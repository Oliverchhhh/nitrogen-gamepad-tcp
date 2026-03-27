import abc
import argparse
import dataclasses
import logging
import json
import os
import queue
import asyncio
import signal
import threading
import numpy as np
import sys
from collections import deque
from typing import AsyncGenerator, AsyncIterator, Awaitable, List, Dict, Tuple, Optional
from elefant.config import load_config
import torch
import time
import lightning as pl
from torch.utils import _pytree as pytree
from elefant.text_tokenizer.factory import get_text_tokenizer
from elefant.text_tokenizer.config import TextTokenizerConfig
from elefant.data.action_mapping import (
    UniversalAutoregressiveActionMapping,
    GamepadAutoregressiveActionMapping,
    StructuredAction,
    DecodedGamepadAction,
)
import elefant.data.proto.video_inference_pb2 as video_inference_pb2
from elefant.policy_model.stage3_finetune import (
    Stage3LabelledBCLightning,
    count_model_parameters,
)
from elefant.torch import pytorch_setup
from elefant.policy_model.config import LightningPolicyConfig
from elefant.data.video_proto_dataset import resize_image_for_model
from elefant.data.proto import shared_pb2
import collections
from PIL import Image

from elefant.inference.unix_socket_server import (
    UnixDomainSocketInferenceServer,
    UDS_PATH,
)
from elefant.torch.util import log_time


class SharedTextInputState:
    """A thread-safe class to hold the latest text input from the terminal."""

    def __init__(self, input_text: bool = False):
        if input_text:
            self._text = ""
        else:
            self._text = None
        self._lock = threading.Lock()

    def get(self) -> Optional[str]:
        """Get the current text input."""
        with self._lock:
            return self._text

    def set(self, new_text: str):
        """Update the text input."""
        with self._lock:
            logging.info(f"Instruction updated to: '{new_text}'")
            self._text = new_text


class TimingMetrics:
    """Track and calculate timing metrics for the inference server."""

    def __init__(self, window_seconds=5.0):
        """Initialize tracking for timing metrics.

        Args:
            window_seconds: The reporting window in seconds
        """
        self.window_seconds = window_seconds
        self.last_report_time = time.time()
        self.metrics: Dict[str, deque] = {
            "request_total_time": deque(),
            "get_action_time": deque(),
        }
        # Add mutex for thread safety
        self.mutex = threading.Lock()

    def record(self, metric_name: str, value: float):
        """Record a timing value for a specific metric.

        Args:
            metric_name: The name of the metric to record
            value: The timing value in seconds
        """
        with self.mutex:
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)

    def should_report(self) -> bool:
        """Check if enough time has passed to report metrics."""
        with self.mutex:
            current_time = time.time()
            if current_time - self.last_report_time >= self.window_seconds:
                return True
            return False

    def calculate_stats(self, values):
        """Calculate statistics for a list of timing values.

        Returns:
            Dict containing min, mean, max, and p99 values
        """
        if not values:
            return {"min": 0, "mean": 0, "max": 0, "p99": 0}

        values_arr = np.array(values)
        return {
            "min": float(np.min(values_arr)),
            "mean": float(np.mean(values_arr)),
            "max": float(np.max(values_arr)),
            "p99": float(np.percentile(values_arr, 99)),
            "p95": float(np.percentile(values_arr, 95)),
        }

    def report_and_reset(self):
        """Report the current metrics and reset for the next window.

        Returns:
            Dict containing stats for each tracked metric
        """
        with self.mutex:
            stats = {}
            for metric_name, values in self.metrics.items():
                if values:
                    # Make a copy of the values to process outside the lock
                    values_copy = list(values)
                    stats[metric_name] = self.calculate_stats(values_copy)

                    # Convert to ms for better readability
                    for stat_name, stat_value in stats[metric_name].items():
                        stats[metric_name][stat_name] = stat_value * 1000

                    # Add count
                    stats[metric_name]["count"] = len(values)

                    # Clear the values
                    # TODO: revert
                    values.clear()

            self.last_report_time = time.time()
            return stats


# TODO: should get this from the model config.
MODEL_INPUT_HEIGHT = 192
MODEL_INPUT_WIDTH = 192


class BaseInferenceState(abc.ABC):
    """Abstract base class for inference state."""

    def __init__(self, config: LightningPolicyConfig):
        self.config = config

    @abc.abstractmethod
    def get_action(
        self, frame: torch.Tensor, text: Optional[str] = None
    ) -> StructuredAction | DecodedGamepadAction:
        """Get the action for the given frame."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset the inference state."""
        pass


class KVCacheInferenceState(BaseInferenceState):
    """Keep track of the state for running the model in inference mode with KV cache.

    Note that this is not really functional, you can only have one of these objects on a given model at a time.
    """

    def __init__(
        self,
        config: LightningPolicyConfig,
        action_mapping: UniversalAutoregressiveActionMapping
        | GamepadAutoregressiveActionMapping,
        model: Stage3LabelledBCLightning,
        max_virtual_steps: int = 20 * 60 * 60,
        use_manual_sampling: bool = False,
        model_records_path: str = None,  # New argument for logging
    ):
        super().__init__(config)
        self.action_mapping = action_mapping
        self.model = model
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.use_manual_sampling = use_manual_sampling
        self.model_records_path = model_records_path
        self.record_model_data = model_records_path is not None
        self.max_virtual_idx = max_virtual_steps * self.model.bc_transformer.step_size
        # TODO: is the rope cache dtype consistent between training and inference?
        self.model.bc_transformer.rebuild_rope_cache(self.max_virtual_idx)
        self.model.bc_transformer.setup_kv_cache(batch_size=1, device=self.device)
        self.step_size = self.model.bc_transformer.step_size
        self.virtual_idx_cpu = 0
        self.text_tokenizer_model_name = self.model._get_text_tokenizer_name()
        if self.text_tokenizer_model_name is not None:
            print(f"text_tokenizer_name is not None: {self.text_tokenizer_model_name}")
            self.text_tokenizer_model = get_text_tokenizer(
                TextTokenizerConfig(text_tokenizer_name=self.text_tokenizer_model_name)
            )
        else:
            self.text_tokenizer_model = None
        self.text_embedding_dim = self.model._get_text_embedding_dim()
        
        # ========================================================================
        # V-JEPA 2 帧缓冲区支持
        # ========================================================================
        # 检查 image_tokenizer 是否是 Vjepa2Tokenizer，如果是，需要维护帧缓冲区
        # 因为 V-JEPA 2 需要至少 tubelet_size 帧（通常是 2 帧）
        image_tokenizer = self.model.bc_transformer.image_tokenizer
        if hasattr(image_tokenizer, 'tubelet_size'):
            # 这是 V-JEPA 2 tokenizer，需要维护帧缓冲区
            self.tubelet_size = image_tokenizer.tubelet_size
            self.frame_buffer = None  # 维护最近 tubelet_size 帧
            self.use_frame_buffer = True
            logging.info(f"检测到 V-JEPA 2 tokenizer，启用帧缓冲区（tubelet_size={self.tubelet_size}）")
        else:
            # 其他 tokenizer（如 ViT、Conv），不需要帧缓冲区
            self.tubelet_size = None
            self.frame_buffer = None
            self.use_frame_buffer = False
        
        self.reset()

    def reset(self):
        self.idx = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.kv_cache_state = self.model.bc_transformer.init_kv_cache_state()
        self.virtual_idx_cpu = 0
        # 重置帧缓冲区
        if self.use_frame_buffer:
            self.frame_buffer = None
        if self.record_model_data and os.path.exists(self.model_records_path):
            # If the log file already exists, we need to remove it to avoid appending to it.
            os.remove(self.model_records_path)

    def get_action(
        self, frame: torch.Tensor, text: Optional[str] = None
    ) -> StructuredAction | DecodedGamepadAction:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            unif_rand_in = None
            if self.use_manual_sampling:
                num_sampling_steps = self.action_mapping.get_seq_len()
                unif_rand_in = torch.rand(
                    size=(num_sampling_steps,), device=self.device
                )
            if self.text_tokenizer_model is not None and text is not None:
                text_input = self.text_tokenizer_model.tokenize(text)
                text_tokens_embed = self.text_tokenizer_model(**text_input)
                assert len(text_tokens_embed.shape) == 4
                # squeeze the batch and T dimension
                text_tokens_embed = text_tokens_embed.squeeze(dim=(0, 1))
            elif self.text_tokenizer_model is not None and text is None:
                # Notice that this initial value needs to be the same as the
                # default value in elefant/data/action_label_video_proto_dataset.py default value
                text_tokens_embed = torch.zeros(
                    size=(
                        self.text_tokenizer_model.get_n_text_tokens(),
                        self.text_tokenizer_model.get_text_embed_dim(),
                    ),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            else:
                # under this case the policy transformer will not take any text embedding
                # so the input will be ignored
                text_tokens_embed = torch.zeros(
                    size=(
                        1,
                        self.text_embedding_dim,
                    ),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            
            # ========================================================================
            # V-JEPA 2 帧缓冲区处理（框架已就绪，待进一步优化）
            # ========================================================================
            # 如果使用 V-JEPA 2 tokenizer，维护帧缓冲区框架
            # 注意：当前由于 online_forward 接口限制（要求 T=1），仍然传入单帧
            # tokenizer 内部会复制帧以满足最小帧数要求
            # TODO: 未来可以优化 online_forward 接口或 tokenizer 接口来真正利用帧缓冲区
            if self.use_frame_buffer:
                # 将新帧添加到缓冲区（为未来优化做准备）
                # frame 形状: [C, H, W] 或 [1, C, H, W]
                if frame.dim() == 3:
                    frame = frame.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                
                if self.frame_buffer is None:
                    # 第一帧，初始化缓冲区
                    self.frame_buffer = frame.unsqueeze(0)  # [1, C, H, W] -> [1, 1, C, H, W]
                else:
                    # 将新帧添加到缓冲区，保持最近 tubelet_size 帧
                    # 例如：tubelet_size=2，保留最近 2 帧
                    self.frame_buffer = torch.cat([
                        self.frame_buffer[:, -self.tubelet_size+1:, :, :, :],  # 保留最近 tubelet_size-1 帧
                        frame.unsqueeze(0)  # 添加新帧
                    ], dim=1)
                
                # 当前实现：由于 online_forward 要求 T=1，仍然传入单帧
                # tokenizer 内部会检测到 T=1 并复制帧
                # 未来优化：可以修改 online_forward 支持多帧，或修改 tokenizer 支持外部帧缓冲区
                frame_to_process = frame.unsqueeze(0)  # [1, C, H, W] -> [1, 1, C, H, W]
            else:
                # 非 V-JEPA 2 tokenizer，直接使用单帧
                # frame 形状: [C, H, W] 或 [1, C, H, W]
                if frame.dim() == 3:
                    frame_to_process = frame.unsqueeze(0).unsqueeze(0)  # [C, H, W] -> [1, 1, C, H, W]
                else:
                    frame_to_process = frame.unsqueeze(0)  # [1, C, H, W] -> [1, 1, C, H, W]
            
            action_tensor, self.idx, self.kv_cache_state = (
                self.model.online_kv_cache_predict(
                    frame_to_process,
                    idx=self.idx,
                    kv_cache_state=self.kv_cache_state,
                    unif_rand=unif_rand_in,
                    sampling_temperature=self.config.inference.sampling_temperature,
                    text_tokens_embed=text_tokens_embed,
                )
            )
            self.virtual_idx_cpu += self.step_size
            if self.virtual_idx_cpu >= self.max_virtual_idx:
                logging.info("Resetting RopE virtual index")
                self.idx = torch.tensor(0, dtype=torch.int64, device=self.device)
                self.virtual_idx_cpu = 0
            action = self.action_mapping.tensor_to_action(
                action_tensor,
                mouse_sampling_approach=self.config.inference.mouse_sampling_approach,
            )
            # Logging block
            if self.record_model_data:
                idx_in = self.idx.cpu().numpy()
                # Convert tensors to CPU and lists for JSON serialization
                kv_cache_sum = 0
                for kv in self.kv_cache_state:
                    kv_cache_sum += kv.k_cache.sum().cpu().item()
                    kv_cache_sum += kv.v_cache.sum().cpu().item()
                data = {
                    "frame": frame.detach().cpu().numpy().flatten().tolist(),
                    "unif_rand_in": unif_rand_in.detach()
                    .cpu()
                    .numpy()
                    .flatten()
                    .tolist()
                    if unif_rand_in is not None
                    else None,
                    "idx_out": int(self.idx.detach().cpu().item()),
                    "idx_in": int(idx_in.item()),
                    "action": action,
                    "kv_cache_sum": kv_cache_sum,
                }
                with open(self.model_records_path, "a") as f:
                    f.write(json.dumps(data) + "\n")
            return action


class FullInferenceState(BaseInferenceState):
    """Keep track of the state for running the model in inference mode."""

    def __init__(
        self,
        config: LightningPolicyConfig,
        action_mapping: UniversalAutoregressiveActionMapping,
        model: Stage3LabelledBCLightning,
    ):
        self.config = config
        self.action_mapping = action_mapping
        self.model = model
        self.device = self.model.device

        self.T = self.config.shared.n_seq_timesteps
        self.n_actions = self.action_mapping.get_seq_len()

        self.frame_history = collections.deque(maxlen=self.T)
        self.action_history = collections.deque(maxlen=self.T)
        self.text_embedding_dim = self.model._get_text_embedding_dim()
        self.text_tokenizer_model_name = self.model._get_text_tokenizer_name()
        if self.text_tokenizer_model_name is not None:
            print(f"text_tokenizer_name is not None: {self.text_tokenizer_model_name}")
            self.text_tokenizer_model = get_text_tokenizer(
                TextTokenizerConfig(text_tokenizer_name=self.text_tokenizer_model_name)
            )
            self.text_tokens_embed = torch.zeros(
                size=(
                    1,
                    self.T,
                    self.text_tokenizer_model.get_n_text_tokens(),
                    self.text_tokenizer_model.get_text_embed_dim(),
                ),
                device=self.device,
                dtype=torch.bfloat16,
            )
        else:
            self.text_tokenizer_model = None
            self.text_tokens_embed = torch.zeros(
                size=(
                    1,
                    self.T,
                    1,
                    self.text_embedding_dim,
                ),
                device=self.device,
                dtype=torch.bfloat16,
            )

        self.frame_in = torch.zeros(
            size=(1, self.T, 3, 192, 192),
            device=self.device,
            dtype=torch.uint8,
        )
        self.action_in = self.action_mapping.make_empty_action(self.T)
        # Add the batch dimension to the actions.
        # To add a batch dimension to every tensor in the action_in namedtuple, use the _replace method and apply unsqueeze(0) to each field.
        self.action_in = pytree.tree_map(
            lambda x: x.unsqueeze(0).to(self.device), self.action_in
        )
        self.n_prior_frames = 0

    def get_action(
        self, frame: torch.Tensor, text: Optional[str] = None
    ) -> StructuredAction:
        if self.n_prior_frames < self.T:
            # We don't worry about inserting the action for this frame yet, because it doesn't matter what is set in the action_in tensor
            # for this frame.
            self.frame_in[:, self.n_prior_frames, :, :, :] = frame
            if self.text_tokenizer_model is not None and text is not None:
                text_input = self.text_tokenizer_model.tokenize(text)
                self.text_tokens_embed[:, self.n_prior_frames, :, :] = (
                    self.text_tokenizer_model(**text_input)
                )
            self.n_prior_frames += 1
        else:
            # We have a full history so we need to roll the frames and actions.
            self.frame_in = torch.roll(self.frame_in, -1, dims=1)
            self.action_in = pytree.tree_map(
                lambda x: torch.roll(x, -1, dims=1), self.action_in
            )
            # Put the new frame at the last position..
            self.frame_in[:, -1, :, :, :] = frame
            self.text_tokens_embed = torch.roll(self.text_tokens_embed, -1, dims=1)
            if self.text_tokenizer_model is not None and text is not None:
                text_input = self.text_tokenizer_model.tokenize(text)
                self.text_tokens_embed[:, -1, :, :] = self.text_tokenizer_model(
                    **text_input
                )
            else:
                self.text_tokens_embed[:, -1, :, :] = torch.zeros(
                    size=(
                        1,
                        self.text_embedding_dim,
                    ),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            # The actions for this new frame will be set below when sampling.

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # For now we (inefficiently) recompute the whole history every time we sample.
            for i in range(self.n_actions):
                action, _action_logits = self.model.online_full_predict(
                    frames=self.frame_in,
                    actions=self.action_in,
                    sampling_temperature=self.config.inference.sampling_temperature,
                    text_tokens_embed=self.text_tokens_embed,
                )
                if i < self.action_mapping.get_number_of_keyboard_actions():
                    self.action_in.keys[:, self.n_prior_frames - 1, i] = action.keys[
                        :, self.n_prior_frames - 1, i
                    ]
                elif (
                    i
                    < self.action_mapping.get_number_of_keyboard_actions()
                    + self.action_mapping.get_number_of_mouse_button_actions()
                ):
                    self.action_in.mouse_buttons[
                        :,
                        self.n_prior_frames - 1,
                        i - self.action_mapping.get_number_of_keyboard_actions(),
                    ] = action.mouse_buttons[
                        :,
                        self.n_prior_frames - 1,
                        i - self.action_mapping.get_number_of_keyboard_actions(),
                    ]
                elif (
                    i
                    < self.action_mapping.get_number_of_keyboard_actions()
                    + self.action_mapping.get_number_of_mouse_button_actions()
                    + 1
                ):
                    self.action_in.mouse_delta_x[
                        :,
                        self.n_prior_frames - 1,
                        i
                        - self.action_mapping.get_number_of_keyboard_actions()
                        - self.action_mapping.get_number_of_mouse_button_actions(),
                    ] = action.mouse_delta_x[
                        :,
                        self.n_prior_frames - 1,
                        i
                        - self.action_mapping.get_number_of_keyboard_actions()
                        - self.action_mapping.get_number_of_mouse_button_actions(),
                    ]
                else:
                    self.action_in.mouse_delta_y[
                        :,
                        self.n_prior_frames - 1,
                        i
                        - self.action_mapping.get_number_of_keyboard_actions()
                        - self.action_mapping.get_number_of_mouse_button_actions()
                        - 1,
                    ] = action.mouse_delta_y[
                        :,
                        self.n_prior_frames - 1,
                        i
                        - self.action_mapping.get_number_of_keyboard_actions()
                        - self.action_mapping.get_number_of_mouse_button_actions()
                        - 1,
                    ]

        # Now pick out the action we actually care about and turn it into a real action.
        sampled_action = pytree.tree_map(
            lambda x: x[:, self.n_prior_frames - 1, :], self.action_in
        )
        action = self.action_mapping.tensor_to_action(
            sampled_action,
            mouse_sampling_approach=self.config.inference.mouse_sampling_approach,
        )
        return action

    def reset(self) -> None:
        """Reset the inference state."""
        ...


@dataclasses.dataclass
class _Request:
    frame: video_inference_pb2.Frame
    start_time_ns: float


class InferenceServer(UnixDomainSocketInferenceServer):
    def __init__(
        self,
        config: LightningPolicyConfig,
        port=8089,
        use_full_inference=False,
        compile=True,
        use_manual_sampling=False,
        use_random_weights=False,
        model_records_path: str = None,
        input_records_path: str = None,
        metrics_window_seconds: float = 60.0,
        checkpoint_path: str = None,
        input_text: bool = False,
    ):
        super().__init__(uds_path=UDS_PATH)
        self.shared_text_state = SharedTextInputState(input_text=input_text)
        self.input_text = input_text
        self.terminal_listener_task = None
        if not compile:
            logging.warning("!!!No compile is enabled!!!")
            torch.compiler.set_stance("force_eager")
            self._compile_stance = "force_eager"
        else:
            self._compile_stance = "fail_on_recompile"

        self.config = config
        mapping_type = getattr(self.config.shared, "action_mapping_type", "keyboard_mouse")
        if mapping_type == "gamepad":
            self.action_mapping = GamepadAutoregressiveActionMapping(
                config=self.config.shared.gamepad_action_mapping
            )
            if use_full_inference:
                raise ValueError(
                    "Full inference mode currently only supports keyboard_mouse mapping."
                )
            logging.info("Using gamepad action mapping for inference.")
        else:
            self.action_mapping = UniversalAutoregressiveActionMapping(
                config=config.shared.action_mapping
            )
            logging.info("Using keyboard_mouse action mapping for inference.")
        self.server_port = port
        # Initialize timing metrics
        self.timing_metrics = TimingMetrics(window_seconds=metrics_window_seconds)
        if use_random_weights:
            logging.info("Using randomly initialized weights (no checkpoint loading)")
            checkpoint_path = None
        else:
            logging.info(f"Loading model from {checkpoint_path}")

        self._setup_device()
        dummy_trainer = pl.Trainer(
            precision=config.shared.precision,
            accelerator="gpu",
            devices=[self.device_idx],
        )

        # Use the dummy trainer to setup the precision, device when loading the model.
        with log_time("Stage3 model instantiation / checkpoint load"):
            with dummy_trainer.init_module():
                if use_random_weights:
                    # Initialize model with random weights
                    self.model = Stage3LabelledBCLightning(
                        config=config, inference_mode=True
                    ).to(device="cuda", dtype=torch.bfloat16)
                else:
                    # Load model from checkpoint
                    self.model = Stage3LabelledBCLightning.load_from_checkpoint(
                        checkpoint_path,
                        config=config,
                        inference_mode=True,
                    )
        total_params, expert_params = count_model_parameters(self.model)
        logging.info(
            f"Total parameters: {total_params}, Expert parameters: {expert_params}"
        )

        self.input_records_path = input_records_path
        if self.input_records_path is not None:
            os.makedirs(input_records_path, exist_ok=True)

        # We don't use lighting trainer for prediction because it's slow / designed for batch prediction.
        self.model.eval()
        self.use_full_inference = use_full_inference
        self.use_manual_sampling = use_manual_sampling
        with log_time("Create inference_state object"):
            if not self.use_full_inference:
                self.inference_state = KVCacheInferenceState(
                    self.config,
                    self.action_mapping,
                    self.model,
                    use_manual_sampling=self.use_manual_sampling,
                    model_records_path=model_records_path,
                )
            else:
                logging.warning(
                    "Using full inference mode - will be slower than using KV Cache."
                )
                self.inference_state = FullInferenceState(
                    self.config, self.action_mapping, self.model
                )
        with log_time("Warm-up / compilation loops"):
            self.model_warmup()

        logging.info("Model warmup done, any compilation should have been done.")
        # The warmup should have done all the compilation, so we should fail if we try to compile again.
        with log_time("FPS test"):
            with torch.compiler.set_stance(self._compile_stance):
                self.fps_test(10_000)

        self.active_connections = set()

    def _setup_device(self):
        # Only try and do inference on GPU.
        assert torch.cuda.is_available()
        cuda_devices = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

        device = None
        # Try and find the first RTX 5090
        for i, device_name in enumerate(cuda_devices):
            if "RTX 5090" in device_name:
                logging.info(f"Using GPU {i} ({device_name})")
                device = f"cuda:{i}"
                device_idx = i
                break
        if device is None and len(cuda_devices) == 1:
            logging.warning(f"No RTX 5090 found, using first GPU: {cuda_devices[0]}")
            device = "cuda:0"
            device_idx = 0
        elif device is None:
            raise ValueError("No RTX 5090 found and multiple GPUs available.")
        self.device = device
        self.device_idx = device_idx
        logging.info(f"Using device: {self.device}")

    @staticmethod
    def _empty_mouse_action() -> video_inference_pb2.MouseAction:
        return video_inference_pb2.MouseAction(
            buttons_down=[],
            mouse_delta_px=shared_pb2.Vec2Int(x=0, y=0),
        )

    @staticmethod
    def _legacy_encode_gamepad_to_keys(action: DecodedGamepadAction) -> List[str]:
        encoded_keys = [f"gamepad:{btn}" for btn in action.buttons_down]
        encoded_keys.extend(
            [
                f"gamepad:lx={action.left_stick[0]:.4f}",
                f"gamepad:ly={action.left_stick[1]:.4f}",
                f"gamepad:rx={action.right_stick[0]:.4f}",
                f"gamepad:ry={action.right_stick[1]:.4f}",
                f"gamepad:lt={action.left_trigger:.4f}",
                f"gamepad:rt={action.right_trigger:.4f}",
            ]
        )
        return encoded_keys

    @staticmethod
    def _set_stick_fields(stick_msg, stick_values: Tuple[float, float]) -> None:
        fields = set(stick_msg.DESCRIPTOR.fields_by_name.keys())
        if "x" in fields:
            stick_msg.x = float(stick_values[0])
        if "y" in fields:
            stick_msg.y = float(stick_values[1])

    def _build_native_gamepad_action(
        self, action: DecodedGamepadAction, frame_id: int
    ) -> Optional[video_inference_pb2.Action]:
        """
        Build Action with native gamepad field when proto supports it.
        Returns None when current proto has no gamepad field.
        """
        action_desc = video_inference_pb2.Action.DESCRIPTOR
        native_field_name = None
        for candidate in ("gamepad_action", "game_pad_action", "controller_action"):
            if candidate in action_desc.fields_by_name:
                native_field_name = candidate
                break

        if native_field_name is None:
            return None

        field_desc = action_desc.fields_by_name[native_field_name]
        msg_cls = getattr(video_inference_pb2, field_desc.message_type.name, None)
        if msg_cls is None:
            logging.warning(
                "Found native gamepad field '%s' but message class '%s' is unavailable",
                native_field_name,
                field_desc.message_type.name,
            )
            return None

        gamepad_msg = msg_cls()
        gamepad_fields = set(gamepad_msg.DESCRIPTOR.fields_by_name.keys())

        if "buttons_down" in gamepad_fields:
            gamepad_msg.buttons_down.extend(action.buttons_down)
        elif "pressed_buttons" in gamepad_fields:
            gamepad_msg.pressed_buttons.extend(action.buttons_down)
        elif "buttons" in gamepad_fields:
            # If proto exposes a typed buttons message, set bool fields by name.
            buttons_msg = getattr(gamepad_msg, "buttons")
            button_fields = set(buttons_msg.DESCRIPTOR.fields_by_name.keys())
            for btn in action.buttons_down:
                if btn in button_fields:
                    setattr(buttons_msg, btn, True)

        if "left_stick" in gamepad_fields:
            self._set_stick_fields(gamepad_msg.left_stick, action.left_stick)
        if "right_stick" in gamepad_fields:
            self._set_stick_fields(gamepad_msg.right_stick, action.right_stick)
        if "left_trigger" in gamepad_fields:
            gamepad_msg.left_trigger = float(action.left_trigger)
        if "right_trigger" in gamepad_fields:
            gamepad_msg.right_trigger = float(action.right_trigger)

        response_action = video_inference_pb2.Action(id=frame_id)
        if "keys" in action_desc.fields_by_name:
            response_action.keys.extend([])
        if "mouse_action" in action_desc.fields_by_name:
            response_action.mouse_action.CopyFrom(self._empty_mouse_action())
        getattr(response_action, native_field_name).CopyFrom(gamepad_msg)
        return response_action

    def fps_test(self, n_frames: int = 100):
        with torch.inference_mode():
            self.inference_state.reset()  # Reset the inference state if using KV cache
            start_time = time.time()
            for i in range(n_frames):
                start_time_i = time.time_ns()
                next_frame = torch.randint(
                    0,
                    255,
                    (3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
                    dtype=torch.uint8,
                    device="cpu",
                ).to(self.device, non_blocking=True)
                if self.input_text:
                    action = self.inference_state.get_action(next_frame, "dummy text")
                else:
                    action = self.inference_state.get_action(next_frame)
                print(f"Action: {action}")
                time_taken_ns = time.time_ns() - start_time_i
                self.timing_metrics.record("get_action_time", time_taken_ns / 1e9)
            end_time = time.time()
            print(
                f"Time taken: {end_time - start_time} seconds, fps = {n_frames / (end_time - start_time)}"
            )
            stats = self.timing_metrics.report_and_reset()
            if stats:
                logging.info(
                    f"Timing metrics for fps test (ms): {json.dumps(stats, indent=1)}"
                )

    def model_warmup(self):
        # Make sure to take enough steps to fill the kv cache to hit all compilation.
        self.fps_test(n_frames=self.config.shared.n_seq_timesteps + 3)

    def _handle_requests(self, in_queue, out_callback):
        time_since_last_frame_ns = time.time_ns()
        # The warmup should have done all the compilation, so we should fail if we try to compile again.
        with torch.inference_mode():
            with torch.compiler.set_stance(
                self._compile_stance, skip_guard_eval_unsafe=False
            ):
                self.inference_state.reset()

                while self.running:
                    try:
                        # Get the latest frame from the queue with a timeout to check running status
                        try:
                            # Use a timeout to periodically check if we should continue running
                            inp = in_queue.get(timeout=0.1)
                        except queue.Empty:
                            # No input received, check if we should still be running
                            continue

                        if inp is None:
                            out_callback(None)
                            break

                        # Process the most recent frame, dropping older ones if multiple are queued
                        while inp is not None:
                            try:
                                new_inp = in_queue.get_nowait()
                                if new_inp is not None:
                                    logging.info(f"Dropping old frame {new_inp[1]}")
                                inp = new_inp
                            except queue.Empty:
                                break

                        if inp is None:
                            out_callback(None)
                            break

                        frame_tensor, frame_id, request_start_time = inp
                        time_since_received_frame_ns = time.time_ns()

                        text_input = self.shared_text_state.get()
                        logging.info(f"Received text input: {text_input}")

                        # Time the get_action call
                        get_action_start_time_ns = time.time_ns()
                        action = self.inference_state.get_action(
                            frame_tensor, text_input
                        )
                        get_action_time_ns = time.time_ns() - get_action_start_time_ns
                        get_action_time_sec = get_action_time_ns / 1e9

                        logging.info(f"Sending action: {action}")

                        # Record the timing metrics
                        self.timing_metrics.record(
                            "get_action_time", get_action_time_sec
                        )

                        # Return keyboard action
                        time_since_last_frame_ns = (
                            time.time_ns() - time_since_last_frame_ns
                        )
                        time_to_process_frame = (
                            time.time_ns() - time_since_received_frame_ns
                        )

                        time_since_last_frame_sec = time_since_last_frame_ns / 1e9
                        fps = 1 / time_since_last_frame_sec

                        logging.debug(
                            f"Sending keys: {action}, fps: {fps}, time to process frame: {time_to_process_frame}s"
                        )
                        time_since_last_frame_ns = time.time_ns()
                        if isinstance(action, DecodedGamepadAction):
                            # Prefer native gamepad transport if proto supports it.
                            # Fallback to legacy keys-encoding for backward compatibility.
                            response_action = self._build_native_gamepad_action(
                                action, frame_id
                            )
                            if response_action is None:
                                encoded_keys = self._legacy_encode_gamepad_to_keys(action)
                                response_action = video_inference_pb2.Action(
                                    keys=encoded_keys,
                                    id=frame_id,
                                    mouse_action=self._empty_mouse_action(),
                                )
                        else:
                            mouse = video_inference_pb2.MouseAction(
                                buttons_down=action.mouse_buttons,
                                mouse_delta_px=shared_pb2.Vec2Int(
                                    x=int(action.mouse_delta_x), y=int(action.mouse_delta_y)
                                ),
                            )
                            response_action = video_inference_pb2.Action(
                                keys=action.keys, id=frame_id, mouse_action=mouse
                            )
                        out_callback(
                            (
                                response_action,
                                request_start_time,
                            )
                        )
                    except Exception as e:
                        logging.exception(f"Error processing frame: {e}")
                        # Continue processing other frames

                logging.info("Exiting frame processing thread")

    async def _make_request_iterator(
        self, reader: asyncio.StreamReader
    ) -> AsyncIterator[_Request]:
        """Create an async iterator that reads frames from the StreamReader."""
        while True:
            try:
                frame = await self._read_frame(reader)
                request_start_time_ns = time.time_ns()

                # Ensure frame.id is an int
                frame.id = int(frame.id)

                # Return both the frame and start time
                yield _Request(frame=frame, start_time_ns=request_start_time_ns)
            except asyncio.IncompleteReadError:
                logging.info("Client disconnected")
                break
            except Exception as e:
                logging.error(f"Error reading frame: {e}")
                break

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> Awaitable[None]:
        client_info = writer.get_extra_info("peername")
        logging.info(f"New connection from {client_info}")

        # Add client to active connections
        self.active_connections.add(writer)

        request_iterator = self._make_request_iterator(reader)
        try:
            async for action in self._infer_video(request_iterator):
                # Check if shutdown was requested
                if self.shutdown_event.is_set():
                    logging.info(
                        f"Shutdown requested, closing connection to {client_info}"
                    )
                    break

                # Send the action back to the client
                await self._write_action(writer, action)

        finally:
            logging.info("Closing client connection")
            # Remove from active connections
            self.active_connections.remove(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception as e:
                logging.error(f"Error while closing writer: {e}")
            logging.info("Client connection closed")

    async def _infer_video(
        self,
        request_iterator: AsyncIterator[_Request],
    ) -> AsyncGenerator[video_inference_pb2.Action]:
        # This is a bit convoluted because gRPC recommends using their async methods for speed.
        # but the torch stuff has to happen on another thread otherwise it blocks all the async
        # stuff (and async Queue is not thread safe).
        # So: we have 1 coroutine (_queue_requests) that grabs each from gRPC and places it onto a
        # (thread safe) queue (this never blocks since the queue has no size limit).
        # In a separate thread (_handle_requests) we pull from the queue and do the torch stuff.
        # If there are multiple frames waiting in the queue we drop all but the newest one.
        # But returning the result is a bit tricky, because asyncio.Queue is not thread safe.
        # and we need to be able to await it on the main thread (and a thread-safe queue can't be awaited).
        # So: we have a callback that the torch thread schedules to run on the main thread that queues
        # the result into a thread-safe queue.
        # The final co-routine (below) awaits this output queue and sends it to gRPC.
        out_queue = asyncio.Queue()
        in_queue = queue.Queue()

        async def _process_request_iterator():
            with torch.inference_mode():
                if self.input_records_path is not None:
                    self._original_tensor_dump_fh = open(
                        os.path.join(self.input_records_path, "original_frames.pt"),
                        "wb",
                    )
                    self._resized_tensor_dump_fh = open(
                        os.path.join(self.input_records_path, "resized_frames.pt"), "wb"
                    )
                async for req in request_iterator:
                    if self.shutdown_event.is_set() or not self.running:
                        logging.info("Shutdown requested, stopping frame processing")
                        break

                    # Convert frame data to tensor
                    frame_tensor_flat = torch.frombuffer(
                        req.frame.data, dtype=torch.uint8
                    )
                    # Comes as HWC
                    frame_tensor_hwc = frame_tensor_flat.view(
                        req.frame.height, req.frame.width, 3
                    )
                    frame_tensor_chw = frame_tensor_hwc.permute(2, 0, 1)

                    if (
                        req.frame.width != MODEL_INPUT_WIDTH
                        or req.frame.height != MODEL_INPUT_HEIGHT
                    ):
                        logging.debug("Image is incorrect size, resizing")
                        frame_tensor_resized = resize_image_for_model(
                            frame_tensor_chw, (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
                        )
                    else:
                        frame_tensor_resized = frame_tensor_chw

                    if self.input_records_path is not None:
                        torch.save(
                            {"id": req.frame.id, "tensor": frame_tensor_hwc.cpu()},
                            self._original_tensor_dump_fh,
                            _use_new_zipfile_serialization=False,
                        )
                        torch.save(
                            {
                                "id": req.frame.id,
                                "tensor": frame_tensor_resized.permute(1, 2, 0).cpu(),
                            },
                            self._resized_tensor_dump_fh,
                            _use_new_zipfile_serialization=False,
                        )

                    frame_tensor_device = (
                        frame_tensor_resized.to(self.device, non_blocking=True)
                        .contiguous()
                        .requires_grad_(False)
                    )

                    in_queue.put((frame_tensor_device, req.frame.id, req.start_time_ns))

        actual_compute_task = asyncio.create_task(_process_request_iterator())

        def _close_dump(_):
            if getattr(self, "_original_tensor_dump_fh", None):
                self._original_tensor_dump_fh.close()
            if getattr(self, "_resized_tensor_dump_fh", None):
                self._resized_tensor_dump_fh.close()

        actual_compute_task.add_done_callback(_close_dump)
        event_loop = asyncio.get_running_loop()

        def _out_callback(
            action_and_start: Tuple[video_inference_pb2.Action, float] | None,
        ):
            # We use nowait here to avoid need to await it (the queue is unlimited size so it will never block).
            event_loop.call_soon_threadsafe(out_queue.put_nowait, action_and_start)

        handle_requests_task = asyncio.create_task(
            asyncio.to_thread(self._handle_requests, in_queue, _out_callback)
        )

        try:
            while self.running:
                if self.shutdown_event.is_set():
                    logging.info("Shutdown requested during inference")
                    break

                # Use wait_for with a timeout to periodically check shutdown status
                try:
                    out = await asyncio.wait_for(out_queue.get(), timeout=0.1)
                    if out is None:
                        logging.warning("Socket connection closed.")
                        break

                    action, request_start_time_ns = out
                    # Calculate total request time
                    request_total_time_ns = time.time_ns() - request_start_time_ns
                    self.timing_metrics.record(
                        "request_total_time", request_total_time_ns / 1e9
                    )

                    # Log stats if it's time to report
                    if self.timing_metrics.should_report():
                        stats = self.timing_metrics.report_and_reset()
                        if stats:
                            logging.info(
                                f"Timing metrics (ms): {json.dumps(stats, indent=1)}"
                            )

                    yield action
                except asyncio.TimeoutError:
                    # No message received within timeout, just continue to check shutdown status
                    continue
        finally:
            # Cancel pending tasks
            if not handle_requests_task.done():
                handle_requests_task.cancel()
            if not actual_compute_task.done():
                actual_compute_task.cancel()

            # Put None in the input queue to signal the processing thread to exit
            in_queue.put(None)

            # Wait for tasks to complete with a timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        actual_compute_task,
                        handle_requests_task,
                        return_exceptions=True,
                    ),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                logging.warning("Timeout waiting for inference tasks to complete")

    async def shutdown(self):
        """Gracefully shut down the server and close all connections."""
        logging.info("Starting graceful shutdown...")
        if self.terminal_listener_task and not self.terminal_listener_task.done():
            self.terminal_listener_task.cancel()
            try:
                await self.terminal_listener_task
            except asyncio.CancelledError:
                logging.info("Terminal listener task cancelled.")

        # Signal all tasks to stop
        self.running = False
        self.shutdown_event.set()

        # Close the server using base implementation
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logging.info("Server closed successfully")

        # Close all active connections
        if self.active_connections:
            logging.info(
                f"Closing {len(self.active_connections)} active connections..."
            )
            for writer in list(self.active_connections):
                writer.close()
            self.active_connections.clear()

            # Wait for all connections to close (don't wait indefinitely)
            try:
                await asyncio.gather(
                    *[writer.wait_closed() for writer in self.active_connections],
                    return_exceptions=True,
                )
            except Exception as e:
                logging.error(f"Error while waiting for connections to close: {e}")

        await super().shutdown()

    async def _listen_for_terminal_input(self):
        """Asynchronously listens for user input from stdin and updates the shared state."""
        logging.info(
            "Starting terminal listener for text input. Press Enter to submit."
        )
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while self.running:
            try:
                # Use wait_for to add a timeout, allowing the loop to check self.running
                line_bytes = await asyncio.wait_for(reader.readline(), timeout=1.0)
                if line_bytes:
                    line = line_bytes.decode("utf-8").strip()
                    if line:
                        self.shared_text_state.set(line)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error reading from terminal: {e}")
                break
        logging.info("Terminal listener stopped.")

    async def serve(self):
        self.terminal_listener_task = asyncio.create_task(
            self._listen_for_terminal_input()
        )
        await super().serve()


def serve_model(
    config: LightningPolicyConfig,
    use_full_inference: bool = False,
    compile: bool = True,
    use_manual_sampling: bool = False,
    use_random_weights: bool = False,
    model_records_path: str = None,
    input_records_path: str = None,
    metrics_window_seconds: float = 60.0,
    checkpoint_path: str = None,
    input_text: bool = False,
):
    logging.basicConfig(level=logging.INFO, force=True)

    # Create the inference server
    inference_model = InferenceServer(
        config,
        use_full_inference=use_full_inference,
        use_manual_sampling=use_manual_sampling,
        compile=compile,
        use_random_weights=use_random_weights,
        model_records_path=model_records_path,
        input_records_path=input_records_path,
        metrics_window_seconds=metrics_window_seconds,
        checkpoint_path=checkpoint_path,
        input_text=input_text,
    )

    # Setup signal handlers for non-asyncio contexts
    def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}, initiating shutdown...")
        # We can't call the async shutdown method directly from here
        # Instead, we'll set a flag that the asyncio context can check
        inference_model.running = False

    # Register the signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, signal_handler)

    # Run the server
    try:
        asyncio.run(inference_model.serve())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logging.exception(f"Error running inference server: {e}")
    finally:
        # Clean up if UDS file still exists
        if os.path.exists(UDS_PATH):
            try:
                os.unlink(UDS_PATH)
                logging.info(f"Removed UDS file: {UDS_PATH}")
            except OSError as e:
                logging.error(f"Error removing UDS file: {e}")

        logging.info("Server shutdown complete")


def _main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--use_full_inference", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--use_manual_sampling", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--use_random_weights",
        action="store_true",
        default=False,
        help="Use randomly initialized weights instead of loading from checkpoint. Requires --config",
    )
    parser.add_argument(
        "--input_records_path",
        type=str,
        required=False,
        help="Path to record inference input/outputs.",
    )
    parser.add_argument(
        "--model_records_path",
        type=str,
        required=False,
        help="Path to record model input data.",
    )
    parser.add_argument(
        "--metrics_window_seconds",
        type=float,
        default=60.0,
        help="Time window in seconds for aggregating and reporting metrics.",
    )
    parser.add_argument(
        "--input_text",
        action="store_true",
        default=False,
        help="Input text from stdin.",
    )
    args = parser.parse_args()

    if args.use_full_inference and args.use_manual_sampling:
        raise ValueError(
            "Cannot use both full inference and manual sampling at the same time."
        )

    if args.config is None:
        raise ValueError("Config must be provided.")

    if args.use_random_weights and args.config is None:
        raise ValueError("--use_random_weights requires --config to be specified.")

    with log_time("Load config"):
        config = load_config(args.config, LightningPolicyConfig)

    with log_time("PyTorch global setup"):
        pytorch_setup()

    with log_time("Prepare and serve model"):
        serve_model(
            config,
            use_full_inference=args.use_full_inference,
            use_manual_sampling=args.use_manual_sampling,
            compile=not args.no_compile,
            use_random_weights=args.use_random_weights,
            model_records_path=args.model_records_path,
            input_records_path=args.input_records_path,
            metrics_window_seconds=args.metrics_window_seconds,
            checkpoint_path=args.checkpoint_path,
            input_text=args.input_text,
        )


if __name__ == "__main__":
    _main()
