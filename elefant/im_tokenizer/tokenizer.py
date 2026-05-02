import logging
import os
from abc import abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from elefant.im_tokenizer.config import ImageTokenizerConfig
from huggingface_hub import login, snapshot_download
from elefant.torch import eager_assert
from elefant.torch import eager_assert
from elefant.modules import LayerNormF32
from elefant.im_tokenizer import conv_tokenizer
from elefant.im_tokenizer.base_tokenizer import ImageBaseTokenizer
from typing import Tuple
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the images of shape [B, T, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, T, C, H, W = x.shape
    x = x.reshape(B, T, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # [B, T, H', W', C, p_H, p_W]
    x = x.flatten(2, 3)  # [B, T, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(3, 5)  # [B, T H'*W', C*p_H*p_W]
    return x


class IdentityTokenizer(ImageBaseTokenizer):
    """
    This tokenizer is used when we don't want to tokenize the image just pass it through.
    Only useful for unit testing.
    """

    def __init__(self, config: ImageTokenizerConfig, n_img_tokens: int):
        super().__init__(config)
        self.n_img_tokens = n_img_tokens

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B, T, C * H, W)
        assert x.shape[2] == self.n_img_tokens
        return x


class VitImageTokenizer(ImageBaseTokenizer):
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)
        self.n_img_tokens = (frame_height // config.vit_tokenizer_config.patch_size) * (
            frame_width // config.vit_tokenizer_config.patch_size
        )
        self.patch_size = config.vit_tokenizer_config.patch_size
        self.proj_to_embed_dim = nn.Linear(
            self.patch_size * self.patch_size * 3, embed_dim
        )
        # self.post_norm = LayerNormF32(embed_dim)

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, img: torch.Tensor):
        patches = img_to_patch(img, self.patch_size)
        patches = self.proj_to_embed_dim(patches)
        # TODO: decide if this is good or not.
        # patches = self.post_norm(patches)
        return patches


class DinoV2Tokenizer(ImageBaseTokenizer):
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        # Set model parameters to not trainable.
        for param in self.model.parameters():
            param.requires_grad = False

        if frame_height != 192 or frame_width != 192:
            raise ValueError("DinoV2Tokenizer only supports 192x192 images")
        self.n_image_tokens = 196

        self.embed_dim = embed_dim
        self.proj_to_embed_dim = nn.Linear(768, embed_dim)

    def get_n_img_tokens(self) -> int:
        return self.n_image_tokens

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = input_tensor.shape
        with torch.no_grad():
            # Pad the input tensor to 196.
            x = F.pad(input_tensor, (2, 2, 2, 2), value=0.0)
            features = self.model.forward_features(x.view(B * T, C, 196, 196))

        features = self.proj_to_embed_dim(features["x_norm_patchtokens"])
        eager_assert(features.shape, (B * T, self.n_image_tokens, self.embed_dim))
        features = features.view(B, T, self.n_image_tokens, self.embed_dim)
        return features


class StaMoTokenizer(ImageBaseTokenizer):
    """
    StaMo Tokenizer - 使用 timm ViT backbone + Q-Former 式 Projector 将图像压缩为紧凑 token。

    数据流：
      [B, T, C, H, W] -> 逐帧 -> VisionBackbone -> [B*T, N_patch, C_vit]
                       -> Projector -> [B*T, num_token, output_align_dim]
                       -> proj_to_embed_dim -> [B, T, num_token, embed_dim]

    作为 state_target_tokenizer 使用时，_build_state_target 会对 dim=2 做 mean pool，
    得到 [B, T, embed_dim] 的全局状态表示。
    """

    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)

        if config.stamo_tokenizer_config is None:
            raise ValueError("stamo_tokenizer_config must be provided when type='stamo'")

        stamo_cfg = config.stamo_tokenizer_config
        self.stamo_cfg = stamo_cfg
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.embed_dim = embed_dim

        # ---- 添加 StaMo 源码路径 ----
        import sys
        from pathlib import Path

        stamo_path = os.getenv("STAMO_PATH")
        if stamo_path is None:
            candidate_paths = [
                Path(__file__).resolve().parents[2] / "third_party" / "StaMo",
                Path(__file__).resolve().parents[2] / "third_part" / "StaMo",
            ]
            for p in candidate_paths:
                if p.exists():
                    stamo_path = str(p)
                    break

        if stamo_path and os.path.exists(stamo_path):
            if stamo_path not in sys.path:
                sys.path.insert(0, stamo_path)
            logging.info(f"已添加 StaMo 路径到 sys.path: {stamo_path}")

        # ---- 构建 VisionBackbone ----
        from stamo.renderer.model.backbone import VisionBackbone

        self.vision_backbone = VisionBackbone(
            img_size=stamo_cfg.img_size,
            model_name=stamo_cfg.model_name,
            pretrained=stamo_cfg.backbone_pretrained,
            local_ckpt=stamo_cfg.backbone_local_ckpt,
        )

        # ---- 构建 Projector ----
        from stamo.renderer.model.projector import Projector
        from types import SimpleNamespace

        projector_args = SimpleNamespace(
            projector=SimpleNamespace(
                num_token=stamo_cfg.num_token,
                num_attn_layers=stamo_cfg.num_attn_layers,
                num_attn_compress_layers=stamo_cfg.num_attn_compress_layers,
                hidden_dim=stamo_cfg.hidden_dim,
                cross_attention_dim=stamo_cfg.cross_attention_dim,
                output_align_dim=stamo_cfg.output_align_dim,
            )
        )
        self.projector = Projector(
            projector_args,
            patches=self.vision_backbone.patches,
            channels=self.vision_backbone.channels,
        )

        # ---- 加载 Projector 权重 ----
        if stamo_cfg.projector_ckpt and os.path.exists(stamo_cfg.projector_ckpt):
            projector_state = torch.load(stamo_cfg.projector_ckpt, map_location="cpu")
            msg = self.projector.load_state_dict(projector_state, strict=False)
            if msg.missing_keys:
                logging.warning(f"StaMo Projector missing keys: {msg.missing_keys}")
            if msg.unexpected_keys:
                logging.warning(f"StaMo Projector unexpected keys: {msg.unexpected_keys}")
            logging.info(f"已加载 StaMo Projector 权重: {stamo_cfg.projector_ckpt}")
        else:
            logging.warning("StaMo Projector 未加载预训练权重，使用随机初始化")

        # ---- 冻结 ----
        if stamo_cfg.frozen:
            self.vision_backbone.eval()
            self.projector.eval()
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = False

        # ---- 输出维度对齐 ----
        self.stamo_output_dim = stamo_cfg.output_align_dim
        self.n_img_tokens = stamo_cfg.num_token

        if self.stamo_output_dim != embed_dim:
            self.proj_to_embed_dim = nn.Linear(self.stamo_output_dim, embed_dim)
        else:
            self.proj_to_embed_dim = None

        # ---- 预处理 transform（与 timm backbone 一致）----
        self.backbone_transforms = self.vision_backbone.transforms

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: [B, T, C, H, W]
        Returns:
            [B, T, num_token, embed_dim]
        """
        B, T, C, H, W = input_tensor.shape

        # 逐帧处理：reshape 成 [B*T, C, H, W]
        x = input_tensor.reshape(B * T, C, H, W)

        # resize 到 StaMo 期望的尺寸
        if H != self.stamo_cfg.img_size or W != self.stamo_cfg.img_size:
            x = F.interpolate(
                x,
                size=(self.stamo_cfg.img_size, self.stamo_cfg.img_size),
                mode="bilinear",
                align_corners=False,
            )

        # timm backbone 归一化
        x = self.backbone_transforms(x)

        # VisionBackbone: [B*T, C, H, W] -> [B*T, N_patch, C_vit]
        with torch.set_grad_enabled(self.training and not self.stamo_cfg.frozen):
            patch_features = self.vision_backbone(x)
            # Projector: [B*T, N_patch, C_vit] -> [B*T, num_token, output_align_dim]
            compressed = self.projector(patch_features)

        # 投影到 embed_dim
        if self.proj_to_embed_dim is not None:
            compressed = self.proj_to_embed_dim(compressed)

        # reshape 回 [B, T, num_token, embed_dim]
        compressed = compressed.view(B, T, self.n_img_tokens, self.embed_dim)

        return compressed


class NitrogenSiglipTokenizer(ImageBaseTokenizer):
    """
    NitroGen 风格 SigLIP tokenizer：
    - 保留 dense visual tokens（不做 pooling）
    - 若视觉维度与 policy embed_dim 不一致，自动线性投影
    """

    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)

        if config.nitrogen_siglip_tokenizer_config is None:
            raise ValueError(
                "nitrogen_siglip_tokenizer_config must be provided when type='nitrogen_siglip'"
            )

        siglip_cfg = config.nitrogen_siglip_tokenizer_config
        self.siglip_cfg = siglip_cfg
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.embed_dim = embed_dim
        self.target_img_size = siglip_cfg.image_size

        self.model_source = siglip_cfg.vision_encoder_name
        local_files_only = False
        if siglip_cfg.vision_encoder_local_path:
            local_path = Path(siglip_cfg.vision_encoder_local_path)
            if not local_path.exists():
                raise ValueError(
                    f"vision_encoder_local_path does not exist: {siglip_cfg.vision_encoder_local_path}"
                )
            self.model_source = str(local_path)
            local_files_only = True

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_source,
            local_files_only=local_files_only,
        )
        self.image_mean = getattr(self.image_processor, "image_mean", [0.5, 0.5, 0.5])
        self.image_std = getattr(self.image_processor, "image_std", [0.5, 0.5, 0.5])
        self.use_image_processor_norm = siglip_cfg.use_image_processor_norm

        if "siglip" in siglip_cfg.vision_encoder_name.lower():
            model = SiglipVisionModel.from_pretrained(
                self.model_source,
                local_files_only=local_files_only,
            )
            self.vision_encoder = model.vision_model
        else:
            self.vision_encoder = AutoModel.from_pretrained(
                self.model_source,
                local_files_only=local_files_only,
            )

        if siglip_cfg.frozen:
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # 通过 dummy forward 确定 token 数和视觉维度，确保与主模型接口一致。
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.target_img_size, self.target_img_size)
            dummy_out = self._encode_vision(dummy)
            if dummy_out.ndim != 3:
                raise ValueError(
                    f"NitrogenSiglipTokenizer expects 3D output [B, N, D], got shape {tuple(dummy_out.shape)}"
                )
            _, n_tokens, vision_dim = dummy_out.shape

        if (
            siglip_cfg.expected_n_img_tokens is not None
            and n_tokens != siglip_cfg.expected_n_img_tokens
        ):
            raise ValueError(
                f"Unexpected number of visual tokens from {siglip_cfg.vision_encoder_name}: "
                f"got {n_tokens}, expected {siglip_cfg.expected_n_img_tokens}"
            )
        if (
            siglip_cfg.vision_hidden_size is not None
            and vision_dim != siglip_cfg.vision_hidden_size
        ):
            raise ValueError(
                f"Unexpected vision hidden size from {siglip_cfg.vision_encoder_name}: "
                f"got {vision_dim}, expected {siglip_cfg.vision_hidden_size}"
            )

        self.n_img_tokens = n_tokens
        self.vision_hidden_size = vision_dim

        if vision_dim != embed_dim:
            self.proj_to_embed_dim = nn.Linear(vision_dim, embed_dim)
            logging.info(
                "NitrogenSiglipTokenizer: adding projection layer %s -> %s",
                vision_dim,
                embed_dim,
            )
        else:
            self.proj_to_embed_dim = None

    def _encode_vision(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=x)
        if isinstance(outputs, dict):
            if "last_hidden_state" not in outputs:
                raise ValueError(
                    f"Vision encoder output does not contain last_hidden_state. Keys: {list(outputs.keys())}"
                )
            return outputs["last_hidden_state"]
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError(
                f"Vision encoder output does not have last_hidden_state: {type(outputs)}"
            )
        return outputs.last_hidden_state

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_image_processor_norm:
            return x
        mean = torch.tensor(self.image_mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = input_tensor.shape

        x = input_tensor.reshape(B * T, C, H, W).to(torch.float32)
        if H != self.target_img_size or W != self.target_img_size:
            x = F.interpolate(
                x,
                size=(self.target_img_size, self.target_img_size),
                mode="bilinear",
                align_corners=False,
            )
        x = self._normalize(x)

        with torch.set_grad_enabled(self.training and not self.siglip_cfg.frozen):
            features = self._encode_vision(x)

        eager_assert(features.shape[0], B * T)
        eager_assert(features.shape[1], self.n_img_tokens)
        eager_assert(features.shape[2], self.vision_hidden_size)

        if self.proj_to_embed_dim is not None:
            features = self.proj_to_embed_dim(features)

        features = features.view(B, T, self.n_img_tokens, self.embed_dim)
        return features


class NitrogenCheckpointTokenizer(ImageBaseTokenizer):
    """
    NitroGen checkpoint 对齐版视觉 encoder：
    vision_encoder(SigLIP2) -> vl_self_attention_model(4-layer Transformer)
    """

    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        super().__init__(config)

        if config.nitrogen_checkpoint_tokenizer_config is None:
            raise ValueError(
                "nitrogen_checkpoint_tokenizer_config must be provided when type='nitrogen_checkpoint'"
            )

        ckpt_cfg = config.nitrogen_checkpoint_tokenizer_config
        self.ckpt_cfg = ckpt_cfg
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.embed_dim = embed_dim
        self.target_img_size = ckpt_cfg.image_size
        self.vision_hidden_size = ckpt_cfg.vision_hidden_size

        # Dynamically add third_party/NitroGen to sys.path, then import VL mixing module.
        import sys

        nitrogen_path = os.getenv("NITROGEN_PATH")
        if nitrogen_path is None:
            candidate_paths = [
                Path(__file__).resolve().parents[2] / "third_party" / "NitroGen",
                Path(__file__).resolve().parents[2] / "third_part" / "NitroGen",
            ]
            for p in candidate_paths:
                if p.exists():
                    nitrogen_path = str(p)
                    break
        if not nitrogen_path or not os.path.exists(nitrogen_path):
            raise ValueError(
                "Cannot locate NitroGen source. Set NITROGEN_PATH or place NitroGen under third_party/NitroGen."
            )
        if nitrogen_path not in sys.path:
            sys.path.insert(0, nitrogen_path)

        from nitrogen.flow_matching_transformer.modules import (
            SelfAttentionTransformer,
            SelfAttentionTransformerConfig,
        )

        self.model_source = ckpt_cfg.vision_encoder_name
        local_files_only = False
        if ckpt_cfg.vision_encoder_local_path:
            local_path = Path(ckpt_cfg.vision_encoder_local_path)
            if not local_path.exists():
                raise ValueError(
                    f"vision_encoder_local_path does not exist: {ckpt_cfg.vision_encoder_local_path}"
                )
            self.model_source = str(local_path)
            local_files_only = True

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_source,
            local_files_only=local_files_only,
        )
        self.image_mean = getattr(self.image_processor, "image_mean", [0.5, 0.5, 0.5])
        self.image_std = getattr(self.image_processor, "image_std", [0.5, 0.5, 0.5])
        self.use_image_processor_norm = ckpt_cfg.use_image_processor_norm

        if "siglip" in ckpt_cfg.vision_encoder_name.lower():
            model = SiglipVisionModel.from_pretrained(
                self.model_source,
                local_files_only=local_files_only,
            )
            self.vision_encoder = model.vision_model
        else:
            self.vision_encoder = AutoModel.from_pretrained(
                self.model_source,
                local_files_only=local_files_only,
            )

        vl_cfg = SelfAttentionTransformerConfig(
            num_attention_heads=ckpt_cfg.vl_num_attention_heads,
            attention_head_dim=ckpt_cfg.vl_attention_head_dim,
            num_layers=ckpt_cfg.vl_num_layers,
            dropout=ckpt_cfg.vl_dropout,
            attention_bias=ckpt_cfg.vl_attention_bias,
            activation_fn=ckpt_cfg.vl_activation_fn,
            upcast_attention=ckpt_cfg.vl_upcast_attention,
            max_num_positional_embeddings=ckpt_cfg.vl_max_num_positional_embeddings,
            compute_dtype=ckpt_cfg.vl_compute_dtype,
            final_dropout=ckpt_cfg.vl_final_dropout,
            positional_embeddings=ckpt_cfg.vl_positional_embeddings,
        )
        self.vl_self_attention_model = SelfAttentionTransformer(config=vl_cfg)

        checkpoint_path = Path(ckpt_cfg.checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        vision_state_dict = {
            k.removeprefix("vision_encoder."): v
            for k, v in state_dict.items()
            if k.startswith("vision_encoder.")
        }
        vl_state_dict = {
            k.removeprefix("vl_self_attention_model."): v
            for k, v in state_dict.items()
            if k.startswith("vl_self_attention_model.")
        }
        if not vision_state_dict:
            raise ValueError(f"No 'vision_encoder.*' keys found in checkpoint: {checkpoint_path}")
        if not vl_state_dict:
            raise ValueError(f"No 'vl_self_attention_model.*' keys found in checkpoint: {checkpoint_path}")

        vision_msg = self.vision_encoder.load_state_dict(vision_state_dict, strict=ckpt_cfg.strict_load)
        vl_msg = self.vl_self_attention_model.load_state_dict(vl_state_dict, strict=ckpt_cfg.strict_load)
        if vision_msg.missing_keys:
            logging.warning("NitrogenCheckpointTokenizer vision missing keys: %s", vision_msg.missing_keys)
        if vision_msg.unexpected_keys:
            logging.warning("NitrogenCheckpointTokenizer vision unexpected keys: %s", vision_msg.unexpected_keys)
        if vl_msg.missing_keys:
            logging.warning("NitrogenCheckpointTokenizer vl mixer missing keys: %s", vl_msg.missing_keys)
        if vl_msg.unexpected_keys:
            logging.warning("NitrogenCheckpointTokenizer vl mixer unexpected keys: %s", vl_msg.unexpected_keys)

        if ckpt_cfg.freeze_vision_encoder:
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if ckpt_cfg.freeze_vl_self_attention_model:
            self.vl_self_attention_model.eval()
            for param in self.vl_self_attention_model.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.target_img_size, self.target_img_size)
            dummy_features = self._encode_vision(dummy)
            if dummy_features.ndim != 3:
                raise ValueError(
                    f"NitrogenCheckpointTokenizer expects 3D output [B, N, D], got shape {tuple(dummy_features.shape)}"
                )
            _, n_tokens, hidden_dim = dummy_features.shape
            dummy_mixed = self.vl_self_attention_model(dummy_features)
            if dummy_mixed.shape[-1] != hidden_dim:
                raise ValueError(
                    f"Unexpected VL mixer output dim {dummy_mixed.shape[-1]}, expected {hidden_dim}"
                )

        if (
            ckpt_cfg.expected_n_img_tokens is not None
            and n_tokens != ckpt_cfg.expected_n_img_tokens
        ):
            raise ValueError(
                f"Unexpected number of visual tokens: got {n_tokens}, expected {ckpt_cfg.expected_n_img_tokens}"
            )
        if (
            ckpt_cfg.vision_hidden_size is not None
            and hidden_dim != ckpt_cfg.vision_hidden_size
        ):
            raise ValueError(
                f"Unexpected vision hidden size: got {hidden_dim}, expected {ckpt_cfg.vision_hidden_size}"
            )

        self.n_img_tokens = n_tokens
        self.vision_hidden_size = hidden_dim

        if hidden_dim != embed_dim:
            self.proj_to_embed_dim = nn.Linear(hidden_dim, embed_dim)
            logging.info(
                "NitrogenCheckpointTokenizer: adding projection layer %s -> %s",
                hidden_dim,
                embed_dim,
            )
        else:
            self.proj_to_embed_dim = None

    def _encode_vision(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(pixel_values=x)
        if isinstance(outputs, dict):
            if "last_hidden_state" not in outputs:
                raise ValueError(
                    f"Vision encoder output does not contain last_hidden_state. Keys: {list(outputs.keys())}"
                )
            return outputs["last_hidden_state"]
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError(
                f"Vision encoder output does not have last_hidden_state: {type(outputs)}"
            )
        return outputs.last_hidden_state

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_image_processor_norm:
            return x
        mean = torch.tensor(self.image_mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - mean) / std

    def get_n_img_tokens(self) -> int:
        return self.n_img_tokens

    def forward_legacy(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = input_tensor.shape
        vl_seq_len = T * self.n_img_tokens
        vl_max_seq_len = self.ckpt_cfg.vl_max_num_positional_embeddings
        if vl_seq_len > vl_max_seq_len:
            raise ValueError(
                "NitrogenCheckpointTokenizer sequence length exceeds vl positional embedding limit: "
                f"T={T}, n_img_tokens={self.n_img_tokens}, T*N={vl_seq_len}, "
                f"vl_max_num_positional_embeddings={vl_max_seq_len}. "
                "Increase vl_max_num_positional_embeddings or reduce n_seq_timesteps."
            )

        x = input_tensor.reshape(B * T, C, H, W).to(torch.float32)
        if H != self.target_img_size or W != self.target_img_size:
            x = F.interpolate(
                x,
                size=(self.target_img_size, self.target_img_size),
                mode="bilinear",
                align_corners=False,
            )
        x = self._normalize(x)

        with torch.set_grad_enabled(self.training and not self.ckpt_cfg.freeze_vision_encoder):
            features = self._encode_vision(x)  # [B*T, N, D]

        features = features.view(B, T, self.n_img_tokens, self.vision_hidden_size)
        vl_input = features.reshape(B, T * self.n_img_tokens, self.vision_hidden_size)
        with torch.set_grad_enabled(
            self.training and not self.ckpt_cfg.freeze_vl_self_attention_model
        ):
            vl_mixed = self.vl_self_attention_model(vl_input)
        features = vl_mixed.view(B, T, self.n_img_tokens, self.vision_hidden_size)

        if self.proj_to_embed_dim is not None:
            features = self.proj_to_embed_dim(features)

        eager_assert(features.shape[0], B)
        eager_assert(features.shape[1], T)
        eager_assert(features.shape[2], self.n_img_tokens)
        eager_assert(features.shape[3], self.embed_dim)
        return features
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = input_tensor.shape
        # vl_seq_len = T * self.n_img_tokens
        # vl_max_seq_len = self.ckpt_cfg.vl_max_num_positional_embeddings
        # if vl_seq_len > vl_max_seq_len:
        #     raise ValueError(
        #         "NitrogenCheckpointTokenizer sequence length exceeds vl positional embedding limit: "
        #         f"T={T}, n_img_tokens={self.n_img_tokens}, T*N={vl_seq_len}, "
        #         f"vl_max_num_positional_embeddings={vl_max_seq_len}. "
        #         "Increase vl_max_num_positional_embeddings or reduce n_seq_timesteps."
        #     )

        x = input_tensor.reshape(B * T, C, H, W).to(torch.float32)
        if H != self.target_img_size or W != self.target_img_size:
            x = F.interpolate(
                x,
                size=(self.target_img_size, self.target_img_size),
                mode="bilinear",
                align_corners=False,
            )
        x = self._normalize(x)

        with torch.set_grad_enabled(self.training and not self.ckpt_cfg.freeze_vision_encoder):
            features = self._encode_vision(x)  # [B*T, N, D]

        features = features.view(B, T, self.n_img_tokens, self.vision_hidden_size)
        # vl_input = features.reshape(B, T * self.n_img_tokens, self.vision_hidden_size)
        # print(features.shape)
        vl_input = features.reshape(B*T, self.n_img_tokens, self.vision_hidden_size) #只进行单帧图像内的Visian 融合
        with torch.set_grad_enabled(
            self.training and not self.ckpt_cfg.freeze_vl_self_attention_model
        ):
            vl_mixed = self.vl_self_attention_model(vl_input)
        features = vl_mixed.view(B, T, self.n_img_tokens, self.vision_hidden_size)

        if self.proj_to_embed_dim is not None:
            features = self.proj_to_embed_dim(features)

        eager_assert(features.shape[0], B)
        eager_assert(features.shape[1], T)
        eager_assert(features.shape[2], self.n_img_tokens)
        eager_assert(features.shape[3], self.embed_dim)
        return features
    


class Vjepa2Tokenizer(ImageBaseTokenizer):
    """
    V-JEPA 2 Tokenizer - 使用 V-JEPA 2 作为视觉编码器
    
    V-JEPA 2 (Video Joint-Embedding Predictive Architecture 2) 是 Meta 开发的视频预训练模型，
    通过时空 patch 编码来提取视频特征。本类将 V-JEPA 2 封装为 OpenP2P 的视觉 tokenizer。
    
    关键特性：
    1. 自动处理单帧输入：V-JEPA 2 需要 T >= tubelet_size（通常是 2），
       对于单帧输入（T=1），会自动复制最后一帧以满足最小帧数要求
    2. 时间维度扩展：V-JEPA 2 会压缩时间维度（T // tubelet_size），
       本实现会自动扩展回原始帧数以保持接口一致性
    3. 分辨率适配：自动将输入分辨率 resize 到 V-JEPA 2 期望的尺寸
    4. 特征维度投影：如果 V-JEPA 2 的输出维度与 policy model 的 embed_dim 不同，
       会自动添加投影层
    
    输入格式：[B, T, C, H, W] - batch, time, channels, height, width
    输出格式：[B, T, N, embed_dim] - batch, time, num_tokens, embedding_dim
    """
    
    def __init__(
        self,
        config: ImageTokenizerConfig,
        frame_height: int,
        frame_width: int,
        embed_dim: int,
    ):
        """
        初始化 V-JEPA 2 Tokenizer
        
        Args:
            config: ImageTokenizerConfig，必须包含 vjepa_tokenizer_config
            frame_height: OpenP2P 的输入帧高度（实际输入分辨率）
            frame_width: OpenP2P 的输入帧宽度（实际输入分辨率）
            embed_dim: Policy model 的 embedding 维度（需要与 transformer_dim 一致）
        """
        super().__init__(config)
        
        # 检查配置
        if config.vjepa_tokenizer_config is None:
            raise ValueError("vjepa_tokenizer_config must be provided when type='vjepa2'")
        
        vjepa_config = config.vjepa_tokenizer_config
        
        # ========================================================================
        # 导入 vjepa2 相关模块（延迟导入，避免依赖问题）
        # ========================================================================
        import sys
        from pathlib import Path

        # 尝试从环境变量或默认路径添加 vjepa2 项目路径
        vjepa2_path = os.getenv("VJEPA2_PATH")
        if vjepa2_path is None:
            # 默认：项目内 third_party/vjepa2，其次回退到常见本地路径
            candidate_paths = [
                Path(__file__).resolve().parents[2] / "third_party" / "vjepa2",
                Path(__file__).resolve().parents[2] / "third_part" / "vjepa2",
                Path("D:/project/vjepa2"),
                Path("/mnt/d/project/vjepa2"),
            ]
            for p in candidate_paths:
                if p.exists():
                    vjepa2_path = str(p)
                    break

        if vjepa2_path and Path(vjepa2_path).exists():
            if vjepa2_path not in sys.path:
                sys.path.insert(0, vjepa2_path)
            logging.info(f"已添加 vjepa2 路径到 sys.path: {vjepa2_path}")
        else:
            logging.warning("vjepa2 路径不存在，尝试直接导入")
        
        # 存储配置
        self.vjepa_config = vjepa_config
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.embed_dim = embed_dim
        self.tubelet_size = vjepa_config.tubelet_size
        self.patch_size = vjepa_config.patch_size
        
        # 判断是否为 V-JEPA 2.1 模型
        self._is_vjepa21 = vjepa_config.model_name.startswith("vjepa2_1_")

        # 加载 V-JEPA 2 / 2.1 encoder
        self.encoder = self._load_vjepa_model(vjepa_config)
        
        # 设置冻结模式
        if vjepa_config.frozen:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 计算每帧的空间 token 数（不考虑时间维度）
        # 注意：实际输出 token 数 = (T // tubelet_size) * n_spatial_tokens
        self.n_spatial_tokens = (vjepa_config.img_size // vjepa_config.patch_size) ** 2
        
        # 对于 OpenP2P，我们需要返回每帧 token 数。
        # 默认返回所有空间 tokens；可选开启全局聚合（每帧压成 1 token）以降低序列长度。
        self.pool_to_global = bool(getattr(vjepa_config, "pool_to_global", False))
        self.n_img_tokens = 1 if self.pool_to_global else self.n_spatial_tokens
        
        # 如果 V-JEPA 2 的输出维度与 embed_dim 不同，添加投影层
        vjepa_embed_dim = self.encoder.embed_dim
        if vjepa_embed_dim != embed_dim:
            self.proj_to_embed_dim = nn.Linear(vjepa_embed_dim, embed_dim)
        else:
            self.proj_to_embed_dim = None

        if self.pool_to_global:
            hidden_dim = getattr(vjepa_config, "aggregation_mlp_hidden_dim", None)
            if hidden_dim is None:
                hidden_dim = embed_dim * 2
            self.global_token_norm = LayerNormF32(embed_dim)
            self.global_token_mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.global_token_norm = None
            self.global_token_mlp = None
        
        # 存储 vjepa_embed_dim 供 forward 使用
        self.vjepa_embed_dim = vjepa_embed_dim
    
    def _load_vjepa_model(self, config):
        """加载 V-JEPA 2/2.1 模型（优先本地 checkpoint）。"""
        checkpoint_path = config.checkpoint_path
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"从本地 checkpoint 加载 V-JEPA: {checkpoint_path}")
            return self._load_from_checkpoint(checkpoint_path, config)
        if config.use_hub_fallback:
            logging.info(f"从 torch.hub 加载 V-JEPA: {config.model_name}")
            return self._load_via_hub(config)
        raise ValueError(
            f"未找到本地 checkpoint ({checkpoint_path})，且 use_hub_fallback=False。"
            "请设置 checkpoint_path 或 use_hub_fallback=True"
        )

    def _load_via_hub(self, config):
        """通过 hub 函数加载模型。"""
        if self._is_vjepa21:
            from src.hub.backbones import (
                vjepa2_1_vit_base_384,
                vjepa2_1_vit_large_384,
            )

            hub_map = {
                "vjepa2_1_vit_base_384": vjepa2_1_vit_base_384,
                "vjepa2_1_vit_large_384": vjepa2_1_vit_large_384,
            }
        else:
            from src.hub.backbones import (
                vjepa2_vit_large,
                vjepa2_vit_huge,
                vjepa2_vit_giant,
                vjepa2_vit_giant_384,
            )

            hub_map = {
                "vit_large": vjepa2_vit_large,
                "vit_huge": vjepa2_vit_huge,
                "vit_giant": vjepa2_vit_giant,
                "vit_giant_384": vjepa2_vit_giant_384,
            }

        if config.model_name not in hub_map:
            raise ValueError(
                f"不支持的 model_name: {config.model_name}。"
                f"支持的值: {list(hub_map.keys())}"
            )

        encoder, _ = hub_map[config.model_name](pretrained=True)
        return encoder

    def _resolve_checkpoint_key(self, checkpoint, config):
        """自动解析 checkpoint 中 encoder 权重 key。"""
        if config.checkpoint_key != "auto":
            return config.checkpoint_key

        for candidate in ["ema_encoder", "encoder", "target_encoder"]:
            if candidate in checkpoint:
                logging.info(f"自动检测到 checkpoint key: {candidate}")
                return candidate

        raise KeyError(f"checkpoint 中未找到 encoder 权重，可用 keys: {list(checkpoint.keys())}")

    def _load_from_checkpoint(self, checkpoint_path, config):
        """从本地 checkpoint 文件加载模型权重。"""

        def _clean_backbone_key(state_dict):
            cleaned = {}
            for key, val in state_dict.items():
                new_key = key.replace("module.", "").replace("backbone.", "")
                cleaned[new_key] = val
            return cleaned

        # 创建 encoder 结构
        if self._is_vjepa21:
            from app.vjepa_2_1.models import vision_transformer as vit_encoder
            from src.hub.backbones import ARCH_NAME_MAP

            arch_name = ARCH_NAME_MAP[config.model_name][0]
            encoder = vit_encoder.__dict__[arch_name](**self._get_model_kwargs(config))
        else:
            from src.models import vision_transformer as vit_encoder

            model_map = {
                "vit_large": "vit_large",
                "vit_huge": "vit_huge",
                "vit_giant": "vit_giant_xformers",
                "vit_giant_384": "vit_giant_xformers",
            }
            if config.model_name not in model_map:
                raise ValueError(
                    f"不支持的 model_name: {config.model_name}。"
                    f"支持的值: {list(model_map.keys())}"
                )
            arch_name = model_map[config.model_name]
            encoder = vit_encoder.__dict__[arch_name](**self._get_model_kwargs(config))

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_key = self._resolve_checkpoint_key(checkpoint, config)
        encoder_state_dict = _clean_backbone_key(checkpoint[ckpt_key])

        msg = encoder.load_state_dict(encoder_state_dict, strict=False)
        if msg.missing_keys:
            logging.warning(f"Encoder missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            logging.warning(f"Encoder unexpected keys: {msg.unexpected_keys}")

        return encoder
    
    def _get_model_kwargs(self, config):
        """
        获取模型初始化参数
        
        返回用于初始化 V-JEPA 2 模型的参数字典。
        这些参数与 V-JEPA 2 预训练时的配置保持一致。
        
        Args:
            config: VjepaTokenizerConfig 配置对象
        
        Returns:
            dict: 模型初始化参数字典
        """
        kwargs = dict(
            img_size=(config.img_size, config.img_size),  # 输入图像尺寸
            patch_size=config.patch_size,                  # 空间 patch 大小
            num_frames=config.num_frames,                   # 预训练时使用的帧数
            tubelet_size=config.tubelet_size,               # 时间维度压缩比例
            use_sdpa=True,                                 # 使用 Scaled Dot Product Attention（更高效）
            use_SiLU=False,                                # 不使用 SiLU 激活函数
            wide_SiLU=True,                                # 使用 wide SiLU（如果启用 SiLU）
            uniform_power=False,                           # 不使用均匀功率的位置编码
            use_rope=True,                                 # 使用 RoPE（Rotary Position Embedding）位置编码
        )
        if self._is_vjepa21:
            kwargs["img_temporal_dim_size"] = 1
            kwargs["interpolate_rope"] = True
        return kwargs
    
    def get_n_img_tokens(self) -> int:
        """
        返回每帧的图像 token 数
        
        此方法用于告诉 policy_transformer 每帧有多少个图像 tokens，
        以便正确计算序列长度和位置编码。
        
        注意：
        - V-JEPA 2 会压缩时间维度，实际输出 token 数 = (T // tubelet_size) * n_spatial_tokens
        - 例如：输入 T=200 帧，tubelet_size=2 -> 输出 100 个时间步，每个时间步 n_spatial_tokens 个 tokens
        - 但为了与 OpenP2P 的接口兼容，这里返回单帧时的 token 数（即 n_spatial_tokens）
        - 在 forward() 中，我们会通过时间维度扩展来确保输出形状为 [B, T, n_spatial_tokens, embed_dim]
        
        Returns:
            int: 每帧的图像 token 数（等于 n_spatial_tokens）
                例如：img_size=256, patch_size=16 -> 返回 256
        """
        return self.n_img_tokens
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 将视频帧编码为 token 序列
        
        处理流程：
        1. 检查并处理单帧输入（V-JEPA 2 需要至少 2 帧）
        2. 转换输入格式为 V-JEPA 2 期望的格式
        3. 调整输入分辨率（如果需要）
        4. 调用 V-JEPA 2 encoder 提取特征
        5. 处理时间维度压缩（V-JEPA 2 会压缩时间维度）
        6. 投影特征维度（如果需要）
        7. 扩展时间维度回原始帧数
        
        Args:
            input_tensor: 输入视频张量，形状为 [B, T, C, H, W]
                - B: batch size
                - T: 时间步数（帧数）
                - C: 通道数（通常是 3，RGB）
                - H: 帧高度
                - W: 帧宽度
        
        Returns:
            输出特征张量，形状为 [B, T, N, embed_dim]
                - B: batch size（与输入相同）
                - T: 时间步数（与输入相同，经过扩展后）
                - N: 每帧的 token 数（n_img_tokens = n_spatial_tokens）
                - embed_dim: embedding 维度（与 policy model 的 transformer_dim 一致）
        """
        B, T, C, H, W = input_tensor.shape
        
        # ========================================================================
        # 步骤 1: 处理单帧输入
        # ========================================================================
        # V-JEPA 2 的核心限制：需要 T >= tubelet_size（通常是 2）
        # 这是因为 V-JEPA 2 使用 PatchEmbed3D，其时间维度的卷积核大小为 tubelet_size
        # 如果 T < tubelet_size，会导致卷积核大于输入尺寸，从而报错
        #
        # 当前处理方案（训练和推理都适用）：
        # - 如果 T < tubelet_size（通常是单帧 T=1），复制最后一帧以满足最小要求
        # - 例如：T=1 -> 复制成 T=2，然后 V-JEPA 2 可以正常处理
        # - 后续在步骤 7 中，我们会将输出扩展回原始帧数（T=1）
        #
        # 注意：这个方案虽然能工作，但有一些限制：
        # 1. 单帧时复制帧会浪费一些计算（处理了2帧但只用1帧的结果）
        # 2. 对于推理场景，更好的方案是在推理流程中维护帧缓冲区（后续优化）
        #
        # 未来优化方向：
        # - 在推理时，维护一个帧缓冲区，始终保留至少 tubelet_size 帧
        # - 这样可以利用真实的时序信息，而不是复制帧
        # - 但这需要修改 inference.py 中的推理流程
        if T < self.tubelet_size:
            # 复制最后一帧，拼接在末尾
            # 例如：[B, 1, C, H, W] -> [B, 2, C, H, W]
            input_tensor = torch.cat([input_tensor, input_tensor[:, -1:]], dim=1)
            T_padded = input_tensor.shape[1]  # 填充后的帧数（例如：1 -> 2）
            need_slice = True  # 标记需要后续截取（因为原始输入是单帧）
        else:
            T_padded = T  # 不需要填充，直接使用原始帧数
            need_slice = False
        
        # ========================================================================
        # 步骤 2: 转换输入格式
        # ========================================================================
        # V-JEPA 2 期望的输入格式是 (B, C, T, H, W)，需要从 (B, T, C, H, W) permute
        # 将通道维度移到时间维度之前
        x = input_tensor.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        
        # ========================================================================
        # 步骤 3: 调整输入分辨率
        # ========================================================================
        # 如果输入尺寸与 V-JEPA 2 期望的不同，需要 resize
        # 例如：OpenP2P 使用 192x192，但 V-JEPA 2 ViT-L 期望 256x256
        if H != self.vjepa_config.img_size or W != self.vjepa_config.img_size:
            # 使用 bilinear 插值进行 resize
            # 先将 (B, C, T, H, W) reshape 成 (B*C, T, H, W) 以便使用 F.interpolate
            # 注意：x 在经过 permute 之后不再保证内存连续，这里使用 reshape 而不是 view，
            # 以避免 stride 不兼容导致的 RuntimeError。
            x = F.interpolate(
                x.reshape(B * C, T_padded, H, W),
                size=(self.vjepa_config.img_size, self.vjepa_config.img_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(
                B,
                C,
                T_padded,
                self.vjepa_config.img_size,
                self.vjepa_config.img_size,
            )
        
        # ========================================================================
        # 步骤 4: 调用 V-JEPA 2 encoder
        # ========================================================================
        # 根据是否冻结模型决定是否计算梯度
        # 如果模型被冻结（frozen=True），即使 self.training=True 也不计算梯度
        with torch.set_grad_enabled(self.training and not self.vjepa_config.frozen):
            # V-JEPA 2 encoder 输出: [B, N_total, D_vjepa]
            # 其中：
            #   - N_total = (T_padded // tubelet_size) * n_spatial_tokens
            #   - 例如：T_padded=200, tubelet_size=2, n_spatial_tokens=256
            #     -> N_total = (200//2) * 256 = 100 * 256 = 25600
            #   - D_vjepa = encoder.embed_dim（例如 ViT-Large 是 1024）
            features = self.encoder(x)
        
        # ========================================================================
        # 步骤 5: Reshape 输出以分离时间和空间维度
        # ========================================================================
        # 将 [B, N_total, D_vjepa] reshape 成 [B, T_temporal, n_spatial_tokens, D_vjepa]
        # 其中 T_temporal = T_padded // tubelet_size（压缩后的时间步数）
        # 例如：T_padded=200, tubelet_size=2 -> T_temporal=100
        T_temporal = T_padded // self.tubelet_size
        features = features.view(B, T_temporal, self.n_spatial_tokens, self.vjepa_embed_dim)
        
        # ========================================================================
        # 步骤 6: 投影特征维度（如果需要）
        # ========================================================================
        # 如果 V-JEPA 2 的输出维度与 policy model 的 embed_dim 不同，需要投影
        # 例如：ViT-Large 输出 1024 维，但 policy model 需要 768 维
        if self.proj_to_embed_dim is not None:
            features = self.proj_to_embed_dim(features)
        
        # ========================================================================
        # 步骤 7: 可选空间聚合（patch tokens -> 全局 token）
        # ========================================================================
        if self.pool_to_global:
            features = features.mean(dim=2, keepdim=True)
            features = self.global_token_norm(features)
            features = features + self.global_token_mlp(features)

        # ========================================================================
        # 步骤 8: 扩展时间维度回原始帧数
        # ========================================================================
        # 现在 features 是 [B, T_temporal, n_img_tokens, embed_dim]
        # 但 OpenP2P 期望的是 [B, T, n_img_tokens, embed_dim]，其中 T 是原始输入帧数
        # 由于 V-JEPA 2 压缩了时间维度（T_temporal = T_padded // tubelet_size），
        # 我们需要将时间维度扩展回原始帧数
        
        # 计算原始输入帧数（考虑是否复制了帧）
        if need_slice:
            original_T = 1  # 原始输入是单帧
        else:
            original_T = T  # 原始输入帧数
        
        # 将压缩后的时间步扩展回原始帧数
        # 方法：每个时间步重复 tubelet_size 次
        # 例如：T_temporal=100, tubelet_size=2 -> 扩展到 200
        #
        # 特殊情况处理（单帧输入）：
        # - 如果 original_T=1（单帧输入），T_temporal=1（因为 T_padded=2, 2//2=1）
        # - 扩展后：1 * 2 = 2，但 original_T=1，所以只取前1个
        # - 这样我们就得到了单帧对应的特征
        if T_temporal < original_T:
            # 正常情况：每个时间步重复 tubelet_size 次
            # features: [B, T_temporal, N, D] -> [B, T_temporal * tubelet_size, N, D]
            # 例如：[B, 100, 256, 1024] -> [B, 200, 256, 1024]
            features_expanded = features.repeat_interleave(self.tubelet_size, dim=1)
            
            # 如果扩展后还是不够（可能因为 original_T 不是 tubelet_size 的倍数），
            # 重复最后一个时间步来填充
            if features_expanded.shape[1] < original_T:
                n_missing = original_T - features_expanded.shape[1]
                features_expanded = torch.cat([
                    features_expanded,
                    features[:, -1:, :, :].repeat(1, n_missing, 1, 1)
                ], dim=1)
            elif features_expanded.shape[1] > original_T:
                # 如果扩展后超过了，只取前 original_T 个
                # 这通常发生在单帧输入时：扩展后是2，但 original_T=1，所以只取前1个
                features_expanded = features_expanded[:, :original_T, :, :]
        elif T_temporal > original_T:
            # 如果压缩后的时间步数大于原始帧数（理论上不应该发生），只取前 original_T 个
            features_expanded = features[:, :original_T, :, :]
        else:
            # T_temporal == original_T，不需要扩展
            # 理论上不应该发生（因为 tubelet_size > 1），但为了代码健壮性保留此分支
            features_expanded = features
        
        # ========================================================================
        # 验证输出形状
        # ========================================================================
        # 确保输出形状符合预期：[B, original_T, n_img_tokens, embed_dim]
        eager_assert(
            features_expanded.shape,
            (B, original_T, self.n_img_tokens, self.embed_dim),
        )
        
        return features_expanded
