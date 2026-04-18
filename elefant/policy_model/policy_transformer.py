from typing import List, Optional, Callable
from elefant.policy_model.transformer import TransformerConfig, MoETransformer
from elefant.policy_model.action_decoder import ActionDecoderConfig
from elefant.im_tokenizer import ImageBaseTokenizer
import torch
import torch.utils._pytree as pytree
from elefant.policy_model.transformer import (
    Transformer,
    TransformerSelfAttentionLayer,
    SparseMoEConfig,
)
from elefant.policy_model.kv_cache import RollingStepKVCache, KVCacheState
from elefant.torch import eager_assert
import torch.nn as nn
from elefant.policy_model.action_decoder import ActionDecoder
from torch.nn.attention import flex_attention as fa
from typing import Tuple


class PolicyCausalTransformerConfig(TransformerConfig):
    n_steps: int = 3
    n_thinking_tokens: int = 1
    # This needs to be set to the number of actions in the action mapping + 1
    n_action_tokens: int = 5
    # Number of future frames to predict actions for (F=1 degrades to original single-frame)
    n_future_action_tokens: int = 1
    mask_block_size: int = 128
    attention_history_len: List[int]
    n_kv_sink_tokens: int = 1

    action_decoder: ActionDecoderConfig
    text_token_size: int
    text_tokenizer_embed_dim: int
    # Set True for gamepad_direct mode: ActionDecoder is not used, skip init to
    # avoid DDP "unused parameters" errors.
    skip_action_decoder: bool = False


class MoEPolicyCausalTransformerConfig(PolicyCausalTransformerConfig):
    sparse_moe: SparseMoEConfig = SparseMoEConfig()


def _img_policy_causal_mask(
    layer_idx: int,
    n_img_tokens: int,
    n_thinking_tokens: int,
    n_action_tokens: int,
    history_len: Optional[int] = None,
    n_text_tokens: int = 0,
    n_kv_sink_tokens: int = 0,
):
    """
    This is a causal mask for the image policy transformer.
    Any image or thinking token can attend to any other image or thinking token but in order to causal with respect
    to actions the action tokens must be causally auto-regressive, also prior actions are masked out to encourage the
    model to learn causally.
    There are extra KV sink tokens at the start of the sequence, always attendable by query
    """

    def _mask(b, h, q_idx, kv_idx):
        is_sink = kv_idx < n_kv_sink_tokens

        # Shift non-sink KV positions into the base index space [0, max_seq_len)
        kv_idx_base = kv_idx - n_kv_sink_tokens

        one_step_len = (
            n_img_tokens + n_text_tokens + n_thinking_tokens + n_action_tokens
        )
        # However, within a single image it's fine to look ahead.
        token_to_action_out = n_img_tokens + n_thinking_tokens + n_text_tokens

        def _step_and_img_mask(idx):
            is_img_or_text_or_thinking_token = (idx % one_step_len) < (
                token_to_action_out
            )
            step_idx = idx // one_step_len

            return is_img_or_text_or_thinking_token, step_idx

        def _step_and_real_action_mask(idx):
            is_action_token = (idx % one_step_len) > (token_to_action_out)
            step_idx = idx // one_step_len

            return is_action_token, step_idx

        def _step_and_action_out_mask(idx):
            is_action_out_token = (idx % one_step_len) == (token_to_action_out)
            step_idx = idx // one_step_len

            return is_action_out_token, step_idx

        q_is_img, q_step_idx = _step_and_img_mask(q_idx)
        kv_is_img, kv_step_idx = _step_and_img_mask(kv_idx_base)
        q_is_real_action, _ = _step_and_real_action_mask(q_idx)
        kv_is_real_action, _ = _step_and_real_action_mask(kv_idx_base)
        q_is_action_out, _ = _step_and_action_out_mask(q_idx)
        kv_is_action_out, _ = _step_and_action_out_mask(kv_idx_base)
        full_mask = (
            (
                q_is_img & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # image attend to anything except for action out
            | (
                q_is_action_out & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # action out attend to anything except for action out
            | (
                q_is_real_action & ~kv_is_action_out & (kv_step_idx < q_step_idx)
            )  # real action attend to anything except for action out
            | (q_is_img & kv_is_img & (kv_step_idx == q_step_idx))  # img attend to img
            | (
                q_is_action_out
                & (kv_is_img | kv_is_action_out)
                & (kv_step_idx == q_step_idx)
            )  # action out attend to action out or img
            | (
                q_is_real_action & ~kv_is_action_out & (kv_step_idx == q_step_idx)
            )  # real action attend to anything except for action out
        )

        if history_len is not None:
            history_mask = (q_step_idx - kv_step_idx) <= history_len
            full_mask = full_mask & history_mask
        return is_sink | full_mask

    return _mask


def _img_policy_causal_mask_multi_future(
    layer_idx: int,
    n_img_tokens: int,
    n_thinking_tokens: int,
    n_action_tokens: int,
    n_future_action_tokens: int = 1,
    history_len: Optional[int] = None,
    n_text_tokens: int = 0,
    n_kv_sink_tokens: int = 0,
):
    """
    Causal mask for multi-future-action prediction.

    Each step layout:
      [text(n_text) | img(n_img) | think(n_think) | a_in^0 ... a_in^{F-1} | a1 ... aN]
       ^--- img_think region ---^  ^--- action_out region (F tokens) ---^  ^-- real_action --^

    Rules (same principle as _img_policy_causal_mask, action_out never used as KV):
      - img/think Q:    cross-step sees img/think + real_action (no action_out as KV)
                        same-step sees img/think only
      - a_in^f Q:       cross-step sees img/think + real_action (no action_out as KV)
                        same-step sees img/think + a_in^0..f (causal within action_out)
      - real_action Q:  cross-step sees img/think + real_action (no action_out as KV)
                        same-step sees img/think + real_action (causal), NO action_out
    """

    def _mask(b, h, q_idx, kv_idx):
        is_sink = kv_idx < n_kv_sink_tokens
        kv_idx_base = kv_idx - n_kv_sink_tokens

        one_step_len = (
            n_img_tokens + n_text_tokens + n_thinking_tokens
            + n_future_action_tokens
            + n_action_tokens
        )
        # position boundary: [0, token_to_first_ao) = img/think/text region
        token_to_first_ao = n_img_tokens + n_thinking_tokens + n_text_tokens
        # position boundary: [token_to_first_ao, token_to_last_ao) = action_out region
        token_to_last_ao = token_to_first_ao + n_future_action_tokens

        def _classify(idx):
            pos = idx % one_step_len
            step = idx // one_step_len
            is_img_think = pos < token_to_first_ao
            is_ao = (pos >= token_to_first_ao) & (pos < token_to_last_ao)
            ao_sub = pos - token_to_first_ao   # relative index within action_out region
            is_real = pos >= token_to_last_ao
            return is_img_think, is_ao, ao_sub, is_real, step

        q_is_img, q_is_ao, q_ao_sub, q_is_real, q_step = _classify(q_idx)
        kv_is_img, kv_is_ao, kv_ao_sub, kv_is_real, kv_step = _classify(kv_idx_base)

        # Cross-step: all Q types see img/think + real_action from history, never action_out
        cross_step = (kv_step < q_step) & ~kv_is_ao

        # Same-step: img/think sees only img/think
        same_step_img = q_is_img & kv_is_img & (kv_step == q_step)

        # Same-step: a_in^f sees img/think + a_in^0..f (causal)
        same_step_ao = (
            q_is_ao
            & (kv_step == q_step)
            & (kv_is_img | (kv_is_ao & (kv_ao_sub <= q_ao_sub)))
        )

        # Same-step: real_action sees img/think + real_action (causal by position), NOT action_out
        same_step_real = (
            q_is_real
            & (kv_step == q_step)
            & ~kv_is_ao
            & ((q_idx % one_step_len) >= (kv_idx_base % one_step_len))
        )

        full_mask = cross_step | same_step_img | same_step_ao | same_step_real

        if history_len is not None:
            full_mask = full_mask & ((q_step - kv_step) <= history_len)

        return is_sink | full_mask

    return _mask


def _img_policy_future_causal_mask(
    layer_idx: int,
    n_img_tokens: int,
    n_thinking_tokens: int,
    n_action_tokens: int,
    history_len: Optional[int] = None,
    n_text_tokens: int = 0,
    n_kv_sink_tokens: int = 0,
):
    """
    Future-policy causal mask with two special output tokens per step:
    [text, img, thinking, s0, a0, real_action_tokens...]
    """

    def _mask(b, h, q_idx, kv_idx):
        is_sink = kv_idx < n_kv_sink_tokens
        kv_idx_base = kv_idx - n_kv_sink_tokens

        one_step_len = n_img_tokens + n_text_tokens + n_thinking_tokens + n_action_tokens
        token_to_state_out = n_img_tokens + n_thinking_tokens + n_text_tokens
        token_to_action_out = token_to_state_out + 1

        def _step_and_img_mask(idx):
            is_img_or_text_or_thinking = (idx % one_step_len) < token_to_state_out
            step_idx = idx // one_step_len
            return is_img_or_text_or_thinking, step_idx

        def _step_and_state_out_mask(idx):
            is_state_out = (idx % one_step_len) == token_to_state_out
            step_idx = idx // one_step_len
            return is_state_out, step_idx

        def _step_and_action_out_mask(idx):
            is_action_out = (idx % one_step_len) == token_to_action_out
            step_idx = idx // one_step_len
            return is_action_out, step_idx

        def _step_and_real_action_mask(idx):
            is_real_action = (idx % one_step_len) > token_to_action_out
            step_idx = idx // one_step_len
            return is_real_action, step_idx

        q_is_img, q_step_idx = _step_and_img_mask(q_idx)
        kv_is_img, kv_step_idx = _step_and_img_mask(kv_idx_base)
        q_is_state_out, _ = _step_and_state_out_mask(q_idx)
        kv_is_state_out, _ = _step_and_state_out_mask(kv_idx_base)
        q_is_action_out, _ = _step_and_action_out_mask(q_idx)
        kv_is_action_out, _ = _step_and_action_out_mask(kv_idx_base)
        q_is_real_action, _ = _step_and_real_action_mask(q_idx)

        full_mask = (
            (
                q_is_img
                & ~kv_is_action_out
                & ~kv_is_state_out
                & (kv_step_idx < q_step_idx)
            )
            | (
                q_is_state_out
                & ~kv_is_action_out
                & ~kv_is_state_out
                & (kv_step_idx < q_step_idx)
            )
            | (
                q_is_action_out
                & ~kv_is_action_out
                & ~kv_is_state_out
                & (kv_step_idx < q_step_idx)
            )
            | (
                q_is_real_action
                & ~kv_is_action_out
                & ~kv_is_state_out
                & (kv_step_idx < q_step_idx)
            )
            | (q_is_img & kv_is_img & (kv_step_idx == q_step_idx))
            | (
                q_is_state_out
                & (kv_is_img | kv_is_state_out)
                & (kv_step_idx == q_step_idx)
            )
            | (
                q_is_action_out
                & (kv_is_img | kv_is_state_out | kv_is_action_out)
                & (kv_step_idx == q_step_idx)
            )
            | (
                q_is_real_action
                & ~kv_is_action_out
                & ~kv_is_state_out
                & (kv_step_idx == q_step_idx)
            )
        )

        if history_len is not None:
            history_mask = (q_step_idx - kv_step_idx) <= history_len
            full_mask = full_mask & history_mask
        return is_sink | full_mask

    return _mask


class PolicyCausalTransformer(torch.nn.Module):
    def __init__(
        self,
        config: PolicyCausalTransformerConfig,
        image_tokenizer: ImageBaseTokenizer,
        inference_mode: bool = False,
        mask_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.config = config

        self._construct_transformer()

        self.image_tokenizer = image_tokenizer
        self.text_token_size = self.config.text_token_size
        self.text_tokenizer_embed_dim = self.config.text_tokenizer_embed_dim

        # n_future_action_tokens action_out tokens per step, the rest are real action tokens
        self.max_seq_len = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + self.config.n_future_action_tokens  # F action_out tokens (a_in^0..F-1)
            + self.config.n_action_tokens
        ) * config.n_steps
        self._transformer.max_seq_len = self.max_seq_len

        self.n_tokens_to_first_action = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + self.config.n_future_action_tokens
        )

        if mask_fn is None:
            if config.n_future_action_tokens > 1:
                mask_fn = _img_policy_causal_mask_multi_future
            else:
                mask_fn = _img_policy_causal_mask
        self._mask_fn = mask_fn

        self.embedding_std = 0.05

        self.text_embed_mlp = nn.Linear(
            self.text_tokenizer_embed_dim, config.embed_dim, bias=False
        )

        # Img token position makers.
        self.img_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.image_tokenizer.get_n_img_tokens(),
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        torch.nn.init.normal_(self.img_pos_tokens, mean=0.0, std=self.embedding_std)

        self.action_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.config.n_action_tokens,
                config.action_decoder.embed_dim,
                dtype=torch.bfloat16,
            )
        )

        torch.nn.init.normal_(self.action_pos_tokens, mean=0.0, std=self.embedding_std)

        self.text_pos_tokens = nn.Parameter(
            torch.empty(
                1,
                self.text_token_size,
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        self.text_embedding_for_no_text_input = nn.Parameter(
            torch.empty(
                1,
                self.text_token_size,
                config.embed_dim,
                dtype=torch.bfloat16,
            )
        )
        torch.nn.init.normal_(
            self.text_embedding_for_no_text_input, mean=0.0, std=self.embedding_std
        )
        torch.nn.init.normal_(self.text_pos_tokens, mean=0.0, std=self.embedding_std)

        # Thinking token position makers.
        self.thinking_pos_tokens = nn.Parameter(
            torch.empty(
                1, self.config.n_thinking_tokens, config.embed_dim, dtype=torch.bfloat16
            )
        )
        torch.nn.init.normal_(
            self.thinking_pos_tokens, mean=0.0, std=self.embedding_std
        )

        # F independent learned tokens, one per future frame to predict (a_in^0..F-1)
        self.action_out_tokens = nn.Parameter(
            torch.empty(1, self.config.n_future_action_tokens, config.embed_dim, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.action_out_tokens, mean=0.0, std=self.embedding_std)

        # ActionDecoder is only used in autoregressive modes (gamepad, keyboard_mouse).
        # In gamepad_direct mode it is skipped entirely (skip_action_decoder=True),
        # so we avoid initialising it to prevent DDP "unused parameters" errors.
        if config.skip_action_decoder:
            self.action_decoder = None
        else:
            self.action_decoder = ActionDecoder(
                cfg=config.action_decoder,
            )

        # Pre-compute indices of transformer output for each (step, future_frame) pair.
        # output_action_token_idx shape: [n_steps * n_future_action_tokens] (flat)
        # Reshaped to [n_steps, n_future_action_tokens] when used.
        F = self.config.n_future_action_tokens
        output_action_token_idx = []
        for i in range(self.config.n_steps):
            step_start_idx = i * (
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + F
                + self.config.n_action_tokens
            )
            base_ao_idx = (
                step_start_idx
                + self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
            )
            for f in range(F):
                output_action_token_idx.append(base_ao_idx + f)
        self.register_buffer(
            "output_action_token_idx",
            torch.tensor(output_action_token_idx, dtype=torch.long),
            persistent=False,
        )

        self._transformer.construct_transformer_layers()

        # After constructing layers, assign appropriate masks to each layer
        self._assign_layer_masks()

    def _construct_transformer(self):
        self._transformer = Transformer(config=self.config)

    def block_mask_to_device(self, device):
        for layer in self._transformer.transformer_layers:
            if layer.self_attention.block_mask is not None:
                layer.self_attention.block_mask = layer.self_attention.block_mask.to(
                    device
                )

    def rebuild_rope_cache(self, inference_seq_len: int):
        self._transformer.rebuild_rope_cache(inference_seq_len)

    def init_kv_cache_state(self):
        return self._transformer.init_kv_cache_state()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def kv_cache(self):
        return self._transformer.kv_cache

    def _assign_layer_masks(self):
        """
        Assign layer-specific masks based on the per-layer allowed frame counts specified in
        self.config.attention_history_len
        """
        assert len(self.config.attention_history_len) == len(
            self._transformer.transformer_layers
        ), "Number of transformer layers and list of attention n frames should match"
        assert max(self.config.attention_history_len) == self.config.n_steps, (
            "Max attention history len should be equal to the n_steps"
        )
        self._transformer._layer_mask_fns = []

        F = self.config.n_future_action_tokens
        for i, layer in enumerate(self._transformer.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                self._transformer._layer_mask_fns.append(None)
                continue
            allowed_frames = self.config.attention_history_len[i]

            # _img_policy_causal_mask_multi_future takes n_future_action_tokens explicitly;
            # legacy _img_policy_causal_mask folds action_out into n_action_tokens (F=1 path).
            if F > 1:
                layer_mask_fn = self._mask_fn(
                    layer_idx=i,
                    n_img_tokens=self.image_tokenizer.get_n_img_tokens(),
                    n_thinking_tokens=self.config.n_thinking_tokens,
                    n_action_tokens=self.config.n_action_tokens,
                    n_future_action_tokens=F,
                    history_len=allowed_frames,
                    n_kv_sink_tokens=self.config.n_kv_sink_tokens,
                    n_text_tokens=self.text_token_size,
                )
            else:
                layer_mask_fn = self._mask_fn(
                    layer_idx=i,
                    n_img_tokens=self.image_tokenizer.get_n_img_tokens(),
                    n_thinking_tokens=self.config.n_thinking_tokens,
                    n_action_tokens=1 + self.config.n_action_tokens,
                    history_len=allowed_frames,
                    n_kv_sink_tokens=self.config.n_kv_sink_tokens,
                    n_text_tokens=self.text_token_size,
                )
            self._transformer._layer_mask_fns.append(layer_mask_fn)

            layer.self_attention.block_mask = fa.create_block_mask(
                layer_mask_fn,
                B=None,
                H=None,
                Q_LEN=self.max_seq_len,
                KV_LEN=self.max_seq_len + self.config.n_kv_sink_tokens,
                BLOCK_SIZE=self.config.mask_block_size,
                device=self.device,
            )

    @property
    def step_size(self):
        return self.max_seq_len // self.config.n_steps

    def setup_kv_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ):
        # For now at least we assume the length and head is the same at every layer so we only need on kv cache
        # (but separate state for each layer).
        self._transformer.kv_cache = RollingStepKVCache(
            batch_size=batch_size,
            step_size=self.step_size,
            max_T=self.config.n_steps,
            num_kv_heads=self.config.n_kv_head,
            embed_size_per_head=self.config.embed_dim // self.config.n_kv_head,
            device=device,
            dtype=dtype,
        )
        # Tell all the self attention layers about the kv cache object
        # Note this object does not contain kv state, just config for manipulating the cache.
        for i, layer in enumerate(self._transformer.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                continue

            sa = layer.self_attention
            sa.kv_cache = self._transformer.kv_cache
            sa._mask_fn = self._transformer._layer_mask_fns[i]
            # Precompute decode masks only if we have a block mask and a concrete mask fn
            if sa.block_mask is not None and sa._mask_fn is not None:
                sa.precompute_decode_masks(
                    q_seq_length=self.n_tokens_to_first_action
                    + self.config.n_action_tokens
                )

    def _impute_no_text_embedding(
        self, text_tokens_embed: torch.Tensor
    ) -> torch.Tensor:
        BxT, action_tokens, dim = text_tokens_embed.shape
        text_tokens_embed_reshaped = text_tokens_embed.reshape(BxT, -1)
        no_text_input_mask = ~torch.any(text_tokens_embed_reshaped, dim=1)
        broadcast_no_text_input_mask = no_text_input_mask.reshape(BxT, 1, 1)
        imputed_text_tokens_embed = torch.where(
            broadcast_no_text_input_mask,
            self.text_embedding_for_no_text_input,
            text_tokens_embed,
        )
        return imputed_text_tokens_embed.reshape(BxT, action_tokens, dim)

    def online_forward(
        self,
        img: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        idx: torch.Tensor,
        kv_cache_state: Optional[List[KVCacheState]] = None,
        should_grow_cache: bool = None,
        action_sampler: Callable = None,
        empty_sampled_action_fn: Callable = None,
        reshape_structured_action_fn: Callable = None,
        action_in_to_tokens_fn: Callable = None,
        direct_action_head_fn: Callable = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with KV cache for inference.

        Two-pass design:
          Pass 1 (dummy real-actions): get a_in^0 output -> sample current-frame action.
          Pass 2 (real action embeddings): write sampled action into KV cache for next step.
        Only a_in^0 is used for action sampling; a_in^1..F-1 are present but ignored at inference.
        """
        B, T, C, H, W = img.shape
        eager_assert(T, 1)
        eager_assert(B, 1)

        F = self.config.n_future_action_tokens
        step_len = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + F
            + self.config.n_action_tokens
        )

        img_tokens = self.image_tokenizer(img)
        eager_assert(
            img_tokens.shape,
            (B, T, self.image_tokenizer.get_n_img_tokens(), self.config.embed_dim),
        )
        eager_assert(
            text_tokens_embed.shape,
            (self.text_token_size, self.text_tokenizer_embed_dim),
        )
        img_tokens = img_tokens.view(
            B, self.image_tokenizer.get_n_img_tokens(), self.config.embed_dim
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens
        text_tokens_embed = text_tokens_embed.view(
            B, self.text_token_size, self.text_tokenizer_embed_dim
        ).to(self.device)
        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed_with_pos = text_tokens_embed + self.text_pos_tokens

        input_pos = torch.arange(step_len, device=idx.device, dtype=idx.dtype) + idx

        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        batch_action_out_tokens = self.action_out_tokens.repeat(B, 1, 1)  # [B, F, D]
        dummy_action_embeddings_in = torch.zeros(
            B, self.config.n_action_tokens, self.config.embed_dim,
            device=batch_action_out_tokens.device,
        )

        # ── Pass 1: dummy real-actions, get a_in^0 output ──
        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_action_out_tokens,
                dummy_action_embeddings_in,
            ],
            dim=1,
        )
        eager_assert(x.shape, (B, step_len, self.config.embed_dim))

        y, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )
        eager_assert(y.shape, (B, step_len, self.config.embed_dim))

        # a_in^0 is at position: n_img + n_text + n_think (first of the F action_out tokens)
        action_out_position = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
        )
        action_token_out = y[:, action_out_position : action_out_position + 1]
        eager_assert(action_token_out.shape, (B, 1, self.config.embed_dim))

        if direct_action_head_fn is not None:
            # Direct mode: MLP → 3 heads, no autoregressive decoding
            sampled_action = direct_action_head_fn(action_token_out)
        else:
            # Sample current-frame action from a_in^0
            sampled_action = self.action_decoder.autogressive_sample(
                action_token_out,
                action_sampler,
                empty_sampled_action_fn,
                reshape_structured_action_fn,
                inference_mode=True,
            )
        sampled_action_reshaped = pytree.tree_map(
            lambda x: x.unsqueeze(0), sampled_action
        )

        # ── Pass 2: fill in real action embeddings, update KV cache ──
        action_embeddings_in = action_in_to_tokens_fn(sampled_action_reshaped)
        action_embeddings_in = action_embeddings_in.view(
            B, self.config.n_action_tokens, self.config.embed_dim
        )
        action_embeddings_in_with_pos = action_embeddings_in + self.action_pos_tokens
        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_action_out_tokens,
                action_embeddings_in_with_pos,
            ],
            dim=1,
        )
        eager_assert(x.shape, (B, step_len, self.config.embed_dim))

        y_new, kv_cache_state, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )
        eager_assert(y_new.shape, (B, step_len, self.config.embed_dim))
        next_idx = input_pos[-1] + 1

        return sampled_action, next_idx, kv_cache_state

    def forward(
        self,
        img: torch.Tensor,
        action_embeddings_in: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        should_grow_cache: bool = None,
        input_pos: Optional[torch.Tensor] = None,
        future_offsets: Optional[List[int]] = None,
        skip_action_decoder: bool = False,
    ):
        """
        output:
        - action_out: (B, T, F, n_actions-1, D): decoded action embeddings for each future offset
        - action_out_tokens: (B, T, F, D): transformer output at a_in positions
        - auxiliary_losses, auxiliary_outputs

        future_offsets: list of F ints, offset[0] must be 0 (current frame).
            e.g. [0, 6, 15] means a_in^0->t, a_in^1->t+6, a_in^2->t+15.
            Defaults to [0, 1, ..., F-1] (consecutive frames).
        """
        B, T, *_ = img.shape
        img_tokens = self.image_tokenizer(img)
        eager_assert(
            text_tokens_embed.shape,
            (
                B,
                T,
                self.text_token_size,
                self.text_tokenizer_embed_dim,
            ),
        )
        eager_assert(
            img_tokens.shape,
            (
                B,
                T,
                self.image_tokenizer.get_n_img_tokens(),
                self.config.embed_dim,
            ),
        )
        eager_assert(
            action_embeddings_in.shape,
            (
                B,
                T,
                self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens.unsqueeze(0)

        action_tokens_with_pos = (
            action_embeddings_in + self.action_pos_tokens.unsqueeze(0)
        )
        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B * T, self.text_token_size, self.config.embed_dim
        )
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B, T, self.text_token_size, self.config.embed_dim
        )
        text_embeddings_with_pos = text_tokens_embed + self.text_pos_tokens

        # Ok, now we are ready to concat the input together of img, thinking, action
        x = []
        # Need to repeat the thinking token along the batch dimension.
        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        eager_assert(
            batch_thinking_pos_tokens.shape,
            (
                B,
                self.config.n_thinking_tokens,
                self.config.embed_dim,
            ),
        )
        batch_action_out_pos_tokens = self.action_out_tokens.repeat(B, 1, 1)
        eager_assert(
            batch_action_out_pos_tokens.shape,
            (B, self.config.n_future_action_tokens, self.config.embed_dim),
        )

        for i in range(self.config.n_steps):
            this_img = img_tokens_with_pos[:, i]
            this_actions = action_tokens_with_pos[:, i]
            this_text = text_embeddings_with_pos[:, i]
            eager_assert(
                this_img.shape,
                (
                    B,
                    self.image_tokenizer.get_n_img_tokens(),
                    self.config.embed_dim,
                ),
            )
            eager_assert(
                this_actions.shape,
                (
                    B,
                    self.config.n_action_tokens,
                    self.config.embed_dim,
                ),
            )
            eager_assert(
                this_text.shape,
                (B, self.text_token_size, self.config.embed_dim),
            )

            this_step_in = torch.cat(
                [
                    this_text,
                    this_img,
                    batch_thinking_pos_tokens,
                    batch_action_out_pos_tokens,  # [B, F, D]
                    this_actions,
                ],
                dim=1,
            )
            eager_assert(
                this_step_in.shape,
                (
                    B,
                    self.image_tokenizer.get_n_img_tokens()
                    + self.text_token_size
                    + self.config.n_thinking_tokens
                    + self.config.n_future_action_tokens
                    + self.config.n_action_tokens,
                    self.config.embed_dim,
                ),
            )

            x.append(this_step_in)

        x = torch.cat(x, dim=1)

        eager_assert(
            x.shape,
            (B, self.max_seq_len, self.config.embed_dim),
        )
        y, _, auxiliary_losses, auxiliary_outputs = self._transformer.forward(
            x, input_pos=input_pos, should_grow_cache=should_grow_cache
        )
        eager_assert(
            y.shape,
            (B, self.max_seq_len, self.config.embed_dim),
        )

        # Select all action_out positions: [B, n_steps * F, D] -> [B, n_steps, F, D]
        F = self.config.n_future_action_tokens
        action_out_tokens_flat = torch.index_select(
            y, dim=1, index=self.output_action_token_idx
        )
        action_out_tokens = action_out_tokens_flat.view(
            B, self.config.n_steps, F, self.config.embed_dim
        )
        eager_assert(
            action_out_tokens.shape,
            (B, self.config.n_steps, F, self.config.embed_dim),
        )

        # Reuse ActionDecoder for each future frame f.
        # future_offsets[f] is the frame offset for a_in^f.
        # offset=0 -> GT_action[t], offset=k -> GT_action[t+k] (pad zeros beyond T).
        T = self.config.n_steps
        n_act = self.config.n_action_tokens
        D = self.config.embed_dim

        # Default: consecutive offsets [0, 1, ..., F-1]
        if future_offsets is None:
            future_offsets = list(range(F))
        assert len(future_offsets) == F, f"future_offsets length {len(future_offsets)} != F {F}"
        assert future_offsets[0] == 0, "future_offsets[0] must be 0 (current frame)"

        # When skip_action_decoder=True (gamepad_direct mode), return the raw
        # transformer a_in outputs directly — no autoregressive decoding.
        if skip_action_decoder:
            # action_out_tokens: [B, T, F, D] — each a_in^f token is the full representation
            return action_out_tokens, action_out_tokens, auxiliary_losses, auxiliary_outputs

        # Flatten (B, T, F) -> (B, T*F, D) for a single ActionDecoder call
        input_action_token_for_decoder = action_out_tokens.reshape(B, T * F, D)

        # Build shifted action embeddings per offset
        shifted_parts = []
        for offset in future_offsets:
            if offset == 0:
                shifted_parts.append(action_embeddings_in)  # [B, T, n_act, D]
            else:
                # shift left by offset, pad with zeros on the right
                valid = action_embeddings_in[:, offset:, :, :]  # [B, T-offset, n_act, D]
                pad = torch.zeros(
                    B, offset, n_act, D,
                    device=action_embeddings_in.device,
                    dtype=action_embeddings_in.dtype,
                )
                shifted_parts.append(torch.cat([valid, pad], dim=1))  # [B, T, n_act, D]

        # Stack -> [B, T, F, n_act, D], reshape -> [B, T*F, n_act, D]
        action_embeddings_shifted = torch.stack(shifted_parts, dim=2).reshape(
            B, T * F, n_act, D
        )

        action_out = self.action_decoder(
            action_embeddings_in=action_embeddings_shifted,
            input_action_token=input_action_token_for_decoder,
        )
        # action_out: [B, T*F, n_actions-1, D] -> [B, T, F, n_actions-1, D]
        action_out = action_out.view(B, T, F, -1, D)

        return action_out, action_out_tokens, auxiliary_losses, auxiliary_outputs


class MoEPolicyCausalTransformer(PolicyCausalTransformer):
    def _construct_transformer(self):
        self._transformer = MoETransformer(config=self.config)


class PolicyFutureCausalTransformer(PolicyCausalTransformer):
    """
    Policy Transformer with future vision prediction head.
    
    Compared to PolicyCausalTransformer, this model adds a head to predict
    the visual representation of the next frame from the current frames' a⁰.
    """

    def __init__(
        self,
        config: PolicyCausalTransformerConfig,
        image_tokenizer: ImageBaseTokenizer,
        inference_mode: bool = False,
        mask_fn: Optional[Callable] = None,
        future_vision_loss_weight: float = 0.1,
    ):
        if mask_fn is None:
            mask_fn = _img_policy_future_causal_mask
        super().__init__(config, image_tokenizer, inference_mode, mask_fn)

        self.future_vision_loss_weight = future_vision_loss_weight
        self.state_out_token = nn.Parameter(
            torch.empty(1, 1, config.embed_dim, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.state_out_token, mean=0.0, std=self.embedding_std)

        # Two output tokens per step in future mode: s0 + a0.
        self.max_seq_len = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + 2
            + self.config.n_action_tokens
        ) * config.n_steps
        self._transformer.max_seq_len = self.max_seq_len
        self._transformer.rebuild_rope_cache(self.max_seq_len)
        self.n_tokens_to_first_action = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + 2
        )

        output_action_token_idx = []
        output_state_token_idx = []
        for i in range(self.config.n_steps):
            step_start_idx = i * (
                self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 2
                + self.config.n_action_tokens
            )
            state_idx = (
                step_start_idx
                + self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
            )
            output_state_token_idx.append(state_idx)
            output_action_token_idx.append(state_idx + 1)

        self.output_action_token_idx = torch.tensor(
            output_action_token_idx, dtype=torch.long
        )
        self.register_buffer(
            "output_state_token_idx",
            torch.tensor(output_state_token_idx, dtype=torch.long),
            persistent=False,
        )
        self._assign_layer_masks()

        self.future_vision_head = nn.Linear(
            config.embed_dim,
            config.embed_dim,
            bias=False,
        )
        torch.nn.init.normal_(self.future_vision_head.weight, mean=0.0, std=0.02)

    def _assign_layer_masks(self):
        assert len(self.config.attention_history_len) == len(
            self._transformer.transformer_layers
        ), "Number of transformer layers and list of attention n frames should match"
        assert max(self.config.attention_history_len) == self.config.n_steps, (
            "Max attention history len should be equal to the n_steps"
        )
        self._transformer._layer_mask_fns = []

        for i, layer in enumerate(self._transformer.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                self._transformer._layer_mask_fns.append(None)
                continue
            allowed_frames = self.config.attention_history_len[i]
            layer_mask_fn = self._mask_fn(
                layer_idx=i,
                n_img_tokens=self.image_tokenizer.get_n_img_tokens(),
                n_thinking_tokens=self.config.n_thinking_tokens,
                n_action_tokens=2 + self.config.n_action_tokens,
                history_len=allowed_frames,
                n_kv_sink_tokens=self.config.n_kv_sink_tokens,
                n_text_tokens=self.text_token_size,
            )
            self._transformer._layer_mask_fns.append(layer_mask_fn)
            layer.self_attention.block_mask = fa.create_block_mask(
                layer_mask_fn,
                B=None,
                H=None,
                Q_LEN=self.max_seq_len,
                KV_LEN=self.max_seq_len + self.config.n_kv_sink_tokens,
                BLOCK_SIZE=self.config.mask_block_size,
                device=self.device,
            )

    def forward(
        self,
        img: torch.Tensor,
        action_embeddings_in: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        should_grow_cache: bool = None,
        input_pos: Optional[torch.Tensor] = None,
        skip_action_decoder: bool = False,
    ):
        B, T, *_ = img.shape
        img_tokens = self.image_tokenizer(img)
        eager_assert(
            text_tokens_embed.shape,
            (
                B,
                T,
                self.text_token_size,
                self.text_tokenizer_embed_dim,
            ),
        )
        eager_assert(
            img_tokens.shape,
            (
                B,
                T,
                self.image_tokenizer.get_n_img_tokens(),
                self.config.embed_dim,
            ),
        )
        eager_assert(
            action_embeddings_in.shape,
            (
                B,
                T,
                self.config.n_action_tokens,
                self.config.embed_dim,
            ),
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens.unsqueeze(0)
        action_tokens_with_pos = action_embeddings_in + self.action_pos_tokens.unsqueeze(0)

        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B * T, self.text_token_size, self.config.embed_dim
        )
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed = text_tokens_embed.reshape(
            B, T, self.text_token_size, self.config.embed_dim
        )
        text_embeddings_with_pos = text_tokens_embed + self.text_pos_tokens

        x = []
        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        batch_state_out_pos_token = self.state_out_token.repeat(B, 1, 1)
        batch_action_out_pos_token = self.action_out_token.repeat(B, 1, 1)
        for i in range(self.config.n_steps):
            this_step_in = torch.cat(
                [
                    text_embeddings_with_pos[:, i],
                    img_tokens_with_pos[:, i],
                    batch_thinking_pos_tokens,
                    batch_state_out_pos_token,
                    batch_action_out_pos_token,
                    action_tokens_with_pos[:, i],
                ],
                dim=1,
            )
            x.append(this_step_in)
        x = torch.cat(x, dim=1)
        eager_assert(x.shape, (B, self.max_seq_len, self.config.embed_dim))

        y, _, auxiliary_losses, auxiliary_outputs = self._transformer.forward(
            x, input_pos=input_pos, should_grow_cache=should_grow_cache
        )
        eager_assert(y.shape, (B, self.max_seq_len, self.config.embed_dim))

        action_out_tokens = torch.index_select(
            y, dim=1, index=self.output_action_token_idx
        )
        state_out_tokens = torch.index_select(
            y, dim=1, index=self.output_state_token_idx
        )

        if skip_action_decoder:
            # Direct mode: return raw transformer a_in output, no ActionDecoder
            action_out_tokens_reshaped = action_out_tokens.unsqueeze(1)  # [B, 1, D] -> [B, 1, 1, D] (T=1, F=1)
            future_vision_pred = self.future_vision_head(state_out_tokens)
            return action_out_tokens_reshaped, action_out_tokens, future_vision_pred, auxiliary_losses, auxiliary_outputs

        action_out = self.action_decoder(
            action_embeddings_in=action_embeddings_in,
            input_action_token=action_out_tokens,
        )
        future_vision_pred = self.future_vision_head(state_out_tokens)
        eager_assert(
            future_vision_pred.shape,
            (
                state_out_tokens.shape[0],
                state_out_tokens.shape[1],
                self.config.embed_dim,
            ),
        )
        return action_out, action_out_tokens, future_vision_pred, auxiliary_losses, auxiliary_outputs

    def online_forward(
        self,
        img: torch.Tensor,
        text_tokens_embed: torch.Tensor,
        idx: torch.Tensor,
        kv_cache_state: Optional[List[KVCacheState]] = None,
        should_grow_cache: bool = None,
        action_sampler: Callable = None,
        empty_sampled_action_fn: Callable = None,
        reshape_structured_action_fn: Callable = None,
        action_in_to_tokens_fn: Callable = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C, H, W = img.shape
        eager_assert(T, 1)
        eager_assert(B, 1)

        img_tokens = self.image_tokenizer(img)
        img_tokens = img_tokens.view(
            B, self.image_tokenizer.get_n_img_tokens(), self.config.embed_dim
        )
        img_tokens_with_pos = img_tokens + self.img_pos_tokens
        text_tokens_embed = text_tokens_embed.view(
            B, self.text_token_size, self.text_tokenizer_embed_dim
        ).to(self.device)
        text_tokens_embed = self.text_embed_mlp(text_tokens_embed)
        text_tokens_embed = self._impute_no_text_embedding(text_tokens_embed)
        text_tokens_embed_with_pos = text_tokens_embed + self.text_pos_tokens

        input_pos = (
            torch.arange(
                +self.image_tokenizer.get_n_img_tokens()
                + self.text_token_size
                + self.config.n_thinking_tokens
                + 2
                + self.config.n_action_tokens,
                device=idx.device,
                dtype=idx.dtype,
            )
            + idx
        )

        batch_thinking_pos_tokens = self.thinking_pos_tokens.repeat(B, 1, 1)
        batch_state_out_token = self.state_out_token.repeat(B, 1, 1)
        batch_action_out_token = self.action_out_token.repeat(B, 1, 1)
        dummy_action_embeddings_in = torch.zeros(
            B,
            self.config.n_action_tokens,
            self.config.embed_dim,
            device=batch_action_out_token.device,
        )
        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_state_out_token,
                batch_action_out_token,
                dummy_action_embeddings_in,
            ],
            dim=1,
        )
        y, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )

        action_out_position = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
            + 1
        )
        action_token_out = y[:, action_out_position : action_out_position + 1]
        sampled_action = self.action_decoder.autogressive_sample(
            action_token_out,
            action_sampler,
            empty_sampled_action_fn,
            reshape_structured_action_fn,
            inference_mode=True,
        )
        sampled_action_reshaped = pytree.tree_map(
            lambda x: x.unsqueeze(0), sampled_action
        )
        action_embeddings_in = action_in_to_tokens_fn(sampled_action_reshaped)
        action_embeddings_in = action_embeddings_in.view(
            B, self.config.n_action_tokens, self.config.embed_dim
        )
        action_embeddings_in_with_pos = action_embeddings_in + self.action_pos_tokens
        x = torch.cat(
            [
                text_tokens_embed_with_pos,
                img_tokens_with_pos,
                batch_thinking_pos_tokens,
                batch_state_out_token,
                batch_action_out_token,
                action_embeddings_in_with_pos,
            ],
            dim=1,
        )

        y_new, kv_cache_state, *_ = self._transformer.forward(
            x,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
            should_grow_cache=should_grow_cache,
            use_decode_mask=True,
        )
        next_idx = input_pos[-1] + 1
        state_out_position = (
            self.image_tokenizer.get_n_img_tokens()
            + self.text_token_size
            + self.config.n_thinking_tokens
        )
        state_token_out = y_new[:, state_out_position : state_out_position + 1]
        future_vision_pred = self.future_vision_head(state_token_out)
        return sampled_action, next_idx, kv_cache_state, future_vision_pred
