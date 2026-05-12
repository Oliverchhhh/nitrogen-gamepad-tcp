"""
Lightweight StaMo encoder wrapper.

Loads only the encoder components (VisionBackbone + Projector + DiTConditionHead)
from a StaMo checkpoint directory, without loading the large SD3 transformer or VAE.
Used to produce supervision targets for StaMo co-training.

Output shape: image_embeds [B, 2, 4096] (2 StaMo tokens × 4096-dim).
"""

import logging
import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_STAMO_REPO = "/home/ch/StaMo"

_STAMO_ARCH = SimpleNamespace(
    # VisionBackbone: DINOv2 ViT-Base with register tokens, 336×336 input
    vision_backbone=SimpleNamespace(
        model_name="vit_base_patch14_reg4_dinov2.lvd142m",
        img_size=336,
    ),
    # Projector compresses 576 patches to 2 tokens
    projector=SimpleNamespace(
        hidden_dim=1024,
        cross_attention_dim=512,
        output_align_dim=4096,
        num_token=2,
        num_attn_layers=6,
        num_attn_compress_layers=6,
    ),
)


class StaMoEncoder(nn.Module):
    """
    Frozen StaMo encoder: VisionBackbone + Projector + DiTConditionHead.

    Encodes 256×256 (or any size) images to [B, 2, 4096] StaMo representations
    by first resizing to 336×336 (the ViT-Base expected input).

    All parameters are frozen. Use encode_batch() for [B, T, 3, H, W] inputs.
    """

    IMG_SIZE = 336
    N_OUTPUT_TOKENS = 2
    TOKEN_DIM = 4096
    FLAT_DIM = N_OUTPUT_TOKENS * TOKEN_DIM  # 8192

    def __init__(self, checkpoint_dir: Optional[str] = None):
        super().__init__()

        if _STAMO_REPO not in sys.path:
            sys.path.insert(0, _STAMO_REPO)

        from stamo.renderer.model.backbone import VisionBackbone, DiTConditionHead
        from stamo.renderer.model.projector import Projector

        arch = _STAMO_ARCH
        self.vision_backbone = VisionBackbone(
            img_size=arch.vision_backbone.img_size,
            model_name=arch.vision_backbone.model_name,
            pretrained=(checkpoint_dir is None),
        )

        self.projector = Projector(
            arch,
            patches=self.vision_backbone.patches,
            channels=self.vision_backbone.channels,
        )
        self.dit_condition_head = DiTConditionHead(pooled_dim=2048)

        if checkpoint_dir is not None:
            self._load_from_checkpoint(checkpoint_dir)
        else:
            logging.warning(
                "StaMoEncoder: no checkpoint_dir provided, using random/pretrained weights. "
                "Only the ViT backbone has pretrained weights; projector is random."
            )

        # Freeze all parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _load_from_checkpoint(self, checkpoint_dir: str) -> None:
        rendernet_path = os.path.join(checkpoint_dir, "RenderNet.pth")
        projector_path = os.path.join(checkpoint_dir, "Projector.pth")

        if not os.path.exists(rendernet_path):
            raise FileNotFoundError(
                f"StaMo RenderNet checkpoint not found at {rendernet_path}. "
                "Set stamo_checkpoint_dir to the directory containing RenderNet.pth and Projector.pth."
            )

        rendernet_ckpt = torch.load(rendernet_path, map_location="cpu")
        model_state = rendernet_ckpt.get("model", rendernet_ckpt)

        def _extract_and_load(module: nn.Module, prefix: str) -> None:
            sub = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
            if sub:
                missing, unexpected = module.load_state_dict(sub, strict=False)
                if missing:
                    logging.warning("StaMoEncoder %s missing keys: %s", prefix, missing[:5])
                if unexpected:
                    logging.warning("StaMoEncoder %s unexpected keys: %s", prefix, unexpected[:5])
            else:
                logging.warning("StaMoEncoder: no keys with prefix '%s' in checkpoint", prefix)

        _extract_and_load(self.vision_backbone, "vision_backbone.")
        _extract_and_load(self.dit_condition_head, "dit_condition_head.")

        if os.path.exists(projector_path):
            proj_ckpt = torch.load(projector_path, map_location="cpu")
            missing, unexpected = self.projector.load_state_dict(proj_ckpt, strict=False)
            if missing:
                logging.warning("StaMoEncoder projector missing keys: %s", missing[:5])
            if unexpected:
                logging.warning("StaMoEncoder projector unexpected keys: %s", unexpected[:5])
        else:
            _extract_and_load(self.projector, "projector.")

        logging.info("StaMoEncoder loaded from %s", checkpoint_dir)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to StaMo representations.

        Args:
            images: [B, 3, H, W], any spatial resolution.

        Returns:
            image_embeds: [B, 2, 4096]
        """
        if images.shape[-1] != self.IMG_SIZE or images.shape[-2] != self.IMG_SIZE:
            images = F.interpolate(
                images.float(),
                size=(self.IMG_SIZE, self.IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            ).to(images.dtype)

        images = self.vision_backbone.transforms(images)
        image_embeds = self.vision_backbone(images)   # [B, 576, 768]
        image_embeds = self.projector(image_embeds)   # [B, 2, 4096]
        return image_embeds

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a video clip to StaMo representations.

        Args:
            images: [B, T, 3, H, W]

        Returns:
            flat_embeds: [B, T, FLAT_DIM] where FLAT_DIM = 8192 (= 2 × 4096)
        """
        B, T, C, H, W = images.shape
        flat = images.reshape(B * T, C, H, W)
        embeds = self.encode(flat)           # [B*T, 2, 4096]
        flat_embeds = embeds.reshape(B * T, self.FLAT_DIM)
        return flat_embeds.reshape(B, T, self.FLAT_DIM)
