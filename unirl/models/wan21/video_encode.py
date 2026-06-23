"""WAN21VideoLatentEncodeStage — Videos → normalized WAN VAE latents.

Used by trainside WAN V2V. This mirrors the latent-prep part of diffusers'
``WanVideoToVideoPipeline.prepare_latents``: encode the input video with the
WAN VAE using deterministic mode latents, then apply the same per-channel
normalization that decode later reverses.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from unirl.models.types.codec import EncodeStage
from unirl.types.primitives import Video, Videos

_SPATIAL_DOWNSAMPLE: int = 8
_TEMPORAL_DOWNSAMPLE: int = 4


@runtime_checkable
class _VAEBundle(Protocol):
    vae: Any
    device: torch.device
    dtype: torch.dtype


class WAN21VideoLatentEncodeStage(EncodeStage[Videos, torch.Tensor]):
    """Encode reference videos into normalized WAN latent space."""

    def __init__(
        self,
        bundle: _VAEBundle,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> None:
        self.bundle = bundle
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)

    def encode(self, p: Videos) -> torch.Tensor:
        if not isinstance(p, Videos):
            raise TypeError(f"WAN21VideoLatentEncodeStage.encode: expected Videos, got {type(p).__name__}")
        videos = p.to_list()
        if not videos:
            raise ValueError("WAN21VideoLatentEncodeStage.encode: empty Videos primitive")

        prepared = [self._prepare_one(video) for video in videos]
        stacked = torch.stack(prepared, dim=0)  # [B, T, C, H, W]
        video_condition = stacked.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        video_condition = video_condition.to(device=self.bundle.device, dtype=torch.float32)
        video_condition = video_condition * 2.0 - 1.0
        video_condition = video_condition.to(dtype=self.bundle.vae.dtype)

        with torch.no_grad():
            latent_condition = self.bundle.vae.encode(video_condition).latent_dist.mode()

        latent_condition = latent_condition.to(device=self.bundle.device, dtype=self.bundle.dtype)
        return self._normalize_latents(latent_condition)

    def _prepare_one(self, video: Video) -> torch.Tensor:
        frames = video.frames
        if frames is None or frames.ndim != 4 or int(frames.shape[1]) != 3:
            raise ValueError(
                f"WAN21VideoLatentEncodeStage.encode: expected frames [T, 3, H, W], "
                f"got shape {None if frames is None else tuple(frames.shape)}"
            )
        frames = frames.to(dtype=torch.float32).clamp_(0.0, 1.0)
        frames = self._sample_frames(frames, target_frames=self.num_frames)
        frames = F.interpolate(
            frames,
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return frames

    @staticmethod
    def _sample_frames(frames: torch.Tensor, *, target_frames: int) -> torch.Tensor:
        total = int(frames.shape[0])
        if total < 1:
            raise ValueError("WAN21VideoLatentEncodeStage.encode: condition video has no frames")
        if int(target_frames) < 1:
            raise ValueError(f"WAN21VideoLatentEncodeStage.encode: num_frames must be >=1, got {target_frames}")
        if total == int(target_frames):
            return frames
        indices = torch.linspace(0, total - 1, steps=int(target_frames), device=frames.device)
        indices = indices.round().to(dtype=torch.long).clamp_(0, total - 1)
        return frames.index_select(0, indices)

    def _normalize_latents(self, latent_condition: torch.Tensor) -> torch.Tensor:
        vae_config = self.bundle.vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            z_dim = int(getattr(vae_config, "z_dim", latent_condition.shape[1]))
            mean = torch.tensor(latents_mean, device=self.bundle.device, dtype=self.bundle.dtype).view(1, z_dim, 1, 1, 1)
            std = torch.tensor(latents_std, device=self.bundle.device, dtype=self.bundle.dtype).view(1, z_dim, 1, 1, 1)
            return (latent_condition - mean) / std

        scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
        return latent_condition * scaling_factor


def add_flowmatch_noise(init_latents: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """FlowMatch ``scale_noise`` equivalent: ``x_sigma = (1-sigma) * x0 + sigma * eps``."""
    sigma = sigma.to(device=init_latents.device, dtype=init_latents.dtype)
    while sigma.ndim < init_latents.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return (1.0 - sigma) * init_latents + sigma * noise.to(device=init_latents.device, dtype=init_latents.dtype)


__all__ = ["WAN21VideoLatentEncodeStage", "add_flowmatch_noise"]
