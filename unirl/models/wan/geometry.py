"""Latent geometry shared by WAN 2.1 and WAN 2.2."""

from __future__ import annotations

from typing import Tuple


def wan_latent_shape(
    *,
    num_frames: int,
    height: int,
    width: int,
    latent_channels: int = 16,
    spatial_downsample: int = 8,
    temporal_downsample: int = 4,
) -> Tuple[int, int, int, int]:
    """Return the WAN latent shape ``(C, T, H, W)``.

    Both supported WAN generations use the same VAE geometry. Keeping the
    temporal invariant here prevents driver-side noise construction and the
    train-side diffusion stages from drifting apart.
    """
    num_frames = int(num_frames)
    temporal_downsample = int(temporal_downsample)
    if (num_frames - 1) % temporal_downsample != 0:
        valid_choices = ", ".join(str(1 + i * temporal_downsample) for i in range(6))
        raise ValueError(
            f"WAN VAE temporal_downsample={temporal_downsample} requires "
            f"(num_frames - 1) % {temporal_downsample} == 0, got num_frames={num_frames}; "
            f"valid choices: {valid_choices}, ..."
        )
    latent_t = (num_frames - 1) // temporal_downsample + 1
    return (
        int(latent_channels),
        latent_t,
        int(height) // int(spatial_downsample),
        int(width) // int(spatial_downsample),
    )


__all__ = ["wan_latent_shape"]
