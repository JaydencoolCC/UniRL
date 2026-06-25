"""VideoCLIPDelta — an edit-delta CLIP reward for video-to-video.

Plain PickScore on a content-anchored V2V output starts near its ceiling
(~0.8): the edit caption mostly describes content the *source* video already
shows, so an un-edited output already scores high and there is little headroom
to train on. This scorer subtracts how much the edited frame still looks like
the SOURCE condition frame, so the reward measures the *edit*, not the content
that was free to begin with:

    reward = pickscore(edited_first_frame, target_caption)
             - lambda_source * cos(edited_first_frame, source_first_frame)

- The first term (identical scaling to ``PickScoreRewardScorer``) keeps the
  edit on-target for the caption.
- The second term (CLIP image-image cosine to the source) is high when the
  output barely changed, so an un-edited V2V output nets ~0 and the reward only
  climbs as the model actually applies the requested edit.

The source condition frame is available because ``RewardService`` copies the
rollout request's ``primitives`` (including the ``video`` condition) into the
reward request and ``repeat_interleave``s them to per-sample alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest
from unirl.utils.media import tensor_frame_to_pil

from .pickscore import PickScoreRewardScorer

if TYPE_CHECKING:
    from PIL import Image

    from unirl.types.primitives import Video


class VideoCLIPDeltaScorer(PickScoreRewardScorer):
    """PickScore-to-target minus CLIP-similarity-to-source, on the first frame.

    Inherits CLIP/PickScore model loading from ``PickScoreRewardScorer``; only
    the reward computation differs (it also reads the source condition video
    from ``request.primitives['video']``).
    """

    canonical_model_name = "videoclipdelta"
    input_kind = "video"

    def __init__(self, *, config: "VideoCLIPDeltaSpec", base_device: str) -> None:
        self.lambda_source = float(getattr(config, "lambda_source", 1.0))
        super().__init__(config=config, base_device=base_device)

    # ------------------------------------------------------------------
    # Frame + embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_frame_pil(video: "Video") -> "Image.Image":
        """First frame of a per-sample ``Video`` (frames ``[T, C, H, W]``)."""
        frames = video.frames
        if frames is None or frames.ndim != 4:
            raise ValueError(
                "VideoCLIPDeltaScorer: expected per-sample frames [T, C, H, W], got "
                f"{None if frames is None else tuple(frames.shape)}"
            )
        frame = frames[0].detach().cpu()
        if not frame.is_floating_point():
            frame = frame.float() / 255.0
        elif frame.numel() > 0 and frame.max() > 1.0:
            frame = (frame / 255.0).clamp(0.0, 1.0)
        else:
            frame = frame.clamp(0.0, 1.0)
        return tensor_frame_to_pil(frame)

    def _embed_images(self, pil_images: List["Image.Image"]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        edited = request.generated.get("video")
        if edited is None:
            raise ValueError(
                "VideoCLIPDeltaScorer: request.generated['video'] is missing; this scorer needs input_kind='video'."
            )
        source = request.primitives.get("video")
        if source is None:
            raise ValueError(
                "VideoCLIPDeltaScorer: request.primitives['video'] is missing — the V2V condition video must reach "
                "the reward (only V2V recipes provide it). Use a V2V recipe, or switch back to VideoPickScoreScorer."
            )

        prompts = request.prompts
        edited_videos = edited.to_list()
        source_videos = source.to_list()
        n = len(edited_videos)
        if len(source_videos) != n or len(prompts) != n:
            raise ValueError(
                f"VideoCLIPDeltaScorer: misaligned counts edited={n}, source={len(source_videos)}, "
                f"prompts={len(prompts)}."
            )

        edited_frames = [self._first_frame_pil(v) for v in edited_videos]
        source_frames = [self._first_frame_pil(v) for v in source_videos]

        rewards: List[float] = []
        with torch.no_grad():
            logit_scale = self.model.logit_scale.exp()
            for i in range(0, n, self.batch_size):
                e = edited_frames[i : i + self.batch_size]
                s = source_frames[i : i + self.batch_size]
                p = prompts[i : i + self.batch_size]

                edited_emb = self._embed_images(e)
                source_emb = self._embed_images(s)
                text_emb = self._embed_texts(p)

                # Same PickScore scaling for the on-target term; raw CLIP cosine
                # for the source-similarity penalty (the two land in a similar
                # ~0.8 range, so lambda_source=1.0 nets ~0 on an un-edited output).
                text_align = (logit_scale * (text_emb * edited_emb).sum(dim=-1)) / 26
                source_sim = (edited_emb * source_emb).sum(dim=-1)

                reward = text_align - self.lambda_source * source_sim
                rewards.extend(reward.float().cpu().tolist())
        return rewards


@dataclass
class VideoCLIPDeltaSpec(BaseRewardComponentSpec):
    """Typed config for the VideoCLIPDelta reward component.

    Mirrors ``VideoPickScoreSpec`` plus ``lambda_source`` — the weight on the
    "still looks like the source" penalty. Higher pushes the policy to change
    more from the condition video; lower keeps it closer to the source.
    """

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    lambda_source: float = 1.0
