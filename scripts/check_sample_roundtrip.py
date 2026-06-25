#!/usr/bin/env python3
"""Runtime oracle for the Sample → Sample rollout-engine refactor (LIN-454).

Exercises the PURE Sample-assembly logic the converted engines rely on — the
path-id lineage, the part-extraction routing, ``assemble_sample`` fill, the σ
round-trip verifier, and the OD-2 noise-key derivation — with fabricated
Samples + ``SimpleNamespace`` fakes (no GPU, no backend, no vllm/sglang). It
does NOT boot an engine; it guards the conversion's structural contracts so a
broken fork / fill / route surfaces here instead of mid-rollout.

Only needs the unirl runtime types (torch) — the imported ``vllm_omni.utils``
helpers pull in no engine runtime. Run in the engine venv:

    python scripts/check_sample_roundtrip.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Callable, Tuple

import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.utils import (
    ar_gen_part,
    assemble_sample,
    cot_text_from_sample,
    diffusion_gen_part,
    image_input_part,
    texts_from_sample,
)
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_raises(fn: Callable[[], object], exc: type, msg: str) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{msg}: expected {exc.__name__} but none was raised")


def _two_stage_sample() -> Tuple[Sample, Part, Part, Part]:
    """``[input(P=2), ar(P*N, N=1), image(P*N*M, M=1)]`` — the HI3 t2i shape."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a cat", "a dog"]))
    ar_shell = inp.fork(1, sampling_params=ARSamplingParams())
    img_shell = ar_shell.fork(
        1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    )
    return Sample(parts=[inp, ar_shell, img_shell]), inp, ar_shell, img_shell


def check_part_extraction() -> None:
    """The part-extraction helpers route by ``sampling_params`` type / primitive."""
    sample, _inp, ar_shell, img_shell = _two_stage_sample()
    _check(ar_gen_part(sample) is ar_shell, "ar_gen_part must return the AR shell")
    _check(diffusion_gen_part(sample) is img_shell, "diffusion_gen_part must return the image shell")
    _check(image_input_part(sample) is None, "image_input_part must be None (no Images input Part present)")
    _check(list(texts_from_sample(sample).texts) == ["a cat", "a dog"], "texts_from_sample returns input prompts")

    # A prompt/gen count mismatch must fail fast (mis-forked Sample).
    mismatched = Sample(
        parts=[
            Part.input(["p0"], primitive=Texts(texts=["x", "y"])),
            Part.input(["p0"], primitive=Texts(texts=["x", "y"])).fork(1, sampling_params=ARSamplingParams()),
        ]
    )
    _expect_raises(lambda: texts_from_sample(mismatched), ValueError, "prompt/gen count mismatch must raise")


def check_assemble_sample() -> None:
    """``assemble_sample`` fills the right gen Parts (by params type), preserves ids."""
    sample, inp, _ar_shell, _img_shell = _two_stage_sample()
    ar_seg, img_seg = SimpleNamespace(tag="ar_seg"), SimpleNamespace(tag="img_seg")
    ar_dec, img_dec = Texts(texts=["pe0", "pe1"]), SimpleNamespace(tag="img_dec")
    cond = {"text": SimpleNamespace(tag="cond")}

    out = assemble_sample(
        sample,
        segments_for_track={"ar": ar_seg, "image": img_seg},
        decoded_for_track={"ar": ar_dec, "image": img_dec},
        conditions=cond,
    )
    _check(len(out.parts) == 3, "assemble_sample must preserve the part count")
    out_inp, out_ar, out_img = out.parts
    _check(out_inp is inp, "the input Part passes through unchanged")
    _check(out_ar.segment is ar_seg and out_ar.primitive is ar_dec, "ar Part filled by the 'ar' track")
    _check(out_img.segment is img_seg and out_img.primitive is img_dec, "image Part filled by the 'image' track")
    _check(out_ar.conditions == cond and out_img.conditions == cond, "conditions replicate onto every filled Part")
    _check(list(out_ar.sample_ids) == ["p0/0", "p1/0"], "ar path ids preserved")
    _check(list(out_img.sample_ids) == ["p0/0/0", "p1/0/0"], "image path ids preserved")
    _check(isinstance(out_ar.sampling_params, ARSamplingParams), "ar sampling_params preserved through fill")
    _check(isinstance(out_img.sampling_params, DiffusionSamplingParams), "image sampling_params preserved through fill")

    # Single-DiT shape ``[input, image]``: only the diffusion Part is targeted.
    inp_s = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    img_s = inp_s.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    single = Sample(parts=[inp_s, img_s])
    out2 = assemble_sample(
        single, segments_for_track={"image": img_seg}, decoded_for_track={"image": img_dec}, conditions={}
    )
    _check(out2.parts[-1].segment is img_seg, "single-DiT: image Part filled")


def check_sigma_roundtrip() -> None:
    """``verify_engine_used_sigmas`` passes on echo == sent, raises on drift / missing."""
    sig = torch.linspace(1.0, 0.0, steps=5)
    verify_engine_used_sigmas(sig.clone(), expected=sig, engine_name="oracle")  # match → no raise

    perturbed = sig.clone()
    perturbed[1] += 0.1
    _expect_raises(
        lambda: verify_engine_used_sigmas(perturbed, expected=sig, engine_name="oracle"),
        RuntimeError,
        "perturbed σ echo must raise",
    )
    _expect_raises(
        lambda: verify_engine_used_sigmas(None, expected=sig, engine_name="oracle"),
        RuntimeError,
        "missing σ echo must raise",
    )
    verify_engine_used_sigmas(None, expected=None, engine_name="oracle")  # expected None → skip, no raise


def check_noise_key_lineage() -> None:
    """OD-2: the x_T noise key is the gen Part's lineage — group id (shared) vs
    sample id (per-sample), both derived from the path-id fork."""
    inp = Part.input(["p0"], primitive=Texts(texts=["x"]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(2, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    _check(
        list(img.sample_ids) == ["p0/0/0", "p0/0/1", "p0/1/0", "p0/1/1"],
        "fork builds group-by-parent contiguous path ids",
    )
    _check(len(set(img.sample_ids)) == 4, "per-sample noise key (sample_ids) is unique → distinct x_T")
    _check(
        img.group_ids == ["p0/0", "p0/0", "p0/1", "p0/1"],
        "group noise key (group_ids) = parent id → siblings share x_T",
    )


def check_split_concat_roundtrip() -> None:
    """``Sample.split`` → ``Sample.concat`` round-trips ids (chunked_engine_generate core)."""
    sample, *_ = _two_stage_sample()
    groups = sample.split()
    _check(len(groups) == 2, "split yields one Sample per root group")
    merged = Sample.concat(groups)
    _check(
        [list(p.sample_ids) for p in merged.parts] == [list(p.sample_ids) for p in sample.parts],
        "split→concat round-trips the per-part sample ids in order",
    )


def _images(n: int) -> Images:
    return Images.from_list([Image(pixels=torch.zeros(3, 8, 8)) for _ in range(n)])


def check_multi_input_image_chain() -> None:
    """IT2I-shaped multi-input ``[text, image_input, ar_gen, image_gen]`` via Part.input_child:
    chains a valid Sample, the helpers route past the image input, conditioning() surfaces both
    primitives, and assemble_sample fills the right gen Parts."""
    text = Part.input(["p0", "p1"], primitive=Texts(texts=["edit the cat", "edit the dog"]))
    image_in = text.input_child(_images(2))
    sample = (
        Sample.request(text, image_in)
        .fork(1, sampling_params=ARSamplingParams())
        .fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    )
    _check(len(sample.parts) == 4, "multi-input chain has 4 parts [text, image, ar, image]")
    _check(image_input_part(sample) is sample.parts[1], "image_input_part finds the chained image input")
    _check(ar_gen_part(sample) is sample.parts[2], "ar_gen_part routes past the image input Part")
    _check(diffusion_gen_part(sample) is sample.parts[3], "diffusion_gen_part routes past the image input Part")
    _check(
        list(texts_from_sample(sample).texts) == ["edit the cat", "edit the dog"],
        "texts_from_sample reads the prompt head",
    )
    cond = sample.conditioning()
    _check(
        len(cond) == 2 and isinstance(cond[0], Texts) and isinstance(cond[1], Images),
        "conditioning() returns [Texts, Images] in turn order (the empty gen shells are skipped)",
    )
    out = assemble_sample(
        sample,
        segments_for_track={"ar": SimpleNamespace(tag="ar"), "image": SimpleNamespace(tag="img")},
        decoded_for_track={"ar": Texts(texts=["r0", "r1"]), "image": SimpleNamespace(tag="dec")},
        conditions={"fused": SimpleNamespace(tag="cond")},
    )
    _check(out.parts[0] is sample.parts[0] and out.parts[1] is sample.parts[1], "input Parts pass through unchanged")
    _check(getattr(out.parts[2].segment, "tag", None) == "ar", "ar gen Part filled by the 'ar' track")
    _check(getattr(out.parts[3].segment, "tag", None) == "img", "image gen Part filled by the 'image' track")


def check_cot_text_chain() -> None:
    """dit_recaption-shaped multi-input ``[text, cot_text_input, image_gen]``: cot_text_from_sample
    finds the chained recaption Texts and the count guard fires on mismatch."""
    text = Part.input(["p0", "p1"], primitive=Texts(texts=["a cat", "a dog"]))
    cot_in = text.input_child(Texts(texts=["a fluffy ginger cat", "a happy golden dog"]))
    sample = Sample.request(text, cot_in).fork(
        1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    )
    _check(
        list(cot_text_from_sample(sample).texts) == ["a fluffy ginger cat", "a happy golden dog"],
        "cot_text_from_sample returns the chained recaption Texts",
    )
    # cot count != prompt count must raise (prompts=1, cot=2; gen count matches prompts so the
    # texts_from_sample guard passes and the cot guard is what fires)
    bad_text = Part.input(["p0"], primitive=Texts(texts=["a cat"]))
    bad = Sample.request(bad_text, bad_text.input_child(Texts(texts=["x", "y"]))).fork(
        1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    )
    _expect_raises(lambda: cot_text_from_sample(bad), ValueError, "cot/prompt count mismatch must raise")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_part_extraction,
    check_assemble_sample,
    check_sigma_roundtrip,
    check_noise_key_lineage,
    check_split_concat_roundtrip,
    check_multi_input_image_chain,
    check_cot_text_chain,
)


def main() -> int:
    failures = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("check-sample-roundtrip: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-sample-roundtrip: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
