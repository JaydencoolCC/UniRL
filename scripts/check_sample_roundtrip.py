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
from unirl.types.sample import Part, Sample, _part_with_field
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


def check_root_group_ids() -> None:
    """``Sample.root_group_ids(i)`` climbs each sample of ``parts[i]`` to its root
    prompt — coarser than ``Part.group_ids`` (immediate parent). The PE/unified
    ``compute_track_advantages(group_key="root")`` replacement; the labels stay
    group-by-parent contiguous for ``Part.compute_advantages(group_ids=...)``."""
    # P=2 prompts, N=2 AR children each, M=1 image each → [input, ar(4), image(4)].
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a cat", "a dog"]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img])

    _check(list(img.sample_ids) == ["p0/0/0", "p0/1/0", "p1/0/0", "p1/1/0"], "image fan-out ids")
    _check(sample.root_group_ids(2) == ["p0", "p0", "p1", "p1"], "image part grouped by ROOT prompt")
    _check(img.group_ids == ["p0/0", "p0/1", "p1/0", "p1/1"], "Part.group_ids stays immediate-parent (4 groups)")
    _check(sample.root_group_ids(0) == ["p0", "p1"], "root part: each prompt is its own group")
    _check(sample.root_group_ids(1) == ["p0", "p0", "p1", "p1"], "ar part grouped by root prompt")


def check_gen_part_accessors() -> None:
    """``Sample.gen_parts``/``gen_part``/``gen_part_index`` locate stages by
    ``sampling_params`` TYPE (the migration's part-location convention; mirrors
    the engine-side ``ar_gen_part``/``diffusion_gen_part``). ``with_parts`` swaps
    parts while preserving ``reward_compute_s``."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a cat", "a dog"]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img], reward_compute_s=1.5)

    gps = sample.gen_parts()
    _check(len(gps) == 2 and gps[0] is ar and gps[1] is img, "gen_parts skips the input Part")
    _check(sample.gen_part(ARSamplingParams) is ar, "gen_part locates the AR stage by type")
    _check(sample.gen_part(DiffusionSamplingParams) is img, "gen_part locates the diffusion stage by type")
    _check(sample.gen_part_index(ARSamplingParams) == 1, "gen_part_index: AR at position 1")
    _check(sample.gen_part_index(DiffusionSamplingParams) == 2, "gen_part_index: diffusion at position 2")

    # A request with no diffusion stage → clear ValueError, not a bare StopIteration.
    try:
        Sample(parts=[inp, ar]).gen_part(DiffusionSamplingParams)
        raised = False
    except ValueError:
        raised = True
    _check(raised, "gen_part raises ValueError when the stage type is absent")

    # with_parts swaps the parts list but carries reward_compute_s forward.
    swapped = sample.with_parts([inp, ar])
    _check(len(swapped.parts) == 2 and swapped.parts[1] is ar, "with_parts replaces the parts list")
    _check(swapped.reward_compute_s == 1.5, "with_parts preserves reward_compute_s")


def check_sample_dp_chunk() -> None:
    """``Sample.chunk``/``slice``/``select`` shard by whole prompt-TREE (the dp>1
    rollout-request fix), not by the ``parts`` list — the inherited ``Batch.slice``
    would slice the length-3 parts list and hand every DP rank the full Sample.
    Reuses the tree-correct ``split``/``concat``."""
    inp = Part.input([f"p{i}" for i in range(4)], primitive=Texts(texts=[f"prompt {i}" for i in range(4)]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img], reward_compute_s=2.0)

    shards = sample.chunk(2)
    _check(len(shards) == 2, "chunk(2) yields 2 shards")
    _check([s.parts[0].batch_size for s in shards] == [2, 2], "each shard holds 2 prompt-trees (not the whole batch)")
    _check([s.parts[1].batch_size for s in shards] == [4, 4], "each shard ar part = 2*N")
    _check([s.parts[2].batch_size for s in shards] == [4, 4], "each shard image part = 2*N*M")
    _check(list(shards[0].parts[0].sample_ids) == ["p0", "p1"], "shard 0 = prompts 0,1")
    _check(
        list(shards[1].parts[2].sample_ids) == ["p2/0/0", "p2/1/0", "p3/0/0", "p3/1/0"],
        "image part is tree-sharded (not list-sliced to the whole part)",
    )
    rt = Sample.concat(shards)
    _check(list(rt.parts[2].sample_ids) == list(sample.parts[2].sample_ids), "chunk -> concat round-trips ids")
    _check(rt.reward_compute_s == 2.0, "chunk -> concat preserves reward_compute_s")
    _check(list(sample.slice(0, 2).parts[0].sample_ids) == ["p0", "p1"], "slice(0,2) = first 2 trees")
    _check(
        list(sample.select(torch.tensor([0, 2])).parts[0].sample_ids) == ["p0", "p2"],
        "select gathers whole trees by index",
    )


def check_unified_dit_noise_ids_unique() -> None:
    """The unified DiT sub-request re-roots from the globally-unique image-shell
    lineage (flatten ``/``), so the engine-derived x_T key stays unique across dp>1
    replicas; the replica-local ``d{k}`` scheme collides (each shard restarts at 0)."""
    inp = Part.input([f"r0:p{i}" for i in range(4)], primitive=Texts(texts=[f"prompt {i}" for i in range(4)]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    shards = Sample(parts=[inp, ar, img]).chunk(2)  # dp=2 — what run_rollout does

    fixed = [sid.replace("/", "_") for s in shards for sid in s.gen_part(DiffusionSamplingParams).sample_ids]
    _check(len(fixed) == len(set(fixed)), "lineage-derived DiT ids are globally unique across dp shards")
    old = [f"r0:d{k}" for s in shards for k in range(s.gen_part(DiffusionSamplingParams).batch_size)]
    _check(len(old) != len(set(old)), "sanity: the old replica-local d{k} scheme DOES collide")


def check_advantage_group_ids_recorded() -> None:
    """``Part.compute_advantages`` records the grouping it used (``advantage_group_ids``)
    so zero-std metrics bucket by the actual GRPO baseline — including a coarser
    root-prompt override (PE ``diffusion_group_scope='prompt'``)."""
    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(2, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img])
    image = _part_with_field(sample.parts[2], "rewards", torch.arange(8, dtype=torch.float32))  # 8 = P*N*M

    a_def = image.compute_advantages(normalize=True)
    _check(a_def.advantage_group_ids == list(image.group_ids), "records the immediate-parent grouping by default")
    root = sample.root_group_ids(2)
    a_root = image.compute_advantages(normalize=True, group_ids=root)
    _check(a_root.advantage_group_ids == root, "records the root-scope override grouping")
    _check(a_root.advantage_group_ids != list(image.group_ids), "root grouping is coarser than immediate group_ids")


def check_rollout_metric_naming() -> None:
    """Multi-stage rollout metrics name the diffusion stage ``diffusion`` (matching
    the train-side track key), not ``image`` — so PE's ``rollout/*`` and ``train/*``
    panels correlate."""
    from unirl.utils.wandb_metrics import compute_rollout_sample_metrics

    inp = Part.input(["p0", "p1"], primitive=Texts(texts=["a", "b"]))
    ar = _part_with_field(inp.fork(2, sampling_params=ARSamplingParams()), "rewards", torch.arange(4, dtype=torch.float32))
    img = _part_with_field(
        ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)),
        "rewards",
        torch.arange(4, dtype=torch.float32),
    )
    m = compute_rollout_sample_metrics(sample=Sample(parts=[inp, ar, img]))
    _check(any(k.startswith("diffusion_") for k in m), "diffusion stage logs under the 'diffusion_' prefix")
    _check(not any(k.startswith("image_") for k in m), "no stale 'image_' prefix")


def check_disable_driver_xt_flag() -> None:
    """``DiffusionSamplingParams`` carries the ``disable_driver_xt`` opt-out (default
    off), settable per-recipe and via ``dataclasses.replace`` (how the unified trainer
    stamps the env onto the params)."""
    import dataclasses

    d = DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    _check(d.disable_driver_xt is False, "disable_driver_xt defaults to False")
    _check(dataclasses.replace(d, disable_driver_xt=True).disable_driver_xt is True, "replace sets the flag")
    _check(
        DiffusionSamplingParams(
            num_inference_steps=4, height=256, width=256, disable_driver_xt=True
        ).disable_driver_xt
        is True,
        "recipe can construct with the flag set",
    )


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_part_extraction,
    check_assemble_sample,
    check_sigma_roundtrip,
    check_noise_key_lineage,
    check_split_concat_roundtrip,
    check_multi_input_image_chain,
    check_cot_text_chain,
    check_root_group_ids,
    check_gen_part_accessors,
    check_sample_dp_chunk,
    check_unified_dit_noise_ids_unique,
    check_advantage_group_ids_recorded,
    check_rollout_metric_naming,
    check_disable_driver_xt_flag,
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
