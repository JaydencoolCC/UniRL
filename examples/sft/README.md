# SFT / Behavior-Cloning Recipes (Cosmos3)

Supervised flow-matching finetuning for **NVIDIA Cosmos3-Nano** (16B omnimodal
MoT world model), including robot **action-trajectory behavior cloning** in the
style of Cosmos3-Nano-Policy-DROID.

## Module split

| Layer | Code | Role |
|---|---|---|
| Generic training framework | `unirl/trainer/sft.py`, `unirl/train_sft.py`, `unirl/train/sft/{policy,data}.py` | Model-agnostic loop: manifest records → task loss → `FSDPBackend` optimizer step → checkpoint → periodic samples. Mirrors the ReFL domain (`unirl/trainer/refl.py`). |
| Cosmos3 wrapper | `unirl/models/cosmos3/{config,bundle,packing}.py` | Weights + freeze policy (und/AR tower frozen, gen tower trainable), joint-sequence packing that reuses `diffusers.Cosmos3OmniPipeline`'s own helpers, flow-matching sigma/noise/velocity math. |
| Task adapters | `unirl/models/cosmos3/sft_task.py` | `Cosmos3VideoSFTTask` (t2i / t2v / video prediction) and `Cosmos3ActionBCTask` (policy-mode BC: obs + instruction → action chunk + co-denoised future video). |
| Data prep | `unirl/utils/prepare_droid100.py` | LeRobot-v3 → self-contained debug samples (uint8 clips + z-normalized action chunks + JSONL manifests). |

## Requirements

- `diffusers>=0.39` (first release with `Cosmos3OmniTransformer` / `Cosmos3OmniPipeline`).
- The trainer venv only — no inference engine (sglang / vllm-omni) is involved.
- Checkpoint: `nvidia/Cosmos3-Nano` (diffusers layout; ~33 GB for
  `transformer/ vae/ text_tokenizer/ scheduler/`).

## Quickstart

```bash
# 1. Data: droid_100 (0.93 GB, 100 episodes, LeRobot v3) -> debug windows
python -m unirl.utils.prepare_droid100 --root datasets/droid100_debug

# 2. First milestone — video prediction (obs frame + instruction -> 16 frames)
export PRETRAINED_MODEL=/path/to/Cosmos3-Nano
ray start --head
RAY_ADDRESS=auto python -m unirl.train_sft --config-name sft/cosmos3_droid100_videopred

# 3. Action BC (policy mode, the Cosmos3-Nano-Policy-DROID objective)
RAY_ADDRESS=auto python -m unirl.train_sft --config-name sft/cosmos3_droid100_action_bc
```

Each step covers the full chain: dataset → collate (worker-side) → packed MoT
forward → masked velocity MSE → backward → clip + AdamW step → periodic
checkpoint (`save_interval`) → periodic samples (`eval_interval`, written to
`<save_dir>/samples/step_N.pt` as `{"video": [T,C,H,W], "action": [T,D]}`).

## How training matches inference

- Packing calls the *pipeline's own* `tokenize_prompt` / `_prepare_*_segment`
  helpers, so prompts (chat template + metadata sentences + `<|vision_start|>`),
  mRoPE ids, and sequence layout are bit-identical to `Cosmos3OmniPipeline.__call__`.
- VAE encoding uses `_encode_video` (argmax mode + per-channel mean/std, amp off).
- Velocity target is `v = eps - x0` (the UniPC `flow_prediction` convention:
  `x0_pred = sample - sigma * v`); sigma is logitnormal, warped by the same
  `flow_shift` map `set_timesteps` uses.
- Policy-mode BC: latent frame 0 clean (no timestep embedding, excluded from
  loss), future frames + the whole action chunk noised with one shared sigma —
  the joint objective Cosmos3-Nano-Policy-DROID was post-trained with
  (`action_loss_weight=10`, `flow_shift=5.0` per the upstream action recipes).

## Known debug-scale simplifications

- **droid_100 actions are LeRobot's collapsed 7-D `action` field** (≈ 6-D
  cartesian velocity + gripper), not the 10-D EE-pose layout the
  `droid_lerobot` domain head (id 8) was pretrained on, and not the 8-D
  absolute joint positions Policy-DROID uses. Fine for a debug BC run — the
  head is being finetuned — but for faithful Policy-DROID reproduction, prep
  `nvidia/Cosmos3-DROID` (success split) with `action.joint_position` +
  `gripper_position` instead.
- No proprioceptive-state stream (the released inference stacks don't consume
  one either); NVIDIA's internal recipe additionally conditions on state.
- Full finetune of the gen stream only; `lora_cfg` is wired through
  `FSDPBackend` if adapter training is preferred.
