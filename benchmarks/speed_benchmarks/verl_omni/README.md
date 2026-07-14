# vs verl-omni

[verl-omni](https://github.com/verl-project/verl-omni) is the closest peer framework
(diffusion / unified / omni RL on vLLM-Omni rollout + FSDP2/VeOmni training). Its
published reference number (~25% end-to-end over diffusers-based flow_grpo on
Qwen-Image FlowGRPO) comes without a public setup spec, so any comparison we publish
must pin and disclose both sides per the protocol in [`../README.md`](../README.md).

Overlapping (model, algorithm) pairs, both natively supported:

| pair | UniRL side | verl-omni side |
|---|---|---|
| Qwen-Image + FlowGRPO | `examples/diffusion/qwen_image/qwen_image_grpo_vllmomni.yaml` | their reference recipe |
| Wan2.2-T2V-14B + DanceGRPO | `examples/diffusion/wan22/wan22_t2v_14b_dancegrpo.yaml` | their Wan2.2 recipe |
| SD3.5 + FlowGRPO | `examples/diffusion/sd3/sd3_vllmomni.yaml` | their SD3.5 recipe |

Suggested alignment for the aligned-backend rows: vLLM-Omni rollout on both sides, same
HPSv3 HTTP reward service, 8×(H20|H800), identical prompts/step × group size,
resolution and denoise steps taken from the verl-omni recipe. For the best-config rows
UniRL may additionally use its `sglang`/trainside rollout variants of the same recipes.

Measure both sides steady-state (UniRL: `../parse_perf.py`; verl-omni: its logged
step timing over the same step window) and record: median s/step, samples/GPU-hour,
peak memory, plus both full configs. Results tables land here as they are produced.
