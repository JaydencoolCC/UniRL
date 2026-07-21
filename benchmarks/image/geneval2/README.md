# image/geneval2

GenEval2 — a compositional text-to-image benchmark. Each of the 800 prompts
(`datasets/geneval2/synthetic/test.jsonl`) ships a `vqa_list` of atomic
`(question, expected_answer)` checks (object presence, count, color/attribute, position, verb).
Quality = **Soft-TIFA VQAScore**: a VLM (Qwen3-VL) answers each atom, and the per-atom answer
probabilities are aggregated (geometric mean) into a per-image score in `[0, 1]` (report ×100).

## Run

```bash
python -m benchmarks.run -b image/geneval2 --ckpt <base> [--lora <adapter>] --reward-url http://<host>:8080
```

The runner generates images, then scores them through the reward service's `geneval2` scorer,
sending each prompt's `vqa_list` as request metadata (a canary request guards against a
non-Soft-TIFA service). Use geometric-mean aggregation for the headline number.

## Scoring backends

- **Reward service (vLLM)** — default; fast, for general benchmarking/monitoring. Reads top-k
  logprobs, so it is an approximation of the full-vocab softmax.
- **transformers Qwen3-VL-8B** — full-vocab softmax, geometric mean. **Required to reproduce the
  DPPO paper numbers** (the vLLM top-k service gives close but non-identical scores). Use the
  local scorer `unirl/reward/local/geneval2.py` (Qwen/Qwen3-VL-8B-Instruct).

## Note: DPPO GenEval2 reproduction

The `image/geneval2` spec pins the DPPO eval regime so `benchmarks.run` matches the recipes:
512×512, 40 steps, cfg 1.0, `max_sequence_length=256`, `linspace(1, 1/steps, steps)` flow-match
sigma grid, per-prompt-content seed, 1 sample/prompt; score with transformers Qwen3-VL-8B (GM).
(`max_sequence_length=256` and the linspace grid are load-bearing — the diffusers pipeline
defaults roughly halve the score.)

Two eval configs:
- **exact 800** (default): mean over the 800 unique prompts.
- **simulated 32×32** (`--sim-even-batches 32x32`): the original distributed eval repeated the last
  partial wave of a 32-GPU × batch-32 loader, double-counting a fixed prefix of prompts; this flag
  additionally reports that `*_sim32x32` metric.
