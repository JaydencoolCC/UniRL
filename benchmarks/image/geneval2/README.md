# image/geneval2

In-domain compositional T2I — the prompt set the GenEval2 GRPO/FlowDPPO recipes train
on. Prompts: the 800-line test split already in the repo (`datasets/geneval2/synthetic/test.jsonl`),
so this folder carries no data.

Protocol: 4 images/prompt; scored by the reward service `geneval2` scorer (VQAScore
soft-TIFA via Qwen3-VL-8B). The runner sends each prompt's `vqa_list` as request
metadata, so a stock scorer config works — no service-side `dataset_path` needed.

```bash
python -m benchmarks.run -b image/geneval2 --ckpt <base> [--lora <ckpt>] --reward-url http://<host>:8080
```
