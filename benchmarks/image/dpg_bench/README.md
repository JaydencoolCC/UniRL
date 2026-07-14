# image/dpg_bench

[DPG-Bench](https://github.com/TencentQQGYLab/ELLA) (ELLA, Apache-2.0): 1065 dense
paragraph-length prompts. The 7.4 MB prompts+questions CSV is fetched, not vendored:

```bash
bash benchmarks/image/dpg_bench/fetch.sh
python -m benchmarks.run -b image/dpg_bench --ckpt <base> [--lora <ckpt>]   # generate only
```

Scoring is external (official protocol): tile each prompt's 4 images into a 2×2 grid
named `<item_id>.png`, then run the ELLA repo's `dpg_bench/dist_eval.sh` (mPLUG-large
VQA judge; DPG score = mean accuracy × 100). Our prompt index `p` maps to `item_id` via
the CSV's unique-`text` order — the same first-seen order `run.py` generates in.
