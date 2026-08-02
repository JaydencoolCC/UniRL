#!/usr/bin/env bash
# PR #281 smoke — sd3_pickscore_refl, 10 rollouts, local assets, tf56 venv.
set -uo pipefail
cd /apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/UniRL-pr2
VENVPY=/tmp/tf56venv/bin/python3
export PRETRAINED_MODEL=/apdcephfs_zwfy8/share_305110755/hunyuan/jinlw_sync/models/stable-diffusion-3.5-medium
export OUTPUT_DIR=/apdcephfs_zwfy8/share_305110755/hunyuan/haonan/mmgrpo/UniRL-pr2/outputs_sd3_smoke
export REPORT_TO_WANDB=false
$VENVPY -c "import sys; from ray.scripts.scripts import main; sys.argv=['ray','stop','--force']; main()" >/dev/null 2>&1 || true
$VENVPY -c "import sys; from ray.scripts.scripts import main; sys.argv=['ray','start','--head','--disable-usage-stats']; sys.exit(main())"
RAY_ADDRESS=auto $VENVPY -m experimental.refl.run --config-name=sd3_pickscore_refl \
  num_devices=8 num_rollouts=10 \
  reward.backend.config.model_id=/apdcephfs_zwfy8/share_305110755/hunyuan/jinlw_sync/models/PickScore_v1 \
  reward.backend.config.processor_id=/apdcephfs_zwfy8/share_305110755/hunyuan/jinlw_sync/models/CLIP-ViT-H-14
echo "SD3_SMOKE_EXIT=$?"
