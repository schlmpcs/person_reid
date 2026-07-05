#!/usr/bin/env bash
# Train FastReID SBS ResNet-50-IBN on MSMT17 (RTX 5060 8GB, AMP).
# Usage: bash train_sbs.sh   (logs to logs/msmt17/sbs_R50-ibn/log.txt via FastReID)
set -e
cd /home/amirizimov/MSMT17/fast-reid
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=.
export FASTREID_DATASETS=/home/amirizimov/MSMT17

../.venv/bin/python tools/train_net.py \
  --config-file configs/MSMT17/sbs_R50-ibn.yml \
  SOLVER.AMP.ENABLED True \
  SOLVER.IMS_PER_BATCH 64 \
  DATALOADER.NUM_INSTANCE 16 \
  DATALOADER.NUM_WORKERS 6 \
  TEST.IMS_PER_BATCH 128 \
  OUTPUT_DIR logs/msmt17/sbs_R50-ibn
