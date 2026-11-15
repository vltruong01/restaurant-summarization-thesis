#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
.venv-stage-f-cloud/bin/python scripts/stage_f_cloud_d033.py smoke
echo 'Train-only100photo benchmark saved. Review benchmark_report.json before issuing the production command.'
