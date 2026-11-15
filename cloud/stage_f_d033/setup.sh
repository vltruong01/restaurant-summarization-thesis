#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c 'import sys; assert sys.version_info[:2]==(3,11), "Use a Python3.11 Linux template"'
python3 scripts/verify_stage_f_cloud_package_d033.py
python3 -m venv .venv-stage-f-cloud
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_TRUSTED_HOST PIP_FIND_LINKS
.venv-stage-f-cloud/bin/python -m pip install --index-url https://pypi.org/simple numpy==1.26.4
.venv-stage-f-cloud/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0+cu118 torchvision==0.16.0+cu118
.venv-stage-f-cloud/bin/python -m pip install --index-url https://pypi.org/simple -r requirements/stage_f_cloud_d033.in
.venv-stage-f-cloud/bin/python -m pip check
.venv-stage-f-cloud/bin/python -c 'import torch,transformers; assert torch.cuda.is_available(); assert transformers.__version__=="4.51.3"; print(torch.__version__,torch.cuda.get_device_name(0))'
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=4
.venv-stage-f-cloud/bin/python scripts/stage_f_cloud_d033.py download
.venv-stage-f-cloud/bin/python scripts/stage_f_cloud_d033.py preflight
echo 'Setup/weight checks complete. Next: bash cloud/stage_f_d033/run_smoke.sh'
