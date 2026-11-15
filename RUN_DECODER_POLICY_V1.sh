#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
APPROVAL="${1:?Supply the NEW researcher-signed decoder-policy approval JSON}"
PY=.venv/bin/python
test -x "$PY" || { echo 'Pinned parent runtime missing; STOP'; exit 1; }
"$PY" -c "import sys;from pathlib import Path;sys.path.insert(0,'scripts');import stage_i_led_large_decoder_policy_v1 as m;m.authorize(Path(sys.argv[1]),'feasibility',True)" "$APPROVAL"
test ! -e runs/F_large_decoder_policy_v1 || { echo 'Fresh namespace required; preserve existing evidence'; exit 1; }
test ! -e decoder_policy_launcher_logs || { echo 'Prior launcher evidence exists; STOP without retry'; exit 1; }
mkdir decoder_policy_launcher_logs
exec > >(tee decoder_policy_launcher_logs/run.log) 2>&1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" scripts/run_stage_i_led_large_decoder_policy_v1.py preflight
"$PY" scripts/check_led_large_vast_host_v1.py --report decoder_policy_launcher_logs/HOST_PREFLIGHT.json
for MODE in feasibility train evaluate; do
  "$PY" -u scripts/run_stage_i_led_large_decoder_policy_v1.py "$MODE" --execute --approval "$APPROVAL" --output runs/F_large_decoder_policy_v1
done
"$PY" scripts/report_stage_i_led_large_decoder_policy_v1.py --output runs/F_large_decoder_policy_v1
echo 'STOP: technical sanity4 and final48 outputs preserved. Independent researcher quality review required; no RO/U.'
