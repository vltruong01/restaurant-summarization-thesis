"""Read-only SSH status for the one authorized V3 suite."""
from pathlib import Path
import base64,json,subprocess
OUT=Path(__file__).resolve().parents[1]/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1'
CODE='''from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/workspace/CE_V2_WORKSPACE/runs/CF_EXPLORATORY_NATIVE_CE_V3');ops=Path('/workspace/cf_native_ce_v3_ops')
result=dict(UTC=datetime.now(timezone.utc).isoformat(),suite_exists=root.exists(),safety={},arms={})
for name in ['HOST_PREFLIGHT_PASS.json','STATUS.json','EXPORT_RECEIPT.json','EXPORT_FAILURE.json']:
 if (ops/name).is_file():result[name]=json.loads((ops/name).read_text())
for name in ['FAILURE_STOP_NO_RETRY.json','COMPLETION.json','BUDGET.json']:
 if (root/name).is_file():result[name]=json.loads((root/name).read_text())
for arm in ['ce','pair']:
 p=root/'live_safety'/arm/'SAFETY.json'
 if p.is_file():
  r=json.loads(p.read_text());result['safety'][arm]={k:r[k] for k in ['status','successful_optimizer_updates','memory']}
 p=root/('train_'+arm)/'optimizer_updates.jsonl';count=len(p.read_text().splitlines()) if p.is_file() else 0
 p=root/('train_'+arm)/'progress.jsonl';last=json.loads(p.read_text().splitlines()[-1]) if p.is_file() and p.read_text().strip() else None
 result['arms'][arm]=dict(successful_updates=count,last_progress=last)
result['generation_calls']=len((root/'generation_calls.jsonl').read_text().splitlines()) if (root/'generation_calls.jsonl').is_file() else 0
p=ops/'SUITE.log'
if p.is_file():result['log_tail']=p.read_text(errors='replace').splitlines()[-8:]
result['supervisor']=subprocess.run(['supervisorctl','status','cf_native_ce_v3'],capture_output=True,text=True,timeout=15).stdout.strip()
print(json.dumps(result))
'''
payload=base64.b64encode(CODE.encode()).decode()
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-p','30556','root@163.142.222.112',
    "python3 -c \"import base64;exec(base64.b64decode('"+payload+"'))\""],capture_output=True,text=True,timeout=50)
if r.returncode!=0:raise RuntimeError(r.stderr)
value=json.loads(r.stdout.strip())
with (OUT/'STATUS_SNAPSHOTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
print(json.dumps(value,indent=2))
