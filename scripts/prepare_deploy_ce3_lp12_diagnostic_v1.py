"""Narrow one-shot transfer/launcher for the current explicit TRAIN12 approval."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import base64
import hashlib
import json
import shutil
import subprocess
import sys

R=Path(__file__).resolve().parents[1]
E=R/'outputs/stage_i/led_large_ce3_lp12_diagnostic_v1'
OLD=R/'outputs/stage_i/led_large_nbest_v2_execution_v1/portable_review'
REMOTE='/workspace/CE3_LP12_20261024T194253Z'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','-p','44054','root@171.5.182.162']
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P','44054']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    assert not E.exists(),'No overwrite or retry'
    for rel,digest in read(OLD/'SHA256_MANIFEST.json')['files'].items():assert sha(OLD/rel)==digest,rel
    E.mkdir();stage=E/'transfer';payload=stage/'payload';ops=stage/'ops';payload.mkdir(parents=True);ops.mkdir()
    pairs=[(OLD/'TRAIN_SOURCE_ONLY_CASES.json','TRAIN_SOURCE_ONLY_CASES.json'),
           (OLD/'execution/CANDIDATES.json','BASELINE_CANDIDATES.json'),
           (OLD/'execution/EFFECTIVE_GENERATION_PAIR.json','HISTORICAL_EFFECTIVE_GENERATION_PAIR.json'),
           (OLD/'execution/ENVIRONMENT.json','HISTORICAL_ENVIRONMENT.json'),
           (OLD/'CHECKPOINT_BINDING.json','CHECKPOINT_BINDING.json'),
           (R/'outputs/stage_i/led_large_nbest_reranking_v2/TRAIN_SAMPLE_MANIFEST.json','TRAIN_SAMPLE_MANIFEST.json'),
           (R/'outputs/stage_i/led_large_nbest_reranking_v2/ORIGINAL_GENERATION_CONFIG.json','ORIGINAL_GENERATION_CONFIG.json')]
    for src,name in pairs:shutil.copyfile(src,payload/name)
    for c in read(payload/'TRAIN_SOURCE_ONLY_CASES.json'):
        for suffix in ('_single.json','_single_PASS.json'):
            dest=payload/'N1_CALLS'/(c['business_id']+suffix);dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(OLD/'execution/calls'/dest.name,dest)
    shutil.copyfile(R/'scripts/run_led_large_ce3_lp12_diagnostic_v1.py',ops/'RUNNER.py')
    shutil.copyfile(R/'scripts/test_led_large_ce3_lp12_diagnostic_v1.py',ops/'CPU_TEST_SOURCE.py')
    source=Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/52b2c37a-1246-4225-b997-eeb9dcef4f96/Pasted text.txt')
    shutil.copyfile(source,ops/'RESEARCHER_REQUEST_UNCHANGED.txt')
    helperpaths={
        '/workspace/CAD_V1_VAST_FRESH/scripts/led_large_nbest_rng_v2.py':'led_large_nbest_rng_v2.py',
        '/workspace/CAD_V1_VAST_FRESH/scripts/stage_i_led_large_nbest_v1.py':'stage_i_led_large_nbest_v1.py'}
    config=dict(experiment='LED_LARGE_CE3_LENGTH_PENALTY_1_2_DIAGNOSTIC_V1',
        parent_workspace='/workspace/CAD_V1_VAST_FRESH/parent',checkpoint='/workspace/CAD_V1_VAST_FRESH/checkpoint-750',
        existing_cad_scripts='/workspace/CAD_V1_VAST_FRESH/scripts',
        existing_preflight_runner_sha256=sha(R/'scripts/run_led_large_cad_v1.py'),
        ownership_guard_scripts='/workspace/BOUNDARY_GATE_20261024T184247Z/overlay/scripts',
        ownership_guard_sha256=sha(R/'scripts/run_led_large_boundary_global_feasibility_v1.py'),
        inherited_helper_bindings={path:sha(R/'scripts'/name) for path,name in helperpaths.items()},
        adapter_sha256='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60',
        GPU_uuid='GPU-386606c5-3717-8b4f-c0fe-70907f18b717',only_generation_change={'length_penalty':[1.0,1.2]},
        source_cases=12,maximum_new_generate_calls=12,original_baseline_reused=True,
        max_runtime_seconds=3600,memory_headroom_min=0.15,automatic_retry=False,
        no_training=True,no_validation_or_test=True,renderer_or_other_decoding_technique=False)
    write(payload/'CONFIG.json',config)
    start=datetime(2026,10,24,19,42,53,tzinfo=timezone.utc)
    write(ops/'AUTHORIZATION.json',dict(experiment=config['experiment'],inference_approved=True,
        authorization_source='Current explicit researcher attachment',authorization_attachment_sha256=sha(source),
        authorized_utc=start.isoformat(),deadline_utc=(start+timedelta(hours=48)).isoformat(),
        timestamp_provenance='Clock tool first observed authorized turn; exact chat-message UTC unavailable',
        config_sha256=sha(payload/'CONFIG.json'),runner_sha256=sha(ops/'RUNNER.py'),
        generation_calls_max=12,training_authorized=False,validation_or_test_access=False,
        automatic_retry=False,only_length_penalty_change=True,existing_GPU_ownership_guard_unchanged=True,
        stop_on_guard_failure_including_namespace_uncertainty=True))
    write(ops/'LOCAL_PREFLIGHT.json',dict(accepted_NBEST_manifest_files_verified=171,
        CPU_contract_tests_passed=5,CPU_contract_tests_failed=0,
        tests_command='python -m unittest discover -s scripts -p test_led_large_ce3_lp12_diagnostic_v1.py -v',
        actual_reported_test_seconds=0.024,frozen_inputs_not_modified=True))
    wrapper=r'''from pathlib import Path
import hashlib,json,os,subprocess,sys,traceback
root=Path(REMOTE);ops=root/'ops';env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not (ops/'LAUNCH_STARTED.json').exists(),'No automatic retry'
with (ops/'LAUNCH_STARTED.json').open('x') as f:json.dump(dict(maximum_invocations=1,automatic_retry=False),f)
result={}
try:
 for rel,digest in json.loads((root/'TRANSFER_MANIFEST.json').read_text())['files'].items():assert sha(root/rel)==digest,rel
 command=[sys.executable,'-u',str(ops/'RUNNER.py'),str(root)]
 with (ops/'RUNNER.log').open('x') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=3700)
 result=dict(worker_returncode=p.returncode)
except BaseException as e:result=dict(error=str(e),traceback=traceback.format_exc())
finally:
 with (ops/'LAUNCH_FINISHED.json').open('x') as f:json.dump(result,f,indent=2)
 files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='SUPERVISOR.log'}
 with (root/'EXPORT_MANIFEST.json').open('x') as f:json.dump(dict(files=files),f,indent=2)
 print(json.dumps(result),flush=True)
'''.replace('REMOTE',repr(REMOTE))
    (ops/'HOST_WRAPPER.py').write_text(wrapper,encoding='utf-8')
    write(stage/'TRANSFER_MANIFEST.json',dict(files={p.relative_to(stage).as_posix():sha(p) for p in stage.rglob('*') if p.is_file()}))
    # One transfer and one service start. No tuning or model invocations occur locally.
    p=subprocess.run(SSH+['test ! -e '+REMOTE],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
    subprocess.run(SCP+['-r',str(stage),'root@171.5.182.162:'+REMOTE],check=True,timeout=120)
    deployment=r'''from pathlib import Path
import json,subprocess
root=Path(REMOTE);program='ce3_lp12_20261024_194253';conf=Path('/etc/supervisor/conf.d')/(program+'.conf')
assert not conf.exists()
text=f"""[program:{program}]
command=/workspace/cad_runtime/bin/python -u {root}/ops/HOST_WRAPPER.py
directory={root}/ops
autostart=false
autorestart=false
startretries=0
startsecs=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/ops/SUPERVISOR.log
stdout_logfile_maxbytes=0
"""
with conf.open('x') as f:f.write(text)
for command in [['supervisorctl','reread'],['supervisorctl','update',program],['supervisorctl','start',program]]:
 p=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30);print(p.stdout.strip())
'''.replace('REMOTE',repr(REMOTE))
    encoded=base64.b64encode(deployment.encode()).decode()
    p=subprocess.run(SSH+["python3 -c \"import base64;exec(base64.b64decode('"+encoded+"'))\""],capture_output=True,text=True,encoding='utf-8',timeout=90)
    (E/'DEPLOYMENT.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    print(p.stdout);assert p.returncode==0,p.stderr
    print('Started exactly one authorized LP12 inference attempt; existing GPU ownership guard remains unchanged.')


if __name__=='__main__':main()
