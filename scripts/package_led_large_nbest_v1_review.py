"""Package the offline proposal and actual CPU evidence; never starts model operations."""
from pathlib import Path
from datetime import datetime,timezone
import json
import re
import subprocess
import zipfile
from stage_i_led_large_nbest_v1 import sha,read

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'outputs/stage_i/led_large_nbest_reranking_v1'


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    python=ROOT/'.venv-caption-pilot/Scripts/python.exe'
    result=subprocess.run([str(python),'-m','unittest','discover','-s','tests','-p','test_led_large_nbest_v1.py','-v'],
                          cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (P/'CPU_TESTS.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stderr
    count=int(re.search(r'Ran (\d+) tests',result.stderr).group(1))
    # Reuse the preserved, complete successful workspace; pure metadata/source/hash preflight.
    program="import sys;from pathlib import Path;w=Path('outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE').resolve();sys.path.insert(0,str(w/'scripts'));import stage_i_led_large_decoder_policy_v2 as p;p.preflight();print('FULL_SUCCESSFUL_PARENT_CPU_PREFLIGHT_PASS_NO_MODEL_FORWARD')"
    parent=subprocess.run([str(python),'-c',program],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (P/'PARENT_CPU_PREFLIGHT.txt').write_text(parent.stdout+'\n'+parent.stderr,encoding='utf-8')
    assert parent.returncode==0,parent.stderr
    save(P/'CPU_VALIDATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),tests_passed=count,
         tests_returncode=result.returncode,tests_sha256=sha(P/'CPU_TESTS.txt'),parent_CPU_preflight='PASS',
         parent_preflight_log_sha256=sha(P/'PARENT_CPU_PREFLIGHT.txt'),test_interpreter=str(python),
         test_scope='Synthetic fixtures; CPU BeamSearchScorer finalization; GenerationConfig serialization; no model construction/forward',
         GPU_calls=0,model_inference_calls=0,optimizer_updates=0,teacher_generation_calls=0,
         global_python_runtime_note='Global interpreter Transformers import fails due huggingface-hub1.30; isolated accepted caption-pilot environment hub0.36.2 used, no global package changes.',
         observed_diversity='NOT_RUN',observed_oracle='NOT_RUN',observed_reranker_accuracy='NOT_RUN',
         historical_GPU_reproduction='NOT_RUN'))
    code=['scripts/stage_i_led_large_nbest_v1.py','scripts/run_stage_i_led_large_nbest_v1.py',
          'scripts/report_led_large_nbest_v1.py','scripts/prepare_led_large_nbest_v1.py',
          'scripts/package_led_large_nbest_v1_review.py','tests/test_led_large_nbest_v1.py',
          'configs/stage_i_led_large_nbest_v1.json']
    exclusions={'PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','SHA256_MANIFEST.json','DELIVERY_RECEIPT.json'}
    files=[ROOT/n for n in code]+[x for x in P.rglob('*') if x.is_file() and x.name not in exclusions]
    files=sorted(set(files))
    manifest={x.relative_to(ROOT).as_posix():sha(x) for x in files}
    save(P/'PREFLIGHT_MANIFEST.json',dict(files=manifest,scope='new additive offline payload; parent workspace verified separately at execution'))
    bindings=dict(config_sha256=sha(ROOT/'configs/stage_i_led_large_nbest_v1.json'),
        runner_sha256=sha(ROOT/'scripts/run_stage_i_led_large_nbest_v1.py'),preflight_sha256=sha(P/'PREFLIGHT_MANIFEST.json'))
    save(P/'EXECUTION_BINDINGS.json',dict(experiment='LED_LARGE_NBEST_RERANKING_V1',authorization_bindings=bindings,
        helper_sha256=sha(ROOT/'scripts/stage_i_led_large_nbest_v1.py'),report_sha256=sha(ROOT/'scripts/report_led_large_nbest_v1.py'),
        policy_sha256=sha(P/'RERANKING_POLICY.json'),prompt_sha256=sha(P/'SOURCE_JUDGE_PROTOCOL.md'),
        training_sample_sha256=sha(P/'TRAIN_SOURCE_ONLY_CASES.json'),checkpoint_adapter_sha256=read(P/'CHECKPOINT_BINDING.json')['files']['adapter_model.safetensors'],
        GPU_inference_approved=False,validation_policy_locked=False))
    files += [P/'PREFLIGHT_MANIFEST.json',P/'EXECUTION_BINDINGS.json']
    save(P/'SHA256_MANIFEST.json',dict(files={x.relative_to(ROOT).as_posix():sha(x) for x in sorted(set(files))}))
    files += [P/'SHA256_MANIFEST.json']
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_NBEST_RERANKING_V1_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for x in sorted(set(files)):z.write(x,x.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        proof=json.loads(z.read(P.relative_to(ROOT).as_posix()+'/SHA256_MANIFEST.json'))['files']
        import hashlib
        for name,h in proof.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    digest=sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    save(P/'DELIVERY_RECEIPT.json',dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,
        portable_manifest_entries_verified=len(proof),CPU_tests_passed=count,new_GPU_operations=0,
        researcher_approval_required=True))
    print(json.dumps(dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,
                         entries=len(proof),tests=count,bindings=bindings),ensure_ascii=False))


if __name__=='__main__':main()
