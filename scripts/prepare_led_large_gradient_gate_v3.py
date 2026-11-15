"""Offline preparation of a fresh evidence-only gate revision. Never runs the GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cloud_handoff/led_large_f_colab_v2'
REL = Path('outputs/stage_i/led_large_gradient_gate_v3')
PREP = ROOT / REL


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False))
        f.write('\n')


def prepare():
    assert not PREP.exists()
    PREP.mkdir()
    old_config = json.loads((OLD / 'configs/stage_i_led_large_colab_v2.json').read_text())
    config = dict(old_config, diagnostic_scope=dict(version=3, purpose='distinguish missing versus nonfinite gradients',
                  scientific_settings_changed=False, permitted_mode='feasibility', train_authorized=False,
                  no_automatic_retry=True, original_deadline_utc='2026-10-21T17:11:48+00:00'))
    config_path = ROOT / 'configs/stage_i_led_large_gradient_gate_v3.json'
    save(config_path, config)
    original = (OLD / 'scripts/run_stage_i_led_large_colab_v2.py').read_text(encoding='utf-8')
    source = original[:original.index('\ndef train(')]
    extra = '''
from led_large_gradient_evidence_v3 import inspect_gradients, finite_value, snapshot_raw_gradients, snapshot_rng

PATCH_PREP = ROOT/'outputs/stage_i/led_large_gradient_gate_v3'
PATCH_CONFIG = ROOT/'configs/stage_i_led_large_gradient_gate_v3.json'
PATCH_RUNNER = ROOT/'scripts/run_stage_i_led_large_gradient_gate_v3.py'

def binding():
    return dict(config_sha256=sha(PATCH_CONFIG),runner_sha256=sha(PATCH_RUNNER),
                preflight_manifest_sha256=sha(PATCH_PREP/'PREFLIGHT_MANIFEST.json'))

def authorize_patch(path, execute):
    require(execute and path is not None,'New diagnostic approval required; GPU disabled')
    a=read(path)
    require(a.get('GPU_execution_approved') is True and a.get('researcher_explicit_authorization'), 'New explicit researcher approval required')
    require(a.get('experiment')=='LED_LARGE_GRADIENT_GATE_V3' and a.get('approval_source')=='researcher_message', 'Wrong approval scope/provenance')
    require(a.get('scopes')==['feasibility'] and a.get('training_authorized') is False, 'Gate-only scope required')
    require(all(a.get(k)==v for k,v in binding().items()),'New diagnostic approval hash mismatch')
    require(a.get('execution_deadline_utc')=='2026-10-21T17:11:48+00:00','Original deadline must remain unchanged')
    require(a.get('feasibility_updates_max')==3 and a.get('feasibility_microbatches_max')==24
            and a.get('technical_generation_calls_max')==2,'Gate ceiling changed')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False,'Additional scope forbidden')
    check_deadline(a)
    return dict(a,_approval_sha256=sha(path))

def preflight_patch():
    c=preflight()  # original v2 runtime pins and all immutable frozen input bindings
    new=read(PATCH_CONFIG)
    require(all(new[k]==v for k,v in c.items()),'Original v2 config changed')
    require(new['diagnostic_scope']['train_authorized'] is False,'Gate-only revision required')
    for rel,digest in read(PATCH_PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Diagnostic/frozen dependency changed: '+rel)
    return c

'''
    pos = source.index('\n\nclass MemoryMeter')
    source = source[:pos] + '\n' + extra + source[pos:]
    source = source.replace('start = time.perf_counter()\n    try:',
                            "start = time.perf_counter()\n    diagnostics = dict(stage='before_model_load', completed_optimizer_updates=0)\n    try:",1)
    source = source.replace('for step in range(1,4):', "for step in range(1,4):\n                diagnostics.update(step=step, stage='before_accumulation', scaler_scale=scaler.get_scale())",1)
    source = source.replace('with torch.autocast(\'cuda\',dtype=torch.float16):\n                        loss = model(**batch,use_cache=False).loss',
        "diagnostics.update(microbatch=j+1, business_id=ids[((step-1)*8+j)%len(ids)], stage='forward')\n                    with torch.autocast('cuda',dtype=torch.float16):\n                        loss = model(**batch,use_cache=False).loss\n                    diagnostics.update(stage='forward_complete', CE_loss=finite_value(loss.detach().item()), scaler_scale=scaler.get_scale())\n                    append(p/'MICROBATCH_EVIDENCE.jsonl',dict(diagnostics))",1)
    source = source.replace('losses.append(float(loss.detach()));scaler.scale(loss/8).backward()',
        "losses.append(float(loss.detach()));scaler.scale(loss/8).backward()\n                    diagnostics.update(stage='scaled_backward_complete')\n                    evidence=inspect_gradients(model)\n                    append(p/'SCALED_GRADIENT_EVIDENCE.jsonl',dict(step=step,microbatch=j+1,scaler_scale=scaler.get_scale(),**evidence))",1)
    source = source.replace('scaler.unscale_(opt);grads = gradient_record(model,expect_later_A=step>=3)',
        "scaler.unscale_(opt)\n                diagnostics.update(stage='after_unscale_before_original_gradient_assertion',scaler_scale=scaler.get_scale())\n                evidence=inspect_gradients(model,detailed=True)\n                write(p/f'UNSCALED_GRADIENTS_step{step}.json',dict(diagnostics,**evidence))\n                if evidence['missing'] or evidence['nonfinite']:\n                    snapshot_raw_gradients(model,p/f'RAW_UNSCALED_GRADIENTS_step{step}.pt')\n                grads = gradient_record(model,expect_later_A=step>=3)",1)
    source = source.replace("write(p/f'update_{step}.json',record);records.append(record);memory.check()",
        "write(p/f'update_{step}.json',record);records.append(record);memory.check()\n                diagnostics.update(completed_optimizer_updates=step,stage='update_complete')",1)
    source = source.replace("write(p/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=traceback.format_exc(),automatic_retry=False))",
        "saved_traceback=traceback.format_exc()\n        try:\n            if 'memory' in locals():\n                diagnostics['memory_summary']=memory.summary()\n                write(p/'FAILURE_MEMORY_SAMPLES.json',memory.samples)\n            if 'scaler' in locals():\n                diagnostics['scaler_state']=scaler.state_dict()\n            if 'opt' in locals():\n                diagnostics['optimizer_state_steps']=sorted({int(s['step']) for s in opt.state.values()})\n            if 'model' in locals():\n                diagnostics['RNG_fingerprints']=snapshot_rng(torch)\n            write(p/'FAILURE_DIAGNOSTICS.json',diagnostics)\n        except BaseException as capture_error:\n            write(p/'FAILURE_DIAGNOSTIC_CAPTURE_ERROR.json',dict(error=str(capture_error)))\n        write(p/'FAILURE_STOP_NO_RETRY.json',dict(error=str(exc),traceback=saved_traceback,automatic_retry=False))",1)
    source += '''
def main():
    ap=argparse.ArgumentParser(description='Versioned evidence-only LED-large feasibility gate. No training retry.')
    ap.add_argument('mode',choices=('feasibility',))
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--approval',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    a=authorize_patch(args.approval,args.execute)
    c=preflight_patch()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out=args.output.resolve()
    require(out!=ROOT and not out.is_relative_to(ROOT/'data') and not out.exists(),'Fresh diagnostic output required; preserve failed v2 run')
    feasible(c,a,out)

if __name__=='__main__':main()
'''
    compile(source, 'gradient_gate_v3', 'exec')
    runner_path = ROOT / 'scripts/run_stage_i_led_large_gradient_gate_v3.py'
    save(runner_path, source)
    save(PREP / 'EXACT_RUNNER_DIFF.patch', ''.join(__import__('difflib').unified_diff(
        original.splitlines(True), source.splitlines(True),fromfile='frozen/run_stage_i_led_large_colab_v2.py',tofile='new/run_stage_i_led_large_gradient_gate_v3.py')))
    # Bind all unchanged input/runtime dependencies transitively through the original v2 manifest.
    original_bindings = ['configs/stage_i_led_large_colab_v2.json',
        'scripts/stage_i_led_large_runtime_v2.py','scripts/run_stage_i_led_large_colab_v2.py',
        'outputs/stage_i/led_large_colab_preparation_v2/PREFLIGHT_MANIFEST.json',
        'outputs/stage_i/led_large_colab_preparation_v2/INITIALIZATION_HASHES.json']
    files = {rel:sha(OLD/rel) for rel in original_bindings}
    for p in (config_path,runner_path,ROOT/'scripts/led_large_gradient_evidence_v3.py',ROOT/'tests/test_led_large_gradient_evidence_v3.py'):
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    save(PREP/'PREFLIGHT_MANIFEST.json',dict(files=files, inherited_full_preflight=True,
                scientific_settings_changed=False, GPU_execution_approved=False))
    bindings=dict(config_sha256=sha(config_path),runner_sha256=sha(runner_path),preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))
    save(PREP/'EXECUTION_BINDINGS.json',bindings)
    template=dict(experiment='LED_LARGE_GRADIENT_GATE_V3',GPU_execution_approved=False,
        approval_source='researcher_message',researcher_explicit_authorization=None,
        execution_authorized_at_utc=None,execution_deadline_utc='2026-10-21T17:11:48+00:00',
        defense_reserve_at_least_14_days_confirmed=True,scopes=['feasibility'],training_authorized=False,
        feasibility_updates_max=3,feasibility_microbatches_max=24,technical_generation_calls_max=2,
        held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,**bindings)
    save(PREP/'GPU_APPROVAL_TEMPLATE_NOT_APPROVED.json',template)
    print(json.dumps(bindings,indent=2))


if __name__=='__main__':prepare()
