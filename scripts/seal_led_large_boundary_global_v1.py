"""Finalize only new preparation files; no GPU/network calls or scientific overwrite."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/led_large_boundary_global_v1'
CONFIG=R/'configs/stage_i_led_large_boundary_global_v1.json'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')


def runtime_configs():
    spec=read(CONFIG);old=read(R/spec['parent_config'])
    for arm in spec['arms']:
        value=dict(old,experiment=spec['experiment'],run_root=spec['run_root']+'/'+arm,
            global_attention=old['global_attention'] if arm=='bos_only' else spec['fixed_policy'],
            vast_fresh=dict(old['vast_fresh'],experiment=spec['experiment'],output=spec['run_root']+'/'+arm))
        write(P/f'RUNTIME_CONFIG_{arm}.json',value)
    # Preparation-time adjustment only: lock max-global source first for the full-cache resource trace.
    stress=read(P/'STRESS_CASES.json')
    rows=[json.loads(s) for s in (P/'ANCHOR_ROWS.jsonl').read_text(encoding='utf-8').splitlines()]
    best=max((r for r in rows if r['split']=='train'),key=lambda r:len(r['positions']))
    stress['train_business_ids']=list(dict.fromkeys([best['business_id']]+stress['train_business_ids']))
    (P/'STRESS_CASES.json').write_text(json.dumps(stress,indent=2)+'\n',encoding='utf-8')


def seal():
    require_files=['FOCUSED_TEST_RESULTS.json','PINNED_PARENT_PREFLIGHT.json','EXPERIMENT_SPEC.md']
    assert all((P/n).is_file() for n in require_files)
    assert read(P/'FOCUSED_TEST_RESULTS.json')['status']=='PASS'
    assert read(P/'PINNED_PARENT_PREFLIGHT.json')['status']=='PASS'
    names=['led_large_boundary_global_v1.py','run_led_large_boundary_global_v1.py',
           'prepare_led_large_boundary_global_v1.py','seal_led_large_boundary_global_v1.py']
    technical=[R/'scripts'/n for n in names]+[R/'tests/test_led_large_boundary_global_v1.py',CONFIG,P/'EXPERIMENT_SPEC.md']
    technical+=list(P.glob('*.json'))+list(P.glob('*.jsonl'))
    technical=[p for p in technical if p.name not in ('PREFLIGHT_MANIFEST.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json','EXECUTION_BINDINGS.json','DELIVERY_RECEIPT.json')]
    files={p.relative_to(R).as_posix():sha(p) for p in sorted(set(technical))}
    write(P/'PREFLIGHT_MANIFEST.json',dict(files=files,scope='New implementation/masks; complete immutable V2 parent chain independently required'))
    import sys
    sys.path.insert(0,str(R/'scripts'))
    from run_led_large_boundary_global_v1 import binding
    bindings={arm:binding(arm) for arm in read(CONFIG)['arms']}
    write(P/'EXECUTION_BINDINGS.json',bindings)
    spec=read(CONFIG)
    template=dict(experiment=spec['experiment'],GPU_execution_approved=False,training_authorized=False,
        approval_source='researcher_message',researcher_explicit_authorization=None,approved_arms=spec['arms'],
        scopes=['feasibility','train','evaluate'],bindings=bindings,execution_authorized_at_utc=None,
        execution_deadline_utc=None,researcher_absolute_stop_utc=None,defense_reserve_at_least_14_days_confirmed=None,
        max_training_updates=1250,evaluation_outputs_max=48,early_sanity_outputs_max=4,
        early_sanity_generation_approved=True,technical_generation_calls_max=2,all_generation_calls_max=54,
        feasibility_updates_max=3,feasibility_microbatches_max=88,fresh_only=True,held_out_access=False,
        Teacher_calls=False,RO_U_training_authorized=False,automatic_retry=False,full_cache_stress_approved=True,
        minimum_memory_headroom=.15,amp_scaler=spec['shared']['amp_scaler'],dynamic_overflow_policy=spec['shared']['dynamic_overflow_policy'],
        conditional_train_after_both_complete_gates_pass=True,
        warning='All budgets above are PER ARM. Full suite:6 gate updates+2500 train updates,108generate calls maximum. This false template grants no execution.')
    write(P/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',template)
    original=R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750/adapter_model.safetensors'
    assert sha(original)==spec['old_F_epoch3_adapter_sha256']
    for name,digest in read(P/'FROZEN_INPUT_BINDINGS.json')['files'].items():assert sha(R/name)==digest
    write(P/'FINAL_READINESS.json',dict(status='GO_FOR_RESEARCHER_REVIEW_OF_CONDITIONAL_GPU_AUTHORIZATION',
        current_GPU_execution='BLOCKED_NO_EXPLICIT_APPROVAL',actual_boundary_VRAM='UNVERIFIED_GATE_REQUIRED',
        full_training='BLOCKED_UNTIL_BOTH_REAL_GATES_PASS',frozen_inputs_and_original_CE3_unchanged=True,
        no_model_forward_backward_generation=True,no_GPU_operations=True,no_held_out_access=True,
        no_automatic_policy_fallback_or_mask_search=True))
    bundle=list(P.glob('*'))+technical
    extras=[R/'scripts/stage_i_led_large_decoder_policy_v2.py',R/'scripts/run_stage_i_led_large_decoder_policy_v2.py',
        R/'scripts/led_large_decoder_label_view_v1.py',R/'scripts/led_large_decoder_early_sanity_v2.py',
        R/'scripts/led_large_dynamic_amp_v2.py',R/'scripts/stage_hi_led_data.py',R/'configs/stage_i_led_large_decoder_policy_v2.json',
        R/'scripts/stage_i_paired_pilot.py',R/'scripts/stage_i_d053_evaluation.py',
        R/'outputs/stage_h/led_dataset_v1/manifest.json',R/'assets/tokenizers/led-base-16384.manifest.json',
        R/'configs/stage_i_d053_evaluation_v1.json',
        R/'outputs/stage_i/led_large_preparation_v1/EVALUATION_MANIFEST.json',
        R/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl',
        R/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json',
        R/'outputs/stage_i/led_large_decoder_policy_v1/PINNED_RUNTIME_CONTRACT.json',
        R/'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json',
        R/'outputs/stage_i/d053_final_references_v1/REFERENCE_FREEZE_MANIFEST.json']
    extras += [R/name for name in read(P/'FROZEN_INPUT_BINDINGS.json')['files']]
    extras += [p for p in (R/'assets/tokenizers/led-base-16384').iterdir() if p.is_file()]
    bundle+=extras
    manifest={p.relative_to(R).as_posix():sha(p) for p in sorted(set(bundle)) if p.is_file()}
    archive=R/'outputs/cloud_handoff/LED_LARGE_BOUNDARY_GLOBAL_V1_OFFLINE_REVIEW.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(R/rel,rel)
        doc=(P/'EXPERIMENT_SPEC.md').read_bytes();z.writestr('READ_THIS_FIRST.md',doc)
        manifest['READ_THIS_FIRST.md']=hashlib.sha256(doc).hexdigest()
        z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=manifest),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    write(P/'DELIVERY_RECEIPT.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),
        sha256=digest,bytes=archive.stat().st_size,manifest_entries=len(manifest),GPU_execution_approved=False))
    print(json.dumps(read(P/'DELIVERY_RECEIPT.json'),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('step',choices=['runtime-configs','seal']);a=ap.parse_args()
    runtime_configs() if a.step=='runtime-configs' else seal()
