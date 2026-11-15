"""Offline import-repair restart preparation; no model imports or execution."""
import json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from stage_i_d059_v2 import *


def main():
    require(PREP.exists() and not (PREP/'PREFLIGHT_MANIFEST.json').exists(),'Preserve existing v2 preparation; no overwrite')
    cfg=read(CONFIG);check_config(cfg)
    before=read(PREP/'ORIGINAL_V1_ARTIFACT_SNAPSHOT.json')
    for rel,h in before['files'].items():require(sha(ROOT/rel)==h,'Historical D059 changed: '+rel)
    prior=read(ROOT/'outputs/stage_i/d059_preparation_v1/PREFLIGHT_MANIFEST.json')
    files=dict(prior['files'])
    for rel,h in files.items():require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Prior binding differs: '+rel)
    reused=[]
    for name in ('MODULE_INVENTORY.json','INITIALIZATION_HASHES.json','DATA_AND_MASK_READINESS.json',
                 'FEASIBILITY_CASES.json','EPOCH_ORDERS.json','EVALUATION_MANIFEST.json','BENCHMARKS_UNCHANGED.json','EXECUTION_BUDGET.json'):
        source=ROOT/'outputs/stage_i/d059_preparation_v1'/name;target=PREP/name
        require(not target.exists(),'Preparation collision: '+name);shutil.copyfile(source,target)
        require(sha(source)==sha(target),'Evidence copy differs')
        reused.append({'source':source.relative_to(ROOT).as_posix(),'copy':target.relative_to(ROOT).as_posix(),'sha256':sha(source)})
    write(PREP/'REUSED_OFFLINE_EVIDENCE.json',{'role':'Exact same scientific configuration and preserved input hashes; reuse validated inventory/serialization/alignment evidence without repeating work.',
        'files':reused,'new_meta_model_constructions':0,'new_model_weight_loads':0,'new_GPU_calls':0,'new_source_tokenization':0,
        'old_GPU_feasibility_role':'Historical PASS only; new v2 gate required','D051_strict_replay':'FAIL_UNCHANGED'})
    write(PREP/'HISTORICAL_GATE_BINDING.json',{'path':'outputs/stage_i/d059_expanded_filtered_v1/feasibility/REPORT.json',
        'sha256':sha(ROOT/'outputs/stage_i/d059_expanded_filtered_v1/feasibility/REPORT.json'),
        'role':'HISTORICAL_PASS_ONLY_NOT_V2_EXECUTION_APPROVAL','v2_gate_must_run':True})
    for rel in before['files']:files[rel]=before['files'][rel]
    needed=['scripts/run_stage_i_d059_v2.py','scripts/stage_i_d059_v2.py','scripts/prepare_stage_i_d059_v2.py',
        'configs/stage_i_d059_expanded_lora_v2.json','tests/test_stage_i_d059_v2_contracts.py','tests/test_stage_i_d059_v2_callbacks.py',
        'specs/009m_d059_technical_restart_v2.md']
    for rel in needed:files[rel]=sha(ROOT/rel)
    for path in PREP.rglob('*'):
        if path.is_file():files[path.relative_to(ROOT).as_posix()]=sha(path)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'decision':'D-059','technical_restart_version':2,'files':files,
        'config_sha256':sha(CONFIG),'runner_sha256':sha(RUNNER),'GPU_execution_approved':False,
        'original_execution_authorized_at_utc':'2026-10-18T15:10:48.941817+00:00',
        'improvement_deadline_utc':'2026-10-25T15:10:48.941817+00:00','historical_gate_does_not_authorize_v2':True,
        'model_forward_backward_calls':0,'Teacher_calls':0,'held_out_access':False})
    write(PREP/'AUTHORIZATION_BINDINGS.json',{'decision':'D-059','technical_restart_version':2,
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER),
        'GPU_execution_approved':False,'original_execution_authorized_at_utc':'2026-10-18T15:10:48.941817+00:00',
        'improvement_deadline_utc':'2026-10-25T15:10:48.941817+00:00'})
    template=read(ROOT/'outputs/stage_i/d059_preparation_v1/GPU_AUTHORIZATION_TEMPLATE.json')
    template.update(technical_restart_version=2,GPU_execution_approved=False,training_authorized=False,
        approval_source=None,researcher_explicit_authorization=None,config_sha256=sha(CONFIG),runner_sha256=sha(RUNNER),
        preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'),
        execution_authorized_at_utc='2026-10-18T15:10:48.941817+00:00',improvement_deadline_utc='2026-10-25T15:10:48.941817+00:00',
        restart_approval_recorded_at_utc=None,allow_backbone_download=False,automatic_retry=False,
        approved_feasibility_report_sha256=None,RO_U_training_authorized=False)
    write(PREP/'GPU_AUTHORIZATION_TEMPLATE.json',template)
    write(PREP/'RESEARCHER_REPAIR_PREPARATION_APPROVAL.json',{'approval_source':'researcher_message',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'minimal_repair_and_offline_preparation_approved':True,'GPU_execution_approved':False,
        'decision_quote':'Researcher decision: APPROVE the minimal technical repair and preparation of one versioned fresh restart.',
        'boundary_quote':'Before renewed GPU execution, return the final new hashes, exact diff, regression-test results and versioned gate/training authorization scope for explicit hash-bound approval.',
        'original_clock_preserved':True,'old_authorization_does_not_authorize_v2':True})
    preflight()
    print(json.dumps({'status':'V2_OFFLINE_PREPARED_GPU_NOT_AUTHORIZED','bindings':read(PREP/'AUTHORIZATION_BINDINGS.json'),
        'original_files_verified':len(before['files']),'total_bound_files':len(files)},indent=2))


if __name__=='__main__':main()
