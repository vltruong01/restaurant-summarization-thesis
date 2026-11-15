"""Technical terminal-beam repair/resumption. No model/GPU import here."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json

from stage_i_repetition_pilot_v1 import ROOT, read, rows, sha, write, require, deadline
from stage_i_repetition_pilot_v1 import preflight as original_preflight

CONFIG = ROOT / 'configs/stage_i_repetition_guard_pilot_v2.json'
PREP = ROOT / 'outputs/stage_i/repetition_guard_preparation_v2'
RUNNER = ROOT / 'scripts/run_stage_i_repetition_guard_pilot_v2.py'
OLD_RUN = ROOT / 'outputs/stage_i/repetition_guard_pilot_v1'


def preflight(require_fresh=True):
    old, _, cases, old_proof = original_preflight(require_fresh=False)
    c, manifest = read(CONFIG), read(PREP / 'PREFLIGHT_MANIFEST.json')
    for rel, digest in manifest['files'].items():
        p = Path(rel)
        require(sha(p if p.is_absolute() else ROOT/p) == digest, 'V2 integrity mismatch: '+rel)
    for k in ('model_id','revision','seed','generation','budgets','runtime_pins','adapter_directory','cases',
              'saved_diagnostic_B','expected_generation_config','backbone_fingerprints','final_references',
              'tokenizer_directory','precision','CPU_threads','minimum_memory_headroom','policy',
              'viability_deadline_utc','original_D059_deadline_utc'):
        require(c[k] == old[k], 'Scientific setting changed: '+k)
    require(c['experiment']=='repetition_guard_pilot_v2' and c['scope']==old['scope'], 'V2 scope changed')
    require(c['GPU_execution_approved'] is False and not c['training_authorized'], 'Preparation is not authorization')
    require(c['new_generation_calls_max']==19 and c['reuse_count']==5 and c['terminal_repair_prefix_tokens']==767,
            'Repair/reuse execution ceiling changed')
    original_artifacts = read(ROOT/'outputs/stage_i/repetition_guard_independent_audit_v1/SOURCE_ARTIFACT_SHA256.json')
    # Audit file hashes are also manifest-bound; use its immutable original
    # records rather than trusting an arbitrary current raw output at resume.
    bound = original_artifacts.get('files', original_artifacts)
    for rel, digest in bound.items():
        if isinstance(digest, str) and len(digest)==64:
            p=Path(rel); require(sha(p if p.is_absolute() else ROOT/p)==digest, 'V1 audited artifact changed: '+rel)
    from stage_i_repetition_guard_v1 import analyze
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(ROOT/c['tokenizer_directory']/'tokenizer.json'))
    reused=[]
    for case in cases[:5]:
        p=OLD_RUN/'raw_cases'/(case['business_id']+'.json'); a=OLD_RUN/'analysis'/p.name
        prediction=read(p)
        require(prediction['business_id']==case['business_id'] and prediction['source_sha256']==case['source_sha256'], 'V1 reuse mapping differs')
        require(prediction['condition']=='filtered' and prediction['subset']==case['subset'], 'V1 reuse condition differs')
        require(tokenizer.decode(prediction['token_ids'],skip_special_tokens=True)==prediction['text'], 'V1 reuse text differs')
        require(len(prediction['token_ids'])<767, 'Terminal repair could affect a reused output')
        actual=analyze(prediction['text'],case['source'],True)
        require(not actual['violations'] and actual==read(a), 'V1 reused output violates unchanged rules')
        reused.append(prediction)
    require(len(list((OLD_RUN/'raw_cases').glob('*.json')))==5, 'Unexpected original completed-case count')
    if require_fresh:
        require(not (ROOT/c['run_root']).exists(), 'V2 namespace exists; no automatic retry/overwrite')
    return c,manifest,cases,reused,{'status':'OFFLINE_V2_PREFLIGHT_PASS_NO_MODEL',
            'new_bindings':len(manifest['files']),'original_preflight':old_proof,'reuse':5,'remaining_attempts_max':19,
            'hard_rule_function':'exact_v1_analyze_import','deadline_unchanged':True}


def authorization(path):
    require(path is not None and Path(path).is_file(), 'Explicit repair/continuation authorization missing')
    a=read(path)
    require(a.get('experiment')=='repetition_guard_pilot_v2' and a.get('GPU_execution_approved') is True, 'Wrong V2 approval')
    require(a.get('approval_source')=='researcher_message' and a.get('researcher_explicit_authorization'), 'No explicit user instruction')
    for key,p in [('config_sha256',CONFIG),('preflight_manifest_sha256',PREP/'PREFLIGHT_MANIFEST.json'),('runner_sha256',RUNNER)]:
        require(a.get(key)==sha(p), 'V2 authorization hash mismatch: '+key)
    require(a.get('new_generation_calls_max')==19 and a.get('reused_outputs')==5 and a.get('retry_failed_case6_once') is True,
            'Repair authorization ceiling differs')
    require(a.get('continue_expected_terminal_infeasibility') is True, 'Expected failure handling not declared')
    for key in ('training_authorized','Teacher_calls_authorized','held_out_access','RO_U_Shuffled_authorized','automatic_retry','fallback_authorized','relax_hard_rules'):
        require(a.get(key) is False,'Forbidden or missing denial: '+key)
    require(a.get('viability_deadline_utc')=='2026-10-22T07:11:22.846767+00:00','Deadline reset')
    stamp=datetime.fromisoformat(a['execution_authorized_at_utc'])
    require(stamp.utcoffset().total_seconds()==0 and stamp<=datetime.now(timezone.utc),'Invalid actual approval time')
    deadline();return a
