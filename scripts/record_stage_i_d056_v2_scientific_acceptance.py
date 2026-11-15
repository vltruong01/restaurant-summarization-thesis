"""Record scientific acceptance only. Never imports/loads a model or grants GPU permission."""
import datetime,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d057_authorization_gate_v1'
BINDINGS={
    'configs/stage_i_d056_minimal_diagnostic_v2.json':'6f41647861b62c82186d7911a5166f916b2cd081abf3bef9287cf6c336bd538e',
    'outputs/stage_i/d056_diagnostic_preparation_v2/PREFLIGHT_MANIFEST.json':'9983ad481c61d7886e2b4683fce6f177f9a445c02b34906d3f628f7e3b53d1f0',
    'scripts/run_stage_i_d056_gpu_diagnostic_v2.py':'efe4b279da099aa0c3ebc4b51551df2e05bb4b6dc9ef41bcc9827edff1f2dc3d',
}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write((value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))+'\n')

def main():
    observed={rel:sha(ROOT/rel) for rel in BINDINGS}
    if observed!=BINDINGS:raise ValueError('STOP: researcher-reviewed hash mismatch')
    cfg=json.loads((ROOT/'configs/stage_i_d056_minimal_diagnostic_v2.json').read_text(encoding='utf-8'))
    template=ROOT/'outputs/stage_i/d056_diagnostic_preparation_v2/GPU_AUTHORIZATION_TEMPLATE.json'
    approval=json.loads(template.read_text(encoding='utf-8'))
    if cfg['GPU_execution_approved'] is not False or approval['GPU_execution_approved'] is not False:
        raise ValueError('Unexpected GPU permission state')
    if cfg['D051_strict_replay_status']!='FAIL_UNCHANGED':raise ValueError('D051 history mismatch')
    if (ROOT/cfg['future_output']).exists():raise ValueError('Unexpected GPU diagnostic namespace already exists')
    OUT.mkdir(exist_ok=False)
    excerpts=[
        '**Scientific review decision: ACCEPTED — READY FOR BOUNDED GPU EXECUTION AUTHORIZATION.**',
        "**Important execution boundary:** This message records independent scientific acceptance, not the researcher's separate explicit authorization to execute GPU operations. Do not mark GPU_execution_approved=true or begin model execution until that authorization is actually given by the researcher.",
    ]
    record={
        'status':'SCIENTIFIC_ACCEPTED_AWAITING_SEPARATE_RESEARCHER_GPU_AUTHORIZATION',
        'decision':'D-056-v2','prospective_execution_report':'D-057',
        'approval_source':'current_user_message_reporting_independent_scientific_review',
        'source_excerpts_verbatim':excerpts,'recorded_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scientific_review_accepted':True,'GPU_execution_approved':False,
        'researcher_explicit_GPU_execution_authorization':None,'training_authorized':False,
        'reviewed_bindings':BINDINGS,'observed_bindings':observed,'hash_check_status':'3_OF_3_MATCH',
        'case_ids':cfg['cases'],'model_states':cfg['states'],'ceilings':cfg['ceilings'],
        'cache_prefix_lengths':cfg['cache_prefix_lengths'],'cache_calls_per_comparison':cfg['cache_calls_per_comparison'],
        'trace_new_token_cap':cfg['trace_new_token_cap'],
        'unchanged_false_authorization_template':{'path':template.relative_to(ROOT).as_posix(),'sha256':sha(template)},
        'actions_this_step':{'file_hash_verifications':3,'model_loads':0,'GPU_operations':0,'forwards':0,
            'generation':0,'training':0,'Teacher_calls':0,'held_out_access':False,'frozen_artifacts_modified':False},
        'D051_strict_replay':'FAIL_UNCHANGED',
        'after_explicit_GPU_authorization_only':[
            'Transparently record the researcher message, scope and exact reviewed config/preflight/runner hashes.',
            'Rerun the existing offline preflight in the actual GPU execution environment; stop on mismatch.',
            'Execute the frozen prepared runner exactly once; no fallback, tolerance changes or automatic resume.',
            'Preserve raw numerical/trace/runtime/failure evidence and stop on unexpected failure.',
            'Prepare D057 observations/hypotheses/inconclusive evidence/possible-next-experiments report and portable review bundle.',
            'Update decision/progress logs with actual status and stop; no subsequent experiment.'
        ],
        'recording_script_sha256':sha(Path(__file__).resolve()),
    }
    write(OUT/'SCIENTIFIC_ACCEPTANCE.json',record)
    report='''# D-056-v2 scientific acceptance — separate GPU authorization pending

Independent scientific review is ACCEPTED. All three supplied config/preflight/
runner hashes match the files on disk. No offline protocol revision was made.

**GPU_execution_approved remains false.** This acceptance is not a researcher
execution authorization. The current message explicitly requires a separate
researcher authorization before any model/GPU operation. No model was loaded;
no forward, generation, training, Teacher request or held-out access took place.

The exact proposed scope remains the three frozen Filtered validation cases and
four read-only states in the reviewed v2 config, at most24 teacher-forced forwards,
144 cache-control forwards (p3/11, three calls/comparison), and36 traces of at
most64 newly emitted tokens. All frozen settings, numerical screens, sources,
targets, references and checkpoints remain unchanged. D051 strict replay FAIL is
preserved. The false authorization template was neither modified nor replaced.

Next gate: an explicit researcher message authorizing execution of D-056-v2 bound
to all three accepted hashes. Only then record that authorization, rerun existing
offline preflight in the actual environment, execute the prepared runner once,
and produce the D057 diagnostic report/review bundle. Stop on unexpected failure;
do not change settings, apply fallback, retrain or launch subsequent experiments.

This directory contains authorization-gate records, not GPU diagnostic results.
No D057 numerical findings/report are claimed. Existing manifests and delivered
ZIP remain unchanged. Updates to living decision/progress/handoff documents are
new status entries; their prior versions remain in the delivered preparation
snapshot, and its historical all-files manifest is not rewritten.
'''
    write(OUT/'EXECUTION_GATE_STATUS.md',report)
    entry='''## D-056-v2 — Scientific review accepted; separate GPU execution authorization pending

Independent reviewer reports **ACCEPTED — READY FOR BOUNDED GPU EXECUTION
AUTHORIZATION**. Config, preflight-manifest and GPU-runner SHA256 match all three
researcher-supplied bindings. No further offline protocol revision requested or
made. Acceptance explicitly **does not authorize GPU execution**:
`GPU_execution_approved=false`; no model loading/forward/generation/training,
Teacher call or held-out access. Original authorization template stays unchanged.

Exact proposed scope: same three Filtered validation cases/four read-only states;
24 teacher-forced /144 cache-control forwards at p3/11 (three calls each) /36
traces <=64 newly emitted tokens. Numerical screens and frozen artifacts remain
unchanged. **D051 strict replay remains FAIL_UNCHANGED.**

Gate record: `outputs/stage_i/d057_authorization_gate_v1/SCIENTIFIC_ACCEPTANCE.json`;
status: `outputs/stage_i/d057_authorization_gate_v1/EXECUTION_GATE_STATUS.md`.
Only after a separate explicit researcher execution authorization: record its
source/hash scope, rerun existing offline preflight in the actual environment,
execute frozen D056-v2 once, preserve failures/raw evidence, and return a D057
report plus review ZIP. No automatic subsequent experiment. No GPU results exist
at this gate; scientific acceptance is distinct from execution and observations.

'''
    docs=['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']
    for rel in docs:
        p=ROOT/rel;text=p.read_text(encoding='utf-8')
        if entry.splitlines()[0] in text:raise ValueError('Acceptance entry already present')
        write(OUT/'documentation_before'/p.name,text)
        head,_,body=text.partition('\n')
        p.write_text(head+'\n\n'+entry+body.lstrip('\n'),encoding='utf-8',newline='\n')
    files={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}
    files.update(observed);files.update({rel:sha(ROOT/rel) for rel in docs})
    files[Path(__file__).resolve().relative_to(ROOT).as_posix()]=sha(Path(__file__).resolve())
    write(OUT/'SHA256_MANIFEST.json',{'files':files,'scope':'Scientific acceptance and pending execution gate only; no GPU authorization'})
    if {rel:sha(ROOT/rel) for rel in BINDINGS}!=BINDINGS or sha(template)!=record['unchanged_false_authorization_template']['sha256']:
        raise ValueError('Frozen bindings modified unexpectedly')
    print(json.dumps({'status':record['status'],'reviewed_hashes_matching':3,'GPU_execution_approved':False,
        'record':str(OUT/'SCIENTIFIC_ACCEPTANCE.json'),'gate_manifest_sha256':sha(OUT/'SHA256_MANIFEST.json')},indent=2))

if __name__=='__main__':main()
