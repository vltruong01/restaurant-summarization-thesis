"""Write D058 offline readiness and decision/progress status; no GPU execution."""
import datetime
from stage_i_d058 import *

def main():
    cfg,m=preflight();data=read(PREP/'DATASET_AND_REFERENCE_READINESS.json')
    hashes={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
            'runner_sha256':sha(ROOT/'scripts/run_stage_i_d058_v1.py')}
    write(PREP/'OFFLINE_TEST_RESULT.json',{'status':'PASS','tests_run':18,'failures':0,'errors':0,'skipped':0,
        'command':'.venv-caption-pilot/Scripts/python.exe -m unittest discover -s tests -p test_stage_i_d058.py -v',
        'observed_unittest_duration_seconds':1.522,'result_provenance':'Executed command result in the current offline-preparation turn.',
        'GPU_execution':False,'model_loads':0,'training_updates':0,'new_predictions':0,
        'coverage':['budget/schedule','frozen decoder/architecture','no sixth epoch or pilot resume','fixed24 and permutations',
            'existing reference readiness','false/stale approval rejection','heavy imports behind execution gate','integrity']})
    write(PREP/'D057_ACCEPTANCE_AND_INVESTIGATION_CLOSURE.json',{
        'approval_source':'researcher_message','source_excerpt_verbatim':'**Decision: ACCEPT D-057 diagnostic findings and close the current root-cause investigation.**',
        'accepted':True,'root_cause_investigation_closed':True,'additional_exposure_is_proven_cure':False,
        'D051_strict_replay':'FAIL_UNCHANGED','D058_GPU_execution_approved':False,
        'researcher_next_task':'Prepare one bounded full-data Filtered-only architecture-B viability experiment offline; stop for explicit execution approval.'})
    report={'status':'D058_OFFLINE_READY_FOR_RESEARCHER_AUTHORIZATION','D057_review_accepted_investigation_closed':True,
        'hashes':hashes,'execution_budget':budget(cfg),'bound_files_checked':len(m['files']),
        'offline_tests':{'passed':18,'total':18,'skipped':0},'data':data['dataset_statistics'],
        'zero_caption_validation_case':data['zero_caption_evaluation'],'reference_validation_pass':data['reference_validation']['valid'],
        'proposed_policy_adaptations':['Full2000x5, fresh linear1250/warmup63 instead of512x2/128/7',
            'Disable loss early stopping within hard5-epoch ceiling','Retain all5 epoch snapshots','No automatic best-loss reload; quality compare epochs3/5 only'],
        'quality_policy':'Existing multidomain rubric; researcher-confirmed substantial practical gain, no rigid new percentage cutoffs. Mixed/unconvincing complete result is NO-GO; unreviewed result is PENDING.',
        'runtime_readiness':'Offline data/shape/identity/API-contract tests only. New full run not executed; GPU compatibility/performance not newly measured.',
        'CPU_preparation':'Only original CPU torch.randperm used to freeze5 permutations in GPU-hidden subprocess; no model loaded or CUDA initialized.',
        'GPU_execution_approved':False,'training_updates':0,'generation_calls':0,'Teacher_calls':0,'held_out_access':False,
        'frozen_artifacts_modified':False,'D051_strict_replay':'FAIL_UNCHANGED',
        'strict_stop_loss':'Single attempt. No substantial usable-quality improvement -> end LED-LoRA optimization and return concise alternative strategy; no automatic new diagnostics. GO -> propose fair RO/U/F plan only.',
        'updated_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(PREP/'D058_READINESS_REPORT.json',report)
    text=f'''# D-058 offline readiness — approval pending

D057 findings accepted; current root-cause investigation closed. This is one
practical Filtered-only viability attempt, not a causal architecture comparison.
No model loading, GPU, training, inference, Teacher request or held-out access
took place during this preparation. CPU randperm only; frozen artifacts unchanged.

**Offline checks:18/18 PASS**, no skips. {len(m['files'])} file bindings verified; two original
Filtered NPZ/Teacher JSONL files,2000 train/200 validation, disjoint IDs, no source
or target overflow. All24 evaluation source/target encodes and NPZ alignments
match,20 random/4 targeted unchanged. Existing D053 final-reference validator
PASS; no source annotation restarted. Saved historical full-F predictions match
the frozen baseline records; no regeneration.

## Proposed execution and explicit policy differences

- One fresh B:48 modules/589,824 trainable parameters; pinned LED/tokenizer,
  same LoRA/AdamW/loss/FP16/attention and frozen source/target protocol.
- Five planned epochs maximum:250 updates/epoch,1250 total;10,000 train
  microbatches and1000 validation-loss batches. Fresh linear schedule,warmup63.
- Explicit proposals needing approval: remove loss-triggered early stopping,
  retain5 epoch checkpoints, disable automatic best-loss reload. Do not resume
  the128-step completed pilot or inspect extra generation checkpoints.
- Only epoch3/step750 and epoch5/step1250:24 beam4 outputs each, **48 total**,
  max_length768 including decoder-start, original decoder settings unchanged.
- Historic full F is a saved practical benchmark, not a matched causal control.
  Model-assisted output judgments remain suggestions; researcher confirmation
  is required for human quality claims. No new embedding/paid judge is included.

GO requires researcher-confirmed substantial improvement in usable summaries
on random20 across semantic repetition, important grounded information/opinion
synthesis, with no compensating material grounding/contrast or targeted-zero
failure. Loss/headers/repetition alone do not qualify. Reuse the approved
qualitative rubric, not new numeric win thresholds; show paired counts/evidence
and uncertainty. Complete but mixed/unconvincing -> NO-GO and end this LED-LoRA
optimization route. Unverified quality -> PENDING, never automatic GO. If both
fixed checkpoints qualify comparably, prefer epoch3. Stop after the report.

Full plan/criteria: `D058_PLAN_AND_AUTHORIZATION.md` / spec009l.
Training executable is prepared but **not runtime/GPU tested** in D058. Old B
training is evidence of feasibility, not a guarantee. Planning range roughly4–7h
training locally from the old2194.33s B512x2 observation;48 full beam generations
are additional and runtime is uncertain. No rental or paid execution approved.

## Researcher authorization bindings

| Artifact | SHA-256 |
| --- | --- |
| Config | `{hashes['config_sha256']}` |
| Preflight manifest | `{hashes['preflight_manifest_sha256']}` |
| Runner | `{hashes['runner_sha256']}` |

The new authorization template stays false. To execute, a separate explicit
researcher message must approve this full-data/schedule/checkpoint policy,
one train run<=1250 updates and48 fixed evaluations, binding these hashes.
No RO/U runs, extra epoch/decoder/architecture/seed sweep, unapproved resume,
Teacher generation, reference/dataset/checkpoint edits or test use. D051 strict
replay remains **FAIL_UNCHANGED**. No automatic D059/D060 diagnostics.

After explicit approval only, the prepared commands are:

```powershell
.\\.venv-caption-pilot\\Scripts\\python.exe scripts\\run_stage_i_d058_v1.py train --execute --approval <D058_researcher_approval.json>
.\\.venv-caption-pilot\\Scripts\\python.exe scripts\\run_stage_i_d058_v1.py evaluate --execute --approval <D058_researcher_approval.json>
```

Do not run now. Missing/false/stale authorization stops before model imports.
'''
    write(PREP/'D058_READINESS_REPORT.md',text)
    entry=f'''## D-058 — D057 accepted/closed; one full-data F-only B viability attempt prepared (GPU NOT approved)

Researcher accepts D057 diagnostic findings and closes the current root-cause
investigation. Weak learned early transitions are supported; exposure is not a
proven cure; historical full F remains repetitive/grounding-limited. No further
numerical diagnostic series is planned. **D051 strict replay FAIL unchanged.**

Offline D058 plan/spec009l and gated runner prepared using existing infrastructure.
One fresh encoder+decoder B (48 modules/589,824 parameters), all2000 Filtered
training businesses/200 validation, maximum5 epochs. Same approved inputs/targets,
backbone/tokenizer/LoRA/optimizer/batch/precision and beam4 decoding. New explicit
schedule1250 updates/warmup63 (same5% ratio), no short-pilot resume. Proposed
checkpoint-policy adaptations: disable loss early-stop; retain5 epoch snapshots;
do not reload best loss. Evaluate only epochs3/5 (steps750/1250), fixed random20
and targeted4,48 new outputs. Saved historical full-F is a practical, noncausal
benchmark, not a matched architecture control. No Teacher/test/annotation changes.

GO: researcher confirms substantial multidomain usable-quality improvement on
random20 without compensating grounding/contrast/targeted-zero regressions.
Existing no-rigid-percentage rubric retained; headers/loss/repetition alone fail.
Mixed/unconvincing complete result -> NO-GO and stop LED-LoRA optimization;
unreviewed -> PENDING. No automatic D059/D060 diagnostics. If promising, propose
fair RO/U/F with the same selected protocol, not execute it automatically.

Offline18/18 tests PASS;{len(m['files'])} file bindings verified; original2000/200 Filtered data,
all24 evaluation source/target token bindings and saved baseline match. Existing
D053 final-reference validator PASS; no new annotation. CPU randperm only:
**model loads/GPU/training/new predictions/Teacher/test access=0**. Config/protocol
remain proposed, not approved for execution. Local training planning estimate4–7h
plus48 beam generations, not a fresh GPU benchmark or guarantee.

Plan: `outputs/stage_i/d058_preparation_v1/D058_PLAN_AND_AUTHORIZATION.md`;
readiness: `outputs/stage_i/d058_preparation_v1/D058_READINESS_REPORT.md`.
ConfigSHA `{hashes['config_sha256']}`;
preflightSHA `{hashes['preflight_manifest_sha256']}`;
runnerSHA `{hashes['runner_sha256']}`.
**STOP: explicit researcher execution approval required before training.**

'''
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/rel;old=p.read_text(encoding='utf-8');require(entry.splitlines()[0] not in old,'Status already recorded')
        write(PREP/'documentation_before'/p.name,old);head,_,rest=old.partition('\n')
        p.write_text(head+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8',newline='\n')
    files=m['files'].copy()
    for p in PREP.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','scripts/finalize_stage_i_d058_preparation_v1.py']:
        files[rel]=sha(ROOT/rel)
    write(PREP/'FINAL_PREPARATION_SHA256_MANIFEST.json',{'files':files,'scope':'Frozen inputs and offline preparation; excludes self and later receipts.'})
    write(PREP/'AUTHORIZATION_HASH_RECEIPT.json',{**hashes,'final_preparation_manifest_sha256':sha(PREP/'FINAL_PREPARATION_SHA256_MANIFEST.json'),'GPU_execution_approved':False})
    print(json.dumps({'status':report['status'],'tests':'18/18 PASS','hashes':hashes,'GPU_execution_approved':False},indent=2))

if __name__=='__main__':main()
