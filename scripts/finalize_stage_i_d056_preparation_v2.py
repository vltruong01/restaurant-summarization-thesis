"""Offline final report/docs/receipt. No diagnostic runner execution."""
import datetime,subprocess,sys
from stage_i_d056_v2 import *

def main():
    check_config(read(CONFIG));manifest=verify_manifest(OUT/'PREFLIGHT_MANIFEST.json')
    tests=read(sorted((OUT/'offline_test_attempts').glob('attempt_*.json'))[-1])
    require(tests['status']=='PASS' and tests['tests_run']==111 and tests['skipped']==0,'Final offline suite incomplete')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/check_stage_i_d056_offline_v2.py')],cwd=ROOT,capture_output=True,text=True)
    write(OUT/'FINAL_OFFLINE_RECHECK.log',result.stdout+result.stderr)
    require(result.returncode==0,'Final repeatable checker failed')
    check=json.loads(result.stdout);write(OUT/'FINAL_OFFLINE_RECHECK.json',check)
    cfg=read(CONFIG);authorization=read(OUT/'GPU_AUTHORIZATION_TEMPLATE.json')
    require(authorization['GPU_execution_approved'] is False,'GPU authorization unexpectedly present')
    hashes={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(OUT/'PREFLIGHT_MANIFEST.json'),
            'authorization_template_sha256':sha(OUT/'GPU_AUTHORIZATION_TEMPLATE.json'),
            'runtime_runner_sha256':sha(ROOT/'scripts/run_stage_i_d056_gpu_diagnostic_v2.py')}
    write(OUT/'RESEARCHER_OFFLINE_AMENDMENT_AUTHORIZATION.json',{
        'decision':'D-056-v2','approval_source':'researcher_message','record_created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'researcher_instruction':'Prioritize p=3/11; strengthen implementation validation; create versioned amendment; complete offline tests; stop before GPU authorization.',
        'offline_amendment_preparation_authorized':True,'GPU_execution_approved':False,'training_authorized':False,
        'original_scientific_artifacts_may_change':False,'no_fabricated_UI_interaction':True,'hash_bindings':hashes})
    report={'status':'READY_FOR_RESEARCHER_GPU_EXECUTION_DECISION_OFFLINE_ONLY','amendment':'D-056-v2',
        'case_ids':CASES,'states':STATES,'cache_prefix_lengths':[3,11],'cache_calls_per_comparison':3,
        'ceilings':cfg['ceilings'],'trace_new_token_cap':64,'teacher_forced_retains_p12_visual_boundary':True,
        'full_offline_suite':tests,'bound_workspace_files':len(manifest['files']),
        'preserved_v1_files':len(read(OUT/'PARENT_V1_PRESERVATION_MANIFEST.json')),
        'saved_adapter_tensors_historical_fingerprint_match':144,'canonical_backbone_records':295,
        'independent_raw_and_processed_top10':True,'beam_ancestry_and_EOS_censor_finalization_checked_offline':True,
        'numerical_screens_changed':False,'model_loads':0,'model_forwards':0,'GPU_calls':0,'training_updates':0,
        'Teacher_calls':0,'held_out_access':False,'frozen_artifacts_changed':False,'D051_strict_replay':'FAIL_UNCHANGED',
        'hashes':hashes,'GPU_execution_approved':False,
        'remaining_runtime_limits':['No actual backbone Parameter rehash/model attachment/LED or PEFT forward has occurred.',
            'NumPy collator shim is not a real Torch/GPU compatibility test.',
            'No cache-equivalence, precision, memory, beam-output or quality result is claimed.',
            'Three diagnostic cases cannot select a winning architecture.'],
        'preserved_engineering_failures':['engineering_compile_failure_001.txt','offline_test_attempts/attempt_003.log'],
        'failed_attempt_resolution':'Removed a new test syntax typo; changed package-path inspection from guarded importlib.util lookup to non-importing PathFinder lookup. The import guard and numerical screens were not relaxed.'}
    write(OUT/'FINAL_READINESS_REPORT.json',report)
    text=f'''# D-056-v2 final readiness report

Status: **READY FOR RESEARCHER DECISION — OFFLINE ONLY**. GPU execution is not authorized.

- Cache probes **p=3/11**, targeting previously observed marker/newline failures.
  p12 and visual-boundary observations remain in existing teacher-forced calls.
- **111/111 tests PASS**, 0 errors/failures/skips in the final run: unchanged v1
  suite38 plus v2 suite73. Includes regression and negative controls.
- **{len(manifest['files'])} workspace file hashes verified**, including all original v1 bindings;
  **53 v1 files preserved**, including its delivered ZIP/config/proposal/tests.
- Historical Tensor/Parameter serializer checked using isolated original AST,
 18 synthetic fixtures and **144 saved D052 adapter payloads**. The 295 saved
 canonical backbone Parameter fingerprints agree across A/B. Actual backbone
 tensor rehash is deferred until authorized loading; the cached weight-file SHA
 was verified. Parameter type prefix is preserved, not replaced by Tensor/file SHA.
- Exact tokenization, padded source/labels, global/padding masks, best-checkpoint
 identity, saved adapter shapes/keys/dtypes and resolved pinned runtime source
 paths/versions verified without loading a model. Real collator Python body was
 exercised using a disclosed NumPy shim, not real Torch.
- Independent per-beam raw-logit and processed-score top10 with IDs/values and
 stable token-ID ties. Selected beam transitions are directly ancestry-gathered;
 returned sequence score is reconstructed excluding prompt and including EOS.
- Pinned vectorized finalization distinguishes observed selected EOS, global
 64-token censor and non-EOS censor finalization. It does not append EOS. Censored
 beam winners are not guaranteed prefixes of full768-token winners.
- Numerical screens, all three cases/four states, datasets, checkpoints and
 reference content unchanged. **D051 strict replay remains FAIL**.

## Exact proposed execution scope

Three frozen Filtered validation cases: `{CASES[0]}`, `{CASES[1]}`,
`{CASES[2]}`. Four states: pretrained/no adapter, historical full F, D052 A and B.
At most **24 TF +144 explicit cache forwards +36 traces**, each trace **<=64 newly
emitted tokens**. Cache arithmetic4x3x2x2x3=144; p3 and p11 have explicit uncached,
setup and incremental calls. TF FP32/FP16-autocast are separate; traces use FP16.
The original max_length768 and all generation settings remain frozen apart from
the already proposed diagnostic modes/instrumentation. Trace iterations are
separately bounded<=2304; at most60 encoder evaluations. One model/example live.

FP32 cache atol/rtol1e-4/1e-4, FP16 0.05/0.005, relative floor1e-8;
cross-precision flag maxAbs>0.1 OR argmax change. No tolerance revisions.
2GiB output disk and15% GPU headroom guards. New output only at
`outputs/stage_i/d056_gpu_diagnostic_v2`. Stop/preserve on mismatch, nonfinite,
unknown cache/shape/mask, missing scores or memory failure; no silent fallback.

No backward/training, checkpoint modification, Teacher requests, held-out access,
new decoding experiment or subsequent scientific action. The approval template
remains false; a separate explicit researcher authorization must bind D-056-v2
and the exact hashes below. This is required by the researcher's current instruction.

## Final authorization bindings

| Artifact | SHA-256 |
| --- | --- |
| Config | `{hashes['config_sha256']}` |
| Preflight manifest | `{hashes['preflight_manifest_sha256']}` |
| False authorization template | `{hashes['authorization_template_sha256']}` |
| Prepared GPU runner | `{hashes['runtime_runner_sha256']}` |

## Limits and preserved failures

No model was loaded; forward/generation/backward/GPU/Teacher calls and training
updates are all zero. Actual GPU feasibility, forward-path compatibility, cache
equivalence, numerical behavior and grounded quality remain unmeasured. Existing
D051/D055 failures/evidence remain unchanged. Offline engineering failures are
retained: a test syntax typo and an overly broad test import guard intercepting
metadata-only find_spec. The latter was resolved with non-importing PathFinder
inspection, preserving the guard against actual model imports. Final suite PASS.

Read `D056_V2_AMENDMENT.md`, `REVISED_MINIMAL_GPU_PROTOCOL.md`,
`PROPOSED_GPU_EXECUTION_AUTHORIZATION.md` and the unchanged interpretation tree.
No new training or architecture choice is automatically authorized by any outcome.
'''
    write(OUT/'FINAL_READINESS_REPORT.md',text)
    entry=f'''## D-056-v2 — Final offline amendment complete; GPU authorization pending (2026-10-18)

Researcher independently accepted the D056 literature rationale/design and
authorized only an offline amendment. Cache probes are now **p=3/11** (observed
marker/newline failures), replacing the unexecuted v1 p11/12 proposal. Each uses
three explicit matched-prefix forwards; p12 and visual boundaries remain in TF.
Same three validation cases/four states, 24/144/36 ceilings, <=64 new tokens per
trace, sources/targets/checkpoints/references, numerical screens and generation
policy. No GPU execution or training authorization has been given.

New isolated namespace: `outputs/stage_i/d056_diagnostic_preparation_v2`;
spec `specs/009k_d056_amendment_v2.md`; config
`configs/stage_i_d056_minimal_diagnostic_v2.json`. Original v1 proposal/code/tests/
config/ZIP preserved by53-file parent receipt. Final suite **111/111 PASS**, no
skip; {len(manifest['files'])} bound file hashes verified. Hashing convention checked against original
D050 serializer (reused by D052/D054),144 real adapter payloads and18 synthetic
fixtures. Backbone file verified, but actual295 named-Parameter rehash remains a
future pre-forward check. Independent raw/processed top10, ancestry/direct score
gathers, vectorized finalization and real EOS/censor distinctions prepared. Exact
source/mask/label arrays, adapter checkpoint identity and resolved package source
paths verified offline; NumPy collator shim is not real Torch/GPU execution.

**Model loads/forwards/GPU/Teacher/training/test access=0. D051 strict replay
remains FAIL_UNCHANGED.** No factual-quality or architecture winner claim. Earlier
engineering failures retained. Final report:
`outputs/stage_i/d056_diagnostic_preparation_v2/FINAL_READINESS_REPORT.md`.
Config SHA256 `{hashes['config_sha256']}`; preflight SHA256
`{hashes['preflight_manifest_sha256']}`. Authorization template is false and bound
to these hashes; researcher must separately approve the exact bounded execution.

'''
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/rel;old=p.read_text(encoding='utf-8');require('## D-056-v2 — Final offline amendment complete;' not in old,'Living log already updated')
        first,sep,rest=old.partition('\n');p.write_text(first+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8',newline='\n')
    files=manifest['files'].copy()
    for p in OUT.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','scripts/finalize_stage_i_d056_preparation_v2.py']:
        files[rel]=sha(ROOT/rel)
    write(OUT/'FINAL_PREPARATION_SHA256_MANIFEST.json',{'files':files,'scope':'Preparation and frozen dependencies; excludes this manifest and later delivery receipts to avoid circular hashes.'})
    receipt={**hashes,'final_preparation_manifest_sha256':sha(OUT/'FINAL_PREPARATION_SHA256_MANIFEST.json'),'status':report['status']}
    write(OUT/'FINAL_AUTHORIZATION_HASH_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
