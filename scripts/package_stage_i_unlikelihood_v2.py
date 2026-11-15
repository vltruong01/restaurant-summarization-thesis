"""Freeze ONE offline v2 amendment; portable review ZIP, not GPU execution."""
import difflib
import re
import shutil
import zipfile
from stage_i_unlikelihood_v2 import *


def main():
    require(not (PREP/'PREFLIGHT_MANIFEST.json').exists(),'Final v2 already frozen')
    cfg=read(CONFIG);v1=ROOT/'outputs/stage_i/unlikelihood_preparation_v1'
    log=(PREP/'OFFLINE_TESTS.log').read_text(encoding='utf-8')
    require('Ran 64 tests' in log and log.strip().endswith('OK'),'Expected focused64test suite PASS')
    audit=read(PREP/'NEGATIVE_BINDING_AUDIT.json')
    require(audit['sentence']==512 and audit['word']==0 and audit['businesses']==512,'All512sentence audit not complete')
    write(PREP/'OFFLINE_TEST_RESULT.json',{'status':'PASS','total_tests':64,'v1_regressions':38,'v2_focused':26,
        'synthetic_CPU_autograd_only':True,'LED_model_operations':0,'GPU_operations':0,'optimizer_updates':0})
    protected=dict(read(v1/'PREFLIGHT_MANIFEST.json')['files'])
    for rel,h in protected.items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'V1/historical binding changed: '+rel)
    oldzip=ROOT/'outputs/stage_i/stage_i_unlikelihood_preparation_review_v1.zip'
    require(sha(oldzip)==read(v1/'DELIVERY_RECEIPT.json')['sha256'],'V1 ZIP changed')
    protected[oldzip.relative_to(ROOT).as_posix()]=sha(oldzip)
    for p in v1.rglob('*'):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    benchmark=read(PREP/'D059_EPOCH3_PRACTICAL_BENCHMARK.json')
    for rel in (benchmark['source_path'],benchmark['effective_generation_config_path']):protected[rel]=sha(ROOT/rel)
    diffs=[]
    for base,new in [('stage_i_unlikelihood_v1.py','stage_i_unlikelihood_v2.py'),
                     ('run_stage_i_unlikelihood_pilot_v1.py','run_stage_i_unlikelihood_pilot_v2.py'),
                     ('audit_stage_i_unlikelihood_v1.py','audit_stage_i_unlikelihood_v2.py')]:
        diffs.extend(difflib.unified_diff((ROOT/'scripts'/base).read_text(encoding='utf-8').splitlines(True),
            (ROOT/'scripts'/new).read_text(encoding='utf-8').splitlines(True),fromfile='scripts/'+base,tofile='scripts/'+new))
    write(PREP/'V1_TO_V2_CODE_DIFF.patch',''.join(diffs))
    write(PREP/'READINESS_REPORT.md',f'''# Unlikelihood v2 — offline readiness

READY FOR FINAL RESEARCHER REVIEW; GPU EXECUTION NOT AUTHORIZED.

-512/512 eligible sentence-level negatives;0word negatives;0ineligible cases.
-Previous256sentence views unchanged; all512IDs, epoch orders, positive target
  tokens and source bindings preserved. Full penalty masks independently checked.
-{audit['masked_tokens']}penalized tokens; maximum negative575tokens;3zero-caption
  businesses have review-only negative views. No truncation/fabricated replacement.
-64/64offline tests PASS:38unchanged v1 regressions +26focused v2 tests.
  Synthetic CPU autograd only;0LED model loads/forwards/backwards/generation,
  GPU operations, optimizer updates, Teacher calls or held-out content access.
-445v1/historical bindings reverified, original v1 ZIP receipt matches. Original
  matched RO/U/F, D058/D059/D060 evidence and D051strict replayFAIL unchanged.
-Saved D059 Expanded-LoRA epoch3:24IDs/subsets/source hashes and exact effective
  generation config match the frozen development inputs. No benchmark regeneration.

## Proposed gate and execution scope (unchanged ceilings)

Each arm:3update/24microbatch feasibility gate, no generation/replay updates;
48additional separate autograd traversals maximum, no extra explicit forward calls.
Record current-AMP-scaled/unscaled CE/UL gradients on all48LoRA tensors, norms,
cosine, weighted/effective ratios, zero/missing gradients and .grad/RNG invariance.
Nonzero UL scalar is not evidence of effective optimization. Report weak/zero UL
without adjusting0.1weight. Per-microbatch preclip ratios are not AdamW update
attribution or quality criteria. Checkpoint recomputation may increase gate cost.
Actual memory/AMP/gradients remain UNTESTED; require15%measured memory headroom.

Only if both gates PASS under new approval: two fresh historical-F-initialized
decoder-only continuations (24modules/294912parameters),512train/200validation,
2epochs/128updates per arm,warmup7; same original optimization, seed/order,
full5000source/768labels and beam4. Final epoch2only:24outputs each=48total.
No loss-weight/decoder/search/architecture changes. Control still performs the
same negative forwards with effective UL weight0.

## Quality and interpretation

Matched CE-versus-UL comparison is distinct from non-causal practical comparison
against saved D059 Expanded-LoRA epoch3 (different architecture, initialization,
parameters and training duration). Review random20 and targeted4 separately.
GO requires meaningful reduction of harmful repetition without loss of Core,
opinion balance, grounding, useful visuals or structural/semantic completeness.
Exact repetition/loss alone never passes. AI assessments remain provisional.
These24cases are repeatedly inspected DEVELOPMENT examples. Exact-copy training
does not directly teach paraphrase redundancy or grounding. No result is claimed.

## Authorization / stop

Require explicit approval of new config/preflight/runner hashes, separate exception
to prior final-attempt restrictions, actual UTC approval timestamp and an end time
at most48hours later. Old deadlines unchanged, no automatic clock extension.
No further experiment if failed/mixed/inconclusive; no RO/U or held-out execution.
See AUTHORIZATION_BINDINGS.json and AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json.
''')
    own=['scripts/stage_i_unlikelihood_v2.py','scripts/run_stage_i_unlikelihood_pilot_v2.py',
         'scripts/prepare_stage_i_unlikelihood_v2.py','scripts/audit_stage_i_unlikelihood_v2.py',
         'scripts/compare_stage_i_unlikelihood_v2.py','scripts/package_stage_i_unlikelihood_v2.py',
         'tests/test_stage_i_unlikelihood_v2.py','specs/009q_unlikelihood_sentence_only_amendment_v2.md',
         'thesis_sdd_docs/UNLIKELIHOOD_PILOT_AMENDMENT_V2.md',CONFIG.relative_to(ROOT).as_posix()]
    for rel in own:protected[rel]=sha(ROOT/rel)
    for p in PREP.iterdir():
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'experiment':cfg['experiment'],'files':protected,
        'GPU_execution_approved':False,'previous_v1_manifest_sha256':sha(v1/'PREFLIGHT_MANIFEST.json')})
    bindings={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
              'runner_sha256':sha(RUNNER),'GPU_execution_approved':False}
    write(PREP/'AUTHORIZATION_BINDINGS.json',bindings)
    template=read(v1/'AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json')
    template.update(bindings);template['experiment']=cfg['experiment']
    write(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json',template)
    bundle=ROOT/'outputs/stage_i/unlikelihood_review_bundle_v2';bundle.mkdir(exist_ok=False)
    def cp(p,rel):
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    extras=['scripts/stage_i_unlikelihood_v1.py','scripts/run_stage_i_unlikelihood_pilot_v1.py',
        'tests/test_stage_i_unlikelihood_v1.py','configs/stage_i_unlikelihood_pilot_v1.json',
        'specs/009p_unlikelihood_paired_pilot_v1.md','thesis_sdd_docs/UNLIKELIHOOD_PILOT_PLAN_V1.md',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json',
        'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json',
        benchmark['source_path'],benchmark['effective_generation_config_path']]
    for rel in own+extras:cp(ROOT/rel,rel)
    for folder in (PREP,v1):
        for p in folder.rglob('*'):
            if p.is_file() and 'documentation_before' not in p.parts:cp(p,p.relative_to(ROOT))
    for p in (ROOT/'assets/tokenizers/led-base-16384').iterdir():
        if p.is_file():cp(p,p.relative_to(ROOT))
    old_sources=ROOT/'outputs/stage_i/unlikelihood_review_bundle_v1/evidence/TRAIN512_ORIGINAL_SOURCES_AND_TEACHER_TARGETS.jsonl'
    cp(old_sources,'evidence/TRAIN512_ORIGINAL_SOURCES_AND_TEACHER_TARGETS.jsonl')
    write(bundle/'README_FIRST.md','''# Sentence-only UL v2 — offline review

Read thesis_sdd_docs/UNLIKELIHOOD_PILOT_AMENDMENT_V2.md, spec009q and
outputs/stage_i/unlikelihood_preparation_v2/READINESS_REPORT.md first.
GPU_execution_approved=false. This is ONE minimal offline amendment, no execution.
Use v2 bindings/template only after NEW explicit researcher approval. V1 preserved.

From extracted root in pinned Python environment:
    python -m unittest discover -s tests -p "test_stage_i_unlikelihood*.py" -v

Tokenizer,64tests, source code,512negative masks/selected original positive rows,
24development inputs, approved references and saved D059epoch3 outputs included.
Full pretrained/adapter weights, StageH packs and historical artifacts are not
duplicated: this is a self-contained REVIEW/test package, not a portable GPU run.
Full preflight and independent source-dataset audit require original workspace.
Photo IDs retained, photos not bundled; no new photo verification is claimed.
SHA256_MANIFEST.json covers every payload. No automatic human labels or qualityGO.
''')
    write(bundle/'CHATGPT_REVIEW_REQUEST.md','''# Final independent review of ONE offline amendment

Confirm512sentence/0word negatives and unchanged512IDs/sources/positives/order,
historical F checkpoint, CE control, weight0.1,128updates/twoepochs and decoding.
Review complete negative-mask alignment/gold-conflict checks and actual gradient
contribution instrumentation: scaled/unscaled per-tensor CE/UL, zero/weak signal,
.grad/RNG invariance, no adaptive loss weight. Distinguish preclip objective
gradients from AdamW updates. Probe overhead is bounded within the existing gate.
Review paired comparison vs separate non-causal D059 Expanded-LoRA epoch3 benchmark.
QualityGO must preserve Core/opinions/grounding/visual usefulness/completion while
reducing harmful repetition; exact repetition alone is insufficient. All24cases
are repeatedly inspected development examples. No actual quality result yet.
No GPU authorized; any approval must bind v2 hashes, explicitly amend prior final-
attempt restrictions and set at most48hours for execution AND review. If failed,
no further experiment. D051FAIL and all historical deadlines remain unchanged.
''')
    files={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'SHA256_MANIFEST.json',{'files':files,'payload_files':len(files)})
    archive=ROOT/'outputs/stage_i/stage_i_unlikelihood_preparation_review_v2.zip'
    require(not archive.exists(),'Preserve delivered ZIP')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    receipt={**bindings,'archive':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),
        'bytes':archive.stat().st_size,'payload_files':len(files),'workspace_bindings':len(protected)}
    write(PREP/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
