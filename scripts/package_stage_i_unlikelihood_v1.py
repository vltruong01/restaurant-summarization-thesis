"""Freeze final offline preparation and create a portable REVIEW, not GPU, ZIP."""
import contextlib
import io
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime,timezone
from stage_i_unlikelihood_v1 import *


def main():
    cfg=read(CONFIG)
    # Keep the initial engineering manifest; no user-reviewed artifact is replaced.
    draft=PREP/'engineering_drafts'
    draft.mkdir(exist_ok=False)
    for name in ('PREFLIGHT_MANIFEST.json','AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json'):
        shutil.copy2(PREP/name,draft/name)
    protected=dict(read(ROOT/'outputs/stage_i/d060_preparation_v1/PREFLIGHT_MANIFEST.json')['files'])
    for rel,h in protected.items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Historical file modified: '+rel)
    test_log=(PREP/'OFFLINE_TESTS.log').read_text(encoding='utf-8')
    require('Ran 38 tests' in test_log and test_log.strip().endswith('OK'),'Offline tests did not pass')
    result={'status':'PASS','tests':38,'GPU_operations':0,'model_loads':0,
        'model_forward_backward_generation':0,'synthetic_CPU_autograd_only':True,
        'notes':'Loss and fake-forward callback tests; not actual LED/PEFT/AMP feasibility'}
    write(PREP/'OFFLINE_TEST_RESULT.json',result)
    sampling=read(PREP/'TRAINING_MANIFEST.json')
    report=f'''# Unlikelihood paired pilot: offline readiness

PREPARED — AWAITING NEW EXPLICIT GPU / METHODOLOGY AUTHORIZATION.

38/38 CPU tests PASS. Independent negative binding audit PASS512/512.
Historical426file bindings verified unchanged.512full positive labels roundtrip.
256sentence +256word synthetic negative views,4719penalized tokens,
maximum negative575tokens;0overflow/skipped views. {sampling['zero_caption_businesses']}zero-caption
train businesses retained naturally; none receive visual negatives.
No LED model load/forward/backward/generation, GPU or optimizer update performed.
Only synthetic CPU tensors were differentiated in unit tests. No Teacher/test use.

Both proposed arms start from the same exact historical full-F decoder-only adapter,
24modules/294912parameters. Serialized adapter inventory verified; actual runtime
attachment/gradients/FP16 AMP/memory/restoration remain GPU-gate checks.

Proposed:512train/200validation,2epochs,128updates per arm,warmup7,seed42;
CE control and CE+0.1custom pseudo-repetition UL. Original beam4,max_length768.
Both arms perform negative forwards; validation CE only. Gate3updates each,
fresh reset before training,48final validation predictions total. No search.

Risks: negative-word penalties may suppress legitimate reuse; synthetic prefix
distribution differs from free decoding; exact-copy UL need not improve semantic
redundancy, grounding or Core coverage;512examples may be insufficient. The24cases
are repeatedly inspected development cases, not independent validation. A paired
pilot controls extra-exposure confounding but one seed does not establish broad
generalization. No quality GO is inferred from software PASS.

New approval must explicitly amend prior final-attempt restrictions and provide
a UTC deadline within48hours. Original D060/D059 clocks are not reset by this
preparation. GPU permission remainsfalse; D051strict replayFAIL preserved.

Review UNLIKELIHOOD_PILOT_PLAN_V1.md and spec009p for scientific/stop criteria.
Both repetition and information/grounding must improve or be preserved; there is
no automatic GO from loss, headers, length or repetition alone.
'''
    write(PREP/'READINESS_REPORT.md',report)
    # Bind reusable sources needed by the planned GPU runner; these are historical
    # helpers, never edited by this preparation.
    extras=['scripts/stage_hi_led_data.py','scripts/stage_i_paired_pilot.py',
        'scripts/stage_i_d058.py','scripts/run_stage_i_d050_feasibility_v1.py',
        'scripts/run_stage_i_paired_lora_pilot_v1.py','scripts/stage_i_d053_evaluation.py',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json',
        'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl',
        'outputs/stage_i/d053_final_references_v1/FINAL_RESEARCHER_APPROVED_REFERENCES.json']
    own=['scripts/stage_i_unlikelihood_v1.py','scripts/run_stage_i_unlikelihood_pilot_v1.py',
        'scripts/prepare_stage_i_unlikelihood_v1.py','scripts/audit_stage_i_unlikelihood_v1.py',
        'scripts/package_stage_i_unlikelihood_v1.py','tests/test_stage_i_unlikelihood_v1.py',
        'specs/009p_unlikelihood_paired_pilot_v1.md','thesis_sdd_docs/UNLIKELIHOOD_PILOT_PLAN_V1.md',
        CONFIG.relative_to(ROOT).as_posix()]
    for rel in extras+own:
        protected[rel]=sha(ROOT/rel)
    for p in PREP.iterdir():
        if p.is_file() and p.name not in ('PREFLIGHT_MANIFEST.json','AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json'):
            protected[p.relative_to(ROOT).as_posix()]=sha(p)
    final={'experiment':cfg['experiment'],'files':protected,'GPU_execution_approved':False,
           'initial_engineering_manifest_preserved':'engineering_drafts/PREFLIGHT_MANIFEST.json'}
    # Explicit finalization of new, not-yet-approved engineering bindings only.
    (PREP/'PREFLIGHT_MANIFEST.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    template=read(draft/'AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json')
    template.update(config_sha256=sha(CONFIG),preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'),runner_sha256=sha(RUNNER))
    (PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json').write_text(json.dumps(template,indent=2)+'\n',encoding='utf-8')
    bundle=ROOT/'outputs/stage_i/unlikelihood_review_bundle_v1'
    bundle.mkdir(exist_ok=False)
    def copy_file(p,rel):
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for rel in own+extras:
        copy_file(ROOT/rel,rel)
    for p in PREP.rglob('*'):
        if p.is_file():copy_file(p,Path('preparation')/p.relative_to(PREP))
    # Tokenizer assets make unit tests directly runnable after extraction; no
    # model weights, checkpoint or full scientific dataset is duplicated here.
    for p in (ROOT/'assets/tokenizers/led-base-16384').iterdir():
        if p.is_file():copy_file(p,p.relative_to(ROOT))
    eval_rows=[json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=set(read(PREP/'TRAINING_MANIFEST.json')['business_ids'])
    hm=read(ROOT/cfg['stage_h_manifest'])
    rows=[json.loads(s) for s in (ROOT/hm['datasets']['train_filtered']['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
    write(bundle/'evidence/TRAIN512_ORIGINAL_SOURCES_AND_TEACHER_TARGETS.jsonl',
          '\n'.join(json.dumps(r,ensure_ascii=False) for r in rows if r['business_id'] in selected))
    # Reviews/captions are in original source strings; source photo bytes are not
    # required to judge this objective implementation and are not duplicated.
    write(bundle/'README_FIRST.md', '''# Offline Unlikelihood pilot review

No GPU execution authorized or performed. Read preparation/READINESS_REPORT.md,
thesis_sdd_docs/UNLIKELIHOOD_PILOT_PLAN_V1.md and specs/009p first.
Proposed matched CE/UL pilot, NOT actual quality results. All approvals are pending.

Portable offline test command from extracted root (pinned Python environment):
    python -m unittest discover -s tests -p test_stage_i_unlikelihood_v1.py -v

This is a self-contained scientific/code REVIEW package, not a standalone GPU
execution handoff. Full StageH packs,426historical inputs, pretrained weights and
checkpoint binaries remain in original workspace. GPU preflight deliberately
fails if those are absent. No hashes are rewritten for extraction.
Package includes original selected train source/target rows, fixed24development
inputs, approved references and negative masks, tokenizer, configs, code and tests.
Photo IDs are preserved; photo bytes are not bundled. No photo verification is
claimed by this objective or package. SHA256_MANIFEST.json hashes every payload.
Do not convert the false authorization template into a genuine researcher approval.
''')
    write(bundle/'CHATGPT_REVIEW_REQUEST.md','''# Independent review request

Review the proposed CUSTOM teacher-derived pseudo-repetition UL intervention.
Is the control fair, negative/gold prefix alignment correct, loss numerically safe,
and proposed GPU budget worth testing? Distinguish this from reproducing DITTO.
Check risk of suppressing useful repeated words/names, loss of Core/opinion content,
remaining hallucinations, and development-set reuse. Do not assume actual quality
improvement or infer researcher authorization. Review initial F checkpoint choice,
512sample/twoepoch schedule, alpha0.1 and explicit deadline amendment before approval.
Expected next action is bounded GPU authorization only after researcher review.
No Teacher/test access, architecture changes, RO/U training or extra search.
''')
    files={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'SHA256_MANIFEST.json',{'files':files,'payload_files':len(files)})
    archive=ROOT/'outputs/stage_i/stage_i_unlikelihood_preparation_review_v1.zip'
    require(not archive.exists(),'Do not overwrite delivered archive')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    receipt={'archive':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),'bytes':archive.stat().st_size,
        'payload_files':len(files),'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
        'runner_sha256':sha(RUNNER),'GPU_execution_approved':False}
    write(PREP/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
