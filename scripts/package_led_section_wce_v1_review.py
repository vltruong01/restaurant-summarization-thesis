"""Package already computed offline diagnostics, implementation and CPU evidence."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/stage_i/led_large_section_wce_v1'
OUT=ROOT/'outputs/cloud_handoff/LED_LARGE_SECTION_WCE_V1_REVIEW'
EXEC=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1'
QUALITY=EXEC/'final_evidence/runs/F_large_decoder_policy_v2/quality_review'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists()
    shutil.copyfile(src,dst);assert sha(src)==sha(dst)


def documents():
    d=read(PREP/'DIAGNOSTIC_MEASUREMENTS.json')
    train,val,dev=(d['targets'][x] for x in ('train','validation','fixed_validation_24'))
    e3,e5=(d['generated'][x]['all'] for x in ('epoch3','epoch5'))
    report=f'''# Section-aware Weighted CE V1: offline diagnostic

Status: IMPLEMENTATION PREPARATION ONLY. No new GPU/model training/generation.
Retains ORIGINAL frozen Teacher supervision. Compact/A/V2/V2.1 redesign packages
are preserved but neither consumed nor executed by this objective experiment.

## Confirmed measurements

|Group|businesses|review body tokens|visual body tokens|review share of valid labels|visual share of bodies|
|---|---:|---:|---:|---:|---:|
|Train original F|2000|{train['review']}|{train['visual']}|{train['review_fraction_valid']:.2%}|{train['visual_fraction_bodies']:.2%}|
|Validation original F|200|{val['review']}|{val['visual']}|{val['review_fraction_valid']:.2%}|{val['visual_fraction_bodies']:.2%}|
|Fixed24 target|24|{dev['review']}|{dev['visual']}|{dev['review_fraction_valid']:.2%}|{dev['visual_fraction_bodies']:.2%}|
|Saved epoch3 output|24|{e3['review']}|{e3['visual']}|{e3['review_fraction_valid']:.2%}|{e3['visual_fraction_bodies']:.2%}|
|Saved epoch5 output|24|{e5['review']}|{e5['visual']}|{e5['review_fraction_valid']:.2%}|{e5['visual_fraction_bodies']:.2%}|

Valid labels include both headers/separators and one EOS; bodies exclude them.
Initial tokenizer BOS, generated decoder-start EOS and padding are excluded.
Raw generated IDs (minus decoder start) match tokenized decoded text exactly in
all48. Detailed random20/targeted4 denominators are in DIAGNOSTIC_MEASUREMENTS.json.
All2200F target boundaries are unambiguous after offset verification, and all
content/EOS IDs equal original StageH labels with only the accepted initialBOS
removal. No target text/tokenizer/input artifact was edited.

Zero F captions:9/2000 train (0.45%),1/200 validation (0.50%). Original zero-caption
targets have empty visual bodies. Both saved model epochs generate a visual claim
for the zero-caption Brooklyn Water Bagel case. Exact IDs/counts are in the audit.

Review body shorter than paired target:21/24 at epoch3,23/24 at epoch5. Visual body
longer than target:11/24 and15/24. Mean target visual body66.375tokens, versus
65.833 at epoch3 and71.542 at epoch5. Thus the visual proportion increase reflects
substantial review compression, not simply uniformly larger visual text.
Train visual targets do NOT dominate label count:review already occupies73.49%
of valid tokens. The hypothesis is relative emphasis, not correcting a measured
visual-majority training corpus. No alternative weight sweep is proposed.

## Semantic evidence and limits

Saved references and prior source-grounded observations document repeated Core
omissions across distinct validation businesses: Novella tasting wait/price,
Mission BBQ meat-quality dissent, Bonefish Bang Bang shrimp contrast, JOEY severe
service/steak failures, and Brooklyn coffee/order contrasts. Aquarium shows an
important counter-risk: its visual fish-tank Core can be lost when review dominates.
CORE_OMISSION_EVIDENCE_PROVISIONAL.json preserves exact approved reference unit IDs,
source quotes and identity-aware prior model-assisted notes for24cases. Full48
predictions and24sources/approved D053references/90photos are included unchanged.

These examples and the token-budget pattern support investigating undercoverage.
They do NOT establish exhaustive semantic recall, a population-wide systematic
Core deficit rate or a causal link to ordinary CE. No automated substring match
is treated as semantic coverage; no new researcher-confirmed scores are fabricated.
Existing reference approvals remain valid; output-coverage judgments stay provisional.

## Hypotheses, not findings

Review1.5/visual1.0 may improve useful review coverage, but also rewards generic
review prose and possibly unsupported original Teacher review claims. It does not
identify Core tokens or ground visuals. Relative EOS/header emphasis falls as
review weight grows; zero-caption compliance could worsen. The9rare training cases
suggest limited exposure, but rarity is not proven to cause the hallucination.
Reducing visual emphasis may lose useful visual Core. Quality gate must check all.

Pooled train weight mass assigned to review would rise from73.49% to80.61%; that is
algebraic token mass, NOT measured loss/gradient contribution, training-average
sample mass, semantic improvement or a GPU finding. Actual section CE means and
normalized loss contributions will be logged on the single joint forward/backward;
no separately scaled component-gradient probe is introduced.

## Boundary method

The headers are ordinary multi-token strings, not special IDs. Exact formatting
and each original token sequence are checked. RoBERTa trim_offsets can collapse a
space-only token to(a,a). For offset analysis only, an in-memory tokenizer clone
disables offset trimming; every sequence must retain EXACT original token IDs.
Frozen tokenizer files and actual training labels remain unchanged. Any token
crossing a body/header boundary, duplicate/missing marker or ambiguous alignment
fails closed. Header/separator tokens get1, review body1.5, visual body1, EOS1.
The resulting static roles are keyed by exact original content/EOS IDs; runtime
never infers a mask from approximate header matching. Unknown labels stop.
'''
    put(PREP/'DIAGNOSTIC_FINDINGS.md',report)
    put(PREP/'MATHEMATICAL_LOSS_SPEC.md',r'''# Fixed section-aware weighted cross-entropy

For one physical microbatch, let labels y_bt be the original corrected content/EOS
IDs, logits z_bt the LED output at the SAME position, and I_bt=1[y_bt != -100].
LED itself shifts decoder inputs once; do not shift logits or loss targets again.

ell_bt = -log softmax(z_bt)[y_bt]
w_bt = 1.5 for REVIEW_CONSENSUS body; 1.0 for VISUAL_EVIDENCE body,
       both headers/separators and EOS; 0 for ignored/padded positions.

L_micro = sum_bt(I_bt * w_bt * ell_bt) / sum_bt(I_bt * w_bt)

With n_R,n_V,n_H,n_E valid role counts, denominator =1.5*n_R+n_V+n_H+n_E.
This is a token-position weighting, NOT vocabulary-class weights. All weights=1
reduces to ordinary mean CE over nonignored labels, within tested FP32 tolerance,
with the same gradient. No normalization by padded length or division by a fixed768.
Ignoring PAD means labels=-100 from the trusted collator; source PAD remains1.
EOS stays supervised even for an empty visual body; tokenizer BOS never enters loss.

Gradient at a valid position is (w_bt/sum_valid_weights)*(softmax(z_bt)-onehot(y_bt)).
The relative coefficient for a review versus visual token is1.5 within the same
example; the example normalizer means visual gradients are relatively reduced.
No separate auxiliary CE, label smoothing, per-section normalization or UL loss.
No Core importance labels, new targets, altered sampling, architecture or decoder.

The physical batch remains1 and accumulation8. Preserve baseline Trainer scaling:
L_update = (L_micro_1+...+L_micro_8)/8, and one optimizer/scheduler update after8.
This is NOT a pooled denominator across8examples of different lengths. Changing
to a pooled accumulated-token denominator would add a sample-weighting change.
Trainer model_accepts_loss_kwargs=False remains; num_items_in_batch is not used.
AMP replay recomputes this same objective on the same buffered8microbatches/RNG.

Implementation computes the original model forward with exactly the original
labels/use_cache=False, retaining LED's existing single shift and numerical path.
It releases the unused native CE graph and replaces ONLY output.loss using FP32
unreduced CE from the SAME logits. Native CE is computed once by the model and
again unreduced for weighting: small extra work, no auxiliary backward. One joint
backward/scaler path, unchanged clipping/optimizer/gradient checkpointing. Record
native/weighted forward finiteness and per-section token counts, CE means, weight
sums and normalized loss contributions. Diagnostics are detached, not objectives.

Roles are a new immutable sidecar, not changed labels. Full2200audit verifies exact
original texts/IDs, markers, untrimmed-offset equality, EOS and no padding in raw
labels. Runtime hash lookup checks exact padded labels before applying the mask.
No collator/model signature change, forward hook or generation-time weight.
Validation uses the same weighted objective. Its magnitude is not directly
comparable to saved unweighted validation CE and does not select quality GO.
''')
    put(PREP/'QUALITY_GO_NO_GO.md','''# Scientific gate — one fresh F objective experiment

Current status: NOT_RUN; new research training/outputs=0. No GPU approval.
Original frozen Teacher targets only. Same official backbone revision, fresh
seeded Expanded-LoRA, data/order/optimizer/schedule/precision and beam4 decoding.
5epochs/1250successfulupdates; report both epoch3andepoch5,24fixedvalidationcases
each (20random+4targeted). Compare directly with SAVED unweighted LED-large V2
epochs3/5, without regenerating baseline outputs. Repeated development exposure
and a historical single-run comparator limit causal/generalization conclusions.

Technical acceptance: exact preflight/roles/input bindings; feasible gate and
memory>=15%; finite forward loss and456LoRAgradient records; accepted bounded AMP
with no unapproved fallback; exact state save/restore; epoch1four technical sanity
outputs/noBOScollapse/continuity. Technical PASS is necessary, not scientific GO.

Source-grounded researcher review should be blinded by per-business model labels
where practical, preserving raw text/IDs. Review20random and4targeted separately.
Use unchanged D053approved Core/Supporting units, fullreviews, exactFcaptions and
original photos; Teacher is supervision, not independent gold. Researcher output
annotations stay blank until actually supplied; model assistance is provisional.

GO requires meaningful, researcher-confirmed gain in important Core information,
distinctive dishes/business facts and same-aspect positive/negative synthesis,
without material factual/visual-grounding or overall-usability regression.
No new serious unsupported claims, misleading sentiment or important visual Core
loss. Check zero-caption Brooklyn explicitly: a visual assertion is unsupported,
not evidence of successful avoidance. Require correct zero-caption compliance
before recommending GO. Review semantic repetition/coherence/completeness and
useful concise visuals, not just headers/exact repeats or length.

Predeclared stress questions: Novella wait/service/value, Mission meat dissent,
Bonefish BangBang contrast, JOEY concrete food/service failures, Brooklyn
coffee/order contrast+emptyF, Aquarium fish-tank visual Core, Scarlett visual
invented egg/lemon details and cross-caption relationships. Inspect every other
case for regressions, not only these known problems. Unit coverage and support
must be judged semantically with exact evidence, not substring-hit scores.

Report both epochs and all known errors. Recommend one checkpoint only if
substantive evidence supports it; lower weightedloss, longer review, shorter
visuals or greater Teacher resemblance alone cannot choose a winner. No automatic
numeric win-rate or fabricated researcher labels. Mixed/weak/inconclusive or
substantive regressions => NO-GO recommendation, retain successful unweighted
largeF and complete established matchedRO/U/F/Shuffled; no follow-up search.
Confirmed meaningful improvement => PREPARE matchedRO/U under the same section
policy, subject to separate authorization. RO would keep review1.5/headerEOS1;
do not silently normalize/change its objective. Shuffled reuses matchedF.
Historical LED-base matchedfamily remains primary until a complete accepted
matchedreplacement exists. No automatic training or mixed-objective family.
''')
    put(PREP/'GPU_GATE_RESOURCE_AND_RECOVERY.md','''# Bounded gate, resource estimate and continuity

No GPU executed. Intended GPU: one RTX3090,24GB. Keep successfulV2's exact runtime
pins, model binary, fresh named LoRA initialization, seed/order, batch1/accum8,
AdamWlr2e-4/warmup63/linear/clip1 and FP32storage/FP16AMP unchanged. All prior
scientific artifacts/deadlines remain intact; no Colab/oldFadapter continuation.

Exactly one gate:3successfulupdates, nominal24microbatches; accepted bounded
nativeAMP can replay the SAME accumulation window with restoredRNG, at most8
overflow skips/minimumscale256 and88actualgate microbatches. No extra optimizer
updates or unapproved run retry. Existing initialscale65536 is retained. A
nonfinite forward, missing gradients, exhausted AMP bound, integrity/restoration,
CUDA, memory-query or <15%headroom failure stops without fallback. Inspect all456
trainableLoRAtensors and save existing rawgradient/AMP/memory/state receipts.
Record new SECTION_LOSS_DIAGNOSTICS.jsonl section means/counts/contributions.
Nonzero review CE is not proof of improved Core optimization or generation.

Existing gate3checkpoint roundtrip checks optimizer/scheduler/scaler/RNG/adapters
and frozen backbone;2unchanged train-input stress generations. Discard gate model
states before fresh training. After epoch1, exactly4predeclared raw sanity outputs
and accepted state guard: RNG, mixedmodulemodes, use_cache, adapteractivation/
values, optimizer/scheduler/scaler/AMP counters and gradients must remain exact.
SustainedBOS>=5, blank/special-dominated or state failures stop; isolatedBOS and
cap warnings alone do not. Sanity is technical, not qualityselection.

Then at most1250successfulupdates/5epochs,10064actualmicrobatches under unchanged
AMP bound; checkpoints250/500/750/1000/1250. Final48calls at epochs3/5 plus gate2
and early4 =>54total. No additional inference/search or automaticRO/U.

Checkpoint recovery honesty: baseline runner's CLI is fresh-only; it has NO
automatic resume entrypoint and rejects resume. This preparation preserves that
restriction, not a falsely claimed working resume feature. Save all full-state
checkpoints and receipts. SyntheticCPU state-roundtrip and the real GPU gate
roundtrip check numerical/restoration contracts. If interrupted, preserve logs/
state and stop; any later recovery requires separate researcher scope and a
hash-bound verified checkpoint with no unreported rollback. No numerical failure
is an excuse to resume or reset the schedule.

Measured OLD run: gate184.38s/~5.08GBpeak; training+epoch1continuity14823.82s,
~17.68GBconservativepeak/30.10%headroom; final48outputs610.23s/~4.92GBpeak. These
are historical readings, not a new feasibility PASS. Plan6–8GPUhours including
setup/export; double CE calculation and diagnostic IO can add overhead. Measure
actual per-update time in the bounded gate; do not promise identical throughput.

Sidecar roles are host data, and device weights/roles are only O(batch*labels).
An extra full FP32logit temporary can be up to768*50265*4=154414080bytes (~147MiB)
for batch1, before backend CE workspace/allocator effects. Native CE graph is
released before weighted CE, but exact VRAM overhead remains GPU-unmeasured.
Do not infer fit solely by adding this number to the old peak. Real gate must
measure >=15%headroom. This is a planned single24GBcard, no new rental/autolaunch.

After independent preparation approval, require exact config/code/preflighthashes,
actualUTCauthorization, hardstop<=48hours and researcher absolute stop/reserve
confirmation protecting at least14days for final evaluation/humanstudy/thesis.
No historical deadline is modified/reset. Stop and return evidence at failure,
completion or expiry. Off-instance full-state backups+SHA256 before destruction.
''')
    put(PREP/'GPU_COMMANDS_AFTER_SEPARATE_APPROVAL.md','''# Commands — FUTURE GPU EXECUTION NOT APPROVED

This review ZIP is a VERSIONED OVERLAY, not the 1.84GB pretrained-model/data
transport. Apply only new versioned paths under code_overlay/ to an intact complete
successful DecoderPolicyV2 workspace. Keep all parent files, model/tokenizer/data
and preflight-chain artifacts. Do not substitute an older D059-only workspace.
Historical read-only baseline scripts/configs are included separately for review,
not instructions to overwrite the approved original payload. Recheck overlay SHA.

Use the successful pinned Linux Python3.11 environment (versions in config).
The prepared original workspace is available locally at
outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE; no upload/rental occurs
in this task. Missing assets or version drift must fail; no dependency fallback.

CPU only (allowed after placing the overlay):
```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_stage_i_led_large_section_wce_v1.py preflight
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_led_section_weighted_ce_v1.py -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_led_large_decoder_policy_v2.py -v
```

GPU steps ONLY after explicit researcher approval of EXECUTION_BINDINGS.json.
AUTHORIZATION_TEMPLATE_NOT_APPROVED.json is intentionally false/null. Do not
auto-enable it or borrow prior approval. A researcher-approved new receipt at
SECTION_WCE_RESEARCHER_APPROVAL.json must bind objective/mask/hashes/clocks/scope.
Use separate commands and inspect each receipt before proceeding:

```bash
python -u scripts/run_stage_i_led_large_section_wce_v1.py feasibility --execute --approval SECTION_WCE_RESEARCHER_APPROVAL.json
# Require exactly one new PASS gate, measured headroom>=0.15, no FAIL marker.
python -u scripts/run_stage_i_led_large_section_wce_v1.py train --execute --approval SECTION_WCE_RESEARCHER_APPROVAL.json
# Require1250updates/5epochs, epoch1technicalPASS, all5checkpointreceipts.
python -u scripts/run_stage_i_led_large_section_wce_v1.py evaluate --execute --approval SECTION_WCE_RESEARCHER_APPROVAL.json
# Exactly48newoutputs, no baseline regeneration.
python scripts/report_stage_i_led_large_section_wce_v1.py --output runs/F_large_section_wce_v1
```

Run root is fixed/fresh: runs/F_large_section_wce_v1. No --resume, no overwrite,
automatic launcher/service/retry or alternate weight flag. Keep launcher stdout/
stderr logs and externally enforce the approved hard deadline. Copy all artifacts
and full checkpoints off-instance, verify hashes, then stop for independent review.
New quality bundle includes48new+48savedbaseline summaries, approvedreferences,
sources/photos and96blankoutputannotationrecords; full checkpoints stay separate.
''')
    put(PREP/'VALIDATION_LIMITATIONS.md','''# Verification scope and environment

All computations in this preparation are CPU/file operations. No pretrained
LED-large forward/generation/training or scientific optimizer update occurred.
One tiny synthetic CPU AdamW update is a checkpoint unit-test fixture only.

Initial test invocation found system huggingface-hub1.30.0 incompatible with
Transformers4.51.3;16/18 tests passed and2imports failed, before any research model
operation. No package/pin was upgraded or patched. Tests were rerun with the
existing local .venv-preprocess site-packages first on PYTHONPATH (hub0.36.2),
retaining torch2.1.0+cu118/Transformers4.51.3/tokenizers0.21.1/numpy1.26.4/
peft0.15.2/accelerate1.6.0/safetensors0.8.0 and CUDA_VISIBLE_DEVICES=''. This is
explicit local CPU environment selection, not an approved GPU runtime amendment.
Portable execution still requires exact successful Linux pins/source contracts.

CPU tests establish CE all-ones value/gradient equivalence, weighted normalization,
padding/EOS, all2200masklookup, synthetic finiteness/scaledbackward/accumulation/
state roundtrip and unchanged single decoder shift. CPU bfloat16-autocast fixture
does not prove CUDAFP16behavior. ReusedV2tests check BOS/source masks and fake
inference state continuity. GPU LoRA gradients, trueAMPbackoff, throughput and
VRAM remain unmeasured until the separately authorized gate. No GPU PASS inferred.

The sidecar stores a byte-exact duplicate of corrected labels solely as hash-
lookup keys and new role IDs; original NPZ/views/text/tokenizers are untouched.
No Compact or other target redesign supervision is read/created by this experiment.
No held-out test content or testNPZ is opened. Parent manifests may describe
historical artifacts; tests/ directories are software tests, not held-out data.
''')


def main():
    assert not OUT.exists(),'No overwrite of review delivery'
    for name,n in [('WEIGHTED_CE_TESTS.txt',18),('INHERITED_V2_REGRESSION_TESTS.txt',35)]:
        value=(PREP/'validation'/name).read_text(encoding='utf-8')
        assert f'Ran {n} tests' in value and value.rstrip().endswith('OK'),name
    assert 'CPU preflight PASS' in (PREP/'validation/PINNED_WORKSPACE_CPU_PREFLIGHT.txt').read_text(encoding='utf-8')
    documents()
    # Preserve source-linked provisional notes and exact approvals, not new coverage labels.
    copy(ROOT/'outputs/stage_i/led_large_v2_blinded_quality_review_v1/ADMIN_KEEP_SEPARATE/PROVISIONAL_MODEL_ASSISTED_OBSERVATIONS.json',
         PREP/'CORE_OMISSION_EVIDENCE_PROVISIONAL.json')
    bindings=dict(config_sha256=sha(ROOT/'configs/stage_i_led_large_section_wce_v1.json'),
        runner_sha256=sha(ROOT/'scripts/run_stage_i_led_large_section_wce_v1.py'),
        preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'),
        helper_sha256=sha(ROOT/'scripts/stage_i_led_large_section_wce_v1.py'),
        loss_module_sha256=sha(ROOT/'scripts/led_section_weighted_ce_v1.py'),
        section_mask_sha256=sha(PREP/'SECTION_MASK_VIEWS.npz'),
        report_sha256=sha(ROOT/'scripts/report_stage_i_led_large_section_wce_v1.py'),
        tests_sha256=sha(ROOT/'tests/test_led_section_weighted_ce_v1.py'))
    put(PREP/'EXECUTION_BINDINGS.json',bindings)
    approval=read(ROOT/'outputs/stage_i/led_large_decoder_policy_v2/AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')
    approval.update(experiment='LED_LARGE_SECTION_WCE_V1',section_weighted_ce=read(ROOT/'configs/stage_i_led_large_section_wce_v1.json')['section_weighted_ce'],
                    **{k:bindings[k] for k in ('config_sha256','runner_sha256','preflight_manifest_sha256')})
    put(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',approval)
    put(PREP/'OFFLINE_READINESS.json',dict(status='READY_FOR_INDEPENDENT_REVIEW_GPU_NOT_APPROVED',
        diagnostic_targets=2200,diagnostic_predictions=48,ambiguous_boundaries=0,
        new_unit_tests=18,inherited_regression_tests=35,complete_pinned_workspace_CPU_preflight='PASS',
        frozen_inputs_before_after='PASS',GPU_executed=False,new_research_training_updates=0,
        new_generations=0,Teacher_calls=0,held_out_content_access=False,**bindings))
    put(PREP/'README_FIRST_VI.md','''# Section-aware Weighted Cross-Entropy V1 — chỉ chuẩn bị offline

Đọc DIAGNOSTIC_FINDINGS.md và MATHEMATICAL_LOSS_SPEC.md trước. Giữ ORIGINAL Teacher
targets và decoder-label policy đã sửa BOS. Chỉ đổi trọng số reviewbody1.5,
visual/header/EOS1.0; pad bị loại và chia cho tổng trọng số token hợp lệ.
Không dùng Compact/V2.1 targets, không đổi model/optimizer/decoder.

code_overlay/ chứa các file MỚI để áp vào đầy đủ workspace DecoderPolicyV2; đây
là gói REVIEW, không chứa lại model1.84GB. reference_baseline_code/ chỉ để đối
chiếu. Toàn2200nhãn/role audit và48savedoutputs/24sources/references/90photos có
trong gói. 18testsmới +35regressiontests và completeCPUpreflightPASS.
Các PASS này không phải GPUfeasibility hoặc qualityGO. Xem VALIDATION_LIMITATIONS.

Future GPUcommands và receipt chưa được duyệt nằm trong tài liệu; không tự bật
approval hoặc chạy. Gửi MỘT ZIP này cho independentreview, rồi STOP. Nếu được duyệt
riêng mới gate3updates và conditionalfreshF5epochs/1250updates; không retry/search.
GO cần substantiveCore/balance/grounding/visualquality, không chỉ loss/length.
''')
    overlay_files=[
        'scripts/led_section_weighted_ce_v1.py','scripts/stage_i_led_large_section_wce_v1.py',
        'scripts/run_stage_i_led_large_section_wce_v1.py','scripts/report_stage_i_led_large_section_wce_v1.py',
        'configs/stage_i_led_large_section_wce_v1.json','tests/test_led_section_weighted_ce_v1.py']
    for n in overlay_files:copy(ROOT/n,OUT/'code_overlay'/n)
    for p in sorted(PREP.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            # Duplicate only the actual runtime mask/manifest/predictions under the overlay namespace.
            rel=p.relative_to(PREP)
            copy(p,OUT/'preparation'/rel)
            if rel.as_posix() in ['SECTION_MASK_VIEWS.npz','PREFLIGHT_MANIFEST.json'] or rel.parts[0]=='baseline_predictions':
                copy(p,OUT/'code_overlay/outputs/stage_i/led_large_section_wce_v1'/rel)
    for n in ['scripts/run_stage_i_led_large_decoder_policy_v2.py','scripts/stage_i_led_large_decoder_policy_v2.py',
              'scripts/led_large_decoder_label_view_v1.py','scripts/led_large_decoder_early_sanity_v2.py',
              'scripts/led_large_dynamic_amp_v2.py','configs/stage_i_led_large_decoder_policy_v2.json',
              'tests/test_led_large_decoder_policy_v2.py','tests/test_led_large_decoder_policy_v1.py',
              'outputs/stage_i/led_large_decoder_policy_v1/PINNED_RUNTIME_CONTRACT.json']:
        copy(ROOT/n,OUT/'reference_baseline_code'/n)
    for n in ['SOURCES_AND_TEACHER_TARGETS.json','APPROVED_REFERENCES.json']:
        copy(QUALITY/n,OUT/'saved_quality_evidence'/n)
    for p in sorted((QUALITY/'images').glob('*.jpg')):copy(p,OUT/'saved_quality_evidence/images'/p.name)
    for e in (3,5):copy(QUALITY/f'predictions/epoch{e}.json',OUT/f'saved_quality_evidence/predictions/epoch{e}.json')
    copy(EXEC/'EXECUTION_REPORT.md',OUT/'saved_quality_evidence/BASELINE_EXECUTION_REPORT.md')
    for p in (ROOT/'outputs/stage_i/teacher_target_v2_preparation_v1/stage_a_sources_32/tokenizer').iterdir():
        if p.is_file():copy(p,OUT/'frozen_tokenizer'/p.name)
    copy(ROOT/'scripts/prepare_led_section_wce_v1.py',OUT/'preparation_code/prepare_led_section_wce_v1.py')
    copy(Path(__file__),OUT/'preparation_code/package_led_section_wce_v1_review.py')
    put(OUT/'README_FIRST.md','Start with preparation/README_FIRST_VI.md, DIAGNOSTIC_FINDINGS.md, MATHEMATICAL_LOSS_SPEC.md and EXECUTION_BINDINGS.json. OFFLINE REVIEW ONLY. No GPU approval. code_overlay/ contains new namespaced files; apply only to the complete intact successful baseline workspace.\n')
    put(OUT/'SHA256_MANIFEST.json',dict(files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    archive=ROOT/'outputs/cloud_handoff/LED_LARGE_SECTION_WCE_V1_OFFLINE_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        files=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert set(z.namelist())==set(files)|{'SHA256_MANIFEST.json'}
        for n,h in files.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    dest=Path.home()/'Downloads'/archive.name;copy(archive,dest)
    receipt=dict(archive=str(archive),download=str(dest),bytes=archive.stat().st_size,sha256=sha(archive),
        manifested_files=len(files),CPU_tests=53,full_CPU_preflight='PASS',GPU_approved=False,
        new_training=0,new_generation=0,status='STOP_FOR_INDEPENDENT_REVIEW',**bindings)
    put(PREP/'DELIVERY_RECEIPT.json',receipt)
    for p in (archive,dest):put(p.with_suffix(p.suffix+'.sha256'),sha(archive)+'  '+p.name+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
