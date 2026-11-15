"""Offline documents, same32 sources, schemas and ID-only future batch layout."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/teacher_target_v2_preparation_v1'
PREP = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_v1/bundle'
AUDIT = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_result_review_v1'
ADJ = ROOT / 'outputs/stage_i/teacher_redesign_adjudication12_v1/bundle'
EXEC = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2_execution_v1'
REQUEST = Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/26f6cae6-ffe9-41a6-be71-2b8403fe20e1/Pasted text.txt')
POLICY = 'TEACHER_TARGET_V2_EVIDENCE_DENSE_V1'
BASELINE_BINDINGS = {
    'configs/stage_i_led_large_decoder_policy_v2.json': 'd079f41aa5ae5b1a680da5174eb3ef83235cb9b82e54c96b3d5d487215a168f4',
    'scripts/run_stage_i_led_large_decoder_policy_v2.py': 'bea436309fadb632a04db82549ad63139f31bd77365e95271b77c1ff68470a61',
    'outputs/stage_i/led_large_decoder_policy_v2/PREFLIGHT_MANIFEST.json': '782de3373434ff01b934818cd143e0d6c0be32fc0658e924c19ba2711401a3d3',
    'scripts/stage_i_led_large_decoder_policy_v2.py': 'a37f7ee37047df01051d9e13cd45ab4ee0d189f3167f2b4167d384ae581a4777',
    'scripts/led_large_decoder_early_sanity_v2.py': '10975bf25d559c248320e7070ea45387bce6cf2f8dde48dc02f7974e64b39306',
    'scripts/led_large_decoder_label_view_v1.py': '413a408ed384bdd5d40ef21ea7c99891a2ee42e88554a1f6e543d1ad6f535b96',
}


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)


POLICY_TEXT = '''# Teacher Target V2 policy

Policy ID: TEACHER_TARGET_V2_EVIDENCE_DENSE_V1. Preparation only; no V2 targets exist yet.
This is the single final authorized redesign direction, not an open policy search.

One example is one business. Read every frozen selected review in its retained
order. Identify recurring/distinctive information and important disagreements
before compressing. Preserve named dishes, distinctive services, prices/value,
order/access problems and meaningful usage constraints when substantively relevant.
Generic praise does not replace a recurring signature or concrete failure.
Preserve disagreement about the SAME dish/aspect explicitly when supported;
an unrelated negative sentence does not balance a positive named-dish assertion.
Do not force equal sentiment if the source is asymmetric. Do not promote isolated
opinions to consensus or count near-duplicate accounts as independent support.

Serious isolated reports require judgment, not automatic inclusion/exclusion.
Distinguish firsthand experience, secondhand hearsay, allegation and verified fact.
Use explicit attribution/uncertainty, and ask whether the report's importance
justifies space relative to recurring business information. Attribution alone does
not justify making a rumor central. Avoid unsupported current-status or safety
claims from dated reviews. Do not add process commentary to the restaurant summary.

Each proposed review claim cites literal quotes and exact review evidence IDs.
Claim metadata records evidence characterization and salience rationale separately
from the target. Metadata is provisional, not an approved Core label. Important
single-source information is allowed with honest attribution; never invent support.

Generate ONE canonical REVIEW_CONSENSUS from reviews only. Its body is the exact
single-space join of review_claims.text and is copied identically into RO/U/F.
RO: '[REVIEW_CONSENSUS]' + newline + canonical body, no visual assertions/header.
U/F: identical RO text + two newlines + '[VISUAL_EVIDENCE]' + newline + that
condition's visual body. Visual body is the single-space join of visual_claims.text.
Do not silently normalize or repair accepted wording/whitespace.

U visual claims use exact U captions; F uses exact F captions. Cite the global
condition-specific evidence ID, literal model_visible_caption quote and photo_id.
Local IMG_CAP_i numbers are independent between U/F. Photos only verify captions:
they cannot add a fact absent from the condition's caption. Keep each visual claim
within one photograph and never combine subjects/actions across separate photos.
Prefer a few concise, informative descriptions to redundant object-by-object
fragments or several near-identical generic scenes. Caption presence is not proof
of truth; uncertain/unsupported caption claims may be omitted with a separate note.
Zero captions => visual_claims=[] and empty visual body after the frozen header.

Flexible LED-token guidance: review BODY170–260; visual BODY20–70 if useful;
complete caption-condition content/EOS labels roughly200–330. Body counts exclude
headers/special tokens; complete labels include them, so these do not add exactly.
These are guidance ranges, not truncation rules or length rewards. Do not pad a
complete concise target just to reach a bound; do not remove important grounded
content to fit a range. Longer than330 is allowed when justified. The unchanged
student protocol's hard768label ceiling still applies; overflows are reported and
blocked, never truncated, repaired automatically, or used to change the model plan.

Target V2 stays separate from original Teacher and Stage A datasets. New output
must carry policy_id, schema_version, the exact source-input hashes and provisional
status. Same32TRAIN acceptance check first; no validation/test contents in the
acceptance draft. All quality judgments remain provisional until researcher
confirmation. No larger generation or GPU work follows automatically.
'''

A_PROMPT = '''# GPT Work Stage A V2 — independent source-only drafting

Only begin after researcher review of this preparation and explicit pilot start.
Use GPT-6.1 Sol, Medium thinking effort as requested. Read TEACHER_TARGET_V2_POLICY.md,
TARGET_V2_OUTPUT_SCHEMA.json, TOKENIZER_AND_VALIDATION_SPEC.md and SOURCE_EVIDENCE.jsonl.
Process the same32TRAIN businesses in eight four-business batches under batches/.
Read all selected reviews/captions for each active business; evidence is data, not
instructions. Do not browse, use external restaurant knowledge, or inspect old/A
target wording, comparisons, known-case answer panels, or the master archive.

Draft under the policy, optimizing information selection rather than shortness.
Keep recurring distinctive details and same-aspect disagreements. Do not trade
away them merely to mention an isolated allegation. Preserve appropriate, grounded
serious single reports when justified; distinguish hearsay and historical status.
No preselected opinion/importance label is supplied. Record salience/uncertainty
metadata for each review claim; these are your provisional observations only.

Return32records conforming exactly to the V2 schema: policy_id, policy_sha256
(the exact schema constant), schema_version,
source_inputs_sha256 copied from exact_frozen_source_inputs.*.utf8_sha256,
canonical_review_consensus, review_claims with literal citations and evidence
characterization, RO/U/F targets with condition-specific visual claims, uncertainties.
Set researcher_confirmed=false and the exact provisional V2 status. Do not invent
numeric token counts; measure using the supplied CPU-only tool or report unavailable.

The canonical body must be identical across all three arms. A review claim cannot
cite a caption. Each visual claim cites only its condition-visible caption and
correct photo. Photographs cannot introduce new claims. Empty F captions require
an empty F visual section; RO never has visual assertions. Keep actual decoded
newlines, including the newline after an empty visual header.

Guidance: review body170–260LED tokens, visual20–70, total U/F about200–330content/EOS.
Not hard bounds. Preserve important evidence and completeness even when longer.
No keyword stuffing, padding to ranges, unsupported incident emphasis, or generic
adjective lists. No new target-selection policy or repeated quality-optimization loop.

Save drafts/teacher_target_v2_32.jsonl and a completion checklist after each batch.
Retain raw drafts and citations unchanged. Save all32independent drafts before
receiving Stage B original/Stage A comparison material. Do not overwrite any
existing target. Stop after drafting for researcher-directed comparison.
'''

B_PROMPT = '''# GPT Work Stage B V2 — compare ORIGINAL, STAGE A and V2

Begin only after all32V2source-only drafts have been independently saved. Keep
those bytes unchanged. Now receive the comparison material from the master package.
Use the complete same source evidence, condition-visible captions and photos.
Compare all three candidates per business and condition. Neither the original
Teacher nor the previous Stage A draft is independent gold truth.

Use KNOWN_CASE_ACCEPTANCE_CHECKS.json as questions requiring source adjudication,
not a forced word list or already approved Core labels. Explicitly test the eight
loss/salience concerns and four potential-benefit cases. Check all remaining20 too;
good performance on12alone is insufficient. Keep random24 and targeted2 strata
separate. Shared canonical review means96arms are not96independent observations.

Evaluate important Core retention, recurring/distinctive facts, same-aspect opinion
disagreements, factual support, attribution/salience of serious isolated or hearsay
reports, useful caption-conditioned visuals, cross-photo errors, redundancy and
completeness. Cite exact output/source spans and stable evidence IDs. Importance
panels are provisional; don't fabricate researcher-confirmed Core or preferences.
Identify facts the original or A retains better as well as gains in V2. Prefer
tie/inconclusive if an important tradeoff is unresolved. Do not force a V2 win.

Use the pinned local tokenizer utility for exact counts. Original/A counts are
provided; V2 counts do not exist before drafting. If unavailable, report that and
leave V2 counts missing for local measurement; do not estimate. Length decrease,
correct headers, literal citations or fewer exact repetitions alone are not GO.

Return evidence-linked comparisons for32businesses/96conditions with three-way
preferences (ORIGINAL/STAGE_A/V2/TIE/INCONCLUSIVE), missing important facts on each
side, unsupported/weak claims, uncertainty and exact token results. Do not revise
drafts during comparison. Disclose if you authored the draft: self-review is not
independent adjudication. Researcher confirmation of QUALITY_GATE_V2.md is required
before any production preparation becomes executable. No training or corpus generation.
Stop for researcher review, without selecting a scientific outcome on their behalf.
'''

GATE_TEXT = '''# Predeclared V2 acceptance gate — frozen before drafts exist

Status: NOT_RUN. Researcher quality confirmation: PENDING. No V2 generation performed.

Technical gate (all32businesses/96targets): exact ID/split/source-policy bindings;
schema; no missing/duplicate condition; identical canonical body across RO/U/F;
literal evidence IDs/quotes/photo mappings; correct condition-specific captions;
RO no visual; zero-caption visual empty; no source edits or silent truncation;
all actual labels <=unchanged768 ceiling. All sources are the exact32TRAIN cases.
Guidance-range departures only warn; important complete content outranks a range.
Mechanical citation checks cannot certify entailment or absence of hallucination.

Source-quality gate, requiring explicit independent researcher adjudication:
- Review all32against ORIGINAL and A, with evidence-linked importance/coverage.
- Resolve all8known loss/salience concerns; do not merely insert named keywords.
  If excluding a serious single report is appropriate, explain that source-based
  choice; including every one-off is not required. Preserve meaningful same-aspect
  disagreement and recurring business signatures/constraints.
- Check the4promising cases retain useful benefits without new unsupported
  policy/safety/current-status claims. Check the other20 for new regressions.
- No unresolved MAJOR important-information regression, unsupported material claim,
  cross-photo relationship, empty-caption violation, or misleading consensus may
  remain. Minor Supporting tradeoffs require an explicit researcher rationale.
- Researcher must find a substantive benefit in Core retention, balance, support
  and usable information density against BOTH prior versions, with no unacceptable
  regressions. No automatic win-rate or weighted-score threshold substitutes for
  this judgment. Ties/inconclusive do not establish meaningful improvement.

Record unresolved items rather than inventing approval. Core/Supporting judgments
for theseTRAIN businesses are not D053 human labels. Prior B was unblinded
self-assessment; the researcher reports an independent review/adjudication in the
new decision message, but no filled detailed adjudication table was supplied.
Do not materialize grades or signatures from that general statement.

If gate passes: freeze policy/prompt/schema/source and accepted V2 draft hashes;
then seek explicit production-generation authorization. If mixed/inconclusive/fail:
stop and return evidence. No automatic third target policy or repeated redesign.
Passing Teacher quality does NOT authorize GPU or imply improved student quality.
All historical deadlines remain unchanged; future execution needs a separate scope.
'''

TOKEN_SPEC = '''# Exact tokenizer / validation specification

Use the frozen LED tokenizer allenai/led-base-16384 revision
38335783885b338d93791936c54bb4be46bebed9, identical IDs as the accepted large setup.
No tokenizer/model download. Assets and SHA256 are under tokenizer/ with binding.
Pinned CPU dependencies: tokenizers==0.21.1, jsonschema==4.23.0 (Draft2020-12).
The unchanged counting/citation helper plus its9symbolic CPU regressions are reused.
Six additional V2 boundary tests cover policy/source bindings, false researcher
confirmation, TRAIN-only scope, flexible guidance and the hard768label limit.
The V2 acceptance wrapper adds the version/policy/source-binding schema and full
32-record membership. It is explicitly TRAIN32-only, not a production runner.

From extracted source-only package after actual V2 drafts exist:
python tools/validate_teacher_target_v2_acceptance_v1.py --drafts drafts/teacher_target_v2_32.jsonl --sources SOURCE_EVIDENCE.jsonl --schema TARGET_V2_OUTPUT_SCHEMA.json --tokenizer tokenizer --output drafts/V2_MECHANICAL_AND_COUNTS.json

It writes a new report, refuses overwrite, and never changes inputs or returns
quality GO. No V2 counts can be reported before actual targets exist. Literal
citation/one-photo checks do not prove semantics, recurrence or importance.

Measured fields: full original-tokenizer BOS/content/EOS; content without specials;
corrected content/EOS labels; standalone review/visual body and section token counts.
Bodies exclude markers/specials, full labels include marker/whitespace boundaries;
do not add body counts to infer complete length. No padding/truncation in counting.

Original Stage H BOS/content/EOS labels remain unchanged. Any FUTURE V2 dataset
uses exactly initial-BOS removal with content and finalEOS intact. Model shift
prepends decoder-startEOS2 once. Input IDs/masks remain unchanged. BOS0/PAD1/EOS2;
markers remain ordinary BPE content. A hard >768label exception stops data readiness;
do not trim targets, extend decoder settings, or silently change training protocol.

Production later needs exact2200business/6600condition completeness, source hashes
and order matched to original StageH, canonical equality, all-target token checks,
append-only attempt/technical-validation receipts and researcher corpus sign-off.
Do not pretend this small acceptance validator implements the entire gated workflow.
'''

PRODUCTION_PLAN = '''# Conditional production generation plan — 2200businesses / 6600targets

NOT_EXECUTABLE / NOT_AUTHORIZED. The32V2quality gate must pass and production scope
must be explicitly authorized first. Today exports ID-only layout, schemas and
workflow contracts, not full production sources, targets, attempts or API runners.

Frozen population:2000train +200validation;100held-out IDs excluded, no held-out
source content. Metadata-only batch layout has550four-business batches:
train first8batches are the32pilot IDs in original order; remaining1968train IDs
sorted form492batches;200validation IDs sorted form50batches. Generation order
does not change the frozen downstream training order. No sources are selected by
checkpoint outputs. Source manifests must be materialized/validated only after GO.

After authorization, reuse the exact source exporter and accepted source template
to provide complete reviews/U/Fcaptions, photo mappings and input hashes per batch.
Do not change review selection, captions, split or tokenizer. Source-only drafting
uses the one frozen V2 policy/prompt, one canonical review per business and U/F
visuals; no original/A target wording in generation context. Keep batches in fresh
contexts or source-only work sessions with policy/schema reapplied; no policy drift.
No API-generation automation is provided or authorized here; workflow is GPT Work.

Reuse the32pilot outputs ONLY if researcher accepts them unchanged under exactly
the frozen production V2policy/prompt/schema/source hashes. Then96condition records
are reused and2168businesses/6504condition records remain. Otherwise production
stays blocked pending researcher decision; do not silently regenerate accepted cases.

Output/ledger contracts (versioned new teacher_target_v2 directory only):
- raw_attempts/<batch_id>/attempt01... : original returned bytes, prompt/source
  manifest hashes, operator/time, parse/validation diagnostics. Never overwrite.
- progress.jsonl append-only: PLANNED -> RECEIVED -> TECHNICALLY_VALID or INCOMPLETE
  or INVALID; quality confirmation is a separate state, never inferred from syntax.
- technically_valid_businesses.jsonl: one immutable record per business, with
  target text/citation hashes. Repeated identical bytes are idempotent; conflicting
  duplicate wording is a hard conflict, not an opportunity to choose the better draft.
- final frozen corpus after researcher sign-off:2200canonical business records,
  six train/validation xRO/U/F JSONLs (2000or200rows each),6600total target records,
  policy/tokenizer/source/output manifests and exact lengths. Flattening adds
  provenance only and cannot rewrite text. Original Teacher dataset is never overwritten.
- new packed label views in a separate directory. Copy original source arrays
  exactly and verify source IDs/masks/order; encode V2content/EOS under unchanged
  label768protocol. No original StageH or corrected historical view is edited.

Resume: recompute receipt/hash inventory, skip only complete technically valid
unchanged businesses. Maximum2additional technical retries for genuinely incomplete
or malformed business records, same frozen instructions/full source; retain every
attempt. Do not retry already valid records for better style/quality, rewrite
accepted content, change policy or search seeds. After the cap/conflict, stop and
report. Semantic quality failure is not a technical retry excuse.

Corpus readiness: all2200IDs,6600conditions, no extras/duplicates; canonical same
across each triple; condition-scoped captions; all zero-caption conditions empty;
literal source/caption links; per-target token counts/768limit; exact original
input IDs/order; all hashes and raw attempt lineage. Quality audit plan must be
frozen before production starts and researcher-approved: review the32pilot anchors,
all detected technical/source-grounding concerns, and deterministic50new train
plus10validation source-linked cases (ID ranks seed20261022|V2_CORPUS_AUDIT|ID,
excluding pilot32 for train). Do not claim the whole corpus is human-verified
from this sample. Any material concern blocks corpus sign-off, not a silent rewrite.
Record added validation exposure. No student training until complete validated
corpus, corpus SHA256 freeze and separate fresh-F GPU authorization.
'''

TRAIN_PLAN = '''# Conditional fresh LED-large F comparison and resource estimate

Preparation only. No new corpus exists and no GPU execution or rental is authorized.
After a complete validated V2corpus and separate approval, prepare exactly ONE
fresh F student, not a continuation of the historical F adapter. Use the successful
Decoder Policy V2 engineering protocol, with only supervision target/version and
the new run/output/hash namespace changed. A new versioned runner/preflight must
bind the new label corpus; the historical runner/hashes cannot be patched or
treated as approval for new data. No ready-to-run GPU script is provided today.

Unchanged: official allenai/led-large-16384 revision
cd59d11c3528415b7dda4dfc95cc8f138aceda2e;459801600backbone parameters;
fresh named seeded KaimingA/zeroB Expanded-LoRA all-linear excluding tied lm_head,
228modules/456trainable tensors/4915200parameters,r8/alpha16/dropout0;
same seed42, fixed2000/200businesses and source order/reviews/captions/input IDs;
batch1/accumulation8; AdamW lr0.0002,decay0.01,betas0.9/0.999,eps1e-8;
5epochs/1250successful updates,linear schedule/warmup63/clip1;
same FP32 parameter storage,FP16AMP,init65536 and bounded native backoff,
same checkpointing and corrected content/EOS labels; beam4/max_length768 settings.
Exact parent config/bindings and measured run evidence are provided; they govern
details rather than abbreviated prose. No coefficient/architecture/decoder search.

Future gate:3successful updates, nominal24microbatches, existing bounded overflow
maximum88actual gate microbatches,2stress-generation calls; require>=15%measured
headroom, finite/missing/zero gradient checks, adapter/backbone integrity and exact
save/restore. Discard feasibility states before fresh training. After epoch1,
exactly4predeclared technical sanity outputs under the accepted BOS-loop>=5,
blank/special-dominance and continuity checks; isolated BOS/cap warnings alone
do not fail. Preserve all5epoch checkpoints. Final generation:24fixed development
cases at epochs3and5 =>48outputs; total generation-call ceiling54including gate
and early sanity. No extra validation decoding or old-model regeneration.

Compare saved old LED-large F V2epoch3/5 against new F epoch3/5 on the SAME24cases
and existing D053approved references/source evidence. The Teacher targets are
supervision, never independent gold. Prioritize Core retention, opinion balance,
factual/visual grounding, useful visuals, semantic repetition, zero-caption and
usability. Substantive researcher-confirmed improvement is required; technical
success or loss reduction alone is insufficient. Raw loss across target versions
has different text/token denominators, so cannot isolate quality. Record repeated
development exposure; model-assisted judgments remain provisional. D059base is a
practical unmatched benchmark, not causal evidence for the target intervention.

If new F is weak/mixed/inconclusive or fails substantive improvement, STOP this
Teacher Target V2 direction. Preserve results and use existing baselines; no new
redesign, fallback tuning or automatic RO/U. If confirmed improved, PROPOSE matched
V2RO/U with identical initialization,optimization,epoch selection and policy.
Shuffled reuses matchedF. Never mix Teacher versions for primary filtering claims.
Only historical LED-base has a complete established matched family currently;
do not imply the existing largeFalone is matchedRO/U/F. Keep all baselines intact.

Measured RTX3090 evidence (old successful run, not a new benchmark):
- GPU total25298141184bytes (~23.56GiB marketed24GB).
- Gate184.38s,peak conservative5.08GB,79.93%headroom.
- Training including early guard14823.82s (~4h07m),peak conservative17.68GB
  (~16.47GiB),30.10%headroom; this includes continuity snapshots/allocator retention,
  NOT pure training-only required memory.
- Final48generation610.23s (~10.2m),peak4.92GB,80.55%headroom.
Known model phases totaled about4h20m. Planning allowance:6–8GPU-hours for one F
run including overhead/export, conditional on gate and available host. This is an
estimate from measured local evidence, not a speed/memory guarantee for new labels
or a current rental-price quote. Provision a single24GB RTX3090-class instance,
at least32GBhost RAM and50GBdisk as planning allocations, then verify actual free
resources. No claim that these RAM/disk allocations were measured minima.
Do not rent or execute from this plan. Future authorization must set its actual
clock/stop and preserve the final evaluation/defense reserve; no old deadline is
modified or reset here. Back up full-state checkpoints/results off-instance and
verify SHA256 before any later resource destruction.
'''


def main():
    assert not OUT.exists(), 'Do not overwrite an existing preparation version'
    source_path = PREP/'SOURCE_EVIDENCE.jsonl'
    source_rows = jl(source_path)
    pilot_ids = [r['business_id'] for r in source_rows]
    assert len(pilot_ids)==len(set(pilot_ids))==32 and all(r['split']=='train' for r in source_rows)
    for rel,h in BASELINE_BINDINGS.items():
        assert sha(ROOT/rel)==h, ('Historical baseline binding differs',rel)
    original_files = [source_path, PREP/'PILOT_SAMPLE_MANIFEST.json',
        AUDIT/'stage_b_verified/locked_inputs/ORIGINAL_TARGETS_UNCHANGED.jsonl',
        AUDIT/'stage_b_verified/locked_inputs/proposed_targets_STAGE_A_UNCHANGED.jsonl',
        ADJ/'RESEARCHER_ADJUDICATION_TABLE.json', REQUEST]
    input_hashes = {str(p):sha(p) for p in original_files}
    OUT.mkdir(parents=True)
    bundle=OUT/'bundle';bundle.mkdir()
    stage=OUT/'stage_a_sources_32';stage.mkdir()
    documents={
        'TEACHER_TARGET_V2_POLICY.md':POLICY_TEXT,
        'GPT_WORK_STAGE_A_V2_PROMPT.md':A_PROMPT,
        'GPT_WORK_STAGE_B_V2_COMPARISON_PROMPT.md':B_PROMPT,
        'QUALITY_GATE_V2.md':GATE_TEXT,
        'TOKENIZER_AND_VALIDATION_SPEC.md':TOKEN_SPEC,
        'PRODUCTION_GENERATION_PLAN_6600.md':PRODUCTION_PLAN,
        'RETRAINING_COMPARISON_AND_GPU_ESTIMATE.md':TRAIN_PLAN,
    }
    for name,text in documents.items():
        (bundle/name).write_text(text,encoding='utf-8')
    policy_hash=sha(bundle/'TEACHER_TARGET_V2_POLICY.md')
    schema=read(PREP/'TARGET_OUTPUT_SCHEMA.json')
    schema['title']='Teacher Target V2: one provisional business record'
    extra={
        'schema_version':{'const':2}, 'policy_id':{'const':POLICY},
        'policy_sha256':{'const':policy_hash},
        'source_inputs_sha256':{'type':'object','additionalProperties':False,
            'required':['reviews_only','unfiltered','filtered'],
            'properties':{c:{'type':'string','pattern':'^[a-f0-9]{64}$'} for c in ('reviews_only','unfiltered','filtered')}},
    }
    schema['required']+=list(extra)
    schema['properties'].update(extra)
    schema['properties']['status']={'const':'PROVISIONAL_GPT_WORK_V2_REQUIRES_RESEARCHER_REVIEW'}
    claim=schema['properties']['review_claims']['items']
    claim['required'].append('evidence_characterization')
    claim['properties']['evidence_characterization']={
        'type':'object','additionalProperties':False,'required':['basis','salience_reason','uncertainty_note'],
        'properties':{'basis':{'enum':['recurring','same_aspect_disagreement','isolated_firsthand_report','secondhand_or_allegation','mixed_or_uncertain']},
          'salience_reason':{'type':'string','minLength':1},'uncertainty_note':{'type':'string'}}}
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(schema)
    save(bundle/'TARGET_V2_OUTPUT_SCHEMA.json',schema)
    for name in ('TEACHER_TARGET_V2_POLICY.md','GPT_WORK_STAGE_A_V2_PROMPT.md','TOKENIZER_AND_VALIDATION_SPEC.md','TARGET_V2_OUTPUT_SCHEMA.json'):
        cp(bundle/name,stage/name)
    for name in ('SOURCE_EVIDENCE.jsonl','PILOT_SAMPLE_MANIFEST.json','CAPTION_PHOTO_MAPPING.json'):
        cp(PREP/name,stage/name)
    for folder in ('photos','frozen_inputs','batches','tokenizer','tools'):
        for p in (PREP/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                cp(p,stage/p.relative_to(PREP))
    cp(ROOT/'scripts/validate_teacher_target_v2_acceptance_v1.py',stage/'tools/validate_teacher_target_v2_acceptance_v1.py')
    cp(ROOT/'scripts/test_teacher_target_v2_acceptance_v1.py',stage/'tools/test_teacher_target_v2_acceptance_v1.py')
    (stage/'tools/requirements-v2-validation.txt').write_text('tokenizers==0.21.1\njsonschema==4.23.0\n',encoding='utf-8')
    (stage/'README_FIRST.md').write_text('''# Source-only V2 acceptance check — not yet started

Exactly the same32TRAIN cases; no resampling and no old/A target wording.
Read the V2policy, schema, tokenizer spec and Stage A V2prompt. Start ONLY after
researcher review/explicit start. Process8batches of4, save all32V2drafts before
seeing originals or A. No old110–170token target guide or prior drafting prompt
is supplied. Extra historical source references are provenance, not new evidence.
No targets or V2token counts exist in this package. No external knowledge/API,
model training, held-out access or scientific decision is authorized by the export.
''',encoding='utf-8')
    # Prior targets/comparisons stay outside the source-only drafting archive.
    comparison=bundle/'stage_b_comparison_32';comparison.mkdir()
    for name,p in {
        'ORIGINAL_TARGETS_UNCHANGED.jsonl':original_files[2],
        'STAGE_A_TARGETS_UNCHANGED.jsonl':original_files[3],
        'EXACT_LOCAL_ORIGINAL_A_TOKEN_COUNTS_96.json':AUDIT/'LOCAL_EXACT_LED_TOKEN_COUNTS_96.json',
        'PRIOR_STAGE_B_REPORT_UNCHANGED.md':AUDIT/'stage_b_verified/STAGE_B_REPORT_VI.md',
        'PRIOR_STAGE_B_OBSERVATIONS_32_UNCONFIRMED.json':AUDIT/'stage_b_verified/BUSINESS_COMPARISONS_32.json',
        'PRIOR_12_ADJUDICATION_TABLE_UNFILLED.json':ADJ/'RESEARCHER_ADJUDICATION_TABLE.json',
    }.items():
        cp(p,comparison/name)
    priority=read(ADJ/'PRIORITY_CASE_MANIFEST.json')['selected12']
    checks=[]
    for c in priority:
        data=read(ADJ/f'case_data/{c["case_id"]}.json')
        checks.append(dict(business_id=c['business_id'],business_name=c['business_name'],
            concern_or_potential_gain=data['researcher_requested_focus'],
            source_evidence_ids_to_inspect=data['relevant_review_ids'],
            test_basis='source-linked research question, not approved importance label or keyword requirement',
            researcher_result=None,researcher_confirmed=False))
    save(bundle/'KNOWN_CASE_ACCEPTANCE_CHECKS.json',checks)
    save(bundle/'QUALITY_GATE_RESEARCHER_FORM.json',dict(status='NOT_RUN_NO_V2_TARGETS',
        reviewer_id=None,reviewed_at=None,policy_sha256=policy_hash,
        known_case_checks=checks,all32_comparisons_completed=False,
        no_unresolved_major_Core_regressions=None,no_unsupported_material_claims=None,
        source_balance_and_visual_quality_acceptable=None,
        substantive_improvement_against_original_and_A=None,
        decision=None,decision_rationale=None,researcher_confirmed=False,
        production_generation_authorized=False,GPU_authorized=False))
    form_path=bundle/'QUALITY_GATE_RESEARCHER_FORM.json'
    form=read(form_path)
    form['comparison_rows_32']=[dict(business_id=r['business_id'],selection_category=r['selection_category'],
        important_information_reviewed=None,opinion_balance_reviewed=None,grounding_reviewed=None,
        visual_usefulness_reviewed=None,major_regressions_resolved=None,evidence_ids=[],
        original_A_V2_preference=None,researcher_notes=None,researcher_confirmed=False) for r in source_rows]
    form_path.write_text(json.dumps(form,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # ID-only conditional production layout, never loading validation/test source contents.
    split_path=ROOT/'data/processed/yelp_2300_v1/business_split.csv'
    assert sha(split_path)=='f3331ce71da531099349ef680e18ab38255f38db8752dcf61e8998c7a191d93b'
    with split_path.open(encoding='utf-8',newline='') as f:
        splits={r['business_id']:r['split'] for r in csv.DictReader(f)}
    assert Counter(splits.values())=={'train':2000,'validation':200,'test':100}
    train=pilot_ids+sorted(b for b,s in splits.items() if s=='train' and b not in pilot_ids)
    val=sorted(b for b,s in splits.items() if s=='validation')
    batches=[]
    for split,ids in [('train',train),('validation',val)]:
        for n in range(0,len(ids),4):
            batches.append(dict(batch_id=f'{split}_B{n//4+1:04d}',split=split,business_ids=ids[n:n+4],
                expected_business_records=4,expected_target_conditions=12,
                state='PLANNED_METADATA_ONLY_NOT_AUTHORIZED',source_manifest='PENDING_POST_GATE',
                pilot_batch_reusable_if_frozen=split=='train' and n<32))
    assert len(batches)==550 and len({x for b in batches for x in b['business_ids']})==2200
    assert not any(splits[x]=='test' for b in batches for x in b['business_ids'])
    save(bundle/'production_plan/DETERMINISTIC_BATCH_LAYOUT_IDS_ONLY.json',dict(seed=20261022,
        population=dict(train=2000,validation=200),source_contents_exported=False,batches=batches))
    save(bundle/'production_plan/RESUME_LEDGER_TEMPLATE.json',dict(records=[],
        allowed_states=['PLANNED','RECEIVED','INCOMPLETE','INVALID','TECHNICALLY_VALID','RESEARCHER_CORPUS_FROZEN'],
        maximum_additional_technical_retries=2,accepted_records_immutable=True,
        conflicting_duplicates='STOP_NO_QUALITY_BASED_RETRY',policy_optimization=False))
    save(bundle/'production_plan/SOURCE_AND_OUTPUT_MANIFEST_CONTRACT.json',dict(
        source_fields=['business_id','split','ordered_review_ids','U_and_F_caption_photo_order',
          'exact_source_utf8_sha256_per_condition','original_StageH_input_ids_sha256','policy_prompt_schema_tokenizer_SHA256'],
        output_fields=['business_id','split','condition','canonical_review_sha256','target_utf8_sha256',
          'exact_BOS_content_EOS_and_corrected_content_EOS_lengths','raw_attempt_SHA256','citation_validation_receipt'],
        final_business_records=2200,final_condition_records=6600,
        materialization_state='PENDING_QUALITY_GATE_AND_SEPARATE_AUTHORIZATION'))
    cp(REQUEST,bundle/'provenance/RESEARCHER_DECISION_VERBATIM.txt')
    cp(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json',bundle/'retraining_reference/FROZEN_PARENT_CONFIG_UNCHANGED.json')
    cp(EXEC/'EXECUTION_REPORT.md',bundle/'retraining_reference/MEASURED_PRIOR_EXECUTION_REPORT_UNCHANGED.md')
    execution=read(EXEC/'EXECUTION_AUDIT.json')
    save(bundle/'retraining_reference/MEASURED_PRIOR_GPU_RESOURCES.json',dict(
        provenance_file=str((EXEC/'EXECUTION_AUDIT.json').relative_to(ROOT)),provenance_sha256=sha(EXEC/'EXECUTION_AUDIT.json'),
        gate=execution['gate'],training=execution['training'],evaluation=execution['evaluation'],
        estimated_future_F_budget_GPU_hours=[6,8],estimate_is_not_measured_V2_target_run=True))
    save(bundle/'retraining_reference/PENDING_F_V2_DATASET_BINDING.json',dict(
        historical_baseline_files_SHA256=BASELINE_BINDINGS,
        only_allowed_change='Teacher target version/text/label corpus and new provenance/output namespace',
        new_corpus='PENDING_NOT_GENERATED',new_config_runner_preflight='PENDING_POST_CORPUS_VALIDATION',
        fresh_pretrained_backbone_and_fresh_LoRA=True,continue_old_F_adapter=False,
        GPU_execution_authorized=False,RO_U_authorized=False,held_out_access=False))
    # Run existing symbolic boundary tests from the portable source-only payload.
    result=subprocess.run([sys.executable,str(stage/'tools/test_export_boundaries.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0 and 'Ran 9 tests' in result.stderr,result.stdout+result.stderr
    save(bundle/'validation/REUSED_CPU_REGRESSION_RESULTS.json',dict(status='PASS',tests=9,
        stdout=result.stdout,stderr=result.stderr,fixtures_are_not_business_targets=True))
    v2tests=subprocess.run([sys.executable,str(stage/'tools/test_teacher_target_v2_acceptance_v1.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert v2tests.returncode==0 and 'Ran 6 tests' in v2tests.stderr,v2tests.stdout+v2tests.stderr
    save(bundle/'validation/V2_BOUNDARY_TEST_RESULTS.json',dict(status='PASS',tests=6,
        stdout=v2tests.stdout,stderr=v2tests.stderr,fixtures_are_not_business_targets=True))
    jsonschema.Draft202012Validator.check_schema(read(stage/'TARGET_V2_OUTPUT_SCHEMA.json'))
    # Check every32source row remains exact and every included original file is hash-bound.
    assert (stage/'SOURCE_EVIDENCE.jsonl').read_bytes()==source_path.read_bytes()
    photo_checks=source_checks=0
    for row in source_rows:
        for v in row['exact_frozen_source_inputs'].values():
            assert sha(stage/v['file'])==v['utf8_sha256'];source_checks+=1
        for caps in row['captions'].values():
            for c in caps:
                if c['photo_available']:
                    assert sha(stage/c['photo_file'])==c['original_photo_sha256'];photo_checks+=1
    # Exclude every original/A full target from the drafting payload; do not copy old prompts/guides.
    old_rows=jl(original_files[2]);a_rows=jl(original_files[3])
    prior_texts=[v for r in old_rows for v in r['targets'].values()]+[
        v['text'] for r in a_rows for v in r['targets'].values()]
    for p in stage.rglob('*'):
        if p.is_file() and p.suffix in ('.md','.json','.jsonl','.txt','.py'):
            text=p.read_text(encoding='utf-8')
            assert not any(t in text for t in prior_texts), ('Anchoring leak',p)
    assert all(sha(Path(p))==h for p,h in input_hashes.items())
    assert all(sha(ROOT/p)==h for p,h in BASELINE_BINDINGS.items())
    def archive(path,folder):
        files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256_MANIFEST.json'}
        save(folder/'SHA256_MANIFEST.json',dict(files=files))
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n in list(files)+['SHA256_MANIFEST.json']:z.write(folder/n,n)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None and len(z.namelist())==len(files)+1
            assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
        return dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,manifest_files=len(files))
    source_archive=bundle/'STAGE_A_V2_32_SOURCES_ONLY.zip'
    source_receipt=archive(source_archive,stage)
    save(bundle/'validation/PREPARATION_VALIDATION.json',dict(status='PASS_OFFLINE_PREPARATION_ONLY',
        same32_training_businesses=True,sample_membership_changed=False,validation_test_contents_accessed=False,
        V2_targets_generated=0,source_views_reverified=source_checks,caption_photo_mapping_checks=photo_checks,
        unique_original_photos=206,old_and_StageA_wording_in_source_bundle=False,
        schema_Draft202012_valid=True,reused_CPU_tests=9,V2_boundary_tests=6,total_tests=15,
        original_files_rehashed_unchanged=True,
        baseline_six_SHA256_bindings_unchanged=True,known_quality_checks=12,known_researcher_results_assigned=0,
        production_ID_batches=550,production_sources_generated=0,production_targets_generated=0,
        GPU_model_Teacher_API_calls=0,quality_gate='NOT_RUN',new_target_counts='NOT_AVAILABLE_NOT_GENERATED'))
    save(bundle/'provenance/INPUT_BINDINGS.json',dict(original_file_SHA256=input_hashes,
        baseline_bindings=BASELINE_BINDINGS,source_only_archive=source_receipt,policy_sha256=policy_hash,
        preparation_request_reports_adjudication_findings=True,separately_supplied_filled_adjudication_table=False,
        researcher_grades_or_preferences_invented=False))
    (bundle/'README_FIRST.md').write_text('''# Teacher Target V2 — complete offline preparation

THIS MASTER PACKAGE IS FOR INDEPENDENT PREPARATION REVIEW, not source-only drafting.
Read policy, prompts, QUALITY_GATE_V2.md, tokenizer/validation specification,
production plan and retraining/resource plan. No V2targets or V2student exist yet.

After researcher approves and explicitly starts the32case pilot, use ONLY the
included STAGE_A_V2_32_SOURCES_ONLY.zip in a FRESH GPT Work conversation. The old
targets and Stage A wording are deliberately outside it. Read the V2prompt and
process8batches of4. Save all32V2drafts before any comparison material is opened.
Then supply stage_b_comparison_32/ plus the Stage B V2prompt/known-case checklist
from this master package to compare ORIGINAL/A/V2. Do not revise drafts silently.
Use the supplied local utility for exact counts and stop at the researcher gate.

Only if researcher confirms the predefined quality gate and separately authorizes
production, materialize the planned2200sources/6600targets workflow. The550batch
layout today is ID metadata only; no validation/test source content was read here.
No new GPUrunner or executable production-generation client is supplied. One fresh
Filtered training run would need the complete validated corpus and separate
hash-bound authorization. Historical approvals/deadlines are not reused/reset.

The new request reports32review/12adjudication findings; it is preserved verbatim.
No completed researcher adjudication table was separately supplied. Existing
unfilled table and earlier provisional self-review remain identifiable; no Core
labels, scores, quality GO or signatures are invented from a general decision.
These are reusedTRAIN cases, not a fresh held-out measure of generalization.

The single final target-redesign direction is V2. Failure does not trigger a third
policy, architecture/decoder search, automatic RO/U, mass regeneration or training.
Original Teacher/Stage A/source/split/checkpoint artifacts remain unchanged.
Stop for researcher review. See validation/ and SHA256_MANIFEST.json for receipts.
''',encoding='utf-8')
    master=ROOT/'outputs/cloud_handoff/TEACHER_TARGET_V2_PREPARATION_V1.zip'
    assert not master.exists()
    receipt=archive(master,bundle)
    master.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+master.name+'\n',encoding='utf-8')
    download=Path('C:/Users/VUONGLOCTRUONG/Downloads')/master.name
    assert not download.exists();cp(master,download)
    save(OUT/'DELIVERY_RECEIPT.json',dict(status='OFFLINE_READY_STOP_FOR_INDEPENDENT_REVIEW',
        master=receipt,download=str(download),source_only=source_receipt,
        policy_sha256=policy_hash,V2_targets_generated=0,production_generation_authorized=False,GPU_authorized=False))
    print(json.dumps(read(OUT/'DELIVERY_RECEIPT.json'),indent=2))


if __name__=='__main__':
    main()
