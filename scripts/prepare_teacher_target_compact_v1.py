"""Offline Compact V1 policy, source-only pilot and conditional research plans."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_i/teacher_target_compact_v1_preparation'
BUNDLE = OUT/'bundle'
SOURCE = OUT/'source_only_32'
DOWNLOADS = Path.home()/'Downloads'
PREP = ROOT/'outputs/stage_i/teacher_target_v2_preparation_v1/bundle'
ERRATA = ROOT/'outputs/stage_i/teacher_target_v2_1_errata_v1/bundle'
REQUEST = Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/8d8ee07b-4196-40ea-b020-15621a0ac087/Pasted text.txt')
PREFIX = 'TEACHER_TARGET_V2_1_CORRECTION_32/'
SRC_PATH = 'inputs/stage_a_v2/source_package/'
ORIGINAL = 'inputs/preparation/stage_b_comparison_32/ORIGINAL_TARGETS_UNCHANGED.jsonl'
V21 = 'correction_v2_1/drafts/teacher_target_v2_1_32.jsonl'
CONDITIONS = ('reviews_only','unfiltered','filtered')
POLICY_ID = 'TEACHER_TARGET_COMPACT_V1'


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def put(root,rel,data):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f: f.write(data if isinstance(data,bytes) else data.encode('utf-8'))


def js(root,rel,value): put(root,rel,json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def parse(raw): return [json.loads(x) for x in raw.decode('utf-8-sig').splitlines()]


def manifest(root):
    js(root,'SHA256_MANIFEST.json',dict(files={p.relative_to(root).as_posix():sha(p)
       for p in sorted(root.rglob('*')) if p.is_file()}))


def pack(root,path):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        files=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert set(z.namelist())==set(files)|{'SHA256_MANIFEST.json'}
        for n,h in files.items(): assert sha_bytes(z.read(n))==h,n


POLICY = '''# Teacher Target Compact V1 policy

Policy ID: TEACHER_TARGET_COMPACT_V1. OFFLINE PREPARATION ONLY; no Compact drafts
or execution approval. This is ONE final compact-supervision hypothesis, not a
new open-ended target-redesign cycle.

Parent: frozen V2.1 bounded errata policy, SHA256
a140f0bf7ca793640380c3922baa8892d9f014a1e7eaf11623f0169485c351a9,
itself a delta to V2 evidence-dense policy. The present document restates all
operative source/support rules generically so a source-only generator never
needs case-specific answers or existing target wording. The only intended policy
change is wording/redundancy compression with flexible complete-label guidance.
V2.1 candidate texts are frozen comparators, not researcher-approved gold.

Exact pinned LED-tokenizer guidance:
- RO COMPLETE serialized target including header and final EOS, excluding initial
  BOS and all padding: approximately 155–175 label tokens.
- U/F COMPLETE serialized targets including both headers and final EOS, excluding
  initial BOS and padding: approximately 185–210 label tokens.
- Visual BODY alone, no header/special tokens: preferably 20–40 tokens WHEN useful
  condition-caption evidence exists. No minimum for empty/unhelpful evidence.
- Intended reduction approximately 15–20% versus frozen V2.1 complete labels over
  the SAME cases/conditions. This is a hypothesis to measure, not a result or quota.

These are flexible ranges, never hard truncation, mandatory padding or quality
thresholds. Some V2.1 cases are already shorter than this guidance: do not expand
them to meet a range or remove important facts to meet a percentage. The hard
unchanged 768-label ceiling still applies; overlength is reported without silent
truncation. No Stage A 110–170 review-body compression target applies here.

Read ALL frozen selected reviews in order. Before drafting identify recurring,
distinctive and useful business information and important same-aspect contrasts.
Compress phrasing and semantic redundancy FIRST: integrate related food/service/
value observations instead of appending parallel clauses or repeating the topic.
Keep named dishes, important complaints and positive experiences, relevant
price/portion/value information and practical constraints where source-salient.
Specific source-backed content outranks generic praise. Do not force all minor
details into the summary or give isolated incidents the weight of consensus.

Preserve BOTH supported sides about the SAME dish or service aspect; unrelated
positive and negative sentences do not establish balance. Keep condiment/dish/
service-event identities separate. Distinguish two located accounts from a broad
consensus and duplicated descriptions of one event from independent recurrence.
Do not manufacture symmetric sentiment when evidence is asymmetric. Source-based
salience choices are recorded outside target prose, not hidden as token savings.

Serious isolated reports need both honest attribution and a salience rationale.
Do not automatically include or exclude them. Distinguish firsthand, hearsay and
allegation from established fact; no unsupported safety, diagnosis, cause, intent,
current-status or external assertion. A temporal sequence does not prove a cause.
Keep classification/process wording in claim metadata, never restaurant prose.

One canonical REVIEW_CONSENSUS is derived ONLY from reviews. It equals the exact
single-space join of review_claims.text and is copied byte-identically into RO/U/F.
RO text = '[REVIEW_CONSENSUS]' + newline + canonical body. No visual header or
visual assertion in RO. U/F = RO + two newlines + '[VISUAL_EVIDENCE]' + newline +
that condition's visual body, exactly the single-space join of visual_claims.text.
The same review cannot be selectively weakened or enriched across conditions.

Visuals use ONLY the relevant condition's model_visible_caption text. U/F local
caption numbers are independent: bind global evidence ID, literal quote and exact
photo ID. Keep a claim's subjects/actions/relations in ONE photograph. Never infer
relationships across photographs or supplement visuals using review descriptions.
Use a few useful concise food/setting descriptions, avoiding repeated generic
objects/scenes. Caption presence is not factual verification. Original photos may
veto uncertain caption details or verify a less-specific entailment of that SAME
caption; they may not introduce photo-only names, objects, counts or relationships.
Do not label questionable BLIP-2 identity as verified fact. Prefer a useful safe
partial caption proposition over discarding an entire scene when defensible.
Photo audit metadata records consistent/uncertain/contradicted/not_checked and
rationale separately from caption entailment; all model judgments are provisional.
If captions are empty, visual_claims=[] and visual body empty after the fixed
header. Do not invent an absence description. If nonempty captions offer nothing
defensible/useful, explain omission outside the target; do not fill a token quota.

Claim evidence uses exact IDs and literal source quotes. Preserve original source
inputs, selected review IDs/order, captions, images, mappings, split and tokenizer.
Save new Compact records separately; never overwrite Original/A/V2/V2.1. No
semantic retries, alternative candidates or policy tuning after outcome inspection.
Mechanical checks/counts do not establish semantic support, Core importance or GO.
Stop for independent researcher review after the bounded pilot and fixed QA.
'''

GENERATION = '''# GPT Work Compact V1 generation prompt — frozen version 1

STATUS: PREPARATION ONLY. Run only after explicit researcher start approval.
Use ONLY STAGE_A_COMPACT_SOURCE_ONLY_32.zip in a fresh source-only conversation.
Do not upload the master preparation/comparison files to the drafting conversation.
Requested authoring setting inherited from the pilot: GPT-6.1 Sol / Medium; record
what is actually verifiable, never claim the model/setting was independently proved.

Read COMPACT_V1_POLICY.md, COMPACT_OUTPUT_SCHEMA.json and TOKENIZER_SPEC.md.
Exactly the same32TRAIN businesses, eight ordered batches of four, no resampling.
Read every selected review and U/F caption; inspect linked original photos solely
to verify/veto caption content. Do not open old targets, comparison notes or known-
case answer panels. The generic source policy contains the needed support rules.

For each business identify recurring business-specific information, same-dish/
aspect disagreements, useful practical/value information and serious isolated
reports before compression. Draft ONE family with one canonical review shared
RO/U/F. Remove repetition and tighten clauses before dropping substantive content.
Preserve material qualifications, attribution, useful visual evidence and zero-
caption compliance. No shortest-output reward, invented balancing, causal glue,
cross-photo relationship, photo-only fact or external knowledge.

Save first complete candidate records, unchanged, at drafts/compact_v1_32.jsonl
and ordered batch_01...08.jsonl, under the supplied schema. Save raw returned
bytes, prompt/policy/schema/source hashes, time/operator and known runtime details
in append-only raw_outputs/ and batch receipts. No parent target wording is needed.
For claims cite exact source IDs/literal quotes/photo IDs. Keep uncertainty and
evidence-characterization metadata outside summaries. Save VISUAL_SUPPORT_AUDIT
with condition, claim_id, caption ID/quote, photo_id, photo-check status/rationale,
researcher_confirmed=false. Report source-based omissions/salience rationale
without purporting to compare with an unseen parent.

Measure EXACT tokenizer counts using the provided local assets, no network/model:
python tools/validate_compact_v1.py --drafts drafts/compact_v1_32.jsonl \\
 --sources SOURCE_EVIDENCE.jsonl --schema COMPACT_OUTPUT_SCHEMA.json \\
 --tokenizer tokenizer --output drafts/COMPACT_MECHANICAL_AND_COUNTS.json
Pinned tokenizers==0.21.1 and jsonschema==4.23.0. RO155–175 and U/F185–210 are
COMPLETE content/EOS label counts including headers, not review-body counts.
Visual20–40 is BODY-only. Guidance departures warn; information outranks ranges.
Never claim exact counts if dependencies cannot run; report NOT_MEASURED instead.
Do not rewrite completed targets after counts/review. Overlength or support failure
is evidence to report, not permission for semantic regeneration.

Only genuinely incomplete/malformed records may receive at most two additional
technical attempts under identical policy/source; retain all attempts, never
redraft technically valid businesses for nicer style/length. Conflicting complete
duplicates or retry cap => stop and report, do not choose a preferred candidate.

Return one result ZIP with32records/96texts, raw attempts, source/claim links,
photo checks, exact counts, issue notes and SHA256 manifest. STOP before seeing
comparators. No production6600, GPU, held-out or policy iteration is authorized.

The separately prespecified8newTRAIN QA cases are not part of these32. Their IDs
and source packaging are withheld from this drafting bundle. After the32are frozen
and a separate approved source-only QA handoff is supplied, use this SAME frozen
policy/schema/prompt for ONE QA pass without optimization. Do not invent QA IDs.
'''

COMPARISON = '''# Independent comparison workflow — Compact / Original / frozen V2.1

PREPARATION ONLY; no Compact targets exist yet. The32pilot businesses remain the
same24random plus four targeted strata of2. V2.1 is the exact downloaded frozen
candidate version; its quality gate remains unconfirmed. Freezing bytes is not
accepting its quality. Original and V2.1 are comparators, never independent gold.

1. Freeze completed Compact32raw output/record/target/source hashes BEFORE opening
   Original or V2.1. Use a different reviewer/conversation with no authoring history
   where practical. Record authorship, previous exposure and whether blindness was
   actually achieved; a new chat alone cannot establish reviewer independence.
2. Source-only first: reviewer records a source-grounded Core/Supporting information
   panel, exact evidence IDs/quotes and important same-aspect contrasts for all32.
   Freeze panel hash/time BEFORE viewing candidate identities/texts. These TRAIN
   panels are provisional reviewer judgments, not D-053 human-approved validation
   references. Do not retrofit Core panels to reward compression.
3. ADMIN_IDENTITY_KEY_KEEP_SEPARATE.json maps per-business A/B/C labels. An operator
   builds reviewer packets from the exact saved32×3×3target texts using that key;
   no header rewriting, style normalization or regenerated baseline. The reviewer
   receives sources/photos/panel and anonymous texts, not the key, provenance,
   token lengths, old preferences, case-specific answers or this master ZIP.
   Review RO/U/F separately, preserving within-business pairing/canonical review.
   Use one reviewer tool/workflow, not a comparison-author self-evaluation presented
   as independent. If independence is unavailable, keep outcomes provisional.
4. Score/annotate Core coverage, same-aspect balance, factual support, recurring
   business specificity, useful condition-caption visuals, semantic redundancy,
   conciseness and usability. Record exact missing/unsupported spans, severity,
   evidence and confidence. Treat zero-caption assertions as unsupported. Preserve
   tie/inconclusive and component-versus-family distinctions. After this provisional
   blind pass, unblind for technical counts and the known failure-case audit;
   timestamp both phases and preserve changes to judgments. Do not tune targets.
5. Known-case check after unblinding: all32register, all13StageBissue entries and
   V2.1 residual/post-save findings. Pat same-dish balance/fork specificity; Gus
   caption versus photo; Mystic unsupported cancellation; Torch/Pita incident
   attribution and salience; Jollibee/TopOut safe useful food/gym; OnTheBorder queso
   versus salsa; Backyard two reports/zeroF; Convivio readability; all other
   photograph-dependent, recurring-detail and appended-topic tradeoffs. Do not
   revive withdrawn photo doubts or require every minor item by checklist.
6. Exact counts: compare Compact with Original AND V2.1 for all96 paired labels,
   preserving hash-equal inputs and content/EOS denominator. Report per-condition
   totals/mean/median/range, per-business differences and random/targeted strata.
   Aggregate reduction =1-sum(Compact label counts)/sum(V2.1 label counts), with
   separateRO/U/F values; same N/conditions. Do not average condition observations
   as independent businesses or claim reduction by counting body tokens alone.
   Approximately15–20% is the planned hypothesis, not a requirement to force edits.
7. Run the prespecified8otherTRAIN QA after policy freeze; no selection/replacement
   from outcomes. The QA reviewer must be independent of authoring where practical.
   Use source panels and complete Original/Compact families, NOT invented V2.1 for
   unseen cases (no V2.1 exists there). Thus QA tests generalization versus Original;
   it cannot estimate Compact-versus-V2.1 generalization. Record this asymmetry.
   Do not optimize prompt after QA. Unexpected failures remain failures/evidence.

All scores/confirmations in RESEARCHER_COMPACT_GATE_FORM.json start null/false.
Only explicit researcher adjudication can satisfy QUALITY_GATE_COMPACT_V1.md.
No automatic win-rate/loss/length rule. If material concerns or meaningful gain
cannot be resolved, stop final Compact direction and finish frozen-supervision
matched RO/U/F/Shuffled. Passing this review does not authorize corpus or GPU.
'''

GATE = '''# Compact V1 predefined quality gate

Status NOT_RUN: Compact32=0; independentQA8=0; no corpus/GPU approval.
Exactly32repeatedly reviewed TRAIN businesses plus8prespecified OTHER TRAIN cases.
Report original24random and four targeted strata separately; QA8is distinct and
does not erase the repeated-exposure/targeted-overfitting limitation of the32.

Technical checks on every32family/96target (and separately8QA/24target):
- exact predeclared membership/order, source and policy/schema/prompt hashes;
  Original/A/V2/V2.1 bytes intact, raw complete candidates immutable/new namespace;
- one identical canonical review acrossRO/U/F; exact claim/target serialization;
- literal review/condition-caption/photo links, no U-to-F leakage or cross-photo
  relationship; RO no visual; zero-caption claims/body empty;
- supplied pinned tokenizer, EOS retained/initialBOSremoved for labels only,
  exact complete-label counts<=768, no truncation, no silent rewrite/replacement;
- photo verification separated from caption entailment; no photo-only substitutions.
Mechanical PASS is necessary but cannot establish support, salience or quality.

Researcher-confirmed source-quality criteria:
1. No serious unsupported claim introduced or retained as verified by compression,
   including ungrounded causal explanations, allegations or unsafe specificity.
2. No new important same-aspect positive/negative loss in priority cases. Preserve
   appropriate recurrence/asymmetry and useful named dishes/value/constraints.
3. No systematic recurring Core omission. For each omission record evidence,
   importance, scope, comparator coverage and rationale; a repeated pattern of
   losing meaningful constraints/dissent is blocking even if each case sounds fluent.
4. No zero-caption visual assertions; canonical/body/condition boundaries hold.
5. Meaningful complete-label reduction versus V2.1 without substantive information
   loss, judged jointly with grounding, balance, useful visuals, redundancy,
   conciseness and overall usability; report against Original too. Shortness or
   resemblance to Teacher cannot establish GO. Minor supporting omissions/range
   deviations need explicit exception rationale; material errors cannot be waived
   by token savings. Do not force 15–20% per case or reject already-compact inputs
   for failing a band. If information preservation prevents meaningful reduction,
   report the hypothesis unsupported/inconclusive rather than truncating.
6. IndependentQA8must show no serious introduced support errors, material balance
   losses or systematic Core loss versus frozen Original. Single sample cannot
   prove population-wide generalization. No prompt changes after QA; unavailable
   source/evidence/independence or unresolved major issue remains pending/NO-GO.

All32known-case/residual questions, including serious-incident salience and safe
visual specificity, receive actual researcher disposition. Author self-review and
inconclusive preferences are not approval. Independent source-first blinded review
is preferred but report actual independence honestly; researcher confirmation is
still required. Per-case confidence/severity and final sign-off fields stay blank
until supplied. No invented human Core labels or signatures.

Outcome: PASS only with defensible source-based quality, meaningful reduction and
explicit researcher acceptance of evidence/exceptions/risks. Then freeze policy,
prompt, schema, outputs and counts; seek separate production approval. PASS never
automatically starts6600generation or training. FAIL/mixed/inconclusive => stop
Compact optimization, preserve evidence and complete established frozen-supervision
matchedRO/U/F/Shuffled. No CompactV2, alternate target search or semantic retry.

Residual risks: same32exposure, smallQA8, subjective salience, uncertain captions,
model-assisted photo/review judgments, no guarantee shorter supervision improves
student. Student quality is a SECOND gate after separately authorized freshF.
'''

PRODUCTION = '''# Conditional Compact V1 production plan

PREPARATION ONLY. Need accepted32pilot + fixed8QA and separate explicit researcher
production authorization. No full source materialization, Teacher calls or corpus
generated here. Current Compact count0; existing frozen Teacher files unchanged.

Population: frozen2000train/200validation;6600RO/U/Frecords, one canonical review
perbusiness. Use copied deterministic550batches of4(first8pilot32, remainingtrain
IDssorted, then200valsorted). Generation order never changes downstream source/
training order. No held-out content. Use new teacher_target_compact_v1 namespace.
Freeze exact policy/prompt/schema/tokenizer/source/acceptedoutputhashes and runtime
provenance; source-only contexts carry no old targets or case-specific answers.

After authorized source export, hash each business's exact three model inputs,
selected reviews/captions/order, photo mapping and source assets. Never change a
source based on outcome. Keep U/F independent, zero-caption empty, safe caption
subset/photo-veto rules and claim links at scale. Original/A/V2/V2.1 remain separate.
Produce2200canonicalrecords plus six split×conditionJSONLs, then6600targetrecords
with unique(split,business,condition), and a full manifest/count/audit ledger.

Pilot32+QA8may be reused ONLY if explicitly accepted unchanged under exactly the
frozen production policy/source/schema bindings; they occupy their predeclared
batches/IDs with lineage receipts. Up to40families/120targets reusable, leaving
2160businesses/6480targets. The QA8retain their 'policy-generalization check' history;
they are training businesses, not a held-out model test. If not accepted, stop for
decision rather than silently regenerating or changing supervision version.

Append-only raw_attempts/<batch>/attemptNN, response bytes, hashes, source envelope,
time/operator/runtime settings and RECEIVED/TECHNICAL_VALID/QUALITY_PENDING/ACCEPTED
states. Resume by recomputing hashes/membership and skipping ONLY complete valid
unchanged families. Identical duplicates idempotent; conflicting valid duplicates
stop. At most2additional technical retries for incomplete/malformed records under
identical instructions; no rewording semantically valid drafts, cherry-picking or
policy search. Material semantic failure stops for adjudication, not a retry loop.

100%mechanical checks: split/IDs/order, canonical consistency, evidence references,
caption/photo scope, all zero-caption cases, exact tokenizer complete-label counts
and hard768limit, no source edits/truncation. Source-only utility is pilotTRAIN32
by default; future production bounds/splits require an approved version binding,
including tests for2000train/200valmembership and exclusion of any other ID. Do not
misuse the32casechecker as a full-corpus certificate. Preserve new content/EOS
labels separately; copy original inputIDs/masks/globalattention/shapes/order exactly.

Predeclare corpus QA before any generation: all40accepted anchors, every detected
support/salience/caption/zero-caption flag, and deterministic50additionalTRAIN plus
10validation businesses ranked SHA256 UTF8 '20261022|COMPACT_V1_CORPUS_QA|ID',
excluding the40anchors fromtrain. Source-evidence/photo review all3conditions for
every selected business; all zero-caption compliance checked mechanically. Source-
only reviewer panels before candidate identities where feasible, true researcher
sign-off, severity/confidence and audit coverage disclosed bysplit. Sample QA does
not establish allcorpushumantruth; record extra validation exposure. Major concerns
block release; no silent fixes or follow-up Teacher policy iterations.

Freeze entire2200/6600corpus only after integrity+sourcequalitysignoff. A separate
freshFproposal must bind new corpus/runner/preflight and receive explicit GPU
authorization. No automaticRO/U. If gates cannot be justified, stop the final
Compact direction and finish matchedRO/U/F/Shuffled with existing frozenTeacher.
'''

RETRAIN = '''# Controlled fresh LED-large F comparison ? conditional plan

PREPARATION ONLY. After Compact pilot/QA acceptance, a complete validated corpus
and separate hash-bound GPU authorization, train ONE fresh Filtered model. Keep
all Original-supervised checkpoints and historical evidence intact. Do not continue
an old F adapter or interrupted Colab checkpoint.

Use the successful LED-large Decoder Policy V2 settings unchanged; the exact
parent config and measured receipts are in retraining_reference/. Only the
supervision corpus/version and new output/provenance bindings change:
- Official allenai/led-large-16384 revision
  cd59d11c3528415b7dda4dfc95cc8f138aceda2e, fresh seed-42 initialization.
- Fresh Expanded-LoRA, all linear modules except tied lm_head: 228 modules,
  456 trainable tensors, 4,915,200 parameters; r=8, alpha=16, dropout=0.
- Frozen 2,000 train / 200 validation businesses, same selected reviews, captions,
  inputs and training order; physical batch 1, accumulation 8.
- AdamW lr=0.0002, decay=0.01, betas=(0.9,0.999), epsilon=1e-8;
  5 epochs / 1,250 successful updates, linear schedule, warmup 63, clipping 1.
- FP32 parameter storage, FP16 AMP starting at 65,536 with the accepted bounded
  native backoff, unchanged checkpointing and state-continuity checks.
- Labels [content..., EOS], removing ONLY initial BOS. Sources remain unchanged;
  shift once with decoder-start EOS=2. Loss scale is unrelated to target length.

A future versioned runner/preflight must bind the new corpus and parent code/pins.
Do not patch the historical runner or reuse its authorization for new supervision.
The existing feasibility gate requires 3 successful optimizer updates, 24 nominal
and at most 88 actual microbatches under its bounded AMP policy, 2 stress
generations, at least 15% measured memory headroom, and numerical/gradient,
backbone-integrity and exact checkpoint-restoration checks. Discard gate state
before fresh training. Failure stops execution; no unapproved retry or fallback.

After epoch 1, run exactly 4 predeclared technical sanity generations with the
accepted BOS-loop >=5, blank/special-dominance and state-continuity criteria.
Isolated BOS or cap warnings alone do not fail. Technical recovery is not quality
GO. Preserve full-state checkpoints at updates 250/500/750/1000/1250. Generate the
fixed 24 development businesses (20 random + 4 targeted) at epochs 3 and 5: 48 new
outputs, 54 maximum generation calls including gate 2 and early sanity 4. Preserve
beam4/max_length768 and all other parent decoding settings. No decoding search.

Compare with SAVED successful Original-supervised LED-large F Decoder Policy V2
epochs 3 and 5 on these same businesses; do not regenerate baseline predictions.
Use approved D-053 Core/Supporting references and source reviews/captions/photos,
never Teacher targets as independent gold. Prefer source-first anonymous paired
packets with the identity key withheld. Assess Core coverage, balanced opinion
synthesis, factual/visual grounding, useful visuals, semantic repetition,
zero-caption compliance and usability. Report 20 random and 4 targeted separately.
Report both new checkpoints; recommend one only on substantive quality, then use
one consistent checkpoint-selection rule for any eventual matched family.

This tests a frozen Compact supervision package, including wording/content
tradeoffs, rather than isolating token length as a causal variable. The saved
baseline shares architecture/protocol, but hardware/runtime numerical trajectories
may differ. One seed and repeated development exposure limit causal and
population-wide claims. Raw loss across different target texts has different
denominators and is not a direct quality comparison. D-059 Expanded LED-base
epoch 3 can remain a practical unmatched benchmark, not causal target evidence.

If quality is weak, mixed, inconclusive or lacks meaningful researcher-confirmed
improvement, STOP target redesign and complete the established matched
RO/U/F/Shuffled evaluation using existing frozen supervision. GO needs substantive
benefit without material Core, balance, grounding or visual regression. Only then
prepare matched RO and U using identical Compact supervision, initialization,
optimization and checkpoint-selection rules, subject to separate authorization.
Shuffled reuses matched F. Do not mix Original RO/U with Compact F in the primary
matched comparison. The established LED-base family remains primary until a
complete accepted matched replacement exists.

Planning estimate from the prior RTX 3090 run: model phases about 4h20; allow
6?8 GPU-hours for one F including setup/export, not a new benchmark or guarantee.
Prior training plus continuity checks peaked near 17.68 GB with 30.10% headroom;
shorter labels may change runtime/memory, but that benefit is unmeasured. Future
authorization needs an actual UTC start/hard stop and must protect at least 14 days
for final evaluation, the 30-business human study, thesis writing and defense.
Do not reset historical deadlines. Verify full off-instance backups before any
resource destruction. No rental, training, additional experiment or tuning now.
'''


VALIDATOR = r'''"""CPU-only Compact draft checks; no generation, model or API calls."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
from count_led_tokens import load_tokenizer, target_counts, validate_proposal, CONDITIONS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('drafts','sources','schema','tokenizer','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-count',type=int,choices=(32,8),default=32,
                    help='8 only with a separately approved prespecified QA source package')
    args=ap.parse_args()
    if importlib.metadata.version('jsonschema')!='4.23.0':
        raise ValueError('Pinned jsonschema==4.23.0 required')
    import jsonschema
    parse=lambda p:[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
    sources,drafts=parse(args.sources),parse(args.drafts)
    if len(sources)!=args.expected_count or len(drafts)!=args.expected_count:
        raise ValueError('Predeclared source/draft count differs')
    ids=[x['business_id'] for x in sources]
    if len(set(ids))!=len(ids) or ids!=[x['business_id'] for x in drafts]:
        raise ValueError('Source/draft IDs missing, duplicated, unexpected or reordered')
    if any(s['split']!='train' for s in sources):
        raise ValueError('TRAIN pilot/QA only; no validation or held-out content')
    schema=json.loads(args.schema.read_text(encoding='utf-8'))
    tok=load_tokenizer(args.tokenizer)
    results=[]
    for row,source in zip(drafts,sources):
        jsonschema.Draft202012Validator(schema).validate(row)
        expected={c:v['utf8_sha256'] for c,v in source['exact_frozen_source_inputs'].items()}
        if row['source_inputs_sha256']!=expected: raise ValueError('Source hash binding differs')
        mechanical=validate_proposal(row,source)
        counts={c:target_counts(tok,row['targets'][c]['text'],c) for c in CONDITIONS}
        if not all(x['within_current_768_label_limit'] for x in counts.values()):
            raise ValueError('Over hard768label limit; no truncation or silent repair')
        warnings=[]
        for c in CONDITIONS:
            n=counts[c]['corrected_training_content_EOS_label_tokens']
            low,high=(155,175) if c=='reviews_only' else (185,210)
            if not low<=n<=high: warnings.append(c+': outside flexible COMPLETE-label guidance')
            if c!='reviews_only' and source['caption_counts'][c]:
                v=counts[c]['visual_body_standalone_no_special']
                if not 20<=v<=40: warnings.append(c+': outside flexible visual-BODY guidance')
        results.append(dict(business_id=row['business_id'],mechanical=mechanical,
                            exact_LED_counts=counts,guidance_warnings=warnings))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.sources,args.drafts,args.schema)}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(dict(status='PASS_MECHANICAL_ONLY_QUALITY_PENDING',businesses=len(drafts),
            target_conditions=len(drafts)*3,input_sha256=hashes,results=results,
            reduction_vs_V2_1='NOT_MEASURED_WITHOUT_COMPARATORS',
            semantic_quality_proven=False,researcher_confirmed=False,
            production_authorized=False,GPU_authorized=False),f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('Mechanical checks/counts complete; no quality GO, target edits or generation.')
if __name__=='__main__': main()
'''


def main():
    assert not OUT.exists(),'No overwrite; version already exists'
    OUT.mkdir(parents=True)
    archive=DOWNLOADS/'TEACHER_TARGET_V2_1_CORRECTION_32_RESULTS.zip'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
        for n in names:
            p=PurePosixPath(n)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n
        entries=json.loads(z.read(PREFIX+'RESULT_INTEGRITY_MANIFEST.json'))['files']
        assert {PREFIX+n for n in entries}==set(names)-{PREFIX+'RESULT_INTEGRITY_MANIFEST.json'}
        for n,spec in entries.items():
            raw=z.read(PREFIX+n)
            assert len(raw)==spec['bytes'] and sha_bytes(raw)==spec['sha256'],n
        parent=json.loads(z.read(PREFIX+'SHA256_MANIFEST.json'))['files']
        assert len(parent)==373
        for n,h in parent.items():
            assert sha_bytes(z.read(PREFIX+n))==h
            assert z.read(PREFIX+n)==(ERRATA/n).read_bytes(),n
        assert z.read(PREFIX+'V2.1_ERRATA_POLICY.md')==(ERRATA/'V2.1_ERRATA_POLICY.md').read_bytes()
        incoming=OUT/'incoming';incoming.mkdir()
        shutil.copyfile(archive,incoming/archive.name);assert sha(archive)==sha(incoming/archive.name)
        sources=parse(z.read(PREFIX+SRC_PATH+'SOURCE_EVIDENCE.jsonl'))
        originals=parse(z.read(PREFIX+ORIGINAL));v21=parse(z.read(PREFIX+V21))
        ids=[s['business_id'] for s in sources]
        assert len(set(ids))==len(sources)==len(originals)==len(v21)==32
        assert ids==[r['business_id'] for r in originals]==[r['business_id'] for r in v21]
        assert all(s['split']=='train' for s in sources)
        # Parent source manifest is fully verified; select only source/tokenizer material.
        source_manifest=json.loads(z.read(PREFIX+SRC_PATH+'SHA256_MANIFEST.json'))['files']
        for n,h in source_manifest.items(): assert sha_bytes(z.read(PREFIX+SRC_PATH+n))==h,n
        keep_roots=['SOURCE_EVIDENCE.jsonl','CAPTION_PHOTO_MAPPING.json','PILOT_SAMPLE_MANIFEST.json']
        copied={}
        for n in z.namelist():
            if not n.startswith(PREFIX+SRC_PATH): continue
            rel=n[len(PREFIX+SRC_PATH):]
            if rel in keep_roots or rel.startswith(('photos/','frozen_inputs/','batches/','tokenizer/')) or rel in (
                    'tools/count_led_tokens.py','tools/test_export_boundaries.py'):
                put(SOURCE,rel,z.read(n));copied[rel]=sha_bytes(z.read(n))
        put(BUNDLE,'COMPACT_V1_POLICY.md',POLICY)
        put(BUNDLE,'GPT_WORK_COMPACT_GENERATION_PROMPT.md',GENERATION)
        put(BUNDLE,'INDEPENDENT_COMPARISON_WORKFLOW.md',COMPARISON)
        put(BUNDLE,'QUALITY_GATE_COMPACT_V1.md',GATE)
        put(BUNDLE,'PRODUCTION_GENERATION_PLAN.md',PRODUCTION)
        put(BUNDLE,'CONTROLLED_LED_LARGE_RETRAINING_PLAN.md',RETRAIN)
        policy_hash=sha(BUNDLE/'COMPACT_V1_POLICY.md')
        for n in ('COMPACT_V1_POLICY.md','GPT_WORK_COMPACT_GENERATION_PROMPT.md'):
            put(SOURCE,n,(BUNDLE/n).read_bytes())
        schema=json.loads(z.read(PREFIX+'TARGET_V2_1_OUTPUT_SCHEMA.json'))
        schema['$id']='urn:teacher-target-compact-v1';schema['title']='One provisional Compact V1 TRAIN business'
        schema['properties']['policy_id']['const']=POLICY_ID
        schema['properties']['policy_sha256']['const']=policy_hash
        schema['properties']['status']['const']='PROVISIONAL_GPT_WORK_COMPACT_V1_REQUIRES_RESEARCHER_REVIEW'
        js(SOURCE,'COMPACT_OUTPUT_SCHEMA.json',schema);js(BUNDLE,'COMPACT_OUTPUT_SCHEMA.json',schema)
        put(SOURCE,'tools/validate_compact_v1.py',VALIDATOR)
        put(SOURCE,'tools/requirements.txt','tokenizers==0.21.1\njsonschema==4.23.0\n')
        tokenizer_spec='''# Exact LED tokenizer and count denominators

Use only tokenizer/ with tokenizers==0.21.1. load_tokenizer verifies all5asset hashes
and IDs BOS0/PAD1/EOS2. See TOKENIZER_BINDING.json for revision and source hashes.
No padding/truncation/download. Complete serialized labels=count(encoded target
with BOS/content/EOS)-1, removing initialBOSonly, retaining EOS. Headers/separators
are INCLUDED. RO155–175/U-F185–210guidance refers to these complete labels.
Visual20–40 counts BODY only, no marker/EOS. ReviewBODY also reported separately.
Do not add body counts to infer full-label counts; tokenization crosses boundaries.
Guidance is flexible; hard768unchanged. No actual Compact counts exist yet.
Inherited count utility and9symbolic boundary tests validate serialization/evidence
mechanics only, not entailment/salience/quality. New compact utility supports32TRAIN
by default;8QA only with the separate predeclared source handoff. No productionval
or held-out validation. No word/character-based estimates called LED tokens.
'''
        put(SOURCE,'TOKENIZER_SPEC.md',tokenizer_spec);put(BUNDLE,'TOKENIZER_SPEC.md',tokenizer_spec)
        put(SOURCE,'README_FIRST.md','''# Compact V1 SOURCE ONLY — preparation, not started

Same32TRAIN/8batchesof4, unchanged reviews/captions/206photos/96model inputs.
Read COMPACT_V1_POLICY.md and GPT_WORK_COMPACT_GENERATION_PROMPT.md after explicit
researcher start. No old targets, prior comparison judgments or case-specific answer
panels included. Use a fresh source-only authoring context; save32drafts before
comparators are opened. Generic source rules inherit V2.1 without revealing answers.
No generation authorized by this export. QA8sources are not included/inspected;
the fixed sample exists separately for later approved source-only handoff.
''')
        js(SOURCE,'SOURCE_INTEGRITY_AND_POLICY_BINDINGS.json',dict(
            parent_V2_1_policy_sha256=sha(ERRATA/'V2.1_ERRATA_POLICY.md'),
            compact_policy_id=POLICY_ID,compact_policy_sha256=policy_hash,
            copied_source_files=copied,source_and_tokenizer_bytes_unchanged=True,
            parent_prompts_targets_case_notes_excluded=True,qa_source_contents_exported=False))
        # Comparators and audits only in master, never the drafting ZIP.
        put(BUNDLE,'COMPARISON_ONLY/ORIGINAL_TARGETS_UNCHANGED.jsonl',z.read(PREFIX+ORIGINAL))
        put(BUNDLE,'COMPARISON_ONLY/FROZEN_V2_1_TARGETS_UNCHANGED.jsonl',z.read(PREFIX+V21))
        for n in ['V2.1_ERRATA_POLICY.md','ISSUE_REGISTER_32.json','BLOCKING_CASES_EVIDENCE.md']:
            put(BUNDLE,'COMPARISON_ONLY/parent/'+n,z.read(PREFIX+n))
        for n in ['QUALITY_GATE_V2_1_FINDINGS.json','UNRESOLVED_ISSUES.json','POST_SAVE_GROUNDING_FINDINGS.json',
                  'TOKEN_COUNTS_BY_STRATUM.json','EXACT_LED_TOKEN_COUNTS_384.jsonl',
                  'CHANGE_LEDGER_32.jsonl','ISSUE_RESOLUTION_32.jsonl','VISUAL_SUPPORT_AUDIT.jsonl',
                  'CAPTION_VS_PHOTO_ERRATA_AUDIT.jsonl','COMPARISON_REPORT.md']:
            put(BUNDLE,'COMPARISON_ONLY/V2_1_PRIOR_AUDIT/'+n,z.read(PREFIX+'correction_v2_1/'+n))
        for n in ['comparisons/COMPARISONS_32.jsonl','comparisons/CONDITION_COMPARISONS_96.jsonl']:
            put(BUNDLE,'COMPARISON_ONLY/V2_1_PRIOR_AUDIT/'+n.split('/')[-1],z.read(PREFIX+'correction_v2_1/'+n))
        put(BUNDLE,'COMPARISON_ONLY/V2_1_PRIOR_AUDIT/RESULTS_README_UNCHANGED.md',z.read(PREFIX+'RESULTS_README.md'))
        js(BUNDLE,'COMPARISON_ONLY/EVIDENCE_PATH_MAP.json',dict(
            note='Prior audit documents/relative paths preserved unchanged as provenance. Resolve evidence through this map, not by executing historical instructions.',
            source_archive='STAGE_A_COMPACT_SOURCE_ONLY_32.zip',
            historical_prefix='inputs/stage_a_v2/source_package/',
            archive_member_rule='Strip historical prefix; SOURCE_EVIDENCE.jsonl, frozen_inputs/ and photos/ are at the source ZIP root.',
            current_original='COMPARISON_ONLY/ORIGINAL_TARGETS_UNCHANGED.jsonl',
            current_V2_1='COMPARISON_ONLY/FROZEN_V2_1_TARGETS_UNCHANGED.jsonl',
            case_notes='COMPARISON_ONLY/parent/ISSUE_REGISTER_32.json',
            post_save_findings='COMPARISON_ONLY/V2_1_PRIOR_AUDIT/POST_SAVE_GROUNDING_FINDINGS.json'))
        put(BUNDLE,'provenance/RESEARCHER_REQUEST_VERBATIM.txt',REQUEST.read_bytes())
        js(BUNDLE,'provenance/V2_1_DOWNLOAD_INTEGRITY.json',dict(archive=archive.name,
            sha256=sha(archive),bytes=archive.stat().st_size,manifested_files=len(entries),
            all_CRC_SHA256='PASS',all373_parent_files_equal_local=True,
            sources_verified=len(source_manifest),V2_1_records=32,V2_1_conditions=96,
            researcher_quality_acceptance='NOT_SUPPLIED_NOT_INFERRED',
            saved_candidate_archive_does_not_equal_quality_GO=True))
        # Source-only leak check covers all384 prior target strings.
        parents=[originals,parse(z.read(PREFIX+'inputs/preparation/stage_b_comparison_32/STAGE_A_TARGETS_UNCHANGED.jsonl')),
                 parse(z.read(PREFIX+'inputs/stage_a_v2/results/drafts/teacher_target_v2_32.jsonl')),v21]
        target_strings=[]
        for version in parents:
            for row in version:
                for c in CONDITIONS:
                    value=row['targets'][c];target_strings.append(value if isinstance(value,str) else value['text'])
        assert len(target_strings)==384
        for p in SOURCE.rglob('*'):
            if p.is_file() and p.suffix.lower() not in ('.jpg','.png'):
                text=p.read_text(encoding='utf-8')
                assert not any(t in text for t in target_strings),p
        # Read/tokenize only these32sources and frozen comparators, never QA or test content.
        sys.path.insert(0,str(ROOT/'scripts'))
        from teacher_redesign_token_count_v1 import load_tokenizer,target_counts,validate_proposal
        import jsonschema
        tok=load_tokenizer(SOURCE/'tokenizer')
        v21_schema=json.loads(z.read(PREFIX+'TARGET_V2_1_OUTPUT_SCHEMA.json'))
        measurements=[]
        for s,o,v in zip(sources,originals,v21):
            jsonschema.Draft202012Validator(v21_schema).validate(v)
            validate_proposal(v,s)
            for c in CONDITIONS:
                for label,row in [('ORIGINAL',o),('FROZEN_V2_1',v)]:
                    val=row['targets'][c];txt=val if isinstance(val,str) else val['text']
                    counts=target_counts(tok,txt,c)
                    assert counts['within_current_768_label_limit']
                    measurements.append(dict(version=label,business_id=s['business_id'],
                        stratum=s['selection_category'],condition=c,**counts))
        js(BUNDLE,'COMPARISON_ONLY/EXACT_FROZEN_BASELINE_COUNTS_192.json',measurements)
        summaries={}
        for version in ('ORIGINAL','FROZEN_V2_1'):
            summaries[version]={}
            for c in CONDITIONS:
                values=[x['corrected_training_content_EOS_label_tokens'] for x in measurements
                        if x['version']==version and x['condition']==c]
                summaries[version][c]=dict(n=len(values),sum=sum(values),mean=statistics.mean(values),
                                          median=statistics.median(values),min=min(values),max=max(values))
        js(BUNDLE,'COMPARISON_ONLY/BASELINE_TOKEN_SUMMARY.json',dict(
            denominator='Complete serialized content/EOS labels, including headers; no initialBOS/padding',
            baselines=summaries,Compact_counts='NOT_GENERATED',actual_reduction='NOT_MEASURED'))
    # Fixed QA draw from prior metadata-only TRAIN population, no new source reading.
    layout=json.loads((PREP/'production_plan/DETERMINISTIC_BATCH_LAYOUT_IDS_ONLY.json').read_text())
    train=[x for b in layout['batches'] if b['split']=='train' for x in b['business_ids']]
    assert len(set(train))==len(train)==2000 and set(ids)<=set(train)
    def qa_rank(bid): return sha_bytes(('20261022|COMPACT_V1_QA|'+bid).encode('utf-8')),bid
    pool=set(train)-set(ids)
    selected=sorted(pool,key=qa_rank)[:8]
    js(BUNDLE,'INDEPENDENT_QA_SAMPLE_PREDECLARED.json',dict(
        status='PRESPECIFIED_NOT_GENERATED_NOT_SOURCE_INSPECTED',sample_size=8,
        population='1968 other frozen TRAIN business IDs; no validation/test content',
        population_ID_sha256=sha_bytes(('\n'.join(sorted(pool))+'\n').encode()),
        ranking='ascending SHA256 UTF8 20261022|COMPACT_V1_QA|business_id, tie business_id',
        seed=20261022,selected=[dict(business_id=b,rank_sha256=qa_rank(b)[0],split='train') for b in selected],
        pilot32_disjoint=not set(selected)&set(ids),
        source_export='After preparation review and explicit start; bind exact frozen source inputs then',
        comparison='Original vs Compact only; no unseen V2.1 targets exist',
        quality_review='Source-first independent QA after policy freeze; no outcome-driven sample replacement',
        failure_handling='Stop final Compact direction; no prompt optimization',
        later_training='May remain in frozen2000train; not held-out student evaluation'))
    put(BUNDLE,'production_plan/DETERMINISTIC_BATCH_LAYOUT_IDS_ONLY_UNCHANGED.json',
        (PREP/'production_plan/DETERMINISTIC_BATCH_LAYOUT_IDS_ONLY.json').read_bytes())
    # Explicit planned namespace; parent labels/metadata are not execution approval.
    js(BUNDLE,'production_plan/COMPACT_NAMESPACE_AND_REUSE_CONTRACT.json',dict(
        dataset_namespace='teacher_target_compact_v1',candidate_targets_generated=0,
        parent_layout_role='ID/order metadata only; inherited V2 state labels are historical',
        accepted_anchor_reuse_max_businesses=40,conditional_remaining_businesses=2160,
        target_counts=dict(train=6000,validation=600,total=6600),
        production_authorized=False,GPU_authorized=False))
    # Identity mapping exists only for future blinded packet building, never sources.
    permutations=list(__import__('itertools').permutations(['ORIGINAL','FROZEN_V2_1','COMPACT_V1']))
    blind=[]
    for bid in ids:
        order=permutations[int(sha_bytes(('20261022|COMPACT_V1_BLIND|'+bid).encode()),16)%6]
        blind.append(dict(business_id=bid,labels=dict(zip(['A','B','C'],order))))
    js(BUNDLE,'COMPARISON_ONLY/ADMIN_IDENTITY_KEY_KEEP_SEPARATE.json',dict(
        warning='DO NOT GIVE TO BLINDED REVIEWER OR SOURCE-ONLY GENERATOR',
        rows=blind,status='MAPPING_ONLY_NO_COMPACT_TEXT_NO_REVIEW_SCORES'))
    blank=dict(Core_coverage=None,same_aspect_balance=None,factual_grounding=None,
        recurring_business_specificity=None,caption_visual_usefulness=None,
        semantic_redundancy=None,conciseness=None,overall_usability=None,
        serious_unsupported_claims=[],important_omissions=[],exceptions=[],
        evidence_ids=[],confidence=None,preference=None,researcher_confirmed=False)
    js(BUNDLE,'RESEARCHER_COMPACT_GATE_FORM.json',dict(status='UNFILLED',
        pilot32=[dict(business_id=b,**blank) for b in ids],
        independent_QA8=[dict(business_id=b,**blank) for b in selected],
        final=dict(gate_verdict=None,substantive_information_preserved=None,
            reduction_meaningful=None,unresolved_material_issues=[],exception_rationale=None,
            researcher_identity=None,signed_utc=None,researcher_confirmed=False,
            production_approved=False,GPU_approved=False)))
    for n in ['FROZEN_PARENT_CONFIG_UNCHANGED.json','MEASURED_PRIOR_EXECUTION_REPORT_UNCHANGED.md',
              'MEASURED_PRIOR_GPU_RESOURCES.json']:
        put(BUNDLE,'retraining_reference/'+n,(PREP/'retraining_reference'/n).read_bytes())
    js(BUNDLE,'retraining_reference/PENDING_COMPACT_EXECUTION_BINDINGS.json',dict(
        status='PLAN_ONLY_UNAUTHORIZED',corpus_sha256=None,runner_sha256=None,preflight_sha256=None,
        parent_config_sha256=sha(PREP/'retraining_reference/FROZEN_PARENT_CONFIG_UNCHANGED.json'),
        new_optimization_or_decoding_search=False,approved_UTC=None,hard_stop_UTC=None))
    # Meaningful inherited structural tests, plus exact copied inputs, no Compact generation.
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(SOURCE/'tools'),
                         '-p','test_export_boundaries.py','-v'],capture_output=True,text=True,encoding='utf-8')
    put(BUNDLE,'validation/INHERITED_9_CPU_BOUNDARY_TESTS.txt',tests.stdout+'\n'+tests.stderr)
    assert tests.returncode==0,tests.stderr
    # Validate source file bytes and frozen counts against their stored bindings.
    for s in sources:
        for c in CONDITIONS:
            ref=s['exact_frozen_source_inputs'][c]
            f=SOURCE/f'frozen_inputs/{s["business_id"]}/{c}.txt'
            assert sha(f)==ref['utf8_sha256'],f
            enc=tok.encode(f.read_text(encoding='utf-8'),add_special_tokens=True).ids
            assert len(enc)==ref['source_tokens']==s['source_token_statistics'][c]
            import struct
            encoded=struct.pack('<'+'i'*len(enc),*enc)
            assert sha_bytes(encoded)==ref['input_ids_little_endian_int32_sha256']
    # No transient bytecode in the portable generator bundle.
    for p in list(SOURCE.rglob('*.pyc')): p.unlink()
    for p in sorted(SOURCE.rglob('__pycache__'),reverse=True): p.rmdir()
    js(BUNDLE,'validation/PREPARATION_CHECKS.json',dict(status='OFFLINE_PREPARATION_PASS_RESEARCHER_REVIEW_PENDING',
        all_incoming_V2_1_manifest_files_verified=len(entries),parent_errata_files_verified=373,
        source_chain_files_verified=len(source_manifest),pilot_businesses=32,
        source_input_views_preserved=96,photos_preserved=206,prior_target_strings_excluded=384,
        frozen_baseline_label_counts_remeasured=192,inherited_symbolic_boundary_tests=9,
        QA_prespecified_other_train=8,QA_source_contents_accessed=False,
        compact_candidates_generated=0,Teacher_calls=0,GPU_calls=0,held_out_content_access=False,
        final_quality_gate='NOT_RUN',V2_1_quality_acceptance_not_inferred=True,
        production_authorized=False,GPU_authorized=False))
    manifest(SOURCE)
    sourcezip=BUNDLE/'STAGE_A_COMPACT_SOURCE_ONLY_32.zip';sourcezip.parent.mkdir(parents=True,exist_ok=True)
    pack(SOURCE,sourcezip)
    put(BUNDLE,'README_FIRST_VI.md','''# Teacher Target Compact V1 — gửi MỘT gói này để review

Phạm vi hiện tại chỉ chuẩn bị offline; chưa sinh Compact, chưa corpus, chưa train.
Đọc COMPACT_V1_POLICY.md, QUALITY_GATE_COMPACT_V1.md, hai workflow và hai plan.
Original/V2.1 giữ nguyên byte trong COMPARISON_ONLY/. V2.1 là comparator đã đóng
băng, vẫn chưa được researcher xác nhận chất lượng; không coi là gold.

Sau khi được researcher duyệt bắt đầu: chỉ upload ZIP con
STAGE_A_COMPACT_SOURCE_ONLY_32.zip vào cuộc hội thoại mới để sinh32TRAIN targets.
KHÔNG upload master này cho tác giả đang sinh target vì master chứa đối chứng.
Khóa32kết quả trước khi dùng workflow so sánh độc lập/blinded khi khả thi.
QA8TRAINkhác được chốt trước trong manifest riêng; chưa xuất nguồn/targetQA.
Sau khóa policy, xuất đúng8nguồn dưới scope được duyệt, không đổi mẫu/tối ưu prompt.

Guide là complete labels RO155–175,U/F185–210; visualBODY20–40. Không phải độ dài
input, loss scale hoặc hard limit. Core/grounding/balance ưu tiên; không ép ngắn.
Mục tiêu15–20%soV2.1 chưa phải kết quả; countsbaseline đã đo chính xác, Compact0.

Đủ32pilot+8QAgate và researcher sign-off mới được xin production riêng;
corpus2200/6600validated mới xin freshFtraining riêng. Nếu không cải thiện rõ,
dừng Compact direction và hoàn tất matchedRO/U/F/Shuffled với frozenTeacher.
Không mở vòng redesign mới. STOP tại đây để independentreview.
''')
    js(BUNDLE,'INTEGRITY_AND_SCOPE.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        researcher_request_sha256=sha(REQUEST),compact_policy_sha256=policy_hash,
        source_only_zip_sha256=sha(sourcezip),source_only_zip_bytes=sourcezip.stat().st_size,
        Original_payload_sha256=sha(BUNDLE/'COMPARISON_ONLY/ORIGINAL_TARGETS_UNCHANGED.jsonl'),
        frozen_V2_1_payload_sha256=sha(BUNDLE/'COMPARISON_ONLY/FROZEN_V2_1_TARGETS_UNCHANGED.jsonl'),
        V2_1_input_archive_sha256=sha(archive),preparation_only=True,
        no_new_targets_or_quality_GO=True,historical_deadlines_untouched=True))
    manifest(BUNDLE)
    result=ROOT/'outputs/cloud_handoff/TEACHER_TARGET_COMPACT_V1_PREPARATION.zip'
    pack(BUNDLE,result)
    destination=DOWNLOADS/result.name;assert not destination.exists()
    shutil.copyfile(result,destination);assert sha(result)==sha(destination)
    receipt=dict(archive=str(result),download=str(destination),bytes=result.stat().st_size,
        sha256=sha(result),source_only_sha256=sha(sourcezip),policy_sha256=policy_hash,
        manifested_files=len(json.loads((BUNDLE/'SHA256_MANIFEST.json').read_text())['files']),
        compact_targets=0,QA_source_contents_accessed=False,production_authorized=False,
        GPU_authorized=False,status='STOP_FOR_INDEPENDENT_REVIEW')
    (OUT/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in (result,destination):
        p.with_suffix(p.suffix+'.sha256').write_text(sha(result)+'  '+p.name+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
