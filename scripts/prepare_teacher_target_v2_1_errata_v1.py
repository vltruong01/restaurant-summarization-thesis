"""Bounded offline errata export. No target generation, edits, API or model calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/teacher_target_v2_1_errata_v1'
BUNDLE = OUT / 'bundle'
DOWNLOADS = Path.home() / 'Downloads'
PREP = ROOT / 'outputs/stage_i/teacher_target_v2_preparation_v1/bundle'
S = 'inputs/stage_a_v2/source_package/'
V = 'inputs/stage_a_v2/results/drafts/teacher_target_v2_32.jsonl'
O = 'inputs/preparation/stage_b_comparison_32/ORIGINAL_TARGETS_UNCHANGED.jsonl'
A = 'inputs/preparation/stage_b_comparison_32/STAGE_A_TARGETS_UNCHANGED.jsonl'
POLICY_ID = 'TEACHER_TARGET_V2_1_BOUNDED_ERRATA_V1'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def put(rel, data):
    p = BUNDLE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(data if isinstance(data, bytes) else data.encode('utf-8'))


def js(rel, data):
    put(rel, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def verify_zip(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    assert len(names) == len(set(names)) and z.testzip() is None
    for n in names:
        p = PurePosixPath(n)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n
    manifest = json.loads(z.read('PORTABLE_SHA256_MANIFEST.json'))['files']
    assert set(manifest) == set(names) - {'PORTABLE_SHA256_MANIFEST.json'}
    for n, h in manifest.items():
        assert sha_bytes(z.read(n)) == h, n
    return z, dict(archive=path.name, bytes=path.stat().st_size, sha256=sha(path),
                   manifested_files=len(manifest), CRC='PASS', SHA256='PASS')


POLICY = '''# V2.1 errata policy — minimal delta to V2

ID: TEACHER_TARGET_V2_1_BOUNDED_ERRATA_V1. Status: PREPARATION ONLY.
Parent: preserved TEACHER_TARGET_V2_EVIDENCE_DENSE_V1, SHA256
6622a743194e66dac7d6266619e6ddcfe42748ea4738cbb911b0b7f74eaae17a.
Read the unchanged parent at inputs/preparation/TEACHER_TARGET_V2_POLICY.md.
This delta takes precedence only on the clarifications below. No V2.1 drafts
exist; preparing this document does not authorize applying corrections.

1. Preserve both supported sides of an important SAME dish/aspect. Before deleting
   one side, check all selected accounts of that item. Pat O'Brien's shrimp/grits
   needs positive accounts REV_2/8 and dissent REV_14 together; retain that REV_14
   also praises the grits. Queso heat cannot be reassigned to salsa. Count distinct
   accounts/events accurately; two late-hours requests are not a larger consensus.
2. Separate caption support from photo verification. A literal BLIP-2 caption is
   model-visible evidence, not verified object/action truth. For each visual claim,
   record its condition caption ID/quote/photo and photo-check status in metadata:
   consistent / uncertain / contradicted / not_checked, with rationale. These are
   provisional reviewer judgments, never researcher-confirmed by default. A photo
   may veto a detail or support a less specific entailment of the SAME caption.
   It cannot supply new names, objects, counts, actions or relations absent from
   that condition's caption. Never replace Gus's beans with a photo-only side name.
   A photo-only finding may appear in the audit, never the model target.
3. Preserve safe partial caption information when useful. Removing a dubious
   modifier is preferable to discarding an entire useful food/setting scene if
   the remaining proposition is both caption-entailed and visually defensible.
   Jollibee F1 supports fried chicken in a generic container; do not import a
   photo-only bucket identity or dubious separate cup. Top Out U2/F2 supports a
   large gym without the unverified 'lot of people'. These are correction rules,
   not pre-approved replacement targets. An uncertain photo check stays unresolved.
4. Do not invent causes. Mystic Fish REV_29 reports a table opening; it does not
   report a cancellation. Temporal sequence, mechanism, diagnosis and intention
   require source support; plausible explanations are not summary evidence.
5. Keep classification/process language in metadata. The Torch-Wood phrase 'this
   is a serious isolated firsthand account' must not appear in a proposed summary.
   An included serious report should read as an attributed customer report, without
   declaring proven injury, general danger or a verified cause. Its salience must
   be justified against repeated smoothie praise; attribution alone is insufficient.
6. Resolve documented support/salience concerns, not a quota of missing facts.
   Use ISSUE_REGISTER_32.json and BLOCKING_CASES_EVIDENCE.md. Evaluate recurring,
   distinctive and practical details before generic praise or repeated scenery.
   Minor omissions may remain with explicit source-based adjudication. Do not
   force every dish/complaint into the target, maximize counts, or prefer brevity.
   Keep V2's existing supported gains and check all 32 for unintended regressions.

Unchanged: same 32 TRAIN businesses and all original selected reviews, captions,
photos, model inputs, order, tokenizer and hard 768-label limit. One canonical
review from reviews ONLY, byte-identical in RO/U/F; independent U/F caption scopes;
RO no visual; zero captions => empty visual body/claims. No photo-derived review
facts. Guidance 170–260 review-body / 20–70 visual-body / 200–330 full U/F corrected
labels remains flexible, not a shortening objective or automatic pass/fail.
No change to student architecture, training, decoder policy or hyperparameters.

Original, Stage A and V2 records remain immutable. Any future candidate is saved
in a separate V2.1 namespace with a per-change ledger; unchanged text stays exact.
No candidate shopping, new policy loop or semantic retry. If the final gate cannot
be justified, revert the proposed direction to frozen existing Teacher supervision
and finish the established matched RO/U/F/Shuffled evaluation.
'''

PROMPT = '''# GPT Work: ONE bounded V2.1 correction — version 1

CURRENT STATUS: preparation only. Do not execute this workflow until the researcher
explicitly approves the errata package and starts the fixed-32 correction pass.
This ZIP is an evidence-linked correction input, not an independent/blinded prompt.
Do not claim model/thinking settings were verified unless the runtime proves them.

After explicit authorization:
1. Verify SHA256_MANIFEST.json and INPUT_BINDINGS.json; read V2.1_ERRATA_POLICY.md,
   parent V2 policy, BLOCKING_CASES_EVIDENCE.md, ISSUE_REGISTER_32.json and the final
   gate. Read all frozen selected reviews and condition captions for each case.
   Inspect linked original images for visual checks only. Review prior Stage B
   assertions against sources; do not treat their provisional labels as gold.
2. Use exactly the same 32 TRAIN IDs/order, eight batches of four. ONE correction
   pass over existing V2; no resampling, new examples or alternative candidates.
   Restrict changes to evidence-supported errata and adjudicated salience issues.
   No-change decisions are allowed. Keep valid unrelated claims verbatim. Review
   all 32, including the 11 provisionally preferred families, for regression.
3. Save new records at correction_v2_1/drafts/teacher_target_v2_1_32.jsonl using
   TARGET_V2_1_OUTPUT_SCHEMA.json (same payload structure as V2; only policy/status
   constants change). Keep exact full record bytes of all parents untouched.
   Canonical review and review_claims must agree; regenerate the three serialized
   candidate target strings consistently from that ONE revised canonical body.
   Do not silently mix Original/A/V2 wording or label versions.
4. Alongside candidates save CHANGE_LEDGER_32.jsonl, one row per business:
   business_id, parent_v2_record_utf8_sha256 (exact original JSONL line without LF),
   parent_v2_target_sha256 per condition, outcome CHANGED/UNCHANGED/UNRESOLVED,
   and changes [{issue_id, condition_or_canonical_review, exact_old_span,
   exact_new_span, review_or_caption_evidence_ids, literal_quotes, rationale}].
   For deletions use empty new span; insertions empty old span plus a location.
   Each span must match the corresponding parent/candidate text, not a paraphrase.
   Save per-claim photo checks separately in VISUAL_SUPPORT_AUDIT.jsonl:
   business_id, condition, claim_id, caption_evidence_id, caption_quote, photo_id,
   candidate_span, photo_check_status, rationale, researcher_confirmed=false.
   Distinguish caption-entailed from photo-checked; a photo cannot expand content.
   Unchanged records still bind their V2 parents and receive a no-change rationale.
5. Preserve all raw batch outputs/attempts, candidate hashes, source bindings,
   errata-policy/prompt/schema hashes and unresolved flags. Do not overwrite a raw
   output or choose a better-sounding second candidate. Only incomplete/malformed
   business records may have at most two extra technical attempts under identical
   instructions. A semantic failure is not permission to regenerate; stop/report.
6. Run the inherited CPU mechanical/count utility against the NEW schema:
   python inputs/stage_a_v2/source_package/tools/validate_teacher_target_v2_acceptance_v1.py \\
     --drafts correction_v2_1/drafts/teacher_target_v2_1_32.jsonl \\
     --sources inputs/stage_a_v2/source_package/SOURCE_EVIDENCE.jsonl \\
     --schema TARGET_V2_1_OUTPUT_SCHEMA.json \\
     --tokenizer inputs/stage_a_v2/source_package/tokenizer \\
     --output correction_v2_1/MECHANICAL_AND_COUNTS.json
   This TRAIN32 utility supports the supplied schema but does NOT verify change
   ledgers, photo entailment, salience or researcher approval. Check those separately
   against the final gate. Never truncate/repair to fit a guidance range. If exact
   dependencies are unavailable report NOT_MEASURED, never guess token counts.
7. Return all 32 candidates/96 targets, ledger, raw outputs, literal evidence,
   photo-check audit, exact counts and unresolved findings in one versioned result
   ZIP with SHA256. Compare against preserved V2 AND Original/A, separating review,
   visual and full family; preserve ties and inconclusive. Prefer an independent
   researcher for semantic adjudication. Author self-review remains provisional.

Stop after this pass for researcher review. Do not fill the researcher's form,
produce 6600 targets, train, use held-out content, alter sources/tokenizers, or
start V2.2/another Teacher optimization cycle. If important support/salience remains
unresolved or substantive benefit cannot be justified, recommend frozen existing
Teacher supervision and completion of matched RO/U/F/Shuffled experiments.
'''

GATE = '''# Final quality gate — fixed 32 TRAIN businesses

State: NOT_RUN_FOR_V2_1. V2.1 candidates=0. Researcher adjudication pending.
The prior V2 mechanical PASS and 11/32 provisional full-family preferences are
context, not acceptance. The comparison author also authored V2; no blind or
independent-human status is inferred. 96 conditions are dependent within 32 cases.
Report 24 random cases and four targeted strata of two separately. Repeated source
and development exposure limits generalization; this is a targeted correction.

Technical acceptance for EVERY candidate/condition:
- exact 32 IDs/order/source hashes and 96 complete targets; Original/A/V2 bytes
  unchanged; all new records/attempts/ledgers use a separate append-only namespace;
- schema/policy/prompt lineage; correct evidence IDs/literal quotes; review-only
  canonical body shared exactly across RO/U/F; no U-to-F caption leakage;
- visual claim grounded in its condition caption, one-photo relation only; photo
  verification status recorded separately. No photo-only target information;
- RO has no visual assertions; both zero-F cases keep an empty visual section;
- exact supplied-tokenizer counts, content/EOS labels, ceiling 768, no truncation.
  Flexible guidance deviations warn only. No preference for shorter outputs.
- ledger spans/hashes match parents/candidates; claimed unchanged targets compare
  byte-exactly; no semantic retry or silent repair.

Source-quality acceptance requires actual independent researcher adjudication:
1. Resolve all 13 enumerated Stage B issue entries, the six named priority cases
   (Pat/Gus/Mystic/Torch/Jollibee/Top Out) and all additional support/salience queues
   in ISSUE_REGISTER_32.json. A withdrawn photo doubt must not be reclassified as
   a confirmed error without new evidence. Review all other cases for regression.
2. No unresolved material unsupported claim/cause, misleading same-aspect consensus,
   important Core regression, cross-photo relation, zero-caption violation or
   photo-only substitution. Source citation alone is not entailment proof.
3. Preserve supported business-specific gains, opinion asymmetry, useful visuals,
   recurrence precision and appropriate attribution; keep classification outside
   summary prose. For serious isolated reports adjudicate both support and salience.
4. Judge specificity, balanced synthesis, factual/visual grounding, semantic
   repetition, conciseness, completeness and usability together against Original,
   A and V2. No forced inclusion of every minor dish; retained minor tradeoffs need
   an explicit rationale and confidence. No automatic win-rate or token/loss GO.
5. Record substantive net benefit over the existing frozen supervision, including
   remaining risks. Ties/inconclusive self-review preferences are not a pass.

RESEARCHER_ADJUDICATION_FINAL.json supplies blank per-issue/per-case fields:
support verdict, evidence IDs, importance/severity, resolution accept/change/reject/
inconclusive, omitted-detail rationale, no-regression decision, overall preference,
confidence, researcher identity/time, confirmation and signatures. No values are
pre-filled as human labels. Core panels for these TRAIN cases are provisional,
not D-053 approved validation Core labels.

Outcome branches:
- Defensible PASS with actual researcher sign-off: freeze exact policy/prompt/schema,
  accepted candidate/source/count/audit hashes; request separate production scope.
  This does not authorize corpus generation or GPU execution by itself.
- FAIL, mixed or inconclusive with important unresolved concerns: recommend reverting
  to existing frozen Teacher supervision and completing the established matched
  RO/U/F/Shuffled experiments. No V2.2, alternative Teacher policy or retry loop.

Residual risks even on PASS: 32 selected cases are small; targeted overfitting,
self-review bias and subjective salience remain; BLIP-2/photo disagreements may
persist at scale; source-grounded targets do not guarantee better student outputs.
No final V2.1 gate verdict can be justified before candidates and adjudication exist.
'''

PRODUCTION = '''# Production readiness plan — conditional, not executable

Current decision: NOT READY; 6600-target production and LED-large retraining remain
unauthorized. No full-corpus sources or targets are materialized by this package.
This is the final targeted correction preparation, not a policy search.

After final-32 gate PASS and separate explicit researcher production approval:
1. Freeze the accepted V2.1 policy/prompt/schema, all accepted 32 records, change
   ledger, photo audit, exact counts and source hashes. Bind production authorization
   to these and the frozen split/input/caption/photo/tokenizer manifests. Record
   actual authorization time/scope. Historical deadlines remain unchanged.
2. Population remains exactly 2000 TRAIN + 200 validation businesses; no held-out
   content or test selection. Reuse existing deterministic ID-only layout (550
   batches of four: first eight pilot32, remaining train IDs sorted, then val IDs
   sorted). Preserve downstream training order, selected reviews/captions and input
   arrays byte-for-byte. No source changes prompted by a target quality problem.
3. For new businesses use the one frozen combined V2+V2.1 policy and source-only
   drafting prompt/schema, NOT case-specific answers, old targets or student outputs.
   The correction prompt is pilot-only; a production source-only rendering with
   identical substantive rules must be frozen and approved before generation.
   Use original images only to veto/verify caption-derived propositions, never to
   add photo-only evidence. Record photo checks, uncertain claims and omitted-safe-
   detail rationales. No unreviewed BLIP-2 identity is labeled verified.
4. Reuse pilot96 targets only if explicitly accepted under exactly the frozen
   combined policy and source bindings, with per-record lineage. 2168 businesses/
   6504 condition targets then remain. No silent mixing with Original/A/old V2.
   If a pilot unchanged text is retained, it is explicitly accepted into V2.1,
   retaining both parent and accepted hashes, rather than relabeled without audit.
5. Append-only raw attempts, source manifests and completion/technical/quality
   states. One immutable accepted family per business; identical retries are
   idempotent, conflicting duplicates stop. At most two extra technical attempts
   for incomplete/malformed records; no retry for nicer wording or semantic failure.
   After cap, conflict or material semantic defect stop/report for adjudication.
6. Validate all 2200 families/6600 targets: exact membership, split/order, one
   canonical review, U/F caption/photo scopes, RO/zero-caption behavior, literal
   evidence, input hashes, no edits/truncation and exact content/EOS counts <=768.
   The supplied mechanical utility is TRAIN32-only: it cannot certify full corpus
   or validation records. Adapt its bounds/split checks only after production scope
   approval, keeping the same rules; test train/val membership and test exclusion.
   Freeze six condition/split JSONLs and separate new packed labels; verify original
   input IDs/masks/global attention/order unchanged. Original Stage H stays intact.
7. Predeclare source-quality QA before production: all 32 pilot anchors, every
   flagged support/attribution/same-aspect/zero-caption problem, plus deterministic
   50 NEW train and 10 validation businesses ranked by SHA256 UTF8
   '20261022|V2_CORPUS_AUDIT|business_id' (exclude pilot32 from train sample).
   Use the inherited fixed seed, no outcome-based sampling. Inspect all selected
   sources/captions/photos and all RO/U/F for each audited business; record blinded
   identity where practical, reviewer independence and case-level verdicts. Include
   all zero-caption families in compliance checks. A material issue blocks corpus
   release; do not silently repair/regenerate or launch another policy cycle.
   Sampling cannot establish all-corpus human truth; disclose additional validation
   exposure. Report counts and severity by split/stratum, not inflated condition N.
8. Full integrity PASS plus explicit researcher corpus QA/sign-off and complete
   backup precede ANY fresh-F execution proposal. Future fresh F retains the
   successful corrected-label LED-large backbone/fresh Expanded-LoRA, fixed data,
   numerical policy, 5 epochs/1250 successful updates, epoch3/5 fixed24 generation
   protocol and approved references. Separate hash-bound GPU authorization needed;
   no continuing old adapter. Quality benefit must be substantive, not lower loss.
   Only confirmed F gain can justify a separately authorized matched RO/U proposal;
   Shuffled reuses matched F. No mixing target versions across the matched family.

If final gate or corpus quality cannot be justified, recommend the frozen existing
Teacher supervision and established matched RO/U/F/Shuffled thesis evaluation.
Keep historical LED-base family primary until a complete matched replacement exists.
No Teacher/API/GPU/held-out execution occurs in readiness preparation.
'''

# Rules are instructions for a future pass, never replacement drafts.
RULES = {
 2: 'Adjudicate vegan-specific review detail and food-platter F5 usefulness versus repeated room context; restore a useful safe caption subset only if it merits space.',
 3: 'Coffee identity in F2 is uncertain: retain cake/cup at caption-entailed specificity without guessing beverage contents. Keep F3 separate. Assess garlic-wing dissent and lunch-time constraint by source salience.',
 4: 'Preserve positive shrimp/grits REV_2/8 together with REV_14 dissent (including favorable grits qualifier). Do not replace this contrast with unrelated dish praise. Judge bathroom-access specificity separately.',
 5: 'Remove unsupported cutting action if not defensible; retain dessert/jar as a safe subset of F1. Do not add spoon action from the photo. Preserve Mongolian-beef same-dish contrast.',
 7: 'Adjudicate menu/recovery specifics versus generic moralizing; no need to restore tip/rodent allegations. Preserve online/counter failures and inspect real support rather than length.',
 8: 'Assign heat reports to the exact condiment: queso REV_7; salsa REV_8/13. Do not fabricate disagreement on the same condiment. Adjudicate margarita dissent and practical surcharge separately.',
 9: 'Do not assert disputed sushi identity in F1; retain caption-entailed generic rolled food if useful. Check U4 garnish identity without substituting a photo-only ingredient. Assess ramen/crab-portion caveats.',
 14: 'Separate caption-supported from photo-confirmed: U2/F1 beans is not established. Keep caption-entailed fried chicken if useful; never substitute photo-only side identities. Apply separately to U and F.',
 15: 'Do not assert uncertain counter location in U3; a safe food/tray subset may remain. Adjudicate recurring torta/taco/salsa identity versus isolated eggshell report and omitted dissent/noise.',
 16: 'Adjudicate named appetizer/dessert contrasts, calm/noisy room and repeated takeout calamari against generic division of opinion. Evaluate F3 bar/bottles versus less useful clock/interior without quota-filling.',
 18: 'Assess distinctive U1 buffalo-head display versus generic room text using that exact caption/photo; do not invent wall mounting or animal identity beyond the caption. No forced visual replacement.',
 19: 'For F1 evaluate safe fried-chicken subset despite questionable fries/count/container details; never import photo-only details. Adjudicate dirty-table/floor omission and redundant quick-service phrasing.',
 20: 'Remove process-classification clause from future summary, keep attribution if serious report retained, and justify salience versus repeated smoothie praise REV_12/19. No proven injury/general danger inference.',
 21: 'Adjudicate allergy attentiveness, seafood/cake/menu constraints versus dated BYOB/license/closure information; no current-status assertion or automatic restoration of every dish.',
 22: 'Delete unsupported cancellation cause; source REV_29 establishes only table availability after initial refusal. Preserve small desserts/reservation information. Do not revive withdrawn U2 empty-room doubt.',
 23: 'Evaluate restoration of safe F1 fried-chicken/container content alongside F2 seating; do not carry over dubious cup relation or substitute photo-only bucket identity.',
 24: 'Evaluate large-gym scene from U2 and independently F2 alongside food; remove unsupported many-people count, do not enrich with photo-only equipment/climbing descriptions. Keep sticky-wing consensus.',
 25: 'Keep F visual empty; adjudicate night-harbor/date-night counterexamples versus practical boarding/buffet/seating facts. No return of counterfeit allegation merely to match previous text.',
 26: 'Use two located late-hours reports REV_6/16 or a noninflated attributed formulation; no current opening-hours promise. Judge cold-room constraint and redundant U bowl scenes; F remains empty.',
 30: 'Compare useful food/darts/game/bar caption scenes against redundant patio/room description. Treat duplicated last-call report as one event, not recurrence; no mandatory menu checklist.',
 32: 'Adjudicate omitted Maggi/biryani taste dissent REV_14/16 and Botanical Gin REV_11 versus other Core details; do not force all three in by quota or treat every omission as Core.',
}


def main():
    assert not OUT.exists(), 'Version already exists; no overwrite'
    OUT.mkdir(parents=True)
    incoming = OUT / 'incoming'; incoming.mkdir()
    paths = [DOWNLOADS / ('TEACHER_TARGET_V2_STAGE_' + x + '.zip') for x in
             ('A_32_RESULTS', 'B_COMPARISON_32')]
    za, ra = verify_zip(paths[0]); zb, rb = verify_zip(paths[1])
    assert ra['manifested_files'] == 516 and rb['manifested_files'] == 666
    for p in paths:
        shutil.copyfile(p, incoming / p.name)
        assert sha(p) == sha(incoming / p.name)
    # Cross-package chain, not just the self-supplied manifest.
    assert za.read('results/drafts/teacher_target_v2_32.jsonl') == zb.read(V)
    for n in za.namelist():
        assert zb.read('inputs/stage_a_v2/' + n) == za.read(n), n
    assert zb.read(O) == (PREP/'stage_b_comparison_32/ORIGINAL_TARGETS_UNCHANGED.jsonl').read_bytes()
    assert zb.read(A) == (PREP/'stage_b_comparison_32/STAGE_A_TARGETS_UNCHANGED.jsonl').read_bytes()
    source_manifest = json.loads(zb.read(S+'SHA256_MANIFEST.json'))['files']
    for n, h in source_manifest.items():
        assert sha_bytes(zb.read(S+n)) == h
        assert zb.read(S+n) == (OUT.parent/'teacher_target_v2_preparation_v1/stage_a_sources_32'/n).read_bytes()
    sources = [json.loads(x) for x in zb.read(S+'SOURCE_EVIDENCE.jsonl').decode().splitlines()]
    drafts = [json.loads(x) for x in zb.read(V).decode().splitlines()]
    cases = [json.loads(x) for x in zb.read('review/COMPARISONS_32.jsonl').decode().splitlines()]
    issues = json.loads(zb.read('review/UNRESOLVED_ISSUES.json'))['issues']
    assert len(sources)==len(drafts)==len(cases)==32 and len(issues)==13
    ids = [x['business_id'] for x in sources]
    assert len(set(ids))==32 and ids == [x['business_id'] for x in drafts] == [x['business_id'] for x in cases]
    assert all(x['split']=='train' for x in sources)
    assert Counter(x['overall_three_condition_preference'] for x in cases)==Counter(V2=11,ORIGINAL=1,INCONCLUSIVE=20)
    # One copy of full sources; omit duplicate historical nested ZIPs/authoring scripts.
    for n in zb.namelist():
        if n.startswith(S) or n in (V,O,A) or n.startswith('review/') and '/case_comparisons/' not in n:
            put(n, zb.read(n))
    for n in ['TEACHER_TARGET_V2_POLICY.md','TARGET_V2_OUTPUT_SCHEMA.json','TOKENIZER_AND_VALIDATION_SPEC.md']:
        put('inputs/preparation/'+n, zb.read('inputs/preparation/'+n))
    for n in ['PRODUCTION_GENERATION_PLAN_6600.md','production_plan/DETERMINISTIC_BATCH_LAYOUT_IDS_ONLY.json']:
        put('parent_plan/'+n, (PREP/n).read_bytes())
    put('V2.1_ERRATA_POLICY.md', POLICY)
    policy_hash = sha(BUNDLE/'V2.1_ERRATA_POLICY.md')
    put('GPT_WORK_V2_1_CORRECTION_PROMPT.md', PROMPT)
    put('QUALITY_GATE_FINAL.md', GATE)
    put('PRODUCTION_READINESS_PLAN.md', PRODUCTION)
    schema = json.loads(zb.read('inputs/preparation/TARGET_V2_OUTPUT_SCHEMA.json'))
    schema['$id']='urn:teacher-target-v2-1-bounded-errata-v1'
    schema['properties']['policy_id']['const']=POLICY_ID
    schema['properties']['policy_sha256']['const']=policy_hash
    schema['properties']['status']['const']='PROVISIONAL_GPT_WORK_V2_1_REQUIRES_RESEARCHER_REVIEW'
    js('TARGET_V2_1_OUTPUT_SCHEMA.json', schema)
    evidence_index={}
    for s in sources:
        for r in s['reviews']:
            evidence_index[r['evidence_id']]=dict(text=r['text'],kind='review')
        for cond in ('unfiltered','filtered'):
            for c in s['captions'][cond]:
                evidence_index[c['evidence_id']]=dict(text=c['model_visible_caption'],kind='caption',
                    condition=cond,photo_file=S+c['photo_file'],photo_id=c['photo_id'])
    register=[]
    for i,(s,d,c) in enumerate(zip(sources,drafts,cases),1):
        own=[dict(issue_id='B%02d'%k,**q) for k,q in enumerate(issues,1) if q['case']==i]
        register.append(dict(case=i,business_id=s['business_id'],business_name=s['business_name'],
            stratum=s['selection_category'],prior_AI_family_preference=c['overall_three_condition_preference'],
            prior_preference_is_not_researcher_confirmation=True,enumerated_stage_b_issues=own,
            remaining_V2_omissions_from_self_review=c['missing_or_less_specific_information']['V2'],
            additional_source_links=c.get('additional_omission_source_links',[]),
            prior_review_assessment=c['review_assessment'],
            prior_condition_assessments={k:v.get('assessment') for k,v in c['condition_comparisons'].items()},
            correction_rule=RULES.get(i,'Preserve supported V2 gains. Check reported omissions for salience and all claims for support; no rewrite by default and no invented requirement to include minor details.'),
            disposition='PENDING_RESEARCHER_ADJUDICATION',researcher_confirmed=False))
    js('ISSUE_REGISTER_32.json', dict(status='PREPARATION_ONLY',cases=register,
        note='All 32 covered, including every inconclusive case. Rules are not corrected targets or accepted verdicts.'))
    # Evidence file: full frozen V2 bodies + exact selected source records, no drafted replacement prose.
    md=['# Blocking cases and remaining evidence\n',
        'Status: correction PREPARATION ONLY. Prior Stage B findings remain AI-assisted self-review. '
        'Rules below are not new target drafts. Original/A/V2 JSONL bytes are preserved separately. '
        'All complete sources and 206 original photos are included; no held-out content.\n',
        'Authoritative prior review: [Stage B](review/STAGE_B_REVIEW_VI.md); '
        '[all 32 case reports](review/COMPARISONS_32.jsonl); [withdrawn photo doubts](review/ORIGINAL_RESOLUTION_REVIEW_LOG.json).\n',
        'Photo checks performed locally for eight listed images are provisional observations only. '
        'Gus: fried chicken is visible; beans not established. Jollibee: chicken/container supported, '
        'separate cup not established. Top Out: gym supported, many-people count not established. '
        'Willie: chicken supported; detailed counts/fries need caution. Gina: cake/cup visible; coffee '
        'identity uncertain. P.F.: dessert/jar supported; cutting action uncertain. Fish Bol: rolled '
        'food supported, sushi identity uncertain. Los Potrillos: food/tray visible; counter location '
        'uncertain. Do not substitute photo-only identities in any candidate.\n']
    link_checks=0
    for i in [4,14,22,20,23,24]+[x for x in sorted(RULES) if x not in [4,14,22,20,23,24]]:
        s,d,c=sources[i-1],drafts[i-1],cases[i-1]
        md.extend([f'\n## Case {i:02d} — {s["business_name"]} (`{s["business_id"]}`)\n',
            '**Proposed correction rule (no draft edited):** '+RULES[i]+'\n',
            '**Exact old V2 canonical review, shared across RO/U/F:**\n', '> '+d['canonical_review_consensus']+'\n'])
        for cond in ('unfiltered','filtered'):
            visual=d['targets'][cond]['text'].split('[VISUAL_EVIDENCE]\n',1)[1]
            md.append(f'**Exact old V2 {cond} visual body:**\n\n> '+(visual if visual else '(EMPTY BODY — annotation only)')+'\n')
        md.append('**Prior unresolved omissions (provisional, not a mandatory checklist):**\n\n'+
                  '\n'.join('- '+x for x in c['missing_or_less_specific_information']['V2'])+'\n')
        eids=[]
        for q in issues:
            if q['case']==i:
                md.append('**Stage B issue:** '+q['issue']+'\n')
                eids+=q['evidence_ids']
        for q in c.get('additional_omission_source_links',[]): eids+=q['evidence_ids']
        for q in c.get('information_panel',[]):
            for e in q.get('source_evidence',[]): eids.append(e['evidence_id'])
        # Include every condition caption for visual/salience choices, not just cited winners.
        for cond in ('unfiltered','filtered'): eids += [q['evidence_id'] for q in s['captions'][cond]]
        md.append('**Exact source records:** review excerpts below are FULL selected review bodies; '
                  'caption text is exact model-visible text. Images can veto, not expand, these captions.\n')
        for eid in dict.fromkeys(eids):
            assert eid in evidence_index,eid
            e=evidence_index[eid];link_checks+=1
            md.append(f'\n### `{eid}` ({e["kind"]})\n\n```text\n{e["text"]}\n```\n')
            if e['kind']=='caption': md.append(f'[Original photo `{e["photo_id"]}`]({e["photo_file"]}) — verification pending/provisional, not researcher-confirmed.\n')
    md.append('\n## Other 11 cases\n\nAll remain in ISSUE_REGISTER_32.json and complete sources. '
              'Prior V2 preference does not exempt them from regression/support review. No edits are prescribed by default.\n')
    put('BLOCKING_CASES_EVIDENCE.md','\n'.join(md))
    fields=dict(support_verdict=None,evidence_ids=[],importance_or_severity=None,
                resolution=None,omission_or_tradeoff_rationale=None,confidence=None,
                researcher_confirmed=False,researcher_identity=None,adjudicated_utc=None)
    js('RESEARCHER_ADJUDICATION_FINAL.json',dict(status='UNFILLED',
        issue_rows=[dict(issue_id='B%02d'%i,case=q['case'],business_id=q['business_id'],**fields) for i,q in enumerate(issues,1)],
        case_rows=[dict(case=x['case'],business_id=x['business_id'],**fields,
                       no_regression_verdict=None,review_preference=None,visual_preferences=None,
                       full_family_preference=None,substantive_benefit_vs_frozen_supervision=None) for x in register],
        overall=dict(final_gate_verdict=None,remaining_material_issues=[],rationale=None,
                     researcher_confirmed=False,researcher_identity=None,signed_utc=None,
                     production_execution_approved=False,GPU_execution_approved=False)))
    parent_lines=zb.read(V).splitlines()
    js('INPUT_BINDINGS.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        archives=[ra,rb],parent_source_files_verified=len(source_manifest),
        cross_package_all_stage_a_members_equal=True,
        preserved_payloads={p:sha(BUNDLE/p) for p in (V,O,A,S+'SOURCE_EVIDENCE.jsonl')},
        parent_V2_record_line_sha256={x['business_id']:sha_bytes(line) for x,line in zip(drafts,parent_lines)},
        policy_id=POLICY_ID,policy_sha256=policy_hash,
        versioning='Original/A/V2 untouched; V2.1 policy/schema only, candidate count zero'))
    put('README_FIRST_VI.md', '''# Chỉ cần gửi ZIP này để researcher/GPT review

Gói V2.1 là errata preparation cuối cùng, chưa có target V2.1.
Đọc 5 tài liệu ở thư mục gốc; ISSUE_REGISTER_32.json bao phủ đủ 32 ca,
RESEARCHER_ADJUDICATION_FINAL.json còn trống. BLOCKING_CASES_EVIDENCE.md chứa
nguyên văn V2 và nguồn/caption/ảnh cho 21 ca cần phân xử; toàn bộ 32 nguồn vẫn có.

Hãy review tính tối thiểu của policy delta, đúng bằng chứng, ranh giới caption/ảnh,
workflow một lượt và final gate. Trả readiness cho correction pass, không sinh
target ngay. Chỉ researcher mới được phê duyệt bước áp dụng. Không production,
không train, không held-out. Nếu gate cuối không đủ căn cứ: dùng lại frozen Teacher
supervision và hoàn tất matched RO/U/F/Shuffled, không mở V2.2.

Original, Stage A và V2 được copy byte-exact vào inputs/. review/ giữ nguyên Stage B;
mọi preference/Core panel là provisional. Dữ liệu nguồn dưới
inputs/stage_a_v2/source_package/ có README/prompt V2 cũ: đó là provenance, không
phải lệnh bắt đầu workflow mới. Chỉ dùng prompt V2.1 sau khi được duyệt.
Không cần gửi ZIP Stage A/Stage B cũ cùng gói này. Không có ZIP lồng bên trong.
''')
    js('validation/INPUT_ARCHIVE_VERIFICATION.json',dict(stage_a=ra,stage_b=rb,
        source_chain_files=len(source_manifest),original_A_V2_preserved=True,
        unique_training_businesses=32,condition_comparisons=96,
        prior_preferences=dict(Counter(x['overall_three_condition_preference'] for x in cases)),
        source_evidence_blocks_resolved=link_checks,enumerated_unresolved_issues=13,
        additional_case_rules=len(RULES),no_V2_1_generation=True,
        production_authorized=False,GPU_authorized=False))
    # Real CPU validation of existing, immutable V2; no candidate is generated.
    python=ROOT/'.venv-caption-pilot/Scripts/python.exe'
    cmd=[str(python),str(BUNDLE/(S+'tools/validate_teacher_target_v2_acceptance_v1.py')),
         '--drafts',str(BUNDLE/V),'--sources',str(BUNDLE/(S+'SOURCE_EVIDENCE.jsonl')),
         '--schema',str(BUNDLE/'inputs/preparation/TARGET_V2_OUTPUT_SCHEMA.json'),
         '--tokenizer',str(BUNDLE/(S+'tokenizer')),
         '--output',str(BUNDLE/'validation/PRESERVED_V2_MECHANICAL_RECHECK.json')]
    check=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')
    put('validation/PRESERVED_V2_RECHECK_LOG.txt',check.stdout+'\n'+check.stderr)
    assert check.returncode==0, check.stderr
    tests=subprocess.run([str(python),'-m','unittest','discover','-s',str(BUNDLE/(S+'tools')),
                         '-p','test_*.py','-v'],capture_output=True,text=True,encoding='utf-8')
    put('validation/REUSED_CPU_BOUNDARY_TESTS.txt',tests.stdout+'\n'+tests.stderr)
    assert tests.returncode==0, tests.stderr
    # Do not package runtime-generated bytecode.
    for p in list(BUNDLE.rglob('*.pyc')): p.unlink()
    for p in sorted(BUNDLE.rglob('__pycache__'),reverse=True): p.rmdir()
    # Schema diff restricted to identifiers/constants; verify no hidden new constraints.
    old=json.loads(zb.read('inputs/preparation/TARGET_V2_OUTPUT_SCHEMA.json'))
    restored=json.loads((BUNDLE/'TARGET_V2_1_OUTPUT_SCHEMA.json').read_text(encoding='utf-8'))
    if '$id' in old: restored['$id']=old['$id']
    else: restored.pop('$id')
    for k in ('policy_id','policy_sha256','status'): restored['properties'][k]=old['properties'][k]
    assert restored==old
    # Recheck all portable parent copies and original downloaded archive bytes after preparation.
    for n in zb.namelist():
        if (BUNDLE/n).is_file(): assert (BUNDLE/n).read_bytes()==zb.read(n),n
    assert sha(paths[0])==ra['sha256'] and sha(paths[1])==rb['sha256']
    za.close();zb.close()
    js('validation/PREPARATION_RESULT.json',dict(status='OFFLINE_PREPARATION_COMPLETE_PENDING_RESEARCHER_REVIEW',
        parent_v2_32_records_96_targets_mechanical_recheck='PASS',reused_symbolic_tests=15,
        schema_delta_only_policy_status_and_id=True,source_inputs_and_targets_byte_preserved=True,
        V2_1_candidates=0,new_Teacher_calls=0,GPU_calls=0,held_out_content_access=False,
        final_quality_gate='NOT_RUN',recommendation_now='NOT_READY_FOR_PRODUCTION_OR_RETRAINING',
        fallback_if_gate_unjustified='Existing frozen Teacher supervision and matched RO/U/F/Shuffled completion'))
    js('SHA256_MANIFEST.json',dict(files={p.relative_to(BUNDLE).as_posix():sha(p)
        for p in sorted(BUNDLE.rglob('*')) if p.is_file()}))
    archive=ROOT/'outputs/cloud_handoff/TEACHER_TARGET_V2_1_ERRATA_REVIEW_V1.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(BUNDLE.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(BUNDLE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert set(z.namelist())==set(manifest)|{'SHA256_MANIFEST.json'}
        for n,h in manifest.items(): assert sha_bytes(z.read(n))==h
    target=DOWNLOADS/archive.name
    assert not target.exists()
    shutil.copyfile(archive,target);assert sha(target)==sha(archive)
    receipt=dict(archive=str(archive),download=str(target),bytes=archive.stat().st_size,
                 sha256=sha(archive),manifested_files=len(manifest),CRC_SHA256='PASS',
                 policy_sha256=policy_hash,status='STOP_FOR_RESEARCHER_REVIEW',
                 candidates_generated=0,production_authorized=False,GPU_authorized=False)
    (OUT/'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in [archive,target]:
        p.with_suffix(p.suffix+'.sha256').write_text(sha(archive)+'  '+p.name+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
