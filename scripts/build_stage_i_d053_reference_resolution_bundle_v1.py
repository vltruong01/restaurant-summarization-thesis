"""Export source-only, blocked D053 reference-resolution handoff. No final freeze.

The payload contains approved decisions plus explicitly NONFINAL unit proposals.
It must never be described as ready for scoring or prediction generation.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import html
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE=OUT/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
NAME='stage_i_d053_researcher_reference_resolution_bundle_v1'
DEST=ROOT/'outputs/stage_i'/NAME


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not DEST.exists(), 'Preserve all existing versions'
    status=read(OUT/'REFERENCE_CONSTRUCTION_READINESS.json')
    assert status['final_reference_freeze_created'] is False
    cases=read(SOURCE/'source_cases.json');state=read(OUT/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json')
    corrections=read(OUT/'CORRECTION_CAPTION_PHOTO_AUDIT.json')
    omissions=read(OUT/'POTENTIAL_IMPORTANT_OMISSIONS_FOR_DECISION.json')
    uncertain=read(OUT/'UNCERTAIN_EXCLUDED_FROM_SCORING.json')
    business=[]
    for case in cases:
        bid=case['business_id'];parent_ids={c['unit_id'] for c in map(json.loads,(SOURCE/'candidates_UNVERIFIED.jsonl').read_text().splitlines()) if c['business_id']==bid}
        business.append({'business_id':bid,'subset':case['subset'],'selected_reviews':len(case['reviews']),
            'captions':len(case['captions']),'candidate_decisions':len(parent_ids),
            'recorded_human_approved_scope':True,
            'candidate_labels':dict(collections.Counter(state['candidate_annotations'][cid]['support_label'] for cid in parent_ids)),
            'proposed_span_units':sum(u['business_id']==bid for u in state['units']),
            'caption_scope_conflicts':sum(r['business_id']==bid and r['caption_scope_assessment']=='REQUIRES_RESEARCHER_DECISION' for r in corrections),
            'potential_omission_examples':sum(r['business_id']==bid for r in omissions),
            'final_reference_units':0,'final_completeness_confirmed':False,
            'blocker':'Unit-level atomicity, priorities/metadata, evidence conflicts and any missing information must resolve before final acceptance'})
    write(OUT/'BUSINESS_COMPLETENESS_AUDIT.json',business)
    index=[{'unit_id':u['unit_id'],'business_id':u['business_id'],'parent_candidate_id':u['parent_candidate_id'],
            'domain':u['domain'],'text':u['text'],'exact_parent_wording':u['exact_parent_wording'],
            'evidence':u['evidence'],'caption_scope_conflict':u['caption_scope_conflict'],
            'eligible_for_scoring':False} for u in state['units']]
    write(OUT/'UNIT_EVIDENCE_INDEX_NOT_FINAL.json',index)
    report='''# D-053 researcher-approval import and reference construction audit

Date:2026-10-17 (Asia/Saigon). Status: **REFERENCE FREEZE BLOCKED — RESEARCHER DECISIONS REQUIRED**.
Candidate decisions are approved; a scoring-ready atomic reference set is not yet complete.

## Approval and provenance

The attached handoff matches its two input SHA-256 values. All202 nested source
bundle manifest entries match. The raw edited JSON remains byte-identical with
human_verified=false, blank reviewer identity, no units and no history. It is not
misrepresented as a finalized export.

The user's direct conversation explicitly approves203 Supported labels,21 exact
corrected wordings subject to evidence eligibility, and two Uncertain exclusions.
RESEARCHER_APPROVAL_RECORD.json binds that message, the submitted JSON and source
ZIP hashes. The actual prior inspection time and platform message ID are unknown
and remain null. The recorded timestamp means import/recording time, not a forged
inspection time or browser click. Candidate confirmation snapshots materialize
the explicit chat approval; they are not claims that the original JSON had them.
All226 candidate decisions are recorded without changing labels, original claim
text, corrected_claim_text, existing priorities/informativeness or AI notes.

## Scope and structural integrity

24 fixed validation Filtered businesses:20 random +4 targeted;652 selected full
reviews;90 original photos/captions;226 candidates;1,407 exact evidence bindings.
IDs, source hashes, review/caption order and quote substrings match. All90 photos
were inspected via contact sheets; originals remain available and unchanged.
The overlapping burger photo was reopened: four burgers are visible, so its
Supported count is not incorrectly flagged. No new clear conflict was identified
in the67 Supported visual candidates at this inspection resolution. This is an
AI evidence audit, not independent human ground truth or guaranteed fine-detail
verification of every image.

The single zero-caption business has no visual proposals. The two Uncertain
candidates remain in a separate exclusion file, with no unit links. One uncertain
case has nonblank corrected text; it is still NOT promoted or scored.

## Why final freeze is blocked

1. **15 of21 exact corrections require a caption-scope decision.** They introduce
   or replace at least one detail not supported by the exact LED-visible caption.
   Examples: shrimp from generic seafood; a green bottle where caption says wine
   bottle; beer replacing wine in an adjacent glass; standing replacing sitting;
   barbecue meat/sides replacing generic food. Some are photo-compatible, but
   photo compatibility cannot expand the permitted caption content. These are
   audit flags for researcher adjudication, not silent relabeling/exclusion.
   All21 approved texts remain exact in the approval ledger and construction
   proposals. Six corrections have no new caption fact identified. Fifteen
   common-caption alternatives are proposed separately, NEVER applied.
2. **Unit metadata/atomicity are incomplete.** All226 candidate importance fields
   are null. Candidate importance is optional, but each FINAL unit requires
   core/supporting, aspect, specificity, polarity for review units and explicit
   unit confirmation under the unchanged D049/D053 protocol. Existing candidate
   informativeness is retained; no new human priority or unit judgment is forged.
   The321 exact clause spans are construction proposals, NOT321 finalized atomic
   facts. Parent context, attribution, residual lists and per-unit evidence scope
   still require semantic atomization. Do not use these fragments as a scoring
   denominator. No final atomic unit or reference freeze has been emitted.
3. **Five concrete possible review omissions require a decision.** These include
   loud music impeding conversation; advertised happy-hour refusal/rudeness;
   repeated pickup failures; cold-brew unavailability; and a non-chicken dish
   flavor concern. Full exact REV quotes/IDs and nearby candidates are supplied.
   These were not automatically added. Only the researcher decides importance
   and inclusion.203 reviews are not directly cited by candidates, but uncited
   does NOT mean semantically uncovered. The inventory covers all652 source
   reviews; the five semantic findings are nonexhaustive. Independent semantic
   completeness of the whole55,236-word review pool is not claimed.

Business-level approval of decisions for all24 is preserved. Final-reference
completeness is not silently asserted while these new conflicts/omissions and
unit-level requirements remain. Existing business/global reference flags stay
pending, distinct from the recorded approval of candidate decisions.

## Validation and limits

The unchanged annotation schema accepts the imported working draft, and its
semantic validator reports structurally valid but not ready_for_freeze. Strict
final acceptance and reference_freeze correctly BLOCK. This expected rejection
is not hidden, repaired by relaxing criteria, or counted as a final-reference
PASS. No final freeze is produced under a misleading filename.

Run the original83 tests,30 corrected-UI tests,eight portable-readiness tests and
16 handoff/provenance/guard tests from the review ZIP. Final receipts and exact
counts are supplied beside the ZIP in FINAL_ZIP_TESTS/; in-package tests refer to
their recorded execution, not to a fictional all-references-PASS state.

Original protocols, schema, code, source data, tokenizer settings and both adapter
hashes remain frozen. No Teacher call, target generation, new prediction, training,
held-out access or checkpoint modification occurred. No scoring is authorized.

## Next researcher decisions

Review FINALIZATION_REVIEW.html and RESEARCHER_DECISION_CHECKLIST.md. Resolve the
15 flagged corrections (approve a proposed caption-safe alternative, provide
another supported wording, or explicitly exclude); decide whether to add each
of the five omission examples; then approve a completed atomic-unit/metadata
sheet before freeze. This does NOT reopen the203 approved candidate labels or
the two Uncertain exclusions. No Pilot A/B prediction generation may start yet.
'''
    write(OUT/'ANNOTATION_AND_PROVENANCE_AUDIT_REPORT.md',report)
    checklist='''# D-053 decisions needed before reference freeze

Your203 Supported decisions,21 conditional corrected wordings and two Uncertain
exclusions have been recorded. They do not need to be relabeled just because the
uploaded export stored human_verified=false. No final reference freeze exists.

## A. Fifteen caption-scope conflicts

For each row below, approve the proposed caption-safe alternative, provide a
different caption+photo-supported wording, or explicitly exclude the revision.
No alternative has been applied. Original labels/approved text are preserved.

'''
    for r in corrections:
        if r['caption_scope_assessment']=='REQUIRES_RESEARCHER_DECISION':
            checklist+=f"- `{r['candidate_id']}`\n  - Approved correction: {r['approved_corrected_claim_text']}\n  - Scope issue: {r['reason']}\n  - Proposed alternative (NOT applied): {r['alternative_for_researcher_review_only']}\n\n"
    checklist+='''## B. Important omitted review information

Select include / exclude / needs clarification for each example. Exclusion is a
researcher relevance/importance decision, not an automatic finding that the
reviewer's report is false. Exact sources are in the linked JSON/HTML.

'''
    for row in omissions:checklist+=f"- {row['issue_id']} / `{row['business_id']}` / {row['marker']}: {row['proposed_information_for_decision']}\n"
    checklist+='''
## C. Finish atomic units and required metadata

321 extracted clause spans preserve exact approved parent wording, but are NOT
finished atomic references. Inspect parent context, split residual lists and make
each final proposition independently assessable without changing its scope,
attribution, polarity or prevalence. Choose the exact supporting evidence for
each unit. Complete core/supporting, aspect, specificity and review polarity;
confirm visual units against the caption AND its photo. The importance column
is blank in all226 submitted candidates; no default was silently invented.

The pending fields are in UNIT_METADATA_AND_ATOMICITY_PENDING.json. No new
metadata or unit wording is represented as already human confirmed. Only after
these items and business completeness are resolved can strict final acceptance
pass and an immutable scoring reference be produced. No48-output inference yet.
'''
    write(OUT/'RESEARCHER_DECISION_CHECKLIST.md',checklist)
    esc=lambda s:html.escape(str(s))
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>D053 reference finalization — pending decisions</title>
<style>body{font:16px/1.55 system-ui;max-width:1180px;margin:28px auto;padding:0 18px;color:#17212d}h1{line-height:1.2}.alert{background:#fff1d6;border:2px solid #a16500;padding:18px}article,details{border:1px solid #b9c3cc;padding:16px;margin:20px 0;border-radius:8px}img{max-width:440px;max-height:390px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}.proposed{background:#edf4ff;padding:12px}.approved{background:#eff8ed;padding:12px}small{color:#52606c}summary{cursor:pointer;font-weight:650}a{color:#0751a3}</style>
<h1>D-053 — Researcher approval recorded; reference freeze pending</h1>
<div class="alert"><b>Không phải reference cuối để scoring.</b> 203 Supported +21 Needs Revision +2 Uncertain đã ghi nhận theo hội thoại. Bản JSON gốc vẫn được giữ nguyên. Cần chốt15 correction xung đột caption, các ý có thể bị bỏ sót và atomic units/metadata. Chưa sinh summaries Pilot A/B.</div>
<p>Đọc toàn bộ review, caption và đúng ảnh gốc trước khi xác nhận completeness. Teacher candidates không phải danh sách đầy đủ. Không thêm chi tiết chỉ có trong ảnh vào reference caption-grounded. Các alternatives dưới đây chỉ là đề xuất, chưa áp dụng.</p>
<p><a href="researcher_finalization/ANNOTATION_AND_PROVENANCE_AUDIT_REPORT.md">Audit report</a> · <a href="researcher_finalization/RESEARCHER_DECISION_CHECKLIST.md">Decision checklist</a> · <a href="researcher_finalization/RESEARCHER_APPROVAL_RECORD.json">Approval provenance</a> · <a href="#omissions">Omissions</a> · <a href="#sources">All source evidence</a></p><h2>21 corrections: exact approved text and evidence scope</h2>'''
    for r in corrections:
        bid=r['business_id']; page+=f'''<article id="{esc(r['candidate_id'])}"><h3><code>{esc(r['candidate_id'])}</code></h3>
<p><b>{esc(r['caption_scope_assessment'])}</b> · <a href="#source_{esc(bid)}">Full business source</a></p>
<a href="{esc(r['image_file'])}" target="_blank"><img src="{esc(r['image_file'])}" alt="Original photo {esc(r['photo_id'])}"></a>
<p><code>{esc(r['caption_id'])} / PHOTO {esc(r['photo_id'])}</code></p><p><b>LED caption:</b> {esc(r['caption'])}</p>
<p class="approved"><b>Exact researcher-approved correction (conditional on evidence):</b> {esc(r['approved_corrected_claim_text'])}</p>
<p><b>Audit:</b> {esc(r['reason'])}</p>'''
        if r['alternative_for_researcher_review_only']:page+=f'<p class="proposed"><b>Alternative for your decision — NOT applied:</b> {esc(r["alternative_for_researcher_review_only"])}</p>'
        page+='</article>'
    page+='<h2 id="omissions">Five potential important omissions — not automatically added</h2>'
    for r in omissions:page+=f'''<article><h3>{esc(r['issue_id'])} / {esc(r['business_id'])}</h3><p>{esc(r['proposed_information_for_decision'])}</p><p>{esc(r['reason'])}</p><code>{esc(r['marker'])} / {esc(r['review_id'])}</code><pre>{esc(r['quote'])}</pre><a href="#source_{esc(r['business_id'])}">Full original source</a></article>'''
    page+='<h2>Two Uncertain exclusions</h2>'
    for r in uncertain:page+=f'<article><code>{esc(r["candidate_id"])}</code><p>{esc(r["original_candidate_text"])}</p><p>Preserved separately, zero scoring units. Any nonblank corrected text is not promoted.</p></article>'
    page+='<h2 id="sources">All24 businesses — complete652 reviews and90 original photos</h2>'
    for case in cases:
        page+=f'<details id="source_{esc(case["business_id"])}"><summary>{esc(case["business_id"])} · {esc(case["subset"])} · {len(case["reviews"])} reviews / {len(case["captions"])} captions</summary>'
        for r in case['reviews']:page+=f'<h4>{esc(r["marker"])} / {esc(r["review_id"])}</h4><pre>{esc(r["text"])}</pre>'
        if not case['captions']:page+='<p><b>Zero captions: no visual reference units.</b></p>'
        for c in case['captions']:page+=f'<h4>{esc(c["caption_id"])} / PHOTO {esc(c["photo_id"])}</h4><p>{esc(c["text"])}</p><a href="{esc(c["image_file"])}" target="_blank"><img loading="lazy" src="{esc(c["image_file"])}" alt="Original photo"></a>'
        page+='</details>'
    page+='</html>'
    write(OUT/'FINALIZATION_REVIEW.html',page)
    # Derived package: source/science bytes stay exact, previous entry points
    # are archived inside the new namespace rather than changed in place.
    shutil.copytree(SOURCE,DEST)
    hist=DEST/'provenance/preparation_v2';hist.mkdir(parents=True)
    for name in ('README_FIRST.md','CHATGPT_REVIEW_REQUEST.md','SHA256_MANIFEST.json','SHA256SUMS.txt'):
        (DEST/name).rename(hist/name)
    data=DEST/'researcher_finalization';data.mkdir()
    skip={'incoming','source_bundle','inspection_only','FINAL_ZIP_TESTS'}
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='FINALIZATION_REVIEW.html':shutil.copyfile(p,data/p.name)
    (data/'incoming').mkdir()
    for name in ('README_FOR_CODEX.md','D053_AI_DRAFT_24_RESTAURANTS_EDITED.json','SHA256_INPUTS.json'):
        shutil.copyfile(OUT/'incoming'/name,data/'incoming'/name)
    shutil.copyfile(OUT/'FINALIZATION_REVIEW.html',DEST/'FINALIZATION_REVIEW.html')
    # The report-local HTML is explicitly a copy; open root HTML for relative photos.
    for name in ('prepare_stage_i_d053_researcher_handoff_v1.py','test_stage_i_d053_researcher_handoff_v1.py','build_stage_i_d053_reference_resolution_bundle_v1.py'):
        shutil.copyfile(ROOT/'scripts'/name,DEST/'code'/name)
    readme='''# D-053 researcher-approved decisions: resolution review package

**STOP: NOT A FINAL REFERENCE FREEZE; NOT READY FOR SCORING OR PREDICTIONS.**

Open FINALIZATION_REVIEW.html at the ZIP root. It contains all21 exact approved
corrections beside original captions/photos,15 flagged scope conflicts with
unapplied alternatives,five possible missing review aspects,two Uncertain
exclusions and every full source review/photo. No external files/server needed.

Your direct conversation approves203 Supported decisions,21 corrections subject
to original evidence rules and two Uncertain exclusions. Approval was recorded
separately; the original uploaded JSON remains unchanged/human_verified=false.
See researcher_finalization/RESEARCHER_APPROVAL_RECORD.json and
researcher_finalization/RESEARCHER_APPROVED_DECISIONS.json. Imported candidate
confirmations are materializations of this chat approval, not fabricated UI clicks.

321 exact clause-span proposals preserve approved text but are NOT completed
atomic references. No final unit, importance, atomicity or business completeness
was invented. All226 submitted importance fields are blank; final units require
additional mandatory metadata under the unchanged protocol. Source/semantic
issues remain explicit. See researcher_finalization/ANNOTATION_AND_PROVENANCE_AUDIT_REPORT.md
and researcher_finalization/RESEARCHER_DECISION_CHECKLIST.md.

The unchanged final validator correctly BLOCKS this draft. Its rejection is
expected, not a software test failure or permission to relax the protocol.
Do not use span counts or parent-label counts as final reference denominators.
Do not apply proposed alternative wordings or omission additions without the
researcher's explicit decision. Existing approved labels are not silently changed.

## Reproducible CPU-only verification from the extracted root

Python3.11+, NumPy1.26.4, jsonschema4.23.0 and Node.js22.14.0 were used. Install
ordinary dependencies using requirements-offline.txt if needed.

```text
python -s -X utf8 code/run_stage_i_d053_extracted_tests_v2.py --root . --output NEW_ORIGINAL83_REPORT.json --portable-readiness-script code/check_stage_i_d053_bundle_readiness_v2.py
node code/test_stage_i_d053_ui_v2.js SOURCE_REFERENCE_ANNOTATION.html
python -s -X utf8 -m unittest discover -s code -p test_stage_i_d053_portable_readiness_v2.py -v
python -s -X utf8 code/test_stage_i_d053_researcher_handoff_v1.py
python -s -X utf8 code/stage_i_d053_annotations.py --package . --annotations researcher_finalization/REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json
```

Draft structural validation passes. Adding --final must FAIL for the documented
pending criteria. No final reference JSON is falsely published. Exact final ZIP
test receipts are supplied beside the ZIP; old validation_v2/ logs are historical
preparation evidence. Portable checker verifies source-preparation integrity only,
not final-reference acceptance. The historical full case inputs/blind key remain
separate exactly as in v2, preserving source-only review boundaries.

No Teacher/model/API call, new Pilot A/B prediction, training, held-out access,
checkpoint modification or accepted-data change. Stop after this resolution
review. Confirm the outstanding decisions before a true reference freeze;48
Pilot A/B summaries require later independent review and explicit authorization.
'''
    write(DEST/'README_FIRST.md',readme)
    write(DEST/'CHATGPT_REVIEW_REQUEST.md','''# Independent D-053 researcher-reference resolution review

The researcher personally approved203 Supported labels,21 corrected texts subject
to original evidence eligibility,and two Uncertain exclusions. The submitted
JSON had false flags; the approval came from the direct conversation. Inspect
the explicit import provenance and unchanged archival JSON; do not pretend the
file originally contained human-confirmed units or fabricate review timestamps.

Open FINALIZATION_REVIEW.html. Review the15 flagged correction/caption scope
conflicts: distinguish ordinary entailed paraphrases from added identities,
attributes,counts,actions or relations visible only in photos. Alternatives are
proposals, never silently applied. Check all21 exact approved corrections and
their IMG_CAP/PHOTO mappings. Do not automatically overrule the researcher.

Review the five possible review omissions with exact REV quotations and decide
whether they matter for the reference denominator. They have not been approved
or added. The uncited-review inventory is not an automatic omission detector.

321 exact clause spans are NOT finalized atomic units. Check parent context,
attribution,polarity,lists,prevalence and evidence scope before proposing final
atomic formulations and unit priorities/metadata for researcher approval. No
unit-level human confirmation or core/supporting value has been fabricated.

The strict final gate currently BLOCKS. Report remaining decisions honestly;
do not claim reference completeness based on passing software tests. Preserve
the two uncertain cases outside scoring,all frozen sources/configs/checkpoints
and the zero-caption behavior. Teacher candidates are not independent truth.
No automatic human labeling, method changes, teacher calls, new predictions,
training or held-out use. Return concrete decisions needed before final freeze.
''')
    # Verify local accepted source/code/config and adapters without loading.
    config=read(SOURCE/'FROZEN_EVALUATION_CONFIG.json');preservation=[]
    for name,item in config['bindings'].items():
        if name in ('case_inputs','blinding_key'):continue
        p=ROOT/item['path'];assert sha(p)==item['sha256'],name
        preservation.append({'name':name,'sha256':item['sha256']})
    for name,item in config['arms'].items():
        assert sha(ROOT/item['adapter_path'])==item['adapter_sha256']
        preservation.append({'name':name,'sha256':item['adapter_sha256']})
    write(OUT/'ACCEPTED_SCIENTIFIC_ARTIFACT_PRESERVATION.json',{'status':'PASS','checked':preservation,
      'model_weights_loaded':False,'held_out_files_accessed':False,'original_input_sha256':sha(OUT/'incoming/D053_CODEX_FINAL_REFERENCE_HANDOFF.zip'),
      'original_edited_json_sha256':sha(OUT/'incoming/D053_AI_DRAFT_24_RESTAURANTS_EDITED.json')})
    shutil.copyfile(OUT/'ACCEPTED_SCIENTIFIC_ARTIFACT_PRESERVATION.json',data/'ACCEPTED_SCIENTIFIC_ARTIFACT_PRESERVATION.json')
    print(json.dumps({'status':'BLOCKED_REFERENCE_REVIEW_PREPARED','bundle':str(DEST),'data':str(OUT)},indent=2))


def pack():
    assert DEST.is_dir()
    zpath=DEST.with_suffix('.zip');assert not zpath.exists()
    files=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(DEST).as_posix()
        if relative in ('SHA256_MANIFEST.json','SHA256SUMS.txt'):continue
        if p.suffix.lower() in ('.txt','.json','.jsonl','.py','.js','.md','.html'):
            import re
            assert not re.search(rb'\bsk-[A-Za-z0-9_-]{20,}',p.read_bytes()),relative
        files.append({'path':relative,'bytes':p.stat().st_size,'sha256':sha(p)})
    write(DEST/'SHA256_MANIFEST.json',{'status':'IMMUTABLE_BLOCKED_REVIEW_SNAPSHOT_NOT_FINAL_REFERENCES',
        'excludes':['SHA256_MANIFEST.json','SHA256SUMS.txt'],'files':files})
    write(DEST/'SHA256SUMS.txt',''.join(r['sha256']+'  '+r['path']+'\n' for r in files))
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,NAME+'/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+2
        for row in files:assert hashlib.sha256(z.read(NAME+'/'+row['path'])).hexdigest()==row['sha256']
    zpath.with_suffix('.zip.sha256').write_text(sha(zpath)+'  '+zpath.name+'\n',encoding='ascii')
    receipt={'status':'ZIP_INTEGRITY_PASS_REFERENCE_ACCEPTANCE_BLOCKED','zip':str(zpath),'bytes':zpath.stat().st_size,
             'MiB':zpath.stat().st_size/2**20,'sha256':sha(zpath),'files':len(files)+2,'final_reference_freeze_created':False}
    write(OUT/'REVIEW_BUNDLE_EXPORT_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pack',action='store_true');args=parser.parse_args()
    if args.pack:pack()
    else:main()
