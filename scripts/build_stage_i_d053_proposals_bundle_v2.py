"""Export new unconfirmed proposals and the unchanged source/protocol evidence."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,html,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d053_reference_proposals_v2'
OLD=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1'
SOURCE=OLD/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
NAME='stage_i_d053_reference_proposals_review_bundle_v2'
DEST=ROOT/'outputs/stage_i'/NAME
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def prepare():
 assert not DEST.exists(),'Preserve existing export versions'
 units=read(OUT/'PROPOSED_ATOMIC_REFERENCE_UNITS.json')['units'];conflicts=read(OUT/'CAPTION_SCOPE_RECOMMENDATIONS.json');omissions=read(OUT/'OMISSION_RECOMMENDATIONS.json');version=read(OUT/'PROPOSAL_VERSION.json');cases=read(SOURCE/'source_cases.json')
 payload={'units':units,'conflicts':conflicts,'omissions':omissions,'version':version,'cases':cases,'template':read(OUT/'RESEARCHER_DECISIONS_TEMPLATE.json')}
 page=(ROOT/'scripts/stage_i_d053_resolution_ui_v2.html').read_text(encoding='utf-8').replace('__STATE_JS__',(ROOT/'scripts/stage_i_d053_resolution_state_v2.js').read_text(encoding='utf-8')).replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
 write(OUT/'RESEARCHER_APPROVAL_V2.html',page)
 for c in cases:
  for cap in c['captions']:
   p=OUT/cap['image_file'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/cap['image_file'],p)
 report=f'''# D-053 revised reference proposals — researcher review required

Date: 2026-10-17. Status: **PROPOSALS ONLY / FINAL REFERENCE ACCEPTANCE BLOCKED**.

The 226 researcher-approved candidate decisions remain unchanged: 203 Supported,
21 approved corrections subject to evidence rules, and 2 Uncertain exclusions.
The original submitted JSON and approval ledger are preserved byte for byte.
No newly proposed wording, metadata, binding, exclusion, or omission is recorded
as human-confirmed. No scoring-ready references or reference freeze is emitted.

## Caption scope and omissions

All 15 conflicts have explicit caption-safe alternatives and linked original
captions/photos. Recommendation: **11 include safe revisions, 4 exclude** because
the remaining safe overlap is too generic relative to the original corrected
claim. Generic compatibility is not a factual error; these are importance
recommendations for the four named conflicted candidates, not a new blanket
caption-quality rule. All original approved corrections remain unchanged.

The five omissions have recommendations: **3 core, 1 supporting, 1 excluded**.
Noise preventing conversation; refused advertised happy hour/rude manager;
and repeated pickup unreliability supply meaningful negative contrasts. Cold-brew
unavailability is a secondary product-availability detail. The isolated pork-bowl
spiciness complaint is proposed for exclusion as peripheral, not false. The
optional supporting unit remains available if the researcher prefers inclusion.
Five findings require seven atomic proposals because two contain distinct issues.

## Atomization and evidence

321 historical clause spans are retained as provenance, not assumed final units.
Explicit semantic recipes split lists into attributed opinions, attributes, or
relations; positive and negative views remain separate. Discourse introductions,
mixed-opinion summaries, synonymous praise, and duplicate generic summaries are
not scored as extra facts. Five additional redundant drafts were retired during
context curation, with reasons. There are **{len(units)} proposed units**:
657 reformulations of the 224 non-uncertain candidate parents, plus 7 omission
proposals; 531 review and 133 visual proposals. This larger number reflects
fine-grained list splitting, not an increased count of independently approved
claims, a final denominator, or a requirement to retain every proposal. The
researcher may merge, edit or exclude proposals before final reference validation.

Each proposal supplies a stable ID, parent/old-span lineage, standalone wording,
core/supporting importance, informativeness, aspect, specificity, review polarity
(visual polarity null), exact evidence IDs/quotes, and unconfirmed provenance.
One primary parent per unit is retained for frozen-validator compatibility. Old
span mapping is at parent meaning level; it is explicitly not exact one-to-one
semantic correspondence. The unchanged parent context is separately accessible.

Evidence lookup is a retrieval aid, not a factual verifier. Source-context
curation corrected 54 compact bindings/wordings; all bindings are exact substrings
of the permitted selected reviews or original captions. Single-citation review
units use singular attribution to avoid transferring parent-level prevalence to
each dish. Count-specific claims retain multiple cited reviews. Matching quote
IDs and software tests do not establish semantic truth or independent human
ground truth. Image scope reuses the earlier 90-photo inspection and original
photos; the 15 alternatives remove flagged photo-only details. New alternatives
still require the researcher's approval. Semantic completeness of all 652 full
reviews is not independently certified by this package.

One additional wording caution is surfaced in the relevant business package:
7u40cyMxw4iGosMdkesoEQ / REVIEW_CANDIDATE_4 / REV_6 literally says "did tell me
how much". Do not silently change it to "did not". Recommend excluding only the
ambiguous pricing-clarity unit; retain the separate bun/chicken value complaint
and all original candidate decisions. This is a new proposed-unit caution, not
a relabeling of the previously approved parent.

## Efficient researcher workflow

Open RESEARCHER_APPROVAL_V2.html. Initial choices and reviewer identity are blank;
no default is approval. Resolve 15 caption and 5 omission cards, then review
wording/metadata by business. Full source evidence is accessible, including the
instruction to identify important information omitted by Teacher candidates.
Confirm packages individually or use the explicit reviewed-all action after
reviewing them and checking the attestation. No need to re-label 226 candidates
or click every atomic unit. Optional overrides save edits/exclusions separately.
Changing an exception invalidates package confirmations; changing a unit
invalidates that business. Export/import is bound to the proposal content hashes.
Incomplete approval export is blocked; progress export remains available.

The exported file is a researcher decision handoff, not a finalized reference
JSON. New decisions and edited evidence must be checked under the unchanged
schema/protocol before any final freeze. The two Uncertain parents produce no
units. The zero-caption case has no visual proposals. There is no prediction,
training, Teacher API use, checkpoint change, or held-out access.

## Previous archive receipt reconciliation

The delivered v1 archive is 11,055,577 bytes, SHA-256:
af82bc27e934924c67ec813114caaa24748532e4a257c23a04411c73d7219a3d.
Its external receipt, sidecar, and published-archive extracted-test record match
the actual bytes. The different 2b9e52e708f60425ffaecce56d60af971af18b9d369b99f9c9a3b9c53b315f49
belongs to the earlier release candidate before test evidence was embedded. It
does not identify the delivered archive. Both remain unchanged as provenance.
ZIP_RECEIPT_RECONCILIATION.json records this distinction. For the new ZIP, use
the external final receipt/sidecar: an archive cannot contain its own final byte
hash without a self-reference problem. The internal manifest hashes its files.

## Acceptance and remaining decisions

Software/provenance validity and scientific acceptance are separate. Run the
original 137 tests plus the new proposal/state tests from the delivered archive.
Exact measured results are in the external FINAL_ZIP_TESTS record and completion
receipt. Synthetic state tests never write real researcher decisions. A new
interactive browser smoke test is not claimed in this offline preparation.

Final acceptance remains blocked pending researcher approval of alternatives,
omissions, atomic wording/metadata and completeness. No evaluation threshold,
caption filtering, test split, or model configuration has changed. Stop here.
'''
 write(OUT/'D053_REFERENCE_PROPOSALS_REPORT.md',report)
 checklist='# D-053 concise approval checklist\n\nAll choices below are proposals, not applied decisions. Use the HTML for evidence and batch approval. Original 226 labels remain unchanged.\n\n## Caption scope\n\n'
 for r in conflicts:checklist+=f"- [ ] `{r['candidate_id']}` — **{r['recommended_action']}**\n  - Safe wording: {r['proposed_caption_safe_wording']}\n  - Reason: {r['recommendation_reason']}\n\n"
 checklist+='## Five omissions\n\n'
 for r in omissions:checklist+=f"- [ ] `{r['issue_id']}` / `{r['business_id']}` / `{r['marker']}` — **{r['recommended_reference_role']}**: {r['recommendation_reason']}\n\n"
 checklist+='''## Atomic wording and metadata

- [ ] Review the 24 business packages; all new text and metadata remain proposals.
- [ ] Check positive/negative attribution, exact source evidence, duplicated meaning,
  and meaningful information absent from Teacher candidates. Core means important
  to the business summary; not every named dish must become a core unit.
- [ ] Review the literal pricing-clarity caution for 7u40cyMxw4iGosMdkesoEQ / REV_6.
- [ ] Confirm packages only after reviewing them; bulk confirmation is an optional
  explicit action, never a preselected approval. Add/override or reject as needed.
- [ ] Export the researcher decision handoff for schema/evidence validation.

This checklist alone does not automatically assign labels or freeze references.
Do not generate Pilot A/B predictions until later approval.
'''
 write(OUT/'RESEARCHER_APPROVAL_CHECKLIST.md',checklist)
 readiness={'status':'PREPARED_FOR_RESEARCHER_DECISIONS_FINAL_ACCEPTANCE_BLOCKED','existing_approved_candidate_decisions':226,'new_proposed_units':len(units),'new_human_confirmed_units':0,'caption_conflicts':15,'caption_recommendations':{'include_safe':11,'exclude':4},'omission_recommendations':{'core':3,'supporting':1,'excluded':1},'additional_literal_wording_caution':1,'final_reference_freeze_created':False,'new_predictions':0,'teacher_API_calls':0,'training':False,'held_out_access':False,'required_next_step':'Researcher decisions on alternatives, omission roles, atomic wording/metadata and completeness; then unchanged final validator.'}
 write(OUT/'REFERENCE_PROPOSAL_READINESS.json',readiness)
 # Reuse the exact former portable source/code bundle, retaining historical state
 # for its 137 tests. Do not rewrite or relabel any previous artifact.
 previous=ROOT/'outputs/stage_i/stage_i_d053_researcher_reference_resolution_bundle_v1.zip'
 with zipfile.ZipFile(previous) as z:
  for item in z.infolist():
   parts=PurePosixPath(item.filename).parts[1:]
   if not parts or item.is_dir() or '__pycache__' in parts:continue
   assert '..' not in parts and not PurePosixPath(*parts).is_absolute()
   rel=Path(*parts)
   if rel.as_posix() in ['README_FIRST.md','CHATGPT_REVIEW_REQUEST.md','SHA256_MANIFEST.json']:rel=Path('provenance/previous_resolution_v1')/rel
   target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
 (DEST/'reference_proposals').mkdir(exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and not p.name.startswith('WORKING_'):shutil.copy2(p,DEST/'reference_proposals'/p.name)
 shutil.copy2(OUT/'RESEARCHER_APPROVAL_V2.html',DEST/'RESEARCHER_APPROVAL_V2.html')
 for pattern in ['*d053*proposals*v2.py','*d053*resolution_state_v2*','stage_i_d053_resolution_ui_v2.html','d053_atomic_recipes_v2*.txt','d053_evidence_curation_v2.json']:
  for p in (ROOT/'scripts').glob(pattern):shutil.copy2(p,DEST/'code'/p.name)
 write(DEST/'README_FIRST.md','''# Start here — D-053 reference proposals v2

This is a researcher decision package, **not finalized evaluation references**.
Open **RESEARCHER_APPROVAL_V2.html** (new interface), not the archived original
SOURCE_REFERENCE_ANNOTATION.html or FINALIZATION_REVIEW.html.

1. Review 15 caption decisions and 5 omission recommendations.
2. Review the proposed wording/metadata by business, with exact source evidence.
3. Enter researcher identity; confirm packages explicitly; export decision JSON.
4. Return the export for validation. No reference freeze or predictions yet.

Read reference_proposals/D053_REFERENCE_PROPOSALS_REPORT.md and
reference_proposals/RESEARCHER_APPROVAL_CHECKLIST.md. All 226 previously approved
candidate decisions are unchanged in researcher_finalization/. New proposals
have human_verified=false and scoring_eligible=false. Two uncertain parents
remain excluded. Sources are validation-only; photos are original relative files.

The original 321 spans are historical, not the new scoring denominator. The
664 finer-grained proposals are optional wording/metadata proposals, not 664
approved facts. They may still be merged, edited or excluded by the researcher.
No software check certifies factual completeness or independent human truth.

Offline commands (from this extracted folder; Python 3.11+, jsonschema, Node):

```
python code/run_stage_i_d053_extracted_tests_v2.py --root . --output ORIGINAL_83_TESTS.json --portable-readiness-script code/check_stage_i_d053_bundle_readiness_v2.py
node code/test_stage_i_d053_ui_v2.js SOURCE_REFERENCE_ANNOTATION.html
python -m unittest discover -s code -p test_stage_i_d053_portable_readiness_v2.py -v
python -m unittest discover -s code -p test_stage_i_d053_researcher_handoff_v1.py -v
python -m unittest discover -s code -p test_stage_i_d053_atomic_proposals_v2.py -v
node code/test_stage_i_d053_resolution_state_v2.js RESEARCHER_APPROVAL_V2.html
python code/check_stage_i_d053_bundle_readiness_v2.py --readiness
```

The original 83 + 30 + 8 + 16 = 137 tests apply to the preserved original source
and imported approval state. New tests inspect this proposal package separately.
Synthetic tests never assign actual labels. Portable readiness checks supplied
frozen files and adapter hash receipts; original checkpoint tensors and historical
target/output-containing inputs remain outside this source-only export. Full
evaluation/reference readiness remains blocked under the unchanged validator.

SHA256_MANIFEST.json hashes package files. The archive's final byte SHA belongs
to the external .zip.sha256 and receipt, not an internal self-hash. Previous
release-candidate receipts are clearly historical; see the reconciliation JSON.
''')
 write(DEST/'CHATGPT_REVIEW_REQUEST.md','''# Independent D-053 proposal review request

Please inspect the new reference_proposals/ files and RESEARCHER_APPROVAL_V2.html.
The 226 prior candidate decisions must remain unchanged. New wording, metadata,
evidence bindings, alternatives and omission choices are NOT human-confirmed.

Review: 15 caption-safe revisions (11 recommended keep, 4 exclude as too generic);
5 omissions (3 core, 1 supporting, 1 exclude); transformation of 321 historical
fragments into 664 properly formulated proposals; duplicate/synonym consolidation;
preservation of review sentiment and cautious attribution; metadata priorities;
caption-plus-photo scope; and the isolated literal pricing ambiguity. Propose
further merges/exclusions if fine-grained units duplicate meaning or importance.

Check the interface: blank initial choices, explicit researcher identity, easy
batch review, no auto human labels, version-bound progress export/import,
incomplete handoff blocking, and invalidation after changes. Check whether source
evidence and instructions to identify meaningful omissions are sufficiently clear.
Teacher wording is not independent factual ground truth. Quote matching and
software PASS do not establish semantic support. Do not treat 664 as a final
scoring denominator. Reference freeze remains blocked pending researcher review.

Verify SHA receipt reconciliation: published prior ZIP af82bc... is distinct from
earlier release candidate 2b9e52... . Both old archives remain unchanged.
No predictions, training, Teacher calls or held-out evaluation are authorized.
''')
 print(json.dumps({'prepared':str(DEST),'proposed_units':len(units),'html':str(OUT/'RESEARCHER_APPROVAL_V2.html')}))
def pack():
 assert DEST.is_dir()
 manifest={'scope':'All payload files except this manifest itself; final archive byte hash is external',
  'files':[{'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p!=DEST/'SHA256_MANIFEST.json' and '__pycache__' not in p.parts]}
 write(DEST/'SHA256_MANIFEST.json',manifest)
 archive=DEST.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(DEST.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,NAME+'/'+p.relative_to(DEST).as_posix())
 digest=sha(archive);write(archive.with_suffix('.zip.sha256'),digest+'  '+archive.name+'\n')
 receipt={'archive':str(archive.relative_to(ROOT)).replace('\\','/'),'sha256':digest,'bytes':archive.stat().st_size,'payload_files':len(manifest['files']),'scope':'Exact delivered archive bytes; verify FINAL_ZIP_TESTS against this hash','scientific_acceptance':'BLOCKED_PENDING_RESEARCHER_DECISIONS','new_human_confirmed_units':0}
 write(OUT/'DELIVERED_ZIP_RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pack',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.pack:pack()
