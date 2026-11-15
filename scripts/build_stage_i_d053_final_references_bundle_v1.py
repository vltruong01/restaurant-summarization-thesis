"""Package accepted D053 references and immutable evidence; no GPU/API calls."""
from pathlib import Path,PurePosixPath
from collections import Counter
import argparse,hashlib,html,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage_i/d053_final_references_v1'
OLD=ROOT/'outputs/stage_i/d053_researcher_reference_finalization_v1';SOURCE=OLD/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2'
NAME='stage_i_d053_final_references_chatgpt_bundle_v1';DEST=ROOT/'outputs/stage_i'/NAME
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prepare():
 assert not DEST.exists(),'Preserve export versions'
 rows=read(OUT/'REFERENCE_DISTRIBUTIONS_BY_BUSINESS.json');state=read(OUT/'FINAL_REFERENCE_ANNOTATIONS.json');audit=read(OUT/'APPROVED_ALL_664_UNIT_DECISIONS_AUDIT.json')['units'];cases=read(SOURCE/'source_cases.json');validation=read(OUT/'COMPLETE_FINAL_REFERENCE_VALIDATION.json')
 assert validation['valid'] and len(state['units'])==576
 table='| Business | Core | Supporting | Excluded | Review units | Visual units |\n|---|---:|---:|---:|---:|---:|\n'
 for r in rows:table+=f"| {r['display_name']} (`{r['business_id']}`) | {r['core']} | {r['supporting']} | {r['excluded']} | {r['review_scoring_units']} | {r['visual_scoring_units']} |\n"
 table+='| **Total** | **261** | **315** | **88** | **502** | **74** |\n'
 write(OUT/'REFERENCE_DISTRIBUTIONS_BY_BUSINESS.md','# D-053 approved distributions\n\n'+table)
 note='''# D-053 versioned exclusion representation — exact compatibility disclosure

The frozen v1 annotation code, v1 JSON Schema, evaluation configuration, D049
protocol, datasets and targets are unchanged. The v1 structural schema accepts
the finalized annotation structure; its semantic evidence/confirmation checks
pass. Its final readiness check correctly BLOCKS 24 Supported parents because
all their proposed children were explicitly excluded from scoring. Nine further
all-child-excluded parents are Needs Revision and already use v1's supported
excluded_from_reference disposition. There are 33 all-child-excluded parents.

The researcher's new instruction explicitly approves all88 exclusions and asks
that Supported parent labels remain preserved. Changing those labels, inserting
dummy scoring facts, or counting excluded units would misrepresent the approval.

New implementation: stage_i_d053_curated_references_v2.py, final format version2,
and a supplementary curated_reference_policy schema. Candidate and unit shapes
remain v1-compatible. The original validator is called unmodified. Every v1
error and every pending unit, evidence, identity, timestamp, business or snapshot
condition is still fatal. Only the EXACT mandatory-child pending messages may
be resolved when ALL original children are present in the canonical664-proposal
set and ALL have explicit, source-bound researcher-approved exclusions. Every
excluded row needs the approved reason, text, IDs, evidence, metadata, identity,
timestamp and conversation provenance. Every accepted unit must match the
approved text/metadata/source. All88 must be outside the576 scoring units.

This is an explicit representational extension, not an unchanged-v1 final PASS.
UNCHANGED_V1_FINAL_VALIDATOR_RESULT.json preserves all24 original blockers.
COMPLETE_FINAL_REFERENCE_VALIDATION.json discloses the v2 result. No arbitrary
errors are filtered and no missing approval/evidence is waived. Negative controls
test rejected mutations of labels, evidence, metadata, snapshots, approval and
exclusion proof. No v1 gate or enum is deleted or relaxed in its original files.

The final status is RESEARCHER_CONFIRMED_SOURCE_REFERENCES_D053_CURATED_V2.
The old v1 reference_freeze() and future-comparison loader intentionally do not
accept this file. A future D-054 preflight must explicitly dispatch this version
to the supplied strict v2 validator before using its approved reference set.
Do not monkey-patch the old evaluator, impersonate Needs Revision labels, or
replace its configuration/hash bindings. No future-comparison code was run or
modified here. Independent review and separate D-054 authorization remain needed.
'''
 write(OUT/'VALIDATOR_COMPATIBILITY_AND_SCHEMA_CHANGE.md',note)
 report='''# D-053 final researcher-approved references

Date:2026-10-17. **Reference construction and versioned validation PASS.**
Await independent package review; D-054 inference remains NOT AUTHORIZED.

## Approved reference population

24 validation Filtered businesses (20 fixed random +4 targeted),652 original
selected reviews and90 original captions/photos. The final reference set has
**576 units:261 Core +315 Supporting**. **88 exclusions** are audit-only, outside
every scoring population. All664 approved unit decisions are accounted for.
502 retained units describe selected customer opinions/observations:285 positive,
199 negative and18 neutral.74 visual units link56 original photos. Every business
retains both positive and negative review perspectives. The one zero-caption
business, j3kGicU3mPG_ciIKG1Vuyg, has zero visual units. No evidence was fabricated.

Core denotes important content, not a requirement to repeat each granular detail
verbatim. Supporting details may reward useful additions; their omission is not
automatically an error. Excluded units never enter scoring denominators. Do not
use664 as an automatic informativeness denominator or treat all576 as mandatory
summary sentences. Frozen metric definitions/config remain unchanged; source
coverage must be interpreted separately by importance and semantic coverage.
Review references remain attributed customer opinions, not objective assertions
about restaurants. Teacher targets are not independent factual ground truth.

## Input identity, exact wording and evidence

Attached archive SHA-256:
a6b297e11491d8328402288b9273df2a5c052c8d251fd1882bfa8094569660ad.
All98 members were extracted and inspected;97 SHA256SUMS entries and CRC pass.
All664 original proposal IDs and the three original per-file SHA bindings match.
Proposal identity:90b7208c2ec75f63e067757969d0753b82f658df11a6c96dc4fb16d585e11898.
The embedded source cases and units equal the unchanged original records. The
HTML's version descriptor omits a decorative status field, but its exact ID/file
hashes and separately supplied unchanged version JSON match. All90 images are
byte-identical to their originals. All original quotes remain exact and untouched.

149 metadata recommendations differ from the initial proposals; all are applied
as explicitly approved. There is one approved grammar change, describe→describes,
on an excluded audit unit. Retained576 wording matches the approved source
recommendations exactly. No Codex-generated replacement factual claims were added.
Exact quote/ID matching and metadata checks are structural evidence integrity,
not an independent software finding of factual correctness or full semantic
coverage of every source review. Photo corroboration flags record the researcher's
stated review/approval, with earlier AI inspection retained as separate provenance.

## Resolutions and existing decisions

15 caption-scope choices:10 approved caption-safe revisions and5 approved scope
exclusions. Scope acceptance and informativeness priority are separate:three
accepted-safe captions still have all units excluded for low information
(j9fzpALNCgvDwCXmA5Vcjw IMG_CAP_4; Ktg3ahIxk0JIkJwXAqu2ew IMG_CAP_1;
haPQJ2oy3529RMzpWrA39A IMG_CAP_3). These are consistent approved decisions,
not a silent override of caption safety. All five scope-excluded candidates have
zero scoring units. No photo-only fact is introduced by an alternative correction.

Five omissions:3 Core,1 Supporting,1 Exclude. The four included findings produce
six atomic units; the pork-bowl finding produces one excluded audit unit. Literal
pricing ambiguity at7u40cyMxw4iGosMdkesoEQ / REV_6 remains literal and its ambiguous
unit is among the approved exclusions; no correction to the source was invented.

All226 parent decisions are preserved:203 Supported,21 Needs Revision,2 Uncertain.
Original corrected text, optional parent metadata and historic annotations remain
unchanged. Final unit IDs/dispositions are new derived links with transparent new
confirmation snapshots. Of21 revisions,12 resolve to confirmed scoring units and
9 are excluded by scope/priority. Both uncertain parents remain outside scoring,
including the one with nonblank unapproved suggested correction. Historical
proposal/import files remain unmodified and keep their original confirmation flags.

## Explicit approval, not fabricated interface use

RESEARCHER_APPROVAL_RECORD.json records the current direct user instruction
verbatim, its hash, archive/curation hashes, exact scope and original proposal ID.
The original recommendation JSON still says unapproved. New confirmation fields
materialize the explicit conversation approval. Reviewer identity denotes the
requesting researcher; no name, platform message ID, inspection time, signature
or browser interaction is fabricated. Recorded timestamps mean import time.
RESEARCHER_DECISIONS_CONVERSATION_HANDOFF.json is accepted by the existing decision
state validator without calling UI action APIs. It contains225 approved overrides
and clearly identifies its conversation origin. No extra signature or repeat
664-unit annotation is necessary for this explicit authorization.

## Validation and exact compatibility limitation

V1 structural JSON Schema:PASS. Original v1 evidence and confirmation errors:none.
The unchanged v1 final gate still reports24 Supported-parent mandatory-child
blockers. Versioned v2 resolves ONLY those explicit all-child-excluded cases with
additional approval/exclusion checks. The original output is preserved, not
reported as PASS. See VALIDATOR_COMPATIBILITY_AND_SCHEMA_CHANGE.md. Final status
uses a new version so an old loader cannot silently consume it.

All known caption/omission/priority decisions are resolved. Business completeness
flags record the researcher's explicit package review; software does not claim
independent exhaustive semantic completeness. Exact duplicates in a common
business/domain/photo/polarity scope are rejected. Distinct attributes and
opposed customer opinions remain distinct; approved unit granularity is retained.
No silent merging/reclassification of the approved664 decisions was performed.

Offline regression scope:the preserved183 tests plus40 final-reference mutation/
integrity tests and5 conversation-handoff tests (228 total). Actual final-ZIP
extracted-run results and byte hash are recorded externally after packaging;
do not mistake a prior release-candidate receipt for the delivered final archive.
No browser smoke, new independent annotation, model inference or API use is
claimed by these tests. The initial construction diagnostics are retained locally;
an import check was corrected to compare a nonblank Uncertain suggested text
against the original annotation, while its approved correction stays null. No
original label or accepted artifact was changed to fix that software comparison.

## Freeze and next-stage readiness

FINAL_RESEARCHER_APPROVED_REFERENCES.json is the active frozen576-unit reference
file. REFERENCE_FREEZE_MANIFEST.json binds final data, decisions, approval, policy,
code and schemas with SHA-256. The ZIP has its own complete file manifest and an
external final byte-hash receipt. Old archives/receipts are preserved separately.
Scientific readiness:source references finalized under the approved curated
roles; independent review pending. D-054 must verify final manifest, new-format
validation compatibility, identical inputs and frozen generation configuration
before any separately authorized48-output inference. No inference/training or
Teacher regeneration occurred; no held-out test input was accessed.

'''+table
 write(OUT/'D053_FINAL_REFERENCE_ACCEPTANCE_REPORT.md',report)
 readiness={'status':'D053_REFERENCES_FINALIZED_AND_VALIDATED_V2_AWAITING_INDEPENDENT_REVIEW','businesses':24,'scoring_units':576,'core':261,'supporting':315,'excluded':88,'prior_parent_decisions_preserved':226,'uncertain_parents_excluded':2,'unresolved_researcher_decisions':0,'separate_attestation_required':False,
  'legacy_v1_final_gate':'EXPECTED_BLOCK_24_SUPPORTED_PARENTS_WITH_ALL_CHILDREN_APPROVED_NON_SCORING','v2_strict_approved_exclusion_gate':'PASS','all_original_evidence_and_confirmation_gates_preserved':True,
  'D054_preflight_requirement':'Explicit version-aware dispatch to strict v2 reference validator; keep frozen generation config/source hashes unchanged','D054_inference_authorized':False,'training_authorized':False,'teacher_calls_made':0,'held_out_access':False}
 write(OUT/'D053_FINAL_REFERENCE_READINESS.json',readiness)
 # Read-only review page; there are no label buttons, predictions or model calls.
 esc=lambda s:html.escape(str(s if s is not None else ''))
 page='<!doctype html><html lang="vi"><meta charset="utf-8"><title>D-053 finalized references</title><style>body{font:16px system-ui;max-width:1120px;margin:auto;padding:24px}summary{cursor:pointer;padding:10px}article{border:1px solid #ccd5df;padding:14px;margin:12px 0}blockquote{white-space:pre-wrap;font-size:14px;background:#f4f6f8;padding:12px}img{max-height:500px;max-width:100%}.mono{font:12px monospace;overflow-wrap:anywhere}.exclude{background:#f0f0f0}</style><h1>D-053: reference đã được researcher phê duyệt</h1><p>576 scoring units:261 Core +315 Supporting;88 Exclude chỉ lưu audit. Trang chỉ đọc, không yêu cầu gán nhãn lại. Phê duyệt được ghi nhận từ hội thoại, không giả lập thao tác UI.</p><p>Core không đòi hỏi mọi chi tiết xuất hiện nguyên văn. Thiếu Supporting không tự động là lỗi. Review là ý kiến khách hàng; các flag xác nhận là provenance của researcher, không phải phần mềm tự xác minh sự thật.</p>'
 for c in cases:
  bid=c['business_id'];rs=[a for a in audit if a['business_id']==bid];count=Counter(a['approved_reference_role'] for a in rs)
  page+=f'<details><summary>{esc(c["display_name"])} — Core {count["core"]} / Supporting {count["supporting"]} / Exclude {count["exclude"]}</summary><p class="mono">{esc(bid)} / {esc(c["subset"])}</p><details><summary>Toàn bộ selected reviews</summary>'
  for r in c['reviews']:page+=f'<article id="{bid}_{r["marker"]}"><b>{r["marker"]}</b><span class="mono"> {esc(r["review_id"])}</span><blockquote>{esc(r["text"])}</blockquote></article>'
  page+='</details><details><summary>Original captions + photos</summary>'
  for cap in c['captions']:page+=f'<article id="{bid}_{cap["caption_id"]}"><b>{cap["caption_id"]}</b><p class="mono">PHOTO {esc(cap["photo_id"])}</p><p>{esc(cap["text"])}</p><a href="{esc(cap["image_file"])}" target="_blank"><img loading="lazy" src="{esc(cap["image_file"])}" alt="{esc(cap["photo_id"])}"></a></article>'
  page+='</details>'
  for role in ['core','supporting','exclude']:
   page+=f'<details><summary>{role.upper()} ({count[role]})</summary>'
   for u in rs:
    if u['approved_reference_role']!=role:continue
    page+=f'<article class="{role}"><p><b>{esc(u["text"])}</b></p><p class="mono">{esc(u["proposal_id"])}</p><p>{esc(json.dumps(u["metadata"],ensure_ascii=False))}</p><p>{esc(u["reason"])}</p>'
    for e in u['evidence']:page+=f'<a href="#{bid}_{e["marker"]}">{esc(e["marker"])} / {esc(e.get("review_id",e.get("photo_id")))}</a> '
    page+='</article>'
   page+='</details>'
  page+='</details>'
 page+='</html>';write(OUT/'FINAL_REFERENCE_REVIEW.html',page)
 for c in cases:
  for cap in c['captions']:
   target=OUT/cap['image_file'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/cap['image_file'],target)
 # Freeze manifest is not self-referential; archive byte identity is external.
 important=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.txt']]
 code_names=['stage_i_d053_curated_references_v2.py','finalize_stage_i_d053_curated_references_v1.py','check_stage_i_d053_final_references_v2.py','test_stage_i_d053_final_references_v2.py','test_stage_i_d053_conversation_handoff_v1.js']
 freeze_manifest={'status':'FROZEN_RESEARCHER_APPROVED_REFERENCES_CURATED_V2','active_reference':'FINAL_RESEARCHER_APPROVED_REFERENCES.json','counts':{'core':261,'supporting':315,'excluded':88},'reference_format_version':2,
  'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(important)],
  'construction_and_validation_code':[{'path':'code/'+name,'sha256':sha(ROOT/'scripts'/name)} for name in code_names],
  'supplemental_schema_sha256':sha(ROOT/'schemas/stage_i_d053_curated_exclusion_extension_v2.schema.json'),
  'unchanged_original_schema_sha256':sha(SOURCE/'schemas/stage_i_d053_annotation_v1.schema.json'),
  'unchanged_original_validator_sha256':sha(SOURCE/'code/stage_i_d053_annotations.py'),
  'source_proposal_id':read(OUT/'RESEARCHER_APPROVAL_RECORD.json')['proposal_id'],'new_inference_authorized':False}
 write(OUT/'REFERENCE_FREEZE_MANIFEST.json',freeze_manifest)
 previous=ROOT/'outputs/stage_i/stage_i_d053_reference_proposals_review_bundle_v2.zip'
 with zipfile.ZipFile(previous) as z:
  for info in z.infolist():
   parts=PurePosixPath(info.filename).parts[1:]
   if not parts or info.is_dir() or '__pycache__' in parts:continue
   assert '..' not in parts and not PurePosixPath(*parts).is_absolute()
   rel=Path(*parts)
   if rel.as_posix() in ['README_FIRST.md','CHATGPT_REVIEW_REQUEST.md','SHA256_MANIFEST.json']:rel=Path('provenance/reference_proposals_v2')/rel
   q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(info))
 final=DEST/'final_references';final.mkdir()
 for p in OUT.iterdir():
  if p.is_file():shutil.copy2(p,final/p.name)
 shutil.copytree(OUT/'incoming',final/'incoming')
 shutil.copy2(OUT/'FINAL_REFERENCE_REVIEW.html',DEST/'FINAL_REFERENCE_REVIEW.html')
 for name in code_names+[Path(__file__).name]:shutil.copy2(ROOT/'scripts'/name,DEST/'code'/name)
 shutil.copy2(ROOT/'schemas/stage_i_d053_curated_exclusion_extension_v2.schema.json',DEST/'schemas/stage_i_d053_curated_exclusion_extension_v2.schema.json')
 write(DEST/'README_FIRST.md','''# D-053 final researcher-approved references — start here

Active reference: final_references/FINAL_RESEARCHER_APPROVED_REFERENCES.json.
576 units:261 Core +315 Supporting.88 exclusions are audit-only.24 validation
Filtered businesses; no held-out test. No predictions, training or Teacher calls.

Read final_references/D053_FINAL_REFERENCE_ACCEPTANCE_REPORT.md and open
FINAL_REFERENCE_REVIEW.html for a read-only source/evidence view. No annotation
redo or extra attestation is required for the explicit conversation approval.
Review source is the actual user instruction, not attachment instructions or
fabricated UI actions. Original664 proposals/curation flags and226 parent
decisions remain unchanged in their historical files.

IMPORTANT VERSION DISCLOSURE:the unchanged v1 final validator still blocks24
Supported parents with all children approved non-scoring. V1 structural/evidence
checks pass. A new strict approved-admissibility v2 bridge verifies every explicit
exclusion and resolves ONLY those mandatory-child pending messages. All other
gates remain fatal. Read final_references/VALIDATOR_COMPATIBILITY_AND_SCHEMA_CHANGE.md.
The old evaluator is not silently patched; before separately authorized D-054,
preflight must explicitly load this new reference version through its validator.

Offline validation (Python3.11+,jsonschema,numpy and Node):

```
python code/check_stage_i_d053_final_references_v2.py --root .
python -m unittest discover -s code -p test_stage_i_d053_final_references_v2.py -v
node code/test_stage_i_d053_conversation_handoff_v1.js .
```

Run the preserved183 tests with the included proposal-archive verifier and the
40 new reference tests +5 handoff tests (228 total). The final extracted-archive
receipt is external and binds the exact delivered ZIP; historical pre-finalization
reports in provenance folders describe their own archived states, not current
acceptance. No automated test is independent human factual ground truth.

Core content is important, not mandatory verbatim detail. Supporting omission
is not automatically an error. Excluded units never enter scoring. Do not use
all664 as a denominator. SHA256_MANIFEST.json covers all files except itself;
final_references/REFERENCE_FREEZE_MANIFEST.json binds the accepted data/approval.
The final ZIP byte hash belongs to its external receipt/sidecar, not a self-hash.

STOP:independent review and separate D-054 authorization are still required.
''')
 write(DEST/'CHATGPT_REVIEW_REQUEST.md','''# Independently review finalized D-053 references

The researcher explicitly approved the attached curated recommendations in the
current conversation:664 decisions =261 Core,315 Supporting,88 Exclude;15 caption
choices =10 safe revisions,5 exclusions;5 omissions =3 Core,1 Supporting,1 Exclude.
All226 prior parent decisions and2 uncertain exclusions remain preserved.

Verify the attached original archive, proposal IDs/file hashes, approval message
and provenance, final576 units,88 audit-only exclusions, exact evidence mappings,
caption-plus-photo scope, sentiment, metadata, omissions, per-business counts,
and no automatic664 denominator. Inspect read-only FINAL_REFERENCE_REVIEW.html.
Core does not require each granular detail verbatim; Supporting omission is not
automatically an error; review statements remain attributed customer opinions.

Please specifically review the versioned representational extension:the unchanged
v1 final gate requires children for all Supported parents and therefore blocks24
all-child-excluded Supported parents. The explicit approved-admissibility v2
bridge checks complete source-bound exclusions and resolves only that exact
pending condition; it does not relabel parents, invent scoring units, weaken
evidence gates or patch the frozen evaluation configuration. The raw v1 BLOCK
and v2 PASS are both preserved. Three caption-safe wordings are approved but
remain importance-excluded; these are different axes, not contradictions.

Assess whether the new freeze is valid for independent approval and a future
version-aware D-054 preflight. The unmodified v1 loader intentionally refuses
the new reference format. Software PASS and researcher confirmation are not
independent machine factual truth. Original proposal/curation files remain
unapproved as historical inputs; only the new approval record materializes the
explicit user instruction. No extra664-unit review should be demanded unless
a concrete unresolved issue is found.

No Pilot A/B inference, training, Teacher calls or held-out access is authorized.
''')
 print(json.dumps({'prepared':str(DEST),'active_reference':sha(OUT/'FINAL_RESEARCHER_APPROVED_REFERENCES.json')}))
def pack():
 assert DEST.is_dir();manifest_path=DEST/'SHA256_MANIFEST.json'
 rows=[{'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file() and p!=manifest_path and '__pycache__' not in p.parts]
 write(manifest_path,{'scope':'Every payload file except this manifest itself; final ZIP byte identity is external','files':rows})
 archive=DEST.with_suffix('.zip');assert not archive.exists(),'Do not overwrite a delivered archive'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(DEST.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,NAME+'/'+p.relative_to(DEST).as_posix())
 digest=sha(archive);write(archive.with_suffix('.zip.sha256'),digest+'  '+archive.name+'\n')
 write(OUT/'DELIVERED_ZIP_RECEIPT.json',{'archive':str(archive.relative_to(ROOT)).replace('\\','/'),'sha256':digest,'bytes':archive.stat().st_size,'active_reference_sha256':sha(OUT/'FINAL_RESEARCHER_APPROVED_REFERENCES.json'),'counts':{'core':261,'supporting':315,'excluded':88},'final_extracted_test_report':'FINAL_ZIP_TESTS/COMPLETE_D053_FINAL_REFERENCE_TEST_SUMMARY.json','new_inference_authorized':False})
 print(json.dumps({'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,'files':len(rows)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--pack',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 if a.pack:pack()
