"""Complete the read-only cross-split investigation, current audit docs and small review ZIP."""
from pathlib import Path
import json,csv,hashlib,collections,datetime,shutil,zipfile,tempfile,difflib
from audit_cross_split_images_fresh_v1 import pixel_info
R=Path(__file__).resolve().parents[1];O=R/'outputs/data_integrity/cross_split_images_s1000_v1';E=O/'EVIDENCE';M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def text(n,t):(O/n).write_text(t.strip()+'\n',encoding='utf8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf8')))
def csvout(p,data):
 with p.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
def table(head,rr):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in r)+' |\n' for r in rr)
assert not (O/'CROSS_SPLIT_IMAGE_DUPLICATION_AUDIT.md').exists(),'Preserve completed audit'
fresh=read(E/'FRESH_RECOMPUTATION_RECEIPT.json');exp=read(E/'EXPOSURE_SUMMARY.json');hf=read(E/'HF_ORIGINAL_JPEG_VERIFICATION.json');pm=read(E/'ORIGINAL_YELP_MATCHED_PHOTO_METADATA.json');bm=read(E/'ORIGINAL_BUSINESS_METADATA_VERIFICATION.json');mr=read(E/'HF_ORIGINAL_METADATA_SHARD_RECEIPT.json')
assert fresh['photos_fresh_decoded']==30027 and fresh['cached_pixel_results_used_to_construct_matches'] is False
assert hf['status']=='PASS_ALL_MATCHED_ORIGINAL_JPEG_MEMBERS' and pm['status']=='PASS_SOURCE_ASSIGNMENTS_MATCH_CURRENT_COHORT' and bm['all_current_identity_fields_equal_original']
assert mr['status']=='PASS_FRESH_REMOTE_ARCHIVE_SHA_AND_ORIGINAL_PHOTO_JSON'
orig={r['photo_id']:r for r in hf['records']};meta={r['photo_id']:r for r in pm['records']};current=rows(E/'PAIR_INPUT_AND_SUPERVISION_FLAGS.csv');fs=read(E/'FILESYSTEM_OBSERVATIONS.json');vol=read(E/'CURRENT_VOLUME.json');business=rows(E/'BUSINESS_IDENTITY_COMPARISON.csv')
assert len(current)==490 and len(orig)==541 and len(meta)==541 and bm['matched_business_records']==113
original_pixels={}
for pid,r in orig.items():
 p=Path(r['local_evidence_path']);assert sha(p)==r['member_sha256'];original_pixels[pid]=pixel_info(p.read_bytes())[0]
matched=[]
for row in current:
 r=dict(row);t=orig[row['test_photo_id']];v=orig[row['other_photo_id']];ti=original_pixels[row['test_photo_id']];vi=original_pixels[row['other_photo_id']]
 assert t['member_sha256']==v['member_sha256']==row['test_file_sha256']==row['other_file_sha256']
 assert ti['pixel_sha256']==vi['pixel_sha256']==row['test_pixel_sha256']
 assert meta[row['test_photo_id']]['business_id']==row['test_business_id'] and meta[row['other_photo_id']]['business_id']==row['other_business_id']
 assert row['physical_samefile']=='False' and row['resolved_paths_equal']=='False'
 r['TEST_F_basic_eligible_before_full_filter']=r.pop('TEST_F_candidate')
 r.update(cause_class='CONFIRMED_SOURCE_DATA_DUPLICATION',cause_basis='Distinct original JPEG tar members and original raw photo metadata freshly verified from pinned HF snapshot; processed bytes/assignments agree',
   HF_repository=hf['repo_id'],HF_revision=hf['revision'],test_original_archive=t['archive'],other_original_archive=v['archive'],
   test_original_member=t['source_member'],other_original_member=v['source_member'],test_original_header_offset=t['header_offset'],other_original_header_offset=v['header_offset'],
   test_original_file_sha256=t['member_sha256'],other_original_file_sha256=v['member_sha256'],test_original_pixel_sha256=ti['pixel_sha256'],other_original_pixel_sha256=vi['pixel_sha256'],
   test_original_width=ti['width'],test_original_height=ti['height'],other_original_width=vi['width'],other_original_height=vi['height'],
   test_original_HF_locator=f"hf://datasets/{hf['repo_id']}@{hf['revision']}/{t['archive']}::{t['source_member']}",other_original_HF_locator=f"hf://datasets/{hf['repo_id']}@{hf['revision']}/{v['archive']}::{v['source_member']}",
   test_restored_original_evidence=t['local_evidence_path'],other_restored_original_evidence=v['local_evidence_path'],
   original_photo_business_assignment_verified=True,preprocessing_changed_image_bytes=False,current_mount_or_path_aliasing_observed=False)
 matched.append(r)
csvout(O/'CROSS_SPLIT_MATCHED_IMAGE_PAIRS.csv',matched)
br=rows(O/'TEST100_DUPLICATE_IMAGE_EXPOSURE.csv');assert len(br)==100 and len({r['test_business_id'] for r in br})==100
affected=[r for r in br if int(r['duplicate_photo_ids'])]
assert len(affected)==17 and sum(int(r['duplicate_photo_ids']) for r in br)==123
un=exp['conditions']['U'];fn=exp['conditions']['F'];count_hashes=len({r['test_pixel_sha256'] for r in matched})
exposure_table=table(['Measure','U','F'],[
 ['All selected TEST caption/photo entries',500,341],['Selected TEST duplicate photo IDs',un['duplicated_TEST_selected_photos'],fn['duplicated_TEST_selected_photos']],
 ['Businesses with selected duplicate photos',un['TEST_businesses_with_selected_duplicates'],fn['TEST_businesses_with_selected_duplicates']],
 ['Selected TEST duplicate photos also in own-condition TRAIN source',un['own_TRAIN_source_overlap_photos'],fn['own_TRAIN_source_overlap_photos']],
 ['Businesses with that own-condition TRAIN source overlap',un['own_TRAIN_source_overlap_businesses'],fn['own_TRAIN_source_overlap_businesses']],
 ['Selected TEST duplicates with own TRAIN retained Visual-label support',un['own_TRAIN_target_supported_overlap_photos'],fn['own_TRAIN_target_supported_overlap_photos']],
 ['Selected TEST duplicates with exact same caption text in own TRAIN',un['same_caption_TRAIN_overlap_photos'],fn['same_caption_TRAIN_overlap_photos']],
 ['Own-condition VALIDATION-only overlap among selected TEST duplicates',un['own_VALIDATION_only_overlap_photos'],fn['own_VALIDATION_only_overlap_photos']]])
brief=table(['TEST business','Raw duplicate photos','U selected duplicates','F selected duplicates','U also own TRAIN source','F also own TRAIN source'],[
 [r['business_name'],r['duplicate_photo_ids'],r['duplicate_U_selected_count'],r['duplicate_F_selected_count'],r['U_own_condition_TEST_input_and_TRAIN_input_overlap_photos'],r['F_own_condition_TEST_input_and_TRAIN_input_overlap_photos']] for r in affected])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=f'''# Cross-split image duplication — independent read-only audit

Audit completed {now}. **The complete frozen TEST100 is retained. No business, split, review, photo ID, caption, filtering rule, checkpoint or Shuffled policy was changed. No TEST prediction, reference or quality metric was generated/read.**

## Direct answers

**Q1. Duplicate businesses or certain duplicate images?** The fresh comparison finds17TEST businesses with123distinct TESTphoto IDs participating in490TEST-to-TRAIN/VALIDATIONimage pairs, covering108distinct byte/pixel content groups and418other photo IDs. There are127business-pair relationships, not490duplicated businesses. All business IDs remain split-disjoint. Original business metadata for113involved records was independently retrieved from HF and matches the current cohort. No matched business pair shares a normalized full address; the minimum coordinate separation is8,914.93m.112/127pairs have equal normalized names, consistent with separate locations of similarly named businesses. These records do not establish duplicate business entities; what is proven is identical images under different photo IDs associated with separate business records/locations. This is not a legal-entity or historical-relocation investigation.

**Q2. Original Yelp image duplication or pipeline/mount error?** All490pairs are byte-identical and full-resolution normalized-RGB-pixel-identical. The541distinct original JPEGmembers involved were freshly fetched from the private HF original-source backup. Each header/member path/size/SHA and decoded pixels matches its processed copy. Original raw `photos.json` assignments were independently recovered and match every involved photo/business pair. Original business identity fields also match. Hence all490are classified **confirmed source-data duplication in the preserved original-Yelp-source snapshot**,0confirmed preprocessing errors,0confirmed current mount/path aliases,0unresolved pair origins within the available evidence. The reason identical images were posted/assigned to multiple businesses is not established; no claim of photographer/corporate-upload intent or independent authentication against a newly downloaded Yelp release is made.

**Q3. How many are actually selected TESTinputs?** Uselects66duplicate photos across17businesses; Fselects44across16. These are distinct selected photo IDs, not pair counts.59Uphotos and42Fphotos also have same-condition TRAINcaption counterparts, exact caption text matches, and retained TRAINVisual-label support; both conditions have16businesses with such direct input overlap.42of123raw duplicate TESTphotos are selected by neither U nor F. Raw overlap therefore overstates direct input exposure, but direct exposure is substantial and confirmed. See the counts and complete100-row ledger below.

**Q4. Scientific validity and impact?** Business-level split independence remains intact. Photo-level and individual-caption-level independence do not hold for the identified exposures. LEDreceives textual captions, not raw images; thus the material exposure is reused visual evidence/caption strings in own-condition TRAINand TESTinputs. These overlaps are a plausible route to easier Visual prediction and reduced independence of a photo-grounding assessment. Actual memorization, Teacher mimicry or inflated TESTscores are **not demonstrated**: no TESToutputs or scores exist in this audit. VALIDATIONoverlap is reported separately and is not mislabeled optimizer training. ROhas no experimental image/caption input and no corresponding direct visual-input exposure; language-pretraining or review-content overlap is outside this photo audit.

**Q5. Can complete TEST100 be retained?** Yes: follow the researcher's explicit decision and retain the exact frozen cohort/inputs, presenting it as a business-held-out evaluation with a disclosed photo/caption-reuse limitation. Do not describe its Visual evidence as wholly unseen/independent. Do not remove the17, create a TEST83replacement, retrain, retune filters or alter Shuffled policy. The pre-existing17/83diagnostic remains historical evidence only; no new83-case dataset was created. Any later exposure-stratified descriptive analysis must be specified in the evaluation protocol within the retained100, not chosen from model outcomes. The overall evaluation may remain informative for the stated business-level task, while claims of generalization to entirely novel visual evidence must be qualified.

## Fresh reproduction

All30,027actual canonical JPEGfiles were read and fully decoded anew:26,387TRAIN,2,504VALIDATION,1,136TEST. RawSHA-256was checked against the accepted copy manifest. Full-resolution EXIF-transposed RGBbytes and dimensions were hashed with the declared framing `EXIF_RGB_dimensions_v1:` plus8-byte-big-endian width and height. No resize, pHashthreshold, cached StageDpixel values or manual visual-similarity classifier was used. Pair pixels were also compared directly, not only by hash. Results:490byte+pixel-identical pairs(454TRAIN,36VALIDATION),0pixel-only pairs. The independent affected-business set exactly matches the prior17; cached diagnostics were read only after constructing fresh matches. This audit does not exhaustively search near-duplicate/visually-similar images and makes no absence claim about them.

HFsource: `{hf['repo_id']}` at revision `{hf['revision']}`; original snapshot created `{hf['original_source_snapshot_created_UTC']}`. The local53,505,783-byte backup manifest matches the pinned repository LFS SHA `{hf['backup_manifest_sha256']}`. For uncompressed original-JPEGshards, only the541needed tar headers/member bodies were retrieved by HTTP206ranges; every returned range/header/path/size/member SHA was verified. FullJPEGshard archives were not freshly downloaded/rehashed. One779,670,344-byte compressed metadata shard was streamed without saving the full archive; its complete archive SHAand original25,653,907-byte `photos.json` SHAwere verified. A29,076,481-byte original-business shard was streamed and verified, retaining only113relevant business records. Remote originals were recovered into this new audit evidence directory; the absent original local path was not recreated or overwritten. No HFpublication/authentication settings changed.

Current filesystem: zero same-physical-file matching pairs, zero repeated(dev,inode)groups across30,027processed images, zero symlink/reparse files, and no multi-hardlink files. Resolved paths are distinct and correctly photo-IDnamed. Relevant ancestor directories are not reparse points; all use the observed local NTFSvolume. Equal content on the same volume is not evidence of a mount error. Source metadata and byte identity rule out an image-copy or business-IDmapping discrepancy for these cases; historical storage configurations beyond the available source/code evidence were not independently replayed.

## Actual frozen input/supervision exposure

{exposure_table}

All123duplicate TESTphotos are Basic-eligible Ucandidates, but only66enter U's frozen K<=5selection. Fselects44, using its unchanged frozen ranking/filtering.29photos are selected in both conditions;81are selected in at least one;42in neither. Selected duplicate shares: U66/500=13.20% of caption entries; F44/341=12.90%. Own-TRAINoverlap shares: U59/500=11.80%; F42/341=12.32%. These are descriptive exposure fractions, not error rates or estimates of performance inflation.

Across all123duplicate TESTphotos,93have any U-selected TRAINcounterpart and60have any F-selected TRAINcounterpart, regardless of whether the TESTphoto is itself selected. Those raw-side counts must not be confused with59/42actual selected own-condition overlaps. U-selected VALIDATIONcounterparts exist for21raw duplicate TESTphotos and Ffor10; they are separately traceable. No selected TESTduplicate has **only** own-condition VALIDATIONexposure without own TRAINexposure; the zero in that row does not mean no VALIDATIONoverlap exists.

Source/target trace: frozen TRAIN/VALIDATIONU/Fdataset file hashes and selected photo IDs match StageF. `visual_claims_with_evidence` support links resolve to caption strings/photo IDs; their immutable original target artifacts are hash-bound. Actual retained1,000-update window logs confirm two successful Visual presentations for each involved TRAINbusiness in its own U/Fcondition. The original750parent source exposure is inherited through frozen own-condition inputs and source/sampler-derived accounting, not invented direct historical photo-IDjournals. No direct TRAINclaim is made for VALIDATIONrecords. A paired photo used only in another condition's training is not automatically treated as exposure for this model.

{brief}

For example, Fulin's Asian Cuisine has13raw duplicate TESTphotos,3selected by U, but0selected by F. Only1of those Uselected duplicates has an own-U TRAINselected counterpart. This demonstrates why raw photo-pool counts and final model-input counts need separate columns.

## Evidence and reproduction

- `CROSS_SPLIT_MATCHED_IMAGE_PAIRS.csv`: all490pairs, IDs/splits/local+HFsource/processed paths, file+pixel hashes, dimensions, byte/pixel classes, physical IDs/link checks, original metadata/cause and model/Teacher/label exposure flags.
- `TEST100_DUPLICATE_IMAGE_EXPOSURE.csv`: exactly100unchanged business IDs, raw duplicate/candidate/selected U/Fcounts and same-condition TRAIN/VALIDATIONexposure.
- `EVIDENCE/FRESH_IMAGE_INVENTORY.csv`: fresh all30,027file/pixel/filesystem measurements. `PER_TEST_DUPLICATE_PHOTO_EXPOSURE.csv` provides the complete123-photo detail, so many-to-many pairs cannot inflate counts.
- `HF_ORIGINAL_JPEG_VERIFICATION.json`, original raw photo/business metadata receipts and `TRAIN_VALIDATION_TARGET_SUPPORT_RECORDS.json` preserve the source chain. Raw original JPEGs and original `photos.json` are retained locally outside the small review ZIP.

Fresh local reproduction into a new evidence directory (refuses overwrite):

```powershell
.venv-caption-pilot/Scripts/python.exe -X utf8 -B scripts/audit_cross_split_images_fresh_v1.py --output-directory outputs/data_integrity/cross_split_images_recheck
```

The delivered scripts `verify_cross_split_hf_originals_v1.py`, `verify_cross_split_original_business_metadata_v1.py` and `audit_cross_split_model_exposure_v1.py` show exact read-only pinned-source/exposure verification. They target this audit namespace and intentionally refuse to replace existing fetched image evidence/CSVoutputs; inspect receipts rather than blindly rerunning into completed outputs. HFGETuses existing authenticated access without recording credentials. The complete input/script hash manifest accompanies this package.

Confirmed facts and risk interpretation are separated throughout. FullTEST100 remains unchanged; no reference construction, inference, metric computation, filtering/caption generation, training, Shuffled mapping or GPU activity followed this investigation. STOP for researcher review.
'''
text('CROSS_SPLIT_IMAGE_DUPLICATION_AUDIT.md',report)
text('MOUNT_AND_PATH_VERIFICATION.md',f'''# Mount/path investigation

Observed local volume: `{vol['UniqueId']}`, drive `{vol['DriveLetter']}`, filesystem `{vol['FileSystem']}`. Volume status is `{vol['HealthStatus']}`/`{vol['OperationalStatus']}` at observation; it is not a historical physical-drive diagnostic.

The fresh pass obtained `lstat`, resolved path, device ID, file ID/inode, link count and Windows reparse attributes for all30,027processed photo files and their relevant path ancestors. All photo stems match frozen `photo_id`values; each canonical path resolves inside the cohort directory. No photo symlink/reparse flags, multi-hardlink files or repeated(dev,inode)groups were observed. Every one of490matching TEST/other pairs has distinct resolved paths and `os.path.samefile=False`. Pair-level identifiers are included in the complete CSV. Ancestor reparse/symlink checks are recorded in FILESYSTEM_OBSERVATIONS.json. The common NTFSvolume and common photos directory are expected storage layout, not proof of aliasing.

Current raw local `data/yelp_photos/photos` is absent; no cause was inferred from that absence. The researcher identified the existing private HFbackup. Fresh HTTPRange retrieval recovered541original JPEGmembers from distinct photo-ID paths/tar offsets; original byte/pixel hashes match processed files. The original raw photo metadata and business records were independently streamed and hash-verified. Photo/business assignments and113original business identity records agree with current cohort metadata. This supplies independent original-source evidence despite local raw files being absent.

The actual StageBcopy implementation uses `shutil.copyfile`, hashes source and destination, and preserves metadata assignment. Its implementation hash matches the historical dataset manifest in the preservation checks. The available original snapshot and current byte/metadata equality establish no copying/indexing discrepancy for these matches. No current mount/path aliasing is found. This does not claim a complete forensic reconstruction of every historical mount or the underlying HFstorage filesystem; HFarchives preserve distinct named members with identical contents, regardless of backend deduplication.

Classification for the490investigated pairs: confirmed original-source content duplication490; confirmed preprocessing/copy error0; confirmed mount/path aliasing0; unresolved pair origin0within the preserved-source scope. The reason images were reused under different YelpIDs remains unestablished. No unrelated storage cleanup command was executed, and no original files or manifests were changed.
''')
text('DUPLICATION_CAUSE_AND_IMPACT_SUMMARY.md',f'''# Duplication cause and scientific impact

**Retain the complete frozen TEST100. No TEST83replacement or data/model modification.**

- Fresh actual-file recomputation:17affected TESTbusinesses,123TESTphoto IDs,490cross-split pairs(454TRAIN/36VALIDATION),108underlying content groups. All490pairs are byte- and pixel-identical, not merely visually similar. No same-physical-file pair.
- Source cause confirmed against541original JPEGmembers plus original photo/business metadata in the pinned private HFsnapshot. Processed bytes/paths/metadata agree. Source duplication is established; posting intent and independent upstream Yelp-distribution authenticity are not additionally established.
- Business split independence remains:2,000/200/100IDs disjoint,127matched business pairs have0same full addresses and minimum coordinate distance8.915km. Duplicate image content does not demonstrate duplicate business entities.

{exposure_table}

Own-condition TRAINoverlap of actual selected input captions is confirmed, including exact caption-string reuse for59U/42Fphotos and retained Visual-label support. This is not just an unused raw-pool issue. Conversely42raw duplicate TESTphotos never enter either final U/Finput. VALIDATIONis development exposure, not optimizer training. ROhas no direct image/caption input; LEDsees captions rather than pixels.

Scientific conclusion: the retained100is business-held-out, but its Visual component is partly photo/caption-reused. A plausible leakage/generalization limitation must be disclosed with raw-pool and actual-input counts, condition-specific TRAINand VALIDATIONdistinctions, and original-source provenance. Performance inflation, memorization and Teacher mimicry are unmeasured and must not be inferred from these counts. No TESToutput/reference/quality metric was used. Any later exposure-stratified reporting requires protocol review within the frozen100; do not redefine the benchmark or optimize subgroups from results. Shuffled policy is unchanged. Stop after this audit.
''')

# Verify original target artifacts referenced by the source/label evidence, without any TEST target.
support=read(E/'TRAIN_VALIDATION_TARGET_SUPPORT_RECORDS.json');targets={r['target_artifact']:r['target_artifact_sha256'] for r in support}
for rel,h in targets.items():assert sha(R/rel)==h
models=read(R/'outputs/pre_evaluation/s1000_readiness_v1/EVIDENCE/MODEL_FREEZE_AUDIT.json')['models']
for m in models.values():assert sha(R/m['path']/'adapter_model.safetensors')==m['adapter_sha256']
stage=R/'outputs/stage_f/production_caption_construction_d033_colab_v1'
for rel,h in read(stage/'completion_report.json')['artifact_sha256'].items():assert sha(stage/rel)==h
ds=read(R/'data/processed/yelp_2300_v1/dataset_manifest.json');assert sha(R/'scripts/build_yelp_cohort.py')==ds['implementation_sha256']['build_yelp_cohort.py']
assert sha(R/'data/processed/yelp_2300_v1/business_split.csv')==ds['business_split_sha256']
preserve=dict(status='PASS_UNCHANGED_FROZEN_ARTIFACTS',model_adapter_hashes={a:m['adapter_sha256'] for a,m in models.items()},StageF_manifests_reverified=19,TRAIN_VALIDATION_target_artifacts_reverified=len(targets),business_split_unchanged=True,stageB_copy_code_matches_creation_hash=True,
 Shuffled_policy_sha256=sha(R/'outputs/stage_i/led_large_matched_signoff_v3/SHUFFLED_POLICY_PROPOSED_UNCHANGED.json'),existing_image_overlap_policy_sha256=sha(R/'configs/image_overlap_evaluation_policy.yaml'),old_StageD_summary_sha256=sha(R/'outputs/stage_d/photo_integrity_duplicate_audit_v1/test_image_disjoint_diagnostic_summary.json'),
 TEST_references_read=0,TEST_predictions_generated=0,TEST_quality_metrics=0,filtering_or_caption_generation=0,new_training_or_GPU=0,cohort_removals=0,new_TEST83_replacement=False)
save(E/'FROZEN_ARTIFACT_PRESERVATION.json',preserve)
save(E/'CAUSE_AND_IMPACT_RECEIPT.json',dict(status='COMPLETE_READONLY_INVESTIGATION',UTC=now,cause_counts=dict(confirmed_source_data_duplication=490,confirmed_preprocessing_error=0,confirmed_mount_alias=0,unresolved_pair_origin=0),fresh=fresh,exposure=exp,HF_repository=hf['repo_id'],HF_revision=hf['revision'],original_source_scope='Preserved pre-training original-Yelp-source snapshot; not newly downloaded upstream Yelp distribution',complete_frozen_TEST100_retained=True))

# Update current researcher-facing audit status without rewriting historical evidence or frozen policy.
docs=['thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','DECISIONS.md'];changes=[]
note=f'''## Cross-split image integrity update — 2026-10-28

An independent read-only pass decoded all30,027images and verified matched originals against the pinned private HFbackup.17TESTbusinesses contain123photo IDs forming490byte/pixel-identical pairs with TRAIN/VALIDATION. Distinct original photo IDs/bytes and raw photo/business metadata confirm source-data duplication; no current filesystem alias or processed-copy/assignment error was found for these pairs. Uselects66duplicate TESTphotos, F44;59U/42Falso occur as own-condition TRAINinput captions with retained Visual-label support. These counts establish photo/caption-level reuse, not memorization or inflated TESTscores. Business IDs remain disjoint; the17are not established duplicate businesses. Original posting intent remains unknown.

**Latest researcher decision: retain the complete frozen TEST100, unchanged. Do not create a TEST83replacement.** Prior17/83files remain historical diagnostics; they do not replace TEST100. The benchmark is business-held-out with a disclosed partial visual-evidence reuse limitation. Final protocol must distinguish raw pools from selected inputs and TRAINfrom VALIDATION. No TESTinference/reference/quality metric, filtering, caption generation, checkpoint change or Shuffled-policy change occurred.

Audit reports, complete490-pair table and100-business exposure ledger: `outputs/data_integrity/cross_split_images_s1000_v1/`. Stop for researcher review before any further evaluation stage.

'''
# Readability of the newly authored note only; never alter quoted source evidence or existing logs.
for a,b in [('all30','all 30'),('027images','027 images'),('HFbackup','HF backup'),('17TEST','17 TEST'),('123photo','123 photo'),('490byte','490 byte'),('Uselects66','U selects 66'),('F44','F 44'),('59U','59 U'),('42F','42 F'),('TRAINinput','TRAIN input'),('TESTphotos','TEST photos'),('TESTscores','TEST scores'),('TEST83replacement','TEST83 replacement'),('prior17','prior 17'),('complete490','complete 490'),('100-business','100-business'),('TRAINfrom','TRAIN from'),('TESTinference','TEST inference')]:note=note.replace(a,b)
for name in docs:
 p=R/name;before=p.read_bytes();backup=O/'DOCUMENTATION_BEFORE'/name;backup.parent.mkdir(parents=True,exist_ok=True);assert not backup.exists();backup.write_bytes(before)
 if name=='DECISIONS.md':after=('## IMAGE-AUDIT-20261028 — source duplication verified; retain TEST100\n\n'+note.split('\n\n',1)[1]).encode('utf8')+before
 else:
  s=before.decode('utf8');heading_end=s.find('\n\n');s=s[:heading_end+2]+note+s[heading_end+2:]
  if name.endswith('THESIS_PROGRESS.md'):
   s=s.replace('Preserve the previously approved main TEST100 and secondary 83-business exact-image-disjoint analysis; business disjointness alone does not imply image independence.','Retain complete TEST100 under the latest researcher decision; prior17/83image flags are diagnostics, not a TEST83 replacement. Business disjointness alone does not imply image independence.')
  after=s.encode('utf8')
 p.write_bytes(after);diff=''.join(difflib.unified_diff(before.decode('utf8').splitlines(True),after.decode('utf8').splitlines(True),fromfile=name+' (before)',tofile=name+' (after)'))
 dp=O/'DOCUMENTATION_DIFFS'/Path(name+'.patch');dp.parent.mkdir(parents=True,exist_ok=True);dp.write_text(diff,encoding='utf8')
 changes.append(dict(path=name,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=sha(p),snapshot=backup.relative_to(O).as_posix()))
save(E/'DOCUMENTATION_CHANGE_MANIFEST.json',changes)

# Package all small reports and reproducible tables, excluding recovered raw JPEGs and full raw JSON.
P=O/'REVIEW_PACKAGE';P.mkdir(exist_ok=False)
for p in O.iterdir():
 if p.is_file() and p.suffix in ['.md','.csv']:cp(p,P/p.name)
for p in E.iterdir():
 if p.is_file():cp(p,P/'EVIDENCE'/p.name)
for name in ['audit_cross_split_images_fresh_v1.py','verify_cross_split_hf_originals_v1.py','verify_cross_split_original_business_metadata_v1.py','audit_cross_split_model_exposure_v1.py','finalize_cross_split_image_audit_v1.py']:cp(R/'scripts'/name,P/'SCRIPTS'/name)
for c in changes:
 if c['path'].startswith('thesis_sdd_docs/'):cp(R/c['path'],P/'UPDATED_DOCUMENTATION'/c['path'])
cp(R/'configs/image_overlap_evaluation_policy.yaml',P/'HISTORICAL_POLICY_UNCHANGED/image_overlap_evaluation_policy.yaml')
(P/'README.md').write_text('''# Read-only cross-split image duplication audit

Read CROSS_SPLIT_IMAGE_DUPLICATION_AUDIT.md for Q1–Q5, then the complete490-pair CSV and100-business exposure CSV. EVIDENCE includes the fresh all30,027-file pixel/byte/filesystem ledger,123-photo exposure ledger, pinned HFsource/member/hash receipts, original metadata verification and TRAIN/VALIDATIONtarget support links. No TEST reference/output/quality data or new83-business cohort exists here. Source originals are recovered only into local audit evidence, not into frozen dataset paths.

All490pairs are confirmed source-data duplication within the preserved original-source HFsnapshot. This does not establish why images were posted to multiple businesses or prove performance inflation/memorization. U66/F44selected photos have duplicates; U59/F42also have own-condition TRAINcaption/label-supported exposure. Many-to-many pair counts must not inflate distinct photo/business counts. VALIDATIONis not optimizer training. LEDinput is text captions, not image pixels.

Original JPEG evidence and full photos.json are retained locally outside this small ZIP. Credentials/signed download URLs, checkpoint binaries and bulk original datasets are excluded. Scripts perform only CPU image/text/hash processing and authenticated HFGET. The fresh local recomputation supports a new output directory; never rerun into existing evidence or edit the original dataset. Exact HFsnapshot revision and member SHA bindings are recorded; JPEGshards were range-read, not freshly full-shard verified. Original metadata shards were fully streamed and hash-verified.

Current thesis_sdd_docs files have a concise integrity update. Exact before-documents and diffs remain locally outside the small ZIP; their hashes are in the change manifest. The older83-case diagnostic policy file is included explicitly as historical and unchanged, not a new replacement cohort. Complete frozen TEST100 is retained per latest researcher decision. No filtering/model/Shuffled-policy changes, inference, references or evaluation were performed. STOP for researcher review.
''',encoding='utf8')
files=sorted(p for p in P.rglob('*') if p.is_file());manifest=[dict(filename=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
with (P/'PACKAGE_MANIFEST.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=['filename','bytes','sha256']);w.writeheader();w.writerows(manifest)
Z=O/'CROSS_SPLIT_IMAGE_DUPLICATION_S1000_REVIEW.zip';assert not Z.exists()
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(P.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(P).as_posix())
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
 with tempfile.TemporaryDirectory(prefix='image_audit_zip_check_',dir=O) as tmp:
  t=Path(tmp);z.extractall(t)
  for m in manifest:assert (t/m['filename']).stat().st_size==m['bytes'] and sha(t/m['filename'])==m['sha256']
  assert (t/'PACKAGE_MANIFEST.csv').read_bytes()==(P/'PACKAGE_MANIFEST.csv').read_bytes()
  for p in t.rglob('*.json'):read(p)
  assert len(rows(t/'CROSS_SPLIT_MATCHED_IMAGE_PAIRS.csv'))==490 and len(rows(t/'TEST100_DUPLICATE_IMAGE_EXPOSURE.csv'))==100
receipt=dict(status='PASS_FRESH_INPUTS_ORIGINAL_SOURCE_EXPOSURE_AND_ZIP_INTEGRITY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=Z.name,bytes=Z.stat().st_size,sha256=sha(Z),payload_files=len(manifest),ZIP_members=len(manifest)+1,
  CRC_reopen_extract_parse_and_every_member_SHA256=True,matched_pairs=490,affected_TEST_businesses=17,duplicated_TEST_photo_ids=123,U_selected_duplicate_photos=66,F_selected_duplicate_photos=44,U_own_TRAIN_input_and_label_overlap_photos=59,F_own_TRAIN_input_and_label_overlap_photos=42,
  original_JPEG_members_verified=541,original_business_metadata_records_verified=113,cause='CONFIRMED_ORIGINAL_SOURCE_SNAPSHOT_DUPLICATION',current_filesystem_alias_pairs=0,TEST100_retained=True,TEST83_replacement_created=False,TEST_predictions_or_quality_metrics=0,
  scientific_limitation='Business-held-out evaluation with confirmed partial visual evidence/caption reuse; performance impact unmeasured',documentation_files_updated=5)
save(O/'PACKAGE_INTEGRITY_REPORT.json',receipt);(O/'CROSS_SPLIT_IMAGE_DUPLICATION_S1000_REVIEW.zip.sha256').write_text(receipt['sha256']+'  '+Z.name+'\n',encoding='utf8')
print(json.dumps(receipt,indent=2))
