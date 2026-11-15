# Stage D1 — Photo integrity and duplicate indexing

2026-10-06. User authorizes the next step after003l: inspect readable/corrupt and duplicate photos before the real caption pilot. No caption/model/API/GPU/rental job, K selection or final research-setting freeze in D1.

## Scope and invariants

All30,027 photo IDs from the fixed2300 cohort (2000train/200validation/100test). Deterministic file integrity/unsupervised hashing applies uniformly to test; no thresholds or policies tuned on test, no test galleries for research decisions. Preserve raw/v1/English data, photo/review/business membership/splits and stored bytes. Existing local-photo eligibility counts are pre-selection conditions; flag businesses below5 readable/unique photos without automatic exclusion/replacement.

Verify manifest path/size/SHA-256 against the accepted Stage B inventory; compute every current file digest again. Missing/changed bytes are an integrity blocker, not an silently excluded input. A byte-matching corrupt image is recorded as unreadable. Verify container and separately force full decode with truncated-image loading disabled; record dimensions/mode/format/EXIF orientation and warnings. Decompression-bomb protection remains enabled. No destructive image rewrite or physical deletion.

Create a readable-photo view for shared image/file validation. Keep original IDs available for the unfiltered condition. Exact/near duplicate indexing belongs to the filtered pipeline; do not deduplicate the unfiltered baseline. Representative IDs in duplicate groups are deterministic diagnostic references, not an adopted caption-selection policy.

## Fixed diagnostic hashes

- Raw file SHA-256: byte identity.
- Normalized RGB pixel SHA-256: EXIF transpose applied in memory, convertRGB, hash explicit width/height and pixel bytes. Normalization is for hashing/display only; final caption preprocessing must be declared in E.
- Diagnostic pHash64 `cosine32_round6_v1`: normalized RGB→Pillow grayscale→32x32 LANCZOS, unnormalized DCT-II top-left8x8 using explicit cosine basis, coefficients rounded to6 decimal places; threshold strictly above median of all64 coefficients, including DC; row-major bits to16-character hex. Declared implementation, not a claimed third-party ImageHash equivalence. Uniform/low-detail colors can collide; matching pHash is not proof of duplication or semantic identity.
- Independently check the DCT reference using direct coefficient summation on256 precommitted train/validation photo IDs. Meaningful synthetic checks cover truncated/corrupt decoding, metadata-only file changes versus pixel identity, orientation and deliberate solid-color pHash collisions.

Raw/pixel exact groups indexed globally and within businesses; flag cross-business/cross-split groups as possible repeated visual evidence, never automatically reassign splits/delete records. Globally index equal-pHash collisions separately, not as confirmed leakage. Near pairs: exhaustive ALL within-business readable pairs only, exploratory Hamming thresholds0/4/8/12. No global approximate-distance sweep/semantic duplicate claim. Save pairs up to12 once; diagnostics reuse them for each threshold.

Report connected components, not all-pairs-equivalent clusters: chaining can yield endpoints farther than the threshold. Record component diameter/chained flags, edge count, deterministic representative and per-business group memberships/count reductions. No automatic near-duplicate exclusion/threshold adoption; filtered-caption policy remains TBD after train/validation visual review.

## Verification/output

Manifest provenance/IDs/splits/raw hashes and readable/normalized metadata verified independently from all original files. Recompute all within-business distances/components/counts/groups/global exact/collision summaries, compare reports. Verify256 fixed pHash reference outputs; no accepted failure. Show counts bysplit/business/photo label, invalid files, below5 readable/exact/near diagnostic views, extreme dimensions, global cross-split groups. Manually inspect a small purposive train/validation pair gallery and document what is observed, including ambiguity/false positives; not formal human evaluation or a chosen threshold.

Keep output compact (<30MB scientific metadata/reports, no new photo copies); only small rendered previews. Snapshot current18-cell notebook/docs/seals, integrate results/gallery into notebook00, execute cached/read-only, inspect new plots and verify previous9 figures unchanged. Update root/SDD progress, README/D-021/master plan. Stop after image audit; caption pilot needs its own spec/declared generation/environment feasibility, and source/metadata/caption allocations stay pending. Core3 models/4 evaluation conditions, external generalization/representative30-business human evaluation unchanged.

## Measured acceptance - 2026-10-06

All30027 readable,0unreadable; source-countmin5/max231; source and original raw hashes equal.37.058s production/43.300s independent validator:30k full fresh pixel/phash/decode replay,256 direct-coefficient references,432699 within-business distances,580max12pairs,2041local+2584global groups/2300 counts PASS. Four tests PASS0.119s after fixing pre-production PNG.getexif/verify order; no failed data run accepted. JPEG29875/PNG152, allRGB/orientation1.165 exactwithin groups;5businesses have4 distinct normalized images after diagnostic collapse. Nearedges432/481/504/580, chainedcomponents0/0/2/7 andbelow5 flags13/17/17/19; nopolicy applied.

New confirmed exact visual-overlap finding:821 raw andpixel groups,664cross-business/227cross-split (119trainval,77traintest,31allthree). No actual caption/evaluation/training result is claimed. Non-applied protect-heldout exclusion scenario removes785 (751train/34val), affects172businesses, leaves124below5/14train zero images. Proposal independently verified by group split priority. No pools/business/splits changed. This is a research decision blocker before finalcaption/training/evaluation protocol, not authorization to exclude/resample. Seven train/val original pairs inspected with notes;no test-policy gallery. Notebook00 section9.7,19/19 cells/twelve figures PASS20.931s,three newfigures inspected/previous9 unchanged. Outputs~24.3MB/no imagecopies; prior18-cell docs/notebook/source audit acceptance snapshot retained. Readable/file/hash acceptance complete; final duplicate/overlap/zero-image handling,caption feasibility/source allocations pending.

Additional structural diagnostic verified:83testbusinesses have no exactpixelreuse withtrain/val,17any/8allreuse,123of1136testphotoIDs overlap. Secondary83 evaluation preserving currentpools/main100 is a non-adopted option, not a claim of complete semantic/pretraining image independence or an executed evaluation. Updated mainnotebook19/19cachedcells/12figures, UIstatus/footer and gallery/report reflect both quantified options. No photo/evaluation policy changed.
