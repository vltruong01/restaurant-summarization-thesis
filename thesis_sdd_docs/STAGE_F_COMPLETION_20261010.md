# Stage F completion ?2026-10-10

**Production caption construction is complete and independently validated. Stop before Stage G / GPTteacher / LEDtraining.**

Frozen D033K5; D0325000/4000/750source contract/template. CLIPsoft-only, pHashdiagnostic-only, exactpixelcollapseFilteredonly, relevance0.08/red0.59/group0.46and D026COCO/beam3/batch1/QC/retry unchanged. Actual ColabGPU:TeslaT4. No held-out test tuning, human testreference use or testteacher targets.

| Split | Businesses | Generated | Basicpass | Basicfail | Unfiltered retained | Filtered retained | ZeroFiltered businesses |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2000 | 26387 | 26331 | 56 | 9989 | 6880 | 9 |
| validation | 200 | 2504 | 2497 | 7 | 998 | 698 | 1 |
| test | 100 | 1136 | 1134 | 2 | 500 | 341 | 2 |

Total30027generated,29962Basicpass/65fail (raw/retry evidence retained),11487Unfiltered/7919Filtered selected captions. ZeroFiltered12businesses (9train/1validation/2test), preserved without fabricatedcaptions or exclusion; Unfilteredzero0. FinalK5is a maximum, not a forcedcount.

Filtered removalcounts after Basic:182exactduplicatecollapses,9070relevance removals,11295semanticredundancy removals;9415postredundancy candidates before up-to5selection. No hardCLIP/pHash removals.

## Actual budgets

| Split | Condition | Caption min / mean / p95 / max | Source min / mean / p95 / max | Caption>750 | Source>5000 |
|---|---|---|---|---:|---:|
| train | unfiltered | 98 / 129.036 / 136.050 / 152 | 1020 / 3847.059 / 4181.000 / 4211 | 0 | 0 |
| train | filtered | 10 / 91.410 / 133.000 / 141 | 971 / 3809.433 / 4171.000 / 4213 | 0 | 0 |
| validation | unfiltered | 109 / 128.720 / 135.000 / 144 | 1159 / 3818.165 / 4179.000 / 4195 | 0 | 0 |
| validation | filtered | 10 / 92.435 / 131.050 / 136 | 1114 / 3781.880 / 4169.050 / 4193 | 0 | 0 |
| test | unfiltered | 120 / 129.430 / 136.050 / 142 | 1267 / 3765.510 / 4178.100 / 4197 | 0 | 0 |
| test | filtered | 10 / 90.860 / 131.050 / 138 | 1218 / 3726.940 / 4164.150 / 4185 | 0 | 0 |

Maxcaptionblock152tokens; maxfullsource4213tokens. EmptyFilteredretains the auditedvisualheader (10incrementtokens). Whole reviews/captions remaincomplete, no silent truncation/removal tofit.

## Independent import/validation

NewDriveexportthesis_stage_f-20261010T062552Z-1-002.zip contains205checkpointbundles/27913latestfiles. AllnestedmemberSHA256and19finalmanifestSHA256checksPASS. Imported27318newColabrecords, combinedwith original2709compatiblelocal/validationrecords. Complete30027caption/model/QC/image/provenance bindings and30027scoringbindings checked. All2300cohortrows present; originalbusinessmetadataexact. Accepted2200train/validationreview IDs/text/order unchanged;100testreviewinputreferences bind tosame frozenfullv3/v4/v4.1/v4.2code/config/inputidentity. Validation160K5sets/sources match prior sensitivityevidence. Unfilteredseedorder and bothconditionfrozenrankingprefixes checked. All3025priorprotectedscientificfiles unchanged.

IndependentCPUaudit re-encoded6900complete Reviews-only/Unfiltered/Filtered sources with native andslow pinnedLEDtokenizers; exacttokenIDs matched, zero review/caption/sourceoverflows. Verification used inputs only, with no inference/recalibration/teacher/training.

Sourcepackageparts002are sufficient for actualColaboutputs because the existingoriginal sourcepackage is already local; earlierolder002export was partial20807, not a failureof the currentcompletedrun. It remains historicalreviewevidence, supersededbythiscompleteexport.

## Artifact entrypoints

- `outputs/stage_f/production_manifest_pointer.json` ? canonicalpointer to completednamespace/files.
- `outputs/stage_f/production_caption_construction_d033_colab_v1/completion_report.json` and `.md` ? cloudcompletioncounts/distributions/manifestsSHA.
- `outputs/stage_f/production_caption_construction_d033_colab_v1/manifests/{train,validation,test}/{unfiltered,filtered}_captions.jsonl` ? finalcaptionconditions.
- `.../manifests/{train,validation,test}/{reviews_only,unfiltered,filtered}_sources.jsonl` ? wholeLEDsources.
- `.../manifests/shared_review_inputs.jsonl`, `token_ledger.jsonl`, `complete_caption_pool_inventory.jsonl`, `frozen_filter_pools.jsonl` ? input/provenance/token/filteraudit.
- `outputs/stage_f/colab_result_review_20261010/complete_import_report.json` and `independent_validation_report.json` ? independentreceipts.

Shuffled actualassignment remains later and is not claimedconstructed/audited. Finalteacher/protocol/targetvalidation and LEDtraining are subsequentstages; no calls/runs authorized or performed as part of thisreview. Main human30-business evaluation and externalgeneralization remain separate.
