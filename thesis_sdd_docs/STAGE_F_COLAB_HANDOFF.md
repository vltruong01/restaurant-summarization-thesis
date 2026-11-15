# Colab execution preference —2026-10-10

User changes the execution preference from paidVastRTX3090 to freeColabGPU, likely NVIDIA T4 (nominal16GB, typicallyabout15GBavailable). This is infrastructure only. K5,5000/4000/750template, original D026COCOrevision/beam3/batch1/prompts/Basic/retry, allscorers and D030/D031threshold/ranking logic remain unchanged. Localjob stays stopped;2709existingcaptions retained,27318missing.

Notebook: `notebooks/09_stage_f_colab_smoke_resume.ipynb`, an executionharness; Notebook00 remains primarydataset/filtering/history entrypoint. Newadapter/output namespace: `scripts/stage_f_colab_d033.py`, `outputs/stage_f/production_caption_construction_d033_colab_v1`. T4engineering preflight requiresCUDA>=14GiB and the originalminimumRAM/diskreserves; it does not alter any scientificparameter. ActualColabGPU feasibility has NOT been tested locally.

## Upload and run

Upload these into GoogleDrivefolder `thesis_stage_f`:

1. Existing `outputs/cloud_handoff/stage_f_d033_v1/stage_f_d033_vast_bundle.zip` (2.91GB). This remains the reusable, independentlyverified source package.
2. `outputs/cloud_handoff/stage_f_colab_d033_v1/stage_f_colab_addon.zip` (small addition, no original basepackage members overwritten).

Open notebook09onColab and choose GPUruntime. RunAllcopies/extracts archives to `/content`, verifies bytes, prepares an isolatedPython3.11/unchangedpinnedscientificstack usinguv if needed, restores checkpointbank, downloads the exactBLIPweights and runs100pendingTRAIN-only smoke. GPUtype/availability and sessionduration are not guaranteed. Inspect/send benchmark_report.json before setting flags for remainingcaptionchunks/finalprocessing. NoSSH/remoteCodexshell/keepaliveworkarounds or distributedworkers are part of the freeColabworkflow.

## Persistence

Originalphotos/code/captionrecords come from the verified archives. GeneratedColaboutputs are incrementally compressed and copied to `thesis_stage_f/colab_checkpoints/` onDrive every120seconds during foregroundcell execution and when the cellreturns/interruption is handled. Each archive is byteverified before being finalized. Storeonlycheckpointoutputs, not15.5GBBLIPweights, inDrive; freeDrivecapacity is separate fromruntimeVRAM.

VMtermination can lose the unsaved interval; verifiedbankZIPs survive. Nextsessionre-extracts the same source package, reinstalls the pinnedvenv and restores the bank. DifferentGPU/runtime/device-map identity causes a stop; do not bypass guards, silentlychangegeneration/quantization/budgets or overwritecaptionrecords. A separate runtime/provenance namespace can be prepared when justified by actualoutput, preserving completedrecordprovenance.

Operationalchunks default500newphotos/invocation, distinct from the modelbatch1setting; noK-based photopreselection. After27318remainingphotos are completed, user-enabledfinalcell runs originalscoring/review/filter/source validation and stops after StageFmanifests, before GPTteacher/LEDtraining. No testtuning or human testreferences.

Offline validation:3Colabadapter/checkpoint guardsPASS; notebookcellscompile; addonfileSHAchecksPASS, basepackage/scientificcore unchanged. These are portability checks, not actualT4runtime or trainingfeasibility results.

Sources: [ColabFAQ](https://research.google.com/colaboratory/faq.html), [NVIDIAT4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [uvPython](https://docs.astral.sh/uv/guides/install-python/).


Observeduser-output update2026-10-10: notebooksmokePASS100TRAINIDs, sourcecellsexact,100Basicpass/1retry,0.80614s/photo,7.43994GiBpeakGPUallocated.27218remainingcaption-only estimate6.095h. This is reviewedusernotebookevidence, not localimport of actualcheckpointbytes/fullruntime. ActualGPUname not shown; fullproduction remainspending. See outputs/stage_f/colab_smoke_review_20261010/review_report.json.


Productioncontinuationv2afteruserapproval: useupdatedNotebook09andupload stage_f_colab_runtime_patch_v2.zip (2.6KB) to thesameDrivefolder. Base/addonZIPs remainunchanged. Notebookrestores100smokecaptions, continuesallremainingpool inone modelload, thenrunsfinalprocessing onlyif27318cloudplanphotoscomplete. Snapshot-safecheckpointIOfix preservescurrentbankformat/captions;4guardsPASS. StopbeforeStageG/LEDtraining. LocalagenthasnotexecutedthisonColab; remoteactualprogress stillawaitsoutput.


FINALcompletion2026-10-10: completeColaboutputs imported locally,all30027captions/scoringbindings and2300manifests verified;6900native/slowLEDsourceIDs identical,zerooverflows.12zeroFilteredbusinesses retained; no method orlabelchanges. Earlierpartialexport/smoke-only/pendingnotes are provenance. See STAGE_F_COMPLETION_20261010.md and outputs/stage_f/production_manifest_pointer.json. StopbeforeStageG.
