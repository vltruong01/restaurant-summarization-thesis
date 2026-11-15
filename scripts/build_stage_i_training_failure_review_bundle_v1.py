"""Package an offline audit/proposal and existing evidence. No model or API execution."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import zipfile

from stage_i_decoding_diagnostics import CONDITIONS, read_jsonl, write_json

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'outputs/stage_i/training_failure_audit_v1'
CLOUD=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
PREP=ROOT/'outputs/stage_i/decoding_diagnostic_v1'
NAME='stage_i_training_failure_audit_chatgpt_bundle_v1'
DEST=ROOT/'outputs/stage_i'/NAME
ZIP=DEST.with_suffix('.zip')


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert not DEST.exists() and not ZIP.exists(),'Preserve old exports'
    mapping={}
    def include(relative,source):
        assert source.is_file() and relative not in mapping
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        mapping[relative]=source
    for entry in read(AUDIT/'AUDIT_MANIFEST.json')['files']:
        p=AUDIT/entry['path'];assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
        include('audit/'+entry['path'],p)
    include('audit/AUDIT_MANIFEST.json',AUDIT/'AUDIT_MANIFEST.json')
    include('audit/original_execution.log',ROOT/'outputs/stage_i/training_failure_audit_v1.log')
    for n in ['cases.jsonl','subset_manifest.json','decoding_configurations.json']:
        include('frozen_diagnostic_inputs/'+n,PREP/n)
    stageh=read(CLOUD/'outputs/stage_h/led_dataset_v1/manifest.json')
    include('frozen_configs/stage_h_manifest.json',CLOUD/'outputs/stage_h/led_dataset_v1/manifest.json')
    include('frozen_configs/approved_original_training_config.json',CLOUD/'configs/stage_i_lora_led_training_approved_v2.json')
    mm=read(CLOUD/'assets/models/led-base-16384.config_manifest.json')
    include('frozen_configs/led_model_config.json',CLOUD/mm['config_file'])
    include('frozen_configs/model_config_manifest.json',CLOUD/'assets/models/led-base-16384.config_manifest.json')
    tm=read(CLOUD/'assets/tokenizers/led-base-16384.manifest.json')
    include('assets/tokenizers/led-base-16384.manifest.json',CLOUD/'assets/tokenizers/led-base-16384.manifest.json')
    for name in tm['sha256']:
        include('assets/tokenizers/led-base-16384/'+name,CLOUD/tm['relative_directory']/name)
    include('frozen_configs/source_template.json',ROOT/'configs/led_source_template_d032.json')
    include('provenance/imported_archive_manifest.json',CLOUD/'BACKUP_MANIFEST.json')
    for c in CONDITIONS:
        info=stageh['datasets']['validation_'+c];p=ROOT/info['stage_g_jsonl'];assert sha(p)==info['stage_g_jsonl_sha256']
        include('validation_full/validation_'+c+'.jsonl',p)
        original=CLOUD/'outputs/stage_i/runs_v3'/c
        include(f'saved_predictions/Baseline/{c}.jsonl',original/'validation_predictions.jsonl')
        for name in ['trainer_state.json','completion_report.json','runtime_initial.json','run_binding.json']:
            include(f'training_evidence/{c}/'+name,original/name)
        include(f'training_evidence/{c}/best_adapter_config.json',original/'best_adapter/adapter_config.json')
        for experiment,folder in [('A','decoding_greedy_A_local_v2'),('B','decoding_beam_ngram_B_local_v1')]:
            run=ROOT/'outputs/stage_i'/folder
            include(f'saved_predictions/{experiment}/{c}.jsonl',run/c/'predictions.jsonl')
            for path in (run/'attempts').glob(f'*/{c}_generation_config.json'):
                include(f'actual_generation_configs/{experiment}/{c}.json',path)
    bdir=ROOT/'outputs/stage_i/decoding_beam_ngram_B_local_v1_evaluation_v1'
    adir=ROOT/'outputs/stage_i/decoding_greedy_A_local_v2_evaluation_v1'
    for name in ['baseline_A_B_cases.json','baseline_A_B_comparison.csv','ngram_lexical_summary.json','HEADER_CONSTRAINT_AUDIT.json',
                 'AI_QUALITATIVE_REVIEW.json','COMPARISON.html','execution_integrity_and_runtime.json']:
        include('existing_B_evidence/'+name,bdir/name)
    include('existing_A_evidence/AI_QUALITATIVE_REVIEW.json',adir/'AI_QUALITATIVE_REVIEW.json')
    for relative in ['scripts/run_stage_i_lora_led_cloud_v2.py','scripts/run_stage_i_lora_led.py','scripts/stage_hi_led_data.py']:
        include('actual_training_code/'+Path(relative).name,CLOUD/relative)
    for name in ['audit_stage_i_training_failures_v1.py','report_stage_i_training_failure_audit_v1.py',
                 'build_stage_i_training_failure_review_bundle_v1.py','stage_i_decoding_diagnostics.py']:
        include('audit_code/'+name,ROOT/'scripts'/name)
    for relative in ['thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md',
                     'DECISIONS.md','specs/009c_post_decoding_training_audit.md']:
        include('documentation/'+Path(relative).name,ROOT/relative)
    # Scan before copying so an encoding/log issue does not leave a partial export.
    for source in mapping.values():
        raw=source.read_bytes();text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
        assert not re.search(r'\bsk-[A-Za-z0-9_-]{20,}',text),'Potential credential: '+source.name
    DEST.mkdir()
    copied=[]
    for rel,src in mapping.items():
        target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);digest=sha(src)
        shutil.copyfile(src,target);assert sha(target)==digest
        copied.append({'path':rel,'bytes':target.stat().st_size,'sha256':digest})
    # Full target-only table permits independent reproduction of the6,600 target text metrics.
    targets=[];frame=[]
    for key,info in stageh['datasets'].items():
        source=ROOT/info['stage_g_jsonl'];assert sha(source)==info['stage_g_jsonl_sha256']
        for r in read_jsonl(source):
            targets.append({k:r[k] for k in ['business_id','split','condition','target','target_tokens','source_sha256']})
            if key=='train_filtered':
                frame.append({'business_id':r['business_id'],'source_tokens':r['source_tokens'],
                              'caption_count':len(r['selected_photo_ids']),'source_sha256':r['source_sha256']})
    assert len(targets)==6600 and len(frame)==2000
    extracted={(x['split'],x['condition'],x['business_id']):hashlib.sha256(x['target'].encode()).hexdigest() for x in targets}
    for r in read_jsonl(AUDIT/'input_target_bindings.jsonl'):
        assert extracted[r['split'],r['condition'],r['business_id']]==r['target_sha256']
    with (DEST/'teacher_targets_train_validation.jsonl').open('x',encoding='utf-8') as f:
        for r in targets:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (DEST/'train_pilot_sampling_frame.jsonl').open('x',encoding='utf-8') as f:
        for r in frame:f.write(json.dumps(r)+'\n')
    cases=read_jsonl(PREP/'cases.jsonl');subset=read(PREP/'subset_manifest.json')
    population=sorted(r['business_id'] for r in read_jsonl(DEST/'validation_full/validation_filtered.jsonl'))
    assert random.Random(subset['seed']).sample(population,20)==subset['random_business_ids']
    index={'all_paths_relative_to_bundle_root':True,'businesses':[],'all_teacher_target_rows':6600,'all_validation_source_rows':600,
           'proposed_pilot_manifest':'audit/proposed_pilot_manifest.json','pilot_train_frame':'train_pilot_sampling_frame.jsonl'}
    for bid in subset['execution_business_ids']:
        record={'business_id':bid,'split':'validation','subset':'random' if bid in subset['random_business_ids'] else 'targeted','conditions':{}}
        for c in CONDITIONS:
            full=read_jsonl(DEST/f'validation_full/validation_{c}.jsonl')
            record['conditions'][c]={'source_teacher_evidence_file':f'validation_full/validation_{c}.jsonl',
                'source_teacher_evidence_row_zero_based':next(i for i,r in enumerate(full) if r['business_id']==bid),
                'fixed_case_file':'frozen_diagnostic_inputs/cases.jsonl',
                'fixed_case_row_zero_based':next(i for i,r in enumerate(cases) if r['business_id']==bid and r['condition']==c),
                'predictions':{}}
            for setting in ['Baseline','A','B']:
                path=f'saved_predictions/{setting}/{c}.jsonl';rows=read_jsonl(DEST/path)
                pos=next(i for i,r in enumerate(rows) if r['business_id']==bid)
                record['conditions'][c]['predictions'][setting]={'file':path,'row_zero_based':pos}
        index['businesses'].append(record)
    write_json(DEST/'INDEX.json',index)
    (DEST/'README_FIRST.md').write_text('''# Stage I post-decoding failure audit and training proposal

**Offline review only. No new inference/training was run. No pilot is approved.**

Read:
1. `CHATGPT_REVIEW_REQUEST.md`
2. `audit/STAGE_I_TRAINING_FAILURE_AUDIT.md`
3. `audit/STAGE_I_TRAINING_IMPROVEMENT_PROPOSAL.md`
4. `audit/architecture_audit.json`, `audit/optimization_audit.json`, `audit/dataset_target_audit.json`
5. `audit/proposed_training_config.json` and the fixed proposed pilot manifest.

Verified current scope:24 decoder Q/V projections,48 tensors,294,912 parameters; encoder/base weights frozen. Adam state/LR/checkpoints support1,250 updates/five epochs and successful best-checkpoint selection. No critical implementation bug explaining the quality failures was established. All6,600 target texts have0 exact repeated sentences. Expanded encoder local/global Q/V plus current decoder Q/V meta-verifies48 projections/96 tensors/589,824 parameters, but no new-scope forward/backward or GPU feasibility has run.

The proposed screen uses two fresh F-only arms on the same512 train businesses, two epochs/128 updates each, same200 validation loss inputs and same20 random+4 targeted generation inputs. Never train on validation. Existing full F checkpoint is the primary practical reference but not compute-matched. A new short B versus old full A comparison alone would confound scope with exposure. Shared decoder initialization, fresh scope-bound optimizers, shorter7-step warmup versus old63, untested activation memory and proposed quality criteria are explicit; nothing is automatically adopted.

## Independent verification evidence

- Actual successful training runner, original runner diff, data helper, frozen configs, argument/optimizer/scaler/scheduler summaries and all module names are included. Exact installed library function excerpts support implementation reasoning. Original archive275-file checksums are retained; model/adapter/optimizer binary weights are omitted, so this is not a standalone training/runtime archive.
- `teacher_targets_train_validation.jsonl` contains all6,600 exact target strings/IDs for repetition/template checks; `audit/input_target_bindings.jsonl` supplies per-row text hashes. Extraction was verified without modifying any target.
- `validation_full/` contains all600 existing RO/U/F validation source/target/evidence records. Baseline includes all200 predictions/condition; A/B each use the same24 businesses/72 predictions. `INDEX.json` joins the fixed sample by relative files and zero-based row numbers. Open `existing_B_evidence/COMPARISON.html` for the saved four-column viewer, or read its JSON.
- The20 random and4 targeted groups remain separate. Existing22 B/18 A qualitative notes are AI source/caption assessments, not human ground truth or prevalence estimates. Original photos are absent; photo IDs and teacher verification records are provenance, not new independent image verification.
- `train_pilot_sampling_frame.jsonl` allows the proposed512-ID sample to be reproduced from caption counts/source lengths without shipping all train review sources. The manifest was chosen from input properties, not favorable model results. Accepted train/validation/test splits remain unchanged; no held-out content is included.

Historical configs still carry their old training authorization flags; they are archived evidence, not authorization to run this new proposal. Current proposed execution switches are false. The optional Triton warning in the original PowerShell audit log is documented in `audit/OFFLINE_EXECUTION_NOTE.json`; the complete audit used meta/CPU inspection and did not require CUDA.

No decoder tuning, Teacher calls, dataset/target/threshold/K/budget changes, checkpoint edits, held-out access or training/inference took place in this audit/export. The ZIP has a SHA256 sidecar and complete payload manifest. Review hypotheses critically; more encoder parameters are not assumed to solve informativeness or grounding.
''',encoding='utf-8')
    (DEST/'CHATGPT_REVIEW_REQUEST.md').write_text('''# Independently review the offline audit and proposed next experiment

Baseline, greedyA and no-repeat4B have completed. B suppresses literal strings but leaves poor information/grounding and some header artifacts. We authorized an offline implementation/failure audit and a training proposal, **not new model execution**.

Please:
1. Verify current LoRA module names and counts; explain why q_proj/v_proj selects decoder only, and distinguish a deliberate frozen-encoder design from an implementation bug. Check optimizer steps, accumulation/scaling, LR/warmup, checkpoint selection, source/label masks and special tokens from actual archived code/states. Do not infer skipped training from low VRAM.
2. Check whether the 'no critical implementation error found' conclusion is justified and identify any overlooked critical defect. Separate code/state facts from hypotheses requiring a numerical replay, which was not run.
3. Inspect saved sources/teacher/Baseline/A/B examples for repetition, generic omissions, lost positive/negative contrasts, review-to-visual leakage, wrong cross-caption object relationships and zero-caption behavior. Teacher targets and AI notes are not independent human truth. Simple supported captions are acceptable; use actual REV/IMG_CAP/PHOTO IDs and quote exact text. Keep random/targeted denominators separate.
4. Assess the hypothesis matrix. The encoder+decoder proposal doubles capacity and changes location together; it cannot isolate location alone. No guarantee it solves grounding. Teacher targets have no exact sentence repetition but conventional style and some adjacent repeats; determine whether Codex overstates imitation/underlearning evidence.
5. Review the minimal paired F-only512-train/two-epoch/128-update design, fixed validation inputs, shared decoder initialization, no cross-scope resume, exact48-module/589,824 count, new-scope memory/gradient feasibility, shorter warmup and full-checkpoint reference limits. Training must remain train-only, evaluation/selection validation-only. Existing full A is not compute-matched to a short B candidate; decide whether the proposed fresh small A-control is necessary/fair.
6. Review the proposed technical/repetition/information/grounding/zero-caption go/no-go criteria. They require source-verified content units and researcher-confirmed judgments rather than ROUGE substitution. Are they sufficiently clear and feasible? Suggest revisions before any execution; do not silently approve new thresholds/parameters.

Return verified findings, any corrections, strongest/weakest causal hypotheses and a recommendation to approve/revise/decline the proposed feasibility/pilot. Do not run training, generate predictions, rewrite teacher/student labels, change datasets, continue decoder experiments or use held-out data. No full RO/U/F retraining or new pilot is automatically authorized by this discussion.
''',encoding='utf-8')
    write_json(DEST/'EXPORT_REPORT.json',{'status':'EXPORT_ONLY_PASS','copied_files':copied,'teacher_target_rows':6600,
        'all_validation_source_rows':600,'fixed_diagnostic_businesses':24,'random':20,'targeted':4,
        'targets_and_predictions_unchanged':True,'new_inference_training_teacher_calls':0,'held_out_content':False,
        'original_weights_and_photos_included':False,'proposed_pilot_executed':False})
    payload=[{'path':p.relative_to(DEST).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
    write_json(DEST/'MANIFEST.json',{'files':payload,'manifest_excludes_itself':True})
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,NAME+'/'+p.relative_to(DEST).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for e in payload:
            raw=z.read(NAME+'/'+e['path']);assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
    for relative,source in mapping.items():assert sha(source)==sha(DEST/relative)
    digest=sha(ZIP);ZIP.with_suffix('.zip.sha256').write_text(f'{digest}  {ZIP.name}\n',encoding='ascii')
    print(json.dumps({'status':'PASS','zip':str(ZIP),'bytes':ZIP.stat().st_size,'MiB':ZIP.stat().st_size/1024**2,
                      'sha256':digest,'files':len(payload)+1},indent=2))


if __name__=='__main__':main()
