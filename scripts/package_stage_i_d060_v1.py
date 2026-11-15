"""Finalize offline tests, receipts and review packet. Never invokes the GPU runner."""
import hashlib,json,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
from stage_i_d060 import *

def main():
    require(read(PREP/'OFFLINE_CHECKS.json')['status']=='PASS','CPU preparation incomplete')
    require(not (PREP/'PREFLIGHT_MANIFEST.json').exists(),'Preserve final preparation')
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_stage_i_d060_offline.py','-v'],cwd=ROOT,capture_output=True,text=True)
    write(PREP/'OFFLINE_REGRESSION_LOG.txt',result.stdout+result.stderr)
    require(result.returncode==0,'Offline regressions failed; preserve log')
    import re
    count=int(re.search(r'Ran (\d+) tests',result.stderr).group(1))
    write(PREP/'OFFLINE_REGRESSION_RESULT.json',{'returncode':0,'tests':count,'status':'PASS','scope':'D060 contracts/negative controls/actual callbacks with doubles; no GPU/model execution in tests'})
    cpu=read(PREP/'CPU_ADAPTER_COMPATIBILITY.json');data=read(PREP/'DATA_ALIGNMENT.json')
    report=f'''# D-060 readiness — offline only

**READY FOR RESEARCHER REVIEW; GPU/TRAINING NOT AUTHORIZED.**

- Actual CPU PEFT load of trained RO:24 modules,{cpu['adapter_trainable_count']:,} trainable parameters,48 adapter tensors exactly equal to the saved RO checkpoint.
- RO/F names,dimensions,dtypes and finite tensor screens agree. All{cpu['backbone_parameter_count']:,} pretrained backbone parameters remain frozen; canonical fingerprints match historical D052. CPU adapter save roundtrip is exact.
- Saved adapter revision is null. Revision is established through historical runtime/approved configuration and pinned snapshot plus actual backbone tensor fingerprints, not inferred from null metadata.
- Fresh AdamW was instantiated without a step:0 inherited state entries. New linear schedule750/warmup38 is proposed; old optimizer/scheduler/scaler/RNG will not be loaded.
- All6600 train/validation condition rows were audited with{data['exact_source_and_target_encodes']:,} exact source/label re-encodes. Shared reviews/order/consensus match. No truncation, split overlap or held-out contents. Filtered zero-caption counts:9train/1validation.
- {count} offline tests PASS, including negative scope/hash/deadline controls and actual callback JSON/checkpoint/AMP/headroom guards with synthetic doubles. No GPU gradients or numerical training result is claimed.
- Proposed run:one RO-initialized F continuation,2000/200,max3epochs/750updates,all3 checkpoints;48 outputs at epochs1/3,original beam4/max_length768.
- Fixed random20/targeted4,existing references and saved historicalF/D058/D059 benchmarks only. HistoricalRO repetition context:5/20 exact-sentence repeats versus historicalF15/20, but18/20 RO adjacent-word repetition; RO is not a matched F control.
- D051 strict replay FAIL unchanged. D059 remains historically intact; the new researcher hypothesis is recorded without fabricating D059 output-quality annotations or formal GO/NO-GO.

## Remaining risks and required approval

CPU loading proves compatibility, not GPU trainability/AMP/gradient/memory behavior for this continuation. First approved training updates must pass inherited finite-gradient/AMP/mask/integrity checks and15% headroom; failure stops with no retry/fallback. Rare zero-caption learning and RO suppression of visual evidence remain risks.

RO has already seen5epochs; this run adds up to3Fepochs. Initialization/exposure/schedule/architecture and repeatedly reviewed validation cases confound comparison. This is a practical final viability attempt, not a causal staging claim.

Approve exact config/preflight/runner hashes in AUTHORIZATION_BINDINGS.json only after accepting750-step/38-warmup adaptation,epoch1/3 selection,72-hour viability deadline and common Stage2 RO/U/F fairness plan. No clock starts during this preparation. Proposed deadline is the earlier of authorization+72hours and original D059 deadline2026-10-25T15:10:48.941817+00:00.

The decision requires researcher-confirmed substantive Core/opinion/grounding/visual/coherence gains, not just loss,headers or repetition. Inconclusive/poor or overdue review ends optimization. Any matchedRO/U needs separate authorization. Reserve thesis effort for test/Shuffled/30-business human evaluation and writing; none is executed here.

See D060_PLAN_AND_AUTHORIZATION.md for exact budget,scientific criteria,confounds and day-by-day stop conditions. Commands below are **offline checks only**:

    .\\.venv-caption-pilot\\Scripts\\python.exe -m unittest discover -s tests -p test_stage_i_d060_offline.py -v
    .\\.venv-caption-pilot\\Scripts\\python.exe -c "import sys;sys.path.insert(0,'scripts');from stage_i_d060 import preflight;preflight();print('OFFLINE PREFLIGHT PASS')"

The prepared train/evaluate runner has not been invoked. No new predictions,model forward/backward,optimizer updates,Teacher calls or GPU operations occurred.
'''
    write(PREP/'D060_READINESS_REPORT.md',report)
    files=dict(read(PREP/'PROTECTED_INPUTS.json')['files'])
    # Preserve saved benchmark outputs and actual D058/D059 checkpoint receipts/tensors.
    for rel in ['outputs/stage_i/d059_result_report_v2/COMPARISON_RECORDS.json',
                'outputs/stage_i/d059_result_report_v2/SOFTWARE_METRICS_BY_SUBSET.json',
                'outputs/stage_i/d059_expanded_filtered_v2/train/COMPLETION.json',
                'outputs/stage_i/d059_expanded_filtered_v2/train/FROZEN_BACKBONE_INTEGRITY.json',
                'outputs/stage_i/d059_expanded_filtered_v2/evaluate/COMPLETION.json']:
        require((ROOT/rel).is_file(),'Missing completed benchmark '+rel);files[rel]=sha(ROOT/rel)
    for base in ['outputs/stage_i/d058_full_filtered_B_v1/train','outputs/stage_i/d059_expanded_filtered_v2/train']:
        for p in (ROOT/base).glob('checkpoint-*/checkpoint_integrity.json'):
            files[p.relative_to(ROOT).as_posix()]=sha(p)
            for name,h in read(p)['files'].items():
                require(sha(p.parent/name)==h,'Historical checkpoint mismatch')
                files[(p.parent/name).relative_to(ROOT).as_posix()]=h
    for rel in ['scripts/stage_i_d060.py','scripts/run_stage_i_d060_v1.py','scripts/prepare_stage_i_d060_v1.py','scripts/package_stage_i_d060_v1.py','tests/test_stage_i_d060_offline.py','configs/stage_i_d060_ro_to_filtered_v1.json','specs/009n_d060_text_first_continuation.md']:
        files[rel]=sha(ROOT/rel)
    for p in PREP.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    write(PREP/'PREFLIGHT_MANIFEST.json',{'decision':'D-060','files':files,'GPU_execution_approved':False,'CPU_model_load_without_forward':True,'D051_strict_replay':'FAIL_UNCHANGED'})
    preflight()
    bindings={'decision':'D-060','config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),'runner_sha256':sha(RUNNER),'GPU_execution_approved':False,'bound_files_verified':len(files)}
    write(PREP/'AUTHORIZATION_BINDINGS.json',bindings)
    write(PREP/'GPU_AUTHORIZATION_TEMPLATE.json',{**bindings,'approval_source':None,'researcher_explicit_authorization':None,'training_authorized':False,'max_optimizer_steps':750,'generation_outputs_max':48,'RO_U_training_authorized':False,'held_out_access':False,'Teacher_calls_authorized':False,'execution_authorized_at_utc':None,'viability_deadline_utc':None,'original_D059_deadline_utc':'2026-10-25T15:10:48.941817+00:00','automatic_retry':False})
    # Living status logs are intentionally outside frozen scientific input hashes.
    entry='''## D-060 — Final text-first continuation prepared; GPU NOT authorized

Researcher selected one final RO-initialized Filtered continuation hypothesis.
Offline CPU loading proves exact historicalRO adapter equality,24decoder Q/V
modules/294912trainable parameters,unchanged frozen backbone fingerprints and
exact adapter serialization. All6600train/validation condition rows and13200
source/label encodes align; shared reviews/order/consensus unchanged,zeroF9train/1val.
Fresh optimizer state is empty; old optimizer/scheduler/scaler/RNG are not resumed.
Proposed Stage2:F2000/200,max3epochs/750updates,new linear750/warmup38,checkpoints
all3epochs,48beam4 validation outputs at epochs1/3. Original datasets/targets/
captions/references/checkpoints remain frozen. D051 strict replay FAIL unchanged.

Plan/readiness/bindings:outputs/stage_i/d060_preparation_v1/. No new model forward,
backward,generation,optimizer update,Teacher call,GPU execution or held-out access.
Separate explicit hash-bound execution approval required.72-hour decision window
starts at that future approval,capped by unchanged D059 deadline25October2026
15:10:48.941817UTC. Inconclusive/poor/overdue quality ends optimization and uses
original matchedRO/U/F. GO requires researcher-confirmed substantive usable quality;
any laterRO/U must share RO initialization and identical Stage2 protocol/common
checkpoint under separate approval. No automatic further experiments.
New user direction authorizes preparation only; it does not retrospectively label
D059 GO/NO-GO or create human quality annotations. Preserve earlier entries as history.

'''
    for rel in ['DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md']:
        p=ROOT/rel;before=p.read_text(encoding='utf-8');snapshot=PREP/'documentation_before'/rel
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_text(before,encoding='utf-8')
        first,sep,rest=before.partition('\n');p.write_text(first+'\n\n'+entry+rest.lstrip('\n'),encoding='utf-8')
    bundle=ROOT/'outputs/stage_i/stage_i_d060_preparation_review_bundle_v1.zip'
    require(not bundle.exists(),'Bundle collision')
    entries={p.relative_to(PREP).as_posix():p for p in PREP.rglob('*') if p.is_file()}
    for rel in ['scripts/stage_i_d060.py','scripts/run_stage_i_d060_v1.py','scripts/prepare_stage_i_d060_v1.py','scripts/package_stage_i_d060_v1.py','scripts/stage_i_d058.py','scripts/stage_i_paired_pilot.py','scripts/stage_hi_led_data.py','tests/test_stage_i_d060_offline.py','configs/stage_i_d060_ro_to_filtered_v1.json','specs/009n_d060_text_first_continuation.md',
                'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl','outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json','outputs/stage_i/d059_result_report_v2/COMPARISON_RECORDS.json','outputs/stage_i/d059_result_report_v2/SOFTWARE_METRICS_BY_SUBSET.json']:
        entries['workspace/'+rel]=ROOT/rel
    hashes={rel:sha(p) for rel,p in entries.items()}
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for rel,p in entries.items():z.write(p,rel)
        z.writestr('SHA256_MANIFEST.json',json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(bundle) as z:
        for rel,h in hashes.items():require(hashlib.sha256(z.read(rel)).hexdigest()==h,'ZIP binding failed')
    digest=sha(bundle);bundle.with_suffix('.zip.sha256').write_text(digest+'  '+bundle.name+'\n',encoding='utf-8')
    write(PREP/'DELIVERY_RECEIPT.json',{'zip':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest,'archive_entries_verified':len(hashes),'tests_passed':count,'bound_files':len(files),'GPU_operations':0,'status':'OFFLINE_READY_AWAITING_APPROVAL'})
    print(json.dumps(read(PREP/'DELIVERY_RECEIPT.json'),indent=2))

if __name__=='__main__':main()
