"""CPU-only audit of the researcher-downloaded interrupted Colab training snapshot."""
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DOWNLOADS=Path(r'C:\Users\VUONGLOCTRUONG\Downloads')
ZIP=DOWNLOADS/'LED-Large_multimodal-20261020T044732Z-1-001.zip'
NOTEBOOK=DOWNLOADS/'LED_LARGE_F_FULL_TRAIN_V5.ipynb'
OUT=ROOT/'outputs/stage_i/led_large_full_train_v5_interrupted_review_v1'
EXPECTED_ZIP='144c88ca3a06be86b4137dab22ca98e75becb8a2c4fc479d70ffad0f3a828b27'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    require(sha(ZIP)==EXPECTED_ZIP,'Downloaded ZIP changed')
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ZIP,OUT/ZIP.name);shutil.copy2(NOTEBOOK,OUT/NOTEBOOK.name)
    with zipfile.ZipFile(ZIP) as z:
        require(z.testzip() is None,'ZIP CRC failure')
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate paths')
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and ':' not in info.filename
                    and '\\' not in info.filename and (info.external_attr>>16)&0o170000 != 0o120000,'Unsafe ZIP entry')
            dest=OUT/'evidence'/info.filename;dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
    evidence=OUT/'evidence/LED-Large_multimodal'
    train=evidence/'runs/F_full_train_v5/train'
    gate=evidence/'runs/F_gradient_gate_v4/feasibility'
    binding=read(ROOT/'outputs/stage_i/led_large_full_train_v5_preparation/EXECUTION_BINDINGS.json')
    require(read(train/'BINDING.json')==binding,'Training binding differs')
    approval=read(evidence/'LED_LARGE_F_FULL_TRAIN_V5_APPROVAL.json')
    require(all(approval[k]==v for k,v in binding.items()) and approval['training_authorized']
            and approval['approval_source']=='researcher_notebook_cell','Approval binding/provenance differs')
    require(approval['execution_deadline_utc']=='2026-10-21T17:11:48+00:00','Deadline changed')
    require({k:v for k,v in read(train/'AUTHORIZATION.json').items() if not k.startswith('_')}==approval,
            'Actual runner authorization differs')
    original=read(ROOT/'outputs/cloud_handoff/LED_LARGE_F_FULL_TRAIN_V5.ipynb')
    received=read(NOTEBOOK)
    a=[c for c in original['cells'] if c['cell_type']=='code']
    b=[c for c in received['cells'] if c['cell_type']=='code']
    require(len(a)==len(b)==14,'Notebook scope differs')
    for i,(x,y) in enumerate(zip(a,b)):
        sx=''.join(x['source']);sy=''.join(y['source'])
        if i==9:
            require(sy.startswith('RESEARCHER_APPROVES_F_TRAINING=True'),'Actual notebook consent missing')
            sy=sy.replace('RESEARCHER_APPROVES_F_TRAINING=True','RESEARCHER_APPROVES_F_TRAINING=False',1)
        require(ast.dump(ast.parse(sx))==ast.dump(ast.parse(sy)),'Notebook code changed beyond explicit consent')
    checked=[]
    for cp,proof_name in ((train/'checkpoint-250','checkpoint_integrity.json'),(gate/'checkpoint_step3','integrity.json')):
        proof=read(cp/proof_name)
        for name,h in proof['files'].items():
            require(Path(name).name==name and sha(cp/name)==h,'Checkpoint component changed/missing: '+str(cp/name))
        checked.append(dict(checkpoint=str(cp.relative_to(OUT)),files=len(proof['files']),status='PASS'))
    require(sha(gate/'REPORT.json')=='3bfbf2248b676c3fa568af77f205905370390b946d33af15aa2b49301d6d51cc',
            'Accepted gate report differs')
    progress=[json.loads(s) for s in (train/'progress.jsonl').read_text().splitlines()]
    gradients=[json.loads(s) for s in (train/'gradient_checks.jsonl').read_text().splitlines()]
    require([r['step'] for r in progress]==list(range(1,279)),'Progress missing/duplicated steps')
    require([r['step'] for r in gradients]==list(range(1,279)),'Gradient steps differ')
    for row,gr in zip(progress,gradients):
        step=row['step'];expected=.0002*(step/63 if step<63 else (1250-step)/1187)
        require(math.isclose(row['epoch'],step/250) and all(math.isclose(x,expected,abs_tol=1e-15) for x in row['lr']),
                'Actual scheduler/order epoch differs')
        require(row['headroom_fraction']>=.15,'Memory headroom failed')
        require(len(gr['gradients'])==456 and all(v['finite'] and math.isfinite(v['norm']) for v in gr['gradients'].values()),
                'Missing/nonfinite adapter gradients')
        if step>=3:require(all(v['nonzero'] for v in gr['gradients'].values()),'Later adapter gradient missing')
    state=read(train/'checkpoint-250/trainer_state.json')
    require(state['global_step']==250 and state['max_steps']==1250 and state['epoch']==1.,'Checkpoint step differs')
    args=read(train/'ACTUAL_TRAINING_ARGUMENTS.json')
    for key,value in dict(num_train_epochs=5,per_device_train_batch_size=1,per_device_eval_batch_size=1,
                          gradient_accumulation_steps=8,learning_rate=.0002,fp16=True,bf16=False,
                          load_best_model_at_end=False,save_total_limit=5,eval_strategy='epoch',save_strategy='epoch',seed=42,data_seed=42).items():
        require(args[key]==value,'Training argument differs: '+key)
    amp=read(train/'ACTUAL_AMP_INITIALIZATION.json')
    require(amp['after']['scale']==1024. and amp['same_scaler_object'] and amp['initialized_before_any_forward_backward'],
            'Actual Trainer AMP initialization differs')
    initial=read(train/'INITIAL_MODEL_BINDINGS.json')
    require(initial==read(gate/'INITIAL_MODEL_BINDINGS.json'),'Fresh training initialization differs from named approved initialization')
    require(not (train/'COMPLETION.json').exists() and not (train/'FAILURE_STOP_NO_RETRY.json').exists(),
            'Unexpected completion/failure record: inspect before interpreting interruption')
    evaluations=[x for x in state['log_history'] if 'eval_loss' in x]
    require(len(evaluations)==1 and evaluations[0]['step']==250 and math.isfinite(evaluations[0]['eval_loss']),
            'Epoch1 validation record differs')
    # Safe CPU tensor loading only, after exact component hashes have been verified.
    import torch
    from safetensors.torch import load_file
    from unittest.mock import patch
    cp=train/'checkpoint-250'
    with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden during audit')):
        tensors=load_file(cp/'adapter_model.safetensors',device='cpu')
        require(len(tensors)==456 and sum(t.numel() for t in tensors.values())==4915200,'Adapter geometry differs')
        require(all(torch.isfinite(t).all().item() for t in tensors.values()),'Nonfinite saved adapter')
        optimizer=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=True)
        require(len(optimizer['state'])==456 and {int(v['step']) for v in optimizer['state'].values()}=={250},
                'Saved AdamW step/tensor counts differ')
        require(all(torch.isfinite(t).all().item() for v in optimizer['state'].values() for t in v.values() if isinstance(t,torch.Tensor)),
                'Nonfinite saved optimizer tensor')
        scaler=torch.load(cp/'scaler.pt',map_location='cpu',weights_only=True)
        scheduler=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=True)
        require(scaler['scale']==1024. and scaler['_growth_tracker']==250,'Checkpoint AMP state differs')
        require(scheduler['last_epoch']==250 and all(math.isclose(x,.0002*1000/1187) for x in scheduler['_last_lr']),
                'Checkpoint scheduler differs')
    duration=(datetime.fromisoformat(progress[-1]['utc'])-datetime.fromisoformat(progress[0]['utc'])).total_seconds()
    summary=dict(archive_sha256=EXPECTED_ZIP,archive_bytes=ZIP.stat().st_size,archive_members=len(names),
        notebook_sha256=sha(NOTEBOOK),notebook_code='IDENTICAL_EXCEPT_RESEARCHER_CONSENT_TRUE',
        approval=approval,checkpoint_integrity=checked,status='INTERRUPTED_INCOMPLETE_NOT_NUMERICAL_FAILURE',
        researcher_reports_stopped_or_disconnected=True,last_saved_progress=progress[-1],
        checkpoint_step=250,uncheckpointed_updates=28,complete_training=False,validation_outputs_generated=0,
        gradient_records=278,finite_tensors_per_record=456,epoch1_validation=evaluations[0],
        min_headroom_fraction=min(x['headroom_fraction'] for x in progress),
        mean_seconds_per_update_between_first_and_last=duration/277,
        scaler_checkpoint=scaler,scheduler_checkpoint_step=scheduler['last_epoch'],
        recovery_policy='No automatic restart or rollback; restoring250 repeats28previously logged updates and needs explicit recovery approval.',
        no_GPU_execution=True,no_scientific_artifact_changes=True,D051_strict_replay='FAIL_UNCHANGED')
    write(OUT/'AUDIT_SUMMARY.json',summary)
    write(OUT/'SHA256_MANIFEST.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()})
    print(json.dumps(dict(status=summary['status'],progress=278,checkpoint=250,validation_loss=evaluations[0]['eval_loss'],
                         checkpoint_files=checked,seconds_per_update=duration/277),indent=2))


if __name__=='__main__':main()
