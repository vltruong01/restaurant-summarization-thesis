"""CPU closeout repair after all1250 updates; evaluation only, never training."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import zipfile
import run_stage_i_led_large_section_wce_recovery_v1 as r
import stage_i_led_large_section_wce_v1 as h

RUN=r.OUTPUT

def checked_closeout():
    proof=h.read(RUN/'FINALIZATION_VERIFIED.json')
    h.require(proof['optimizer_updates']==1250 and proof['epochs']==5
        and proof['only_post_training_checkpoint_rotation_failure'] is True,'Closeout repair not verified')
    h.require(h.sha(RUN/'train/FAILURE_STOP_NO_RETRY.json')==proof['preserved_failure_sha256'],'Failure evidence changed')
    for step in (250,500,750,1000,1250):r.verify_checkpoint(RUN/'train'/f'checkpoint-{step}')
    return proof

def finalize():
    import torch
    r.preflight()
    failure=RUN/'train/FAILURE_STOP_NO_RETRY.json';error=h.read(failure)
    expected=str(RUN/'train/checkpoint-250/checkpoint_integrity.json')
    h.require(error['error']=="[Errno 2] No such file or directory: '"+expected+"'",'Unexpected failure; do not relabel numerical error')
    state=h.read(RUN/'train/checkpoint-1250/trainer_state.json')
    h.require(state['global_step']==1250 and state['max_steps']==1250 and state['epoch']==5,'Training did not finish')
    dest=RUN/'train/checkpoint-250'
    if not dest.exists():shutil.copytree(r.SOURCE/'train/checkpoint-250',dest)
    for step in (250,500,750,1000,1250):r.verify_checkpoint(RUN/'train'/f'checkpoint-{step}')
    for name in ('progress.jsonl','gradient_checks.jsonl'):
        seen=[]
        with (RUN/'train'/name).open() as f:
            for line in f:
                row=json.loads(line);seen.append(row['step'])
                if name=='gradient_checks.jsonl':h.require(len(row['gradients'])==456,'Missing gradient tensor record')
        h.require(seen==list(range(1,1251)),'Noncontiguous or incomplete optimizer-update evidence')
    p=RUN/'train/checkpoint-1250'
    opt=torch.load(p/'optimizer.pt',map_location='cpu',weights_only=False)
    h.require(len(opt['state'])==456 and {int(s['step']) for s in opt['state'].values()}=={1250},'Terminal optimizer state mismatch')
    h.require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for s in opt['state'].values() for v in s.values()),'Nonfinite terminal optimizer')
    scheduler=torch.load(p/'scheduler.pt',map_location='cpu',weights_only=False)
    h.require(scheduler['last_epoch']==1250 and all(x==0 for x in scheduler['_last_lr']),'Terminal scheduler mismatch')
    scaler=torch.load(p/'scaler.pt',map_location='cpu',weights_only=False)
    h.require(scaler['scale']>=1,'Invalid terminal scale')
    from safetensors.torch import load_file
    h.require(all(torch.isfinite(v).all().item() for v in load_file(p/'adapter_model.safetensors').values()),'Nonfinite terminal adapter')
    proof=dict(optimizer_updates=1250,epochs=5,terminal_optimizer_finite=True,terminal_adapter_finite=True,
        terminal_scheduler_step=1250,terminal_scaler=scaler,all1250_gradient_records_present=True,
        only_post_training_checkpoint_rotation_failure=True,checkpoint250_restored_byte_identically=True,
        preserved_failure_sha256=h.sha(failure),no_training_or_optimizer_update_during_repair=True,
        training_end_live_backbone_hash_not_reached=True,
        frozen_training_contract='Frozen parameters excluded from optimizer; frozen gradients guarded at every update',
        inference_backbone_live_hash_verification_required=True,
        cause='Extra operational anchor906 plus save_total_limit5 rotated checkpoint250 at final save',
        source250_receipt_sha256=h.sha(r.SOURCE/'train/checkpoint-250/checkpoint_integrity.json'),
        restored250_receipt_sha256=h.sha(dest/'checkpoint_integrity.json'),recovery_binding=r.binding())
    h.write(RUN/'FINALIZATION_VERIFIED.json',proof)
    h.write(RUN/'train/COMPLETION.json',dict(status='TRAINING_COMPLETE_WITH_CPU_CLOSEOUT_REPAIR',
        binding=r.binding(),steps=1250,epochs=5,generation_calls=4,
        checkpoint_steps=[250,500,750,1000,1250],extra_operational_checkpoint=906,
        completion_basis='Verified terminal checkpoint/optimizer/scheduler and contiguous update/gradient receipts',
        amp_dynamic=h.read(p/'AMP_DYNAMIC_STATE.json'),closeout_repair_receipt_sha256=h.sha(RUN/'FINALIZATION_VERIFIED.json'),
        final_in_memory_training_backbone_hash_unavailable=True,quality_GO='PENDING'))
    checked_closeout();r.verify_source();print('CPU CLOSEOUT REPAIR PASS:1250 successful updates; no training replay',flush=True)

def evaluate():
    c=r.preflight();a=r.authorize();checked_closeout();r.install(c,a)
    source=inspect.getsource(r.parent.evaluate)
    before="""require(not (out/'train/FAILURE_STOP_NO_RETRY.json').exists()
            and read(out/'train/FROZEN_BACKBONE_INTEGRITY.json')['unchanged'], 'Training not accepted technically')"""
    after="""require(checked_closeout()['only_post_training_checkpoint_rotation_failure'] is True,
            'CPU closeout repair required; original error receipt remains preserved')"""
    h.require(source.count(before)==1,'Pinned evaluation extension point changed')
    namespace=dict(r.parent.__dict__,checked_closeout=checked_closeout)
    exec(compile(source.replace(before,after),str(Path(__file__))+':evaluation_only','exec'),namespace)
    namespace['evaluate'](c,a,RUN)

def report():
    checked_closeout();sys.argv=[sys.argv[0],'report'];r.main()
    old=RUN/'LED_LARGE_SECTION_WCE_RECOVERY_V1_CHATGPT_REVIEW.zip'
    output=RUN/'LED_LARGE_SECTION_WCE_RECOVERY_FINAL_REVIEW.zip'
    with zipfile.ZipFile(old) as z:files={n:z.read(n) for n in z.namelist() if not n.endswith('/') and n!='SHA256_MANIFEST.json'}
    files['FINALIZATION_VERIFIED.json']=(RUN/'FINALIZATION_VERIFIED.json').read_bytes()
    files['POST_TRAINING_CLOSEOUT_DISCLOSURE.md']=(
        '# Post-training closeout repair\n\nAll1250 optimizer updates and epoch5 validation finished. '
        'The extra operational checkpoint906 caused the unchanged five-checkpoint rotation rule '
        'to remove checkpoint250 in the recovery directory. The final inventory check stopped '
        'before writing completion/live end-of-training backbone hashes. Checkpoint250 was restored '
        'byte-for-byte from the immutable parent, and all five checkpoints, terminal finite '
        'adapter/optimizer states, scheduler1250 and contiguous1250 gradient/update records were '
        'verified on CPU. No additional optimizer update or training replay occurred during this repair. '
        'The closeout exception is preserved. A direct final in-memory TRAINING backbone hash is unavailable; '
        'training used frozen-gradient guards and trainable-only optimizer. Evaluation loads the pinned '
        'backbone afresh and performs the original before/after live backbone hash verification. '
        'This implementation/verification limitation must be disclosed, not mistaken for a clean '
        'uninterrupted execution. Quality GO remains pending independent review.\n').encode()
    files['SHA256_MANIFEST.json']=json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2).encode()
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(files.items()):z.writestr(n,b)
    h.write(RUN/'FINAL_QUALITY_BUNDLE_RECEIPT.json',dict(archive=output.name,sha256=h.sha(output),bytes=output.stat().st_size,
        actual_predictions=48,researcher_quality_decision='PENDING'))
    print(output,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('finalize','evaluate','report'))
    mode=ap.parse_args().mode
    {'finalize':finalize,'evaluate':evaluate,'report':report}[mode]()
