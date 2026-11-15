"""One explicitly authorized recovery; preserve the failed parent and accepted code.

Only numerical recovery bounds and checkpoint continuation are amended. The
accepted FP16 objective, data, model, optimizer and generation code are reused.
"""
import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys

import stage_i_led_large_section_wce_v1 as h
import run_stage_i_led_large_section_wce_v1 as parent

ROOT=h.ROOT
PREP=ROOT/'outputs/stage_i/led_large_section_wce_recovery_v1'
CONFIG=ROOT/'configs/stage_i_led_large_section_wce_recovery_v1.json'
SOURCE=ROOT/'runs/F_large_section_wce_v1'
OUTPUT=ROOT/'runs/F_large_section_wce_v1_recovery_v1'
APPROVAL=ROOT/'SECTION_WCE_RECOVERY_V1_APPROVAL.json'

def binding():
    return dict(config_sha256=h.sha(CONFIG),runner_sha256=h.sha(Path(__file__)),
                preflight_manifest_sha256=h.sha(PREP/'PREFLIGHT_MANIFEST.json'))

def verify_checkpoint(p):
    proof=h.read(p/'checkpoint_integrity.json');step=proof['global_step']
    expected=h.binding() if step<=750 else binding()
    h.require(proof['binding']==expected,'Unexpected checkpoint provenance')
    for name,digest in proof['files'].items():
        h.require(Path(name).name==name and h.sha(p/name)==digest,'Changed checkpoint component: '+name)
    return proof

def verify_source():
    spec=h.read(PREP/'SOURCE_BINDINGS.json')
    h.require(spec['source_run']==str(SOURCE) and spec['last_successful_update']==906,'Wrong failed parent')
    for name,digest in spec['files'].items():
        h.require(h.sha(SOURCE/name)==digest,'Failed source changed: '+name)
    h.require((SOURCE/'train/FAILURE_STOP_NO_RETRY.json').is_file(),'Original failure evidence required')
    for step in (250,500,750):verify_checkpoint(SOURCE/'train'/f'checkpoint-{step}')
    h.require(h.read(SOURCE/'train/early_sanity/REPORT.json')['status']=='PASS_TECHNICAL_ONLY','Original sanity missing')
    return spec

def authorize():
    a=h.read(APPROVAL);c=h.read(CONFIG)
    h.require(a['recovery_execution_approved'] is True and a['approval_source']=='researcher_message'
              and a['recovery_user_instruction'],'Explicit recovery approval required')
    h.require(a['recovery_binding']==binding() and a['recovery_policy']==c['recovery_policy'],'Recovery binding/policy differs')
    h.require(a['restored_checkpoint_step']==750 and a['superseded_uncheckpointed_updates']==156
              and a['additional_successful_updates_max']==500,'Wrong rollback/execution budget')
    h.require(a['parent_authorization_sha256']==h.sha(ROOT/'SECTION_WCE_RESEARCHER_APPROVAL.json'),'Parent approval changed')
    h.require(a['resume_checkpoint_integrity_sha256']==h.sha(SOURCE/'train/checkpoint-750/checkpoint_integrity.json'),'Source checkpoint differs')
    h.require(a['resume_after_interruption_approved'] is True and not a['RO_U_training_authorized']
              and not a['Teacher_calls'] and not a['held_out_access'],'Forbidden scope')
    h.check_deadline(a)  # Original deadline/reserve, never reset or extended.
    return dict(a,_approval_sha256=h.sha(APPROVAL))

def preflight():
    c=h.preflight()  # Complete unchanged parent runtime/data/backbone/objective chain.
    for rel,digest in h.read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        h.require(h.sha(ROOT/rel)==digest,'Recovery file differs: '+rel)
    verify_source();authorize()
    return c

def prefix_bytes(raw,maximum,key='step'):
    rows=[]
    for line in raw.splitlines(keepends=True):
        row=json.loads(line)
        if row[key]<=maximum:rows.append(line)
    return b''.join(rows)

def prepare():
    preflight();h.require(not OUTPUT.exists(),'Recovery namespace exists; do not repeat preparation')
    OUTPUT.mkdir();train=OUTPUT/'train';train.mkdir()
    for step in (250,500,750):
        shutil.copytree(SOURCE/'train'/f'checkpoint-{step}',train/f'checkpoint-{step}')
    shutil.copytree(SOURCE/'train/early_sanity',train/'early_sanity')
    for name in ('AUTHORIZATION.json','INITIAL_MODEL_BINDINGS.json','ACTUAL_TRAINING_ARGUMENTS.json',
                 'ACTUAL_AMP_INITIALIZATION.json','ACCEPTED_VAST_GATE.json'):
        shutil.copy2(SOURCE/'train'/name,train/name)
    for name in ('progress.jsonl','gradient_checks.jsonl','CHECKPOINT_RECEIPTS.jsonl'):
        (train/name).write_bytes(prefix_bytes((SOURCE/'train'/name).read_bytes(),750))
    (train/'AMP_DYNAMIC.jsonl').write_bytes(prefix_bytes((SOURCE/'train/AMP_DYNAMIC.jsonl').read_bytes(),750,'successful_updates'))
    h.write(train/'BINDING.json',binding())
    h.write(OUTPUT/'RECOVERY_LINEAGE.json',dict(source_run=str(SOURCE),source_binding=h.binding(),
        recovery_binding=binding(),checkpoint_bytes_preserved=True,restored_checkpoint_step=750,
        parent_last_successful_step=906,superseded_uncheckpointed_steps=list(range(751,907)),
        recomputed_successful_updates=156,cumulative_successful_updates_including_discarded_branch_max=1406,
        logical_final_step=1250,original_failed_run_unchanged=True,exact_CUDA_replay_claimed=False,
        D051_strict_replay='FAIL_UNCHANGED',original_sanity_reused_without_new_generation=True,
        numerical_policy_amendment=h.read(CONFIG)['recovery_policy']))
    for step in (250,500,750):verify_checkpoint(train/f'checkpoint-{step}')
    verify_source();print('RECOVERY PREPARED; parent unchanged',flush=True)

def installed_train_function():
    # Versioned, exact single-line extension of the accepted implementation.
    # Its existing checkpoint hash, namespace, progress and state guards remain.
    source=inspect.getsource(parent.train)
    before='require(resume is None, "Fresh corrected-label training only; old adapters/state forbidden")'
    after='require(resume == OUTPUT/"train/checkpoint-750", "Only the authorized checkpoint750 recovery is allowed")'
    h.require(source.count(before)==1,'Accepted train extension point changed')
    namespace=dict(parent.__dict__,OUTPUT=OUTPUT)
    exec(compile(source.replace(before,after),str(Path(__file__))+':accepted_train_extension','exec'),namespace)
    return namespace['train'],namespace

def install(c,a):
    import torch
    import led_large_dynamic_amp_v2 as amp
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    config=h.read(CONFIG);policy=config['recovery_policy']
    amp.POLICY=dict(amp.POLICY,max_overflow_skips=policy['max_overflow_skips'],
                    minimum_scale=policy['minimum_scale'],train_microbatches_max=policy['train_microbatches_max'])
    Base=h._core_callback_class()

    class Record(Base):
        def on_train_begin(self,args,state,control,**kw):
            super().on_train_begin(args,state,control,**kw)
            h.require(state.global_step==750 and state.max_steps==1250,'Resume step/schedule differs')
            opt=self.trainer.optimizer;scaler=self.trainer.accelerator.scaler
            expected=h.read(OUTPUT/'train/checkpoint-750/AMP_DYNAMIC_STATE.json')
            h.require({int(v['step']) for v in opt.state.values()}=={750},'Optimizer did not restore step750')
            h.require(scaler.state_dict()==torch.load(OUTPUT/'train/checkpoint-750/scaler.pt',map_location='cpu',weights_only=False),'Scaler was reset instead of restored')
            h.require(self.trainer.lr_scheduler.state_dict()==torch.load(OUTPUT/'train/checkpoint-750/scheduler.pt',map_location='cpu',weights_only=False),'Scheduler restore differs')
            weights=load_file(OUTPUT/'train/checkpoint-750/adapter_model.safetensors',device='cpu')
            actual=get_peft_model_state_dict(self.trainer.model)
            h.require(set(actual)==set(weights) and all(torch.equal(v.detach().cpu(),weights[k]) for k,v in actual.items()),'Adapter restore differs')
            for s in opt.state.values():
                h.require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in s.values()),'Nonfinite restored optimizer state')
            recovery=self.trainer.amp_recovery
            recovery.successful_updates=expected['successful_updates']
            recovery.overflow_skips=expected['overflow_skips']
            recovery.microbatches=expected['actual_forward_backward_microbatches']
            h.require(recovery.successful_updates==750 and recovery.overflow_skips==5
                      and recovery.microbatches==6040 and scaler.get_scale()==2048.,'Recovery counters/scale differ')
            h.write(self.folder/'RECOVERY_STATE_RESTORE_PASS.json',dict(adapter_exact=True,optimizer_step=750,
                optimizer_finite=True,scheduler_exact=True,scaler_exact=True,restored_scale=scaler.get_scale(),
                recovery_counters=recovery.state(scaler),RNG_restored_by_accepted_pinned_hook=True))
            h.write(self.folder/'ACTUAL_RECOVERY_TRAINING_ARGUMENTS.json',args.to_dict())

        def seal(self,state):
            p=self.folder/f'checkpoint-{state.global_step}'
            h.write(p/'AMP_DYNAMIC_STATE.json',self.trainer.amp_recovery.state(self.trainer.accelerator.scaler))
            names=('adapter_model.safetensors','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth','trainer_state.json')
            h.require(all((p/n).is_file() for n in names),'Incomplete recovery checkpoint')
            h.write(p/'checkpoint_integrity.json',dict(binding=binding(),global_step=state.global_step,
                files={q.name:h.sha(q) for q in p.iterdir() if q.is_file() and q.name!='checkpoint_integrity.json'}))
            h.append(self.folder/'CHECKPOINT_RECEIPTS.jsonl',dict(step=state.global_step,
                checkpoint_integrity_sha256=h.sha(p/'checkpoint_integrity.json')))

        def on_step_end(self,args,state,control,**kw):
            super().on_step_end(args,state,control,**kw)
            if state.global_step==753:
                h.write(self.folder/'RECOVERY_FIRST3_UPDATES_PASS.json',dict(status='PASS',updates=[751,752,753],
                    all_456_gradients_finite=True,logical_updates_not_extra_probe_updates=True,**self.memory.summary()))
            if state.global_step==906:
                # One operational full-state anchor before the previously failing window.
                self.trainer._save_checkpoint(self.trainer.model,trial=None)
                self.seal(state)
            if state.global_step==907:
                h.write(self.folder/'PREVIOUS_FAILURE_WINDOW_CROSSED.json',dict(status='FINITE_SUCCESSFUL_UPDATE',
                    step=907,scale=float(self.trainer.accelerator.scaler.get_scale()),
                    exact_reproduction_of_old_trajectory_not_claimed=True))

        def on_save(self,args,state,control,**kw):
            h.require(state.global_step in (1000,1250),'Unexpected new epoch checkpoint')
            self.seal(state)

    def gate(out,approval):
        original=h.authorize(ROOT/'SECTION_WCE_RESEARCHER_APPROVAL.json','train',True)
        return h.require_gate(SOURCE,original)

    train,namespace=installed_train_function()
    namespace.update(binding=binding,checkpoint_verified=verify_checkpoint,callback_class=lambda:Record,require_gate=gate)
    parent.binding=binding;parent.checkpoint_verified=verify_checkpoint;parent.require_gate=gate
    return train

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('preflight','prepare','train','evaluate','report'))
    mode=ap.parse_args().mode
    if mode=='prepare':prepare();return
    c=preflight();a=authorize()
    if mode=='preflight':print('RECOVERY CPU PREFLIGHT PASS',binding(),flush=True);return
    h.require(OUTPUT.is_dir() and h.read(OUTPUT/'RECOVERY_LINEAGE.json')['restored_checkpoint_step']==750,'Prepared branch required')
    if mode=='report':
        import report_stage_i_led_large_section_wce_v1 as report
        report.binding=binding;sys.argv=[sys.argv[0],'--output',str(OUTPUT)];report.main()
        import zipfile
        dest=OUTPUT/'quality_review_recovery_v1';shutil.copytree(OUTPUT/'quality_review',dest)
        for src in (OUTPUT/'RECOVERY_LINEAGE.json',APPROVAL,CONFIG):shutil.copy2(src,dest/src.name)
        (dest/'RECOVERY_DISCLOSURE.md').write_text(
            'Numerical recovery, not an untouched fresh-run completion. Parent stopped after906 updates. '
            'Checkpoint750 restored with full optimizer/scheduler/scaler/RNG; steps751-906 recomputed '
            '(156 discarded-branch successful updates). FP16 retained; AMP bounds explicitly amended '
            'from max8/floor256 to max16/floor1. No exact CUDA replay claim; D051 strict replay FAIL unchanged. '
            'Epoch3 uses original immutable checkpoint750; epoch5 comes from the recovery branch. '
            'Comparison with saved CE baseline is provisional and includes this numerical-policy/recovery confound. '
            'Original targets, source inputs, loss weights, initialization, effective batch, optimizer, '
            'schedule and beam4 decoding remain unchanged. No RO/U, held-out or new Teacher generation. '
            'This Vast instance has no persistent volume: full checkpoints must be backed up locally before destruction. '
            'Quality GO remains pending independent researcher review.\n',encoding='utf-8')
        manifest=dest/'SHA256_MANIFEST.json';manifest.unlink()
        h.write(manifest,dict(files={p.relative_to(dest).as_posix():h.sha(p) for p in dest.rglob('*') if p.is_file()}))
        archive=OUTPUT/'LED_LARGE_SECTION_WCE_RECOVERY_V1_CHATGPT_REVIEW.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(dest.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(dest).as_posix())
        h.write(OUTPUT/'RECOVERY_QUALITY_BUNDLE_RECEIPT.json',dict(archive=archive.name,sha256=h.sha(archive),
            bytes=archive.stat().st_size,actual_predictions=48,researcher_quality_decision='PENDING'))
        print(archive,flush=True);return
    train=install(c,a)
    if mode=='train':train(c,a,OUTPUT,resume=OUTPUT/'train/checkpoint-750')
    else:parent.evaluate(c,a,OUTPUT)
    verify_source()

if __name__=='__main__':main()
