"""Derive versioned native components from frozen source. Never opens data payloads."""
from pathlib import Path
import ast,difflib,hashlib,json,sys
R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/led_large_boundary_dual_v2'
F=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def emit(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)
def derive():
    require_paths=[R/'scripts/led_large_boundary_amp750_v2.py',R/'scripts/led_large_boundary_native_helper750_v2.py',R/'scripts/led_large_boundary_native_runner750_v2.py']
    assert not any(p.exists() for p in require_paths),'No overwrite of versioned source'
    records=[]
    amp=(F/'scripts/led_large_dynamic_amp_v2.py').read_text()
    amp_new=amp.replace('10064','6064').replace('1250','750')
    records.append(('led_large_dynamic_amp_v2.py','led_large_boundary_amp750_v2.py',amp,amp_new))
    helper=(F/'scripts/stage_i_led_large_decoder_policy_v2.py').read_text()
    tree=ast.parse(helper)
    names={'require_gate','configure_amp_scaler','_core_callback_class','callback_class'}
    parts=[ast.get_source_segment(helper,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(parts)==4
    hnew='"""Versioned 750-step callbacks/gate verifier; native CE/scaler preserved."""\nfrom stage_i_led_large_decoder_policy_v2 import *\nfrom led_large_boundary_amp750_v2 import POLICY\nfrom led_large_boundary_dual_contract_v2 import scheduler_accounting\n\n'+'\n\n'.join(parts)+'\n'
    hnew=hnew.replace('(250,500,750,1000,1250)','(250,500,750)').replace('1250','750').replace('Only five approved epoch checkpoints','Only three approved epoch checkpoints')
    hnew=hnew.replace("r['approval_sha256'] == a['_approval_sha256']", "r['approval_sha256'] == a.get('_gate_approval_sha256', a['_approval_sha256'])")
    hnew=hnew.replace("r['deadline_utc'] == a['execution_deadline_utc']", "r['deadline_utc'] == a.get('_gate_deadline_utc', a['execution_deadline_utc'])")
    needle="write(p/'AMP_DYNAMIC_STATE.json',self.trainer.amp_recovery.state(self.trainer.accelerator.scaler))"
    hnew=hnew.replace(needle,needle+"\n            write(p/'SCHEDULER_ACCOUNTING.json',scheduler_accounting(self.trainer.lr_scheduler,state.global_step))")
    records.append(('stage_i_led_large_decoder_policy_v2.py','led_large_boundary_native_helper750_v2.py',helper,hnew))
    base=(F/'scripts/run_stage_i_led_large_decoder_policy_v2.py').read_text()
    assert base.count('\ndef main():')==1
    bnew=base.split('\ndef main():')[0]
    edits=[('from led_large_dynamic_amp_v2 import','from led_large_boundary_amp750_v2 import'),
      ('from stage_i_led_large_decoder_policy_v2 import','from led_large_boundary_native_helper750_v2 import'),
      ('for epoch,step in ((3,750),(5,1250)):','for epoch,step in ((3,750),):'),
      ('(250,500,750,1000,1250)','(250,500,750)'),('[250,500,750,1000,1250]','[250,500,750]'),
      ('trainer.state.epoch == 5','trainer.state.epoch == 3'),('epochs=5','epochs=3'),
      ('count <= 48','count <= 24'),('count == 48','count == 24'),('generation_calls=48','generation_calls=24'),
      ('48_OUTPUTS_COMPLETE','24_OUTPUTS_COMPLETE'),('five-epoch','three-epoch')]
    for a,b in edits:
        assert a in bnew,a;bnew=bnew.replace(a,b)
    bnew=bnew.replace('1250','750')
    needle="['nvidia-smi','--query-gpu=driver_version'"
    assert bnew.count(needle)==1
    bnew=bnew.replace(needle,"['nvidia-smi','-i',os.environ['LED_BOUNDARY_GPU_UUID'],'--query-gpu=driver_version'")
    needle='self.initial_amp_record = configure_amp_scaler(self.accelerator,c)'
    bnew=bnew.replace(needle,needle+"\n                single_runtime(torch,os.environ['LED_BOUNDARY_GPU_UUID'],self.accelerator)")
    bnew='from led_large_boundary_dual_contract_v2 import single_runtime\ndef checkpoint_io():\n    raise ValueError(\'Approved checkpoint I/O coordinator not installed\')\n'+bnew
    bnew=bnew.replace('class Trainer(Seq2SeqTrainer):','class Trainer(Seq2SeqTrainer):\n            def _save_checkpoint(self,*args,**kwargs):\n                with checkpoint_io():\n                    return super()._save_checkpoint(*args,**kwargs)')
    records.append(('run_stage_i_led_large_decoder_policy_v2.py','led_large_boundary_native_runner750_v2.py',base,bnew))
    diffs=[];sources={}
    for old,new,a,b in records:
        ast.parse(b);emit(R/'scripts'/new,b)
        sources[(F/'scripts'/old).relative_to(R).as_posix()]=sha(F/'scripts'/old)
        diffs.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='frozen/'+old,tofile='V2/'+new))
    emit(P/'NATIVE_COMPONENT_DIFF.patch',''.join(diffs))
    old=read(F/'configs/stage_i_led_large_decoder_policy_v2.json')
    from led_large_boundary_dual_contract_v2 import EXPERIMENT,AMP,SCALER,LIMITS,validate_runtime
    for arm in ('bos_only','boundary'):
        c=dict(old,experiment=EXPERIMENT,run_root='runs/'+EXPERIMENT+'/'+arm,
            optimization=dict(old['optimization'],num_train_epochs=3),training=dict(old['training'],epochs=3,steps=750),dynamic_overflow_policy=AMP,evaluation_epochs=[3],
            checkpoint_policy='retain_250_500_750_no_best_loss_reload',checkpoint_selection='fixed750_scientific_and_adoption_decisions_separate',
            global_attention=old['global_attention'] if arm=='bos_only' else 'BOS_PLUS_FIRST_BPE_OF_EACH_EXISTING_REVIEW_CAPTION_MARKER_V1',
            vast_fresh=dict(old['vast_fresh'],experiment=EXPERIMENT,output='runs/'+EXPERIMENT+'/'+arm))
        validate_runtime(old,c,arm)
        emit(P/('RUNTIME_CONFIG_'+arm+'.json'),json.dumps(c,indent=2)+'\n')
    config=dict(experiment=EXPERIMENT,GPU_execution_approved=False,training_authorized=False,
        native_parent_config_sha256=sha(F/'configs/stage_i_led_large_decoder_policy_v2.json'),
        parent_config_path='configs/stage_i_led_large_decoder_policy_v2.json',
        limits=LIMITS,amp_policy=AMP,amp_scaler=SCALER,mask_policy='BOS_PLUS_FIRST_BPE_OF_EACH_EXISTING_REVIEW_CAPTION_MARKER_V1',
        historical_source_bindings=sources,
        frozen_mask_preflight_sha256='3b4d91f67cf48c0757a56c2441cbb3a614359efe1e5a85c2522eabccd678a6ac',
        original_CE3_sha256='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60',
        automatic_evaluation=False,automatic_model_selection=False,held_out_access=False)
    emit(R/'configs/stage_i_led_large_boundary_dual_v2.json',json.dumps(config,indent=2)+'\n')
    print('Derived three new native modules; original source untouched. No payloads opened.')
if __name__=='__main__':derive()
