"""Approved-host bridge to versioned native CE code. No work is performed on import."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import os,sys,time,threading,traceback
import led_large_boundary_dual_contract_v2 as C
from led_large_boundary_dual_control_v2 import lease_check,barrier,checkpoint_io,parent_death_guard
from led_large_boundary_dual_safety_v2 import DeviceSafety

def arm_binding(package,arm):
    p=Path(package);prep=p/'outputs/stage_i/led_large_boundary_dual_v2'
    return dict(**C.package_binding(p),attention_arm=arm,
        runtime_config_sha256=C.sha(prep/('RUNTIME_CONFIG_'+arm+'.json')),
        native_runner_sha256=C.sha(p/'scripts/led_large_boundary_native_runner750_v2.py'),
        native_helper_sha256=C.sha(p/'scripts/led_large_boundary_native_helper750_v2.py'),
        AMP_sha256=C.sha(p/'scripts/led_large_boundary_amp750_v2.py'))

def verify_package(package):
    package=Path(package)
    C.verify_manifest(package,C.read(package/'outputs/stage_i/led_large_boundary_dual_v2/PREFLIGHT_MANIFEST.json')['files'])
    return C.read(package/'configs/stage_i_led_large_boundary_dual_v2.json')

def load_parent(package,parent,mask_root,a):
    """Called only after central authorization, including explicit integrity-data access."""
    package,parent,mask_root=map(Path,(package,parent,mask_root))
    spec=verify_package(package)
    C.require(a.get('validation_integrity_access') is True,'No data-integrity permission')
    for rel,digest in spec['historical_source_bindings'].items():
        C.require(C.sha(parent/'scripts'/Path(rel).name)==digest,'Native historical source differs')
    C.require(C.sha(parent/spec['parent_config_path'])==spec['native_parent_config_sha256'],'Original config differs')
    sys.path[:0]=[str(package/'scripts'),str(mask_root/'scripts'),str(parent/'scripts')]
    import stage_i_led_large_decoder_policy_v2 as original
    C.require(original.ROOT.resolve()==parent.resolve(),'Wrong already-imported native parent; fresh process required')
    original.preflight()  # Future approved integrity verification; never run by CPU suite.
    m=mask_root/'outputs/stage_i/led_large_boundary_global_v1'
    C.require(C.sha(m/'PREFLIGHT_MANIFEST.json')==spec['frozen_mask_preflight_sha256'],'Frozen mask manifest differs')
    C.verify_manifest(mask_root,C.read(m/'PREFLIGHT_MANIFEST.json')['files'])
    from led_large_boundary_global_v1 import MaskIndex
    rows=[original.json.loads(s) for s in (m/'ANCHOR_ROWS.jsonl').read_text().splitlines()]
    C.require(len(rows)==2200,'Frozen anchor population differs')
    index=MaskIndex(rows)
    import led_large_boundary_native_helper750_v2 as h
    import led_large_boundary_native_runner750_v2 as b
    return spec,index,h,b,m

def protected(a):
    p=Path(a['protected_CE3_checkpoint']);receipt=p/'checkpoint_integrity.json'
    C.require(C.sha(receipt)==a['protected_CE3_integrity_sha256'],'Protected CE3 receipt differs')
    r=C.read(receipt);C.require(r['global_step']==750,'Wrong protected CE3 step')
    C.require(r['files'].get('adapter_model.safetensors')=='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60','Wrong protected CE3 adapter')
    for n,d in r['files'].items():
        C.require(Path(n).name==n and C.sha(p/n)==d,'Protected CE3 component changed')
    return dict(receipt_sha256=C.sha(receipt),components=len(r['files']))

def run_arm(package,parent,mask_root,suite,arm,phase,approval_path,run_id):
    package,parent,mask_root,suite=map(lambda p:Path(p).resolve(),(package,parent,mask_root,suite))
    a=C.authorize(approval_path,phase,package)
    parent_death_guard(int(os.environ['LED_BOUNDARY_COORDINATOR_PID']))
    spec=verify_package(package)
    out=C.private_path(suite,arm);out.mkdir(exist_ok=True)
    device=a['common_decoding_device'] if phase=='decode' else a['training_devices'][arm]
    C.require(os.environ.get('CUDA_VISIBLE_DEVICES')==device and os.environ.get('CONTAINER_ID')==str(a['instance_id']),'Unapproved worker environment')
    C.require(suite.parent==parent/'runs' and suite.name.startswith(C.EXPERIMENT),'Output must be fresh isolated parent/runs suite')
    protected_dir=Path(a['protected_CE3_checkpoint']).resolve()
    C.require(not suite.is_relative_to(protected_dir) and not protected_dir.is_relative_to(suite),'Output overlaps protected checkpoint')
    gate_approval=C.read(suite/'GATE_AUTHORIZATION.json')
    expected_gate=a.get('accepted_gate_approval_sha256') or (C.sha(approval_path) if phase=='gates' else None)
    C.require(C.sha(suite/'GATE_AUTHORIZATION.json')==expected_gate,'Accepted gate approval link differs')
    C.require(time.time()<__import__('datetime').datetime.fromisoformat(gate_approval['execution_deadline_utc']).timestamp(),'Gate authorization expired; no extension')
    a.update(_gate_approval_sha256=C.sha(suite/'GATE_AUTHORIZATION.json'),_gate_deadline_utc=gate_approval['execution_deadline_utc'])
    before=protected(a)
    stop=threading.Event()
    def health(*unused):
        lease_check(suite,run_id)
        C.require(time.time()<__import__('datetime').datetime.fromisoformat(a['execution_deadline_utc']).timestamp(),'Authorization deadline expired')
    def watch():
        while not stop.wait(1):
            try:health()
            except BaseException as exc:
                try:C.write(suite/'PAIR_FAILURE.json',dict(status='LEASE_OR_DEADLINE_FAILURE',arm=arm,error=str(exc)))
                except FileExistsError:pass
                os._exit(70)  # Also bounds a stuck CUDA call if coordinator disappears.
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        health();_,index,h,b,m=load_parent(package,parent,mask_root,a)
        c=C.read(package/'outputs/stage_i/led_large_boundary_dual_v2'/('RUNTIME_CONFIG_'+arm+'.json'))
        C.validate_runtime(C.read(parent/spec['parent_config_path']),c,arm)
        binding=arm_binding(package,arm)
        querydir=out/(phase+'_safety');querydir.mkdir(exist_ok=False)
        safety=DeviceSafety(device,lambda n,v:C.write(querydir/n,v));safety.check(loaded=False)
        from led_large_boundary_global_v1 import install_collator
        import led_large_decoder_label_view_v1 as labels
        import run_led_large_boundary_global_v1 as cache_impl
        C.require(Path(cache_impl.__file__).resolve().is_relative_to(mask_root.resolve()),'Wrong frozen cache implementation')
        native_load=b.load_model;NativeMeter=b.MemoryMeter;native_read=b.read;native_args=b.training_arguments
        load_count=0
        stage={'cache':False}
        def load(cfg):
            nonlocal load_count
            model,before_weights,scope=native_load(cfg);load_count+=1
            import torch
            C.single_runtime(torch,device)
            safety.check(loaded=True);health()
            if phase=='gates':
                barrier(suite,'gate_models' if load_count==1 else 'cache_models',arm,run_id,health)
                if load_count==2:
                    stage['cache']=True
                    C.write(out/'CACHE_READY.json',dict(UTC_seconds=time.time()))
            return model,before_weights,scope
        class Meter(NativeMeter):
            def check(self):
                super().check();health();safety.check(loaded=True)
            def __exit__(self,*args):
                result=super().__exit__(*args)
                if stage['cache']:C.write(out/'CACHE_MEMORY_SAMPLES.json',self.samples)
                return result
        stress=C.read(m/'STRESS_CASES.json')['train_business_ids']
        def routed_read(path):
            if Path(path)==b.LARGE_PREP/'FEASIBILITY_CASES.json':return dict(business_ids=stress)
            return native_read(path)
        def train_only(cfg):
            from stage_hi_led_data import PackedLEDDataset
            from led_large_decoder_label_view_v1 import CorrectedLabelDataset
            meta=C.read(parent/cfg['stage_h_manifest']);views=C.read(parent/cfg['label_views_manifest'])
            data=CorrectedLabelDataset(PackedLEDDataset(meta['datasets']['train_filtered'],5000,768,root=parent),views['views']['train_filtered'],parent)
            C.require(len(data)==2000 and data.business_ids==C.read(b.LARGE_PREP/'EPOCH_ORDERS.json')['training_business_ids'],'Train identity/order mismatch')
            return data,None
        def args(cfg,folder):
            result=native_args(cfg,folder)
            result.update(num_train_epochs=3,max_steps=750,warmup_steps=63,save_total_limit=3)
            return result
        def observe(row):C.append(out/(phase+'_MASKS.jsonl'),row)
        with ExitStack() as stack:
            stack.enter_context(install_collator(labels,index,arm,observe))
            for mod in (h,b):
                for name,value in [('binding',lambda:binding),('check_deadline',health)]:stack.enter_context(patch.object(mod,name,value))
            stack.enter_context(patch.object(h,'CONFIG',package/'outputs/stage_i/led_large_boundary_dual_v2'/('RUNTIME_CONFIG_'+arm+'.json')))
            for name,value in [('load_model',load),('MemoryMeter',Meter),('read',routed_read),('training_arguments',args),
                    ('checkpoint_io',lambda:checkpoint_io(suite/'CHECKPOINT_IO.lock',health))]:stack.enter_context(patch.object(b,name,value))
            if phase=='gates':
                stack.enter_context(patch.object(b,'data',train_only))
                stack.enter_context(patch.object(cache_impl,'P',m))
                stack.enter_context(patch.object(cache_impl,'check_time',health))
                b.feasible(c,a,out);cache_impl.cache_stress(c,a,out,b)
                C.write(out/'CACHE_ENDED.json',dict(UTC_seconds=time.time()))
                h.require_gate(out,a)
                resource=C.read(out/'RESOURCE_GATE.json')
                C.require(resource['headroom_fraction']>=.15 and resource['cache_tokens']==768 and resource['full_cache_batch']==4 and resource['global_count']==(1 if arm=='bos_only' else 36),'Cache resource gate invalid')
                C.write(out/'GATE_COMPLETE.json',dict(status='PASS',binding=binding,approval_sha256=a['_approval_sha256'],device_UUID=device))
            elif phase=='train':
                C.require((suite/'PAIR_GATES_COMPLETE.json').is_file(),'Both gates required')
                barrier(suite,'fresh_train_start',arm,run_id,health)
                b.train(c,a,out)
            else:
                C.require((suite/'PAIR_TRAINING_COMPLETE.json').is_file(),'Both training arms required before decoding')
                b.evaluate(c,a,out)
        C.require(protected(a)==before,'Protected CE3 changed')
        # Repeat full immutable parent/mask verification after authorized use.
        load_parent(package,parent,mask_root,a)
        C.write(out/(phase+'_WORKER_COMPLETE.json'),dict(status='PASS',arm=arm,phase=phase,device_UUID=device,
            binding=binding,approval_sha256=a['_approval_sha256'],protected_CE3=before))
    except BaseException as exc:
        try:C.write(suite/'PAIR_FAILURE.json',dict(status='PAIR_INCOMPLETE_STOP_NO_RETRY',arm=arm,phase=phase,error=str(exc),traceback=traceback.format_exc()))
        except FileExistsError:pass
        raise
    finally:stop.set();thread.join(timeout=2)
