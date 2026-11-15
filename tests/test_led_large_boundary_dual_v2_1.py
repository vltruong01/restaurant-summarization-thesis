"""CPU/synthetic contract, process, native-prefix and safety regression suite.

No model/data payloads, CUDA initialization, network or instance contact.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,tempfile,unittest,subprocess,time,copy,math,ast
from types import SimpleNamespace as NS
from datetime import datetime,timedelta,timezone
from unittest.mock import Mock,patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import led_large_boundary_dual_contract_v2 as C
import led_large_boundary_dual_control_v2 as ctl
import led_large_boundary_dual_safety_v2 as S
from led_large_boundary_dual_runtime_v2 import verify_package
U='GPU-11111111-1111-1111-1111-111111111111';V='GPU-22222222-2222-2222-2222-222222222222'

def fixture_arm(suite,arm,binding,step=750,init='same'):
    p=Path(suite)/arm/'train';p.mkdir(parents=True)
    C.write(p/'COMPLETION.json',dict(binding=binding,steps=step,epochs=step/250,checkpoint_steps=[250,500,750],generation_calls=4))
    C.write(p/'early_sanity/REPORT.json',dict(status='PASS_TECHNICAL_ONLY'))
    C.write(p/'INITIAL_MODEL_BINDINGS.json',dict(initialization={'x':init}))
    C.write(Path(suite)/arm/'train_WORKER_COMPLETE.json',dict(status='PASS',arm=arm,phase='train',binding=binding))
    C.write(p/'FROZEN_BACKBONE_INTEGRITY.json',dict(unchanged=True))
    C.write(p/'ACTUAL_TRAINING_ARGUMENTS.json',dict(num_train_epochs=3,max_steps=750,warmup_steps=63,gradient_accumulation_steps=8,per_device_train_batch_size=1,fp16=True,bf16=False,load_best_model_at_end=False))
    (p/'progress.jsonl').write_text('\n'.join(json.dumps({'step':i}) for i in range(1,step+1))+'\n')
    for k in C.CHECKPOINTS:
        folder=p/f'checkpoint-{k}';folder.mkdir()
        for n in ['adapter_model.safetensors','adapter_config.json','training_args.bin','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth']:(folder/n).write_bytes(b'SYNTHETIC_CPU_TEST_ONLY')
        C.write(folder/'trainer_state.json',dict(global_step=k,max_steps=750,epoch=k/250))
        C.write(folder/'AMP_DYNAMIC_STATE.json',dict(successful_updates=k,policy=C.AMP,overflow_skips=0,actual_forward_backward_microbatches=k*8))
        C.write(folder/'SCHEDULER_ACCOUNTING.json',dict(horizon=1250,warmup_steps=63,last_epoch=k,next_learning_rates=[2e-4*C.lr_multiplier(k)]))
        C.write(folder/'checkpoint_integrity.json',dict(binding=binding,global_step=k,files={f.name:C.sha(f) for f in folder.iterdir()}))
    return p

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.p=Path(self.temp.name)
        (self.p/'configs').mkdir();(self.p/'configs/stage_i_led_large_boundary_dual_v2.json').write_text('{}')
        q=self.p/'outputs/stage_i/led_large_boundary_dual_v2';q.mkdir(parents=True);(q/'PREFLIGHT_MANIFEST.json').write_text('{"files":{}}')
        now=datetime.now(timezone.utc)
        self.a=dict(experiment=C.EXPERIMENT,GPU_execution_approved=True,approval_source='researcher_message',researcher_explicit_authorization='SYNTHETIC TEST ONLY',bindings=C.package_binding(self.p),limits=C.LIMITS,amp_policy=C.AMP,amp_scaler=C.SCALER,
            instance_id=1,training_devices=dict(bos_only=U,boundary=V),common_decoding_device=U,
            automatic_retry=False,held_out_access=False,model_selection_authorized=False,validation_integrity_access=True,
            Teacher_calls=False,RO_U_Shuffled_authorized=False,
            validation_payload_access=True,training_authorized=True,early_sanity_generation_approved=True,
            decoding_authorized=True,reference_binding_access=True,scopes=['gates','train','decode'],
            execution_authorized_at_utc=(now-timedelta(minutes=1)).isoformat(),execution_deadline_utc=(now+timedelta(hours=1)).isoformat(),researcher_absolute_stop_utc=(now+timedelta(hours=2)).isoformat(),defense_reserve_at_least_14_days_confirmed=True)
    def authorize(self,phase='gates'):
        f=self.p/'approval.json';f.write_text(json.dumps(self.a));return C.authorize(f,phase,self.p)
    def test_complete_explicit_contract(self):self.assertIn('_approval_sha256',self.authorize())
    def test_CPU_permission_is_not_GPU_permission(self):
        self.a['GPU_execution_approved']=False
        with self.assertRaises(ValueError):self.authorize()
    def test_missing_validation_permission_denies_train_decode(self):
        self.a['validation_payload_access']=False
        self.authorize('gates')
        for mode in ['train','decode']:
            with self.assertRaises(ValueError):self.authorize(mode)
    def test_gate_only_receipt_cannot_train_or_decode(self):
        self.a['scopes']=['gates']
        for phase in ['train','decode']:
            with self.assertRaises(ValueError):self.authorize(phase)
    def test_unknown_or_duplicate_scope(self):
        for scope in [['gates','extra'],['gates','gates']]:
            self.a['scopes']=scope
            with self.assertRaises(ValueError):self.authorize()
    def test_common_device_must_be_predeclared_A(self):
        self.a['common_decoding_device']=V
        with self.assertRaises(ValueError):self.authorize()
    def test_same_device_for_two_arms_rejected(self):
        self.a['training_devices']['boundary']=U
        with self.assertRaises(ValueError):self.authorize()
    def test_wrong_bindings_or_extended_budget_rejected(self):
        for key,value in [('bindings',{}),('limits',dict(C.LIMITS,updates_per_arm=751)),('amp_policy',dict(C.AMP,max_overflow_skips=9))]:
            old=self.a[key];self.a[key]=value
            with self.assertRaises(ValueError):self.authorize()
            self.a[key]=old
    def test_expiry_timezone_reserve(self):
        original=copy.deepcopy(self.a)
        for key,value in [('execution_deadline_utc','2001-01-01T00:00:00+00:00'),('defense_reserve_at_least_14_days_confirmed',False),('execution_authorized_at_utc','2026-01-01T00:00:00')]:
            self.a=copy.deepcopy(original);self.a[key]=value
            with self.assertRaises(ValueError):self.authorize()
    def test_no_resume_or_model_selection(self):
        self.a['model_selection_authorized']=True
        with self.assertRaises(ValueError):self.authorize()
    def test_exact_source_and_config_clone(self):
        old=C.read(R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/configs/stage_i_led_large_decoder_policy_v2.json')
        for arm in C.ARMS:
            new=C.read(R/'outputs/stage_i/led_large_boundary_dual_v2'/('RUNTIME_CONFIG_'+arm+'.json'));C.validate_runtime(old,new,arm)
            new['generation']['length_penalty']=1.2
            with self.assertRaises(ValueError):C.validate_runtime(old,new,arm)
    def test_scheduler_exact_prefix_values(self):
        self.assertEqual(C.lr_multiplier(0),0);self.assertEqual(C.lr_multiplier(63),1)
        self.assertAlmostEqual(2e-4*C.lr_multiplier(750),8.424599831508003e-5)
        self.assertGreater(C.lr_multiplier(750),0);self.assertEqual(C.lr_multiplier(1250),0)
    def test_scalar_scheduler_outside_horizon_rejected(self):
        for s in [-1,1251,True,2.3]:
            with self.assertRaises(ValueError):C.lr_multiplier(s)
    def test_manifest_tamper_and_traversal(self):
        f=self.p/'a';f.write_text('original');m={'a':C.sha(f)};C.verify_manifest(self.p,m);f.write_text('changed')
        with self.assertRaises(ValueError):C.verify_manifest(self.p,m)
        with self.assertRaises(ValueError):C.verify_manifest(self.p,{'../outside':'x'})
    def test_exclusive_receipt_no_overwrite(self):
        C.write(self.p/'one.json',{'first':True})
        with self.assertRaises(FileExistsError):C.write(self.p/'one.json',{'first':False})
        self.assertTrue(C.read(self.p/'one.json')['first'])

class Devices(unittest.TestCase):
    def test_UUID_environment_mapping_and_no_distributed_inheritance(self):
        env=C.clean_worker_env(dict(LOCAL_RANK='1',WORLD_SIZE='2',MASTER_ADDR='x',SAFE='keep'),V,'boundary')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],V);self.assertEqual(env['SAFE'],'keep')
        self.assertNotIn('LOCAL_RANK',env);self.assertNotIn('WORLD_SIZE',env)
    def test_logical_zero_for_physical_B(self):
        torch=NS(cuda=NS(device_count=lambda:1,current_device=lambda:0,get_device_name=lambda n:'NVIDIA GeForce RTX 3090',get_device_capability=lambda n:(8,6)),distributed=NS(is_initialized=lambda:False))
        C.single_runtime(torch,V,NS(num_processes=1,distributed_type='DistributedType.NO'),{'CUDA_VISIBLE_DEVICES':V})
        torch.cuda.current_device=lambda:1
        with self.assertRaises(ValueError):C.single_runtime(torch,V,env={'CUDA_VISIBLE_DEVICES':V})
    def test_accelerator_parallelism_rejected_with_single_visible_GPU(self):
        t=NS(cuda=NS(device_count=lambda:1,current_device=lambda:0,get_device_name=lambda n:'NVIDIA GeForce RTX 3090',get_device_capability=lambda n:(8,6)),distributed=NS(is_initialized=lambda:False))
        with self.assertRaises(ValueError):C.single_runtime(t,U,NS(num_processes=2,distributed_type='MULTI_GPU'),{'CUDA_VISIBLE_DEVICES':U})
    def test_multiple_visible_devices_or_DDP_rejected(self):
        t=NS(cuda=NS(device_count=lambda:2,current_device=lambda:0),distributed=NS(is_initialized=lambda:False))
        with self.assertRaises(ValueError):C.single_runtime(t,U,env={'CUDA_VISIBLE_DEVICES':U})
    def test_full_UUID_only(self):
        for u in ['0','GPU-short','',None]:
            with self.assertRaises(ValueError):C.uuid(u)
    def test_dual_provider_does_not_guess_RAM_field_semantics(self):
        obj=dict(id=1,actual_status='running',intended_status='running',num_gpus=2,gpu_name='RTX 3090',gpu_ram=49152)
        self.assertEqual(S.allocation({'instances':obj},1)['num_gpus'],2)
        obj['num_gpus']=1
        with self.assertRaises(ValueError):S.allocation(obj,1)
    def query(self,processes,device=U,gpu=None):
        text=gpu or (device+', NVIDIA GeForce RTX 3090, 24576, 2000, 570\n')
        return Mock(side_effect=[NS(returncode=0,stdout=text,stderr=''),NS(returncode=0,stdout=processes,stderr='')])
    def test_device_filter_and_opaque_PID(self):
        q=self.query(U+', 8888888\n');s=S.DeviceSafety(U,lambda n,v:None,q);s.check(loaded=True)
        self.assertEqual(s.expected,frozenset([8888888]));self.assertTrue(all(U in x.args[0] for x in q.call_args_list))
    def test_idle_workload_rejects(self):
        with self.assertRaises(ValueError):S.DeviceSafety(U,lambda n,v:None,self.query(U+', 77')).check(loaded=False)
    def test_foreign_multiple_changed_or_wrong_GPU_rejected(self):
        for rows in [U+', 77\n'+U+', 88',V+', 99','not csv','']:
            with self.assertRaises(ValueError):S.DeviceSafety(U,lambda n,v:None,self.query(rows)).check(loaded=True)
        s=S.DeviceSafety(U,lambda n,v:None,self.query(U+', 77'));s.check(loaded=True);s.query=self.query(U+', 78')
        with self.assertRaises(ValueError):s.check()
    def test_other_GPU_rows_not_silently_accepted(self):
        q=self.query('',gpu=U+', NVIDIA GeForce RTX 3090, 24576, 2, 570\n'+V+', NVIDIA GeForce RTX 3090, 24576, 2, 570')
        with self.assertRaises(ValueError):S.DeviceSafety(U,lambda n,v:None,q).check(loaded=False)
    def test_resource_headroom_and_disk_failure(self):
        good=dict(effective_CPU_cores=8,effective_RAM_total=64*1024**3,effective_RAM_available=32*1024**3,free_disk_bytes=30*1024**3)
        S.check_host(good,True)
        for update in [dict(effective_CPU_cores=4),dict(effective_RAM_available=1),dict(free_disk_bytes=0)]:
            with self.assertRaises(ValueError):S.check_host(dict(good,**update),True)

class Coordination(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)
    def spec(self,code,arm='bos_only'):
        return dict(command=[sys.executable,'-B','-c',code],cwd=str(self.p),env=C.clean_worker_env(os.environ,U if arm=='bos_only' else V,arm))
    def test_locks_conflict_release_and_legacy_exclusion(self):
        paths=[self.p/'legacy.lock',self.p/'gpuA.lock',self.p/'gpuB.lock']
        with ctl.Locks(paths):
            with self.assertRaises(ValueError):
                with ctl.Locks([paths[0]]):pass
        with ctl.Locks(paths):pass
    def test_two_real_CPU_processes_have_private_rng_env_and_paths(self):
        code="import os,json,random,time;from pathlib import Path;random.seed(42);time.sleep(.1);Path(os.environ['LED_BOUNDARY_ARM']+'.json').write_text(json.dumps(dict(pid=os.getpid(),UUID=os.environ['CUDA_VISIBLE_DEVICES'],random=random.random())))"
        x=ctl.run_workers(self.p,{a:self.spec(code,a) for a in C.ARMS},time.time()+10,'isolation')
        a=C.read(self.p/'bos_only.json');b=C.read(self.p/'boundary.json')
        self.assertNotEqual(a['pid'],b['pid']);self.assertNotEqual(a['UUID'],b['UUID']);self.assertEqual(a['random'],b['random']);self.assertEqual(set(x['returncodes'].values()),{0})
    def test_one_process_failure_cancels_other_no_compensating_run(self):
        started=time.monotonic()
        with self.assertRaises(ValueError):ctl.run_workers(self.p,{'bos_only':self.spec('import sys;sys.exit(7)'), 'boundary':self.spec('import time;time.sleep(20)','boundary')},time.time()+10,'failure')
        self.assertLess(time.monotonic()-started,8);self.assertTrue((self.p/'PAIR_FAILURE.json').exists())
    def test_missing_or_expired_lease_fails_closed(self):
        with self.assertRaises(FileNotFoundError):ctl.lease_check(self.p,'run')
        ctl.pulse(self.p/'LEASE.json',dict(run_id='run',time=time.time()-61,deadline=time.time()+10))
        with self.assertRaises(ValueError):ctl.lease_check(self.p,'run')
    def test_coordinator_budget_cancels_sleeping_worker(self):
        with self.assertRaises(ValueError):ctl.run_workers(self.p,{'bos_only':self.spec('import time;time.sleep(20)')},time.time()+.3,'budget',poll=.05)
        self.assertEqual(C.read(self.p/'PAIR_FAILURE.json')['status'],'PAIR_INCOMPLETE_STOP_NO_RETRY')
    def test_parent_death_registration_and_race_rejection(self):
        register=Mock();ctl.parent_death_guard(100,register,lambda:100);register.assert_called_once()
        with self.assertRaises(ValueError):ctl.parent_death_guard(100,register,lambda:101)
        with self.assertRaises(ValueError):ctl.parent_death_guard(100,register,Mock(side_effect=[100,101]))
    def test_failed_receipt_does_not_skip_cancellation(self):
        with patch.object(ctl,'write',side_effect=OSError('Synthetic disk full')):
            start=time.monotonic()
            with self.assertRaises(OSError):ctl.run_workers(self.p,{'bos_only':self.spec('import sys;sys.exit(7)'), 'boundary':self.spec('import time;time.sleep(20)','boundary')},time.time()+10,'diskfail')
            self.assertLess(time.monotonic()-start,8)
    def test_no_completion_if_only_one_arm_exists(self):
        fixture_arm(self.p,'bos_only',{'arm':'A'})
        with self.assertRaises(FileNotFoundError):C.joint_training(self.p,{'bos_only':{'arm':'A'},'boundary':{'arm':'B'}})
    def test_exact_joint_completion_and_checkpoint_integrity(self):
        bind={a:{'arm':a} for a in C.ARMS}
        for a in C.ARMS:fixture_arm(self.p,a,bind[a])
        x=C.joint_training(self.p,bind);self.assertFalse(x['automatic_evaluation']);self.assertEqual(x['scientific_GO'],'NOT_ASSESSED')
        (self.p/'boundary/train/checkpoint-750/scaler.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):C.joint_training(self.p,bind)
    def test_partial_step_or_unmatched_initialization_cannot_complete(self):
        bind={a:{'arm':a} for a in C.ARMS}
        fixture_arm(self.p,'bos_only',bind['bos_only']);fixture_arm(self.p,'boundary',bind['boundary'],749)
        with self.assertRaises(ValueError):C.joint_training(self.p,bind)
    def test_checkpoint_1250_not_accepted(self):
        with self.assertRaises(ValueError):C.verify_checkpoint(self.p,{},1250)
    def test_output_path_traversal_rejected(self):
        with self.assertRaises(ValueError):C.private_path(self.p,'../bos_only')
    def test_barrier_checks_both_run_identifiers(self):
        for a in C.ARMS:C.write(self.p/'barriers/cache'/f'{a}.json',dict(run_id='wrong'))
        ctl.pulse(self.p/'LEASE.json',dict(run_id='actual',time=time.time(),deadline=time.time()+10))
        with self.assertRaises(FileExistsError):ctl.barrier(self.p,'cache','bos_only','actual',lambda:None)
    def test_native_modules_do_not_have_execution_CLI(self):
        for n in ['led_large_boundary_native_runner750_v2.py','led_large_boundary_native_helper750_v2.py']:
            tree=ast.parse((R/'scripts'/n).read_text())
            self.assertFalse(any(isinstance(x,ast.If) for x in tree.body))
    def test_training_coordinator_never_chains_decode(self):
        from run_led_large_boundary_dual_v2 import coordinate
        import inspect
        s=inspect.getsource(coordinate)
        self.assertIn("if args.phase=='decode':",s)
        self.assertIn('PAIR_TRAINING_COMPLETE.json',s)
        self.assertNotIn('winner',s)

class TensorCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
    def test_actual_CPU_AdamW_keeps_1250_schedule_through_750(self):
        import led_large_boundary_native_helper750_v2 as h
        torch=self.torch;model=torch.nn.Linear(1,1,bias=False)
        opt,sch=h.optimizer_scheduler(model,{'optimization':dict(weight_decay=.01,learning_rate=2e-4,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8)})
        for step in range(1,751):
            model.weight.grad=torch.ones_like(model.weight);opt.step();sch.step()
            if step in [1,63,250,500,750]:C.scheduler_accounting(sch,step)
        self.assertEqual({int(s['step']) for s in opt.state.values()},{750});self.assertGreater(sch.get_last_lr()[0],0)
    def test_recovery_exact_750_and_microbatch_limit(self):
        import led_large_boundary_amp750_v2 as amp
        with tempfile.TemporaryDirectory() as td:
            r=amp.Recovery(td,750,lambda:None,C.write,C.append)
            for _ in range(750):
                for j in range(8):r.microbatch()
                r.pending=True;r.commit(NS(get_scale=lambda:8192))
            self.assertEqual(r.successful_updates,750);self.assertEqual(r.microbatches,6000)
            r.pending=True
            with self.assertRaises(ValueError):r.commit(NS(get_scale=lambda:8192))
            r.microbatches=6064
            with self.assertRaises(ValueError):r.microbatch()
    def test_recovery_rejects_unreviewed_budget(self):
        import led_large_boundary_amp750_v2 as amp
        with self.assertRaises(ValueError):amp.Recovery('.',1250,None,None,None)
    def test_native_callback_rejects_1250_and_step751(self):
        import led_large_boundary_native_helper750_v2 as h
        with tempfile.TemporaryDirectory() as td,patch.object(h,'check_deadline',lambda a:None):
            rec=h._core_callback_class()(Path(td),{},NS(check=lambda:None))
            rec.trainer=NS(accelerator=NS(scaler=NS(is_enabled=lambda:True)))
            rec.on_train_begin(None,NS(max_steps=750,global_step=0),None)
            with self.assertRaises(ValueError):rec.on_train_begin(None,NS(max_steps=1250,global_step=0),None)
            with self.assertRaises(ValueError):rec.on_step_end(None,NS(global_step=751),None)
    def test_AMP_floor_and_overflow_guard_are_not_relaxed(self):
        import led_large_boundary_amp750_v2 as amp,inspect
        self.assertEqual(amp.POLICY,C.AMP)
        s=inspect.getsource(amp.Recovery.finish)
        self.assertIn("scale*.5<POLICY['minimum_scale']",s);self.assertIn("self.overflow_skips>=POLICY['max_overflow_skips']",s)
    def amp_case(self,td,scale=65536,skips=0,mutate=False):
        import led_large_boundary_amp750_v2 as amp
        torch=self.torch
        model=torch.nn.Module();model.params=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(456)])
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for p in model.parameters():p.grad=torch.ones_like(p)
        next(model.parameters()).grad.fill_(float('inf'))
        scaler=NS(value=scale,get_scale=lambda:scaler.value)
        r=amp.Recovery(td,750,lambda:None,C.write,C.append);r.overflow_skips=skips
        with patch.object(torch.cuda,'is_available',return_value=False):r.begin()
        for _ in range(8):r.microbatch()
        before=[p.detach().clone() for p in model.parameters()];replayed=[]
        def skip():
            scaler.value*=.5
            if mutate:
                with torch.no_grad():next(model.parameters()).add_(1)
        def repeat():
            replayed.append(torch.rand(1).item())
            for _ in range(8):r.microbatch()
            for p in model.parameters():p.grad=torch.ones_like(p)
        r.finish(model,opt,scaler,lambda:None,skip,repeat)
        return r,scaler,before,model,replayed
    def test_CPU_overflow_backoff_replay_no_optimizer_update(self):
        with tempfile.TemporaryDirectory() as td:
            r,s,before,model,replayed=self.amp_case(td)
            self.assertEqual(s.value,32768);self.assertEqual(r.overflow_skips,1);self.assertEqual(r.microbatches,16)
            self.assertEqual(r.successful_updates,0);self.assertTrue(r.pending)
            self.assertTrue(all(self.torch.equal(p,b) for p,b in zip(model.parameters(),before)));self.assertEqual(len(replayed),1)
    def test_AMP_floor_stops_before_skip(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'Bounded AMP'):self.amp_case(td,scale=256)
    def test_ninth_overflow_stops(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'Bounded AMP'):self.amp_case(td,skips=8)
    def test_skip_parameter_mutation_detected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'changed adapter'):self.amp_case(td,mutate=True)
    def test_bos_free_shift_teacher_forcing_CPU(self):
        from led_large_decoder_label_view_v1 import remove_initial_bos
        from transformers.models.led.modeling_led import shift_tokens_right
        torch=self.torch;labels=torch.tensor([remove_initial_bos([0,10,20,2])+[-100]])
        decoder=shift_tokens_right(labels,1,2)
        self.assertEqual(decoder.tolist(),[[2,10,20,2]]);self.assertEqual(labels.tolist(),[[10,20,2,-100]])
    def test_no_CUDA_initialization(self):self.assertFalse(self.torch.cuda.is_initialized())

class JointGateTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)
        self.bind={a:{'arm':a} for a in C.ARMS};self.approval=dict(training_devices=dict(bos_only=U,boundary=V))
        for arm in C.ARMS:
            p=self.p/arm
            C.write(p/'feasibility/REPORT.json',dict(status='PASS',binding=self.bind[arm],optimizer_updates=3,generation_calls=2,state_restore_exact=True,adapter_restore_exact=True,frozen_backbone_unchanged=True,headroom_fraction=.8,overflow_skips=3,microbatches=48,dynamic_overflow_policy=C.AMP,approval_sha256='a',hardware_profile={'driver':'same'},GPU_total_bytes=24*1024**3))
            C.write(p/'RESOURCE_GATE.json',dict(status='PASS',binding=self.bind[arm],headroom_fraction=.7,cache_tokens=768,full_cache_batch=4,source_shape=[1,5120],optimizer_updates=0,new_generate_calls=0,global_count=1 if arm=='bos_only' else 36))
            C.write(p/'GATE_COMPLETE.json',dict(device_UUID=self.approval['training_devices'][arm],binding=self.bind[arm],approval_sha256='a'))
            C.write(p/'gates_WORKER_COMPLETE.json',dict(status='PASS'))
            C.write(p/'CACHE_READY.json',dict(UTC_seconds=1));C.write(p/'CACHE_ENDED.json',dict(UTC_seconds=4))
            C.write(p/'feasibility/INITIAL_MODEL_BINDINGS.json',dict(initialization={'fixture':'same'}))
    def test_complete_gate_pair(self):
        from run_led_large_boundary_dual_v2 import gate_pair
        r=gate_pair(self.p,self.bind,self.approval);self.assertFalse(r['training_automatically_started'])
    def test_cache_overlap_required(self):
        from run_led_large_boundary_dual_v2 import gate_pair
        (self.p/'boundary/CACHE_READY.json').write_text(json.dumps({'UTC_seconds':5}))
        with self.assertRaises(ValueError):gate_pair(self.p,self.bind,self.approval)
    def test_cache_geometry_and_optimizer_invariant(self):
        from run_led_large_boundary_dual_v2 import gate_pair
        p=self.p/'boundary/RESOURCE_GATE.json';original=C.read(p)
        for k,v in [('source_shape',[1,4096]),('optimizer_updates',1),('global_count',1),('headroom_fraction',.14)]:
            p.write_text(json.dumps(dict(original,**{k:v})))
            with self.assertRaises(ValueError):gate_pair(self.p,self.bind,self.approval)
    def test_unmatched_initialization_not_comparable(self):
        from run_led_large_boundary_dual_v2 import gate_pair
        (self.p/'boundary/feasibility/INITIAL_MODEL_BINDINGS.json').write_text(json.dumps({'initialization':{'fixture':'different'}}))
        with self.assertRaises(ValueError):gate_pair(self.p,self.bind,self.approval)

class NativePrefixIntegration(unittest.TestCase):
    def run_native(self,steps,sanity='pass'):
        """Real modified native function, CPU AdamW/scheduler, synthetic Trainer adapter.

        CUDA, actual LED model, AMP scaler and dataset loaders are mocked. This is
        completion/schedule integration evidence, NOT real LED/FP16 equivalence.
        """
        import torch,transformers,types
        from stage_i_paired_pilot import PairedEpochSampler
        original_schedule=transformers.get_linear_schedule_with_warmup
        # Fully initialize CPU optimizer internals before replacing a lazy module.
        warmup_parameter=torch.nn.Parameter(torch.ones(1));warmup_opt=torch.optim.AdamW([warmup_parameter]);warmup_parameter.grad=torch.zeros(1);warmup_opt.step()
        import led_large_boundary_native_runner750_v2 as b
        import led_large_boundary_native_helper750_v2 as h
        import led_large_decoder_early_sanity_v2 as early
        events=[]
        torch_model=torch.nn.Linear(1,1,bias=False)
        class Memory:
            samples=[]
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def check(self):pass
            def summary(self):return dict(headroom_fraction=1.)
        class SyntheticCoreCallback:
            """Mock only the core recording work, retaining the production sanity wrapper."""
            def __init__(self,folder,approval,memory):
                self.folder=folder;self.approval=approval;self.memory=memory;self.trainer=None
            def on_save(self,args,state,control,**kwargs):
                events.append(('save_started',state.global_step))
        def synthetic_early_sanity(model,folder,approval,config,memory,root,binding,write,deadline,*,trainer):
            # Called only by production Record.on_save at the simulated step-250 save.
            self.assertEqual(trainer.state.global_step,250)
            self.assertEqual(events[-1],('save_started',250))
            self.assertFalse((folder/'early_sanity/REPORT.json').exists())
            events.append(('sanity_called',250))
            if sanity!='missing':
                write(folder/'early_sanity/REPORT.json',dict(
                    SYNTHETIC_CPU_FIXTURE_ONLY=True,global_step=250,
                    status='PASS_TECHNICAL_ONLY' if sanity=='pass' else 'FAIL_SYNTHETIC',
                    binding=binding,generation_calls=4))
                events.append(('sanity_receipt_written',250))
        class Arguments(NS):
            def to_dict(self):return self.__dict__
        class Trainer:
            def __init__(self,model,args,optimizers,**kw):
                self.model=model;self.args=args;self.optimizer,self.lr_scheduler=optimizers
                self.accelerator=NS(scaler=None);self.initial_amp_record={'SYNTHETIC_CPU_ONLY':True}
                self.state=NS(global_step=0,epoch=0)
                self.callbacks=kw['callbacks']
            def train(self,**kw):
                for k in range(1,steps+1):
                    self.model.weight.grad=torch.ones_like(self.model.weight);self.optimizer.step();self.lr_scheduler.step()
                    self.state.global_step=k;self.state.epoch=k/250
                    if k in C.CHECKPOINTS:
                        for callback in self.callbacks:
                            callback.on_save(self.args,self.state,NS())
                        events.append(('save_accepted',k))
                C.scheduler_accounting(self.lr_scheduler,steps)
                self.amp_recovery=NS(state=lambda s:dict(successful_updates=steps,policy=C.AMP))
                return NS(metrics={'SYNTHETIC_CPU_ONLY':True})
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            C.write(out/'SYNTHETIC_CONFIG.json',dict(SYNTHETIC_CPU_FIXTURE_ONLY=True))
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.object(h,'_core_callback_class',lambda:SyntheticCoreCallback))
                stack.enter_context(patch.object(h,'binding',lambda:{'fixture':True}))
                stack.enter_context(patch.object(h,'CONFIG',out/'SYNTHETIC_CONFIG.json'))
                stack.enter_context(patch.object(early,'run_early_sanity',synthetic_early_sanity))
                for name,value in [('check_required_gpu',lambda t:None),('require_gate',lambda o,a:{}),('check_gpu_same',lambda g,t:None),
                    ('data',lambda c:([{'synthetic':True}],[])),('load_model',lambda c:(torch_model,{}, {'initialization':{'fixture':True}})),
                    ('read',lambda p:{'indices':[list(range(2000))]*5}),('callback_class',h.callback_class),
                    ('MemoryMeter',Memory),('install_trainer',lambda *a:None),('verify_backbone',lambda *a:None),
                    ('check_deadline',lambda *a:None),('binding',lambda:{'fixture':True}),
                    ('checkpoint_verified',lambda *a:{}),('training_arguments',lambda c,f:dict(max_steps=750,num_train_epochs=3))]:stack.enter_context(patch.object(b,name,value))
                fake_transformers=types.ModuleType('transformers')
                fake_transformers.__spec__=transformers.__spec__
                fake_transformers.Seq2SeqTrainer=Trainer;fake_transformers.Seq2SeqTrainingArguments=lambda **k:Arguments(**k)
                fake_transformers.set_seed=lambda s:None;fake_transformers.get_linear_schedule_with_warmup=original_schedule
                stack.enter_context(patch.dict(sys.modules,{'transformers':fake_transformers}))
                config={'optimization':dict(weight_decay=.01,learning_rate=2e-4,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8)}
                if sanity!='pass':
                    error=FileNotFoundError if sanity=='missing' else ValueError
                    with self.assertRaises(error):b.train(config,{},out)
                    self.assertEqual([k for event,k in events if event=='save_started'],[250,500])
                    self.assertEqual([k for event,k in events if event=='save_accepted'],[250])
                    self.assertFalse((out/'train/COMPLETION.json').exists())
                    self.assertTrue((out/'train/FAILURE_STOP_NO_RETRY.json').is_file())
                elif steps==750:
                    b.train(config,{},out)
                    done=C.read(out/'train/COMPLETION.json');self.assertEqual(done['steps'],750);self.assertEqual(done['epochs'],3);self.assertEqual(done['checkpoint_steps'],[250,500,750])
                    receipt=out/'train/early_sanity/REPORT.json'
                    self.assertEqual(C.read(receipt)['status'],'PASS_TECHNICAL_ONLY')
                    self.assertEqual(C.read(receipt)['binding'],{'fixture':True})
                    self.assertEqual(C.read(receipt)['generation_calls'],4)
                    self.assertEqual(done['early_sanity_report_sha256'],C.sha(receipt))
                    self.assertEqual(events,[('save_started',250),('sanity_called',250),('sanity_receipt_written',250),('save_accepted',250),('save_started',500),('save_accepted',500),('save_started',750),('save_accepted',750)])
                else:
                    with self.assertRaisesRegex(ValueError,'Incomplete planned three-epoch'):b.train(config,{},out)
                    self.assertFalse((out/'train/COMPLETION.json').exists());self.assertTrue((out/'train/FAILURE_STOP_NO_RETRY.json').is_file())
                    self.assertEqual([k for event,k in events if event=='save_accepted'],[250,500])
    def test_native_750_completion(self):self.run_native(750)
    def test_native_749_not_mislabeled_complete(self):self.run_native(749)
    def test_native_missing_early_sanity_blocks_later_checkpoint(self):self.run_native(750,'missing')
    def test_native_failed_early_sanity_blocks_later_checkpoint(self):self.run_native(750,'failed')

if __name__=='__main__':unittest.main(verbosity=2)
