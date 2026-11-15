"""Versioned host entry point. CPU verification now; every GPU phase needs new approval.

Training NEVER chains decoding. No SSH, provisioning, lifecycle or retry command.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,os,sys,time,uuid as ids
import led_large_boundary_dual_contract_v2 as C
from led_large_boundary_dual_control_v2 import Locks,run_workers
from led_large_boundary_dual_safety_v2 import provider,DeviceSafety,host_snapshot,check_host
from led_large_boundary_dual_runtime_v2 import arm_binding,verify_package,run_arm

def gate_pair(suite,bindings,a):
    suite=Path(suite);reports={};initial=[];profiles=[];totals=[];intervals=[]
    for arm in C.ARMS:
        out=suite/arm;r=C.read(out/'feasibility/REPORT.json');resource=C.read(out/'RESOURCE_GATE.json');done=C.read(out/'GATE_COMPLETE.json')
        C.require(r['status']=='PASS' and r['binding']==bindings[arm] and r['optimizer_updates']==3 and r['generation_calls']==2,'Incomplete native gate')
        C.require(r['state_restore_exact'] and r['adapter_restore_exact'] and r['frozen_backbone_unchanged'] and r['headroom_fraction']>=.15,'Invalid restoration/resource gate')
        C.require(r['overflow_skips']<=8 and r['microbatches']<=88 and r['dynamic_overflow_policy']==C.AMP,'Gate AMP budget differs')
        C.require(resource['status']=='PASS' and resource['binding']==bindings[arm] and resource['headroom_fraction']>=.15 and resource['cache_tokens']==768 and resource['full_cache_batch']==4 and resource['source_shape']==[1,5120] and resource['new_generate_calls']==0 and resource['optimizer_updates']==0 and resource['global_count']==(1 if arm=='bos_only' else 36),'Invalid full cache stress')
        C.require(done['device_UUID']==a['training_devices'][arm] and done['binding']==bindings[arm] and done['approval_sha256']==r['approval_sha256'],'Wrong assigned device/gate receipt')
        C.require(C.read(out/'gates_WORKER_COMPLETE.json')['status']=='PASS','Missing gate postflight')
        start=C.read(out/'CACHE_READY.json')['UTC_seconds'];end=C.read(out/'CACHE_ENDED.json')['UTC_seconds'];intervals.append((start,end))
        initial.append(C.read(out/'feasibility/INITIAL_MODEL_BINDINGS.json')['initialization']);profiles.append(r['hardware_profile']);totals.append(r['GPU_total_bytes'])
        reports[arm]=dict(gate_report_sha256=C.sha(out/'feasibility/REPORT.json'),resource_sha256=C.sha(out/'RESOURCE_GATE.json'),device_UUID=done['device_UUID'])
    C.require(initial[0]==initial[1] and profiles[0]==profiles[1] and totals[0]==totals[1],'Unmatched initialization/backend/effective VRAM')
    C.require(max(v[0] for v in intervals)<min(v[1] for v in intervals),'Cache stresses did not overlap')
    return dict(status='PAIR_GATES_COMPLETE_STOP_FOR_REVIEW',arms=reports,initialization_identical=True,cache_overlap=True,training_automatically_started=False)

def coordinate(args):
    package,parent,mask_root,suite=map(lambda p:p.resolve(),(args.package,args.parent,args.mask_root,args.suite))
    a=C.authorize(args.approval,args.phase,package)  # FIRST: no device/data work before approval.
    C.require(os.name=='posix','Production execution requires dedicated Linux host')
    verify_package(package)
    C.require(suite.parent==parent/'runs' and suite.name.startswith(C.EXPERIMENT),'Unsafe suite path')
    if args.phase=='gates':
        suite.mkdir(parents=True,exist_ok=False)
        start=time.time()
        C.write(suite/'SUITE_START.json',dict(start=start,pair_deadline=start+28800,instance_id=a['instance_id'],training_devices=a['training_devices'],common_decoding_device=a['common_decoding_device'],bindings={x:arm_binding(package,x) for x in C.ARMS}))
        args._phase_started=True
        with (suite/'GATE_AUTHORIZATION.json').open('xb') as f:f.write(args.approval.read_bytes())
    start=C.read(suite/'SUITE_START.json')
    C.require(not (suite/'PAIR_FAILURE.json').exists(),'Failed pair cannot retry/continue')
    C.require(start['instance_id']==a['instance_id'] and start['training_devices']==a['training_devices'] and start['common_decoding_device']==a['common_decoding_device'],'Pair/phase allocation changed')
    bindings={x:arm_binding(package,x) for x in C.ARMS};C.require(bindings==start['bindings'],'Scientific/code bindings changed between phases')
    deadline=min(start['pair_deadline'],datetime.fromisoformat(a['execution_deadline_utc']).timestamp(),
        datetime.fromisoformat(C.read(suite/'GATE_AUTHORIZATION.json')['execution_deadline_utc']).timestamp(),
        time.time()+(7200 if args.phase=='gates' else 21600))
    C.require(time.time()<deadline,'No phase budget remains')
    if args.phase=='train':C.require((suite/'PAIR_GATES_COMPLETE.json').is_file(),'Both gates required; no automatic training')
    if args.phase=='decode':
        C.require((suite/'PAIR_TRAINING_COMPLETE.json').is_file(),'Both fresh arms required')
        C.joint_training(suite,bindings)
    C.write(suite/(args.phase+'_PHASE_REQUEST.json'),dict(approval_sha256=C.sha(args.approval),requested_explicitly=True,deadline=deadline))
    args._phase_started=True
    lockpaths=[Path('/workspace/.led_large_experiment.lock'),suite/'COORDINATOR.lock']+[Path('/workspace')/('.led_large_'+u+'.lock') for u in a['training_devices'].values()]
    with Locks(lockpaths):
        C.write(suite/(args.phase+'_ALLOCATION.json'),provider(a['instance_id']))
        C.write(suite/(args.phase+'_HOST_INITIAL.json'),check_host(host_snapshot(suite),initial=True))
        for arm,device in a['training_devices'].items():
            folder=suite/(args.phase+'_idle_'+arm);folder.mkdir()
            DeviceSafety(device,lambda n,v,folder=folder:C.write(folder/n,v)).check(loaded=False)
        def command(arm,run_id):
            device=a['common_decoding_device'] if args.phase=='decode' else a['training_devices'][arm]
            env=C.clean_worker_env(os.environ,device,arm)
            env['LED_BOUNDARY_COORDINATOR_PID']=str(os.getpid())
            env['PYTHONPATH']=str(package/'scripts')+os.pathsep+str(mask_root/'scripts')+os.pathsep+str(parent/'scripts')
            return dict(command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--phase',args.phase,
                '--arm',arm,'--package',str(package),'--parent',str(parent),'--mask-root',str(mask_root),
                '--suite',str(suite),'--approval',str(args.approval.resolve()),'--run-id',run_id],cwd=str(package),env=env)
        def hostcheck():
            snap=check_host(host_snapshot(suite));C.append(suite/'HOST_SAMPLES.jsonl',snap)
        if args.phase=='decode':
            for arm in C.ARMS:
                rid='decode_'+arm+'_'+ids.uuid4().hex
                run_workers(suite,{arm:command(arm,rid)},deadline,rid,hostcheck)
            for arm in C.ARMS:
                r=C.read(suite/arm/'evaluate/COMPLETION.json')
                C.require(r['generation_calls']==24 and r['binding']==bindings[arm],'Incomplete decoded pair')
            C.write(suite/'PAIR_DECODING_COMPLETE.json',dict(status='48_OUTPUTS_STOP_FOR_INDEPENDENT_REVIEW',generated=48,scientific_GO='NOT_ASSESSED',adoption_GO='NOT_ASSESSED',automatic_model_selection=False))
        else:
            rid=args.phase+'_'+ids.uuid4().hex
            run_workers(suite,{x:command(x,rid) for x in C.ARMS},deadline,rid,hostcheck)
            if args.phase=='gates':C.write(suite/'PAIR_GATES_COMPLETE.json',gate_pair(suite,bindings,a))
            else:C.write(suite/'PAIR_TRAINING_COMPLETE.json',C.joint_training(suite,bindings))
    print('STOP_FOR_REVIEW',args.phase,'No automatic next phase or model selection.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['verify-package','coordinate','worker'])
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--parent',type=Path);p.add_argument('--mask-root',type=Path);p.add_argument('--suite',type=Path)
    p.add_argument('--approval',type=Path);p.add_argument('--phase',choices=['gates','train','decode']);p.add_argument('--arm',choices=C.ARMS);p.add_argument('--run-id')
    args=p.parse_args()
    if args.mode=='verify-package':verify_package(args.package);print('PACKAGE_HASHES_PASS_NO_GPU_OR_PAYLOAD_ACCESS');return
    C.require(all(getattr(args,k) is not None for k in ('parent','mask_root','suite','approval','phase')),'Explicit phase/paths/approval required')
    try:
        if args.mode=='worker':
            C.require(args.arm is not None and args.run_id,'Coordinator worker identity missing')
            run_arm(args.package,args.parent,args.mask_root,args.suite,args.arm,args.phase,args.approval,args.run_id)
        else:coordinate(args)
    except BaseException as exc:
        if getattr(args,'_phase_started',False) and args.suite.is_dir() and not (args.suite/'PAIR_FAILURE.json').exists():
            C.write(args.suite/'PAIR_FAILURE.json',dict(status='PAIR_INCOMPLETE_STOP_NO_RETRY',error=str(exc)))
        raise
if __name__=='__main__':main()
