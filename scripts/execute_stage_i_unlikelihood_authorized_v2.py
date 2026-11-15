"""Sequential one-shot orchestration of immutable approved UL v2 runner.

No scientific overrides. Both gates must finish before either fresh training run.
On failure/expiry stop; never retry or move to later GPU stages.
"""
import platform
import subprocess
import sys
import time
import traceback
from stage_i_unlikelihood_v2 import *
from run_stage_i_unlikelihood_pilot_v2 import preflight

AUDIT=ROOT/'outputs/stage_i/unlikelihood_execution_v2'
EXPECTED={
 'config_sha256':'590c9e54013f30f86a81a7194c85ef9f66cd62035ad5bf68b290deb42b9c7ae1',
 'preflight_manifest_sha256':'e01b09cb0c8425c05efd416702751b8545f92112ef8f94a403288e3c625e93a7',
 'runner_sha256':'f2353860dc79445a8e005ef1a2a62ea9d6d888f5bd374d3b5793405282e2e5c5'}


def now():return datetime.now(timezone.utc).isoformat()


def verify():
    actual={'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
            'runner_sha256':sha(RUNNER)}
    require(actual==EXPECTED,'Researcher-approved hash binding mismatch')
    return actual


def main():
    require(sys.argv[1:]==['--execute'],'--execute required')
    approval=AUDIT/'RESEARCHER_EXECUTION_AUTHORIZATION.json'
    require(approval.exists(),'Researcher authorization record required')
    require(not (AUDIT/'ORCHESTRATION_STARTED.json').exists(),'Already started; no retry')
    write(AUDIT/'ORCHESTRATION_STARTED.json',{'UTC':now(),'wrapper_sha256':sha(Path(__file__)),'no_retry':True})
    started=time.perf_counter()
    try:
        bindings=verify();cfg,m=preflight()
        authorization=authorization_gate(approval,cfg,m,'feasibility','control_ce')
        write(AUDIT/'ACTUAL_ENVIRONMENT_PREFLIGHT.json',{'status':'PASS','UTC':now(),'bindings':bindings,
            'file_bindings_verified':len(m['files']),'runtime_pins':cfg['runtime_pins'],
            'python':sys.executable,'platform':platform.platform(),'model_operations_so_far':0,
            'hard_deadline_utc':authorization['new_execution_deadline_utc']})
        print('Exact three SHA256,482workspace bindings and runtime pins PASS.',flush=True)
        steps=[('feasibility','control_ce'),('feasibility','treatment_ul'),
               ('train','control_ce'),('train','treatment_ul'),
               ('evaluate','control_ce'),('evaluate','treatment_ul')]
        for mode,arm in steps:
            verify();cfg,m=preflight();a=authorization_gate(approval,cfg,m,mode,arm)
            if mode in ('train','evaluate'):
                for gate_arm in cfg['arms']:
                    folder=ROOT/cfg['run_root']/gate_arm/'feasibility'
                    report=read(folder/'COMPLETION.json')
                    require(report['status']=='FEASIBILITY_PASS' and report['steps']==3,'Both bounded gates must PASS')
                    require(not (folder/'FAILURE_STOP_NO_RETRY.json').exists(),'Prior gate failure')
                    require(read(folder/'FROZEN_BACKBONE_INTEGRITY.json')['unchanged'],'Gate backbone changed')
                    require(read(folder/'UL_SIGNAL_SUMMARY.json')['microbatches']==24,'Gradient probe evidence incomplete')
            label=f'{mode.upper()}_{arm.upper()}'
            cmd=[sys.executable,str(RUNNER),mode,'--arm',arm,'--execute','--approval',str(approval)]
            write(AUDIT/(label+'_INVOCATION.json'),{'UTC':now(),'command':cmd,'invocation':1,'bindings':verify()})
            remaining=(datetime.fromisoformat(a['new_execution_deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
            require(remaining>0,'Hard deadline reached before next invocation')
            print(f'Invoking {mode}/{arm} once.',flush=True);t=time.perf_counter()
            with (AUDIT/(label+'_STDOUT.log')).open('x',encoding='utf-8') as stdout,(AUDIT/(label+'_STDERR.log')).open('x',encoding='utf-8') as stderr:
                try:
                    result=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=remaining)
                except subprocess.TimeoutExpired:
                    write(AUDIT/'STOP_ON_DEADLINE.json',{'UTC':now(),'stage':label,'child_killed':True,'further_GPU_execution':False})
                    return 124
            write(AUDIT/(label+'_EXIT.json'),{'UTC':now(),'returncode':result.returncode,'seconds':time.perf_counter()-t,
                                            'retry_invocations':0,'bindings_after':verify()})
            if result.returncode:
                write(AUDIT/'STOP_ON_FAILURE_NO_RETRY.json',{'UTC':now(),'stage':label,'returncode':result.returncode,
                    'automatic_retry':False,'further_GPU_execution':False})
                print(f'STOP: {label} failed; no retry or later stage.',flush=True)
                return result.returncode
            completion=read(ROOT/cfg['run_root']/arm/mode/'COMPLETION.json')
            require(completion['config_sha256']==EXPECTED['config_sha256'] and completion['runner_sha256']==EXPECTED['runner_sha256'],
                    'Unexpected completion bindings')
            if mode=='evaluate':require(completion['generation_calls']==24,'Wrong generation count')
            else:require(completion['steps']==(3 if mode=='feasibility' else 128),'Wrong optimizer update count')
            print(f'Completed {mode}/{arm}.',flush=True)
        with (AUDIT/'OFFLINE_COMPARISON_STDOUT.log').open('x',encoding='utf-8') as stdout,(AUDIT/'OFFLINE_COMPARISON_STDERR.log').open('x',encoding='utf-8') as stderr:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/compare_stage_i_unlikelihood_v2.py')],cwd=ROOT,stdout=stdout,stderr=stderr)
        require(result.returncode==0,'Offline comparison assembly failed; no GPU retry')
        write(AUDIT/'ORCHESTRATION_COMPLETION.json',{'status':'TWO_GATES_PAIRED_TRAINING_AND48_OUTPUTS_COMPLETE_STOP_FOR_REVIEW',
            'UTC':now(),'seconds':time.perf_counter()-started,'gate_updates_total':6,'training_updates_total':256,
            'generation_outputs':48,'model_experiments_after_this_pilot':0,'retry_invocations':0})
        print('Approved paired scope complete. Stop GPU; quality review pending.',flush=True)
        return 0
    except Exception as exc:
        write(AUDIT/'ORCHESTRATION_FAILURE_STOP.json',{'UTC':now(),'error':str(exc),'traceback':traceback.format_exc(),
            'no_retry':True,'further_GPU_execution':False})
        raise


if __name__=='__main__':sys.exit(main())
