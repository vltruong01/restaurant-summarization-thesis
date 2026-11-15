"""Portable execution adapter: frozen D026 core, separate Linux runtime, original science untouched."""
from pathlib import Path
from collections import Counter
import argparse,json,os,platform,statistics,subprocess,sys,time
import yaml
import blip2_caption_pilot as legacy
from stage_f_d033_protocol import freeze_method,STATE
from generate_stage_f_validation_captions import generate_locked

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'outputs/stage_f/production_caption_construction_d033_v1'
OUT=ROOT/'outputs/stage_f/production_caption_construction_d033_cloud_v1'
GEN=OUT/'generation_remaining'
CONFIG=ROOT/'configs/stage_f_cloud_d033.yaml'
legacy.ORIGIN='agent'


def immutable(path,value):
    if path.exists():assert legacy.read(path)==value,f'Incompatible existing cloud artifact: {path}'
    else:legacy.save(path,value)


def configuration():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    old=yaml.safe_load((ROOT/'configs/stage_f_production_d033.yaml').read_text(encoding='utf-8'))
    expected=dict(old);expected['output_directory']=GEN.relative_to(ROOT).as_posix();expected['isolated_environment']='.venv-stage-f-cloud'
    assert cfg==expected,'Only output namespace and isolated environment may differ'
    freeze_method();return cfg


def prepare():
    cfg=configuration();assert legacy.read(LOCAL/'user_stop_report.json')['status'].startswith('STOPPED_AT_EXPLICIT_USER_REQUEST')
    plan=legacy.read(LOCAL/'complete_photo_plan.json');original=legacy.read(LOCAL/'reused_caption_sources.json')
    reused={pid:dict(item) for pid,item in original.items()}
    for path in sorted((LOCAL/'generation_train_test/caption_checkpoints').glob('*.json')):
        r=legacy.read(path);pid=r['photo_id'];assert pid not in reused
        assert r['generation_settings']==cfg['generation'] and r['model_id']==cfg['model_id'] and r['resolved_commit']==cfg['requested_revision']
        assert r['basic_quality_policy_sha256']==legacy.sha(ROOT/cfg['basic_quality_policy'])
        reused[pid]={'checkpoint_path':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':legacy.sha(path),
                     'split':r['split'],'basic_eligible':r['basic_eligible'],'generation_settings_exact':True}
    indexed={r['photo_id']:r for r in plan};assert len(indexed)==len(plan)==30027 and len(reused)==2709
    for pid,item in reused.items():
        path=ROOT/item['checkpoint_path'];assert legacy.sha(path)==item['checkpoint_sha256'];r=legacy.read(path)
        assert r['photo_id']==pid and r['business_id']==indexed[pid]['business_id'] and r['split']==indexed[pid]['split']
        assert r['raw_sha256']==indexed[pid]['raw_sha256']
        assert legacy.sha(ROOT/r['image_path'])==r['raw_sha256']
    oldplan={r['photo_id']:r for r in legacy.read(LOCAL/'generation_train_test/pilot_photo_plan.json')}
    pending=[dict(oldplan[pid]) for pid in indexed if pid not in reused]
    pending.sort(key=lambda r:({'train':0,'test':1}[r['split']],r['business_id'],r['photo_id']))
    assert len(pending)==27318 and not (set(reused)&{r['photo_id'] for r in pending})
    assert set(reused)|{r['photo_id'] for r in pending}==set(indexed)
    assert all(r['split']=='train' for r in pending[:100])
    GEN.mkdir(parents=True,exist_ok=True)
    immutable(OUT/'complete_photo_plan.json',plan);immutable(OUT/'reused_caption_sources.json',reused)
    immutable(GEN/'pilot_photo_plan.json',pending)
    immutable(GEN/'model_revision.json',legacy.read(LOCAL/'generation_train_test/model_revision.json'))
    source_names=['configs/stage_f_cloud_d033.yaml','configs/stage_f_production_d033.yaml',
                  'scripts/stage_f_cloud_d033.py','scripts/generate_stage_f_validation_captions.py',
                  'scripts/blip2_caption_pilot.py','scripts/caption_basic_quality.py','configs/basic_caption_quality.yaml',
                  'outputs/stage_d/photo_integrity_duplicate_audit_v1/unfiltered_readable_photo_pool.csv',
                  (GEN/'pilot_photo_plan.json').relative_to(ROOT).as_posix(),
                  (OUT/'reused_caption_sources.json').relative_to(ROOT).as_posix()]
    identity={'config_sha256':legacy.sha(CONFIG),'source_sha256':{n:legacy.sha(ROOT/n) for n in source_names},
              'plan_sha256':legacy.canon_hash(pending),'basic_policy_sha256':legacy.sha(ROOT/cfg['basic_quality_policy']),
              'parent_local_plan_identity':legacy.read(LOCAL/'generation_train_test/pilot_identity.json')['identity_sha256'],
              'execution_only_migration':True,'reused_caption_count':len(reused)}
    identity['identity_sha256']=legacy.canon_hash(identity);immutable(GEN/'pilot_identity.json',identity)
    report={'status':'CLOUD_PACKAGE_PREPARED_NOT_GPU_VALIDATED','cohort_businesses':2300,'full_photo_pool':len(plan),
            'reused_captions':len(reused),'reused_by_split':dict(Counter(i['split'] for i in reused.values())),
            'pending_photos':len(pending),'pending_by_split':dict(Counter(r['split'] for r in pending)),
            'all_eligible_photos_before_K_selection':True,'same_scientific_generator':True,'same_method':True,
            'cloud_namespace':OUT.relative_to(ROOT).as_posix(),'parent_local_namespace_unchanged':True,
            'K':5,'teacher_calls':False,'LED_training':False,'cloud_runtime_validated':False}
    immutable(OUT/'preparation_report.json',report);print(json.dumps(report,indent=2));return report


def check_execution_seal():
    for relative,digest in legacy.read(OUT/'execution_seal.json').items():
        assert legacy.sha(ROOT/relative)==digest,f'Cloud execution source changed: {relative}'
    cfg=configuration();identity=legacy.read(GEN/'pilot_identity.json');legacy.check_sources(identity)
    assert identity['plan_sha256']==legacy.canon_hash(legacy.read(GEN/'pilot_photo_plan.json'))
    return cfg


def verify_blip_weights():
    from huggingface_hub import snapshot_download
    expected=legacy.read(OUT/'blip_weight_files.json')
    snapshot=Path(snapshot_download(expected['model_id'],revision=expected['revision'],cache_dir=ROOT/'assets/model_cache',
                                   local_files_only=True,allow_patterns=list(expected['sha256'])))
    for name,digest in expected['sha256'].items():assert legacy.sha(snapshot/name)==digest,f'Wrong pinned BLIP file: {name}'
    immutable(OUT/'weight_verification.json',{'status':'PASS_PINNED_ORIGINAL_BLIP_BYTES',
              'revision':expected['revision'],'expected_manifest_sha256':legacy.sha(OUT/'blip_weight_files.json')})


def download():
    check_execution_seal()
    from huggingface_hub import snapshot_download
    expected=legacy.read(OUT/'blip_weight_files.json')
    snapshot_download(expected['model_id'],revision=expected['revision'],cache_dir=ROOT/'assets/model_cache',
                      allow_patterns=list(expected['sha256']),max_workers=2)
    verify_blip_weights();print('Pinned BLIP cache ready; no inference performed')


def preflight():
    cfg=check_execution_seal();assert platform.system()=='Linux','Cloud adapter requires Linux; do not resume the stopped local job'
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory/2**30>=23,'One24GB CUDA GPU required'
    expected=legacy.read(LOCAL/'generation_train_test/environment_report.json')
    from importlib.metadata import version
    for name,value in expected['versions'].items():assert version(name)==value,f'Pinned library version mismatch: {name}'
    if (GEN/'environment_report.json').exists():
        r=legacy.read(GEN/'environment_report.json')
        assert r['status']=='preflight_passed_not_model_load_passed' and r['gpu_name']==torch.cuda.get_device_name(0)
        assert r['versions']==expected['versions'];print('Original cloud preflight/device-map profile retained')
    else:legacy.preflight(cfg,GEN)


def smoke():
    cfg=check_execution_seal();preflight()
    from filelock import FileLock
    with FileLock(GEN/'caption_model_execution.lock',timeout=0):
        plan=legacy.read(GEN/'pilot_photo_plan.json');sample=plan[:100]
        assert all(r['split']=='train' for r in sample)
        if (OUT/'benchmark_report.json').exists():
            prior=legacy.read(OUT/'benchmark_report.json')
            assert prior['sample_photo_ids']==[r['photo_id'] for r in sample]
            for pid,digest in prior['caption_checkpoint_sha256'].items():
                assert legacy.sha(GEN/'caption_checkpoints'/f'{pid}.json')==digest
            print(json.dumps(prior,indent=2));return
        completed=sum((GEN/'caption_checkpoints'/f"{r['photo_id']}.json").exists() for r in sample)
        if completed<100:generate_locked(cfg,GEN,max_new_photos=100-completed)
        model=legacy.read(GEN/'model_load_report.json')
        assert set(model['device_map'].values())<={'0','cuda:0'},'Cloud smoke still offloads toCPU/disk; report feasibility blocker'
        records=[legacy.read(GEN/'caption_checkpoints'/f"{r['photo_id']}.json") for r in sample]
        seconds=[sum(a['seconds'] for a in r['attempts']) for r in records]
        benchmark={'status':'PASS_CLOUD_TRAIN_ONLY_SMOKE_NOT_PRODUCTION_COMPLETION','sample_photo_ids':[r['photo_id'] for r in sample],
                   'caption_checkpoint_sha256':{r['photo_id']:legacy.sha(GEN/'caption_checkpoints'/f"{r['photo_id']}.json") for r in sample},
                   'sample_count':100,'mean_seconds_per_photo':statistics.mean(seconds),'median_seconds_per_photo':statistics.median(seconds),
                   'model_load_seconds':model['model_load_seconds'],'device_map_sha256':model['device_map_sha256'],
                   'runtime_sha256':model['runtime_sha256'],'basic_pass':sum(r['basic_eligible'] for r in records),
                   'retries':sum(len(r['attempts'])>1 for r in records),'peak_GPU_allocated_GiB':max(a['peak_GPU_allocated_GiB'] for r in records for a in r['attempts']),
                   'estimate_remaining_caption_hours_at_smoke_rate':(len(plan)-100)*statistics.mean(seconds)/3600,
                   'no_test_tuning':True,'no_settings_changed':True,'K':5,'teacher_calls':False,'LED_training':False}
        immutable(OUT/'benchmark_report.json',benchmark);print(json.dumps(benchmark,indent=2))


def process_step(action):
    check_execution_seal()
    if action in ['score','construct']:
        import construct_stage_f_production_d033 as core
        core.OUT=OUT;core.GEN=GEN
        # Original implementation, original thresholds/tokenizer/validation namespaces; only output roots differ.
        return core.score_production() if action=='score' else core.construct()
    if action=='reviews':
        import prepare_stage_f_production_reviews_d033 as core
        core.OUT=OUT/'review_inputs';return core.run(workers=2)


def progress(phase,state='running',**extra):
    n=len(list((GEN/'caption_checkpoints').glob('*.json')))
    legacy.save(OUT/'worker_progress.json',{'status':state,'phase':phase,'worker_pid':os.getpid(),
                'cloud_new_caption_count':n,'cloud_plan':27318,'compatible_reused_captions':2709,
                'total_generated_captions':2709+n,'total_photo_pool':30027,'remaining_photos':27318-n,
                'K':5,'teacher_calls':False,'LED_training':False,**extra})


def production():
    from filelock import FileLock
    with FileLock(OUT/'managed_worker.lock',timeout=0):
        check_execution_seal();assert legacy.read(OUT/'benchmark_report.json')['status'].startswith('PASS_CLOUD_TRAIN_ONLY_SMOKE')
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',RAYON_NUM_THREADS='4',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES='0')
        for phase in ['generate','score','reviews','construct']:
            with (OUT/f'{phase}.stdout.log').open('a',encoding='utf-8') as stdout,(OUT/f'{phase}.stderr.log').open('a',encoding='utf-8') as stderr:
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),phase],cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=(os.name!='nt'))
                try:
                    while p.poll() is None:progress(phase,child_pid=p.pid);time.sleep(15)
                    if p.returncode:raise RuntimeError(f'{phase} failed exit{p.returncode}; checkpoints retained; no method changes')
                except BaseException:
                    if p.poll() is None:
                        if os.name!='nt':
                            import signal
                            os.killpg(p.pid,signal.SIGTERM)
                        else:p.terminate()
                        p.wait(timeout=30)
                    progress(phase,'STOP_ON_ERROR_OR_INTERRUPT_NO_AUTOMATIC_METHOD_CHANGE');raise
        r=legacy.read(OUT/'completion_report.json');assert r['status']=='STAGE_F_PRODUCTION_COMPLETE_VALIDATED_STOP_BEFORE_STAGE_G'
        progress('complete',r['status']);print(r['status'])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','download','preflight','smoke','generate','score','reviews','construct','production'])
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='download':download()
    elif a.action=='preflight':preflight()
    elif a.action=='smoke':smoke()
    elif a.action=='production':production()
    elif a.action=='generate':
        cfg=check_execution_seal();preflight()
        assert legacy.read(OUT/'benchmark_report.json')['status'].startswith('PASS_CLOUD_TRAIN_ONLY_SMOKE')
        from filelock import FileLock
        with FileLock(GEN/'caption_model_execution.lock',timeout=0):generate_locked(cfg,GEN)
    else:process_step(a.action)


if __name__=='__main__':main()
