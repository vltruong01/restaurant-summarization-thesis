"""Offline only: audit all 6 frozen train/validation sets and prepare one BOS correction."""
from pathlib import Path
from datetime import datetime,timezone
import ast,difflib,hashlib,json,os,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/cloud_handoff/LED_LARGE_VAST_D059_F_V1'
PREP=ROOT/'outputs/stage_i/led_large_decoder_policy_v1'
STAGE=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_WORKSPACE'
NAME='LED_LARGE_DECODER_POLICY_V1'
CONFIG='configs/stage_i_led_large_decoder_policy_v1.json'
HELPER='scripts/stage_i_led_large_decoder_policy_v1.py'
RUNNER='scripts/run_stage_i_led_large_decoder_policy_v1.py'
REPORTER='scripts/report_stage_i_led_large_decoder_policy_v1.py'
POLICY='LED_LARGE_CONTENT_EOS_NO_INITIAL_BOS_V1'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def replace_function(source,name,replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])

def audit_labels():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    print('IMPORT_PINNED_TOKENIZER',flush=True)
    from transformers import LEDTokenizerFast
    from led_large_decoder_label_view_v1 import remove_initial_bos
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    tok=LEDTokenizerFast.from_pretrained(PARENT/'assets/tokenizers/led-base-16384',local_files_only=True)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),historical_stage_h_status=manifest['status'],
        policy=POLICY,scope='All 6 train/validation conditions locally; no held-out files',datasets={})
    records=[];views={}
    for name,info in manifest['datasets'].items():
        print('AUDIT_DATASET_START',name,flush=True)
        assert name.split('_')[0] in ('train','validation')
        npz=ROOT/info['path'];jsonl=ROOT/info['stage_g_jsonl']
        assert sha(npz)==info['sha256'] and sha(jsonl)==info['stage_g_jsonl_sha256']
        rows=[json.loads(x) for x in jsonl.read_text(encoding='utf-8').splitlines()]
        with np.load(npz,allow_pickle=False) as packed:
            # NPZ access decompresses a member on every __getitem__. Cache once
            # so exhaustive auditing does not repeatedly inflate the full corpus.
            z={key:packed[key].copy() for key in packed.files}
            ids=z['business_ids'].tolist();assert ids==[r['business_id'] for r in rows]
            corrected=[];offsets=[0];sizes=[];unknown=0
            for i,row in enumerate(rows):
                if i%250==0:print('AUDIT_ROW',name,i,flush=True)
                a,b=z['label_offsets'][i:i+2];old=z['labels'][a:b].astype(np.int64).tolist()
                new=remove_initial_bos(old)
                expected=tok.encode(row['target'],add_special_tokens=True,truncation=False,padding=False)
                content=tok.encode(row['target'],add_special_tokens=False,truncation=False,padding=False)
                assert expected==old and content+[2]==new and tok.decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)==row['target']
                ia,ib=z['input_offsets'][i:i+2];source=z['input_ids'][ia:ib].astype(np.int64).tolist()
                # Source tensors are never reconstructed by the correction.
                # Entire original NPZ + JSONL hashes are verified above; retain
                # their exact source IDs and per-row hashes without retokenizing.
                assert hashlib.sha256(row['source'].encode()).hexdigest()==row['source_sha256']
                assert not any(x in (0,1,2,50264) for x in new[:-1])
                unknown+=new.count(3);sizes.append(len(new))
                records.append(dict(dataset=name,business_id=row['business_id'],original_length=len(old),view_length=len(new),
                    target_text_sha256=hashlib.sha256(row['target'].encode()).hexdigest(),
                    original_label_sha256=hashlib.sha256(np.asarray(old,dtype='<i4').tobytes()).hexdigest(),
                    corrected_label_sha256=hashlib.sha256(np.asarray(new,dtype='<i4').tobytes()).hexdigest(),
                    source_input_ids_sha256=hashlib.sha256(np.asarray(source,dtype='<i4').tobytes()).hexdigest()))
                corrected.extend(new);offsets.append(len(corrected))
            if name.endswith('_filtered'):
                p=PREP/(name+'_content_eos_labels.npz')
                np.savez_compressed(p,business_ids=np.asarray(ids),labels=np.asarray(corrected,dtype=np.int32),label_offsets=np.asarray(offsets,dtype=np.int64))
                views[name]=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),rows=len(rows),
                    original_npz=info['path'],original_npz_sha256=info['sha256'])
        report['datasets'][name]=dict(rows=len(rows),original_npz_sha256=sha(npz),original_jsonl_sha256=sha(jsonl),
            one_initial_BOS_and_final_EOS=True,no_interior_BOS_EOS_PAD_MASK=True,unknown_tokens=unknown,
            full_target_tokenization_matches=True,content_tokenization_plus_EOS_matches_view=True,
            decoded_target_text_exact=True,source_input_ids_preserved_by_original_npz_hash=True,
            view_min_length=min(sizes),view_max_length=max(sizes),label_tokens_removed=len(rows))
        print('LABEL_AUDIT_PASS',name,len(rows),flush=True)
    report['rows_total']=sum(v['rows'] for v in report['datasets'].values())
    assert report['rows_total']==6600
    put(PREP/'LABEL_AUDIT.json',report)
    put(PREP/'LABEL_VIEWS.json',dict(policy=POLICY,views=views,source_npz_overwritten=False))
    put(PREP/'LABEL_ROW_RECEIPTS.jsonl',''.join(json.dumps(r)+'\n' for r in records))

PREFLIGHT='''def preflight():
    import stage_i_led_large_vast_d059_v1 as parent
    old=parent.preflight()
    c=read(CONFIG)
    expected=dict(old,experiment=EXPERIMENT,run_root='runs/F_large_decoder_policy_v1',
        decoder_label_policy=LABEL_POLICY,label_views_manifest='outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json',
        effective_generation_config_file='outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
        early_sanity=c['early_sanity'],execution_window=EXECUTION_WINDOW,
        vast_fresh=dict(old['vast_fresh'],experiment=EXPERIMENT,output='runs/F_large_decoder_policy_v1'),
        preparation_state='OFFLINE_READY_EXECUTION_NOT_APPROVED')
    require(c==expected,'Unreported scientific/config change')
    require(c['early_sanity']==read(PREP/'EARLY_SANITY_MANIFEST.json'),'Early case/budget drift')
    for rel,digest in read(ROOT/'VAST_SHA256_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Historical parent payload changed: '+rel)
    for rel,digest in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Corrected-policy binding differs: '+rel)
    import importlib
    installed=dict(LED='transformers.models.led.modeling_led',generation='transformers.generation.utils',
        Trainer='transformers.trainer',Accelerator='accelerate.accelerator')
    contract=read(PREP/'PINNED_RUNTIME_CONTRACT.json')['source_sha256']
    require(all(sha(Path(importlib.import_module(module).__file__))==contract[name]
        for name,module in installed.items()),'Pinned runtime source bytes differ')
    require(not c['GPU_execution_approved'] and not c['training_authorized'],'Preparation does not grant GPU approval')
    return c
'''
AUTHORIZE='''def authorize(path,mode,execute):
    require(execute and path is not None,'OFFLINE ONLY: new explicit hash-bound approval required')
    a=read(path)
    require(a.get('GPU_execution_approved') is True and a.get('training_authorized') is True
        and a.get('scientific_amendment_approved') is True,'Corrected decoder-policy execution not approved')
    require(a.get('experiment')==EXPERIMENT and a.get('decoder_label_policy')==LABEL_POLICY
        and a.get('approval_source')=='researcher_message' and a.get('researcher_explicit_authorization'),'Wrong approval scope')
    require(a.get('scopes')==['feasibility','train','evaluate'] and mode in a['scopes'],'Wrong mode scope')
    require(all(a.get(k)==v for k,v in binding().items()),'Authorization hash mismatch')
    require(a.get('execution_window')==EXECUTION_WINDOW and a.get('dynamic_overflow_policy')==POLICY
        and a.get('amp_scaler')==AMP,'Numerical/time policy differs')
    require(a.get('max_training_updates')==1250 and a.get('evaluation_outputs_max')==48
        and a.get('early_sanity_outputs_max')==4 and a.get('early_sanity_generation_approved') is True
        and a.get('technical_generation_calls_max')==2 and a.get('all_generation_calls_max')==54
        and a.get('feasibility_updates_max')==3 and a.get('feasibility_microbatches_max')==88,'Budget differs')
    require(a.get('fresh_only') is True and a.get('conditional_train_after_feasibility_pass') is True
        and a.get('held_out_access') is False and a.get('Teacher_calls') is False
        and a.get('RO_U_training_authorized') is False,'Forbidden extra scope or resume')
    check_deadline(a)
    return dict(a,_approval_sha256=sha(path))
'''
DEADLINE='''def check_deadline(a,now=None):
    from datetime import timedelta
    require(a.get('experiment')==EXPERIMENT and a.get('execution_window')==EXECUTION_WINDOW,'Wrong new clock scope')
    start=datetime.fromisoformat(a['execution_authorized_at_utc'])
    end=datetime.fromisoformat(a['execution_deadline_utc'])
    absolute=datetime.fromisoformat(a['researcher_absolute_stop_utc'])
    require(all(t.tzinfo is not None and t.utcoffset().total_seconds()==0 for t in (start,end,absolute)),'Actual UTC clocks required')
    require(start<=(now or datetime.now(timezone.utc))<end<=min(start+timedelta(hours=48),absolute),'New execution window expired/invalid')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True,'Researcher final-phase reserve confirmation required')
    require(a.get('max_training_updates')==1250 and a.get('all_generation_calls_max')==54,'Budget cannot be relaxed')
    return end.isoformat()
'''
CALLBACK='''def callback_class():
    Base=_core_callback_class()
    class Record(Base):
        def on_save(self,args,state,control,**kwargs):
            super().on_save(args,state,control,**kwargs)
            if state.global_step==250:
                from led_large_decoder_early_sanity_v1 import run_early_sanity
                run_early_sanity(self.trainer.model,self.folder,self.approval,read(CONFIG),self.memory,
                    ROOT,binding(),write,check_deadline)
            elif state.global_step>250:
                receipt=read(self.folder/'early_sanity/REPORT.json')
                require(receipt['status']=='PASS_TECHNICAL_ONLY' and receipt['binding']==binding()
                    and receipt['generation_calls']==4,'Early sanity PASS required before further checkpoints')
    return Record
'''
DATA='''def data(c):
    from stage_hi_led_data import PackedLEDDataset
    from led_large_decoder_label_view_v1 import CorrectedLabelDataset
    manifest=read(ROOT/c['stage_h_manifest'])
    views=read(ROOT/c['label_views_manifest'])
    require(views['policy']==c['decoder_label_policy'],'Label view policy differs')
    train=CorrectedLabelDataset(PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768,root=ROOT),views['views']['train_filtered'],ROOT)
    val=CorrectedLabelDataset(PackedLEDDataset(manifest['datasets']['validation_filtered'],5000,768,root=ROOT),views['views']['validation_filtered'],ROOT)
    require(len(train)==2000 and len(val)==200 and not set(train.business_ids)&set(val.business_ids),'Split mismatch')
    require(train.business_ids==read(LARGE_PREP/'EPOCH_ORDERS.json')['training_business_ids'],'Train ordering differs')
    return train,val
'''
COLLATE='''def collator(stress=False):
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    return CorrectedLabelCollator(stress=stress)
'''

def prepare_code():
    old=read(PARENT/'configs/stage_i_led_large_vast_d059_v1.json')
    cases=[json.loads(x) for x in (PARENT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    early=dict(step=250,epoch=1,outputs=4,cases_file='outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl',
        business_ids=[cases[i]['business_id'] for i in (0,1,20,22)],subsets=['random','random','targeted','targeted'],
        technical_only=True,requires_nonempty_no_BOS_natural_EOS_min16_content=True,semantic_quality_GO=False,
        stop_on_failure_without_retry=True)
    assert not cases[20]['selected_photo_ids']
    window=dict(policy='NEW_APPROVAL_UTC_MAX48H_RESEARCHER_ABSOLUTE_STOP_REQUIRED',historical_deadlines_unchanged=True)
    c=dict(old,experiment=NAME,run_root='runs/F_large_decoder_policy_v1',decoder_label_policy=POLICY,
        label_views_manifest='outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json',
        effective_generation_config_file='outputs/stage_i/led_large_preparation_v1/EXPECTED_EFFECTIVE_GENERATION_CONFIG.json',
        early_sanity=early,execution_window=window,preparation_state='OFFLINE_READY_EXECUTION_NOT_APPROVED',
        vast_fresh=dict(old['vast_fresh'],experiment=NAME,output='runs/F_large_decoder_policy_v1'))
    put(ROOT/CONFIG,c);put(PREP/'EARLY_SANITY_MANIFEST.json',early)
    h=(PARENT/'scripts/stage_i_led_large_vast_d059_v1.py').read_text(encoding='utf-8')
    h=h.replace("CONFIG = ROOT/'configs/stage_i_led_large_vast_d059_v1.json'",f"CONFIG = ROOT/'{CONFIG}'")
    h=h.replace("PREP = ROOT/'outputs/stage_i/led_large_vast_d059_v1_preparation'","PREP = ROOT/'outputs/stage_i/led_large_decoder_policy_v1'")
    h=h.replace("RUNNER = ROOT/'scripts/run_stage_i_led_large_vast_d059_v1.py'",f"RUNNER = ROOT/'{RUNNER}'")
    h=h.replace("EXPERIMENT = 'LED_LARGE_VAST_D059_F_V1'",f"EXPERIMENT = '{NAME}'\nLABEL_POLICY = '{POLICY}'")
    lines=h.splitlines();lines=[('EXECUTION_WINDOW = '+repr(window)) if line.startswith('EXECUTION_WINDOW = ') else line for line in lines];h='\n'.join(lines)+'\n'
    h=replace_function(h,'preflight',PREFLIGHT);h=replace_function(h,'authorize',AUTHORIZE);h=replace_function(h,'check_deadline',DEADLINE)
    h=h.replace("r['deadline_utc'] == DEADLINE","r['deadline_utc'] == a['execution_deadline_utc']")
    h=h.replace('def callback_class():','def _core_callback_class():',1)+'\n'+CALLBACK
    put(ROOT/HELPER,h)
    r=(PARENT/'scripts/run_stage_i_led_large_vast_d059_v1.py').read_text(encoding='utf-8')
    r=r.replace('from stage_i_led_large_vast_d059_v1 import','from stage_i_led_large_decoder_policy_v1 import')
    r=r.replace('from led_large_dynamic_amp_v2 import Recovery, POLICY, install_trainer','from led_large_dynamic_amp_v2 import Recovery, POLICY, install_trainer\nfrom led_large_decoder_label_view_v1 import remove_initial_bos')
    r=replace_function(r,'data',DATA);r=replace_function(r,'collator',COLLATE)
    r=r.replace('def train(c, a, out, resume=None):','def train(c, a, out, resume=None):\n    require(resume is None, "Fresh corrected-label training only; old adapters/state forbidden")')
    r=r.replace("labels=case['label_ids']","labels=remove_initial_bos(case['label_ids'])")
    assert 'labels=remove_initial_bos' in r
    r=r.replace('generation_calls=0,amp_dynamic=',"generation_calls=4,early_sanity_report_sha256=sha(folder/'early_sanity/REPORT.json'),amp_dynamic=")
    r=r.replace('runs/F_large_d059_v1','runs/F_large_decoder_policy_v1')
    r=r.replace('Fresh independent RTX3090 F run; never resume Colab','Fresh corrected-label RTX3090 F; new approval required')
    put(ROOT/RUNNER,r)
    reporter=(PARENT/'scripts/report_stage_i_led_large_vast_d059_v1.py').read_text(encoding='utf-8').replace('from stage_i_led_large_vast_d059_v1 import','from stage_i_led_large_decoder_policy_v1 import')
    reporter=reporter.replace("    write(dest/'ACTUAL_CONFIG.json',cfg)","    shutil.copytree(run/'train/early_sanity',dest/'early_sanity_epoch1')\n    write(dest/'ACTUAL_CONFIG.json',cfg)")
    reporter=reporter.replace('Train: 2000; validation loss: 200;', 'Corrected content/EOS labels; separate epoch1 technical sanity4; Train: 2000; validation loss: 200;')
    put(ROOT/REPORTER,reporter)
    diff=[]
    for before,after in [('scripts/stage_i_led_large_vast_d059_v1.py',HELPER),('scripts/run_stage_i_led_large_vast_d059_v1.py',RUNNER),('scripts/report_stage_i_led_large_vast_d059_v1.py',REPORTER),('configs/stage_i_led_large_vast_d059_v1.json',CONFIG)]:
        diff.extend(difflib.unified_diff((PARENT/before).read_text(encoding='utf-8').splitlines(True),(ROOT/after).read_text(encoding='utf-8').splitlines(True),fromfile=before,tofile=after))
    for name in ('scripts/led_large_decoder_label_view_v1.py','scripts/led_large_decoder_early_sanity_v1.py'):
        diff.extend(difflib.unified_diff([], (ROOT/name).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile=name))
    put(PREP/'EXACT_PROPOSED_PATCH.diff',''.join(diff))

def stage_workspace():
    STAGE.mkdir(exist_ok=True,parents=True)
    parent=read(PARENT/'VAST_SHA256_MANIFEST.json')['files']
    for rel,digest in parent.items():
        source=PARENT/rel;target=STAGE/rel
        assert sha(source)==digest
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    target=STAGE/'VAST_SHA256_MANIFEST.json'
    if not target.exists():shutil.copyfile(PARENT/'VAST_SHA256_MANIFEST.json',target)
    files=[CONFIG,HELPER,RUNNER,REPORTER,'RUN_DECODER_POLICY_V1.sh','scripts/led_large_decoder_label_view_v1.py','scripts/led_large_decoder_early_sanity_v1.py','scripts/prepare_led_large_decoder_policy_v1.py',
           'tests/test_led_large_decoder_policy_v1.py']
    # Bind immutable preparation inputs, not later validation logs/receipts.
    # This prevents circular/stale bindings when tests document this exact build.
    files += ['outputs/stage_i/led_large_decoder_policy_v1/'+name for name in (
        'train_filtered_content_eos_labels.npz','validation_filtered_content_eos_labels.npz',
        'LABEL_AUDIT.json','LABEL_VIEWS.json','LABEL_ROW_RECEIPTS.jsonl',
        'PINNED_RUNTIME_CONTRACT.json','EARLY_SANITY_MANIFEST.json','EXACT_PROPOSED_PATCH.diff')]
    manifest={rel:sha(ROOT/rel) for rel in files}
    manifest['VAST_SHA256_MANIFEST.json']=sha(PARENT/'VAST_SHA256_MANIFEST.json')
    put(PREP/'PREFLIGHT_MANIFEST.json',dict(files=manifest,parent_payload_files=len(parent),parent_is_read_only=True))
    bindings=dict(config_sha256=sha(ROOT/CONFIG),runner_sha256=sha(ROOT/RUNNER),preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'),helper_sha256=sha(ROOT/HELPER),label_module_sha256=sha(ROOT/'scripts/led_large_decoder_label_view_v1.py'))
    put(PREP/'EXECUTION_BINDINGS.json',bindings)
    template=dict(experiment=NAME,decoder_label_policy=POLICY,GPU_execution_approved=False,training_authorized=False,scientific_amendment_approved=False,
        approval_source='researcher_message',researcher_explicit_authorization=None,scopes=['feasibility','train','evaluate'],
        max_training_updates=1250,evaluation_outputs_max=48,early_sanity_outputs_max=4,early_sanity_generation_approved=False,
        technical_generation_calls_max=2,all_generation_calls_max=54,feasibility_updates_max=3,feasibility_microbatches_max=88,
        amp_scaler=read(ROOT/CONFIG)['amp_scaler'],dynamic_overflow_policy=read(ROOT/CONFIG)['dynamic_overflow_policy'],
        execution_window=read(ROOT/CONFIG)['execution_window'],execution_authorized_at_utc=None,execution_deadline_utc=None,
        researcher_absolute_stop_utc=None,defense_reserve_at_least_14_days_confirmed=False,
        fresh_only=True,conditional_train_after_feasibility_pass=True,held_out_access=False,Teacher_calls=False,RO_U_training_authorized=False,
        **{k:v for k,v in bindings.items() if k in ('config_sha256','runner_sha256','preflight_manifest_sha256')})
    put(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json',template)
    files += ['outputs/stage_i/led_large_decoder_policy_v1/'+n for n in ('PREFLIGHT_MANIFEST.json','EXECUTION_BINDINGS.json','AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')]
    for rel in files:
        p=STAGE/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=(ROOT/rel).read_bytes():
            assert rel not in parent,'Cannot replace a historical payload file'
            assert p.resolve().is_relative_to(STAGE.resolve()),'Unsafe candidate refresh path'
            p.unlink()  # Only newly created preparation files; never historical payload links.
        if not p.exists():shutil.copyfile(ROOT/rel,p)
    print('PREPARATION_BINDINGS',json.dumps(bindings),flush=True)

if __name__=='__main__':
    import sys
    import faulthandler
    faulthandler.dump_traceback_later(45,repeat=True)
    PREP.mkdir(parents=True,exist_ok=True)
    if '--stage-only' not in sys.argv:
        if '--code-only' not in sys.argv:audit_labels()
        prepare_code()
    stage_workspace()
