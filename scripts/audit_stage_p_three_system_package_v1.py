"""Portable standard-library audit. Never loads weights, generates text or changes ratings."""
import argparse,csv,hashlib,json,math,sys,tempfile,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True  # Keep the manifest-bound review directory read-only on ordinary execution.
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def tsha(t):return hashlib.sha256(t.encode()).hexdigest()
def audit(root,check_manifest=True):
    n=root/'NATIVE';checks=[]
    def passed(name,count):checks.append(dict(check=name,status='PASS',count=count))
    if check_manifest:
        manifest=read(root/'PACKAGE_MANIFEST.json');expected={x['file'] for x in manifest}|{'PACKAGE_MANIFEST.json'}
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()};assert actual==expected
        for x in manifest:
            p=root/x['file'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['file']
        passed('Exact package member set, byte sizes and SHA256',len(manifest))
    for x in read(n/'NATIVE_EXPORT_MANIFEST.json'):
        p=n/x['file'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    passed('Native download every scientific member SHA256',len(read(n/'NATIVE_EXPORT_MANIFEST.json')))
    plan=read(n/'INPUTS/LED_F_CALL_PLAN_30.json');oldplan=[x for x in read(n/'PROVENANCE/ORIGINAL_LED_PLAN_75.json')['calls'] if x['condition']=='F'];assert plan==oldplan
    ids={x['business_id'] for x in plan};assert len(ids)==15 and len(plan)==30
    passed('LED exact accepted source call plan, business membership and input IDs',30)
    old={(x['business_id'],x['task']):x for x in read(n/'INPUTS/ORIGINAL_LED_F_COMPONENTS_30.json')};led=read(n/'LED_BENCHMARK/BENCHMARK_RECORDS_62.json');assert len(led)==62
    lock=read(n/'INPUTS/LED_MODEL_AND_DECODER_LOCK.json');rest=read(n/'LED_BENCHMARK/RESTORATION.json')
    assert rest['adapter_sha256']==lock['models']['F']['adapter_sha256'] and rest['backbone_sha256']==lock['backbone']['sha256'] and rest['generation']==lock['generation']
    reps={}
    for c in led:
        p=next(p for p in plan if (p['business_id'],p['task'])==(c['business_id'],c['task']))
        assert c['input_ids']==p['input_ids'] and c['source_sha256']==p['source_sha256'] and c['text_sha256']==tsha(c['text'])
        o=old[c['business_id'],c['task']];assert c['original_text_exact']==(c['text']==o['text']);assert c['original_tokens_exact']==(c['token_ids']==o['token_ids'])
        if c['phase']!='WARMUP':reps.setdefault(c['phase'],[]).append(c)
    assert sorted(map(len,reps.values()))==[30,30]
    assert all(read(n/x)['exit_code']==0 for x in ['LED_JOB_EXIT.json','IMAGE_JOB_EXIT.json'])
    assert read(n/'CPU_LED_REPLAY.json')['status']=='PASS' and len(read(n/'CPU_LED_REPLAY.json')['calls'])==62
    passed('LED original/rerun equivalence calculations, locked hashes/decoder, saved CPU decode receipt',62)
    ip=read(n/'IMAGE_PLAN_15.json');pre={c['business_id']:c for c in read(n/'IMAGE_PREFLIGHT.json')['calls']};assert {c['business_id'] for c in ip}==ids
    oldq={c['business_id']:c for c in read(n/'INPUTS/QWEN_CAPTION_OUTPUTS_15.json')};newq={c['business_id']:c for c in read(n/'QWEN_IMAGE/QWEN_IMAGE_OUTPUTS_15.json')};assert set(oldq)==set(newq)==ids
    count=0;photo_ids=[]
    for item in ip:
        bid=item['business_id'];c=read(n/'QWEN_IMAGE/CALLS'/f'{bid}.json');qc=read(n/'INPUTS/QWEN_CAPTION_CALLS'/f'{bid}__visual.json')
        assert c['generation_settings']==qc['generation_settings'] and c['model_revision']==qc['model_revision']
        assert c['selected_photo_ids']==item['selected_photo_ids']==qc['selected_photo_ids']==[x['photo_id'] for x in item['photos']]
        assert c['input_provenance']==pre[bid] and c['text_sha256']==tsha(c['text'])
        for p in item['photos']:
            f=n/p['file'];assert f.stat().st_size==p['bytes'] and sha(f)==p['sha256'];count+=1;photo_ids.append(p['photo_id'])
        z=newq[bid];assert z['review_text']==oldq[bid]['review_text'] and z['review_text_sha256']==oldq[bid]['review_text_sha256']==tsha(z['review_text'])
        assert z['visual_text']==c['text'] and z['combined_text']==z['review_text']+'\n\n'+z['visual_text']
        assert not c['flags']['empty'] and not c['flags']['unexpected_thinking'] and c['flags']['stop_token_seen']
    assert count==73 and len(set(photo_ids))==73
    assert read(n/'CPU_IMAGE_REPLAY.json')['status']=='PASS' and len(read(n/'CPU_IMAGE_REPLAY.json')['calls'])==15
    assert read(n/'QWEN_IMAGE/COMPLETION.json')['new_review_calls']==0
    passed('All73 exact photo bytes/order;15 Visual calls/settings;15 verbatim Review reuse;CPU pixel/token replay receipt',73)
    assert read(n/'IMAGE_EXECUTION_BINDING.json')['worker_sha256']==sha(n/'run_stage_p_qwen_image_v1.py')
    passed('Effective executed image-worker hash',1)
    observations=read(root/'PRELIMINARY_PHOTO_OBSERVATIONS.json')['observations'];assert len(observations)==15 and {c['business_id'] for c in observations}==ids
    for c in observations:
        assert c['output_span'] in newq[c['business_id']]['visual_text']
        assert set(c['photo_ids'])<=set(pre[c['business_id']]['selected_photo_ids'])
    passed('Preliminary observation exact spans and source IDs (not semantic certification)',15)
    records=read(root/'THREE_SYSTEM_PREDICTIONS_45.json');assert len(records)==45
    for c in records:
        bid=c['business_id']
        if c['system']=='LED F original':assert c['review_text']==old[bid,'review']['text'] and c['visual_text']==old[bid,'visual']['text']
        else:
            original=(oldq if c['system']=='Qwen-Caption' else newq)[bid];assert c['review_text']==original['review_text'] and c['visual_text']==original['visual_text']
    passed('45 assembled system records preserve actual original components',45)
    # Recompute numerical reports to a temporary directory; byte-check deterministic tables.
    spec=importlib.util.spec_from_file_location('resource_analysis',root/'CODE/analyze_stage_p_three_system_v1.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);mod.analyze(root,p);files=list(p.iterdir())
        for f in files:assert f.read_bytes()==(root/'REPORTS'/f.name).read_bytes(),f.name
    passed('Recomputed descriptive CSV/JSON/Markdown reports byte for byte',len(files))
    return dict(status='PASS',checks=checks,semantic_quality_certified=False,new_human_ratings=0,limitations='Structural/resource audit and saved CPU replay evidence; participant identity, image semantics, causal attribution and future system quality are not independently certified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--skip-manifest',action='store_true');a=p.parse_args();print(json.dumps(audit(a.root.resolve(),not a.skip_manifest),indent=2))
