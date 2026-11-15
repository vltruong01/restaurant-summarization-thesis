"""Verify a complete downloaded Stage O run and prepare a review-only handoff.

No generation, semantic grading, human submissions or live release.
"""
from pathlib import Path
from collections import Counter
import ast,hashlib,html,json,shutil,subprocess,sys,zipfile
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/stage_o/tripadvisor_external_v1';E=O/'RTX3080_EXECUTION_V1'
D=E/'DOWNLOADED_NATIVE/STAGE_O'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def textsha(x):return hashlib.sha256(x.encode()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    assert read(E/'LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json')['status']=='PASS'
    assert read(D/'RTX3080_EXECUTION_V1/WORKER_STATUS.json')['state']=='NATIVE_GENERATION_COMPLETE_PENDING_LOCAL_DOWNLOAD_VERIFICATION'
    assert sha(D/'COHORT15_FREEZE.json')==sha(O/'COHORT15_FREEZE.json')
    freeze=read(O/'COHORT15_FREEZE.json');ids={r['business_id'] for r in freeze['restaurants']}
    from caption_basic_quality import evaluate
    import yaml
    basic=yaml.safe_load((ROOT/'configs/basic_caption_quality.yaml').read_text(encoding='utf8'))
    plan=read(D/'PREPARED_CAPTION_JOB/pilot_photo_plan.json');assert len(plan)==5143
    captions={}
    for p in plan:
        r=read(D/'PREPARED_CAPTION_JOB/caption_checkpoints'/(p['photo_id']+'.json'))
        assert r['business_id']==p['business_id'] and r['raw_sha256']==p['raw_sha256']==sha(ROOT/p['image_path'])
        assert r['model_id']=='Salesforce/blip2-opt-2.7b-coco' and r['resolved_commit']=='cda95b9319b722f79c9451a3d8ff92eea02048dc'
        assert len(r['attempts'])==1+int(r['attempts'][0]['basic_quality_fail'])
        for a in r['attempts']:
            for key,value in evaluate(a,basic).items():assert a[key]==value,(p['photo_id'],key)
        chosen=r['attempts'][r['chosen_attempt']-1]
        assert r['raw_caption']==chosen['raw_caption'] and r['normalized_caption']==chosen['normalized_caption']
        assert r['basic_eligible']==(not chosen['basic_quality_fail'])
        captions[p['photo_id']]=r
    assert len(captions)==5143
    assert sha(D/'PREPROCESSING/shared_review_inputs.jsonl')==sha(O/'PREPROCESSING/shared_review_inputs.jsonl')
    tok=Tokenizer.from_file(str(ROOT/'assets/tokenizers/led-base-16384/tokenizer.json'));tok.no_padding();tok.no_truncation()
    calls=read(D/'LED_INFERENCE_CALL_PLAN_75.json')['calls'];assert len(calls)==75
    index={(c['condition'],c['business_id'],c['task']):c for c in calls};assert len(index)==75
    lock=read(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')
    condition_inputs=read(D/'FINAL_CONDITION_INPUTS_45.json');assert len(condition_inputs)==45
    from source_budget_audit_template import counts
    from s1000_final_eval_core_v1 import isolated
    reviews={r['business_id']:r for r in map(json.loads,(O/'PREPROCESSING/shared_review_inputs.jsonl').read_text(encoding='utf8').splitlines())}
    budget=[]
    for row in condition_inputs:
        b=row['business_id'];rr=reviews[b]
        assert len(row['selected_photo_ids'])<=5
        assert all(captions[p]['business_id']==b for p in row['selected_photo_ids'])
        assert row['captions']==[captions[p]['normalized_caption'] for p in row['selected_photo_ids']]
        if row['condition']=='RO':assert row['source']==rr['base_source'];continue
        info,source=counts(tok,rr['metadata'],rr['review_block'],row['captions']);assert source==row['source']
        assert not any(info[k] for k in ['review_budget_overflow','caption_budget_overflow','source_budget_overflow'])
        rs,vs=isolated(source);assert rs==rr['review_task_source']
        for task,text in [('review',rs),('visual',vs)]:assert index[row['condition'],b,task]['source']==text
        budget.append(dict(business_id=b,condition=row['condition'],**info))
    records=[];assembled=[]
    for condition in ['RO','U','F']:
        completion=read(D/'LED_OUTPUTS'/condition/'COMPLETION.json')
        assert completion['status']=='COMPLETE' and completion['businesses']==15 and completion['checkpoint_unchanged']
        raw=read(D/'LED_OUTPUTS'/condition/'raw_predictions.json')
        assert sha(D/'LED_OUTPUTS'/condition/'raw_predictions.json')==completion['raw_sha256']
        assert len(raw)==(15 if condition=='RO' else 30)
        assert {r['business_id'] for r in raw}==ids
        for r in raw:
            c=index[condition,r['business_id'],r['task']]
            assert textsha(r['text'])==r['text_sha256'] and tok.decode(r['token_ids'],skip_special_tokens=True)==r['text']
            assert r['generation_settings']==lock['generation'] and r['generation_config_sha256']==digest(lock['generation'])
            assert r['checkpoint_sha256']==lock['models'][condition]['adapter_sha256']==c['checkpoint_sha256']
            assert r['source_sha256']==c['source_sha256']==textsha(c['source'])
            assert r['input_ids']==c['input_ids']==tok.encode(c['source']).ids and digest(r['input_ids'])==r['input_ids_sha256']==c['input_ids_sha256']
        records.extend(raw)
        predictions=read(D/'LED_OUTPUTS'/condition/'predictions.json');assert len(predictions)==15
        for row in predictions:
            rr={r['task']:r for r in raw if r['business_id']==row['business_id']}
            assert row['combined_text']=='\n\n'.join(rr[t]['text'] for t in ['review','visual'] if t in rr)
            assert not row['rewriting']
        assembled.extend(predictions)
    assert len(records)==75 and len(assembled)==45
    assert assembled==read(D/'TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json')
    # Materialize only newly generated artifacts; pre-existing frozen sources must compare byte-for-byte.
    for folder in ['PREPARED_CAPTION_JOB','CAPTION_SCORES','LED_OUTPUTS']:
        for p in (D/folder).rglob('*'):
            if p.is_file():
                dest=O/p.relative_to(D);dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert sha(dest)==sha(p),dest
                else:shutil.copyfile(p,dest)
    for name in ['FINAL_U_F_CAPTION_POOLS.json','FINAL_CONDITION_INPUTS_45.json','LED_INFERENCE_CALL_PLAN_75.json','TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json']:
        dest=O/name
        if dest.exists():assert sha(dest)==sha(D/name)
        else:shutil.copyfile(D/name,dest)
    # Real outputs are now present. Prepare aliases/sources only; the builder never releases ratings.
    if not (O/'HUMAN_APP/WORKSPACE').exists():subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/prepare_stage_o_human_workspace_v1.py')],check=True)
    assert not (O/'HUMAN_APP/WORKSPACE/LIVE_RELEASE.json').exists()
    app=O/'HUMAN_APP/WORKSPACE'
    private=read(app/'COORDINATOR_ONLY/PRIVATE_ITEMS.json');assert len(private)==90
    for rater in ['A01','A02']:
        a=[r for r in private if r['annotator']==rater];assert len(a)==45 and {(r['business_id'],r['condition']) for r in a}=={(b,c) for b in ids for c in ['RO','U','F']}
    assert not list(app.rglob('*.sqlite3'))
    qwen=read(O/'QWEN_SHARED_INPUT_MANIFEST.json');qwen['status']='FROZEN_SHARED_INPUTS_WITH_ACTUAL_CAPTIONS_QWEN_NOT_AUTHORIZED'
    for row in qwen['restaurants']:
        b=row['business_id']
        for c,label in [('U','unfiltered'),('F','filtered')]:
            x=next(x for x in condition_inputs if x['business_id']==b and x['condition']==c)
            row[label+'_captions']=[dict(photo_id=p,caption=t) for p,t in zip(x['selected_photo_ids'],x['captions'])]
        row['caption_status']='COMPLETE_FROZEN_K5_INPUTS'
    save(E/'QWEN_SHARED_INPUT_MANIFEST_COMPLETE.json',qwen)
    report=dict(status='PASS_NATIVE_75_COMPONENTS_45_OUTPUTS_VERIFIED_LOCALLY_STOP_BEFORE_HUMAN',cohort_restaurants=15,source_photos=5143,selected_reviews=247,
                raw_BLIP_caption_records=5143,Basic_eligible=sum(r['basic_eligible'] for r in captions.values()),
                native_LED_components=75,assembled_outputs=45,actual_human_ratings=0,Qwen_inference=0,training=0,
                adapter_backbone_runtime_source_input_token_output_integrity='PASS',all_caption_QC_recomputed='PASS',
                all_30_U_F_source_budgets_recomputed='PASS',all_90_blinded_assignments_real_output_and_ownership_checks='PASS',
                full_HUMAN_APP_HTTP_runtime_test='PENDING_RELEASE_AND_ACCOUNT_SETUP',
                output_format_flags=[dict(condition=r['condition'],business_id=r['business_id'],task=r['task'],**r['format_flags']) for r in records if not r['format_flags']['expected_header'] or r['format_flags']['foreign_header']],
                capped_outputs=[dict(condition=r['condition'],business_id=r['business_id'],task=r['task']) for r in records if r['termination']['at_length_limit']],
                no_selective_repair_or_regeneration=True,no_semantic_factuality_claim=True)
    save(E/'COMPLETED_NATIVE_VALIDATION.json',report);save(E/'FINAL_TOKEN_BUDGET_LEDGER.json',budget)
    P=E/'COMPLETED_REVIEW_PACKAGE';P.mkdir(exist_ok=False)
    shutil.copytree(O/'REVIEW_PACKAGE/SOURCE_PHOTOS',P/'SOURCE_PHOTOS')
    for folder in ['RESTAURANTS','PREPROCESSING','PREPARED_CAPTION_JOB','CAPTION_SCORES','LED_OUTPUTS','PROTOCOL','HUMAN_APP']:
        if (O/folder).exists():shutil.copytree(O/folder,P/folder,ignore=shutil.ignore_patterns('*.pyc','__pycache__','*.sqlite3','*.lock'))
    for name in ['COHORT15_FREEZE.json','SOURCE_IMAGE_DIAGNOSTICS.json','FINAL_U_F_CAPTION_POOLS.json','FINAL_CONDITION_INPUTS_45.json','LED_INFERENCE_CALL_PLAN_75.json','TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json','SOURCE_READER.html']:
        shutil.copyfile(O/name,P/name)
    shutil.copyfile(E/'QWEN_SHARED_INPUT_MANIFEST_COMPLETE.json',P/'QWEN_SHARED_INPUT_MANIFEST.json')
    shutil.copyfile(E/'COMPLETED_NATIVE_VALIDATION.json',P/'VALIDATION_REPORT.json')
    shutil.copyfile(E/'FINAL_TOKEN_BUDGET_LEDGER.json',P/'FINAL_TOKEN_BUDGET_LEDGER.json')
    shutil.copytree(E/'DOWNLOADED_NATIVE/RUNTIME',P/'RUNTIME')
    shutil.copytree(E/'DOWNLOADED_NATIVE/CODE',P/'EXECUTED_CODE')
    for name in ['NATIVE_EXPORT_RECEIPT.json','LOCAL_NATIVE_DOWNLOAD_VERIFICATION.json']:
        shutil.copyfile(E/name,P/name)
    shutil.copytree(O/'REVIEW_PACKAGE/REPRODUCIBILITY',P/'REPRODUCIBILITY')
    shutil.copytree(O/'REVIEW_PACKAGE/DATASET_PROVENANCE',P/'DATASET_PROVENANCE')
    (P/'HUMAN_APP/README.md').write_text('''# Reference-free Stage O app — real outputs prepared, live rating release pending

WORKSPACE contains 90 genuine-output assignments, common selected reviews and source photographs. COORDINATOR_ONLY holds identities/seeds and must remain hidden from evaluators. The reused Stage N app is adapted to three conditions, two evaluators, 45 items per evaluator and localhost port 8767. No accounts or passwords, live release, SQLite rating store, submissions or approvals were generated. The researcher must inspect the completed package before human collection. After explicit release and account setup, run the adapted application's serve command against this WORKSPACE. Full HTTP/session/persistence checks remain pending that setup; only assignment/ownership/source/output integrity has been checked here.
''',encoding='utf8')
    for name in ['DATASET_FEASIBILITY.md','EXECUTION_PLAN_AND_COST.md']:
        dest=P/'HISTORICAL_PREPARATION'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(O/name,dest)
    (P/'README.md').write_text('''# Stage O — completed native inference, independent review required

The frozen 15 TripAdvisor restaurants, 247 selected reviews and all 5,143 source photographs were retained. Native BLIP captions and frozen U/F selection are complete. All 75 original LED components and 45 RO/U/F output records have been downloaded and independently checked locally against the frozen model/source/decoder bindings. See VALIDATION_REPORT.json for actual counts and preserved cap/format flags. No quality repair or selective regeneration was performed. This is not a claim that every generated assertion is true.

Open SOURCE_READER.html for sources. Exact outputs are in TRIPADVISOR15_RO_U_F_PREDICTIONS_45.json and LED_OUTPUTS/. Raw caption attempts, token IDs, Basic-QC evidence, scoring embeddings, source/input token IDs, budgets, native logs and file hashes are included. Weight binaries are not repeated in this review ZIP.

HUMAN_APP/WORKSPACE contains 90 real-output blinded assignments (45 per evaluator), selected reviews and all photographs. COORDINATOR_ONLY contains the model identity mapping and must not be given to evaluators. No accounts, live release or rating submissions exist. This package must be inspected before real human collection. Full HTTP/account/persistence testing is still pending the later release/setup; no simulated human results were created.

QWEN_SHARED_INPUT_MANIFEST.json now contains the actual U/F caption inputs with the same frozen review/photo evidence. Qwen inference/configuration is not authorized by this stage. Teacher targets, references and training were not generated. This exploratory single-city external study cannot conclusively disprove Teacher mimicry. Source photographs are publisher-linked CDN renderings, not attested camera-original files. Cost is not part of the methodology report, as requested.

HISTORICAL_PREPARATION contains the prior feasibility/resource report and is explicitly historical; its pending markers describe the state before this completed inference. Frozen Yelp A–N results remain unchanged. Instance lifecycle actions are not authorized or performed.
''',encoding='utf8')
    lifecycle = E/'INSTANCE_DESTROY_AUTHORIZATION.json'
    if lifecycle.exists():
        shutil.copyfile(lifecycle,P/lifecycle.name)
        shutil.copyfile(E/'INSTANCE_DESTROY_README.md',P/'INSTANCE_DESTROY_README.md')
        readme = P/'README.md'
        text = readme.read_text(encoding='utf8')
        text = text.replace('Instance lifecycle actions are not authorized or performed.',
            'The researcher subsequently authorized destruction of this exact instance after verified successful local closeout. A separate local watcher performs final preservation and identity checks before one provider DELETE. This package is built before that operation; only INSTANCE_DESTRUCTION_RESULT.json can establish its actual outcome. See INSTANCE_DESTROY_README.md.')
        readme.write_text(text,encoding='utf8')
    files=[p for p in P.rglob('*') if p.is_file()]
    save(P/'PACKAGE_MEMBER_SHA256.json',[dict(file=p.relative_to(P).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)])
    archive=E/'STAGE_O_TRIPADVISOR_EVALUATION_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(P))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in read(P/'PACKAGE_MEMBER_SHA256.json'):assert hashlib.sha256(z.read(r['file'])).hexdigest()==r['sha256']
    save(E/'FINAL_COMPLETED_PACKAGE_RECEIPT.json',dict(status='PASS',ZIP=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),outputs=45,components=75,ratings=0))
    print(json.dumps(read(E/'FINAL_COMPLETED_PACKAGE_RECEIPT.json'),indent=2),flush=True)
if __name__=='__main__':main()
