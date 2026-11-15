"""Bind real completed Qwen outputs to pre-frozen slots. Does not release scoring.

Refuses stored ratings, stale hashes, incomplete output and overwrites. The staged
workspace has sources/slots only until this succeeds; missing Qwen is never an
empty summary. Use --previous-workspace to recheck the actual earlier study.
"""
from pathlib import Path
import argparse,hashlib,json,sqlite3,sys
from contextlib import closing
from stage_p_qwen_inference_v1 import load,read,put,sha,textsha,require,tokenizer,messages

def no_ratings(workspace):
    require(workspace.is_dir(),'Previous workspace unavailable: cannot verify no collected ratings')
    found=[]
    for db in workspace.rglob('*.sqlite3'):
        with closing(sqlite3.connect('file:'+db.resolve().as_posix()+'?mode=ro',uri=True)) as c:
            tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for name in ['ratings','events']:
                if name in tables:require(c.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]==0,'STOP: existing ratings/journal in '+str(db))
        found.append(str(db))
    return found

def finalize(root,execution,previous_workspace):
    no_ratings(previous_workspace);w=root/'HUMAN_APP/WORKSPACE';no_ratings(w)
    binding=read(root/'PREPARATION_BINDINGS.json')
    require(sha(previous_workspace/'CONFIG.json')==binding['original_stage_o_config_sha256'],'Wrong earlier study configuration; inspect actual collecting workspace')
    require(sha(previous_workspace/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')==binding['original_stage_o_private_items_sha256'],'Earlier assignments differ')
    for name,expected in binding['prepared_files'].items():require(sha(root/name)==expected,'Prepared artifact changed: '+name)
    require(not (w/'CONFIG.json').exists() and not (w/'COORDINATOR_ONLY/PRIVATE_ITEMS.json').exists(),'Finalized workspace already exists; never overwrite assignments')
    spec,prompts,inputs=load(root);plan=read(w/'PLANNED_CONFIG.json');slots=read(w/'COORDINATOR_ONLY/ASSIGNMENT_SLOTS_120.json')
    require(sha(root/'FROZEN_SPEC.json')==plan['inference_spec_sha256'],'Plan spec binding changed')
    require(sha(w/'COORDINATOR_ONLY/ASSIGNMENT_SLOTS_120.json')==plan['assignment_slots_sha256'],'Frozen slots changed')
    receipt_path=execution/'COMPLETION_RECEIPT.json'
    require(receipt_path.is_file(),'PENDING: no native 30-call Qwen completion receipt; do not create placeholder assignments')
    receipt=read(receipt_path);outputpath=execution/'QWEN_OUTPUTS_15.json'
    require(receipt['status']=='COMPLETE_30_NATIVE_CALLS' and receipt['calls']==30 and receipt['businesses']==15,'Incomplete native run')
    require(receipt['spec_sha256']==sha(root/'FROZEN_SPEC.json') and receipt['outputs_sha256']==sha(outputpath),'Run does not match frozen spec/output')
    name=next((n for n,m in spec['models'].items() if m['model_id']==receipt['model']['model_id'] and m['revision']==receipt['model']['revision']),None)
    require(name is not None and receipt['model']==spec['models'][name],'Wrong model provenance')
    require(receipt['precision']==spec['precision'],'Precision changed')
    require(receipt['weights_verified']==spec['models'][name]['weights'],'Weight verification differs')
    require(set(receipt['native_call_hashes'])=={r['call_id'] for r in inputs},'Missing/extra native calls')
    tok,_=tokenizer(root,name);native={}
    for source in inputs:
        call=source['call_id'];path=execution/'CALLS'/(call+'.json');require(sha(path)==receipt['native_call_hashes'][call],'Native call hash mismatch')
        row=read(path)
        require(all(row[k]==source[k] for k in ['business_id','task','call_id','source_sha256','selected_review_ids','selected_photo_ids']),'Source/identity mismatch')
        require(row['model_id']==spec['models'][name]['model_id'] and row['model_revision']==spec['models'][name]['revision'],'Mixed models')
        require(row['spec_sha256']==sha(root/'FROZEN_SPEC.json') and row['prompt_version']==spec['prompt_version'],'Prompt/spec mismatch')
        expected_ids=tok.apply_chat_template(messages(prompts,source),tokenize=True,add_generation_prompt=True,enable_thinking=False)
        require(row['input_ids']==expected_ids and row['input_ids_sha256']==textsha(json.dumps(expected_ids,separators=(',',':'))),'Native input token IDs differ from frozen prompts/sources')
        settings={k:v for k,v in spec['generation'].items() if k!='max_new_tokens_by_task'}
        settings.update(max_new_tokens=spec['generation']['max_new_tokens_by_task'][source['task']],pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=tok.eos_token_id)
        require(row['generation_settings']==settings,'Generation settings changed')
        require(textsha(row['text'])==row['text_sha256'] and tok.decode(row['generation_token_ids'],skip_special_tokens=True)==row['text'],'Generated text differs from native token IDs')
        require(tok.decode(row['generation_token_ids'],skip_special_tokens=False)==row['raw_decoded_with_special_tokens'],'Raw token decoding differs')
        native[source['business_id'],source['task']]=row
    outputs=read(outputpath);require(len(outputs)==15 and len({r['business_id'] for r in outputs})==15,'15 unique real Qwen outputs required')
    led=read(root/'LED_FROZEN_COMPONENTS_45.json');by={(r['business_id'],r['condition']):r for r in led}
    for row in outputs:
        b=row['business_id'];require(row['condition']=='QWEN','Wrong condition')
        for task in ['review','visual']:
            require(row[task+'_text']==native[b,task]['text'] and row[task+'_text_sha256']==native[b,task]['text_sha256'],'Assembled text changed')
        require(row['combined_text']==row['review_text']+'\n\n'+row['visual_text'],'Assembly is not verbatim')
        by[b,'QWEN']=row
    items=[]
    require(len(slots)==120 and len({s['alias'] for s in slots})==120,'Invalid slots')
    for slot in slots:
        row=by[slot['business_id'],slot['condition']]
        items.append({**slot,**{k:row[k] for k in ['review_text','visual_text','review_text_sha256','visual_text_sha256']}})
    for rater in ['A01','A02']:
        assigned=[i for i in items if i['annotator']==rater]
        require(len(assigned)==60 and len({(i['business_id'],i['condition']) for i in assigned})==60,'Unbalanced assignment')
    for b,bind in plan['sources'].items():
        source=read(w/bind['file'])
        # Stage O canonical digest (JSON sort + separators, UTF-8).
        digest=hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(digest==bind['canonical_sha256'],'Original source binding changed')
        for photo in source['photos']:
            f=w/'PHOTOS'/(photo['photo_id']+'.jpg')
            require(f.stat().st_size==photo['bytes'] and sha(f)==photo['sha256']==photo['original_source_sha256'],'Original photo changed')
    put(w/'COORDINATOR_ONLY/PRIVATE_ITEMS.json',items)
    config={k:v for k,v in plan.items() if k not in ['assignment_slots_sha256','inference_spec_sha256']}
    config['file_bindings']={n:sha(w/n) for n in ['COORDINATOR_ONLY/PRIVATE_ITEMS.json','COORDINATOR_ONLY/ASSIGNMENT_SLOTS_120.json','COORDINATOR_ONLY/RANDOMIZATION_SEEDS.json','RUBRIC.json','TWO_RATER_PROTOCOL_AMENDMENT.md']}
    config['Qwen_native_receipt_sha256']=sha(receipt_path);config['Qwen_spec_sha256']=sha(root/'FROZEN_SPEC.json')
    put(w/'CONFIG.json',config)
    put(root/'VALIDATION/WORKSPACE_FINALIZATION.json',dict(status='FINALIZED_NOT_RELEASED',outputs=60,assignments=120,ratings_created=0,previous_workspace_rechecked=str(previous_workspace),
        config_sha256=sha(w/'CONFIG.json'),Qwen_native_receipt_sha256=sha(receipt_path),stage_p_protocol_sha256=sha(w/'TWO_RATER_PROTOCOL_AMENDMENT.md')))
    print('Finalized 120 genuine blinded items, 60 per rater. Live release absent; review and E2E validation are still required before collection.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--execution',type=Path);p.add_argument('--previous-workspace',type=Path,required=True)
    a=p.parse_args();r=a.root.resolve();finalize(r,a.execution or r/'QWEN_EXECUTION',a.previous_workspace.resolve())
