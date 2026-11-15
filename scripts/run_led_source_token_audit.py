"""Measure accepted selections once, keep a small token ledger, no model calls."""
from pathlib import Path
import collections,hashlib,json,os,time
import pandas as pd,orjson,yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from review_selection_pilot import formatted_input
from led_source_token_audit import ROOT,CONFIG,load_inputs,metadata_text,source_text,empty_caption_source,digest_text,digest_ids,summaries


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    started=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];accepted=ROOT/cfg['accepted_directory']
    assert not (out/'completion_report.json').exists(),'Completed audit exists; verify, do not overwrite.'
    assert json.loads((accepted/'validation_report.json').read_text())['validation_passed']
    assert json.loads((accepted/'acceptance_report.json').read_text())['status']=='PASS'
    completion=json.loads((accepted/'completion_report.json').read_text());parent=json.loads((accepted/'scientific_identity.json').read_text())['identity']
    protected={}
    for name in ['selection_runs.jsonl','businesses.csv']:
        path=accepted/name;assert file_hash(path)==completion['artifact_sha256'][name]
        protected[path.relative_to(ROOT).as_posix()]=file_hash(path)
    for split in ['train','validation']:
        name=(Path(cfg['english_directory'])/'reviews'/f'{split}.jsonl').as_posix()
        assert file_hash(ROOT/name)==parent['protected_sha256'][name];protected[name]=parent['protected_sha256'][name]
    checksums=json.loads((ROOT/'data/processed/yelp_2300_v1/checksums.json').read_text())
    assert file_hash(ROOT/cfg['business_records'])==checksums['businesses.jsonl']['sha256']
    for name in [cfg['business_records'],cfg['tokenizer'],cfg['selector_config'],'configs/research_pending.yaml','scripts/review_selection_pilot.py',cfg['accepted_directory']+'/completion_report.json',cfg['accepted_directory']+'/validation_report.json',cfg['accepted_directory']+'/acceptance_report.json']:
        protected[name]=file_hash(ROOT/name)
    assert protected[cfg['tokenizer']]==parent['protected_sha256'][cfg['tokenizer']]
    assert protected['scripts/review_selection_pilot.py']==parent['scripts']['review_selection_pilot.py']
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text());assert all(v is None for section in ['review_selection','token_budget'] for v in pending[section].values())
    identity={'config_sha256':file_hash(CONFIG),'protected_sha256':protected,'scripts_sha256':{name:file_hash(ROOT/'scripts'/name) for name in ['led_source_token_audit.py','run_led_source_token_audit.py']}}
    out.mkdir(parents=True,exist_ok=True);write(out/'scientific_identity.json',identity);(out/'config_at_run.yaml').write_bytes(CONFIG.read_bytes())
    rows,business,reviews,info=load_inputs(cfg)
    os.environ['TOKENIZERS_PARALLELISM']='true';os.environ['RAYON_NUM_THREADS']=str(cfg['execution']['tokenizer_threads'])
    tok=Tokenizer.from_file(str(ROOT/cfg['tokenizer']));tok.no_truncation();tok.no_padding()
    body={};ids=sorted(reviews);batch=cfg['execution']['body_batch_size']
    for offset in range(0,len(ids),batch):
        part=ids[offset:offset+batch];encoded=tok.encode_batch([reviews[rid]['text'] for rid in part],add_special_tokens=False)
        for rid,value in zip(part,encoded):body[rid]=len(value.ids)
    print(f'575749 originals streamed; {len(reviews)} selected unique originals BODY-encoded.',flush=True)
    selector_cfg=yaml.safe_load((ROOT/cfg['selector_config']).read_text(encoding='utf-8'));ledger=[];references=[];length_cache={}
    def lengths(texts):
        missing=list(dict.fromkeys(t for t in texts if digest_text(t) not in length_cache))
        batch=cfg['execution']['encode_batch_size']
        for offset in range(0,len(missing),batch):
            part=missing[offset:offset+batch];encoded=tok.encode_batch(part,add_special_tokens=True)
            for text,value in zip(part,encoded):
                assert sum(value.special_tokens_mask)==2
                length_cache[digest_text(text)]=len(value.ids)
        return [length_cache[digest_text(t)] for t in texts]
    for number,((bid,budget),row) in enumerate(sorted(rows.items()),1):
        selected=[reviews[rid] for rid in row['selected_review_ids']]
        assert all(r['business_id']==bid for r in selected) and selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
        block=formatted_input(selected,selector_cfg);block_tokens=lengths([block])[0]
        assert block_tokens==row['tokens_after_budget_enforcement']<=budget
        reference={'business_id':bid,'review_budget':budget,'selected_review_ids':row['selected_review_ids'],'selected_ids_sha256':digest_ids(row['selected_review_ids']),'review_block_sha256':digest_text(block),'method':cfg['accepted_method'],'final_manifest':False}
        references.append(reference)
        metas=[metadata_text(business[bid],fields,cfg) for fields in cfg['metadata_profiles'].values()]
        bases=[source_text(block,meta,cfg) for meta in metas];empty=[empty_caption_source(base,cfg) for base in bases]
        counts=lengths(bases+empty);metadata_counts=[len(e.ids) for e in tok.encode_batch(metas,add_special_tokens=False)]
        for index,profile in enumerate(cfg['metadata_profiles']):
            base_tokens=counts[index];empty_tokens=counts[index+len(metas)]
            ledger.append({'business_id':bid,'name':business[bid]['name'],'split':row['split'],'volume_band':row['volume_band'],'review_budget':budget,'metadata_profile':profile,'retained_reviews':len(selected),'below10_flag':len(selected)<10,'below5_flag':len(selected)<5,'review_BODY_token_sum':sum(body[r['review_id']] for r in selected),'review_block_tokens':block_tokens,'review_format_special_boundary_delta':block_tokens-sum(body[r['review_id']] for r in selected),'metadata_standalone_tokens':metadata_counts[index],'text_only_base_tokens':base_tokens,'marginal_source_overhead':base_tokens-block_tokens,'empty_caption_base_tokens':empty_tokens,'empty_caption_header_increment':empty_tokens-base_tokens,'caption_capacity':cfg['diagnostic_source_ceiling']-base_tokens,'capacity_after_empty_caption_header':cfg['diagnostic_source_ceiling']-empty_tokens,'added_special_tokens':2,'rating_jsd':row['rating_jsd_after'],'temporal_jsd':row['temporal_jsd_after'],'length_jsd':row['length_jsd'],'selected_source_BODY_mean_ratio':row['selected_source_mean_ratio'],'selected_ids_sha256':reference['selected_ids_sha256'],'review_block_sha256':reference['review_block_sha256'],'metadata_sha256':digest_text(metas[index]),'text_base_sha256':digest_text(bases[index]),'empty_caption_base_sha256':digest_text(empty[index]),'actual_captions_available':False,'actual_caption_conditioned_source_tokens':None,'final_model_input':False,'final_manifest':False})
        if number%500==0:print(f'Complete review/source blocks audited {number}/6600.',flush=True)
    frame=pd.DataFrame(ledger);assert len(frame)==19800
    frame.to_csv(out/'source_token_ledger.csv',index=False)
    with (out/'selection_reference.jsonl').open('wb') as stream:
        for row in references:stream.write(orjson.dumps(row)+b'\n')
    summaries(frame,cfg).to_csv(out/'caption_capacity_summary.csv',index=False)
    for dimension in ['split','volume_band']:summaries(frame,cfg,dimension).to_csv(out/f'capacity_by_{dimension}.csv',index=False)
    extremes=[]
    for (profile,budget),part in frame.groupby(['metadata_profile','review_budget']):
        for kind,column,ascending in [('max_metadata_overhead','marginal_source_overhead',False),('min_caption_capacity','caption_capacity',True),('max_caption_capacity','caption_capacity',False)]:
            chosen=part.sort_values([column,'business_id'],ascending=[ascending,True]).iloc[0]
            extremes.append({'extreme':kind,**chosen.to_dict()})
    pd.DataFrame(extremes).to_csv(out/'source_token_extremes.csv',index=False)
    # Limited full examples, no all-business source duplication.
    profile='name_categories_location';part=frame[(frame.metadata_profile==profile)&frame.review_budget.eq(4500)];choices={}
    def choose(bid,budget,reason):
        if bid not in choices:choices[bid]={'budget':budget,'reasons':[]}
        choices[bid]['reasons'].append(reason)
    choose(part.sort_values(['marginal_source_overhead','business_id'],ascending=[False,True]).iloc[0].business_id,4500,'max_metadata_overhead')
    choose(part.sort_values(['caption_capacity','business_id']).iloc[0].business_id,4500,'min_caption_capacity')
    choose(part.sort_values(['caption_capacity','business_id'],ascending=[False,True]).iloc[0].business_id,3500,'max_caption_capacity_low_source_count')
    low=frame[(frame.metadata_profile==profile)&frame.review_budget.eq(3500)&frame.retained_reviews.eq(30)]
    choose(low.sort_values(['review_block_tokens','business_id']).iloc[0].business_id,3500,'minimum_exact30_review_block')
    choose(cfg['example_default_business_id'],3500,'default_notebook_view')
    drops=json.loads((accepted/'v4_2_count_drop_explanations.json').read_text(encoding='utf-8'))
    for drop in drops:choose(drop['business_id'],4000,'existing_primary_count_drop_no_change')
    examples=[]
    for bid,choice in choices.items():
        budget=choice['budget'];row=rows[bid,budget];selected=[reviews[rid] for rid in row['selected_review_ids']]
        block=formatted_input(selected,selector_cfg);meta=metadata_text(business[bid],cfg['metadata_profiles'][profile],cfg);base=source_text(block,meta,cfg)
        entry=frame[(frame.business_id==bid)&frame.review_budget.eq(budget)&frame.metadata_profile.eq(profile)].iloc[0].to_dict()
        examples.append({'reasons':choice['reasons'],**entry,'selected_review_ids':row['selected_review_ids'],'metadata_text':meta,'complete_text_only_source':base,'complete_source_with_EMPTY_caption_header':empty_caption_source(base,cfg),'actual_captions':None,'provisional_template':True})
    write(out/'source_examples.json',examples)
    pd.DataFrame([{k:v for k,v in example.items() if not isinstance(v,(dict,list)) and k not in ['metadata_text','complete_text_only_source','complete_source_with_EMPTY_caption_header']} for example in examples]).to_csv(out/'example_businesses.csv',index=False)
    for name,digest in protected.items():assert file_hash(ROOT/name)==digest,name
    scientific_files=[p for p in out.iterdir() if p.is_file()];output_bytes=sum(p.stat().st_size for p in scientific_files);assert output_bytes<=cfg['storage']['maximum_scientific_output_bytes'],output_bytes
    report={'status':'PASS','businesses':2200,'splits':{'train':2000,'validation':200},'source_English_originals_streamed':575749,'selected_unique_originals_BODY_encoded':len(reviews),'complete_review_blocks':6600,'metadata_profiles':3,'source_ledger_rows':19800,'caption_summary_rows':63,'unique_full_strings_encoded':len(length_cache),'examples':len(examples),'actual_captions_available':False,'review_ids_text_order_counts_primary_metrics_unchanged':True,'test_businesses':0,'diagnostic_source_ceiling':5000,'final_manifest':False,'final_source_tokens':None,'final_review_tokens':None,'final_caption_tokens':None,'final_metadata_fields':None,'methodology_locked':False,'scientific_output_bytes':output_bytes,'elapsed_seconds':round(time.perf_counter()-started,3),'artifact_sha256':{p.name:file_hash(p) for p in scientific_files}}
    write(out/'completion_report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
