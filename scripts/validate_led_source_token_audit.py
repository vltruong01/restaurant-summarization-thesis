"""Independent serialization/token/capacity checks from original selected reviews."""
import hashlib,json,math,os,time
import numpy as np,pandas as pd,orjson,yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from led_source_token_audit import ROOT,CONFIG,load_inputs


def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    started=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    completion=json.loads((out/'completion_report.json').read_text());assert completion['status']=='PASS'
    for name,digest in completion['artifact_sha256'].items():assert file_hash(out/name)==digest,name
    identity=json.loads((out/'scientific_identity.json').read_text());assert file_hash(CONFIG)==identity['config_sha256']
    for name,digest in identity['scripts_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest,name
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    rows,business,reviews,info=load_inputs(cfg)
    frame=pd.read_csv(out/'source_token_ledger.csv',keep_default_na=False);assert len(frame)==19800
    assert frame[['business_id','review_budget','metadata_profile']].duplicated().sum()==0
    assert frame.actual_captions_available.eq(False).all() and frame.actual_caption_conditioned_source_tokens.eq('').all()
    assert frame.final_model_input.eq(False).all() and frame.final_manifest.eq(False).all()
    os.environ['TOKENIZERS_PARALLELISM']='true';os.environ['RAYON_NUM_THREADS']='8'
    tok=Tokenizer.from_file(str(ROOT/cfg['tokenizer']));tok.no_truncation();tok.no_padding()
    body={};ids=sorted(reviews)
    for offset in range(0,len(ids),1024):
        part=ids[offset:offset+1024]
        for rid,encoded in zip(part,tok.encode_batch([reviews[rid]['text'] for rid in part],add_special_tokens=False)):body[rid]=len(encoded.ids)
    lengths={};special_checks=0
    def encode(texts):
        nonlocal special_checks
        missing=list(dict.fromkeys(text for text in texts if sha(text) not in lengths))
        for start in range(0,len(missing),64):
            part=missing[start:start+64]
            for text,encoding in zip(part,tok.encode_batch(part,add_special_tokens=True)):
                assert sum(encoding.special_tokens_mask)==2
                lengths[sha(text)]=len(encoding.ids);special_checks+=1
        return [lengths[sha(text)] for text in texts]
    references={(r['business_id'],r['review_budget']):r for r in map(orjson.loads,(out/'selection_reference.jsonl').open('rb'))}
    assert set(references)==set(rows)
    examples=json.loads((out/'source_examples.json').read_text(encoding='utf-8'))
    example_keys={(r['business_id'],r['review_budget'],r['metadata_profile']) for r in examples}
    indexed=frame.set_index(['business_id','review_budget','metadata_profile']);source_cache={};checked=0
    for number,((bid,budget),old) in enumerate(sorted(rows.items()),1):
        selected=[reviews[rid] for rid in old['selected_review_ids']]
        assert all(r['business_id']==bid for r in selected) and selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
        block='\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i,r in enumerate(selected,1))
        block_tokens=encode([block])[0];assert block_tokens==old['tokens_after_budget_enforcement']<=budget
        ids_digest=hashlib.sha256(orjson.dumps(old['selected_review_ids'])).hexdigest();reference=references[bid,budget]
        assert reference['selected_review_ids']==old['selected_review_ids'] and reference['selected_ids_sha256']==ids_digest and reference['review_block_sha256']==sha(block) and not reference['final_manifest']
        metas=[];bases=[];empties=[]
        for profile,fields in cfg['metadata_profiles'].items():
            lines=[]
            for field in fields:
                value=business[bid].get(field);value='' if value is None else value;assert isinstance(value,str)
                lines.append(field+'='+value)
            meta='[BUSINESS]\n'+'\n'.join(lines) if fields else ''
            base=(meta+'\n\n' if meta else '')+'[REVIEWS]\n'+block
            metas.append(meta);bases.append(base);empties.append(base+'\n\n[CAPTIONS]\n')
        counts=encode(bases+empties);standalone=[len(e.ids) for e in tok.encode_batch(metas,add_special_tokens=False)]
        for index,profile in enumerate(cfg['metadata_profiles']):
            row=indexed.loc[bid,budget,profile];base_tokens=counts[index];empty_tokens=counts[index+3]
            exact={'retained_reviews':len(selected),'below10_flag':len(selected)<10,'below5_flag':len(selected)<5,'review_BODY_token_sum':sum(body[r['review_id']] for r in selected),'review_block_tokens':block_tokens,'review_format_special_boundary_delta':block_tokens-sum(body[r['review_id']] for r in selected),'metadata_standalone_tokens':standalone[index],'text_only_base_tokens':base_tokens,'marginal_source_overhead':base_tokens-block_tokens,'empty_caption_base_tokens':empty_tokens,'empty_caption_header_increment':empty_tokens-base_tokens,'caption_capacity':5000-base_tokens,'capacity_after_empty_caption_header':5000-empty_tokens,'added_special_tokens':2,'selected_ids_sha256':ids_digest,'review_block_sha256':sha(block),'metadata_sha256':sha(metas[index]),'text_base_sha256':sha(bases[index]),'empty_caption_base_sha256':sha(empties[index]),'split':old['split'],'name':business[bid]['name']}
            for field,value in exact.items():assert row[field]==value,(bid,budget,profile,field)
            for field,parent_field in [('rating_jsd','rating_jsd_after'),('temporal_jsd','temporal_jsd_after'),('length_jsd','length_jsd'),('selected_source_BODY_mean_ratio','selected_source_mean_ratio')]:
                assert math.isclose(row[field],old[parent_field],abs_tol=1e-12),(bid,budget,profile,field)
            assert bases[index].endswith(block) and all(r['text'] in bases[index] for r in selected)
            if (bid,budget,profile) in example_keys:source_cache[bid,budget,profile]=(metas[index],bases[index],empties[index])
            checked+=1
        if number%1000==0:print(f'Independent source rows {checked}/19800.',flush=True)
    summary_checks=0
    for filename,extra in [('caption_capacity_summary.csv',None),('capacity_by_split.csv','split'),('capacity_by_volume_band.csv','volume_band')]:
        summary=pd.read_csv(out/filename,keep_default_na=False);expected_rows=63*(1 if extra is None else frame[extra].nunique());assert len(summary)==expected_rows
        for _,row in summary.iterrows():
            g=frame[(frame.metadata_profile==row.metadata_profile)&frame.review_budget.eq(row.review_budget)]
            if extra:g=g[g[extra].eq(row[extra])]
            allowance=int(row.caption_allowance);fit=(g.text_only_base_tokens+allowance<=5000);bound=(g.review_budget+g.marginal_source_overhead+allowance<=5000)
            assert row.businesses==len(g) and row.observed_fit==fit.sum() and row.observed_overflow==(~fit).sum()
            assert row.conservative_envelope_fit==bound.sum() and row.conservative_envelope_overflow==(~bound).sum()
            assert math.isclose(row.observed_fit_pct,fit.mean()*100,abs_tol=1e-10)
            assert row.observed_allocated_total_max==(g.text_only_base_tokens+allowance).max()
            assert row.conservative_envelope_total_max==(g.review_budget+g.marginal_source_overhead+allowance).max()
            scalar={'retained_mean':g.retained_reviews.mean(),'retained_min':g.retained_reviews.min(),'retained_max':g.retained_reviews.max(),'below10':g.below10_flag.sum(),'below5':g.below5_flag.sum(),'source_only_overflows':g.text_only_base_tokens.gt(5000).sum(),'metadata_no_special_min':g.metadata_standalone_tokens.min(),'metadata_no_special_mean':g.metadata_standalone_tokens.mean(),'metadata_no_special_max':g.metadata_standalone_tokens.max(),'marginal_overhead_min':g.marginal_source_overhead.min(),'marginal_overhead_median':g.marginal_source_overhead.median(),'marginal_overhead_p95':g.marginal_source_overhead.quantile(.95),'marginal_overhead_max':g.marginal_source_overhead.max(),'exact_text_base_min':g.text_only_base_tokens.min(),'exact_text_base_median':g.text_only_base_tokens.median(),'exact_text_base_p95':g.text_only_base_tokens.quantile(.95),'exact_text_base_max':g.text_only_base_tokens.max(),'caption_capacity_min':g.caption_capacity.min(),'caption_capacity_p05':g.caption_capacity.quantile(.05),'caption_capacity_median':g.caption_capacity.median(),'caption_capacity_max':g.caption_capacity.max(),'review_tokens_min':g.review_block_tokens.min(),'review_tokens_max':g.review_block_tokens.max(),'empty_caption_header_increment_min':g.empty_caption_header_increment.min(),'empty_caption_header_increment_max':g.empty_caption_header_increment.max(),'rating_jsd_mean':g.rating_jsd.mean(),'temporal_jsd_mean':g.temporal_jsd.mean(),'length_jsd_mean':g.length_jsd.mean()}
            for field,value in scalar.items():assert math.isclose(row[field],value,abs_tol=1e-10),(filename,field)
            assert not row.actual_captions_available and row.actual_caption_conditioned_tokens=='' and not row.final_manifest
            summary_checks+=1
    for example in examples:
        key=(example['business_id'],example['review_budget'],example['metadata_profile']);meta,base,empty=source_cache[key]
        assert example['metadata_text']==meta and example['complete_text_only_source']==base and example['complete_source_with_EMPTY_caption_header']==empty
        assert example['actual_captions'] is None and example['provisional_template'] and not example['final_model_input']
    pending=yaml.safe_load((ROOT/'configs/research_pending.yaml').read_text());assert all(v is None for section in ['review_selection','token_budget'] for v in pending[section].values())
    for name,digest in identity['protected_sha256'].items():assert file_hash(ROOT/name)==digest,name
    report={'validation_passed':True,'source_rows_verified':checked,'review_blocks_verified':6600,'selected_originals_BODY_fresh_encoded':len(body),'unique_full_strings_fresh_encoded':len(lengths),'added_special_token_pairs_verified':special_checks,'summary_rows_verified':summary_checks,'examples_complete_source_equal':len(examples),'whole_review_ids_text_order_primary_metrics_unchanged':True,'actual_captions_available':False,'actual_caption_conditioned_source_tokens':None,'test_businesses':0,'final_manifest':False,'final_parameters_unchanged':True,'source_and_accepted_inputs_unchanged':True,'validator_sha256':file_hash(__file__),'elapsed_seconds':round(time.perf_counter()-started,3)}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
