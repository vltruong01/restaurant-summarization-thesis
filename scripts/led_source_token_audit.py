"""Provisional source serialization and compact allocation summaries, no selection."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,orjson,yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/led_source_token_allocation_audit.yaml'


def digest_text(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def digest_ids(ids):return hashlib.sha256(orjson.dumps(ids)).hexdigest()


def load_inputs(cfg):
    accepted=ROOT/cfg['accepted_directory'];sample=pd.read_csv(accepted/'businesses.csv',keep_default_na=False)
    assert len(sample)==2200 and sample.split.value_counts().to_dict()=={'train':2000,'validation':200}
    info=sample.set_index('business_id').to_dict('index');rows={};wanted=set()
    for line in (accepted/'selection_runs.jsonl').open('rb'):
        row=orjson.loads(line)
        if row['method']!=cfg['accepted_method']:continue
        key=(row['business_id'],row['review_budget']);assert key not in rows
        assert row['split'] in cfg['allowed_splits'] and row['business_id'] in info
        assert row['review_budget'] in cfg['review_budgets'] and not row['final_model_input'] and not row['methodology_locked'] and row['final_review_budget'] is None
        ids=row['selected_review_ids'];assert 0<len(ids)<=30 and len(ids)==len(set(ids))==row['final_retained_review_count']
        rows[key]=row;wanted.update(ids)
    assert len(rows)==6600 and all((bid,b) in rows for bid in info for b in cfg['review_budgets'])
    business={}
    for line in (ROOT/cfg['business_records']).open('rb'):
        record=orjson.loads(line)
        if record['business_id'] in info:business[record['business_id']]=record
    assert set(business)==set(info)
    reviews={};source_count=0
    for split in cfg['allowed_splits']:
        for line in (ROOT/cfg['english_directory']/'reviews'/f'{split}.jsonl').open('rb'):
            record=orjson.loads(line);source_count+=1
            assert record['business_id'] in info and info[record['business_id']]['split']==split
            if record['review_id'] in wanted:
                assert record['review_id'] not in reviews
                reviews[record['review_id']]=record
    assert source_count==575749 and set(reviews)==wanted
    return rows,business,reviews,info


def metadata_text(business,fields,cfg):
    if not fields:return ''
    values=[]
    for field in fields:
        value=business.get(field);value='' if value is None else value
        assert isinstance(value,str),field
        values.append(f'{field}={value}')
    return cfg['format']['business_marker']+'\n'+'\n'.join(values)


def source_text(review_block,metadata,cfg):
    return (metadata+'\n\n' if metadata else '')+cfg['format']['reviews_marker']+'\n'+review_block


def empty_caption_source(base,cfg):return base+'\n\n'+cfg['format']['captions_marker']+'\n'


def summaries(ledger,cfg,extra=None):
    columns=['metadata_profile','review_budget']+([extra] if extra else [])
    results=[];ceiling=cfg['diagnostic_source_ceiling']
    for keys,g in ledger.groupby(columns,dropna=False):
        if not isinstance(keys,tuple):keys=(keys,)
        base={**dict(zip(columns,keys)),'businesses':len(g),'retained_mean':float(g.retained_reviews.mean()),'retained_min':int(g.retained_reviews.min()),'retained_max':int(g.retained_reviews.max()),'below10':int(g.below10_flag.sum()),'below5':int(g.below5_flag.sum()),'review_tokens_min':int(g.review_block_tokens.min()),'review_tokens_max':int(g.review_block_tokens.max()),'metadata_no_special_min':int(g.metadata_standalone_tokens.min()),'metadata_no_special_mean':float(g.metadata_standalone_tokens.mean()),'metadata_no_special_max':int(g.metadata_standalone_tokens.max()),'marginal_overhead_min':int(g.marginal_source_overhead.min()),'marginal_overhead_median':float(g.marginal_source_overhead.median()),'marginal_overhead_p95':float(g.marginal_source_overhead.quantile(.95)),'marginal_overhead_max':int(g.marginal_source_overhead.max()),'exact_text_base_min':int(g.text_only_base_tokens.min()),'exact_text_base_median':float(g.text_only_base_tokens.median()),'exact_text_base_p95':float(g.text_only_base_tokens.quantile(.95)),'exact_text_base_max':int(g.text_only_base_tokens.max()),'source_only_overflows':int(g.text_only_base_tokens.gt(ceiling).sum()),'caption_capacity_min':int(g.caption_capacity.min()),'caption_capacity_p05':float(g.caption_capacity.quantile(.05)),'caption_capacity_median':float(g.caption_capacity.median()),'caption_capacity_max':int(g.caption_capacity.max()),'empty_caption_header_increment_min':int(g.empty_caption_header_increment.min()),'empty_caption_header_increment_max':int(g.empty_caption_header_increment.max()),'rating_jsd_mean':float(g.rating_jsd.mean()),'temporal_jsd_mean':float(g.temporal_jsd.mean()),'length_jsd_mean':float(g.length_jsd.mean())}
        for reserve in cfg['caption_allowances_exploratory']:
            observed=g.caption_capacity.ge(reserve)
            envelope=(g.review_budget+g.marginal_source_overhead+reserve).le(ceiling)
            results.append({**base,'caption_allowance':reserve,'caption_condition':'no_captions_text_only_control' if reserve==0 else 'hypothetical_entire_caption_block_increment','observed_fit':int(observed.sum()),'observed_overflow':int((~observed).sum()),'observed_fit_pct':float(observed.mean()*100),'conservative_envelope_fit':int(envelope.sum()),'conservative_envelope_overflow':int((~envelope).sum()),'observed_allocated_total_max':int((g.text_only_base_tokens+reserve).max()),'conservative_envelope_total_max':int((g.review_budget+g.marginal_source_overhead+reserve).max()),'actual_captions_available':False,'actual_caption_conditioned_tokens':None,'final_manifest':False})
    return pd.DataFrame(results)
