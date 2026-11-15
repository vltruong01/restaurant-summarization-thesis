"""Versioned C2 budget pilot; originals, source split, final methodology remain fixed."""
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import copy
import hashlib
import heapq
import html
import json
import sqlite3
import time
import orjson
import pandas as pd
import yaml
from tokenizers import Tokenizer
from english_eligibility import file_hash
from review_selection_pilot import ordered,select_initial,adjust_budget,diagnostics,formatted_input,token_count
from budget_aware_review_selection import search_population

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sample_businesses(root,cfg):
    c1=root/cfg['c1_directory']; prior=root/cfg['original_pilot_directory']
    base=pd.read_csv(prior/'pilot_businesses.csv',keep_default_na=False)
    all_stats=pd.read_csv(c1/'business_review_statistics.csv',keep_default_na=False)
    allowed=all_stats.loc[all_stats['split'].isin(cfg['allowed_splits'])].copy()
    heaps=defaultdict(list)
    db_path=(c1/'review_metadata.sqlite').resolve()
    db=sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro',uri=True)
    for bid,length in db.execute("SELECT business_id,led_tokens_without_special FROM reviews WHERE language='en' AND split IN ('train','validation')"):
        heap=heaps[bid]
        if len(heap)<8: heapq.heappush(heap,length)
        elif length>heap[0]: heapq.heapreplace(heap,length)
    db.close()
    allowed['top8_individual_base_tokens']=allowed['business_id'].map(lambda b:sum(heaps[b]))
    eligible=allowed.loc[allowed['english_reviews'].le(cfg['pilot']['low_volume_maximum_reviews']) & ~allowed['business_id'].isin(base['business_id'])]
    added=eligible.sort_values(['top8_individual_base_tokens','business_id'],ascending=[False,True]).head(cfg['pilot']['additional_low_volume_top8_length_cases']).copy()
    assert len(added)==8
    added['volume_band']=pd.cut(added['english_reviews'],[9,30,100,300,float('inf')],labels=['10-30','31-100','101-300','301+']).astype(str)
    added['pilot_reasons']='additional_low_volume_top8_length_stress'
    base['sample_group']='original_48'; added['sample_group']='additional_long_review_8'
    base['top8_individual_base_tokens']=base['business_id'].map(lambda b:sum(heaps[b]))
    result=pd.concat([base,added],ignore_index=True).fillna('')
    assert result['business_id'].is_unique and set(result['split'])<={'train','validation'}
    return result


def reference_cfg(cfg):
    value=copy.deepcopy(cfg); value['algorithm_version']='rating_time_quantile_draft_v1'
    return value


def run_business(bid,population,cfg,tokenizer,info):
    ref_cfg=reference_cfg(cfg)
    refs={m:select_initial(population,m,ref_cfg,bid) for m in ['random','rating_time']}
    baselines={}; extra=[]
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        for method in refs:
            selected,trace,feasible=adjust_budget(population,refs[method],budget,cfg,tokenizer)
            baselines[(method,budget)]=(selected if feasible else [],trace,feasible)
            if feasible: extra.append((f'{method}_delete_only_{budget}',selected))
    ev,choices,frontiers,sensitivity,stats=search_population(population,cfg,tokenizer,bid,extra)
    runs=[]; previews=[]
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        for method in ['random_delete_only','rating_time_delete_only','budget_aware']:
            ref_method='random' if method=='random_delete_only' else 'rating_time'
            initial=refs[ref_method]; before=diagnostics(population,initial,cfg)
            if method=='budget_aware':
                chosen=choices[budget]; selected=[ev.by_id[rid] for rid in chosen['ids']] if chosen else []
                path=chosen['path'] if chosen else []; provenance={'origin':chosen['origin'] if chosen else None,'accepted_local_path':path}
            else:
                selected,trace,feasible=baselines[(ref_method,budget)]
                provenance={'delete_only_trace':trace}
            after=diagnostics(population,selected,cfg) if selected else None
            final_ids=[r['review_id'] for r in selected]; initial_ids=[r['review_id'] for r in initial]
            initial_set=set(initial_ids); final_set=set(final_ids)
            row={'business_id':bid,'name':info['name'],'split':info['split'],'volume_band':info['volume_band'],'sample_group':info['sample_group'],
                 'method':method,'review_budget':budget,'initial_eligible_review_count':len(population),'candidate_pool_count':len(population),
                 'initial_candidate_count':len(initial),'final_retained_review_count':len(selected),
                 'tokens_before_budget_enforcement':token_count(initial,cfg,tokenizer),
                 'tokens_after_budget_enforcement':token_count(selected,cfg,tokenizer) if selected else None,
                 'rating_jsd_before':before['rating_jsd'],'temporal_jsd_before':before['temporal_jsd'],
                 'rating_jsd_after':after['rating_jsd'] if after else None,'temporal_jsd_after':after['temporal_jsd'] if after else None,
                 'date_span_coverage_after':after['date_span_coverage'] if after else None,
                 'below_10_retained_flag':len(selected)<10,'below_5_retained_flag':len(selected)<5,
                 'selection_status':'feasible' if selected else 'budget_infeasible',
                 'selected_review_ids':final_ids,'initial_reference_review_ids':initial_ids,
                 'removed_vs_reference_ids':[rid for rid in initial_ids if rid not in final_set],
                 'added_vs_reference_ids':[rid for rid in final_ids if rid not in initial_set],
                 'search_provenance':provenance,'metrics_before':before,'metrics_after':after,
                 'semantic_centroid_similarity':None,'semantic_status':'not_computed_model_TBD',
                 'final_review_budget':None,'methodology_locked':False,'seed':cfg['seed']}
            runs.append(row)
            if selected:
                previews.append({'business_id':bid,'method':method,'review_budget':budget,'selected_review_ids':final_ids,
                                 'input_text':formatted_input(selected,cfg),'led_tokens_with_special':row['tokens_after_budget_enforcement'],'final_model_input':False})
    longest=sorted(population,key=lambda r:(-ev.length_cost[r['review_id']],r['review_id']))[:8]
    stress=[]
    for count in (7,8):
        items=ordered(longest[:count])
        stress.append({'business_id':bid,'count':len(items),'selected_review_ids':[r['review_id'] for r in items],
                       'exact_tokens':token_count(items,cfg,tokenizer),'purpose':'deliberate_long_review_diagnostic_not_selected_input'})
    return {'business_id':bid,'runs':runs,'previews':previews,'sensitivity':sensitivity,
            'frontiers':{str(b):v for b,v in frontiers.items()},'candidate_audit':ev.audit,'search_statistics':stats,'long_review_stress':stress}


def save_details(out,sample,populations,runs,cfg):
    new=[r for r in runs if r['method']=='budget_aware' and r['review_budget']==3500]
    ranked=sorted(new,key=lambda r:(r['final_retained_review_count'],-r['tokens_before_budget_enforcement'],r['business_id']))
    ids=[r['business_id'] for r in ranked[:4]]
    extra=sample.loc[sample['sample_group'].eq('additional_long_review_8')].sort_values(['top8_individual_base_tokens','business_id'],ascending=[False,True])
    for bid in extra['business_id']:
        if bid not in ids: ids.append(bid)
        if len(ids)>=7: break
    for r in sorted(new,key=lambda r:(-r['tokens_before_budget_enforcement'],r['business_id'])):
        if r['business_id'] not in ids: ids.append(r['business_id'])
        if len(ids)>=cfg['pilot']['detail_examples']: break
    parts=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Budget pilot review examples</title>',
           '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:8px;margin:5px 0}.selected{background:#e9f5ff}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}</style>',
           '<h1>Complete-review budget pilot — chưa khóa methodology</h1><p>10 real businesses; toàn bộ original English text có thể mở. Review màu xanh được giữ ở budget-aware 3500-token scenario. Semantic/manual aspect audit chưa thực hiện. Các phương pháp/budgets là exploratory.</p>']
    lookup=sample.set_index('business_id').to_dict('index')
    for bid in ids:
        info=lookup[bid]; rows=[r for r in runs if r['business_id']==bid]
        current=next(r for r in rows if r['method']=='budget_aware' and r['review_budget']==3500)
        kept=set(current['selected_review_ids']); initial=set(current['initial_reference_review_ids'])
        parts.append(f'<h2>{html.escape(info["name"])} — {html.escape(bid)}</h2><p>{info["split"]}; {len(populations[bid])} English reviews; {html.escape(info["sample_group"])}</p>')
        cols=['method','review_budget','final_retained_review_count','tokens_after_budget_enforcement','rating_jsd_after','temporal_jsd_after','below_10_retained_flag']
        parts.append(pd.DataFrame(rows)[cols].to_html(index=False,escape=True,float_format=lambda v:f'{v:.6f}'))
        parts.append('<details><summary>Search provenance / reference differences</summary><pre>'+html.escape(json.dumps({k:current[k] for k in ['removed_vs_reference_ids','added_vs_reference_ids','search_provenance']},ensure_ascii=False,indent=2))+'</pre></details>')
        for review in ordered(populations[bid]):
            rid=review['review_id']; label=('KEPT' if rid in kept else 'not kept')+(' | initial reference' if rid in initial else '')
            css=' class="selected"' if rid in kept else ''
            parts.append(f'<details{css}><summary>{label} | {int(review["stars"])} stars | {html.escape(review["date"])} | {html.escape(rid)}</summary><pre>{html.escape(review["text"])}</pre></details>')
    parts.append('</html>')
    (out/'manual_examples.html').write_text('\n'.join(parts),encoding='utf-8'); write_json(out/'detail_example_business_ids.json',ids)


def run_pilot(root=ROOT):
    root=Path(root); started=time.perf_counter()
    path=root/'configs/review_selection_budget_pilot.yaml'; cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert cfg['allowed_splits']==['train','validation'] and not cfg['final_methodology_locked'] and cfg['semantic']['model_id'] is None
    source=root/cfg['english_pool_directory']; c1=root/cfg['c1_directory']; out=root/cfg['output_directory']
    c1_completion=json.loads((c1/'completion_report.json').read_text(encoding='utf-8'))
    assert json.loads((c1/'validation_report.json').read_text())['validation_passed']
    for name in ('business_review_statistics.csv','scientific_identity.json'):
        assert file_hash(c1/name)==c1_completion['diagnostic_artifact_sha256'][name]
    source_hashes=json.loads((source/'checksums.json').read_text(encoding='utf-8'))
    originals=['reviews/train.jsonl','reviews/validation.jsonl','business_split.csv']
    before={n:((source/n).stat().st_size,(source/n).stat().st_mtime_ns) for n in originals}
    for name in originals: assert file_hash(source/name)==source_hashes[name]
    token_dir=root/'assets/tokenizers/led-base-16384'
    token_manifest=json.loads((root/'assets/tokenizers/led-base-16384.manifest.json').read_text())
    for name,digest in token_manifest['sha256'].items(): assert file_hash(token_dir/name)==digest
    identity={'config_sha256':file_hash(path),'c1_identity':c1_completion['identity_sha256'],
              'runtime_versions':{n:version(n) for n in ['tokenizers','pandas','numpy','orjson']},
              'sources':{n:source_hashes[n] for n in originals},'tokenizer_manifest':token_manifest,
              'scripts':{n:file_hash(root/'scripts'/n) for n in ['run_budget_review_pilot.py','budget_aware_review_selection.py','review_selection_pilot.py']}}
    identity_hash=hashlib.sha256(orjson.dumps(identity,option=orjson.OPT_SORT_KEYS)).hexdigest()
    if (out/'completion_report.json').exists():
        old=json.loads((out/'completion_report.json').read_text())
        if old['identity_sha256']!=identity_hash: raise RuntimeError('Output identity changed; version directory')
        for name,digest in old['artifact_sha256'].items(): assert file_hash(out/name)==digest,name
        print('Using completed exploratory budget pilot.',flush=True); return old
    out.mkdir(parents=True,exist_ok=True)
    identity_path=out/'scientific_identity.json'
    if identity_path.exists(): assert json.loads(identity_path.read_text())['identity_sha256']==identity_hash
    write_json(identity_path,{'identity_sha256':identity_hash,'identity':identity})
    (out/'config_at_run.yaml').write_bytes(path.read_bytes())
    sample=sample_businesses(root,cfg); sample.to_csv(out/'pilot_businesses.csv',index=False)
    membership=sample.set_index('business_id')['split'].to_dict(); info=sample.set_index('business_id').to_dict('index')
    populations={bid:[] for bid in membership}
    for split in cfg['allowed_splits']:
        with (source/'reviews'/f'{split}.jsonl').open('rb') as stream:
            for line in stream:
                row=orjson.loads(line); bid=row['business_id']
                if bid in membership:
                    assert membership[bid]==split; populations[bid].append(row)
    with (out/'pilot_population.jsonl').open('wb') as stream:
        for bid,pool in populations.items():
            assert len(pool)==int(info[bid]['english_reviews'])
            for review in ordered(pool): stream.write(orjson.dumps({'split':membership[bid],'review':review})+b'\n')
    tokenizer=Tokenizer.from_file(str(token_dir/'tokenizer.json')); tokenizer.no_truncation(); tokenizer.no_padding()
    cache=out/'business_runs'; cache.mkdir(exist_ok=True)
    payloads=[]
    for index,(bid,pool) in enumerate(populations.items(),1):
        cached=cache/f'{hashlib.sha256(bid.encode()).hexdigest()}.json'
        if cached.exists(): payload=json.loads(cached.read_text(encoding='utf-8'))
        else:
            payload=run_business(bid,pool,cfg,tokenizer,info[bid]); write_json(cached,payload)
        payloads.append(payload)
        print(f'{index}/{len(sample)} {bid}: budget-aware counts '+str([r['final_retained_review_count'] for r in payload['runs'] if r['method']=='budget_aware']),flush=True)
        write_json(out/'run_progress.json',{'completed_businesses':index,'expected_businesses':len(sample),'elapsed_seconds':round(time.perf_counter()-started,1)})
    runs=[r for p in payloads for r in p['runs']]
    previews=[r for p in payloads for r in p['previews']]
    sensitivity=[dict(business_id=p['business_id'],**r) for p in payloads for r in p['sensitivity']]
    stress=[r for p in payloads for r in p['long_review_stress']]
    for name,rows in [('selection_runs.jsonl',runs),('model_input_previews.jsonl',previews),('long_review_stress.jsonl',stress)]:
        with (out/name).open('wb') as stream:
            for row in rows: stream.write(orjson.dumps(row)+b'\n')
    frame=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in runs])
    frame['retained_count_band']=pd.cut(frame['final_retained_review_count'],[-1,4,9,14,19,24,30],labels=['<5','5-9','10-14','15-19','20-24','25-30']).astype(str)
    frame.to_csv(out/'selection_diagnostics.csv',index=False)
    pd.DataFrame(sensitivity).to_json(out/'tolerance_sensitivity.json',orient='records',indent=2)
    summaries=[]
    for (method,budget),group in frame.groupby(['method','review_budget']):
        summaries.append({'method':method,'review_budget':int(budget),'businesses':len(group),
            'below_10':int(group['below_10_retained_flag'].sum()),'below_5':int(group['below_5_retained_flag'].sum()),
            'infeasible':int(group['selection_status'].eq('budget_infeasible').sum()),
            'retained_count_mean':float(group['final_retained_review_count'].mean()),'retained_count_min':int(group['final_retained_review_count'].min()),
            'rating_jsd_before_mean':float(group['rating_jsd_before'].mean()),'rating_jsd_after_mean':float(group['rating_jsd_after'].mean()),
            'temporal_jsd_before_mean':float(group['temporal_jsd_before'].mean()),'temporal_jsd_after_mean':float(group['temporal_jsd_after'].mean()),
            **{f'band_{band}':int(group['retained_count_band'].eq(band).sum()) for band in ['25-30','20-24','15-19','10-14','5-9','<5']}})
    pd.DataFrame(summaries).to_csv(out/'budget_method_summary.csv',index=False)
    save_details(out,sample,populations,runs,cfg)
    assert before=={n:((source/n).stat().st_size,(source/n).stat().st_mtime_ns) for n in originals}
    artifacts={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name not in ['completion_report.json','run_progress.json']}
    report={'identity_sha256':identity_hash,'businesses':len(sample),'original_48_businesses':48,'additional_long_review_businesses':8,
            'population_reviews':sum(map(len,populations.values())),'selection_runs':len(runs),'test_businesses':0,
            'exploratory_review_budgets':cfg['budget']['exploratory_review_token_scenarios'],'final_review_budget':None,
            'semantic_status':'not_computed_model_TBD','methodology_locked':False,'source_metadata_unchanged':True,
            'artifact_sha256':artifacts,'elapsed_seconds':round(time.perf_counter()-started,3)}
    write_json(out/'completion_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'},indent=2),flush=True)
    return report


if __name__=='__main__': run_pilot()
