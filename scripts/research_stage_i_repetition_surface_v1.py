"""Read saved validation predictions only; preserve metrics, decompose flag types."""
import hashlib,json,re,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from stage_i_decoding_diagnostics import text_diagnostics
OUT=ROOT/'outputs/stage_i/repetition_research_20261019_v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'outputs/stage_i/d060_result_report_v1/COMPARISON_RECORDS.json'
    rows=json.loads(p.read_text(encoding='utf-8'));flags=[];groups=defaultdict(list)
    for r in rows:
        text=re.sub(r'\[[A-Z_]+\]','',r['text'])
        ws=list(re.finditer(r"[a-z0-9]+(?:'[a-z]+)?",text.lower()))
        pairs=[]
        for a,b in zip(ws,ws[1:]):
            if a.group()!=b.group():continue
            sep=text[a.end():b.start()]
            pairs.append({'word':a.group(),'separator':sep,'whitespace_only':bool(sep) and sep.isspace(),
                'sentence_terminator_between':bool(re.search(r'[.!?]',sep)),
                'context':text[max(0,a.start()-55):min(len(text),b.end()+70)]})
        legacy=text_diagnostics(r['text'])
        assert legacy['consecutive_repeated_word_pairs']==len(pairs)
        record={'business_id':r['business_id'],'arm':r['arm'],'subset':r['subset'],
            'prediction_text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),
            'historical_metrics_unchanged':legacy,'adjacent_lexical_pairs':pairs,
            'whitespace_pair_present':any(x['whitespace_only'] for x in pairs),
            'no_sentence_terminator_pair_present':any(not x['sentence_terminator_between'] for x in pairs),
            'classification_is_surface_only_not_error_verdict':True}
        flags.append(record);groups[(r['arm'],r['subset'])].append(record)
    summary=[]
    for (arm,subset),rs in groups.items():
        summary.append({'arm':arm,'subset':subset,'n':len(rs),
            'published_exact_sentence_flag_outputs':sum(x['historical_metrics_unchanged']['exact_repeated_sentence'] for x in rs),
            'published_adjacent_lexical_flag_outputs':sum(bool(x['adjacent_lexical_pairs']) for x in rs),
            'outputs_with_pair_without_sentence_terminator':sum(x['no_sentence_terminator_pair_present'] for x in rs),
            'outputs_with_whitespace_only_pair':sum(x['whitespace_pair_present'] for x in rs),
            'flags_are_not_automatic_factual_or_repetition_error_judgments':True})
    old=json.loads((ROOT/'outputs/stage_i/d060_result_report_v1/SOFTWARE_METRICS_BY_SUBSET.json').read_text())
    for x in summary:
        expected=next(y for y in old if y['arm']==x['arm'] and y['subset']==x['subset'])
        assert x['published_exact_sentence_flag_outputs']==expected['exact_repeated_sentence_outputs']
        assert x['published_adjacent_lexical_flag_outputs']==expected['adjacent_word_repetition_outputs']
    for name,obj in [('SURFACE_FLAG_DECOMPOSITION.json',flags),('SUMMARY_UNCHANGED_METRICS.json',summary)]:
        with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    protected=[p,ROOT/'scripts/stage_i_decoding_diagnostics.py',ROOT/'outputs/stage_i/d060_result_report_v1/SOFTWARE_METRICS_BY_SUBSET.json',ROOT/'outputs/stage_i/stage_i_diagnostic_B_chatgpt_bundle_v1/results/DIAGNOSTIC_REPORT.md',ROOT/'outputs/stage_i/stage_i_diagnostic_B_chatgpt_bundle_v1/results/HEADER_CONSTRAINT_AUDIT.json',ROOT/'outputs/stage_i/d055_root_cause_v2/TEACHER_SURFACE_DIAGNOSTICS.json']
    proof={'files':{q.relative_to(ROOT).as_posix():sha(q) for q in protected},'prediction_records_read':len(rows),
        'model_load_forward_backward_or_generation_calls':0,'GPU_operations':0,'Teacher_calls':0,'held_out_contents_opened':False,'published_metrics_reproduced':True}
    with (OUT/'READ_ONLY_AUDIT_PROVENANCE.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps({'status':'OFFLINE_SAVED_OUTPUT_AUDIT_PASS','records':len(rows),'groups':len(summary),'old_metrics_unchanged':True}))

if __name__=='__main__':main()
