"""Verify an extracted D055 review bundle; no torch/models/network required.

Requires tokenizers0.21.1 for exact saved-ID encoding/decoding checks.
"""
import argparse,hashlib,json,pathlib,sys,statistics
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)
def require(c,m):
    if not c:raise ValueError(m)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=pathlib.Path);a=parser.parse_args();r=a.bundle.resolve()
    manifest=read(r/'SHA256_MANIFEST.json')
    for p,h in manifest.items():
        f=(r/p).resolve();require(f.is_relative_to(r),'Unsafe path')
        require(f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==h,'Missing/hash mismatch: '+p)
    auth=read(r/'reports/RESEARCHER_AUTHORIZATION.json')
    require(hashlib.sha256((r/'reports/RESEARCHER_REQUEST_ORIGINAL.txt').read_bytes()).hexdigest()==auth['request_sha256'],'Authorization request hash binding')
    require(auth['offline_investigation_authorized'] and not auth['GPU_execution_authorized'] and not auth['training_authorized'],'Scope changed')
    sys.path.insert(0,str(r/'code/project'))
    from stage_i_d053_evaluation import structural_diagnostics,repetition_diagnostics,token_termination
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(r/'evidence/tokenizer.json'))
    cases=[json.loads(s) for s in (r/'evidence/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    sourcecases=read(r/'evidence/source_cases.json');sourceby={c['business_id']:c for c in sourcecases}
    require(len(cases)==24 and len({c['business_id'] for c in cases})==24,'Frozen24 IDs')
    cfg=read(r/'evidence/expected_generation_config.json')
    outputs=read(r/'reports/EXISTING_72_OUTPUTS_DIAGNOSTICS.json')
    rawpilots=[json.loads(s) for s in (r/'evidence/D054_predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    require(len(rawpilots)==48,'Raw D054 count');raw={(p['business_id'],p['arm']):p for p in rawpilots}
    for c in cases:raw[(c['business_id'],'historical_full_F')]=c['baseline_prediction']
    require(len(outputs)==72 and len({(p['business_id'],p['arm']) for p in outputs})==72,'72 unique outputs')
    byid={c['business_id']:c for c in cases}
    for c in cases:
        require(c['split']=='validation' and c['condition']=='filtered','Unexpected split/condition')
        require(tok.encode(c['source']).ids==c['input_ids'],'Source IDs')
        require(tok.encode(c['target']).ids==c['label_ids'],'Teacher IDs')
        require(hashlib.sha256(c['source'].encode()).hexdigest()==c['source_sha256'],'Source hash')
        require(sourceby[c['business_id']]['source']==c['source'],'Original evidence source')
    for p in outputs:
        c=byid[p['business_id']];orig=raw[(p['business_id'],p['arm'])]
        require(p['text']==orig['text'] and p['token_ids']==orig['token_ids'],'Raw prediction changed')
        require(tok.decode(p['token_ids'],skip_special_tokens=True)==p['text'],'Stored BPE decode differs')
        expected={'structure':structural_diagnostics(p['text'],len(c['selected_photo_ids'])),
            'repetition':repetition_diagnostics(p['text'],p['token_ids'],cfg),'termination':token_termination(p['token_ids'],cfg)}
        for k,v in expected.items():require(canonical(v)==canonical(p[k]),'Exact diagnostic mismatch: '+k)
        require(p['source']==c['source'] and p['teacher_target']==c['target'],'Evidence alignment')
        ids=p['token_ids'];labels=[2]+c['label_ids'];n=next((i for i,(x,y) in enumerate(zip(ids,labels)) if x!=y),min(len(ids),len(labels)))
        require(n==p['prefix']['first_teacher_token_mismatch'],'Earliest mismatch differs')
    photos=set()
    for c in sourcecases:
        for cap in c['captions']:
            p=(r/'evidence'/cap['image_file']).resolve();require(p.is_relative_to((r/'evidence').resolve()) and p.is_file(),'Missing original photo')
            photos.add(cap['photo_id'])
    summary=read(r/'reports/HISTORICAL_VS_PILOT_SUMMARY.json');summary_checks=0
    for arm,groups in summary.items():
        for subset,record in groups.items():
            z=[p for p in outputs if p['arm']==arm and (subset=='all24' or p['subset']==subset)]
            expected={'n':len(z),'structure_compliant':sum(p['structure']['section_headers_compliant'] for p in z),
                'exact_one_each_marker':sum(p['structure']['review_marker_count']==p['structure']['visual_marker_count']==1 for p in z),
                'sentence_repetition':sum(p['repetition']['exact_repeated_sentence'] for p in z),
                'adjacent_word_repetition':sum(p['repetition']['consecutive_repeated_word'] for p in z),
                'cap_hits':sum(p['termination']['cap_reached'] for p in z),'early_eos':sum(p['termination']['eos_before_length_cap'] for p in z),
                'header_prefix_complete':sum(p['prefix']['required_header_prefix_complete'] for p in z)}
            for k,v in expected.items():require(record[k]==v,'Summary count mismatch: '+arm+'/'+subset+'/'+k);summary_checks+=1
            require(record['lengths']['mean']==statistics.mean(len(p['token_ids']) for p in z),'Mean length mismatch')
    report={'status':'PASS','sha256_files':len(manifest),'frozen_validation_cases':24,'raw_output_bindings':72,
        'source_target_tokenization_checks':48,'stored_prediction_decodes':72,'exact_diagnostic_recomputations':216,
        'photos':len(photos),'summary_count_checks':summary_checks,'authorization_hash_bound':True,'no_model_execution':True,
        'limits':'Full6600-target audit requires the accepted six datasets in the original project. Its immutable report/receipts are included; this portable verifier does not claim to rerun that full scan or independently judge factual quality.'}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
