"""Read-only detailed verification against actual frozen K/L/M evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from s1000_final_eval_core_v1 import read, digest, file_hash

def audit(root,project):
    ws=root/'WORKSPACE';private=ws/'COORDINATOR_ONLY';hist=private/'HISTORICAL';config=read(ws/'CONFIG.json');cohort=read(ws/'COHORT30.json')['business_ids']
    test=read(hist/'COHORT100_FREEZE.json')['business_ids']
    expected=sorted(sorted(test,key=lambda b:(digest([20261028,'HUMAN_PANEL',b]),b))[:30]);assert cohort==expected
    manifest={r['business_id']:r for r in read(private/'SOURCE_PACKAGE_MANIFEST.json')['frozen_sources']}
    key=read(private/'PRIVATE_ITEMS.json');legacy=list(csv.DictReader((hist/'PRIVATE_HUMAN_DESIGN_KEY.csv').open(encoding='utf8')))
    assert [{k:str(r[k]) for k in ['alias','annotator','position','business_id','condition']} for r in key]==[r for r in legacy if r['annotator'] in ['A01','A02']]
    native={};all_native={};m=read(private/'STAGE_M_COMPLETION.json')
    for c in ['RO','U','F','S']:
        p=project/'outputs/stage_k/s1000_test100_phase1_v1/EXECUTION'/c/'raw_predictions.json'
        assert file_hash(p)==m['prediction_integrity']['condition_bindings'][c]['raw_predictions_sha256']
        for r in read(p):all_native[c,r['business_id'],r['task']]=r
    for r in read(private/'NATIVE_COMPONENT_RECEIPTS.json'):
        n=r['native_record'];assert all_native[n['condition'],n['business_id'],n['task']]==n
        native[n['condition'],n['business_id'],n['task']]=n
    mappings={r['recipient_id']:r for r in read(hist/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings']}
    case_rows=[]
    for b in cohort:
        s=read(ws/config['sources'][b]['file']);assert digest(s)==manifest[b]['sha256']
        assert len(s['reviews'])==manifest[b]['reviews'] and len(s['photos'])==manifest[b]['photos']
        assert set(s['business_metadata'])<=set(['name','categories','city','state'])
        for c in ['RO','U','F','S']:
            n=native[c,b,'review'];assert n['selected_review_ids']==s['selected_review_ids']
        f=native['F',b,'visual'];sh=native['S',b,'visual'];donor=sh.get('shuffled_donor_id')
        mapping=mappings.get(b);assert mapping is not None
        if not sh.get('no_visual_intervention'):
            assert donor==mapping['donor_id'];d=all_native['F',donor,'visual']
            assert sh['text']==d['text'] and sh['token_ids']==d['token_ids']
        else:assert sh['text']==f['text'] and sh['token_ids']==f['token_ids']
        case_rows.append({'business_id':b,'selected_reviews':len(s['reviews']),'recipient_photos':len(s['photos']),
            'F_caption_ids':f['caption_ids'],'zero_F_captions':len(f['caption_ids'])==0,
            'F_visual_empty_preserved':not f['text'].removeprefix('[VISUAL_EVIDENCE]\n').strip(),
            'S_visual_empty_preserved':not sh['text'].removeprefix('[VISUAL_EVIDENCE]\n').strip(),
            'private_shuffled_donor_id':donor,'F_review_generation_status':native['F',b,'review'].get('generation_status'),
            'F_review_termination':native['F',b,'review'].get('termination')})
    original=read(root/'REPORTS/PREPARATION_INTEGRITY.json')['protected_input_sha256'];assert all(file_hash(project/p)==h for p,h in original.items())
    result={'status':'PASS','checks':{'cohort_recomputed_from_original_seed':'PASS','original_120_item_orders_and_aliases_each':'PASS',
        'source_manifest_canonical_hash_counts_and_recipient_IDs':'PASS','all_210_native_records_including_token_IDs_unchanged':'PASS',
        'raw_prediction_hashes_match_Stage_M_receipt':'PASS','shuffled_recipient_donor_and_exact_output_reuse':'PASS','selected_review_IDs_match_each_recipient':'PASS',
        'rater_context_contains_only_business_metadata':'PASS','frozen_protected_inputs_unchanged':'PASS'},
        'businesses':30,'case_audit_coordinator_only':case_rows,'zero_F_caption_cases':sum(r['zero_F_captions'] for r in case_rows),
        'live_human_scoring_started':False,'authorship_identity_and_actual_source_inspection':'NOT_INDEPENDENTLY_AUTHENTICATED'}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--project',required=True,type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=audit(a.root,a.project)
    if a.output:a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in r.items() if k!='case_audit_coordinator_only'},indent=2))
