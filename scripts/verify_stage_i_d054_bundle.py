"""Portable offline verifier. No model loading, CUDA, network or inference."""
import hashlib,json,re,sys
from pathlib import Path
from stage_i_d053_evaluation import validate_predictions,repetition_diagnostics,structural_diagnostics,token_termination,paired_comparison,human_metrics

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def same_json_data(a,b):
    """Exact serialized data comparison: tuple spans serialize as JSON lists.

    No float tolerance or boolean/numeric coercion. All numeric values must match.
    """
    options=dict(sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    return json.dumps(a,**options)==json.dumps(b,**options)

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        p=(root/r['path']).resolve();assert p.is_relative_to(root),'Unsafe path'
        assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['path']
    dependencies=json.loads((root/'CODE_DEPENDENCY_MANIFEST.json').read_text(encoding='utf-8'))
    for row in dependencies['files']:
        assert digest(root/'code'/row['file'])==row['sha256']
        for dep in row['local_python_dependencies']:assert (root/'code'/dep).is_file(),dep
    cfg=json.loads((root/'configs/FROZEN_EVALUATION_CONFIG.json').read_text(encoding='utf-8'))
    source=root/'reference_evidence';sys.path.insert(0,str(source/'code'))
    from check_stage_i_d053_final_references_v2 import check
    references=check(source);assert references['valid'] and references['reference_freeze_manifest_checked']
    cases=json.loads((source/'source_cases.json').read_text(encoding='utf-8'));by={c['business_id']:c for c in cases}
    inputs=[json.loads(s) for s in (root/'evidence/FROZEN_EVALUATION_INPUTS.jsonl').read_text(encoding='utf-8').splitlines()]
    input_by={c['business_id']:c for c in inputs}
    frozen=json.loads((source/'final_references/FINAL_RESEARCHER_APPROVED_REFERENCES.json').read_text(encoding='utf-8'))
    predictions=[json.loads(s) for s in (root/'results/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_predictions(predictions,cases,cfg)
    old=json.loads((root/'results/AUTOMATIC_EVALUATION.json').read_text(encoding='utf-8'));rows=[]
    for p in predictions:
        assert hashlib.sha256(p['text'].encode()).hexdigest()==p['text_sha256']
        assert hashlib.sha256(json.dumps(p['token_ids'],separators=(',',':')).encode()).hexdigest()==p['token_ids_sha256']
        assert p['generation_config']==cfg['effective_generation_config']
        assert p['reference_freeze_sha256']==digest(source/'final_references/FINAL_RESEARCHER_APPROVED_REFERENCES.json')
        assert p['source_untruncated'] and p['source_content_tokens']<=5000
        c=by[p['business_id']];assert hashlib.sha256(c['source'].encode()).hexdigest()==p['source_sha256']
        assert p['source_content_tokens']==len(input_by[p['business_id']]['input_ids'])
        assert c['source']==input_by[p['business_id']]['source']
        units=[u for u in frozen['units'] if u['business_id']==p['business_id']]
        rows.append({'arm':p['arm'],'business_id':p['business_id'],'subset':c['subset'],
          'prediction_text_sha256':p['text_sha256'],
          'automatic':{'repetition':repetition_diagnostics(p['text'],p['token_ids'],cfg['effective_generation_config']),
          'structure':structural_diagnostics(p['text'],len(c['captions'])),
          'termination':token_termination(p['token_ids'],cfg['effective_generation_config'])},'human':human_metrics(None,p,c,units)})
    assert same_json_data(rows,old['per_output']),'Metric mismatch'
    assert same_json_data(paired_comparison(rows,cfg),old['paired']),'Bootstrap/comparison mismatch'
    html=(root/'D054_BLINDED_REVIEW.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert len(payload['cases'])==24
    key=json.loads((root/'private/BLINDING_KEY_UNBLINDS_REVIEW.json').read_text(encoding='utf-8'))
    assert 'A_decoder_only' not in html and 'B_encoder_decoder' not in html,'Blinded page leaks arm labels'
    for c in payload['cases']:
        assert c['source']==by[c['business_id']]['source']
        assert sorted(o['text_sha256'] for o in c['outputs'])==sorted(p['text_sha256'] for p in predictions if p['business_id']==c['business_id'])
        for o,position in zip(c['outputs'],['left','right']):
            p=next(p for p in predictions if p['business_id']==c['business_id'] and p['arm']==key['mapping'][c['business_id']][position])
            assert o['text_sha256']==p['text_sha256'] and o['text']==p['text']
        for cap in c['captions']:assert (root/cap['image_file']).is_file()
    pending=json.loads((root/'results/PENDING_RESEARCHER_OUTPUT_ANNOTATIONS.json').read_text(encoding='utf-8'))
    assert len(pending)==48 and all(p['human_verified'] is False for p in pending)
    assessment=json.loads((root/'results/CODEX_SOURCE_BASED_ASSESSMENT.json').read_text(encoding='utf-8'))
    assert len(assessment['outputs'])==48
    for r in assessment['outputs']:
        assert r['human_verified'] is False and r['full_output_atomic_grounding_score'] is None and r['complete_unit_coverage_score'] is None
        p=next(p for p in predictions if p['business_id']==r['business_id'] and p['arm']==r['arm'])
        assert r['prediction_text_sha256']==p['text_sha256']
        c=by[r['business_id']];ee={x['marker']:x for x in c['reviews']};ee.update({x['caption_id']:x for x in c['captions']})
        for claim in r['claims']:
            assert p['text'][claim['start']:claim['end']]==claim['text'] and claim['human_verified'] is False
            for e in claim['evidence']:
                original=ee[e['marker']];assert e['text']==original['text']
                assert e.get('review_id',e.get('photo_id'))==original.get('review_id',original.get('photo_id'))
        for m in r['source_reference_matches_examples']:
            assert p['text'][m['start']:m['end']]==m['output_text'] and m['human_verified'] is False
            assert m['unit'] in frozen['units']
        for u in r['important_core_omission_examples']:assert u in frozen['units'] and u['importance']=='core'
    result={'status':'PASS','hashes_checked':len(manifest['files']),'reference_v2':'PASS','predictions':48,
      'identical_source_pairs':24,'metrics_and_seeded_bootstrap_reproduced':True,'human_output_annotations_created':0,
      'blinded_page_bindings':'PASS','local_code_dependency_closure':'PASS','model_assisted_claim_evidence_spans':'PASS','all_original_photos_present':True,'new_inference':False}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
