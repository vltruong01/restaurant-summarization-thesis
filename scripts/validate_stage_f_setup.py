"""Independent validation-only roles, sampled caption EOS and scorer feature checks."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/validation_calibration_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    from stage_f_d030_protocol import active_d030
    from stage_f_d031_protocol import active_d031
    from stage_f_active_protocol import validate_compatibility
    if active_d030() or active_d031():
        receipt=validate_compatibility()
        print(json.dumps({'status':receipt['status'],'original_setup_report_preserved':True,
                          'scorer_bound_protocol_snapshot_verified':True,'caption_scoring_reexecuted':False}))
        return
    from transformers import AutoTokenizer
    from safetensors.numpy import load_file
    rows=[json.loads(l) for l in (ROOT/'data/processed/yelp_2300_v1/photo_metadata/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    roles={r['business_id']:r for r in read(OUT/'business_roles.json')}
    assert set(roles)=={r['business_id'] for r in rows} and len(roles)==200
    assert Counter(r['role'] for r in roles.values())=={'calibration':40,'K_study':160}
    for band in ['5-7','8-9','10-19','20+']:
        assert sum(r['role']=='calibration' and r['raw_photo_band']==band for r in roles.values())==10
    pixels=defaultdict(set)
    with (ROOT/'outputs/stage_d/photo_integrity_duplicate_audit_v1/unfiltered_readable_photo_pool.csv').open(encoding='utf-8',newline='') as f:
        for p in csv.DictReader(f):
            if p['split']=='validation' and p['business_id'] in roles:pixels[p['pixel_sha256']].add(roles[p['business_id']]['role'])
    assert all(len(g)==1 for g in pixels.values())
    plan=read(OUT/'validation_photo_plan.json');assert len(plan)==len({p['photo_id'] for p in plan})==2504
    assert {p['photo_id'] for p in plan}=={r['photo_id'] for r in rows}
    assert all(p['split']=='validation' and p['role']==roles[p['business_id']]['role'] for p in plan)
    caption_cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_caption_pool.yaml').read_text())
    original_cfg=yaml.safe_load((ROOT/'configs/blip2_caption_basic_quality_pilot.yaml').read_text())
    for k in ['model_id','requested_revision','generation','device','basic_quality_policy']:assert caption_cfg[k]==original_cfg[k]
    identity=read(OUT/'pilot_identity.json')
    for name,expected in identity['source_sha256'].items():assert sha(ROOT/name)==expected,name
    tokenizer=AutoTokenizer.from_pretrained(caption_cfg['model_id'],revision=caption_cfg['requested_revision'],cache_dir=ROOT/caption_cfg['model_cache'],local_files_only=True)
    checkpoints=list((OUT/'caption_checkpoints').glob('*.json'));assert checkpoints
    lookup={p['photo_id']:p for p in plan}
    for path in checkpoints:
        r=read(path);p=lookup[r['photo_id']]
        assert r['split']=='validation' and r['role']==p['role'] and r['raw_sha256']==sha(ROOT/r['image_path'])
        assert r['generation_settings']==caption_cfg['generation']
        for a in r['attempts']:
            assert a['generated_sequence_ids'][a['prefix_tokens']:]==a['generated_suffix_ids']
            assert tokenizer.decode(a['generated_suffix_ids'],skip_special_tokens=True)==a['raw_caption']
            assert a['eos_seen']==any(t in a['eos_token_ids'] for t in a['generated_suffix_ids'])
        assert not r['final_caption_set'] and not r['teacher_target']
    smoke=OUT/'scorer_smoke_v2';scores=read(smoke/'caption_scores.json');features=load_file(str(smoke/'embeddings.safetensors'))
    assert scores and all(r['role']=='calibration' and r['split']=='validation' for r in scores)
    for name in ['SBERT','CLIP_image','CLIP_caption','prototypes']:
        assert features[name].dtype==np.float32 and np.isfinite(features[name]).all()
        assert np.allclose(np.linalg.norm(features[name],axis=1),1,atol=1e-5)
    for r in scores:
        assert r['raw_image_sha256']==sha(ROOT/r['image_path'])
        i=r['embedding_index'];assert abs(float(features['CLIP_image'][i]@features['CLIP_caption'][i])-r['clip_score'])<1e-5
        similarity=features['SBERT'][i]@features['prototypes'].T
        assert abs(float(similarity[:6].max()-similarity[6:].max())-r['relevance_margin'])<1e-5
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text())
    assert all(v is None for v in cfg['thresholds'].values()) and cfg['K'] is None
    assert not (OUT/'frozen_thresholds.json').exists()
    result={'status':'PASS_SETUP_AND_REAL_PARTIAL_CAPTION_SCORING_NOT_FULL_POOL_OR_THRESHOLDS',
            'business_roles_checked':200,'validation_photo_IDs_checked':2504,'exact_cross_role_pixel_groups':0,
            'current_caption_checkpoints_checked':len(checkpoints),'scorer_smoke_records_checked':len(scores),
            'raw_decode_EOS_and_float32_feature_formulas_checked':True,'human_labels_collected':0,
            'thresholds_K_selected':False,'source_hashes_unchanged':True,'test_for_tuning':False}
    (OUT/'setup_validation_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
