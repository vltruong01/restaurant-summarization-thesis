"""Blind human annotation UI and fixed calibration samples; never generate labels."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import os
import base64
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/validation_calibration_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(x):return hashlib.sha256(('42|human_calibration_v1|'+x).encode()).hexdigest()


def build(preview=False):
    from safetensors.numpy import load_file
    folder=OUT/('scorer_smoke_v2' if preview and (OUT/'scorer_smoke_v2/scoring_report.json').exists() else 'scorer_smoke' if preview else 'scores_calibration')
    scores=read(folder/'caption_scores.json')
    rows=[r for r in scores if r['role']=='calibration' and r['score_status']=='scored']
    assert rows and all(r['split']=='validation' for r in rows)
    vectors=load_file(str(folder/'embeddings.safetensors'))['SBERT']
    plan={r['photo_id']:r for r in read(OUT/'validation_photo_plan.json')}
    by_business=defaultdict(list)
    for r in rows:by_business[r['business_id']].append(r)
    selected={}
    for bid,items in sorted(by_business.items()):
        r=min(items,key=lambda r:rank(r['photo_id']));selected[r['photo_id']]=r
    for r in rows:
        if plan[r['photo_id']]['label']=='menu':selected[r['photo_id']]=r
    clip_sorted=sorted(rows,key=lambda r:(r['clip_score'],r['photo_id']))
    rel_sorted=sorted(rows,key=lambda r:(r['relevance_margin'],r['photo_id']))
    cq={r['photo_id']:min(4,5*i//len(rows)) for i,r in enumerate(clip_sorted)}
    rq={r['photo_id']:min(4,5*i//len(rows)) for i,r in enumerate(rel_sorted)}
    cells=defaultdict(list)
    for r in rows:
        if r['photo_id'] not in selected:cells[cq[r['photo_id']],rq[r['photo_id']]].append(r)
    for items in cells.values():items.sort(key=lambda r:rank(r['photo_id']))
    while len(selected)<min(200,len(rows)) and any(cells.values()):
        for cell in sorted(cells):
            if cells[cell] and len(selected)<min(200,len(rows)):
                r=cells[cell].pop(0);selected[r['photo_id']]=r
    cases=[];private_scores={}
    for r in sorted(selected.values(),key=lambda r:rank(r['photo_id'])):
        cid='IMAGE_'+r['photo_id'];p=plan[r['photo_id']]
        cases.append({'case_id':cid,'kind':'image_caption','business_id':r['business_id'],'name':r['name'],
                      'photo_ids':[r['photo_id']],'image_paths':[p['image_path']],
                      'raw_captions':[r['raw_caption']],'normalized_captions':[r['normalized_caption']],
                      'labels':{'consistency':'','relevance':''},'notes':''})
        private_scores[cid]={'clip_score':r['clip_score'],'relevance_margin':r['relevance_margin']}
    for bid,items in sorted(by_business.items()):
        pairs=[]
        for a,b in itertools.combinations(sorted(items,key=lambda r:r['photo_id']),2):
            cosine=float(vectors[a['embedding_index']]@vectors[b['embedding_index']])
            phash=(int(plan[a['photo_id']]['phash64_hex'],16)^int(plan[b['photo_id']]['phash64_hex'],16)).bit_count()
            pairs.append((cosine,phash,a,b))
        if not pairs:continue
        ordered=sorted(pairs,key=lambda x:(x[0],x[2]['photo_id'],x[3]['photo_id']))
        # Five semantic-rank and five image-distance-rank samples, up to10 unique pairs/business.
        chosen={}
        for ordering in [ordered,sorted(pairs,key=lambda x:(x[1],x[2]['photo_id'],x[3]['photo_id']))]:
            for i in range(min(5,len(ordering))):
                pair=ordering[round(i*(len(ordering)-1)/max(1,min(5,len(ordering))-1))]
                chosen[pair[2]['photo_id'],pair[3]['photo_id']]=pair
        for pair in ordered:
            if len(chosen)>=min(10,len(pairs)):break
            chosen.setdefault((pair[2]['photo_id'],pair[3]['photo_id']),pair)
        for cosine,phash,a,b in chosen.values():
            cid='PAIR_'+a['photo_id']+'_'+b['photo_id']
            cases.append({'case_id':cid,'kind':'caption_pair','business_id':bid,'name':a['name'],
                          'photo_ids':[a['photo_id'],b['photo_id']],
                          'image_paths':[plan[a['photo_id']]['image_path'],plan[b['photo_id']]['image_path']],
                          'raw_captions':[a['raw_caption'],b['raw_caption']],
                          'normalized_captions':[a['normalized_caption'],b['normalized_caption']],
                          'labels':{'redundancy':'','broad_topic':'','image_duplicate':''},'notes':''})
            private_scores[cid]={'semantic_cosine':cosine,'phash_hamming':phash}
    target=OUT/('human_check_preview' if preview else 'human_check');target.mkdir(exist_ok=True)
    source={'schema_version':1,'dataset_kind':'preview_not_for_thresholds' if preview else 'official_calibration',
            'scoring_sha256':digest(folder/'caption_scores.json'),
            'protocol_sha256':digest(ROOT/'configs/stage_f_validation_protocol.yaml'),
            'role_manifest_sha256':digest(OUT/'business_roles.json'),'cases':cases}
    payload=json.dumps(source,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    source['dataset_sha256']=hashlib.sha256(payload.encode()).hexdigest()
    (target/'annotation_dataset.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (target/'private_calibration_scores.json').write_text(json.dumps(private_scores,indent=2)+'\n',encoding='utf-8')
    ui_cases=[]
    for c in cases:
        item=dict(c);item['image_paths']=[os.path.relpath(ROOT/p,target).replace('\\','/') for p in c['image_paths']];ui_cases.append(item)
    ui={k:v for k,v in source.items() if k!='cases'};ui['cases']=ui_cases
    image_ids={pid for c in cases for pid in c['photo_ids']}
    ui['images']={pid:'data:image/jpeg;base64,'+base64.b64encode((ROOT/plan[pid]['image_path']).read_bytes()).decode('ascii') for pid in image_ids}
    template=Path(__file__).with_name('stage_f_human_check_template.html').read_text(encoding='utf-8')
    (target/'human_check.html').write_text(template.replace('__ANNOTATION_DATA__',json.dumps(ui,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    report={'status':'HUMAN_CHECK_PREVIEW_ONLY' if preview else 'OFFICIAL_CALIBRATION_QUEUE_READY_HUMAN_LABELS_PENDING',
            'image_caption_cases':sum(c['kind']=='image_caption' for c in cases),
            'pair_cases':sum(c['kind']=='caption_pair' for c in cases),'human_labels_collected':0,
            'calibration_businesses_represented':len(by_business),'model_scores_blinded_in_UI':True,
            'dataset_sha256':source['dataset_sha256'],'preview_cannot_calibrate_thresholds':preview}
    (target/'queue_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');build(p.parse_args().preview)
