"""Pinned FP32 scorers; no threshold selection, K-study or human-label inference."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time
from importlib.metadata import version
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_f/validation_calibration_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def score(records,destination):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel,CLIPProcessor,AutoModel,AutoTokenizer
    from PIL import Image,ImageOps
    from safetensors.torch import save_file
    cfg=yaml.safe_load((ROOT/'configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
    files=read(OUT/'model_files_report.json');destination.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    clip_path=ROOT/files['CLIP_local_safetensors'];semantic_path=ROOT/files['SBERT_local_snapshot']
    assert digest(clip_path/'model.safetensors')==files['CLIP_converted_safetensors_sha256']
    assert digest(semantic_path/'model.safetensors')==cfg['models']['SBERT']['weight_sha256']
    clip,ci=CLIPModel.from_pretrained(clip_path,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    assert not ci['missing_keys'] and not ci['mismatched_keys']
    assert all(k.endswith('position_ids') for k in ci['unexpected_keys'])
    processor=CLIPProcessor.from_pretrained(clip_path,local_files_only=True,use_fast=False)
    semantic,si=AutoModel.from_pretrained(semantic_path,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,add_pooling_layer=False,output_loading_info=True)
    assert not si['missing_keys'] and not si['mismatched_keys']
    assert all('pooler.' in k or k.endswith('position_ids') for k in si['unexpected_keys'])
    tokenizer=AutoTokenizer.from_pretrained(semantic_path,local_files_only=True)
    clip=clip.eval().to('cuda');semantic=semantic.eval().to('cuda')
    def text_embedding(texts):
        batch=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        if batch['input_ids'].shape[1]>384:raise ValueError('SBERT overlength: no silent truncation.')
        batch={k:v.to('cuda') for k,v in batch.items()}
        with torch.inference_mode():
            hidden=semantic(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1).float()
            return F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9),p=2,dim=1)
    prototypes=cfg['relevance']['positive_prototypes']+cfg['relevance']['negative_prototypes']
    proto=text_embedding(prototypes)
    output=[];vectors=[];clip_images=[];clip_texts=[]
    for record in records:
        assert digest(ROOT/record['image_path'])==record['raw_sha256'],'Original image bytes changed since caption generation.'
        text=record['normalized_caption'];row={k:record[k] for k in ['photo_id','business_id','name','split','image_path','raw_caption','normalized_caption']}
        row['role']=record.get('role');row['basic_eligible']=record['basic_eligible']
        row['caption_record_sha256']=record.get('_record_sha256')
        row['raw_image_sha256']=record['raw_sha256']
        if not record['basic_eligible']:
            row.update(score_status='basic_failure_not_scored',clip_score=None,relevance_margin=None,embedding_index=None);output.append(row);continue
        clip_ids=processor.tokenizer(text,truncation=False)['input_ids']
        semantic_ids=tokenizer(text,truncation=False)['input_ids']
        row.update(CLIP_text_tokens=len(clip_ids),SBERT_text_tokens=len(semantic_ids))
        if len(clip_ids)>77 or len(semantic_ids)>384:
            row.update(score_status='overlength_requires_protocol_review',clip_score=None,relevance_margin=None,embedding_index=None);output.append(row);continue
        with Image.open(ROOT/record['image_path']) as original:
            image=ImageOps.exif_transpose(original).convert('RGB')
            inputs=processor(text=[text],images=image,return_tensors='pt',padding=True,truncation=False)
        inputs={k:v.to('cuda') for k,v in inputs.items()}
        with torch.inference_mode():
            image_vec=F.normalize(clip.get_image_features(pixel_values=inputs['pixel_values']),p=2,dim=1)
            text_vec=F.normalize(clip.get_text_features(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask']),p=2,dim=1)
            vec=text_embedding([text]);prototype_cos=(vec@proto.T)[0]
            positive=prototype_cos[:6].max();negative=prototype_cos[6:].max()
            cosine=(image_vec*text_vec).sum().item();margin=(positive-negative).item()
        assert np.isfinite(cosine) and np.isfinite(margin)
        row.update(score_status='scored',clip_score=cosine,relevance_margin=margin,
                   positive_prototype_max_cosine=positive.item(),negative_prototype_max_cosine=negative.item(),
                   embedding_index=len(vectors))
        output.append(row);vectors.append(vec.cpu()[0]);clip_images.append(image_vec.cpu()[0]);clip_texts.append(text_vec.cpu()[0])
    if vectors:
        save_file({'SBERT':torch.stack(vectors),'CLIP_image':torch.stack(clip_images),'CLIP_caption':torch.stack(clip_texts),'prototypes':proto.cpu()},str(destination/'embeddings.safetensors'))
    (destination/'caption_scores.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.cuda.synchronize()
    report={'status':'ACTUAL_PINNED_SCORING_COMPLETE_NOT_FILTER_THRESHOLD_OR_K_SELECTION','input_records':len(records),'scored':len(vectors),'overlength':sum(r['score_status'].startswith('overlength') for r in output),'basic_failures':sum(not r['basic_eligible'] for r in output),'dtype':'float32','device':torch.cuda.get_device_name(0),'TF32':False,'elapsed_seconds':round(time.perf_counter()-started,3),'peak_GPU_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_GPU_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'models':cfg['models'],'config_sha256':digest(ROOT/'configs/stage_f_validation_protocol.yaml'),'code_sha256':digest(Path(__file__)),'versions':{p:version(p) for p in ['torch','transformers','tokenizers','numpy','safetensors']},'CLIP_loading_info':ci,'SBERT_loading_info':si,'human_labels_or_thresholds_K_selected':False}
    (destination/'scoring_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    del clip,semantic,proto;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--role',choices=['calibration','K_study']);args=p.parse_args()
    plan=read(OUT/'validation_photo_plan.json');roles={r['business_id']:r['role'] for r in read(OUT/'business_roles.json')}
    records=[]
    if args.smoke:
        source=OUT/'caption_checkpoints' if list((OUT/'caption_checkpoints').glob('*.json')) else ROOT/'outputs/stage_e/blip2_caption_basic_quality_pilot_v2/caption_checkpoints'
        for path in source.glob('*.json'):
            r=read(path)
            if r['split']=='validation' and roles[r['business_id']]=='calibration':
                r['role']='calibration';r['_record_sha256']=digest(path);records.append(r)
        assert records,'No E1 calibration-role captions available; generate calibration smoke first.'
        destination=OUT/'scorer_smoke_v2'
    else:
        for photo in plan:
            if args.role and photo['role']!=args.role:continue
            path=OUT/'caption_checkpoints'/f"{photo['photo_id']}.json"
            if not path.exists():raise RuntimeError('Complete validation caption pool required before full scoring.')
            r=read(path);r['_record_sha256']=digest(path);records.append(r)
        destination=OUT/('scores_'+args.role if args.role else 'scores')
    score(records,destination)


if __name__=='__main__':main()
