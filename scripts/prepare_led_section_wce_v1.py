"""Offline section diagnostics and minimal versioned weighted-CE derivative."""
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import struct
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/led_large_section_wce_v1'
EXEC=ROOT/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1'
QUALITY=EXEC/'final_evidence/runs/F_large_decoder_policy_v2/quality_review'
sys.path.insert(0,str(ROOT/'scripts'))
from led_section_weighted_ce_v1 import segment_target,OBJECTIVE,label_key
from teacher_redesign_token_count_v1 import load_tokenizer


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def text(p,s):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:f.write(s)


def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]


def summarize(rows):
    totals={k:sum(x[k] for x in rows) for k in ('review','visual','headers_separators','EOS','complete_labels')}
    totals['businesses']=len(rows)
    totals['review_fraction_valid']=totals['review']/totals['complete_labels']
    totals['visual_fraction_valid']=totals['visual']/totals['complete_labels']
    totals['visual_fraction_bodies']=totals['visual']/(totals['review']+totals['visual'])
    totals['review_fraction_after_weighting']=1.5*totals['review']/(totals['complete_labels']+.5*totals['review'])
    totals['mean_review_tokens']=statistics.mean(x['review'] for x in rows)
    totals['mean_visual_tokens']=statistics.mean(x['visual'] for x in rows)
    totals['median_per_business_visual_fraction_bodies']=statistics.median(x['visual']/(x['review']+x['visual']) for x in rows)
    return totals


def replace_function(source,name,replacement):
    node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement+'\n'+''.join(lines[node.end_lineno:])


def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'No overwrite of preparation version'
    OUT.mkdir(parents=True,exist_ok=True)
    from stage_hi_led_data import PackedLEDDataset
    from led_large_decoder_label_view_v1 import CorrectedLabelDataset
    tok=load_tokenizer(ROOT/'outputs/stage_i/teacher_target_v2_preparation_v1/stage_a_sources_32/tokenizer')
    sm=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    vm=read(ROOT/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json')
    input_hashes={}
    for p in [ROOT/'configs/stage_i_led_large_decoder_policy_v2.json',
              ROOT/'scripts/run_stage_i_led_large_decoder_policy_v2.py',
              ROOT/'scripts/stage_i_led_large_decoder_policy_v2.py',
              ROOT/'scripts/led_large_decoder_label_view_v1.py',
              ROOT/'scripts/led_large_decoder_early_sanity_v2.py',
              ROOT/'outputs/stage_i/led_large_decoder_policy_v2/PREFLIGHT_MANIFEST.json']:
        input_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    assert input_hashes['configs/stage_i_led_large_decoder_policy_v2.json']=='d079f41aa5ae5b1a680da5174eb3ef83235cb9b82e54c96b3d5d487215a168f4'
    assert input_hashes['scripts/run_stage_i_led_large_decoder_policy_v2.py']=='bea436309fadb632a04db82549ad63139f31bd77365e95271b77c1ff68470a61'
    target_rows=[];labels_flat=[];roles_flat=[];offsets=[0];business_ids=[];splits=[];frozen_texts=[]
    for split,key in [('train','train_filtered'),('validation','validation_filtered')]:
        info=sm['datasets'][key];data=PackedLEDDataset(info,5000,768,ROOT)
        corrected=CorrectedLabelDataset(data,vm['views'][key],ROOT)
        p=ROOT/info['stage_g_jsonl'];assert sha(p)==info['stage_g_jsonl_sha256']
        input_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
        for q in [ROOT/info['path'],ROOT/vm['views'][key]['path']]:input_hashes[q.relative_to(ROOT).as_posix()]=sha(q)
        records=jl(p);assert [x['business_id'] for x in records]==data.business_ids
        for i,r in enumerate(records):
            ids,roles,spans=segment_target(r['target'],tok)
            assert ids==corrected[i]['labels']==data[i]['labels'][1:]
            assert len(ids)==len(roles) and len(ids)<=768
            rolecount=Counter(roles);zero=len(r['selected_photo_ids'])==0
            assert not zero or rolecount[2]==0,'Original zero-caption target has visual body'
            body=r['source'].split('[VISUAL_EVIDENCE]',1)[1] if '[VISUAL_EVIDENCE]' in r['source'] else ''
            # Caption count comes from unchanged model input markers, cross-checked with photo IDs.
            import re
            caption_markers=re.findall(r'\[IMG_CAP_\d+\]',r['source'])
            assert len(caption_markers)==len(r['selected_photo_ids'])
            row=dict(business_id=r['business_id'],split=split,review=rolecount[1],visual=rolecount[2],
                headers_separators=rolecount[0],EOS=rolecount[3],complete_labels=len(ids),
                zero_filtered_captions=zero,Filtered_caption_count=len(caption_markers),
                spans=spans,label_ids_sha256=label_key(ids),
                target_utf8_sha256=hashlib.sha256(r['target'].encode()).hexdigest(),
                boundary_straddles=0,all_original_content_EOS_ids_equal=True)
            target_rows.append(row);labels_flat+=ids;roles_flat+=roles;offsets.append(len(labels_flat))
            business_ids.append(r['business_id']);splits.append(split)
            frozen_texts.append(dict(business_id=r['business_id'],split=split,target=r['target'],
                target_utf8_sha256=row['target_utf8_sha256'],label_ids_sha256=row['label_ids_sha256']))
    assert len(target_rows)==2200 and len(set(business_ids))==2200
    np.savez_compressed(OUT/'SECTION_MASK_VIEWS.npz',business_ids=np.array(business_ids),
        splits=np.array(splits),labels=np.array(labels_flat,dtype=np.int32),roles=np.array(roles_flat,dtype=np.int8),
        offsets=np.array(offsets,dtype=np.int64))
    with (OUT/'TARGET_BOUNDARY_AUDIT_2200.jsonl').open('x',encoding='utf-8') as f:
        for r in target_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (OUT/'FROZEN_ORIGINAL_FILTERED_TEXTS_2200.jsonl').open('x',encoding='utf-8') as f:
        for r in frozen_texts:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    by_id={x['business_id']:x for x in target_rows}
    cases=read(QUALITY/'SOURCES_AND_TEACHER_TARGETS.json')
    generated=[];pred_summaries={}
    for epoch in (3,5):
        p=QUALITY/f'predictions/epoch{epoch}.json';pred=read(p);input_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
        assert [r['business_id'] for r in pred]==[r['business_id'] for r in cases]
        for r in pred:
            ids,roles,spans=segment_target(r['text'],tok)
            exact=(r['token_ids'][0]==2 and ids==r['token_ids'][1:])
            assert exact,'Stored raw generation differs from retokenized text; do not report as exact'
            cnt=Counter(roles);parent=by_id[r['business_id']]
            generated.append(dict(business_id=r['business_id'],subset=r['subset'],epoch=epoch,
                review=cnt[1],visual=cnt[2],headers_separators=cnt[0],EOS=cnt[3],complete_labels=len(ids),
                decoder_start_excluded=True,raw_token_alignment_exact=exact,
                zero_filtered_captions=parent['zero_filtered_captions'],
                unsupported_zero_caption_visual=parent['zero_filtered_captions'] and cnt[2]>0,
                target_review_tokens=parent['review'],target_visual_tokens=parent['visual'],
                source_target_sha256=parent['target_utf8_sha256'],
                prediction_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest()))
        pred_summaries[f'epoch{epoch}']={}
        for subset in ('all','random','targeted'):
            rows=[x for x in generated if x['epoch']==epoch and (subset=='all' or x['subset']==subset)]
            pred_summaries[f'epoch{epoch}'][subset]=summarize(rows)
            pred_summaries[f'epoch{epoch}'][subset]['visual_body_longer_than_target_cases']=sum(x['visual']>x['target_visual_tokens'] for x in rows)
            pred_summaries[f'epoch{epoch}'][subset]['review_body_shorter_than_target_cases']=sum(x['review']<x['target_review_tokens'] for x in rows)
            pred_summaries[f'epoch{epoch}'][subset]['zero_caption_visual_assertions']=sum(x['unsupported_zero_caption_visual'] for x in rows)
    save(OUT/'GENERATED_SECTION_AUDIT_48.json',generated)
    sums={split:summarize([x for x in target_rows if x['split']==split]) for split in ('train','validation')}
    sums['fixed_validation_24']=summarize([by_id[x['business_id']] for x in cases])
    zero={split:dict(businesses=sum(x['split']==split for x in target_rows),
         zero=sum(x['split']==split and x['zero_filtered_captions'] for x in target_rows),
         business_ids=[x['business_id'] for x in target_rows if x['split']==split and x['zero_filtered_captions']])
         for split in ('train','validation')}
    for value in zero.values():value['fraction']=value['zero']/value['businesses']
    save(OUT/'DIAGNOSTIC_MEASUREMENTS.json',dict(targets=sums,generated=pred_summaries,zero_Filtered_captions=zero,
        boundary_successes=2200,boundary_failures=0,raw_generation_alignments_exact=48,
        normalization='Headers/separators/EOS included in valid denominator; initial BOS/decoder-start/padding excluded',
        semantic_Core_recall_quantified=False,new_model_assessments_confirmed_by_researcher=False))
    save(OUT/'BASELINE_INPUT_BINDINGS.json',dict(files=input_hashes,held_out_content_access=False))
    # Versioned derivative: namespace, objective, additional static mask binding only.
    c=read(ROOT/'configs/stage_i_led_large_decoder_policy_v2.json')
    new=json.loads(json.dumps(c));new['experiment']='LED_LARGE_SECTION_WCE_V1'
    new['run_root']='runs/F_large_section_wce_v1'
    new['vast_fresh'].update(experiment=new['experiment'],output=new['run_root'])
    new['loss']='section_aware_weighted_CE_only_no_auxiliary_loss'
    new['section_weighted_ce']=dict(objective=OBJECTIVE,
        mask_file='outputs/stage_i/led_large_section_wce_v1/SECTION_MASK_VIEWS.npz',
        mask_sha256=sha(OUT/'SECTION_MASK_VIEWS.npz'))
    save(ROOT/'configs/stage_i_led_large_section_wce_v1.json',new)
    helper=(ROOT/'scripts/stage_i_led_large_decoder_policy_v2.py').read_text(encoding='utf-8')
    helper=helper.replace('decoder_policy_v2','section_wce_v1').replace('DECODER_POLICY_V2','SECTION_WCE_V1')
    # Restore accepted early sanity module: not a new implementation.
    helper=helper.replace('led_large_decoder_early_sanity_v2','led_large_decoder_early_sanity_v2')
    preflight='''def preflight():
    import stage_i_led_large_decoder_policy_v2 as parent
    from led_section_weighted_ce_v1 import OBJECTIVE, SectionMaskIndex
    old=parent.preflight()  # Exact accepted V2 data/runtime/backbone chain.
    c=read(CONFIG)
    objective=dict(objective=OBJECTIVE,
        mask_file='outputs/stage_i/led_large_section_wce_v1/SECTION_MASK_VIEWS.npz',
        mask_sha256=sha(PREP/'SECTION_MASK_VIEWS.npz'))
    expected=dict(old,experiment=EXPERIMENT,run_root='runs/F_large_section_wce_v1',
        vast_fresh=dict(old['vast_fresh'],experiment=EXPERIMENT,output='runs/F_large_section_wce_v1'),
        loss='section_aware_weighted_CE_only_no_auxiliary_loss',section_weighted_ce=objective)
    require(c==expected,'Unapproved configuration/objective change')
    for rel,digest in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'Section-CE preparation binding differs: '+rel)
    SectionMaskIndex(ROOT/objective['mask_file'],objective['mask_sha256'])
    return c
'''
    helper=replace_function(helper,'preflight',preflight)
    # The blind replace also affected parent module import only in old preflight, now replaced.
    helper=helper.replace("    check_deadline(a)\n    return dict(a,_approval_sha256=sha(path))",
        "    require(a.get('section_weighted_ce') == read(CONFIG)['section_weighted_ce'], 'Objective/mask approval missing')\n    check_deadline(a)\n    return dict(a,_approval_sha256=sha(path))")
    text(ROOT/'scripts/stage_i_led_large_section_wce_v1.py',helper)
    old=(ROOT/'scripts/run_stage_i_led_large_decoder_policy_v2.py').read_text(encoding='utf-8')
    runner=old.replace('stage_i_led_large_decoder_policy_v2','stage_i_led_large_section_wce_v1')
    runner=runner.replace('runs/F_large_decoder_policy_v2','runs/F_large_section_wce_v1')
    runner=runner.replace('class MemoryMeter:', 'from led_section_weighted_ce_v1 import section_forward\n\n\nclass MemoryMeter:')
    assert runner.count('output = model(**inputs,use_cache=False)')==1
    runner=runner.replace('output = model(**inputs,use_cache=False)',
        "output = section_forward(model,inputs,use_cache=False,audit_path=folder/'SECTION_LOSS_DIAGNOSTICS.jsonl')")
    assert runner.count('loss=model(**batch,use_cache=False).loss')==1
    runner=runner.replace('loss=model(**batch,use_cache=False).loss',
        "loss=section_forward(model,batch,use_cache=False,audit_path=p/'SECTION_LOSS_DIAGNOSTICS.jsonl').loss")
    text(ROOT/'scripts/run_stage_i_led_large_section_wce_v1.py',runner)
    patches=[]
    for oldname,newname in [('scripts/stage_i_led_large_decoder_policy_v2.py','scripts/stage_i_led_large_section_wce_v1.py'),
                            ('scripts/run_stage_i_led_large_decoder_policy_v2.py','scripts/run_stage_i_led_large_section_wce_v1.py'),
                            ('configs/stage_i_led_large_decoder_policy_v2.json','configs/stage_i_led_large_section_wce_v1.json')]:
        patches.append(''.join(difflib.unified_diff((ROOT/oldname).read_text(encoding='utf-8').splitlines(True),
            (ROOT/newname).read_text(encoding='utf-8').splitlines(True),fromfile=oldname,tofile=newname)))
    text(OUT/'EXACT_PROPOSED_PATCH.diff','\n'.join(patches))
    print(json.dumps(dict(targets=sums,generated=pred_summaries,zero=zero),indent=2))


if __name__=='__main__':main()
