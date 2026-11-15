"""Offline, full-population token-offset/mask audit; never imports a model."""
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from led_large_boundary_global_v1 import POLICY, anchors_from_encoding, MaskIndex, MaskCollator
from led_large_decoder_label_view_v1 import CorrectedLabelCollator, remove_initial_bos

R=Path(__file__).resolve().parents[1]
O=R/'outputs/stage_i/led_large_boundary_global_v1'
PARENT=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    O.mkdir(exist_ok=False)
    baseline=read(R/'configs/stage_i_led_large_decoder_policy_v2.json')
    keys=('model_id','revision','weights_sha256','seed','lora','target_modules','expected_modules',
          'expected_trainable_parameters','optimization','amp_scaler','dynamic_overflow_policy',
          'budgets','training','runtime_pins','decoder_label_policy','generation','checkpoint_policy',
          'checkpoint_selection','early_sanity','evaluation_epochs','source_tokenizer','label_views_manifest')
    config=dict(experiment='LED_LARGE_BOUNDARY_GLOBAL_V1',preparation_only=True,GPU_execution_approved=False,
        parent_config='configs/stage_i_led_large_decoder_policy_v2.json',parent_config_sha256=sha(R/'configs/stage_i_led_large_decoder_policy_v2.json'),
        shared={k:baseline[k] for k in keys},arms=['bos_only','boundary'],fixed_policy=POLICY,
        maximum_global_tokens=36,run_root='runs/LED_LARGE_BOUNDARY_GLOBAL_V1',
        initialization='Fresh official backbone plus identical named-seeded fresh Expanded LoRA per arm; no historical F weights/state',
        arm_execution_order=['bos_only','boundary'],full_training_updates_per_arm=1250,
        feasibility_updates_per_arm=3,early_sanity_outputs_per_arm=4,final_outputs_per_arm=48,
        gate_generation_calls_per_arm=2,all_generation_calls_per_arm_max=54,
        minimum_memory_headroom=0.15,automatic_retry=False,automatic_mask_search=False,
        early_quality_selection=False,primary_comparison_epoch=3,epoch5='predeclared secondary; same selection rule for both arms',
        no_current_GPU_authorization=True,held_out_access=False,Teacher_calls=False,
        old_F_epoch3_adapter_sha256='507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60')
    write(R/'configs/stage_i_led_large_boundary_global_v1.json',config)
    tm=read(R/'assets/tokenizers/led-base-16384.manifest.json')
    for n,h in tm['sha256'].items():assert sha(R/tm['relative_directory']/n)==h
    tok=Tokenizer.from_file(str(R/'assets/tokenizers/led-base-16384/tokenizer.json'))
    assert tok.truncation is None and tok.padding is None
    H=read(R/'outputs/stage_h/led_dataset_v1/manifest.json')
    rows=[];features=[];input_bindings={};stats=[];tokens={}
    for value in read(R/baseline['label_views_manifest'])['views'].values():
        assert sha(R/value['path'])==value['sha256']
        input_bindings[value['path']]=value['sha256']
    for split,count in [('train',2000),('validation',200)]:
        info=H['datasets'][split+'_filtered'];pack_path=R/info['path'];src_path=R/info['stage_g_jsonl']
        assert sha(pack_path)==info['sha256'] and sha(src_path)==info['stage_g_jsonl_sha256']
        input_bindings[info['path']]=sha(pack_path);input_bindings[info['stage_g_jsonl']]=sha(src_path)
        with np.load(pack_path,allow_pickle=False) as z:p={k:z[k] for k in z.files}
        source_rows=[json.loads(s) for s in src_path.read_text(encoding='utf-8').splitlines()]
        assert len(source_rows)==count
        start=len(rows)
        for i,record in enumerate(source_rows):
            assert record['business_id']==str(p['business_ids'][i]) and record['split']==split and record['condition']=='filtered'
            a,b=p['input_offsets'][i:i+2];c,d=p['label_offsets'][i:i+2]
            ids=p['input_ids'][a:b].astype(np.int64).tolist();labels=p['labels'][c:d].astype(np.int64).tolist()
            enc=tok.encode(record['source'],add_special_tokens=True)
            assert enc.ids==ids
            assert hashlib.sha256(record['source'].encode('utf-8')).hexdigest()==str(p['source_sha256'][i])==record['source_sha256']
            row=anchors_from_encoding(record['source'],ids,enc.offsets,int(p['review_counts'][i]),int(p['caption_counts'][i]))
            row.update(business_id=record['business_id'],split=split,source_sha256=record['source_sha256'])
            for m in row['markers']:
                segment=ids[m['token_start']:m['token_end_exclusive']]
                assert tok.decode(segment,skip_special_tokens=False)==m['marker']
                key=(m['marker'],tuple(segment))
                tokens[key]=dict(marker=m['marker'],token_ids=segment,multi_token=len(segment)>1)
            rows.append(row);features.append(dict(input_ids=ids,labels=remove_initial_bos(labels)))
        current=rows[start:]
        stats.append(dict(split=split,rows=count,global_min=min(len(x['positions']) for x in current),
            global_max=max(len(x['positions']) for x in current),
            global_mean=sum(len(x['positions']) for x in current)/count,
            global_histogram=dict(sorted(collections.Counter(len(x['positions']) for x in current).items())),
            review_min=min(x['review_count'] for x in current),review_max=max(x['review_count'] for x in current),
            caption_min=min(x['caption_count'] for x in current),caption_max=max(x['caption_count'] for x in current),
            zero_caption=sum(x['caption_count']==0 for x in current),
            max_source_tokens=max(x['source_length'] for x in current)))
    assert len(rows)==2200 and not {r['business_id'] for r in rows if r['split']=='train'} & {r['business_id'] for r in rows if r['split']=='validation'}
    index=MaskIndex(rows)
    import torch
    assert not torch.cuda.is_initialized()
    for arm in config['arms']:
        collate=MaskCollator(CorrectedLabelCollator,index,arm)
        for feature in features:
            original=CorrectedLabelCollator()([feature]);actual=collate([feature])
            assert set(actual)==set(original)
            for k in ('input_ids','attention_mask','labels'):assert torch.equal(actual[k],original[k])
            if arm=='bos_only':assert torch.equal(actual['global_attention_mask'],original['global_attention_mask'])
        print(arm,'2200 exact input/label/shape/mask checks PASS',flush=True)
    # Mixed-size batch and worst padded source/label envelope; no CUDA/model operation.
    selected=[max(range(len(rows)),key=lambda i:rows[i]['source_length']),
              max(range(len(rows)),key=lambda i:len(features[i]['labels'])),
              max(range(len(rows)),key=lambda i:len(rows[i]['positions'])),
              next(i for i,r in enumerate(rows) if r['caption_count']==0)]
    for arm in config['arms']:
        stress=MaskCollator(CorrectedLabelCollator,index,arm,stress=True)([features[i] for i in selected])
        assert list(stress['input_ids'].shape)==[4,5120] and list(stress['labels'].shape)==[4,768]
    evaluate=[json.loads(s) for s in (R/baseline['early_sanity']['cases_file']).read_text(encoding='utf-8').splitlines()]
    assert len(evaluate)==24
    for case in evaluate:
        row=index.lookup(case['input_ids']);assert row['business_id']==case['business_id'] and row['source_sha256']==case['source_sha256']
    with (O/'ANCHOR_ROWS.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    write(O/'MARKER_TOKENIZATION.json',list(tokens.values()))
    write(O/'MASK_VALIDATION.json',dict(status='PASS_CPU_ALL_2200_FILTERED_INPUTS',created_utc=datetime.now(timezone.utc).isoformat(),
        policy=POLICY,summary=stats,exact_full_source_tokenization=2200,
        unchanged_input_attention_labels_and_shapes_per_arm=2200,control_mask_bit_exact=2200,
        fixed24_evaluation_inputs_resolve=24,epoch1_inputs_resolve=4,
        observed_max_globals=max(len(r['positions']) for r in rows),theoretical_max_globals=36,
        marker_occurrences=sum(len(r['markers']) for r in rows),all_markers_exactly_decoded=True,
        stress_batch_shape=[4,5120],stress_label_shape=[4,768],
        GPU_initialized=False,model_loads=0,model_forward_backward=0,teacher_calls=0,held_out_payloads=0,
        source_npz_and_targets_overwritten=False,input_bindings=input_bindings))
    write(O/'STRESS_CASES.json',dict(policy='Fixed TRAIN-only stress population; cross product of maximum source padding5120,labels768,global36',
        train_business_ids=list(dict.fromkeys([max((r for r in rows if r['split']=='train'),key=lambda r:len(r['positions']))['business_id'],
            max((r for r in rows if r['split']=='train'),key=lambda r:r['source_length'])['business_id'],
            max((r for r in rows if r['split']=='train'),key=lambda r:len(features[rows.index(r)]['labels']))['business_id'],
            next(r['business_id'] for r in rows if r['split']=='train' and r['caption_count']==0)]))))
    write(O/'RESOURCE_ESTIMATE.json',dict(measured_boundary_GPU_memory='UNVERIFIED_NO_GPU_AUTHORIZATION',minimum_headroom=0.15,
        theoretical_global_max=36,observed_global_max=max(len(r['positions']) for r in rows),
        source_padded=5120,label_padded=768,batch=1,heads=16,layers=12,
        one_extra_global_score_tensor_FP16_bytes=16*5120*35*2,
        two_direction_raw_score_deltas_all12_layers_MiB=16*5120*35*2*2*12/1024**2,
        formula_not_total_peak=True,projection_activation_backward_allocator_and_sanity_overhead_not_in_formula=True,
        historical_BOS_training_including_sanity_peak_bytes=17684430848,historical_total_bytes=25298141184,
        historical_headroom=0.3009592792064639,historical_gate_headroom_approx=0.7993,
        historical_measurements_do_not_certify_boundary=True,
        paired_training_time_budget_hours=[9,12],estimate_only=True,
        future_gate_required='Both arms; real TRAIN data36-global forward/backward at5120x768,3 successful updates/arm; actual-mask beam4 plus max-global beam4 trace;>=15% measured headroom; discard all gate states'))
    write(O/'FROZEN_INPUT_BINDINGS.json',dict(files=input_bindings))
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':main()
