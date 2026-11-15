"""One minimal sentence-only offline revision. Never load a model or use CUDA."""
import importlib.metadata
import shutil
from datetime import datetime,timezone
from stage_i_unlikelihood_v2 import *


def main():
    from tokenizers import Tokenizer
    import numpy as np
    v1=ROOT/'outputs/stage_i/unlikelihood_preparation_v1'
    require(not PREP.exists() and not CONFIG.exists(),'Preserve existing preparation; no overwrite')
    PREP.mkdir(parents=True)
    old=read(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json')
    receipt=read(v1/'DELIVERY_RECEIPT.json')
    require(sha(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json')==receipt['config_sha256'],'V1 config changed')
    require(sha(v1/'PREFLIGHT_MANIFEST.json')==receipt['preflight_manifest_sha256'],'V1 manifest changed')
    protected=dict(read(v1/'PREFLIGHT_MANIFEST.json')['files'])
    for rel,h in protected.items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==h,'Prior scientific/preparation binding changed: '+rel)
    hm=read(ROOT/old['stage_h_manifest']);tm=read(ROOT/hm['tokenizer_manifest'])
    tok=Tokenizer.from_file(str(ROOT/tm['relative_directory']/'tokenizer.json'))
    info=hm['datasets']['train_filtered']
    require(sha(ROOT/info['stage_g_jsonl'])==info['stage_g_jsonl_sha256'],'Training labels changed')
    rows=[json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
    by_id={r['business_id']:r for r in rows}
    # These files are copied byte-for-byte, never resampled or reordered.
    for name in ('TRAINING_MANIFEST.json','EVALUATION_MANIFEST.json','ADAPTER_INVENTORY.json'):
        shutil.copy2(v1/name,PREP/name)
        require(sha(PREP/name)==sha(v1/name),'Fixed manifest byte mismatch')
    selected=read(PREP/'TRAINING_MANIFEST.json')['business_ids']
    previous=[json.loads(s) for s in (v1/'NEGATIVE_VIEWS.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(selected)==512 and [v['business_id'] for v in previous]==selected,'V1 fixed business mapping differs')
    views=[];same_sentence_views=0
    for bid,prior in zip(selected,previous):
        row=by_id[bid];ids=tok.encode(row['target']).ids
        require(hashlib.sha256(row['target'].encode()).hexdigest()==prior['positive_sha256'],'Positive wording changed')
        require(hashlib.sha256(json.dumps(ids).encode()).hexdigest()==prior['positive_ids_sha256'],'Positive tokens changed')
        v=build_negative(row['target'],tok,bid,'sentence',limit=old['budgets']['target'],seed=old['seed'])
        v.update({k:prior[k] for k in ('business_id','original_row_index','source_sha256','positive_ids_sha256')})
        if prior['kind']=='sentence':
            require(v==prior,'Existing sentence-level view changed unexpectedly')
            same_sentence_views+=1
        views.append(v)
    with np.load(ROOT/info['path'],allow_pickle=False) as pack:
        labels,offsets=pack['labels'],pack['label_offsets']
        require(pack['business_ids'].tolist()==[r['business_id'] for r in rows],'Original packed ID map differs')
        for v in views:
            i=v['original_row_index']
            require(rows[i]['business_id']==v['business_id'] and labels[offsets[i]:offsets[i+1]].tolist()==tok.encode(rows[i]['target']).ids,
                    'Original full positive label alignment differs')
    write(PREP/'NEGATIVE_VIEWS.jsonl','\n'.join(json.dumps(v,ensure_ascii=False) for v in views))
    failures=[{'business_id':v['business_id'],'reason':v['reason'],'rejected_candidates':v['rejected_candidates']} for v in views if not v['eligible']]
    write(PREP/'ELIGIBILITY_REPORT.json',{'businesses':512,'eligible':512-len(failures),'failures':failures,
        'previous256sentence_views_identical':same_sentence_views,'silent_replacements':0,'truncations':0})
    require(not failures,'STOP: ineligible sentence views recorded in ELIGIBILITY_REPORT.json; no fallback')
    cfg=read(ROOT/'configs/stage_i_unlikelihood_pilot_v1.json')
    cfg.update(experiment='post_D060_unlikelihood_paired_v2',
        run_root='outputs/stage_i/unlikelihood_paired_run_v2',
        negative_design='one_sentence_level_teacher_derived_prefix_per_business; no_model_generation',
        gate_gradient_diagnostics={'CE_and_UL_separate_autograd':True,'AMP_scaled_then_unscaled':True,
            'all_trainable_LoRA_tensors':48,'extra_autograd_traversals_max_per_gate_arm':48,
            'new_optimizer_updates':0,'new_model_forward_calls':0,
            'extra_checkpoint_recomputation_possible':True,
            'low_signal_flag_ratio':0.01,'flag_is_descriptive_not_a_PASS_threshold':True,
            'automatic_weight_adjustment':False,'precision_or_tolerance_changes':False},
        practical_quality_benchmark='outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch3/predictions.json')
    write(CONFIG,cfg)
    changed={k for k in set(old)|set(cfg) if old.get(k)!=cfg.get(k)}
    require(changed=={'experiment','run_root','negative_design','gate_gradient_diagnostics','practical_quality_benchmark'},'Unrequested configuration change')
    cases=[json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    benchmark=read(ROOT/cfg['practical_quality_benchmark']);by_case={c['business_id']:c for c in cases}
    require(len(benchmark)==24 and len({b['business_id'] for b in benchmark})==24 and {b['business_id'] for b in benchmark}==set(by_case),'Wrong D059epoch3 case IDs')
    require(all(b['source_sha256']==by_case[b['business_id']]['source_sha256'] and b['subset']==by_case[b['business_id']]['subset'] and b['epoch']==3 for b in benchmark),'D059 benchmark input mismatch')
    gp=ROOT/'outputs/stage_i/d059_expanded_filtered_v2/evaluate/EFFECTIVE_GENERATION_CONFIG.json'
    expected=read(ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/expected_generation_config.json')
    semantic=lambda x:{k:v for k,v in x.items() if k not in ('transformers_version','_from_model_config','_commit_hash')}
    require(semantic(read(gp))==semantic(expected),'D059 benchmark decoding differs')
    write(PREP/'D059_EPOCH3_PRACTICAL_BENCHMARK.json',{'comparison_type':'NON_CAUSAL_CROSS_ARCHITECTURE_PRACTICAL_BENCHMARK',
        'source_path':cfg['practical_quality_benchmark'],'source_sha256':sha(ROOT/cfg['practical_quality_benchmark']),
        'effective_generation_config_path':gp.relative_to(ROOT).as_posix(),'effective_generation_config_sha256':sha(gp),
        'same24businesses_and_sources':True,'predictions':benchmark,'new_generation_calls':0})
    require({k:importlib.metadata.version(k) for k in cfg['runtime_pins']}==cfg['runtime_pins'],'Runtime drift')
    write(PREP/'V1_TO_V2_AMENDMENT.json',{'changed_config_keys':sorted(changed),
        'unchanged_scientific_config_keys':sorted(set(old)-changed),'source_v1_config_sha256':receipt['config_sha256'],
        'v1_manifest_sha256':receipt['preflight_manifest_sha256'],
        'TRAINING_MANIFEST_sha256':sha(PREP/'TRAINING_MANIFEST.json'),
        'EVALUATION_MANIFEST_sha256':sha(PREP/'EVALUATION_MANIFEST.json'),
        'historical_v1_bindings_verified':len(protected),'model_operations':0,'GPU_operations':0,
        'historical_deadlines_unchanged':cfg['old_deadlines_preserved'],'prior_stop_loss_override_approved':False,
        'scoring_change':'add saved D059 Expanded-LoRA epoch3 practical benchmark; no matched causal claim'})
    write(PREP/'OFFLINE_SCOPE_RECORD.json',{'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'source':'researcher_message','decision':'ONE minimal offline revision; no GPU execution of v1 or v2 authorized',
        'GPU_execution_approved':False,'training_authorized':False,'deadline_override_approved':False,
        'max_future_window_hours':48,'no_followup_experiment_if_failed':True})
    print(json.dumps({'eligible':512,'sentence':512,'word':0,'prior_sentence_views_identical':same_sentence_views,
        'penalty_tokens':sum(v['penalized_tokens'] for v in views),'max_negative_tokens':max(len(v['negative_ids']) for v in views),
        'historical_v1_bindings_verified':len(protected),'new_model_operations':0,'GPU_operations':0}))


if __name__=='__main__':main()
