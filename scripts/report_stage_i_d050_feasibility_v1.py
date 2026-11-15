"""Report saved D-050 evidence offline, preserving failure and missing measurements."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/d050_gpu_feasibility_v1'
PREP=ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1'
ARMS=['A_decoder_only','B_encoder_decoder']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        assert read(p)==v,'Preserve differing report: '+str(p)
        return
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def branch(n):return 'encoder_global' if '_global.' in n else 'encoder_local' if '.encoder.layers.' in n else 'decoder'
def main():
    initial=read(OUT/'PRE_EXECUTION_INTEGRITY.json')
    assert sha(ROOT/'configs/stage_i_paired_lora_pilot_v1.json')==initial['config_sha256']
    assert sha(PREP/'PREPARATION_MANIFEST.json')==initial['preparation_manifest_sha256']
    for n,h in initial['prepared_files'].items():assert sha(PREP/n)==h,n
    for group in ['code','upstream','original_archive_files']:
        for n,h in initial[group].items():assert sha(ROOT/n)==h,n
    cache=Path.home()/'.cache/huggingface/hub/models--allenai--led-base-16384/snapshots/38335783885b338d93791936c54bb4be46bebed9'
    for n,x in initial['cached_backbone_files'].items():assert sha(cache/n)==x['sha256']
    write(OUT/'POST_EXECUTION_INTEGRITY.json',{'status':'PASS','original_archive_files_unchanged':275,'prepared_files_unchanged':14,'reviewed_code_files_unchanged':6,'upstream_files_unchanged':22,'cached_backbone_files_unchanged':len(initial['cached_backbone_files']),'held_out_access':False,'targets_or_historical_checkpoints_modified':False})
    comparison={};drift={};grad_summary={};optimizer_summary={}
    for arm in ARMS:
        p=OUT/arm;raw=ROOT/'outputs/stage_i/paired_lora_pilot_runs_v1'/arm/'feasibility'
        observation=read(p/'observation_completion.json');r=read(raw/'report.json') if (raw/'report.json').exists() else None
        runtime=read(p/'runtime.json');integrity=read(p/'actual_parameter_update_integrity.json')
        assert not integrity['changed_frozen_parameters'] and integrity['only_expected_adapters_changed'] and integrity['frozen_parameters_have_no_gradients']
        count=48 if arm==ARMS[0] else 96
        assert len(integrity['changed_trainable_parameters'])==count
        restored=read(p/'checkpoint_restored_state_fingerprints.json');assert all(restored['exact_matches'].values())
        micro=[json.loads(l) for l in (p/'microbatches.jsonl').read_text().splitlines()];assert len(micro)==32
        assert all(m['source_tensor_shape']==[1,5120] and m['label_tensor_shape']==[1,768] and m['global_tokens']==1 for m in micro)
        updates=[read(p/'optimizer_updates'/f'call_{i}.json') for i in range(1,5)]
        assert [u['parameter_steps'] for u in updates]==[[1],[2],[3],[3]]
        assert [u['scheduler_last_epoch_before_scheduler_step'] for u in updates]==[0,1,2,2]
        # AdamW also decays nonzero A weights at step2 despite their zero gradients.
        assert [len(u['changed_parameter_names']) for u in updates]==[0,count,count,count]
        optimizer_summary[arm]=[{k:v for k,v in u.items() if k not in ['adapter_hashes','changed_parameter_names']}|{'changed_adapter_tensors':len(u['changed_parameter_names'])} for u in updates]
        gradients=[read(p/'gradients'/f'update_call_{i}.json') for i in range(1,5)];grad_summary[arm]=[]
        for i,values in enumerate(gradients,1):
            groups={}
            for n,v in values.items():
                assert v['finite']
                if i>=3:assert v['nonzero']
                key=branch(n)+('_A' if '.lora_A.' in n else '_B');groups.setdefault(key,[]).append(v)
            grad_summary[arm].append({'physical_update_call':i,'groups':{k:{'tensors':len(v),'nonzero':sum(t['nonzero'] for t in v),'all_finite':True,'norm_min':min(t['norm'] for t in v),'norm_max':max(t['norm'] for t in v)} for k,v in groups.items()}})
        differences=[]
        for n,v in gradients[2].items():
            w=gradients[3][n];diff=abs(w['norm']-v['norm'])
            differences.append({'parameter':n,'branch':branch(n),'original_norm':v['norm'],'replay_norm':w['norm'],'absolute_norm_difference':diff,'relative_norm_difference':diff/max(abs(v['norm']),1e-30),'adapter_weights_exact':updates[2]['adapter_hashes'][n]==updates[3]['adapter_hashes'][n]})
        drift[arm]=differences
        telemetry=read(p/'device_telemetry.json');device=max(float(t['csv_memory_MiB_total_MiB_util_percent'].split(',')[0]) for t in telemetry if t.get('exit_code')==0 and t.get('csv_memory_MiB_total_MiB_util_percent'))
        comparison[arm]={'overall_status':'TECHNICAL_FEASIBILITY_PASS' if r else 'BLOCKED_REPLAY_TOLERANCE_FAILURE','observation_status':observation['status'],'error':observation['error'],'gpu':runtime['gpu'],'modules':len(runtime['scope']['modules']),'trainable_parameters':runtime['scope']['trainable_parameters'],'unchanged_frozen_tensors':integrity['frozen_parameter_tensors'],'frozen_parameters':161844480,'changed_intended_adapter_tensors':count,'nonzero_finite_gradient_tensors_at_step3':count,'finite_loss_checks_completed_on_all_32_microbatches':True,'loss_means_per_distinct_step':[statistics.mean(s['losses']) for s in r['gradient_records']] if r else None,'peak_allocated_MiB':r['peak_allocated_bytes']/2**20 if r else None,'peak_reserved_MiB':r['peak_reserved_bytes']/2**20 if r else None,'estimated_headroom_fraction':r['estimated_headroom_fraction'] if r else None,'sampled_device_peak_MiB':device,'checkpoint_model_restore_exact_asserted':True,'checkpoint_states_exact':restored['exact_matches'],'replay_exact_tensor_hashes':sum(d['adapter_weights_exact'] for d in differences),'replay_different_tensor_hashes':sum(not d['adapter_weights_exact'] for d in differences),'replay_allclose':r['replay_allclose'] if r else False,'replay_tolerance':{'rtol':1e-5,'atol':1e-6},'parameter_max_abs_error_unavailable':True,'max_relative_gradient_norm_difference':max(d['relative_norm_difference'] for d in differences),'seconds_including_load_and_observation':observation['seconds_including_load_and_observation'],'microbatches':32,'physical_optimizer_calls':4,'distinct_optimizer_steps':3,'OOM_NaN_Inf_skipped_updates_observed':False,'pilot_training_started':False,'generation_calls':0,'missing_numeric_losses_and_allocator_peaks_due_failure_path':r is None}
        checkpoint=read(raw/'checkpoint_step2/integrity.json')
        for n,h in checkpoint['files'].items():assert sha(raw/'checkpoint_step2'/n)==h
    a=read(OUT/ARMS[0]/'parameter_hashes_before.json');b=read(OUT/ARMS[1]/'parameter_hashes_before.json')
    # PEFT wraps the additional B encoder projections; compare canonical base names.
    base_a={k.replace('.base_layer.','.'):v for k,v in a.items() if not v['requires_grad']}
    base_b={k.replace('.base_layer.','.'):v for k,v in b.items() if not v['requires_grad']}
    assert len(base_a)==len(base_b)==295 and base_a==base_b
    assert all(b[k]==v for k,v in a.items() if v['requires_grad'])
    assert not any((ROOT/'outputs/stage_i/paired_lora_pilot_runs_v1'/arm/mode).exists() for arm in ARMS for mode in ['train','generation'])
    write(OUT/'AB_COMPARISON.json',{'status':'A_PASS_B_REPLAY_BLOCKER_PILOT_NOT_READY','architectures':comparison,'same_pretrained_frozen_tensor_hashes':True,'same_common_decoder_initialization':True,'new_pilot_training_or_generation':False})
    write(OUT/'GRADIENT_BRANCH_SUMMARY.json',grad_summary);write(OUT/'OPTIMIZER_SCHEDULER_SUMMARY.json',optimizer_summary)
    write(OUT/'REPLAY_COMPARISON.json',{'architectures':drift,'gradient_norm_differences_are_not_parameter_errors':True,'original_step3_tensor_values_not_retained_only_hashes':True,'elementwise_weight_error_distribution_unavailable_without_new_authorized_replay':True})
    from stage_hi_led_data import PackedLEDDataset
    ids=read(PREP/'feasibility_cases.json')['business_ids'];cloud=ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2';manifest=read(cloud/'outputs/stage_h/led_dataset_v1/manifest.json');packed=PackedLEDDataset(manifest['datasets']['train_filtered'],5000,768,root=cloud)
    dataset=ROOT/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/train_filtered.jsonl';selected={r['business_id']:r for r in (json.loads(l) for l in dataset.read_text(encoding='utf-8').splitlines()) if r['business_id'] in ids};assert len(selected)==3
    evidence=[]
    for bid in ids:
        row=selected[bid];assert row['split']=='train' and hashlib.sha256(row['source'].encode()).hexdigest()==row['source_sha256'];f=packed[packed.business_ids.index(bid)]
        evidence.append({'business_id':bid,'frozen_Stage_G_record':row,'input_ids':f['input_ids'],'label_ids':f['labels']})
    assert any(not r['frozen_Stage_G_record']['selected_photo_ids'] for r in evidence)
    write(OUT/'REAL_TRAIN_CONTROLS.json',{'records':evidence,'existing_content_only':True,'new_targets_created':False})
    import inspect
    from transformers.models.led.modeling_led import LEDEncoderSelfAttention
    excerpts='\n\n'.join(inspect.getsource(getattr(LEDEncoderSelfAttention,n)) for n in ['_chunk','_sliding_chunks_query_key_matmul','_sliding_chunks_matmul_attn_probs_value'])
    (OUT/'snapshots/installed_LED_overlapping_attention_excerpt.py').write_text(excerpts,encoding='utf-8')
    write(OUT/'REPORT_DATA.json',{'comparison':comparison,'gradient_branch_summary':grad_summary,'optimizer_scheduler_summary':optimizer_summary})
    print(json.dumps({'status':'A_PASS_B_REPLAY_BLOCKER','architectures':comparison},indent=2))
if __name__=='__main__':main()
