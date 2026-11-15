"""Offline inspection only: CPU checkpoint arithmetic, meta architecture, saved texts. No forward/generate/train/API."""
from collections import Counter
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LEDConfig, LEDForConditionalGeneration, LEDTokenizerFast, GenerationConfig, Trainer
from transformers.models.led.modeling_led import shift_tokens_right, LEDEncoder
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from peft.tuners.tuners_utils import check_target_module_exists

from stage_hi_led_data import LEDCompleteCollator
from stage_i_decoding_diagnostics import (CONDITIONS, read_jsonl, write_json, text_diagnostics,
                                        sections, distribution, words, captions)

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
RUNS = CLOUD / 'outputs/stage_i/runs_v3'
PREP = ROOT / 'outputs/stage_i/decoding_diagnostic_v1'
OUT = ROOT / 'outputs/stage_i/training_failure_audit_v1'
CONFIG = CLOUD / 'configs/stage_i_lora_led_training_approved_v2.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def probe(config, mc, target_modules):
    with torch.device('meta'):
        base = LEDForConditionalGeneration(LEDConfig(**mc))
        before = {n: [m.in_features, m.out_features] for n,m in base.named_modules() if isinstance(m, torch.nn.Linear)}
        base_count = sum(p.numel() for p in base.parameters())
        lora = {**config['lora'], 'target_modules': target_modules}
        model = get_peft_model(base, LoraConfig(**lora))
        matched = list(model.targeted_module_names)
        trainable = [{'name':n,'shape':list(p.shape),'parameters':p.numel()} for n,p in model.named_parameters() if p.requires_grad]
        require(all(p.device.type == 'meta' for p in model.parameters()), 'Architecture probe allocated real weights')
        require(all('lora_' in p['name'] for p in trainable), 'Unintended base parameter unfrozen')
    count = sum(p['parameters'] for p in trainable)
    enc = [n for n in matched if '.encoder.' in n]
    dec = [n for n in matched if '.decoder.' in n]
    expected = sum(config['lora']['r'] * sum(before[n]) for n in matched)
    require(count == expected, 'LoRA parameter arithmetic differs')
    return {'matched_modules':matched,'encoder_modules':enc,'decoder_modules':dec,
            'matched_module_count':len(matched),'encoder_count':len(enc),'decoder_count':len(dec),
            'base_parameter_count':base_count,'trainable_parameter_count':count,
            'trainable_fraction_percent':100 * count/(base_count+count),
            'trainable_tensors':trainable,'all_encoder_base_weights_frozen':True,
            'all_base_weights_frozen':True,'meta_only':True,'pretrained_weights_loaded':False,
            'forward_backward_or_generation':False, 'target_modules':target_modules}


def main():
    OUT.mkdir(exist_ok=False)
    config = read(CONFIG)
    manifest_path = CLOUD / config['stage_h_manifest']
    require(sha(manifest_path) == config['stage_h_manifest_sha256'], 'Stage H manifest changed')
    manifest = read(manifest_path)
    backups = read(CLOUD / 'BACKUP_MANIFEST.json')
    for e in backups['files']:
        require(sha(CLOUD/e['path']) == e['sha256'], 'Original archive file changed: '+e['path'])
    versions = {p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','numpy','tokenizers','safetensors']}
    pins = read(PREP / 'decoding_configurations.json')['required_runtime_versions']
    require(all(versions[k].split('+')[0] == v for k,v in pins.items()), 'Installed audit runtime drift')
    mc = read(CLOUD / 'assets/models/led-base-16384/config.json') if (CLOUD / 'assets/models/led-base-16384/config.json').is_file() else read(CLOUD / read(CLOUD/config['model_config_manifest'])['config_file'])
    current = probe(config, mc, config['lora']['target_modules'])
    require(current['encoder_count'] == 0 and current['decoder_count'] == 24 and current['trainable_parameter_count'] == 294912, 'Unexpected current LoRA scope')
    proposed_regex = r'led\.(?:encoder\.layers\.\d+\.self_attn\.longformer_self_attn\.(?:query|value|query_global|value_global)|decoder\.layers\.\d+\.(?:self_attn|encoder_attn)\.(?:q_proj|v_proj))'
    proposed = probe(config, mc, proposed_regex)
    require(proposed['encoder_count'] == proposed['decoder_count'] == 24 and proposed['trainable_parameter_count'] == 589824, 'Unexpected proposed LoRA scope')
    architecture = {'current':current,'proposed_not_approved':proposed,
                    'module_shape_formula':'For each 768x768 projection: r*(768+768)=12288 at r8. Current24=294912; proposed48=589824.',
                    'expected_decoder_attention_types':['self_attn','encoder_attn'],
                    'encoder_base_freeze_does_not_mean_encoder_ignored':True}
    write_json(OUT / 'architecture_audit.json', architecture)
    # Save exact installed implementation evidence without executing model methods.
    code_dir = OUT / 'library_sources'; code_dir.mkdir()
    for name, obj in [('Trainer_training_step.py.txt',Trainer.training_step),
                      ('Trainer_create_accelerator.py.txt',Trainer.create_accelerator_and_postprocess),
                      ('Accelerator_backward.py.txt',Accelerator.backward),
                      ('LED_forward.py.txt',LEDForConditionalGeneration.forward),
                      ('LED_encoder_forward.py.txt',LEDEncoder.forward),
                      ('PEFT_module_matching.py.txt',check_target_module_exists)]:
        (code_dir/name).write_text(inspect.getsource(obj),encoding='utf-8')
    # Actual optimizer, scheduler, AMP and argument snapshots; no optimizer or forward is executed.
    optimization = {}
    for c in CONDITIONS:
        folder = RUNS/c
        state = read(folder/'trainer_state.json'); completion = read(folder/'completion_report.json')
        require(state['global_step'] == state['max_steps'] == 1250 and state['epoch'] == 5, 'Unexpected training duration')
        ckpt = folder/'checkpoint-1250'
        optimizer = torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=False)
        scheduler = torch.load(ckpt/'scheduler.pt',map_location='cpu',weights_only=False)
        scaler = torch.load(ckpt/'scaler.pt',map_location='cpu',weights_only=False)
        args = torch.load(ckpt/'training_args.bin',map_location='cpu',weights_only=False)
        required_args = ['per_device_train_batch_size','per_device_eval_batch_size','gradient_accumulation_steps','num_train_epochs',
                         'learning_rate','weight_decay','warmup_ratio','lr_scheduler_type','optim','fp16','bf16',
                         'gradient_checkpointing','gradient_checkpointing_kwargs','label_names','remove_unused_columns',
                         'label_smoothing_factor','predict_with_generate','load_best_model_at_end','metric_for_best_model',
                         'greater_is_better','max_grad_norm','eval_strategy','save_strategy','save_total_limit','seed','data_seed']
        arg_values = {k:getattr(args,k) for k in required_args}
        arg_values = json.loads(json.dumps(arg_values,default=str))
        require(args.label_names == ['labels'] and args.gradient_accumulation_steps == 8 and args.per_device_train_batch_size == 1, 'Training args differ')
        steps = sorted({float(v['step']) for v in optimizer['state'].values()})
        require(len(optimizer['state']) == 48 and steps == [1250.0], 'Adapter optimizer step evidence differs')
        require(all(torch.isfinite(s[k]).all() for s in optimizer['state'].values() for k in ['exp_avg','exp_avg_sq']), 'Nonfinite optimizer state')
        require(scheduler['last_epoch'] == 1250 and scheduler['_last_lr'] == [0.0,0.0], 'Scheduler final state differs')
        warmup = math.ceil(1250*.05)
        logs = [x for x in state['log_history'] if 'learning_rate' in x]
        def lr(t):
            return .0002 * (t/max(1,warmup) if t < warmup else max(0,(1250-t)/(1250-warmup)))
        lr_errors = [abs(x['learning_rate'] - lr(x['step']-1)) for x in logs]
        require(max(lr_errors) < 1e-12, 'Logged used-update learning rates differ from expected schedule')
        evals = [x for x in state['log_history'] if 'eval_loss' in x]
        require([x['step'] for x in evals] == [250,500,750,1000,1250], 'Missing epoch evaluation')
        require(min(x['eval_loss'] for x in evals) == state['best_metric'] == completion['best_eval_loss'], 'Best loss selection differs')
        best = load_file(folder/'best_adapter/adapter_model.safetensors')
        last = load_file(ckpt/'adapter_model.safetensors')
        previous = load_file(folder/'checkpoint-1000/adapter_model.safetensors')
        require(set(best) == set(last) == set(previous) and len(best) == 48, 'Adapter tensor scope differs')
        require(all(torch.equal(best[k], last[k]) and torch.isfinite(best[k]).all() for k in best), 'Best checkpoint differs/nonfinite')
        changed = sum(not torch.equal(previous[k],last[k]) for k in best)
        require(changed == 48, 'Adapter training stalled')
        optimization[c] = {'training_arguments':arg_values,'trainer_state':state,'completion':completion,
            'optimizer_groups':[{**{k:v for k,v in g.items() if k!='params'},'parameter_tensor_count':len(g['params'])} for g in optimizer['param_groups']],
            'optimizer_tensor_state_count':len(optimizer['state']),'optimizer_step_values':steps,
            'all_optimizer_moments_finite':True,'nonzero_first_moment_tensors':sum(torch.count_nonzero(s['exp_avg']).item()>0 for s in optimizer['state'].values()),
            'scheduler':scheduler,'scaler':scaler,'warmup_steps':warmup,
            'LR_log_entries_matching_pre_update_schedule':len(logs),'maximum_LR_error':max(lr_errors),
            'LR_log_interpretation':'Log at global step n records the LR used before scheduler advanced from n-1 to n; final stored LR is zero.',
            'best_adapter_equals_checkpoint1250':True,'changed_adapter_tensors_1000_to_1250':changed,
            'all_48_optimizer_steps_1250_support_no_skipped_updates':'Consistent with1250 successful steps; not a captured per-microbatch gradient replay.'}
        del optimizer,best,last,previous
    write_json(OUT/'optimization_audit.json',optimization)
    tokenizer = LEDTokenizerFast.from_pretrained(ROOT/'assets/tokenizers/led-base-16384',local_files_only=True)
    collator = LEDCompleteCollator(5000,768,1024)
    datasets = {}; teacher_stats = {}; train_f_records = None; sample_batches=[]
    markers = ['food opinions are mixed','service experiences vary','value opinions vary','fresh, flavorful','on a table']
    row_bindings = []
    for split in ['train','validation']:
        for c in CONDITIONS:
            info=manifest['datasets'][f'{split}_{c}']; path=CLOUD/info['path']; rawpath=ROOT/info['stage_g_jsonl']
            require(sha(path)==info['sha256'] and sha(rawpath)==info['stage_g_jsonl_sha256'],'Accepted dataset changed')
            with np.load(path,allow_pickle=False) as z: pack={k:z[k] for k in z.files}
            records=read_jsonl(rawpath); ids=pack['business_ids'].tolist()
            require([r['business_id'] for r in records]==ids and len(ids)==len(set(ids))==info['rows'],'Dataset IDs/order differ')
            if c=='reviews_only':
                common_ids=ids; shared={r['business_id']:(r['source'].split('[REVIEWS]',1)[1].split('[VISUAL_CONTEXT]',1)[0].rstrip(),sections(r['target'])[0],r['selected_review_ids']) for r in records}
            else:
                require(ids==common_ids,'Cross-condition order differs')
                for r in records:
                    require(shared[r['business_id']]==(r['source'].split('[REVIEWS]',1)[1].split('[VISUAL_CONTEXT]',1)[0].rstrip(),sections(r['target'])[0],r['selected_review_ids']),'Shared review inputs/consensus differ')
            lengths=np.diff(pack['label_offsets']); sources=np.diff(pack['input_offsets']); n=len(ids)
            require(max(lengths)<=768 and max(sources)<=5000,'Unexpected source/target overflow')
            for key,off in [('input_ids','input_offsets'),('labels','label_offsets')]:
                require(np.all(pack[key][pack[off][:-1]]==0) and np.all(pack[key][pack[off][1:]-1]==2),'Incomplete source/label BOS/EOS')
            # Revalidate worst-case and empty-caption examples with the actual collator/shift helper on CPU.
            indices={0,int(lengths.argmax()),int(sources.argmax())}
            zeros=np.flatnonzero(pack['caption_counts']==0) if c!='reviews_only' else []
            if len(zeros): indices.add(int(zeros[0]))
            for i in sorted(indices):
                f={key:pack[raw][pack[off][i]:pack[off][i+1]].astype(np.int64).tolist() for key,raw,off in [('input_ids','input_ids','input_offsets'),('labels','labels','label_offsets')]}
                batch=collator([f]); shifted=shift_tokens_right(batch['labels'],1,2); il,ll=len(f['input_ids']),len(f['labels'])
                require(batch['input_ids'][0,:il].tolist()==f['input_ids'] and batch['labels'][0,:ll].tolist()==f['labels'],'Collator altered content')
                require(torch.all(batch['labels'][0,ll:]==-100) and shifted[0,0].item()==2 and shifted[0,1:ll].tolist()==f['labels'][:-1],'Padding/decoder shift differs')
                require(batch['global_attention_mask'][0,0]==1 and batch['global_attention_mask'].sum()==1 and batch['attention_mask'].sum()==il,'Source/global attention differs')
                require(torch.all(batch['attention_mask'][0,il:]==0),'Padding attended')
                sample_batches.append({'split':split,'condition':c,'business_id':ids[i],'source_shape':list(batch['input_ids'].shape),'label_shape':list(batch['labels'].shape),'PASS':True})
            diagnostics=[text_diagnostics(r['target']) for r in records]
            empty_visual=sum(not sections(r['target'])[1] for r in records) if c!='reviews_only' else None
            zero_ids=[r['business_id'] for r in records if c=='filtered' and not r['selected_photo_ids']]
            require(all(not sections(r['target'])[1] for r in records if r['business_id'] in zero_ids),'Zero Filtered teacher target not empty')
            label_headers=sum(r['target'].startswith('[REVIEW_CONSENSUS]\n') and r['target'].count('[REVIEW_CONSENSUS]')==1 and (c=='reviews_only' or '\n\n[VISUAL_EVIDENCE]\n' in r['target']) for r in records)
            tstats={'targets':n,'exact_repeated_sentence_outputs':sum(d['exact_repeated_sentence'] for d in diagnostics),
                'adjacent_word_repeated_outputs':sum(d['consecutive_repeated_word'] for d in diagnostics),
                'mean_adjacent_word_pair_fraction':sum(d['consecutive_repeated_word_pair_fraction'] for d in diagnostics)/n,
                'literal_review_visual_headers_valid':label_headers,'target_tokens':distribution(lengths.tolist()),
                'source_tokens':distribution(sources.tolist()),'empty_visual_targets':empty_visual,
                'zero_caption_filtered_businesses':zero_ids,
                'review_phrase_presence_counts':{m:sum(m in sections(r['target'])[0].lower() for r in records) for m in markers}}
            teacher_stats[f'{split}_{c}']=tstats
            datasets[f'{split}_{c}']={'rows':n,'NPZ_sha256':sha(path),'JSONL_sha256':sha(rawpath),'business_ids':ids,
                'complete_BOS_EOS_all_rows':True,'source_limit':5000,'label_limit':768,'row_order_alignment':True}
            for i,r in enumerate(records):
                row_bindings.append({'split':split,'condition':c,'business_id':r['business_id'],'source_sha256':r['source_sha256'],
                    'target_sha256':hashlib.sha256(r['target'].encode()).hexdigest(),'input_tokens':int(sources[i]),'label_tokens':int(lengths[i])})
            if split=='train' and c=='filtered':train_f_records=records
    require(not set(datasets['train_filtered']['business_ids']) & set(datasets['validation_filtered']['business_ids']),'Train/validation overlap')
    write_json(OUT/'dataset_target_audit.json',{'datasets':datasets,'teacher_target_statistics':teacher_stats,'sample_collator_shift_checks':sample_batches,
              'teacher_is_not_independent_factual_ground_truth':True,'held_out_files_read':False})
    with (OUT/'input_target_bindings.jsonl').open('x',encoding='utf-8') as f:
        for row in row_bindings:f.write(json.dumps(row)+'\n')
    # Same predeclared validation IDs. Analyze saved texts; no generation.
    cases=read_jsonl(PREP/'cases.jsonl'); subset=read(PREP/'subset_manifest.json')
    saved={} ; text_rows=[]
    for name,folder in [('A','decoding_greedy_A_local_v2'),('B','decoding_beam_ngram_B_local_v1')]:
        saved[name]={(p['condition'],p['business_id']):p for c in CONDITIONS for p in read_jsonl(ROOT/'outputs/stage_i'/folder/c/'predictions.jsonl')}
    for case in cases:
        key=case['condition'],case['business_id']
        for name,record in [('baseline',case['baseline_prediction']),('A',saved['A'][key]),('B',saved['B'][key]),('teacher',{'text':case['target']})]:
            review,visual=sections(record['text'])
            text_rows.append({'business_id':case['business_id'],'condition':case['condition'],'subset':case['subset'],'setting':name,
                'word_count':len(words(record['text'])),**text_diagnostics(record['text']),
                'review_phrase_presence':{m:m in review.lower() for m in markers},
                'visual_claims_when_zero_filtered_captions':case['condition']=='filtered' and not case['selected_photo_ids'] and bool(visual),
                'literal_headers':re.findall(r'\[[A-Z_]+\]',record['text'])})
    write_json(OUT/'saved_prediction_audit.json',{'sample':subset,'rows':text_rows,
        'qualitative_evidence_reused':'See unchanged B22-note/A18-note evidence plus full72 source/teacher/Baseline/A/B comparisons',
        'lower_repetition_not_proof_of_quality':True})
    # Proposed, non-executed pilot: a natural-distribution stratified train subset, not validation training.
    # 512 deterministic train IDs stratified on caption count and source length; no outcome-based selection.
    lengths={r['business_id']:r['source_tokens'] for r in train_f_records}
    quartiles=np.quantile(list(lengths.values()),[.25,.5,.75]).tolist()
    strata={}
    for r in train_f_records:
        cap=len(r['selected_photo_ids']); cb='0' if cap==0 else '1-2' if cap<=2 else '3-5'
        q=int(np.searchsorted(quartiles,r['source_tokens'],side='right'))
        strata.setdefault(cb+'_lengthQ'+str(q+1),[]).append(r['business_id'])
    allocations={k:math.floor(len(v)*512/2000) for k,v in strata.items()}
    remainder=512-sum(allocations.values())
    ranked=sorted(strata,key=lambda k:(-(len(strata[k])*512/2000-allocations[k]),k))
    for key in ranked[:remainder]:allocations[key]+=1
    rng=random.Random(20261016)
    chosen=[]
    for k in sorted(strata):chosen+=rng.sample(sorted(strata[k]),allocations[k])
    require(len(chosen)==len(set(chosen))==512,'Pilot sampling count differs')
    ordered=[bid for bid in datasets['train_filtered']['business_ids'] if bid in set(chosen)]
    pilot={'status':'PROPOSED_ONLY_NOT_AUTHORIZED_OR_EXECUTED','training_condition':'filtered','sample_seed':20261016,
        'training_business_ids':ordered,'training_count':512,'sampling':'Largest-remainder proportional allocation over caption count(0,1-2,3-5) x full source length quartiles; fixed-seed within-stratum draw; preserve original row order. No model-output-based selection.',
        'source_quartile_boundaries':quartiles,'strata':{k:{'population':len(v),'sample':allocations[k]} for k,v in sorted(strata.items())},
        'selected_zero_filtered_train_ids':[r['business_id'] for r in train_f_records if r['business_id'] in chosen and not r['selected_photo_ids']],
        'validation_loss_business_ids':datasets['validation_filtered']['business_ids'],
        'generation_evaluation_random20':subset['random_business_ids'],'generation_evaluation_targeted4':[r['business_id'] for r in subset['targeted_businesses']],
        'epochs':2,'microbatches_per_arm':1024,'optimizer_steps_per_arm':128,'effective_batch':8,'warmup_steps':7,
        'matched_arms':['decoder_only_fresh_pilot','encoder_decoder_fresh_pilot'],
        'existing_full_F_checkpoint_primary_practical_reference_not_compute_matched':True,
        'no_backprop_on_validation':True,'train_targets_or_splits_modified':False}
    write_json(OUT/'proposed_pilot_manifest.json',pilot)
    (OUT/'training_runner_original_to_cloud_v2.diff').write_text(''.join(difflib.unified_diff(
        (CLOUD/'scripts/run_stage_i_lora_led.py').read_text(encoding='utf-8').splitlines(True),
        (CLOUD/'scripts/run_stage_i_lora_led_cloud_v2.py').read_text(encoding='utf-8').splitlines(True),fromfile='original_runner',tofile='actual_cloud_v2')),encoding='utf-8')
    sources={'training_config':str(CONFIG.relative_to(ROOT)),'training_config_sha256':sha(CONFIG),
        'actual_runner_sha256':sha(CLOUD/'scripts/run_stage_i_lora_led_cloud_v2.py'),
        'actual_data_helper_sha256':sha(CLOUD/'scripts/stage_hi_led_data.py'),
        'installed_versions':versions,'existing_imported_archive_files_unchanged':len(backups['files']),
        'new_model_forward_backward_generation_training_teacher_calls':0,'pretrained_weights_loaded_for_probe':False,
        'checkpoint_states_read_CPU_only':True,'held_out_content_read':False,'scientific_files_modified':False}
    write_json(OUT/'AUDIT_INTEGRITY.json',sources)
    print(json.dumps({'status':'OFFLINE_AUDIT_COMPLETE_NO_FORWARD_OR_TRAINING','current_parameters':current['trainable_parameter_count'],
        'proposed_parameters':proposed['trainable_parameter_count'],'teacher_stats':teacher_stats,'pilot_zero_train_count':len(pilot['selected_zero_filtered_train_ids'])},indent=2))


if __name__=='__main__':
    main()
