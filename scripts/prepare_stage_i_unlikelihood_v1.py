"""Build train-only negative views and offline bindings; never load a model."""
import importlib.metadata
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from stage_i_unlikelihood_v1 import *


def main():
    require(not (PREP/'PREFLIGHT_MANIFEST.json').exists(), 'Preparation frozen; do not overwrite')
    PREP.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer
    import numpy as np
    from safetensors import safe_open
    old = read(ROOT/'configs/stage_i_d060_ro_to_filtered_v1.json')
    protected = dict(read(ROOT/'outputs/stage_i/d060_preparation_v1/PREFLIGHT_MANIFEST.json')['files'])
    for rel, digest in protected.items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel) == digest, 'Historical binding changed: '+rel)
    manifest = read(ROOT/old['stage_h_manifest'])
    tm = read(ROOT/manifest['tokenizer_manifest'])
    tokenizer = Tokenizer.from_file(str(ROOT/tm['relative_directory']/'tokenizer.json'))
    datasets = {}
    for split in ('train', 'validation'):
        info = manifest['datasets'][split+'_filtered']
        require(sha(ROOT/info['stage_g_jsonl']) == info['stage_g_jsonl_sha256'], 'Source/label file changed')
        require(sha(ROOT/info['path']) == info['sha256'], 'Packed dataset changed')
        datasets[split] = [json.loads(s) for s in (ROOT/info['stage_g_jsonl']).read_text(encoding='utf-8').splitlines()]
        require(len(datasets[split]) == (2000 if split == 'train' else 200), 'Wrong business count')
    train, val = datasets['train'], datasets['validation']
    ti, vi = [r['business_id'] for r in train], [r['business_id'] for r in val]
    require(len(set(ti)) == 2000 and len(set(vi)) == 200 and not set(ti)&set(vi), 'Split/ID overlap')
    # Fixed hash ranking, no score/output/label quality used in sampling.
    selected = sorted(ti, key=lambda b: stable_key(42, b, 'UL_train512'))[:512]
    positions = {b: i for i, b in enumerate(ti)}
    views = []
    for rank, business in enumerate(selected):
        row = train[positions[business]]
        text = row['target']
        ids = tokenizer.encode(text).ids
        require(len(ids) <= 768 and len(tokenizer.encode(row['source']).ids) <= 5000, 'Overflow; no truncation')
        kind = 'sentence' if rank % 2 == 0 else 'word'
        view = build_negative(text, tokenizer, business, kind)
        view.update(business_id=business, original_row_index=positions[business],
                    source_sha256=row['source_sha256'], positive_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        views.append(view)
    # Packed positives must match every original token, including BOS/EOS.
    pack_path = ROOT/manifest['datasets']['train_filtered']['path']
    with np.load(pack_path, allow_pickle=False) as z:
        labels, offsets = z['labels'], z['label_offsets']
        require(z['business_ids'].tolist() == ti, 'Packed mapping differs')
        for v in views:
            i = v['original_row_index']
            require(labels[offsets[i]:offsets[i+1]].tolist() == tokenizer.encode(train[i]['target']).ids,
                    'Positive label mismatch')
    root = ROOT/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2/outputs/stage_i/runs_v3/filtered/best_adapter'
    ac = read(root/'adapter_config.json')
    inventory = {}
    with safe_open(root/'adapter_model.safetensors', framework='numpy') as weights:
        for name in weights.keys():
            value = weights.get_tensor(name)
            require(np.isfinite(value).all(), 'Nonfinite initial adapter')
            require('.decoder.' in name and any(f'.{p}.' in name for p in ('q_proj', 'v_proj')), 'Wrong adapter scope')
            inventory[name] = {'shape': list(value.shape), 'parameters': int(value.size)}
    require(len(inventory) == 48 and sum(v['parameters'] for v in inventory.values()) == 294912,
            'Historical F adapter geometry changed')
    require(ac['r'] == 8 and ac['lora_alpha'] == 16 and set(ac['target_modules']) == {'q_proj','v_proj'}, 'LoRA differs')
    require(ac['base_model_name_or_path'] == old['model_id'], 'Foreign backbone')
    cfg = {k: old[k] for k in ('model_id','revision','seed','lora','generation','budgets','runtime_pins','stage_h_manifest','final_references')}
    cfg.update(experiment='post_D060_unlikelihood_paired_v1', status='OFFLINE_PREPARED_NOT_GPU_AUTHORIZED',
               GPU_execution_approved=False, training_authorized=False, condition='filtered',
               initial_adapter=root.relative_to(ROOT).as_posix(),
               initial_adapter_sha256=sha(root/'adapter_model.safetensors'),
               initial_adapter_config_sha256=sha(root/'adapter_config.json'),
               expected_modules=24, expected_trainable_parameters=294912,
               arms={'control_ce':0.0, 'treatment_ul':0.1},
               negative_design='one_teacher_derived_prefix_per_business; alternating_sentence_and_word; no_model_generation',
               optimization={**old['optimization'], 'num_train_epochs':2, 'save_total_limit':2},
               training_businesses=512, validation_businesses=200, training_steps_per_arm=128,
               warmup_steps=7, evaluation_epochs=[2], generation_outputs_total=48,
               gate_updates_per_arm=3, gate_microbatches_per_arm=24, headroom_minimum=0.15,
               run_root='outputs/stage_i/unlikelihood_paired_run_v1',
               new_execution_deadline_utc=None, prior_deadline_override_approved=False,
               old_deadlines_preserved={'D060':'2026-10-22T07:11:22.846767+00:00','D059':'2026-10-25T15:10:48.941817+00:00'},
               D051_strict_replay='FAIL_UNCHANGED', held_out_access=False, Teacher_calls=False,
               primary_validation_interpretation='fixed24 repeatedly inspected DEVELOPMENT cases; not independent confirmation')
    write(CONFIG, cfg)
    write(PREP/'TRAINING_MANIFEST.json', {'seed':42,'method':'SHA256 ranking seed|UL_train512|business_id; no stratification',
        'business_ids':selected,'row_indices':[positions[b] for b in selected],
        'zero_caption_businesses':sum(not train[positions[b]]['selected_photo_ids'] for b in selected),
        'epochs':[[*sorted(range(512), key=lambda i: stable_key(42+e, selected[i], 'UL_epoch_order'))] for e in range(2)]})
    write(PREP/'NEGATIVE_VIEWS.jsonl', '\n'.join(json.dumps(v, ensure_ascii=False) for v in views))
    cases = [json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(cases)==24 and all(c['business_id'] in vi for c in cases), 'Wrong development evaluation cases')
    write(PREP/'EVALUATION_MANIFEST.json', {'business_ids':[c['business_id'] for c in cases],
        'subsets':{c['business_id']:c['subset'] for c in cases}, 'epoch':2,
        'development_only':True,'confirmation_cohort':'NOT_SELECTED; exposure audit and separate approval needed before independent claim'})
    write(PREP/'ADAPTER_INVENTORY.json', {'actual_tensors':inventory,'actual_trainable_parameter_count':294912,
        'actual_projection_module_count':24,'verification':'serialized tensor inventory only; no new model attachment or gradient execution'})
    runtime = {k:importlib.metadata.version(k) for k in cfg['runtime_pins']}
    require(runtime==cfg['runtime_pins'], 'Pinned runtime drift')
    write(PREP/'OFFLINE_PREFLIGHT_REPORT.json', {'status':'PASS_OFFLINE_NO_MODEL_EXECUTION',
        'historical_file_bindings':len(protected), 'positive_label_roundtrips':512,
        'eligible_sentence_negatives':sum(v['eligible'] and v['kind']=='sentence' for v in views),
        'eligible_word_negatives':sum(v['eligible'] and v['kind']=='word' for v in views),
        'skipped_negative_views':[{'business_id':v['business_id'],'kind':v['kind'],'reason':v.get('reason')} for v in views if not v['eligible']],
        'negative_mask_tokens':sum(v.get('penalized_tokens',0) for v in views),
        'max_negative_tokens':max(len(v.get('negative_ids',[])) for v in views),
        'runtime_pins':runtime,'train_validation_disjoint':True,'teacher_positive_content_changed':False,
        'GPU_operations':0,'model_loads':0,'model_forward_backward_generation':0,'optimizer_steps':0,
        'synthetic_CPU_autograd_tests':'separate unit tests; no LED loaded', 'D051_strict_replay':'FAIL_UNCHANGED'})
    newfiles = [CONFIG,RUNNER,ROOT/'scripts/stage_i_unlikelihood_v1.py',Path(__file__),
        ROOT/'tests/test_stage_i_unlikelihood_v1.py',ROOT/'specs/009p_unlikelihood_paired_pilot_v1.md',
        ROOT/'thesis_sdd_docs/UNLIKELIHOOD_PILOT_PLAN_V1.md'] + list(PREP.glob('*.json')) + list(PREP.glob('*.jsonl'))
    for p in newfiles:
        protected[p.relative_to(ROOT).as_posix()] = sha(p)
    # Reverify protected data after construction; none of these files are edited.
    for rel, digest in protected.items():
        require(sha(Path(rel) if Path(rel).is_absolute() else ROOT/rel)==digest, 'Integrity changed during preparation')
    write(PREP/'PREFLIGHT_MANIFEST.json', {'experiment':cfg['experiment'],'files':protected,'GPU_execution_approved':False})
    write(PREP/'AUTHORIZATION_TEMPLATE_NOT_APPROVAL.json', {
        'experiment':cfg['experiment'],'GPU_execution_approved':False,'approval_source':None,
        'researcher_explicit_authorization':None,'execution_authorized_at_utc':None,'new_execution_deadline_utc':None,
        'prior_stop_loss_override_explicit':False,'approved_arms':['control_ce','treatment_ul'],
        'approved_modes':['feasibility','train','evaluate'],'max_training_updates_total':256,
        'max_gate_updates_total':6,'max_generation_outputs_total':48,'held_out_access':False,'Teacher_calls':False,
        'config_sha256':sha(CONFIG),'preflight_manifest_sha256':sha(PREP/'PREFLIGHT_MANIFEST.json'),
        'runner_sha256':sha(RUNNER)})
    print(json.dumps(read(PREP/'OFFLINE_PREFLIGHT_REPORT.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
