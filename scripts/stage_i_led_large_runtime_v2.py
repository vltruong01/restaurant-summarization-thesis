"""LED-large F contracts and callback implementation. Importing performs no GPU work."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/stage_i_led_large_colab_v2.json'
PREP = ROOT/'outputs/stage_i/led_large_colab_preparation_v2'
ORIGINAL = ROOT/'configs/stage_i_led_large_proposal_v1.json'
LARGE_PREP = ROOT/'outputs/stage_i/led_large_preparation_v1'
RUNNER = ROOT/'scripts/run_stage_i_led_large_colab_v2.py'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(p, value):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False);f.write('\n')


def append(p, value):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def check_config(c):
    old = read(ORIGINAL)
    science = ('seed', 'lora', 'optimization', 'generation', 'budgets', 'runtime_pins',
               'stage_h_manifest', 'final_references', 'training', 'evaluation_epochs',
               'model_id', 'revision', 'target_modules', 'expected_modules',
               'expected_trainable_parameters', 'source_tokenizer', 'source_limit', 'label_limit',
               'global_attention', 'loss', 'initialization', 'checkpoint_policy', 'checkpoint_selection',
               'F_evaluation_outputs', 'technical_gate_generation_outputs_max', 'feasibility', 'timeline')
    require(all(c[k] == old[k] for k in science), 'Frozen proposal setting differs')
    require(c['condition'] == 'filtered' and c['transport_version'] == 2, 'F-only v2 required')
    require(c['model_id'] == 'allenai/led-large-16384' and c['expected_modules'] == 228
            and c['expected_trainable_parameters'] == 4915200, 'Wrong large architecture')
    require(c['weights_sha256'] == 'b31752450099a5eb3fcd5199001c5a6831e4a32e13f17a287a5aa6e10801d081', 'Wrong public checkpoint')
    require(not c['GPU_execution_approved'] and not c['training_authorized'], 'Config must not fabricate approval')
    require(c['D051_strict_replay'] == 'FAIL_UNCHANGED', 'Historical replay evidence changed')
    return c


def check_deadline(a, now=None):
    now = now or datetime.now(timezone.utc)
    start = datetime.fromisoformat(a['execution_authorized_at_utc'])
    end = datetime.fromisoformat(a['execution_deadline_utc'])
    require(start.tzinfo is not None and end.tzinfo is not None, 'UTC-aware approval clock required')
    cap = datetime.fromisoformat(read(CONFIG)['timeline']['proposed_absolute_model_family_freeze_utc'])
    require(start <= now < end <= min(start+timedelta(hours=48), cap), 'Deadline invalid or expired; STOP')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True, 'Researcher must confirm defense reserve')
    return end.isoformat()


def authorize(path, mode, execute):
    require(execute and path is not None, 'GPU disabled: --execute and explicit researcher approval required')
    a = read(path)
    require(a.get('GPU_execution_approved') is True and a.get('researcher_explicit_authorization'),
            'No explicit researcher GPU authorization')
    require(a.get('approval_source') == 'researcher_message' and a.get('experiment') == 'LED_LARGE_F_COLAB_V2',
            'Wrong approval provenance/scope')
    require(mode in a.get('scopes', []) and set(a['scopes']) <= {'feasibility', 'train', 'evaluate'}, 'Mode not approved')
    require(a.get('config_sha256') == sha(CONFIG) and a.get('runner_sha256') == sha(RUNNER)
            and a.get('preflight_manifest_sha256') == sha(PREP/'PREFLIGHT_MANIFEST.json'), 'Approval hash mismatch')
    require(a.get('max_training_updates') == 1250 and a.get('feasibility_updates_max') == 3
            and a.get('feasibility_microbatches_max') == 24 and a.get('technical_generation_calls_max') == 2
            and a.get('evaluation_outputs_max') == 48, 'Execution ceiling differs')
    require(a.get('held_out_access') is False and a.get('Teacher_calls') is False
            and a.get('RO_U_training_authorized') is False, 'Unapproved additional scope')
    require(a.get('conditional_train_after_feasibility_pass') is True, 'Conditional F scope must be explicit')
    check_deadline(a)
    return {**a, '_approval_sha256': sha(path)}


def preflight():
    c = check_config(read(CONFIG))
    actual = {k: importlib.metadata.version(k) for k in c['runtime_pins']}
    require(actual == c['runtime_pins'], 'Pinned ML versions differ; no fallback')
    for rel, h in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel) == h, 'Immutable payload changed: '+rel)
    return c


def binding():
    return dict(config_sha256=sha(CONFIG), runner_sha256=sha(RUNNER),
                preflight_manifest_sha256=sha(PREP/'PREFLIGHT_MANIFEST.json'))


def require_gate(out, a):
    r = read(out/'feasibility/REPORT.json')
    require(r['status'] == 'PASS' and r['binding'] == binding(), 'Matching large GPU feasibility PASS required')
    approved_resume = a.get('resume_after_interruption_approved') is True and a.get('approved_feasibility_report_sha256') == sha(out/'feasibility/REPORT.json')
    require((r['approval_sha256'] == a['_approval_sha256'] or approved_resume)
            and r['deadline_utc'] == a['execution_deadline_utc'], 'Gate/deadline must remain bound to conditional authorization')
    require(not (out/'feasibility/FAILURE_STOP_NO_RETRY.json').exists(), 'Failed gate; stop')
    return r


def generation(c):
    from transformers import GenerationConfig
    expected = read(LARGE_PREP/'EXPECTED_EFFECTIVE_GENERATION_CONFIG.json')
    g = GenerationConfig(**expected)
    semantic = lambda d: {k:v for k,v in d.items() if k not in ('transformers_version', '_from_model_config', '_commit_hash')}
    require(semantic(g.to_dict()) == semantic(expected), 'GenerationConfig serialization drift')
    require(all(getattr(g, k) == v for k,v in c['generation'].items()), 'Beam4 configuration changed')
    require(g.max_new_tokens is None and g.max_length == 768 and g.use_cache, 'Length/cache semantics changed')
    return g


def load_model(c):
    """Explicitly called only after authorization and integrity preflight. CPU load -> verified CUDA transfer."""
    import torch
    from transformers import LEDConfig, LEDForConditionalGeneration, set_seed
    from peft import LoraConfig, get_peft_model
    from unittest.mock import patch
    from stage_i_paired_pilot import initialize_named_adapters
    from run_stage_i_d050_feasibility_v1 import parameters
    require(torch.cuda.is_available(), 'CUDA unavailable; no CPU training fallback')
    require(torch.cuda.device_count() == 1, 'Single GPU protocol required')
    require(torch.cuda.get_device_capability()[0] >= 7, 'Unsupported GPU; stop')
    modeldir = ROOT/c['local_model_directory']
    binary = modeldir/'pytorch_model.bin'
    require(binary.stat().st_size == 1839633783 and sha(binary) == c['weights_sha256'], 'Checkpoint binary differs')
    mc = read(LARGE_PREP/'remote_metadata/official/config.json')
    require(read(modeldir/'config.json') == mc, 'Local large config differs')
    set_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32 = False
    # The public binary is hash-verified before a weights-only CPU load; no remote model code.
    state = torch.load(binary, map_location='cpu', weights_only=True)
    base = LEDForConditionalGeneration(LEDConfig(**mc))
    require(set(state) == set(base.state_dict()), 'Serialized tensor key inventory differs')
    require(all(state[n].shape == t.shape and state[n].dtype == t.dtype for n,t in base.state_dict().items()),
            'Serialized tensor dimensions/precision differ')
    base.load_state_dict(state, strict=True);del state
    base.tie_weights()
    # Give PEFT the verified LOCAL config for serialization/vocabulary checks, not the hub card's './'.
    base.name_or_path = str(modeldir)
    require(sum(p.numel() for p in base.parameters()) == 459801600, 'Unique backbone parameter count differs')
    canonical_before = parameters(base)
    expected = read(LARGE_PREP/'META_LORA_INVENTORY.json')
    raw = {n:m for n,m in base.named_modules() if isinstance(m, torch.nn.Linear)}
    names = {m['name'] for m in expected['modules']}
    require(set(raw) == names|{'lm_head'} and base.get_output_embeddings() is raw['lm_head'], 'Linear inventory/head differs')
    require(base.get_input_embeddings().weight is raw['lm_head'].weight, 'Expected tied embeddings differ')
    with patch('peft.tuners.lora.model.is_bnb_available', return_value=False), \
         patch('peft.tuners.lora.model.is_bnb_4bit_available', return_value=False):
        model = get_peft_model(base, LoraConfig(**c['lora'], target_modules='all-linear'))
    scope = inspect_model(model, expected)
    init = initialize_named_adapters(model, c['seed'])
    require(init == read(PREP/'INITIALIZATION_HASHES.json')['per_module_sha256'], 'Named adapter initialization differs')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    model.generation_config = generation(c)
    model.to('cuda');set_seed(c['seed'])
    verify_backbone(model, canonical_before)
    return model, canonical_before, dict(scope=scope, initialization=init,
        checkpoint_binary_sha256=c['weights_sha256'], pretrained_tensor_inventory_verified=True)


def inspect_model(model, expected):
    import torch
    names = {x['name'] for x in expected['modules']}
    require(set(model.targeted_module_names) == names, 'PEFT target_modules resolution differs')
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    require(len(trainable) == 456 and sum(p.numel() for p in trainable.values()) == 4915200, 'Adapter tensor/parameter count differs')
    require(all('.lora_A.' in n or '.lora_B.' in n for n in trainable), 'Backbone/head unexpectedly trainable')
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'FP32 parameter storage changed')
    for info in expected['modules']:
        m = model.get_submodule('base_model.model.'+info['name'])
        require(list(m.lora_A['default'].weight.shape) == info['A_shape']
                and list(m.lora_B['default'].weight.shape) == info['B_shape']
                and m.scaling['default'] == 2 and not m.disable_adapters, 'Adapter geometry/activation differs')
    return dict(modules=228, trainable_tensors=456, trainable_parameters=4915200, frozen_backbone_parameters=459801600)


def verify_backbone(model, before):
    from run_stage_i_d050_feasibility_v1 import parameters
    after = parameters(model)
    after = {n.removeprefix('base_model.model.').replace('.base_layer.', '.'):v
             for n,v in after.items() if not v['requires_grad']}
    require(set(after) == set(before), 'Frozen parameter names differ')
    require(all(not v['requires_grad'] and v['sha256'] == before[n]['sha256'] for n,v in after.items()),
            'Frozen pretrained backbone changed')
    return True


def optimizer_scheduler(model, c):
    import torch
    from transformers import get_linear_schedule_with_warmup
    o = c['optimization']
    opt = torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad], 'weight_decay':o['weight_decay']},
                            {'params':[], 'weight_decay':0.}], lr=o['learning_rate'],
                           betas=(o['adam_beta1'],o['adam_beta2']), eps=o['adam_epsilon'])
    return opt, get_linear_schedule_with_warmup(opt,63,1250)


def training_arguments(c, out):
    from stage_i_d058 import training_arguments as original
    return original(c, out)


def callback_class():
    from transformers import TrainerCallback
    class Record(TrainerCallback):
        def __init__(self, folder, approval, memory):
            self.folder, self.approval, self.memory = folder, approval, memory
            self.trainer = None

        def on_train_begin(self, args, state, control, **kw):
            require(state.max_steps == 1250 and state.global_step <= 1250, 'Schedule differs')
            check_deadline(self.approval)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            from stage_i_paired_pilot import gradient_record
            gradients = gradient_record(model, expect_later_A=state.global_step>=2)
            require(len(gradients) == 456, 'Missing large adapter gradients')
            append(self.folder/'gradient_checks.jsonl', dict(step=state.global_step+1, gradients=gradients))

        def on_optimizer_step(self, args, state, control, optimizer=None, **kw):
            require(self.trainer is not None and not self.trainer.accelerator.optimizer_step_was_skipped, 'AMP skipped update')
            require({int(v['step']) for v in optimizer.state.values()} == {state.global_step+1}, 'Optimizer state step differs')

        def on_step_end(self, args, state, control, lr_scheduler=None, **kw):
            check_deadline(self.approval);self.memory.check()
            require(state.global_step <= 1250, 'Update ceiling exceeded')
            append(self.folder/'progress.jsonl', dict(step=state.global_step, epoch=state.epoch,
                   lr=lr_scheduler.get_last_lr(), utc=datetime.now(timezone.utc).isoformat(), **self.memory.summary()))

        def on_save(self, args, state, control, **kw):
            require(state.global_step in (250,500,750,1000,1250), 'Only five approved epoch checkpoints')
            p = self.folder/f'checkpoint-{state.global_step}'
            needed = ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                      'scaler.pt','rng_state.pth','trainer_state.json')
            require(all((p/n).is_file() for n in needed), 'Incomplete epoch checkpoint')
            write(p/'checkpoint_integrity.json', dict(binding=binding(), global_step=state.global_step,
                  files={q.name:sha(q) for q in p.iterdir() if q.is_file()}))
            append(self.folder/'CHECKPOINT_RECEIPTS.jsonl', dict(step=state.global_step,
                   checkpoint_integrity_sha256=sha(p/'checkpoint_integrity.json')))
    return Record
