"""One bounded, CPU-only investigation. Never loads pretrained weights or CUDA.

The reduced-width LED fixture is not a numerical qualification of the real model.
No execution runner, objective, threshold, or historical artifact is patched.
"""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse, ast, collections, hashlib, inspect, json, sys, time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
EXEC = ROOT / 'outputs/stage_i/led_large_exploratory_contrastive_v2_1_execution_v1'
RUN = EXEC / 'downloaded/runs/CF_EXPLORATORY_V2_1_DIAGNOSTIC_ONLY/diagnose_pair'
OUT = ROOT / 'outputs/stage_i/led_large_contrastive_v2_1_root_cause_review_v1'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def preserved_files():
    manifest = read(EXEC / 'portable_review/SHA256_MANIFEST.json')['files']
    paths = {str((EXEC / 'portable_review' / rel).relative_to(ROOT)): digest
             for rel, digest in manifest.items()}
    accepted = read(ROOT/'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1/CONFIG.json')
    for rel, digest in accepted['files'].items():
        assert sha(ROOT/rel) == digest, rel
        paths[rel] = digest
    zip_path=ROOT/'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_GPU_DIAGNOSTIC_REVIEW.zip'
    assert sha(zip_path)=='2fc650edf9720b46f8fd696981c93b690746a7a271cae616180c08ec8070c053'
    extra = [
        ROOT / 'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_1_GPU_DIAGNOSTIC_REVIEW.zip',
        ROOT / 'outputs/cloud_handoff/LED_LARGE_EXPLORATORY_CONTRASTIVE_V2_DIAGNOSTIC_REVIEW.zip',
        EXEC / 'V2_1_RESEARCHER_AUTHORIZATION.json',
        ROOT / 'scripts/run_led_large_exploratory_contrastive_diagnostic_v2_1.py',
        ROOT / 'scripts/led_large_contrastive_diagnostic_evidence_v2_1.py',
        ROOT / 'scripts/led_large_contrastive_sequential_v1.py',
        ROOT / 'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1/CONFIG.json',
        ROOT / 'outputs/stage_i/led_large_exploratory_contrastive_diagnostic_v2_1/PREFLIGHT_MANIFEST.json',
    ]
    extra += list((RUN / 'lambda0').glob('*'))
    checkpoint = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
    proof = read(checkpoint / 'checkpoint_integrity.json')
    for name, digest in proof['files'].items():
        assert sha(checkpoint / name) == digest
        extra.append(checkpoint / name)
    extra.append(checkpoint / 'checkpoint_integrity.json')
    for p in extra:
        paths[str(p.relative_to(ROOT))] = sha(p)
    assert all(sha(ROOT / rel) == digest for rel, digest in paths.items())
    return paths


def raw_gradient_analysis(torch):
    a = torch.load(RUN / 'lambda0/native_gradients.pt', map_location='cpu', weights_only=True)
    b = torch.load(RUN / 'lambda0/delegated_gradients.pt', map_location='cpu', weights_only=True)
    from led_large_contrastive_diagnostic_evidence_v2_1 import compare_tensor
    groups = {}
    tensors = []
    for name in a['parameter_names']:
        x, y = a['gradients'][name], b['gradients'][name]
        assert x.device.type == y.device.type == 'cpu'
        metric = compare_tensor(x, y)
        part = name.split('.led.')[1]
        segment = '.'.join(part.split('.')[:3])
        family = 'A' if '.lora_A.' in name else 'B'
        key = segment + '.' + family
        g = groups.setdefault(key, dict(tensors=0, failing_tensors=0, elements=0,
                                        failing_elements=0, max_abs=0., changed_elements=0))
        g['tensors'] += 1
        g['failing_tensors'] += int(not metric['allclose'])
        g['elements'] += x.numel()
        g['failing_elements'] += metric['failing_element_count']
        g['max_abs'] = max(g['max_abs'], metric['max_absolute_difference'])
        g['changed_elements'] += int((x != y).sum())
        tensors.append(dict(name=name, **metric))
    summary = dict(tensors=len(tensors), failing_tensors=sum(not t['allclose'] for t in tensors),
                   failing_elements=sum(t['failing_element_count'] for t in tensors),
                   elements=sum(x.numel() for x in a['gradients'].values()),
                   bit_exact_tensors=sum(t['bit_exact'] for t in tensors),
                   all_finite=all(t['native_finite'] and t['delegated_finite'] for t in tensors),
                   max_abs=max(t['max_absolute_difference'] for t in tensors),
                   native_loss=float(a['loss']), delegated_loss=float(b['loss']))
    assert summary['failing_tensors'] == 106 and summary['failing_elements'] == 63611
    assert summary['elements'] == 4915200 and summary['max_abs'] == 0.000244140625
    contexts = {key: read(RUN / f'lambda0/{key}_context.json') for key in ('native', 'delegated')}
    summary['recorded_RNG_equal_at_start'] = contexts['native']['RNG_start'] == contexts['delegated']['RNG_start']
    summary['recorded_RNG_equal_at_end'] = contexts['native']['RNG_end'] == contexts['delegated']['RNG_end']
    return dict(summary=summary, per_layer=groups, per_tensor=tensors)


def graph_signature(loss, names):
    """Ordered loss-reachable DAG; never traverses the unused manual CE root."""
    queue = [loss.grad_fn]
    ids = {loss.grad_fn: 0}
    rows = []
    for fn in queue:
        children = []
        for child, output_index in fn.next_functions:
            if child is None:
                children.append([None, output_index])
                continue
            if child not in ids:
                ids[child] = len(queue)
                queue.append(child)
            children.append([ids[child], output_index])
        var = getattr(fn, 'variable', None)
        rows.append([type(fn).__name__, names.get(id(var)), children])
    blob = json.dumps(rows, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(blob).hexdigest(), nodes=len(rows),
                node_types=dict(collections.Counter(row[0] for row in rows)))


def cpu_paths(torch):
    from transformers import LEDConfig, LEDForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    import led_large_contrastive_sequential_v1 as seq
    import torch.nn.functional as F
    data = ROOT / 'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
    pos = read(data / 'positives/EzJI5x25U5GgbYBWhu5ZQQ.json')
    pair = read(data / 'pairs/P02.json')
    assert pair['business_id'] == pos['business_id']
    assert pair['positive_labels'] == pos['corrected_labels']
    positive = CorrectedLabelCollator()([dict(input_ids=pos['input_ids'], labels=pos['corrected_labels'])])
    torch.random.default_generator.manual_seed(17)
    config = LEDConfig(vocab_size=50265, d_model=16, encoder_layers=12, decoder_layers=12,
        encoder_attention_heads=1, decoder_attention_heads=1, encoder_ffn_dim=32, decoder_ffn_dim=32,
        attention_window=[1024]*12, max_encoder_position_embeddings=16384,
        max_decoder_position_embeddings=1024, dropout=.1, attention_dropout=0.,
        activation_dropout=0., encoder_layerdrop=0., decoder_layerdrop=0.,
        pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2, use_cache=False)
    # This CPU fixture never uses quantization. Prevent local optional bnb/triton
    # discovery from probing CUDA while dispatching ordinary torch.nn.Linear.
    import peft.tuners.lora.model as lora_dispatch
    with patch.object(lora_dispatch,'is_bnb_available',return_value=False), \
         patch.object(lora_dispatch,'is_bnb_4bit_available',return_value=False):
        model = get_peft_model(LEDForConditionalGeneration(config), LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0., bias='none', task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules='all-linear'))
    for name, p in model.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(p, std=.01)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    params = tuple(p for p in model.parameters() if p.requires_grad)
    assert len(params) == 456 and all(p.device.type == 'cpu' for p in params)
    names = {id(p): name for name, p in model.named_parameters()}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial = seq.capture_rng(torch.device('cpu'))
    modes = seq._mode_signature(model)
    runs = []
    saved_grads = []
    loss_objects = []
    original_dropout = F.dropout
    for kind in ('native_extra_CE', 'delegated_lambda0', 'native_extra_CE_repeat'):
        seq.restore_rng(initial, torch.device('cpu'))
        calls = []
        layer_calls = collections.Counter()
        handles = []
        def pre(name):
            def hook(*_): layer_calls[name] += 1
            return hook
        for name, module in model.named_modules():
            if name.endswith(tuple(f'layers.{i}' for i in range(12))):
                handles.append(module.register_forward_pre_hook(pre(name)))
        def dropout(input, p=.5, training=True, inplace=False):
            rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            result = original_dropout(input, p, training, inplace)
            calls.append(dict(shape=list(input.shape), p=p, training=training, RNG=rng,
                              result_sha256=hashlib.sha256(result.detach().contiguous().numpy().tobytes()).hexdigest()))
            return result
        start = time.monotonic()
        # CPU FP32 fixture only: we do not emulate CUDA by changing to CPU BF16.
        with patch.object(F, 'dropout', dropout):
            if kind.startswith('native'):
                output = model(**positive, use_cache=False)
                loss = output.loss
                manual = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]),
                                         positive['labels'].reshape(-1), ignore_index=-100)
                manual_equal = torch.equal(loss, manual)
                unused_manual_fn_type = type(manual.grad_fn).__name__
                del manual, output  # Same lifetime boundary as the failed diagnostic.
            else:
                loss = seq.loss(model, positive, None, weight=0.)
                manual_equal, unused_manual_fn_type = None, None
            graph = graph_signature(loss, names)
            forward_call_count = len(calls)
            grads = torch.autograd.grad(loss, params)
        for h in handles: h.remove()
        assert seq._mode_signature(model) == modes
        assert all(p.grad is None for p in params)
        saved_grads.append(grads)  # Preserve previous gradient lifetime like the failed diagnostic.
        loss_objects.append(loss)  # Keep loss object alive after its saved graph tensors are freed.
        runs.append(dict(kind=kind, elapsed_seconds=time.monotonic()-start, loss=float(loss.detach()),
            manual_CE_equal=manual_equal, unused_manual_root=unused_manual_fn_type,
            graph=graph, dropout_calls=calls, forward_dropout_calls=forward_call_count,
            layer_forward_and_recompute_counts=dict(layer_calls), RNG_end=seq.fingerprints(seq.capture_rng(torch.device('cpu'))),
            all_finite=all(torch.isfinite(g).all().item() for g in grads)))
        print('CPU_CASE_COMPLETE', kind, flush=True)
        # Retaining the prior loss object cannot preserve its freed saved tensors after autograd.grad.
    comparisons = []
    for i in (1, 2):
        a, b = saved_grads[0], saved_grads[i]
        comparisons.append(dict(reference=runs[0]['kind'], candidate=runs[i]['kind'],
            tensors=len(a), bit_exact_tensors=sum(torch.equal(x,y) for x,y in zip(a,b)),
            max_abs=max(float((x-y).abs().max()) for x,y in zip(a,b)),
            allclose=all(torch.allclose(y,x,atol=1e-5,rtol=1e-4) for x,y in zip(a,b)),
            graph_equal=runs[0]['graph']==runs[i]['graph'],
            RNG_end_equal=runs[0]['RNG_end']==runs[i]['RNG_end'],
            dropout_execution_trace_equal=runs[0]['dropout_calls']==runs[i]['dropout_calls'],
            checkpoint_execution_counts_equal=runs[0]['layer_forward_and_recompute_counts']==runs[i]['layer_forward_and_recompute_counts']))
    assert all(x['bit_exact_tensors']==456 and x['max_abs']==0 and x['graph_equal'] and
               x['RNG_end_equal'] and x['dropout_execution_trace_equal'] and x['checkpoint_execution_counts_equal']
               for x in comparisons)
    assert all(torch.equal(p.detach(), before[name]) for name,p in model.named_parameters())
    assert all(r['all_finite'] for r in runs)
    for r in runs:
        r['dropout_trace_sha256'] = hashlib.sha256(json.dumps(r['dropout_calls'],sort_keys=True).encode()).hexdigest()
    return dict(fixture='Random reduced-width LED, NOT original pretrained model or CUDA qualification',
        width=16, encoder_layers=12, decoder_layers=12, trainable_tensors=456,
        source_shape=list(positive['input_ids'].shape), labels_shape=list(positive['labels'].shape),
        dropout=.1, checkpoint_use_reentrant=False, device='cpu', dtype='torch.float32',
        CPU_fixture_optional_quantization_discovery_disabled=True,
        model_weights_unchanged=True, successful_optimizer_updates=0, generation_calls=0,
        cases=runs, comparisons=comparisons)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    preservation=preserved_files()
    write(args.output/'PRESERVATION_BEFORE.json',dict(files=preservation))
    import torch, transformers, peft
    torch.set_num_threads(2)
    assert torch.__version__=='2.1.0+cu118' and transformers.__version__=='4.51.3' and peft.__version__=='0.15.2'
    # Guard against accidental CUDA initialization/transfer in this investigation.
    def forbidden(*a,**k): raise AssertionError('CUDA is prohibited in this offline investigation')
    with patch.object(torch.cuda,'_lazy_init',forbidden), patch.object(torch.Tensor,'cuda',forbidden):
        raw=raw_gradient_analysis(torch);write(args.output/'SAVED_GRADIENT_REANALYSIS.json',raw)
        result=cpu_paths(torch);write(args.output/'CPU_PATH_TEST_RESULTS.json',result)
    after={rel:sha(ROOT/rel) for rel in preservation}
    assert after==preservation
    write(args.output/'PRESERVATION_AFTER.json',dict(files=after,unchanged=True))
    import torch.utils.checkpoint as checkpoint
    from transformers.models.led import modeling_led
    files=[Path(inspect.getsourcefile(checkpoint)),Path(inspect.getsourcefile(modeling_led))]
    write(args.output/'LOCAL_RUNTIME_SOURCE_RECEIPT.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        peft=peft.__version__,python=sys.version,files={str(p):sha(p) for p in files}))
    write(args.output/'OFFLINE_INVESTIGATION_RECEIPT.json',dict(
        status='CPU_TESTS_PASS_CUDA_ROOT_CAUSE_NOT_ESTABLISHED',elapsed_seconds=time.monotonic()-start,
        reduced_fixture_forward_backward_pairs=3,new_pretrained_LED_forwards=0,new_CUDA_forwards=0,
        new_CUDA_backwards=0,successful_optimizer_updates=0,generation_calls=0,remote_connections=0,
        preserved_files=len(preservation),scientific_files_modified=False,
        conclusion='UNRESOLVED',recommendation='STOP exploratory Contrastive branch; no GPU retry proposed'))
    print('OFFLINE_INVESTIGATION_COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
