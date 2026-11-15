"""Minimal technical-sanity revision; accepted V1 label/collator implementation reused."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from led_large_decoder_label_view_v1 import require, remove_initial_bos, sha

CRITERIA = dict(
    policy='TECHNICAL_BOS_COLLAPSE_V2',
    consecutive_BOS_hard_fail_at=5,
    near_empty_ordinary_tokens_less_than=16,
    special_dominance_fraction_at_least=0.90,
    isolated_BOS='WARNING_ONLY',
    generation_cap='WARNING_ONLY_UNLESS_COLLAPSE',
    empty_decoded_text='HARD_FAIL',
    wrong_decoder_start_or_invalid_token='HARD_FAIL',
    semantic_quality_GO=False,
)
SPECIAL_IDS = frozenset((0, 1, 2, 3, 50264))


def sanity_result(ids, text, max_length=768):
    """Raw-token technical screen; does not assess factual/semantic summary quality."""
    errors, warnings = [], []
    if not ids or ids[0] != 2:
        errors.append('Wrong decoder start')
    if any(type(t) is not int or not 0 <= t < 50265 for t in ids):
        errors.append('Invalid generated token ID')
    body = ids[1:]
    longest = run = 0
    for token in body:
        run = run + 1 if token == 0 else 0
        longest = max(longest, run)
    bos = body.count(0)
    ordinary = sum(t not in SPECIAL_IDS for t in body)
    special = len(body) - ordinary
    fraction = special / len(body) if body else 1.0
    if longest >= CRITERIA['consecutive_BOS_hard_fail_at']:
        errors.append('Pathological consecutive BOS loop (>=5)')
    elif bos:
        warnings.append('BOS after decoder start without sustained BOS loop')
    if not text.strip():
        errors.append('Empty decoded output')
    if (ordinary < CRITERIA['near_empty_ordinary_tokens_less_than'] and
            fraction >= CRITERIA['special_dominance_fraction_at_least']):
        errors.append('Near-empty special-token-dominated output')
    cap = len(ids) >= max_length
    if cap:
        warnings.append('Generation length cap reached')
    if not body or body[-1] != 2:
        warnings.append('No terminal EOS observed')
    return dict(status='FAIL_STOP' if errors else 'PASS_TECHNICAL_ONLY', errors=errors,
                warnings=warnings, criteria=CRITERIA.copy(), semantic_quality_GO=False,
                token_count=len(ids), BOS_after_start=bos, longest_BOS_run=longest,
                ordinary_content_tokens=ordinary, special_tokens_after_start=special,
                special_fraction_after_start=fraction, cap_hit=cap,
                terminal_EOS=bool(body) and body[-1] == 2)


def continuity_snapshot(trainer):
    """Read state only. No optimizer/scaler restore, step, zero_grad or schedule action."""
    from run_stage_i_d050_feasibility_v1 import fingerprint
    model = trainer.model
    scaler = trainer.accelerator.scaler
    require(trainer.optimizer is not None and trainer.lr_scheduler is not None and scaler is not None,
            'Epoch1 optimizer/scheduler/scaler required for continuity verification')
    adapter_flags = {}
    for name, module in model.named_modules():
        flags = {}
        for key in ('active_adapters', 'disable_adapters', 'merged_adapters'):
            if hasattr(module, key):
                flags[key] = copy.deepcopy(getattr(module, key))
        if flags:
            adapter_flags[name] = flags
    return dict(
        optimizer_object=id(trainer.optimizer), scheduler_object=id(trainer.lr_scheduler),
        scaler_object=id(scaler), model_object=id(model),
        optimizer=fingerprint(trainer.optimizer.state_dict()),
        scheduler=fingerprint(trainer.lr_scheduler.state_dict()),
        scaler=fingerprint(scaler.state_dict()),
        AMP_recovery=fingerprint(trainer.amp_recovery.state(scaler)),
        trainer_state=fingerprint(trainer.state.__dict__),
        trainer_control=fingerprint(trainer.control.__dict__),
        accelerator_step_skipped=trainer.accelerator.optimizer_step_was_skipped,
        adapter_activation=fingerprint(adapter_flags),
        parameter_objects={n: id(p) for n, p in model.named_parameters()},
        requires_grad={n: p.requires_grad for n, p in model.named_parameters()},
        gradients=fingerprint({n: p.grad for n, p in model.named_parameters() if p.requires_grad}),
    )


@contextmanager
def preserve_training_state(model, trainer, evidence):
    """Restore RNG/modes/cache, verify adapter and optimizer continuity even on exceptions."""
    import torch
    from peft import get_peft_model_state_dict
    from led_large_dynamic_amp_v2 import capture_rng, restore_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    require(trainer.model is model, 'Sanity model differs from live Trainer model')
    rng = capture_rng()
    modes = {name: m.training for name, m in model.named_modules()}
    cache = model.config.use_cache
    adapters = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    before = continuity_snapshot(trainer)
    evidence['before'] = before
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        for name, module in model.named_modules():
            if name not in modes:
                failures.append('Module tree changed')
            else:
                module.training = modes[name]
        model.config.use_cache = cache
        try:
            restore_rng(rng)
            checks = dict(
                RNG_restored=fingerprint(capture_rng()) == fingerprint(rng),
                module_modes_restored={n: m.training for n, m in model.named_modules()} == modes,
                use_cache_restored=model.config.use_cache == cache,
            )
            after_adapters = get_peft_model_state_dict(model)
            checks['adapter_unchanged'] = set(after_adapters) == set(adapters) and all(
                torch.equal(v.detach().cpu(), adapters[k]) for k, v in after_adapters.items())
            after = continuity_snapshot(trainer)
            evidence['after'] = after
            checks['optimizer_and_training_continuity_exact'] = before == after
            evidence['checks'] = checks
            failures.extend(k for k, ok in checks.items() if not ok)
        except BaseException as check_error:
            failures.append(type(check_error).__name__ + ': ' + str(check_error))
        evidence['status'] = 'PASS' if not failures else 'FAIL_STOP'
        evidence['errors'] = failures
        if failures:
            message = 'Early sanity state continuity failed: ' + '; '.join(failures)
            if primary is not None:
                primary.add_note(message)  # Keep original inference traceback; record cleanup errors too.
            else:
                raise ValueError(message)


def run_early_sanity(model, folder, approval, config, memory, root, binding, write, check_deadline, trainer):
    import torch
    from transformers import LEDTokenizerFast, GenerationConfig
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    require(approval.get('early_sanity_generation_approved') is True, 'Early generation not approved')
    require(approval.get('early_sanity_outputs_max') == 4, 'Wrong early sanity budget')
    require(config['early_sanity']['technical_criteria'] == CRITERIA, 'Technical sanity criteria differ')
    bids = config['early_sanity']['business_ids']
    require(len(bids) == len(set(bids)) == 4, 'Four unique predeclared cases required')
    path = Path(folder) / 'early_sanity'
    path.mkdir(exist_ok=False)
    checkpoint = Path(folder) / 'checkpoint-250'
    require((checkpoint / 'checkpoint_integrity.json').is_file(), 'Epoch1 checkpoint receipt required')
    cases = {c['business_id']: c for c in (
        json.loads(x) for x in (Path(root) / config['early_sanity']['cases_file']).read_text(encoding='utf-8').splitlines())}
    tok = LEDTokenizerFast.from_pretrained(Path(root) / config['source_tokenizer'], local_files_only=True)
    g = GenerationConfig(**json.loads((Path(root) / config['effective_generation_config_file']).read_text(encoding='utf-8')))
    pack = CorrectedLabelCollator()
    rows, continuity = [], {}
    try:
        with preserve_training_state(model, trainer, continuity):
            model.eval()
            for bid in bids:
                check_deadline(approval)
                c = cases[bid]
                batch = pack([dict(input_ids=c['input_ids'], labels=remove_initial_bos(c['label_ids']))])
                batch = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    ids = model.generate(**batch, generation_config=g)[0].cpu().tolist()
                text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                record = dict(business_id=bid, subset=c['subset'], epoch=1, step=250, token_ids=ids, text=text,
                              source_sha256=c['source_sha256'], checkpoint_sha256=sha(checkpoint / 'adapter_model.safetensors'),
                              technical_screen=sanity_result(ids, text, g.max_length), researcher_quality_status='NOT_REVIEWED')
                rows.append(record)
                write(path / (bid + '.json'), record)
                check_deadline(approval)
                memory.check()
    finally:
        # Available even if inference/restoration fails; no success is inferred from partial rows.
        write(path / 'STATE_CONTINUITY.json', continuity)
    report = dict(status='PASS_TECHNICAL_ONLY' if len(rows) == 4 and all(
        r['technical_screen']['status'] == 'PASS_TECHNICAL_ONLY' for r in rows) else 'FAIL_STOP',
        binding=binding, generation_calls=len(rows), epoch=1, step=250, rows=rows,
        state_continuity=continuity, quality_GO=False, criteria=CRITERIA.copy())
    write(path / 'REPORT.json', report)
    require(report['status'] == 'PASS_TECHNICAL_ONLY', 'Early generation collapsed; STOP without retry or setting changes')
    return report
