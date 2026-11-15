"""Import and independently check the researcher-downloaded Colab gate evidence; CPU only."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(r'C:\Users\VUONGLOCTRUONG\Downloads\LED_LARGE_GRADIENT_V4_EVIDENCE_20261019T201158Z.zip')
OUT = ROOT/'outputs/stage_i/led_large_gradient_gate_v4_colab_evidence_v1'
EXPECTED_ZIP = '50d6ddbd3dfb035fa81be4cc929551829cb66e47775038814406c3ebf9d12fcc'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    require(sha(ARCHIVE) == EXPECTED_ZIP, 'Delivered archive changed')
    require(not OUT.exists(), 'Preserve existing imported evidence; no overwrite')
    OUT.mkdir(parents=True)
    shutil.copy2(ARCHIVE, OUT/ARCHIVE.name)
    with zipfile.ZipFile(ARCHIVE) as archive:
        require(archive.testzip() is None, 'ZIP CRC failure')
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP names')
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name,
                    'Unsafe ZIP path')
            dest = OUT/'evidence'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    p = OUT/'evidence/run/feasibility'
    report = read(p/'REPORT.json')
    binding = read(ROOT/'outputs/stage_i/led_large_gradient_gate_v4_preparation/EXECUTION_BINDINGS.json')
    require(report['binding'] == binding == read(p/'BINDING.json'), 'Approved hash bindings differ')
    a = read(p/'AUTHORIZATION.json')
    approval_path = ROOT/'outputs/stage_i/led_large_gradient_gate_v4_authorization_v1/AUTHORIZATION.json'
    local_approval = read(approval_path)
    require({k:v for k,v in a.items() if not k.startswith('_')} == local_approval, 'Researcher approval differs')
    # Notebook serialized the recorded approval as UTF-8 LF, independent of Windows newline bytes.
    lf_approval = (json.dumps(local_approval, ensure_ascii=False, indent=2)+'\n').encode()
    require(hashlib.sha256(lf_approval).hexdigest() == report['approval_sha256'] == a['_approval_sha256'],
            'Actual Colab authorization hash differs')
    require(report['deadline_utc'] == a['execution_deadline_utc'] == '2026-10-21T17:11:48+00:00', 'Deadline changed')
    require(report['status'] == 'PASS' and report['optimizer_updates'] == 3 and report['microbatches'] == 24
            and report['generation_calls'] == 2 and report['replay_updates'] == 0, 'Execution scope differs')
    for key in ('state_restore_exact', 'adapter_restore_exact', 'frozen_backbone_unchanged',
                'feasibility_states_discarded_before_fresh_F_train'):
        require(report[key] is True, key)
    require(report['D051_strict_replay'] == 'FAIL_UNCHANGED', 'D051 history differs')
    old = ROOT/'outputs/stage_i/led_large_gradient_gate_v3_colab_evidence_v1/evidence/run/feasibility'
    init = read(p/'INITIAL_MODEL_BINDINGS.json')
    require(init == read(old/'INITIAL_MODEL_BINDINGS.json'), 'v3/v4 initial model bindings differ')
    require(init['scope'] == dict(modules=228, trainable_tensors=456, trainable_parameters=4915200,
                                 frozen_backbone_parameters=459801600), 'Adapter scope differs')
    require(len(init['initialization']) == 228, 'Missing initialization hashes')
    micro = [json.loads(s) for s in (p/'MICROBATCH_EVIDENCE.jsonl').read_text().splitlines()]
    grads = [json.loads(s) for s in (p/'SCALED_GRADIENT_EVIDENCE.jsonl').read_text().splitlines()]
    require(len(micro) == len(grads) == 24, 'Missing per-microbatch evidence')
    for i, (m, g) in enumerate(zip(micro, grads)):
        require((m['step'],m['microbatch']) == (g['step'],g['microbatch']) == (i//8+1,i%8+1), 'Order differs')
        require(math.isfinite(m['CE_loss']) and m['scaler_scale'] == g['scaler_scale'] == 1024.0, 'Loss/scale failure')
        require(g['trainable_tensors'] == len(g['tensors']) == 456 and not g['missing'] and not g['nonfinite'],
                'Missing/nonfinite scaled gradients')
        require(sum(t['elements'] for t in g['tensors'].values()) == 4915200, 'Gradient element count differs')
        require(all(t['status'] == 'finite' and t['finite'] == t['elements'] for t in g['tensors'].values()),
                'Invalid scaled gradient counts')
    old_micro = [json.loads(s) for s in (old/'MICROBATCH_EVIDENCE.jsonl').read_text().splitlines()]
    require(len(old_micro) == 8 and [m['CE_loss'] for m in micro[:8]] == [m['CE_loss'] for m in old_micro],
            'Controlled first-eight forward losses differ')
    updates = []
    for step in range(1, 4):
        u = read(p/f'update_{step}.json')
        g = read(p/f'UNSCALED_GRADIENTS_step{step}.json')
        require(u['step'] == g['step'] == step and g['completed_optimizer_updates'] == step-1, 'Step mismatch')
        require(g['trainable_tensors'] == len(g['tensors']) == len(u['gradients']) == 456
                and not g['missing'] and not g['nonfinite'], 'Unscaled gradients incomplete')
        require(all(t['finite'] for t in u['gradients'].values()), 'Nonfinite update gradient')
        require(u['losses'] == [m['CE_loss'] for m in micro[(step-1)*8:step*8]], 'Loss records differ')
        require(math.isclose(u['used_lr'], .0002*(step-1)/63, abs_tol=1e-16), 'Warmup changed')
        if step == 3:
            require(all(t['nonzero'] for t in u['gradients'].values()), 'Missing later A/B gradient flow')
        updates.append(dict(step=step, mean_CE_loss=sum(u['losses'])/8, used_lr=u['used_lr'],
                            finite_tensors=456, nonzero_tensors=sum(t['nonzero'] for t in u['gradients'].values())))
    samples = read(p/'MEMORY_SAMPLES.json')
    total = samples[-1]['total']
    used = max(max(s['used'] for s in samples), report['peak_reserved_bytes']+
               max(max(0,s['used']-s['reserved']) for s in samples))
    require(total == report['total_bytes'] == report['GPU_total_bytes'] and used == report['conservative_peak_used_bytes'],
            'Memory reconstruction differs')
    require(1-used/total == report['headroom_fraction'] and report['headroom_fraction'] >= .15, 'Memory gate fails')
    cfg = read(ROOT/'configs/stage_i_led_large_gradient_gate_v4.json')
    effective = read(p/'EFFECTIVE_GENERATION_CONFIG.json')
    require(all(effective[k] == v for k,v in cfg['generation'].items()), 'Frozen generation settings differ')
    traces = [read(p/f'train_memory_trace_{i}.json') for i in (1,2)]
    require(all(t['scope'] == 'TRAIN_INPUT_TECHNICAL_MEMORY_ONLY_NOT_QUALITY_OUTPUT' and
                0 < len(t['token_ids']) <= 768 for t in traces), 'Technical trace scope differs')
    cp = p/'checkpoint_step3'
    receipt = read(cp/'integrity.json')
    require(receipt['binding'] == binding, 'Checkpoint binding differs')
    present, missing = {}, []
    for name,digest in receipt['files'].items():
        require(Path(name).name == name, 'Unsafe checkpoint name')
        if (cp/name).is_file():
            require(sha(cp/name) == digest, 'Delivered checkpoint component hash differs')
            present[name] = digest
        else:
            missing.append(name)
    require(set(missing) == {'README.md','adapter_model.safetensors'}, 'Unexpected missing checkpoint files')
    summary = dict(archive_sha256=EXPECTED_ZIP, archive_bytes=ARCHIVE.stat().st_size, members=len(names),
                   actual_bounded_gate_status='PASS', independent_log_checks='PASS',
                   portable_checkpoint_completeness='INCOMPLETE_EXPORT_ONLY', missing_checkpoint_files=missing,
                   checkpoint_present_hashes=present, original_report=report, updates=updates,
                   first_eight_forward_losses_identical_to_v3=True, all_24_scaled_gradients_finite=True,
                   all_456_later_A_B_gradients_nonzero=True, source='Researcher-run Colab; Codex offline evidence audit',
                   no_new_GPU_execution=True, main_F_training_started=False)
    write(OUT/'AUDIT_SUMMARY.json', summary)
    write(OUT/'SHA256_MANIFEST.json', {p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
    print(json.dumps(dict(status='PASS_WITH_EXPORT_COMPLETENESS_NOTE', destination=str(OUT),
                         updates=updates, headroom=report['headroom_fraction'], missing=missing), indent=2))


if __name__ == '__main__':
    main()
