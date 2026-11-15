"""Condition-aware RO/U transport for the accepted, unchanged DecoderPolicyV2 core.

Preparation/preflight never constructs a model or calls CUDA. Execution requires a
new condition-specific, hash-bound researcher receipt. No evaluate/resume mode.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = Path('outputs/stage_i/led_large_matched_family_v1')
CONDITIONS = ('reviews_only', 'unfiltered')
PARENT_CONFIG_SHA = 'd079f41aa5ae5b1a680da5174eb3ef83235cb9b82e54c96b3d5d487215a168f4'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')


def paths(condition, root=ROOT):
    require(condition in CONDITIONS, 'Only RO/U training packages are prepared')
    return root / 'configs' / f'led_large_matched_{condition}_v1.json'


def expected_config(parent, condition):
    """Exhaustive allowlist: all other F science fields remain byte-equivalent JSON values."""
    require(condition in CONDITIONS, 'Wrong condition')
    c = json.loads(json.dumps(parent))
    name = f'LED_LARGE_MATCHED_{condition.upper()}_V1'
    output = f'runs/{condition}_large_matched_v1'
    c.update(experiment=name, condition=condition, run_root=output,
             label_views_manifest=str(PREP / 'LABEL_VIEWS.json').replace('\\', '/'),
             checkpoint_selection='LOCKED_COMMON_EPOCH3_STEP750_F_DEVELOPMENT_INFORMED',
             training_runner_status='OFFLINE_ONLY_SEPARATE_APPROVAL_REQUIRED',
             preparation_state='OFFLINE_ONLY_NO_GPU_AUTHORIZATION')
    c['vast_fresh'].update(experiment=name, output=output)
    c['early_sanity']['cases_file'] = str(PREP / f'{condition}_validation_cases.jsonl').replace('\\', '/')
    # Keep the original checkpoint collection, per-epoch eval, generation, and all
    # numerical settings. Historical F timeline is provenance only, never a clock.
    return c


def binding(condition, root=ROOT):
    return dict(config_sha256=sha(paths(condition, root)),
                runner_sha256=sha(root / 'scripts/run_led_large_matched_family_v1.py'),
                helper_sha256=sha(root / 'scripts/led_large_matched_family_v1.py'),
                preflight_manifest_sha256=sha(root / PREP / 'PREFLIGHT_MANIFEST.json'))


def verify_overlay(root=ROOT):
    manifest = read(root / PREP / 'PREFLIGHT_MANIFEST.json')
    for rel, digest in manifest['files'].items():
        p = root / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe manifest path')
        require(p.is_file() and not p.is_symlink() and sha(p) == digest, 'Overlay hash differs: ' + rel)
    return len(manifest['files'])


def condition_data(c, parent, root=ROOT):
    from stage_hi_led_data import PackedLEDDataset
    from led_large_decoder_label_view_v1 import CorrectedLabelDataset
    condition = c['condition']
    require(condition in CONDITIONS, 'Refuse F data masquerading as RO/U')
    manifest = read(parent / c['stage_h_manifest'])
    views = read(root / c['label_views_manifest'])
    require(views['policy'] == c['decoder_label_policy'], 'Label policy mismatch')
    result = []
    for split, count in [('train', 2000), ('validation', 200)]:
        key = split + '_' + condition
        original = PackedLEDDataset(manifest['datasets'][key], 5000, 768, root=root)
        ds = CorrectedLabelDataset(original, views['views'][key], root)
        require(len(ds) == count, 'Dataset count mismatch')
        result.append(ds)
    train, val = result
    require(not set(train.business_ids) & set(val.business_ids), 'Split leakage')
    orders = read(parent / 'outputs/stage_i/led_large_preparation_v1/EPOCH_ORDERS.json')
    require(train.business_ids == orders['training_business_ids'], 'Frozen training order mismatch')
    return train, val


def cpu_preflight(condition, parent, root=ROOT, full_parent=True):
    parent = Path(parent).resolve()
    require(parent != root.resolve(), 'Supply the complete accepted V2 parent, not a partial overlay')
    verify_overlay(root)
    sys.path.insert(0, str(parent / 'scripts'))
    import stage_i_led_large_decoder_policy_v2 as h
    require(h.ROOT.resolve() == parent, 'Imported a different parent workspace; start a fresh process')
    require(sha(parent / 'configs/stage_i_led_large_decoder_policy_v2.json') == PARENT_CONFIG_SHA,
            'Wrong successful F configuration')
    old = h.preflight() if full_parent else read(h.CONFIG)
    # full_parent=False is for routing unit tests only; CLI always uses True.
    c = read(paths(condition, root))
    require(c == expected_config(old, condition), 'Unapproved condition/scientific difference')
    for rel, digest in read(root / PREP / 'PARENT_REQUIRED_FILES.json')['files'].items():
        require(sha(parent / rel) == digest, 'Required immutable parent member changed: ' + rel)
    for rel, digest in read(root / PREP / 'ORIGINAL_SOURCE_FILES.json')['files'].items():
        require(sha(root / rel) == digest, 'RO/U original source binding differs: ' + rel)
    train, val = condition_data(c, parent, root)
    cases = [json.loads(s) for s in (root / c['early_sanity']['cases_file']).read_text(encoding='utf-8').splitlines()]
    require(len(cases) == 24 and len({r['business_id'] for r in cases}) == 24, 'Wrong development cases')
    for r in cases:
        i = val.business_ids.index(r['business_id'])
        require(r['condition'] == condition and r['input_ids'] == val[i]['input_ids']
                and r['label_ids'][1:] == val[i]['labels'], 'Cross-condition sanity source/labels')
    gate = read(root / PREP / f'{condition}_FEASIBILITY_CASES.json')
    require(len(gate['business_ids']) >= 2 and all(x in train.business_ids for x in gate['business_ids']),
            'Gate must be TRAIN-only')
    return c, dict(status='CPU_PREFLIGHT_PASS', condition=condition, parent=str(parent),
                   rows=[len(train), len(val)], binding=binding(condition, root), GPU_operations=0,
                   generation_calls=0, optimizer_updates=0)


def check_deadline(a, now=None):
    times = [datetime.fromisoformat(a[k]) for k in
             ('execution_authorized_at_utc', 'execution_deadline_utc', 'researcher_absolute_stop_utc')]
    require(all(t.tzinfo is not None and t.utcoffset().total_seconds() == 0 for t in times), 'UTC clocks required')
    start, end, absolute = times
    require(start <= (now or datetime.now(timezone.utc)) < end <= min(start + timedelta(hours=48), absolute),
            'Expired/invalid fresh authorization')
    require(a.get('defense_reserve_at_least_14_days_confirmed') is True, '14-day reserve confirmation required')
    require(a.get('max_training_updates') == 1250 and a.get('feasibility_updates_max') == 3
            and a.get('all_generation_calls_max') == 6, 'Budget drift')
    return end.isoformat()


def authorize(path, condition, mode, root=ROOT):
    a = read(path)
    require(a.get('GPU_execution_approved') is True and a.get('training_authorized') is True,
            'OFFLINE package: fresh explicit GPU approval required')
    require(a.get('experiment') == read(paths(condition, root))['experiment'] and a.get('condition') == condition,
            'Wrong experiment/condition authorization')
    require(a.get('approval_source') == 'researcher_message' and a.get('researcher_explicit_authorization'),
            'Explicit researcher source required')
    require(a.get('scopes') == ['feasibility', 'train'] and mode in a['scopes'], 'Training-only contract')
    require(a.get('binding') == binding(condition, root), 'Authorization hash mismatch')
    expected = dict(fresh_only=True, held_out_access=False, Teacher_calls=False, resume_authorized=False,
                    conditional_train_after_feasibility_pass=True, early_sanity_generation_approved=True,
                    early_sanity_outputs_max=4, technical_generation_calls_max=2, evaluation_outputs_max=0,
                    feasibility_microbatches_max=88, selected_checkpoint_step=750)
    require(all(a.get(k) == v for k, v in expected.items()), 'Forbidden scope/budget')
    check_deadline(a)
    return dict(a, _approval_sha256=sha(path))


def install_condition_context(condition, parent, root=ROOT):
    """Reuse bound native CE/AMP/gate code; replace only data, case and receipt routing.

    Call only after original parent preflight, in an isolated worker. Never call
    the historical F CLI or evaluate(). A fresh process is used for gate/train.
    """
    import stage_i_led_large_decoder_policy_v2 as h
    import run_stage_i_led_large_decoder_policy_v2 as b
    require(h.ROOT.resolve() == Path(parent).resolve() == b.ROOT.resolve(), 'Parent import mismatch')
    c = read(paths(condition, root))
    get_binding = lambda: binding(condition, root)
    b.binding = h.binding = get_binding
    b.check_deadline = h.check_deadline = check_deadline
    b.data = lambda cfg: condition_data(cfg, Path(parent), root)
    original_read = b.read

    def routed_read(path):
        if Path(path) == b.LARGE_PREP / 'FEASIBILITY_CASES.json':
            return read(root / PREP / f'{condition}_FEASIBILITY_CASES.json')
        return original_read(path)

    b.read = routed_read
    Base = h._core_callback_class()

    class Record(Base):
        def on_save(self, args, state, control, **kwargs):
            super().on_save(args, state, control, **kwargs)
            if state.global_step == 250:
                from led_large_decoder_early_sanity_v2 import run_early_sanity
                operational = dict(c, source_tokenizer=str(Path(parent) / c['source_tokenizer']),
                                   effective_generation_config_file=str(Path(parent) / c['effective_generation_config_file']))
                run_early_sanity(self.trainer.model, self.folder, self.approval, operational,
                                 self.memory, root, get_binding(), h.write, check_deadline, trainer=self.trainer)
            elif state.global_step > 250:
                proof = read(self.folder / 'early_sanity/REPORT.json')
                require(proof['status'] == 'PASS_TECHNICAL_ONLY' and proof['binding'] == get_binding()
                        and proof['generation_calls'] == 4, 'Four-output technical sanity PASS required')

    b.callback_class = lambda: Record
    return b, c
