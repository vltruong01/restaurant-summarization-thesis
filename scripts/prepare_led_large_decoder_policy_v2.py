"""Offline minimal early-sanity revision. Never rewrite accepted V1 artifacts."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2'
BASE = ROOT / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V1_WORKSPACE'
STAGE = ROOT / 'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
V1 = 'outputs/stage_i/led_large_decoder_policy_v1/'
V2 = 'outputs/stage_i/led_large_decoder_policy_v2/'
sys.path.insert(0, str(ROOT / 'scripts'))
from led_large_decoder_early_sanity_v2 import CRITERIA


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def put(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def derivative(rel):
    dest = rel.replace('_v1', '_v2').replace('_V1', '_V2')
    source = (ROOT / rel).read_text(encoding='utf-8')
    # Only execution namespace changes. Accepted label policy/module/views stay V1.
    revised = source.replace('decoder_policy_v1', 'decoder_policy_v2').replace('DECODER_POLICY_V1', 'DECODER_POLICY_V2')
    revised = revised.replace('outputs/stage_i/led_large_decoder_policy_v2/LABEL_VIEWS.json', V1 + 'LABEL_VIEWS.json')
    if rel == 'scripts/stage_i_led_large_decoder_policy_v1.py':
        revised = revised.replace('import stage_i_led_large_vast_d059_v1 as parent', 'import stage_i_led_large_decoder_policy_v1 as parent')
        node = next(n for n in ast.parse(revised).body if isinstance(n, ast.FunctionDef) and n.name == 'preflight')
        lines = revised.splitlines(keepends=True)
        replacement = '''def preflight():
    import stage_i_led_large_decoder_policy_v1 as parent
    from led_large_decoder_early_sanity_v2 import CRITERIA
    old=parent.preflight()  # Accepted V1 + complete original parent/runtime bindings.
    c=read(CONFIG)
    early=dict(old['early_sanity'])
    del early['requires_nonempty_no_BOS_natural_EOS_min16_content']
    early['technical_criteria']=CRITERIA
    expected=dict(old,experiment=EXPERIMENT,run_root='runs/F_large_decoder_policy_v2',early_sanity=early,
        vast_fresh=dict(old['vast_fresh'],experiment=EXPERIMENT,output='runs/F_large_decoder_policy_v2'))
    require(c==expected,'Unrequested scientific/config change in minimal V2 revision')
    require(c['early_sanity']==read(PREP/'EARLY_SANITY_MANIFEST.json'),'Early case/criteria drift')
    for rel,digest in read(PREP/'PREFLIGHT_MANIFEST.json')['files'].items():
        require(sha(ROOT/rel)==digest,'V2 binding differs: '+rel)
    return c
'''
        revised = ''.join(lines[:node.lineno-1]) + replacement + ''.join(lines[node.end_lineno:])
        revised = revised.replace('from led_large_decoder_early_sanity_v1 import', 'from led_large_decoder_early_sanity_v2 import')
        revised = revised.replace('ROOT,binding(),write,check_deadline)', 'ROOT,binding(),write,check_deadline,trainer=self.trainer)')
    if rel == 'RUN_DECODER_POLICY_V1.sh':
        revised = revised.replace('decoder_policy_launcher_logs', 'decoder_policy_v2_launcher_logs')
    put(ROOT / dest, revised)
    return dest, ''.join(difflib.unified_diff(source.splitlines(True), revised.splitlines(True), fromfile=rel, tofile=dest))


def main():
    PREP.mkdir(parents=True, exist_ok=True)
    accepted = read(ROOT / (V1 + 'EXECUTION_BINDINGS.json'))
    expected = dict(config_sha256='89202ab27ec6adecba3ee293cb1c8151b3a5935a20ff0c31da12c9bc784a3d20',
        runner_sha256='6762bed2bf4f44ecb2077d87f5687e3ab8193bce421b291dceaaf48c80298f8f',
        preflight_manifest_sha256='cfa5b638cc7cae4fa5013f4bf220e4a79318d287d80e24d06983f6b4d944946c',
        helper_sha256='390ee6dbcb7a5bd85ca4ff11def1cb1e74e7b87e34cb75e2db33b943b57aad28',
        label_module_sha256='413a408ed384bdd5d40ef21ea7c99891a2ee42e88554a1f6e543d1ad6f535b96')
    assert accepted == expected
    inherited = read(ROOT / (V1 + 'PREFLIGHT_MANIFEST.json'))['files']
    for rel, digest in inherited.items():
        assert sha(BASE / rel) == digest, rel
    assert sha(ROOT / 'scripts/led_large_decoder_label_view_v1.py') == accepted['label_module_sha256']
    files, diffs = [], []
    for rel in ('scripts/stage_i_led_large_decoder_policy_v1.py', 'scripts/run_stage_i_led_large_decoder_policy_v1.py',
                'scripts/report_stage_i_led_large_decoder_policy_v1.py', 'RUN_DECODER_POLICY_V1.sh'):
        dest, diff = derivative(rel)
        files.append(dest)
        diffs.append(diff)
    old = read(ROOT / 'configs/stage_i_led_large_decoder_policy_v1.json')
    c = dict(old, experiment='LED_LARGE_DECODER_POLICY_V2', run_root='runs/F_large_decoder_policy_v2',
             vast_fresh=dict(old['vast_fresh'], experiment='LED_LARGE_DECODER_POLICY_V2', output='runs/F_large_decoder_policy_v2'))
    c['early_sanity'] = dict(old['early_sanity'])
    del c['early_sanity']['requires_nonempty_no_BOS_natural_EOS_min16_content']
    c['early_sanity']['technical_criteria'] = CRITERIA
    config = 'configs/stage_i_led_large_decoder_policy_v2.json'
    put(ROOT / config, c)
    diffs.append(''.join(difflib.unified_diff((ROOT / 'configs/stage_i_led_large_decoder_policy_v1.json').read_text(encoding='utf-8').splitlines(True),
        (ROOT / config).read_text(encoding='utf-8').splitlines(True), fromfile='configs/stage_i_led_large_decoder_policy_v1.json', tofile=config)))
    for rel in ('scripts/led_large_decoder_early_sanity_v2.py', 'tests/test_led_large_decoder_policy_v2.py'):
        source = (ROOT / rel.replace('_v2', '_v1')).read_text(encoding='utf-8')
        diffs.append(''.join(difflib.unified_diff(source.splitlines(True), (ROOT / rel).read_text(encoding='utf-8').splitlines(True),
                                              fromfile=rel.replace('_v2', '_v1'), tofile=rel)))
    put(PREP / 'EARLY_SANITY_MANIFEST.json', c['early_sanity'])
    put(PREP / 'EXACT_PROPOSED_PATCH.diff', ''.join(diffs))
    put(PREP / 'ACCEPTED_V1_BINDINGS.json', accepted)
    files += [config, 'scripts/led_large_decoder_early_sanity_v2.py', 'tests/test_led_large_decoder_policy_v2.py',
              'scripts/prepare_led_large_decoder_policy_v2.py']
    files += [V2 + n for n in ('EARLY_SANITY_MANIFEST.json', 'EXACT_PROPOSED_PATCH.diff', 'ACCEPTED_V1_BINDINGS.json')]
    files += [V1 + n for n in ('PREFLIGHT_MANIFEST.json', 'EXECUTION_BINDINGS.json')]
    manifest = {rel: sha(ROOT / rel) for rel in files}
    put(PREP / 'PREFLIGHT_MANIFEST.json', dict(files=manifest,
        parent_preflight_manifest_sha256=accepted['preflight_manifest_sha256'],
        accepted_label_module_sha256=accepted['label_module_sha256'], parent_payload_files=9961,
        revision_scope='Technical early sanity and exact state-continuity verification only'))
    bindings = dict(config_sha256=sha(ROOT / config), runner_sha256=sha(ROOT / 'scripts/run_stage_i_led_large_decoder_policy_v2.py'),
        preflight_manifest_sha256=sha(PREP / 'PREFLIGHT_MANIFEST.json'), helper_sha256=sha(ROOT / 'scripts/stage_i_led_large_decoder_policy_v2.py'),
        early_sanity_module_sha256=sha(ROOT / 'scripts/led_large_decoder_early_sanity_v2.py'),
        accepted_label_module_sha256=accepted['label_module_sha256'])
    put(PREP / 'EXECUTION_BINDINGS.json', bindings)
    a = read(ROOT / (V1 + 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json'))
    a.update(experiment='LED_LARGE_DECODER_POLICY_V2', **{k: bindings[k] for k in (
        'config_sha256', 'runner_sha256', 'preflight_manifest_sha256')})
    assert not a['GPU_execution_approved'] and a['execution_authorized_at_utc'] is None
    put(PREP / 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json', a)
    # Inherit immutable V1 workspace using hardlinks; write only NEW V2 paths privately.
    inherited = dict(read(BASE / 'VAST_SHA256_MANIFEST.json')['files'], **inherited)
    inherited.update({V1 + n: sha(ROOT / (V1 + n)) for n in (
        'PREFLIGHT_MANIFEST.json', 'EXECUTION_BINDINGS.json', 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')})
    for rel, digest in inherited.items():
        target = STAGE / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(BASE / rel, target)
    new_files = [r for r in files if not r.startswith(V1)] + [V2 + n for n in (
        'PREFLIGHT_MANIFEST.json', 'EXECUTION_BINDINGS.json', 'AUTHORIZATION_TEMPLATE_NOT_APPROVED.json')]
    for rel in new_files:
        assert rel not in inherited, 'Cannot overwrite accepted/historical file'
        target = STAGE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.resolve().is_relative_to(STAGE.resolve())
            target.unlink()  # Only private, unapproved V2 candidate paths.
        shutil.copyfile(ROOT / rel, target)
    print(json.dumps(bindings, indent=2), flush=True)


if __name__ == '__main__':
    main()
