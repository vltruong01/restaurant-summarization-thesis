"""Freeze the minimal technical repair before its single authorized execution.

Offline only: imports stdlib and existing no-model integrity helpers.
"""
import difflib
import shutil
from pathlib import Path
from stage_i_repetition_pilot_v1 import ROOT, read, sha, write


def prepare():
    prep=ROOT/'outputs/stage_i/repetition_guard_preparation_v2'
    prep.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/'configs/stage_i_repetition_guard_pilot_v1.json')
    config={**old,'experiment':'repetition_guard_pilot_v2',
        'status':'OFFLINE_TECHNICAL_REPAIR_PREPARED_APPROVAL_IS_SEPARATE_RECORD',
        'run_root':'outputs/stage_i/repetition_guard_pilot_v2',
        'new_generation_calls_max':19,'reuse_count':5,'terminal_repair_prefix_tokens':767,
        'repair_scope':'Only forced EOS2 at prefix767: prune forbidden rows; observe actual finished hypotheses without changing HF selection. All nonterminal behavior and hard rules unchanged.',
        'known_terminal_infeasibility':'Save missing-output record and continue other businesses once; no EOS bypass, generated-content repair, or retry.',
        'lifetime_call_accounting':'v1 attempted6; v2 maximum19 =25 lifetime calls, one explicitly authorized retry of failed case6;24 unique cases;5 reused byte-for-byte'}
    write(ROOT/'configs/stage_i_repetition_guard_pilot_v2.json',config)
    for rel in ('DECISIONS.md','thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md'):
        dest=prep/'documentation_before_repair'/Path(rel).name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    diff=''.join(difflib.unified_diff(
        (ROOT/'scripts/run_stage_i_repetition_guard_pilot_v1.py').read_text(encoding='utf-8').splitlines(True),
        (ROOT/'scripts/run_stage_i_repetition_guard_pilot_v2.py').read_text(encoding='utf-8').splitlines(True),
        fromfile='frozen_v1_runner',tofile='new_v2_runner'))
    write(prep/'RUNNER_EXACT_DIFF.patch',diff)
    write(prep/'TECHNICAL_REPAIR_AND_SCOPE.md',"""# Terminal beam repair and one authorized continuation

Researcher instruction: “trước hết hãy sửa hết và chạy tiếp”. This explicitly
authorizes technical repair and continuation. The hashes are generated under
that delegated instruction; no claim of an independent advance hash review is made.
First observed workspace clock after the instruction:2026-10-19T17:41:13.021054+07:00
(2026-10-19T10:41:13.021054+00:00). This is observation time, not a hidden message timestamp.

Frozen v1 stopped because row0 had no valid forced EOS at prefix767. It had not
tested the remaining rows. Only this terminal position is repaired. Forbidden
EOS stays masked; allowed beam rows can complete. HF's original finished-beam
selection is observed without changing arguments, results, output flags or scores.
A previously genuinely finished, finite-scored, compliant path can be returned.
An initial unfinished placeholder is never a summary. The unchanged final guard
checks every accepted result. No rule, cap, marker, model, precision or dataset changes.

All24 frozen F cases remain in their exact order/random20+targeted4 strata.
Copy5 completed v1 outputs and analysis files byte-for-byte with separate provenance.
They completed successfully without the terminal exception; no regeneration.
Attempt cases6–24 exactly once:maximum19 new calls. The failed sixth v1 call is
retried once under this new instruction. Total v1+v2 calls are at most25, not24.
Only the typed expected no-compliant-terminal-output failure is recorded as missing
and continued. Any other integrity/numerical/memory/runtime failure stops globally.
No further retry, fallback, unmasking, cap extension or manual text repair.

Same historical F checkpoint, beam4/max_length768, hard rules, seed, source/target
budgets, runtime pins,15% memory headroom and original deadlines. No training,
Teacher API, held-out access, RO/U/Shuffled, checkpoint modification or automatic
subsequent experiment. Preserve D051 strict replay FAIL and all v1 evidence.

Technical completion is not quality GO. Semantic repetition, filler, Core coverage,
source grounding and zero-caption behavior still need evidence-linked assessment
and researcher review. Report missing cases and paired denominators transparently.
""")
    files=dict(read(ROOT/'outputs/stage_i/repetition_guard_preparation_v1/PREFLIGHT_MANIFEST.json')['files'])
    additions=[ROOT/'configs/stage_i_repetition_guard_pilot_v2.json',
        ROOT/'scripts/stage_i_repetition_guard_v2.py',ROOT/'scripts/stage_i_repetition_pilot_v2.py',
        ROOT/'scripts/run_stage_i_repetition_guard_pilot_v2.py',Path(__file__).resolve(),
        ROOT/'tests/test_stage_i_repetition_guard_v2.py',ROOT/'tests/test_stage_i_repetition_pilot_v2.py',
        ROOT/'outputs/stage_i/repetition_guard_preparation_v1/PREFLIGHT_MANIFEST.json']
    for directory in (prep,ROOT/'outputs/stage_i/repetition_guard_pilot_v1',
            ROOT/'outputs/stage_i/repetition_guard_execution_v1',
            ROOT/'outputs/stage_i/repetition_guard_independent_audit_v1',
            ROOT/'outputs/stage_i/repetition_guard_result_report_v1'):
        additions.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for path in additions:
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    write(prep/'PREFLIGHT_MANIFEST.json',{'experiment':'repetition_guard_pilot_v2',
        'files':dict(sorted(files.items())),'historical_bindings':426,
        'GPU_execution_approved':False,'D051_strict_replay':'FAIL_UNCHANGED'})
    print({'new_manifest_bindings':len(files),'config_sha256':sha(ROOT/'configs/stage_i_repetition_guard_pilot_v2.json'),
        'preflight_manifest_sha256':sha(prep/'PREFLIGHT_MANIFEST.json'),
        'runner_sha256':sha(ROOT/'scripts/run_stage_i_repetition_guard_pilot_v2.py')})


if __name__=='__main__':prepare()
