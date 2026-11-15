"""Integration check in a fresh process against the COMPLETE accepted parent.
No model construction or CUDA; early inference is replaced by a recording mock.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import sys
import json
import led_large_matched_family_v1 as m


def main():
    condition = sys.argv[1]
    parent = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(parent / 'scripts'))
    import stage_i_led_large_decoder_policy_v2 as h
    import run_stage_i_led_large_decoder_policy_v2 as b
    import led_large_decoder_early_sanity_v2 as early
    native = (b.train, b.feasible, b.load_model, b.configure_amp_scaler, b.collator, b.require_gate)
    b, c = m.install_condition_context(condition, parent)
    m.require(native == (b.train, b.feasible, b.load_model, b.configure_amp_scaler, b.collator, b.require_gate),
              'Native training/numerical/model functions were replaced')
    train, val = b.data(c)
    expected = m.binding(condition)
    m.require(b.binding() == h.binding() == expected, 'Receipt routing differs')
    m.require(b.require_gate.__globals__['binding']() == expected, 'Native gate verifier uses F binding')
    gate = b.read(b.LARGE_PREP / 'FEASIBILITY_CASES.json')
    m.require(gate == m.read(m.ROOT / m.PREP / f'{condition}_FEASIBILITY_CASES.json'), 'F stress cases leaked')
    Record = b.callback_class()
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        record = Record(folder, {}, object())
        record.trainer = SimpleNamespace(model=object())
        with patch.object(Record.__mro__[1], 'on_save', return_value=None), patch.object(early, 'run_early_sanity') as call:
            record.on_save(None, SimpleNamespace(global_step=250), None)
            m.require(call.call_count == 1, 'Early check must run once at step250')
            args = call.call_args.args
            operational = args[3]
            m.require(operational['condition'] == condition and args[5] == m.ROOT and args[6] == expected,
                      'Early condition/root/binding differs')
            m.require(Path(operational['source_tokenizer']) == parent / c['source_tokenizer'] and
                      Path(operational['effective_generation_config_file']) == parent / c['effective_generation_config_file'],
                      'Path resolution changed tokenizer/generation assets')
            cases = [json.loads(x) for x in (m.ROOT / operational['early_sanity']['cases_file']).read_text().splitlines()]
            m.require(all(x['condition'] == condition for x in cases), 'F case rows leaked into early inference')
            m.write(folder / 'early_sanity/REPORT.json', dict(status='PASS_TECHNICAL_ONLY', binding=expected, generation_calls=4))
            record.on_save(None, SimpleNamespace(global_step=500), None)
            m.require(call.call_count == 1, 'No additional early calls after step250')
    m.write(m.ROOT / m.PREP / f'{condition}_ROUTING_CPU.json', dict(status='PASS',
            condition=condition, rows=[len(train), len(val)], native_functions_unchanged=True,
            own_dataset_label_case_and_receipt_routing=True, early_case_path_resolution_test='MOCK_NO_INFERENCE',
            CUDA_calls=0, model_constructions=0, generation_calls=0, optimizer_updates=0,
            binding=expected))
    print(condition, 'CPU native-core/condition-routing integration PASS')


if __name__ == '__main__':
    main()
