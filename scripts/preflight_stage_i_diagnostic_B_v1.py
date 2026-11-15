"""Offline B integrity audit; no model load, generation, network or scientific edits."""
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from run_stage_i_decoding_diagnostic_local_v2 import checked_inputs, read, sha, require
from stage_i_decoding_diagnostics import CONDITIONS, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/decoding_diagnostic_v1'
CLOUD = ROOT / 'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2'
OUT = ROOT / 'outputs/stage_i/decoding_B_preflight_v1'


def main():
    OUT.mkdir(exist_ok=False)
    manifest, cases = checked_inputs(PREP, CLOUD)
    settings = read(PREP / 'decoding_configurations.json')
    for path, digest in manifest['source_file_bindings'].items():
        require(sha(ROOT / path) == digest, 'Upstream source/dataset/baseline changed: ' + path)
    import torch
    from transformers import LEDConfig, LEDTokenizerFast, GenerationConfig
    from transformers.generation.utils import GenerationMixin
    from stage_hi_led_data import LEDCompleteCollator
    for pkg, version in settings['required_runtime_versions'].items():
        require(importlib.metadata.version(pkg).split('+')[0] == version, 'Runtime drift: ' + pkg)
    tokenizer = LEDTokenizerFast.from_pretrained(ROOT / settings['tokenizer_directory'], local_files_only=True)
    aroot = ROOT / 'outputs/stage_i/decoding_greedy_A_local_v2'
    completion_files = list((aroot / 'attempts').glob('*/completion_report.json'))
    require(len(completion_files) == 1, 'Expected one completed A attempt')
    acomplete = read(completion_files[0])
    collator = LEDCompleteCollator(5000, 768, 1024)
    for condition in CONDITIONS:
        preds = read_jsonl(aroot / condition / 'predictions.jsonl')
        require(sha(aroot / condition / 'predictions.jsonl') == acomplete['predictions_sha256'][condition], 'A predictions changed')
        relevant = [c for c in cases if c['condition'] == condition]
        require([p['business_id'] for p in preds] == [c['business_id'] for c in relevant], 'A IDs/order differ')
        amap = {p['business_id']: p for p in preds}
        for c in relevant:
            require(hashlib.sha256(c['source'].encode()).hexdigest() == c['source_sha256'], 'Source hash mismatch')
            require(tokenizer(c['source'], truncation=False)['input_ids'] == c['input_ids'], 'Source token mismatch')
            require(tokenizer(c['target'], truncation=False)['input_ids'] == c['label_ids'], 'Target token mismatch')
            for p in (c['baseline_prediction'], amap[c['business_id']]):
                require(tokenizer.decode(p['token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False) == p['text'], 'Prediction token/text mismatch')
            batch = collator([{'input_ids': c['input_ids'], 'labels': c['label_ids']}])
            n = len(c['input_ids'])
            require(batch['input_ids'][0, :n].tolist() == c['input_ids'], 'Collator content mismatch')
            require(batch['global_attention_mask'].sum().item() == 1 and batch['global_attention_mask'][0, 0].item() == 1, 'Global attention mismatch')
    # Use the pinned, already-cached config without allocating model weights.
    config = LEDConfig.from_pretrained(settings['model_id'], revision=settings['revision'], local_files_only=True)
    for key, value in settings['backbone_checks'].items():
        require(getattr(config, key) == value, 'Pinned config mismatch: ' + key)
    effective = GenerationConfig.from_model_config(config)
    for key, value in settings['experiments']['beam_ngram_B']['generation_kwargs'].items():
        setattr(effective, key, value)
    effective.use_cache = True
    expected = settings['experiments']['beam_ngram_B']['expected_effective_generation_config_reconstruction']
    ignored = {'transformers_version', '_from_model_config', '_commit_hash'}
    actual = effective.to_dict()
    differences = {k: {'expected': expected.get(k), 'actual': actual.get(k)} for k in set(expected) | set(actual)
                   if k not in ignored and expected.get(k) != actual.get(k)}
    require(not differences, 'Unexpected reconstructed GenerationConfig mismatch')
    # Exercise the installed length-preparation method, without inference/model allocation.
    lengths = {}
    for field, value in [('max_length', 768), ('max_new_tokens', 768)]:
        candidate = deepcopy(effective)
        if field == 'max_new_tokens':
            candidate.max_length = GenerationConfig().max_length
            candidate.max_new_tokens = value
        candidate = GenerationMixin._prepare_generated_length(
            SimpleNamespace(config=config), candidate, field == 'max_new_tokens', True,
            'input_ids', 1, torch.zeros((1, 4000), dtype=torch.long))
        lengths[field + '=' + str(value)] = {'effective_total_max_length_including_decoder_start': candidate.max_length,
                                          'maximum_new_tokens_after_one_token_decoder_start': candidate.max_length - 1}
    require(lengths['max_length=768']['effective_total_max_length_including_decoder_start'] == 768, 'Unexpected baseline length')
    require(lengths['max_new_tokens=768']['effective_total_max_length_including_decoder_start'] == 769, 'Unexpected new-token length')
    source = inspect.getsource(GenerationMixin._prepare_generated_length)
    (OUT / 'installed_transformers_length_preparation.py.txt').write_text(source, encoding='utf-8')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    report = {
        'status': 'INPUT_INTEGRITY_PASS_BLOCKED_ON_CONFLICTING_GENERATION_LENGTH_INSTRUCTIONS',
        'authorization': 'Researcher approved bounded B and local/Colab execution; length wording conflicts with preservation requirement',
        'businesses': 24, 'random_businesses': 20, 'targeted_businesses': 4, 'expected_outputs': 72,
        'new_outputs_generated': 0, 'model_weights_loaded': False,
        'checkpoint_dataset_source_code_tokenizer_hashes': 'PASS', 'A_baseline_ID_alignment': 'PASS',
        'complete_source_target_token_roundtrips': 72, 'global_BOS_attention_checks': 72,
        'source_file_checksums': manifest['source_file_bindings'], 'checkpoint_checksums': manifest['checkpoints'],
        'subset_manifest_sha256': sha(PREP / 'subset_manifest.json'),
        'frozen_decoding_configurations_sha256': sha(PREP / 'decoding_configurations.json'),
        'reconstructed_B_config_matches_frozen_preparation': not differences,
        'reconstructed_B_generation_config': actual,
        'observed_actual_B_model_generation_config': None,
        'baseline_and_A_observed_or_saved_explicit_length': {'max_length': 768, 'max_new_tokens': None},
        'installed_transformers_length_test': lengths,
        'blocker': 'max_new_tokens=768 allows 769 total tokens; max_length=768 preserves the prior total cap. Await researcher choice; do not silently choose.',
        'recommended_resolution': 'Preserve max_length=768, changing only no_repeat_ngram_size from 0 to 4.',
        'GPU_snapshot': gpu,
        'memory_feasibility': 'RTX3050 6GB available; B beam4 peak not measured. A memory is not proof B fits. Stop on OOM; no fallback.',
        'versions': {p: importlib.metadata.version(p) for p in settings['required_runtime_versions']},
        'no_generation_training_teacher_requests_heldout_access': True,
    }
    write_json(OUT / 'PREFLIGHT_REPORT.json', report)
    (OUT / 'PREFLIGHT_REPORT.md').write_text(
        '# Diagnostic B preflight - generation-length clarification required\n\n'
        'Checkpoint, input, dataset, tokenizer, code and baseline/A bindings PASS for the same 24 validation businesses / 72 cases. '
        'Complete tokenization and global-BOS checks PASS. The prepared B GenerationConfig reconstructs without drift.\n\n'
        'The current request both says `max_new_tokens=768` and requires preserving the original length semantics. '
        'The saved baseline and observed A use `max_length=768`, including one decoder-start token. '
        'The installed Transformers 4.51.3 length-preparation function confirms that `max_new_tokens=768` permits 769 total tokens. '
        'These instructions cannot both hold. No choice was made and no B output was generated. '
        'Recommend retaining `max_length=768` so the sole baseline decoding change is no_repeat_ngram_size 0 -> 4; await researcher clarification.\n\n'
        'The local RTX3050 6GB is available. B beam-search memory has not been measured; the earlier A measurement does not establish B feasibility. '
        'Once length is resolved, load the exact adapters, verify the actual full GenerationConfig and stop on OOM or mismatch without changing settings.\n\n'
        'No weights, datasets, earlier predictions or scientific outputs were modified. No generation, training, Teacher request or held-out evaluation occurred.\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'new_outputs_generated', 'GPU_snapshot', 'installed_transformers_length_test']}))


if __name__ == '__main__':
    main()
