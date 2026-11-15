"""Read-only analysis of saved TRAIN/validation outputs. No technique implementation."""
from pathlib import Path
import collections
import hashlib
import json
import re
import shutil
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_i/led_large_output_improvement_decision_v1'
RUN = ROOT / 'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2'
ZIP = ROOT / 'outputs/cloud_handoff/LED_LARGE_OUTPUT_IMPROVEMENT_DECISION_V1_REVIEW.zip'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def body(text):
    assert text.count('[REVIEW_CONSENSUS]') == text.count('[VISUAL_EVIDENCE]') == 1
    review, visual = text.split('[VISUAL_EVIDENCE]')
    return review.split('[REVIEW_CONSENSUS]')[1].strip(), visual.strip()


def words(text):
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text))


def captions(source):
    tail = source.partition('[VISUAL_CONTEXT]')[2]
    return [{'id': m.group(1), 'text': m.group(2).strip()} for m in re.finditer(
        r'\[(IMG_CAP_\d+)\]\s*([\s\S]*?)(?=\[IMG_CAP_\d+\]|$)', tail)]


def main():
    assert not OUT.exists() and not ZIP.exists(), 'Preserve versioned result'
    OUT.mkdir(parents=True)
    source_path = RUN / 'quality_review/SOURCES_AND_TEACHER_TARGETS.json'
    pred_path = RUN / 'evaluate/epoch3/predictions.json'
    sources, predictions = read(source_path), read(pred_path)
    assert len(sources) == len(predictions) == 24
    by_id = {p['business_id']: p for p in predictions}
    assert len(by_id) == 24 and set(by_id) == {s['business_id'] for s in sources}
    assert all(s['split'] == 'validation' for s in sources)
    checkpoint_sha = '507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60'
    rows = []
    for s in sources:
        p = by_id[s['business_id']]
        assert hashlib.sha256(s['source'].encode('utf-8')).hexdigest() == s['source_sha256'] == p['source_sha256']
        assert p['checkpoint_sha256'] == checkpoint_sha
        review, visual = body(p['text'])
        tr, tv = body(s['target'])
        caps = captions(s['source'])
        rows.append({'business_id': s['business_id'], 'name': re.search(r'name=(.*)', s['source']).group(1),
            'subset': p['subset'], 'caption_count': len(caps), 'source_sha256': s['source_sha256'],
            'generated_review_words': words(review), 'generated_visual_words': words(visual),
            'target_review_words': words(tr), 'target_visual_words': words(tv),
            'raw_caption_words': sum(words(c['text']) for c in caps),
            'captions': caps, 'stored_automatic': {k:p['automatic'][k] for k in ('termination','repetition')},
            'raw_prediction': p['text'], 'original_target': s['target']})
    def aggregate(part):
        return {'N': len(part), 'mean_review_words': statistics.mean(x['generated_review_words'] for x in part),
            'mean_visual_words': statistics.mean(x['generated_visual_words'] for x in part),
            'mean_raw_caption_words': statistics.mean(x['raw_caption_words'] for x in part),
            'exact_sentence_repeat_cases_from_stored_flags': sum(x['stored_automatic']['repetition']['exact_repeated_sentence'] for x in part),
            'length_cap_hits': sum(x['stored_automatic']['termination']['cap_reached'] for x in part),
            'BOS_after_start_cases': sum(x['stored_automatic']['termination']['generated_bos_after_start'] for x in part),
            'zero_caption_cases': sum(x['caption_count'] == 0 for x in part)}
    diagnostics = {'scope': '24 saved validation CE3 outputs, original sources/targets and all90 model-visible captions; no new generation',
        'word_count_definition': "Unicode regex \\b\\w+(?:['’-]\\w+)*\\b; bodies exclude section headers; not BPE",
        'overall_descriptive_only': aggregate(rows),
        'random20': aggregate([x for x in rows if x['subset'] == 'random']),
        'targeted4': aggregate([x for x in rows if x['subset'] == 'targeted']),
        'all_captions_raw_shorter_equal_longer_than_generated_visual': {
            'shorter': sum(x['raw_caption_words'] < x['generated_visual_words'] for x in rows),
            'equal': sum(x['raw_caption_words'] == x['generated_visual_words'] for x in rows),
            'longer': sum(x['raw_caption_words'] > x['generated_visual_words'] for x in rows)},
        'word_length_is_not_semantic_quality': True, 'exhaustive_semantic_Core_recall_measured': False,
        'new_researcher_confirmed_output_labels': 0, 'rows': rows}
    write('OFFLINE_DIAGNOSTICS.json', diagnostics)
    # Evidence excerpts preserve exact source character offsets. No candidate summaries created.
    highlights = {
        'JOEY Mayfield': ['preparing food', 'women sitting', 'bowl of food'],
        'Scarlett Begonia': ['Lemon ricotta pancakes and eggs Benedict'],
        "Thai O'Cha": ['Thai hot does not', 'Thai hot spiciness'],
        'Brooklyn Water Bagel': ['coffee ice cubes', 'bagel sandwich'],
        'Peel Wood Fired Pizza': ['particularly bland', 'slightly burnt'],
        'Starbucks': ['Made a mobile order', 'get a refund'],
        "Prairie Fish 'N' Chips": ['bottle of beer'],
        'Pool Bar & Grill - Seminole Hard Rock Tampa': ['a dj is standing'],
    }
    evidence = []
    for s in sources:
        name = re.search(r'name=(.*)', s['source']).group(1)
        if name not in highlights:
            continue
        quotes = []
        for term in highlights[name]:
            m = re.search(re.escape(term), s['source'], re.I)
            assert m, (name, term)
            start, end = max(0,m.start()-110), min(len(s['source']),m.end()+230)
            quote = s['source'][start:end]
            assert s['source'][start:end] == quote
            quotes.append({'search_phrase': term, 'start': start, 'end': end, 'exact_source_quote': quote})
        evidence.append({'business_id': s['business_id'], 'name': name, 'source_sha256': s['source_sha256'],
            'quotes': quotes, 'all_captions': captions(s['source']), 'original_target': s['target'],
            'unchanged_CE3_output': by_id[s['business_id']]['text'],
            'assessment_status': 'MODEL_ASSISTED_PROVISIONAL_NOT_RESEARCHER_CONFIRMED'})
    write('SOURCE_LINKED_EXAMPLES.json', evidence)
    measured = read(ROOT / 'outputs/stage_i/led_large_section_wce_v1/DIAGNOSTIC_MEASUREMENTS.json')
    write('HISTORICAL_TARGET_TOKEN_DIAGNOSTIC_UNCHANGED.json', measured)
    dependencies = [source_path, pred_path, RUN/'evaluate/EFFECTIVE_GENERATION_CONFIG.json',
        ROOT/'scripts/run_stage_i_led_large_decoder_policy_v2.py', ROOT/'scripts/led_large_decoder_label_view_v1.py',
        ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/config.json',
        ROOT/'outputs/stage_i/led_large_section_wce_v1/DIAGNOSTIC_FINDINGS.md',
        ROOT/'outputs/stage_i/led_large_section_wce_quality_review_v1/QUALITY_REVIEW_VI.md',
        ROOT/'outputs/stage_i/led_large_nbest_v2_execution_v1/TRAIN_SOURCE_GROUNDED_PROVISIONAL_REVIEW.md',
        ROOT/'outputs/stage_i/led_large_exploratory_contrastive_v3_execution_v1/QUALITY_COMPARISON.md',
        ROOT/'outputs/stage_i/led_large_cad_v1_vast_execution_v1/CASE_LEVEL_REVIEW.md',
        ROOT/'outputs/stage_i/led_large_boundary_global_feasibility_v1_execution_v1/PRE_GPU_STOP_REVIEW.md']
    write('INPUT_EVIDENCE_MANIFEST.json', {'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
        'CE3_adapter_binding': checkpoint_sha, 'model_weights_loaded_or_modified': False,
        'frozen_targets_modified': False, 'new_inference_calls': 0, 'new_training_updates': 0,
        'new_improvement_technique_implemented': False, 'held_out_payload_access': False,
        'paused_thesis_stage_artifacts_modified': False})
    write('PRIMARY_RESEARCH_SOURCES.json', {'sources': [
        {'url':'https://aclanthology.org/2020.emnlp-main.750/','role':'Motivates checking sentence claims against source spans; does not validate our proposed renderer or a factuality score here.'},
        {'url':'https://arxiv.org/abs/2004.05150','role':'Local+task-motivated global attention; architecture rationale, not evidence for our boundary mask.'},
        {'url':'https://huggingface.co/transformers/v4.3.0/model_doc/led.html','role':'Official advice uses first-token global attention for summarization; counterweight to a presumed boundary benefit.'},
        {'url':'https://arxiv.org/abs/2106.14448','role':'R-Drop bidirectional KL consistency under two dropout passes; transfer to this setting unproven.'},
        {'url':'https://github.com/dropreg/R-Drop','role':'Original authors implementation; no code adopted or executed.'}],
        'access_date_local':'2026-10-25','versioned_v4_51_3_LED_documentation_fetch':'Unavailable through browsing; actual static configuration/code checked locally instead.'})
    # Portable evidence is a byte copy of existing validation artifacts, never a new output.
    raw = OUT / 'EXISTING_VALIDATION_EVIDENCE'
    raw.mkdir()
    shutil.copyfile(source_path, raw / source_path.name)
    shutil.copyfile(pred_path, raw / 'CE3_predictions_unchanged.json')
    shutil.copyfile(RUN/'evaluate/EFFECTIVE_GENERATION_CONFIG.json', raw / 'EFFECTIVE_GENERATION_CONFIG.json')
    (OUT / 'DIAGNOSTIC_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'diagnostics': diagnostics['overall_descriptive_only'], 'source_examples': len(evidence),
                      'raw_caption_counts': sum(r['caption_count'] for r in rows), 'new_outputs_generated': 0},indent=2))


if __name__ == '__main__':
    main()
