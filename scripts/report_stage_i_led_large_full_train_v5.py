"""Offline export of actual saved predictions; no factual judge or model execution."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
import statistics
import zipfile
from stage_i_led_large_full_train_v5 import ROOT, CONFIG, LARGE_PREP, read, write, sha, require, binding


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a = ap.parse_args();run = a.output.resolve()
    complete = read(run/'evaluate/COMPLETION.json')
    require(complete['generation_calls'] == 48 and complete['binding'] == binding(), 'Exactly 48 bound predictions required')
    dest = run/'quality_review';dest.mkdir(exist_ok=False)
    cfg = read(CONFIG)
    cases = [json.loads(s) for s in (ROOT/'outputs/stage_i/paired_lora_pilot_preparation_v1/evaluation_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    references = read(ROOT/cfg['final_references']);models = {};metrics = []
    for epoch in (3,5):
        rows = read(run/f'evaluate/epoch{epoch}/predictions.json')
        require([x['business_id'] for x in rows] == [x['business_id'] for x in cases], 'Case order differs')
        models[f'LED_large_epoch{epoch}'] = {x['business_id']:x for x in rows}
        write(dest/f'predictions/epoch{epoch}.json',rows)
        for subset in ('random','targeted'):
            group = [x for x in rows if x['subset'] == subset]
            require(len(group) == (20 if subset == 'random' else 4), 'Subset size differs')
            metrics.append(dict(epoch=epoch,subset=subset,n=len(group),
                exact_sentence_repeat_cases=sum(x['automatic']['repetition']['exact_repeated_sentence'] for x in group),
                legacy_adjacent_word_flag_cases=sum(x['automatic']['repetition']['consecutive_repeated_word'] for x in group),
                headers_compliant_cases=sum(x['automatic']['structure']['section_headers_compliant'] for x in group),
                length_cap_cases=sum(x['automatic']['termination']['cap_reached'] for x in group),
                EOS_before_cap_cases=sum(x['automatic']['termination']['eos_before_length_cap'] for x in group),
                mean_length_including_decoder_start=statistics.mean(len(x['token_ids']) for x in group)))
    # Saved benchmark rows only; these do NOT constitute a matched causal architecture comparison.
    for name,rel in [('D059_Expanded_epoch3','outputs/stage_i/d059_expanded_filtered_v2/evaluate/epoch3/predictions.json'),
                     ('D058_B_epoch5','outputs/stage_i/d058_full_filtered_B_v1/evaluate/epoch5/predictions.json')]:
        rows = read(ROOT/rel);models[name] = {x['business_id']:x for x in rows}
        write(dest/f'benchmarks/{name}.json',rows)
    write(dest/'SOURCES_AND_TEACHER_TARGETS.json',cases);write(dest/'APPROVED_REFERENCES.json',references)
    write(dest/'AUTOMATIC_METRICS.json',metrics)
    from stage_i_d053_evaluation import human_output_annotation_template
    eligible = [u for u in references['units'] if u['importance'] in ('core','supporting') and u['support_status'] == 'supported']
    templates = [human_output_annotation_template(row,[u for u in eligible if u['business_id'] == bid])
                 for name,rows in models.items() if name.startswith('LED_large') for bid,row in rows.items()]
    require(len(templates) == 48, 'Missing blank human review records')
    write(dest/'BLANK_RESEARCHER_OUTPUT_ANNOTATIONS.json',templates)
    sections = ['<meta charset="utf-8"><title>LED-large F quality review</title><style>body{font:16px system-ui;margin:24px;line-height:1.5}pre{white-space:pre-wrap}article{border-top:3px solid #777;padding:24px 0}details{margin:12px 0}figure{display:inline-block;max-width:260px;vertical-align:top}img{max-width:250px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>',
                '<h1>Actual LED-large F outputs — researcher review pending</h1><p>Automatic repetition/header/EOS metrics are software checks. Core coverage, sentiment contrasts, grounding, semantic repetition, visual usefulness and substantive completion require source-based researcher assessment. Legacy adjacent-word flags can include legitimate names or boundaries. No automatic quality GO.</p>']
    for case in cases:
        bid = case['business_id'];sections.append(f'<article><h2>{html.escape(bid)} — {case["subset"]}</h2>')
        sections.append('<details><summary>Complete frozen source and Teacher target (not independent truth)</summary><pre>'+html.escape(case['source'])+'</pre><h3>Teacher</h3><pre>'+html.escape(case['target'])+'</pre></details>')
        for pid in case['selected_photo_ids']:
            src = ROOT/'data/processed/yelp_2300_v1/photos'/(pid+'.jpg');dst = dest/'images'/(pid+'.jpg')
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():shutil.copy2(src,dst)
            sections.append(f'<figure><a href="images/{pid}.jpg"><img src="images/{pid}.jpg"></a><figcaption>PHOTO {html.escape(pid)}</figcaption></figure>')
        units = [u for u in eligible if u['business_id'] == bid]
        sections.append('<details><summary>Approved Core/Supporting units and evidence IDs</summary><pre>'+html.escape(json.dumps(units,ensure_ascii=False,indent=2))+'</pre></details>')
        old = case.get('baseline_prediction')
        if old is not None:
            sections.append('<h3>Historical full F (practical benchmark)</h3><pre>'+html.escape(old if isinstance(old,str) else json.dumps(old,ensure_ascii=False,indent=2))+'</pre>')
        for name,rows in models.items():
            row = rows[bid];sections.append('<h3>'+html.escape(name)+'</h3><pre>'+html.escape(row['text'])+'</pre>')
        sections.append('</article>')
    (dest/'RESULTS_REVIEW.html').write_text('\n'.join(sections),encoding='utf-8')
    lines = ['# LED-large F execution and software-metric report','',
             '**Researcher quality decision: PENDING. No automatic scientific GO/NO-GO.**','',
             'Train: 2000; validation loss: 200; fixed development generation: 24 × epochs3/5 =48.',
             'Historical F, D058 B and D059 Expanded are practical non-causal benchmarks, not a matched architecture ablation.',
             'Keep random20 and targeted4 separate. Teacher-target agreement is not factual accuracy.',
             '', '|epoch|subset|n|sentence repeats|legacy word flags|headers|cap hits|EOS before cap|mean length|',
             '|---|---|---|---|---|---|---|---|---|']
    for m in metrics:
        lines.append('|'+ '|'.join(str(m[k]) for k in ('epoch','subset','n','exact_sentence_repeat_cases','legacy_adjacent_word_flag_cases',
            'headers_compliant_cases','length_cap_cases','EOS_before_cap_cases','mean_length_including_decoder_start'))+'|')
    lines += ['', 'Read RESULTS_REVIEW.html with complete sources/captions/photos and approved Core/Supporting references.',
              'Complete blank annotations; excluded/uncertain units must not enter scoring denominators.',
              'Lower repetition, lower validation loss or correct headers alone cannot establish usable summary quality.',
              'No RO/U training is triggered. Preserve D051 strict replay FAIL. Stop for independent review.']
    (dest/'EXECUTION_AND_QUALITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for folder in ('feasibility','train','evaluate'):
        for name in ('REPORT.json','COMPLETION.json','AUTHORIZATION.json','EFFECTIVE_GENERATION_CONFIG.json','ACTUAL_TRAINING_ARGUMENTS.json',
                     'FROZEN_BACKBONE_INTEGRITY.json','CHECKPOINT_RECEIPTS.jsonl','progress.jsonl','ACTUAL_AMP_INITIALIZATION.json','ACCEPTED_V4_GATE.json'):
            src = run/folder/name
            if src.exists():
                dst = dest/'execution_evidence'/folder/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    write(dest/'ACTUAL_CONFIG.json',cfg)
    (dest/'README_FIRST.md').write_text('Read EXECUTION_AND_QUALITY_REPORT.md, RESULTS_REVIEW.html and full source evidence. This ZIP contains actual saved outputs, not edited summaries. Human annotations are blank. No RO/U automatic continuation. Full optimizer/adapter checkpoints remain in the persistent run directory; this quality bundle contains receipts.\n',encoding='utf-8')
    (dest/'CHATGPT_REVIEW_REQUEST.md').write_text('Independently assess random20 and targeted4 separately: Core coverage, business specificity, positive/negative opinion synthesis, source/caption/photo grounding, semantic repetition, zero-caption compliance and completeness. Compare epochs3/5 against the saved practical benchmarks; disclose confounds. Software repetition flags are not human labels. Do not declare GO from headers/loss/exact repeats alone. Propose matched RO/U only after researcher confirmation.\n',encoding='utf-8')
    write(dest/'SHA256_MANIFEST.json',dict(files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}))
    archive = run/'LED_LARGE_F_QUALITY_REVIEW.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest).as_posix())
    write(run/'QUALITY_BUNDLE_RECEIPT.json',dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
          actual_predictions=48,researcher_quality_decision='PENDING',human_labels_assigned=0))
    print(archive)


if __name__ == '__main__':
    main()
