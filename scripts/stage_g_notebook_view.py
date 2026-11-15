"""Read-only Stage G notebook views; no credentials or API execution imports."""
from pathlib import Path
import html
import json
import pandas as pd
from IPython.display import display, Markdown, HTML, Image as NotebookImage

ROOT = Path(__file__).resolve().parents[1]
stage_g_cfg = json.loads((ROOT / 'configs/stage_g_teacher_pilot_v3_canonical_ids.json').read_text(encoding='utf-8'))
STAGE_G_DIR = ROOT / stage_g_cfg['output_directory']
STAGE_G_EXECUTION = STAGE_G_DIR / 'execution_v1'
stage_g_jobs = json.loads((STAGE_G_DIR / 'pilot_jobs.json').read_text(encoding='utf-8'))


def view_stage_g_pilot(business_id='EzJI5x25U5GgbYBWhu5ZQQ', split='train',
                       condition='unfiltered', show_reviews=False):
    jobs = [j for j in stage_g_jobs if j['business_id'] == business_id and j['split'] == split]
    if not jobs:
        raise ValueError('Business is outside the predeclared train/validation pilot')
    for job in jobs:
        if job['task'] == 'review_consensus' and show_reviews:
            display(Markdown('### Complete selected review input'))
            for review in job['evidence']['reviews']:
                display(HTML("<pre style='white-space:pre-wrap'>" +
                    html.escape(json.dumps(review, ensure_ascii=False, indent=2)) + '</pre>'))
        elif job['task'] == 'visual_evidence' and job['condition'] == condition:
            display(Markdown(f'### {condition}: exact captions and mapped originals'))
            if job.get('skip_api_empty_captions'):
                display(Markdown('**0 captions: empty visual evidence, no fabricated caption.**'))
            for caption, photo in zip(job['evidence']['captions'], job['images']):
                display(caption)
                display(NotebookImage(filename=str(ROOT / photo['relative_path']), width=450))


def view_stage_g_targets(business_id='EzJI5x25U5GgbYBWhu5ZQQ', show_evidence=True):
    path = STAGE_G_EXECUTION / 'provisional_targets' / f'{business_id}.json'
    if not path.exists():
        display(Markdown('No complete target set for this business yet.'))
        return
    record = json.loads(path.read_text(encoding='utf-8'))
    display(Markdown('**Provisional targets; human grounding verification pending.**'))
    for condition, target in record['targets'].items():
        display(Markdown(f'### {condition}'))
        display(HTML("<pre style='white-space:pre-wrap'>" + html.escape(target) + '</pre>'))
    if show_evidence:
        display(Markdown('### Evidence records (outside LED target text)'))
        display(record['evidence_records'])
        display(Markdown('### Serialized visual claims with all supporting caption/photo records'))
        display(record['visual_target_claims_with_all_evidence'])


def view_stage_g_completion():
    """Show audit statistics and expandable target text for the fixed pilot cohort."""
    path = STAGE_G_EXECUTION / 'pilot_integrity_and_tokens_report.json'
    if not path.exists():
        display(Markdown('Pilot is still running; complete integrity report is not available yet.'))
        return
    report = json.loads(path.read_text(encoding='utf-8'))
    display(Markdown('### Completed pilot integrity and target lengths'))
    display(pd.DataFrame([dict(condition=k, **v)
                          for k, v in report['LED_target_tokens_with_special_tokens'].items()]))
    display(pd.DataFrame(report['business_details']))
    display({k: report[k] for k in ['status', 'condition_target_texts',
        'reconstructed_actual_request_SHA_checks', 'slow_LED_target_token_ID_checks',
        'redundant_reported_review_ID_mismatch_count', 'human_confirmed_claims',
        'human_verified_businesses', 'held_out_teacher_targets', 'full_production_method_frozen']})
    display(Markdown('### Expand a business to read all three targets'))
    for row in report['business_details']:
        record = json.loads((STAGE_G_EXECUTION / 'provisional_targets' /
                            (row['business_id'] + '.json')).read_text(encoding='utf-8'))
        content = ''.join('<h4>' + html.escape(condition) + '</h4>' +
                         "<pre style='white-space:pre-wrap'>" + html.escape(text) + '</pre>'
                         for condition, text in record['targets'].items())
        display(HTML('<details><summary>' + html.escape(row['split'] + ' / ' + row['business_id']) +
                     '</summary>' + content + '</details>'))
    display(Markdown('To inspect source reviews/captions/images, use `view_stage_g_pilot(business_id=..., '
        'split=..., condition=..., show_reviews=True)`; source evidence is available with '
        '`view_stage_g_targets(business_id=..., show_evidence=True)`. '
        'These views do not submit or infer human verification.'))


def view_flagged_straw():
    photo_id = 'HVkUbl7niy2hIADyPh17MA'
    job = next(j for j in stage_g_jobs if j['business_id'] == 'EzJI5x25U5GgbYBWhu5ZQQ'
               and j['condition'] == 'unfiltered')
    pair = next(c for c in job['evidence']['captions'] if c['photo_id'] == photo_id)
    photo = next(i for i in job['images'] if i['photo_id'] == photo_id)
    display(Markdown('### Image to inspect: straw claim'))
    display(Markdown(f"**BLIP-2 caption:** {pair['caption']}"))
    feedback_path = STAGE_G_EXECUTION / 'human_claim_verifications.json'
    records = json.loads(feedback_path.read_text(encoding='utf-8'))['records'] if feedback_path.exists() else []
    confirmed = next((r for r in records if r['photo_id'] == photo_id and r['human_verified']), None)
    display(Markdown('**Teacher claim:** The cup has a straw. **Teacher verdict:** verified.'))
    if confirmed:
        display(Markdown('**User confirmation:** verified after image inspection. '
                         '**Information quality:** user considers it simple but correct. '
                         '**Previous AI flag:** resolved; the AI reviewer had missed the straw.'))
    else:
        display(Markdown('**AI sanity check:** visibility unclear; user confirmation pending.'))
    display(NotebookImage(filename=str(ROOT / photo['relative_path']), width=500))
    display(Markdown('This view does not change teacher outputs or human labels.'))
