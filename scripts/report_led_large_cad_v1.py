"""Package actual CAD evidence after the single run; never calls a model."""
from pathlib import Path
from datetime import datetime,timezone
import collections
import argparse
import hashlib
import json
import re
import zipfile
R=Path(__file__).resolve().parents[1]
P=R/'outputs/stage_i/led_large_cad_v1'
E=R/'outputs/stage_i/led_large_cad_v1_local_execution_v1'
RUN=E/'run'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    global E,RUN
    ap=argparse.ArgumentParser();ap.add_argument('--vast',action='store_true');args=ap.parse_args()
    if args.vast:
        E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1';RUN=E/'downloaded/run'
        assert (E/'DOWNLOAD_VERIFIED.json').exists(),'Verified remote download required'
        assert (E/'downloaded/operations/STATUS.json').exists(),'Do not package a running process'
    else:
        assert (E/'PROCESS_EXIT.json').exists(),'Do not package a running process'
    for rel,h in read(P/'PREFLIGHT_MANIFEST.json')['files'].items():assert sha(R/rel)==h,rel
    cases=read(P/'TRAIN_SOURCE_ONLY_CASES.json');sources={c['business_id']:c for c in cases}
    complete=(RUN/'COMPLETION.json').exists()
    assert complete or (RUN/'FAILURE_STOP_NO_RETRY.json').exists()
    results=[read(x) for x in sorted((RUN/'cases').glob('*.json')) if not x.name.endswith('IDENTITY.json')]
    raw_calls=[read(x) for x in sorted((RUN/'calls').glob('*.json')) if not x.name.endswith('STARTED.json')]
    raw_transcript=['# All returned calls including technical and unpaired outputs','']
    for path in sorted((RUN/'calls').glob('*.json')):
        if path.name.endswith('STARTED.json'):continue
        item=read(path)
        raw_transcript.extend(['## '+path.stem,'',item.get('text') or '[NO TEXT RETURNED]',''])
    (E/'ALL_RETURNED_CALLS.md').write_text('\n'.join(raw_transcript),encoding='utf-8')
    h={c['business_id']:c for c in read(P/'HISTORICAL_TRAIN_N1_REFERENCE.json')}
    metrics=[];transcript=['# All unedited CAD paired TRAIN outputs','',
        'Model-assisted review only. No researcher-confirmed quality labels. No safeguard or rewrite applied.','']
    total_step_rows=0
    for path in (RUN/'calls').glob('*_steps.jsonl'):
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        assert all(x['branches_finite'] and x['conditional_contextless_prefix_exact'] for x in rows)
        if 'alpha0' in path.name:assert all(x['alpha0_raw_logits_bit_exact'] for x in rows)
        assert [x['step'] for x in rows]==list(range(len(rows)))
        total_step_rows+=len(rows)
    for result in results:
        bid=result['business_id'];src=sources[bid];name=src['source'].split('name=',1)[1].split('\n')[0]
        assert result['source_sha256']==src['source_sha256']
        assert result['alpha0']['raw_token_ids']==result['vanilla']['raw_token_ids']
        assert result['alpha0']['text']==result['vanilla']['text']
        row=dict(business_id=bid,name=name,captions=len(src['captions']),alpha0_exact=True,methods={},
            historical_vanilla_raw_exact=result['vanilla']['raw_token_ids']==h[bid]['single']['token_ids'],
            historical_vanilla_text_exact=result['vanilla']['text']==h[bid]['single']['text'])
        transcript.extend(['## '+name,'','Business ID: `'+bid+'`; Filtered captions: '+str(len(src['captions'])), ''])
        for method in ['vanilla','cad']:
            v=result[method];text=v['text'];ids=v['raw_token_ids'];review,sep,visual=text.partition('[VISUAL_EVIDENCE]')
            assert v['RNG']['transition_exact'] and v['RNG']['restoration_exact']
            assert ids[0]==2 and ids[-1]==2 and len(ids)<=768 and not v['errors']
            sentences=[s.strip().casefold() for s in re.split(r'(?<=[.!?])\s+',text) if s.strip()]
            row['methods'][method]=dict(tokens_including_start_EOS=len(ids),words=len(text.split()),
                review_words=len(review.replace('[REVIEW_CONSENSUS]','').split()),visual_words=len(visual.split()),
                generated_BOS=ids[1:].count(0),terminal_EOS=True,cap_hit=len(ids)>=768,
                exact_repeated_sentences={k:n for k,n in collections.Counter(sentences).items() if n>1},
                headers_exactly_once=text.count('[REVIEW_CONSENSUS]')==text.count('[VISUAL_EVIDENCE]')==1,
                seconds=v['seconds'],visual_text=visual.strip(),
                zero_caption_visual_assertions_require_review=not src['captions'] and bool(visual.strip()))
            transcript.extend(['### '+method,'',text,''])
        metrics.append(row)
    (E/'ALL_PAIRED_OUTPUTS.md').write_text('\n'.join(transcript),encoding='utf-8')
    write(E/'PAIRED_TECHNICAL_METRICS.json',metrics)
    status=read(RUN/('COMPLETION.json' if complete else 'FAILURE_STOP_NO_RETRY.json'))
    memory=read(RUN/'MEMORY_SAMPLES.json') if (RUN/'MEMORY_SAMPLES.json').exists() else None
    cpu_audit=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),complete=complete,paired_cases=len(results),
        alpha0_identities_verified=len(results),CAD_step_receipts_verified=total_step_rows,
        returned_call_receipts=len(raw_calls),all_recorded_RNG_transitions_and_restorations_exact=
            all(x.get('RNG',{}).get('transition_exact') and x.get('RNG',{}).get('restoration_exact') for x in raw_calls),
        historical_top1_exact_count=sum(x['historical_vanilla_raw_exact'] and x['historical_vanilla_text_exact'] for x in metrics),
        checkpoint_adapter_unchanged=status.get('weights_unchanged'),actual_run=status,
        no_fabricated_outputs=True,no_optimizer_or_training=True,held_out_access=False,
        technical_success_is_not_quality_GO=True)
    if complete:
        assert len(results)==12 and len(raw_calls)==37 and status['generation_calls']==37
        assert results[0]['cad']['raw_token_ids']==results[0]['cad_repeat']['raw_token_ids']
    write(E/'INDEPENDENT_EXPORTED_ARTIFACT_AUDIT.json',cpu_audit)
    report=['# CAD V1 actual execution report','',
        ('Hardware: Vast.ai NVIDIA RTX3090 24GB; locked Python3.11.7/runtime, original FP32 parameter storage/FP16 autocast; TF32 disabled.'
            if args.vast else 'Hardware: local NVIDIA RTX3050 Laptop6GB, original FP32 parameter storage/FP16 autocast; TF32 disabled.'),
        'No quantization, CPU offload, sampling, beam change, safeguard, optimizer, backward or training.',
        'Fixed checkpoint: Original LED-large CE epoch3 step750. Fixed alpha0.3.',
        'Contextless retains empty source section markers, removing all business metadata/reviews/captions.',
        '',f'Completed paired cases: {len(results)}/12. Returned technical/scientific call receipts: {len(raw_calls)}.',
        'RESOURCE STOP: incomplete pilot; do not present the three completed zero-caption pairs as a full12-case comparison.' if not complete else 'Complete technical pilot; substantive quality requires independent review.',
        f'Verified per-step CAD/prefix/finite-logit receipts: {total_step_rows}.',
        f'Historical RTX3090 vanilla top1 exactly reproduced: {cpu_audit["historical_top1_exact_count"]}/{len(results)}.',
        'Historical comparison is reported separately from mandatory SAME-HOST alpha0 exactness.',
        '', '```json',json.dumps(status,ensure_ascii=False,indent=2),'```','',
        'Quality findings are in CASE_LEVEL_REVIEW.md and reviewer annotations when available. Technical PASS alone is not quality GO.',
        'The12 previously examined TRAIN businesses are stratified9captioned/3zero-caption, not an independent or population-representative benchmark.',
        'No held-out/validation/Teacher/reference/photo evidence was opened or used. No additional experiment or main-study CAD promotion is authorized.']
    (E/'EXECUTION_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    bundle_files=list(x for folder in [P,E] for x in folder.rglob('*') if x.is_file())
    bundle_files += [R/'scripts'/x for x in ['led_large_cad_v1.py','run_led_large_cad_v1.py',
        'prepare_led_large_cad_v1.py','seal_led_large_cad_v1.py','execute_led_large_cad_local_v1.py',
        'close_led_large_cad_local_v1.py','report_led_large_cad_v1.py']]
    bundle_files += [R/'tests/test_led_large_cad_v1.py']
    if args.vast:
        bundle_files += [R/'scripts'/name for name in ['prepare_led_large_cad_v1_vast_deploy.py',
            'deploy_run_led_large_cad_v1_vast.py','inspect_download_led_large_cad_v1_vast.py',
            'review_led_large_cad_v1_vast.py','close_cad_vast_instance_after_verified_backup.py']]
        # The large byte-transfer TAR is reproducible from immutable originals;
        # hash receipt is included, model binary duplication is excluded.
        bundle_files=[p for p in bundle_files if p.name!='CAD_V1_FRESH_HOST_PAYLOAD.tar']
    # Quality annotations are added before invoking this final packager.
    manifest={str(x.relative_to(R)).replace('\\','/'):sha(x) for x in sorted(set(bundle_files))}
    shortcuts={'README_FIRST_VI.md':(E/'README_FIRST_VI.md').read_bytes(),
               'EXACT_BINDINGS.json':(P/'BINDINGS.json').read_bytes()}
    archive=R/'outputs/cloud_handoff'/('LED_LARGE_CAD_V1_VAST_CHATGPT_REVIEW.zip' if args.vast else 'LED_LARGE_CAD_V1_LOCAL_RESOURCE_STOP_REVIEW.zip')
    assert not archive.exists(),'Do not overwrite a sealed review package'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in manifest:z.write(R/rel,rel)
        for rel,content in shortcuts.items():
            z.writestr(rel,content);manifest[rel]=hashlib.sha256(content).hexdigest()
        z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=manifest),indent=2))
    with zipfile.ZipFile(archive) as z:
        for rel,digest in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    write(E/'DELIVERY_RECEIPT.json',dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
        manifest_entries=len(manifest),complete_pairs=len(results),independent_quality_decision='PENDING'))
    print(json.dumps(read(E/'DELIVERY_RECEIPT.json'),indent=2))


if __name__=='__main__':main()
