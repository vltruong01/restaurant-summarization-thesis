"""Portable offline evidence report after terminal v2 status; no model imports."""
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write((v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))+'\n')
def esc(v):return html.escape(str(v))


def build():
    run=ROOT/'outputs/stage_i/repetition_guard_pilot_v2'
    terminal=run/'COMPLETION.json'
    if not terminal.exists():terminal=run/'FAILURE_STOP_NO_RETRY.json'
    if not terminal.exists():raise ValueError('Active run; no packaging of incomplete changing evidence')
    status=read(terminal)
    c=read(ROOT/'configs/stage_i_repetition_guard_pilot_v2.json')
    cases=[json.loads(x) for x in (ROOT/c['cases']).read_text(encoding='utf-8').splitlines() if x.strip()]
    source_path=ROOT/'outputs/stage_i/d053_reference_verification_v1/source_cases.json'
    source={x['business_id']:x for x in read(source_path)}
    baseline_b={x['business_id']:x for x in [json.loads(s) for s in (ROOT/c['saved_diagnostic_B']).read_text(encoding='utf-8').splitlines() if s.strip()]}
    preds={p.stem:read(p) for p in (run/'raw_cases').glob('*.json')}
    out=ROOT/'outputs/stage_i/repetition_guard_result_report_v2'
    out.mkdir(exist_ok=False)
    audit_dir=ROOT/'outputs/stage_i/repetition_guard_independent_audit_v2'
    audit=read(audit_dir/'INDEPENDENT_SOFTWARE_AUDIT.json')
    groups=read(audit_dir/'ACCEPTED_METRICS_RECOMPUTED.json') if (audit_dir/'ACCEPTED_METRICS_RECOMPUTED.json').exists() else {}
    lexical=read(audit_dir/'LEXICAL_AND_ADJACENT_SUMMARY.json') if (audit_dir/'LEXICAL_AND_ADJACENT_SUMMARY.json').exists() else {}
    quality_path=ROOT/'outputs/stage_i/repetition_guard_quality_assessment_v2/AI_QUALITY_REVIEW.json'
    quality=read(quality_path) if quality_path.exists() else {'status':'NO_NEW_QUALITY_ASSESSMENT','human_verified':False}
    targeted_quality_path=quality_path.parent/'AI_TARGETED_REVIEW.json'
    if targeted_quality_path.exists():quality={**quality,'additional_targeted_findings':read(targeted_quality_path)}
    # Every displayed arm keeps the saved original string verbatim.
    blocks=[]
    for index,case in enumerate(cases,1):
        bid=case['business_id'];s=source[bid];p=preds.get(bid)
        arms=[('Historical F (saved)',case['baseline_prediction']),('Diagnostic B (saved)',baseline_b[bid]),
              ('Guard: unchanged v1 reuse' if index<=5 else 'Guard v2 terminal repair',p)]
        pics=[]
        for cap in s['captions']:
            photo=source_path.parent/cap['image_file']
            data=base64.b64encode(photo.read_bytes()).decode('ascii')
            pics.append(f"<figure><img loading='lazy' src='data:image/jpeg;base64,{data}'><figcaption>{esc(cap['caption_id'])} / PHOTO {esc(cap['photo_id'])}<br>{esc(cap['text'])}</figcaption></figure>")
        panels=''.join(f"<section><h3>{esc(name)}</h3><pre>{esc(pred['text']) if pred else 'NO ACCEPTED OUTPUT — see preserved failure record'}</pre></section>" for name,pred in arms)
        proof=(p or {}).get('completion_proof',{})
        blocks.append(f"<article data-subset='{esc(case['subset'])}'><h2>{index}. {esc(s['display_name'])} / {esc(case['subset'])}</h2><code>{esc(bid)}</code><p>Status: {'REUSED v1 bytes' if index<=5 else 'NEW v2 output' if p else 'MISSING OUTPUT'}; tokens={len(p['token_ids']) if p else 'NA'}; terminal cap={esc(proof.get('diagnostic_length_cap_reached','see original v1 termination'))}</p><div class='grid'>{panels}</div><details><summary>Original LED-visible source and Teacher comparison (Teacher is not independent ground truth)</summary><pre>{esc(case['source'])}\n\nTEACHER\n{esc(case['target'])}</pre></details><details><summary>Original captions and photographs ({len(pics)})</summary><div class='photos'>{''.join(pics)}</div></details><details><summary>Teacher evidence mappings / original verdicts</summary><pre>{esc(json.dumps(case['evidence_records'],ensure_ascii=False,indent=2))}</pre></details></article>")
    table='\n'.join('|'+key+'|'+str(g.get('outputs','?'))+'|'+str(g.get('exact_repeated_sentence',{}).get('count','NA'))+'|'+str(lexical.get(key,{}).get('outputs_with_hard_adjacent_word_violation','NA'))+'|' for key,g in groups.items())
    extra_table='\n'.join('|'+key+'|'+str(g.get('outputs','?'))+'|'+str(g.get('consecutive_repeated_word',{}).get('count','NA'))+'|'+str(lexical.get(key,{}).get('outputs_with_hard_list_violation','NA'))+'|'+str(g.get('section_headers_compliant',{}).get('count','NA'))+'|'+str(g.get('cap_reached',{}).get('count','NA'))+'|' for key,g in groups.items())
    text=f"""# Repetition guard v2: execution repair and quality review

Accepted outputs:{len(preds)}/24. Five v1 outputs are reused byte-for-byte;
new attempted calls:{status.get('new_generation_calls',status.get('new_calls_attempted','see failure'))}.
V1 attempted6 calls; v2 permits19, including one explicitly authorized retry
of failed case6. Total maximum25 calls for24 unique validation cases.

Technical repair: only forced-EOS position767 prunes forbidden terminal rows
and observes actual HF finished hypotheses. No rule/cap/score/checkpoint/data
change. No EOS exemption, manual output repair, training or Teacher calls.
Terminal status:{status.get('status',status.get('error'))}.
Software audit:{audit['status']}. Full metric definitions/counts are in the
independent audit JSON. Do not equate software PASS with scientific quality GO.

| Arm/subset | n | Exact-sentence repetition | Hard literal adjacent violation |
|---|---:|---:|---:|
{table}

| Arm/subset | n | Legacy adjacent-word flags | Hard duplicate-list flags | Correct headers | Length-cap hits |
|---|---:|---:|---:|---:|---:|
{extra_table}

The legacy adjacent-word metric drops punctuation and can flag sentence boundaries;
its unchanged values and descriptive decomposition are supplied separately.
For guard/random,7 of8 flags are only sentence boundaries; the eighth includes
the supported name Bang Bang. The one targeted flag is “rude, rude behavior”,
a phrase extension rather than a forbidden identical completed list item.
Lexical n-gram reduction is not evidence of semantic non-redundancy or grounding.

Literal whitespace-only forbidden stutters are0 in all saved arms; do not claim
a measured reduction on that narrow metric. Duplicate-list flags drop from23/24
historical F and24/24 saved B to0/24 under the frozen surface parser, but visible
near-duplicate clauses and extended list items remain. Lexical4gram duplicate
fractions:random F26.80%,B0.35%,guard23.12%;targeted F41.06%,B0.63%,guard20.01%.
The guard performs substantially worse than B on lexical duplication despite
matching its0 exact-sentence count. Headers are19/20 random and2/4 targeted;
zero-caption compliance remains0/1. Those are structural metrics, not human labels.

Runtime:{status.get('seconds','unavailable')} seconds. PyTorch peak allocated:
{status.get('peak_allocated_bytes','unavailable')} bytes; peak reserved:
{status.get('peak_reserved_bytes','unavailable')} bytes. These are not total-GPU
VRAM peaks. Final on-disk and runtime adapter/backbone equality assertions passed.
The process-observation receipt has exit_code=null; no OS exit-code0 is claimed.

Quality assessment status:{quality.get('status','PROVISIONAL_AI_REVIEW')}.
All new assessments are provisional AI-assisted evidence inspection, not human
labels. No researcher GO/NO-GO has been assigned. Core/Supporting references keep
their approved roles; no automatic664-unit denominator is introduced.

Case6 now returns an actual finite-scored finished hypothesis at the768-token
ceiling, but still has severe phrase/semantic looping, missing visual section
and an incomplete ending. Technical crash resolution is not a quality cure.
Inspect the raw side-by-side outputs and evidence, including zero-caption behavior.

Preserved:D051 strict replay FAIL, all original data/targets/references/checkpoints,
v1 failure and both original deadlines. Stop after review; no automatic next experiment.
"""
    write(out/'REPORT.md',text)
    write(out/'EXECUTION_AND_AUDIT_SUMMARY.json',{'execution':status,'software_audit':audit,'human_quality_decision':'PENDING_RESEARCHER','quality_review':quality})
    header=f"<h1>Repetition guard v2 — {len(preds)}/24 outputs</h1><p>Five unchanged v1 outputs + new v2 results. Terminal-only technical repair; quality decision remains researcher-pending. <b>Hard-rule PASS does not establish factual grounding or semantic quality.</b></p><details><summary>Execution and independent software report</summary><pre>{esc(text)}</pre></details><details><summary>Provisional AI evidence-linked quality review</summary><pre>{esc(json.dumps(quality,ensure_ascii=False,indent=2))}</pre></details>"
    page="<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Repetition guard v2 results</title><style>body{font:16px system-ui;margin:24px;background:#f3f5f8;color:#17202a}article,section,details{background:white;padding:16px;margin:12px 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.photos{display:flex;flex-wrap:wrap;gap:12px}figure{margin:8px;max-width:320px}img{max-width:100%;max-height:320px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}code{overflow-wrap:anywhere}summary{cursor:pointer}@media(max-width:1000px){.grid{grid-template-columns:1fr}}</style>"+header+''.join(blocks)+"</html>"
    write(out/'RESULTS_REVIEW.html',page)
    bundle=ROOT/'outputs/stage_i/stage_i_repetition_guard_review_v2_bundle'
    bundle.mkdir(exist_ok=False)
    included={}
    def copy(p):
        p=Path(p);rel=p.relative_to(ROOT);dest=bundle/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);included[rel.as_posix()]=sha(dest)
    for directory in (out,run,audit_dir,quality_path.parent,
            ROOT/'outputs/stage_i/repetition_guard_preparation_v2',ROOT/'outputs/stage_i/repetition_guard_execution_v2',
            ROOT/'outputs/stage_i/repetition_guard_independent_audit_v1',ROOT/'outputs/stage_i/repetition_guard_quality_assessment_v1',
            ROOT/'outputs/stage_i/repetition_guard_preparation_v1',ROOT/'outputs/stage_i/repetition_guard_execution_v1'):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:copy(p)
    # Frozen v1 raw data/incident, without copying previous ZIPs or checkpoints.
    for p in (ROOT/'outputs/stage_i/repetition_guard_pilot_v1').rglob('*'):
        if p.is_file():copy(p)
    for pattern in ('*repetition*guard*v*.py','stage_i_repetition_pilot_v*.py'):
        for p in (ROOT/'scripts').glob(pattern):copy(p)
    for p in (ROOT/'tests').glob('test_stage_i_repetition*.py'):copy(p)
    for rel in (c['cases'],c['saved_diagnostic_B'],c['expected_generation_config'],c['final_references'],
            'configs/stage_i_repetition_guard_pilot_v1.json','configs/stage_i_repetition_guard_pilot_v2.json',
            'scripts/stage_i_decoding_diagnostics.py','scripts/stage_i_d053_evaluation.py',
            'scripts/audit_stage_i_repetition_guard_results_v1.py',
            'thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','DECISIONS.md'):
        copy(ROOT/rel)
    copy(source_path)
    for s in source.values():
        for cap in s['captions']:copy(source_path.parent/cap['image_file'])
    for p in (ROOT/c['tokenizer_directory']).glob('*'):
        if p.is_file():copy(p)
    readme="""# Read first

Open outputs/stage_i/repetition_guard_result_report_v2/RESULTS_REVIEW.html.
It embeds original photos and includes all24 frozen cases, exact saved F/B/new
outputs, LED sources and Teacher comparisons. Five v1 outputs are reused unchanged.
Missing outputs, if any, are explicit. No content is manually rewritten.

Review REPORT.md, independent audit JSON, raw results, AI quality evidence and
authorization/technical repair provenance. Researcher quality acceptance is pending.
The v1 failure, original protocol and strict D051 replay FAIL are preserved.

SHA256_MANIFEST.json hashes every delivered payload except itself. Verify before
review. This is a self-contained evidence review package, not a standalone GPU
workspace: model checkpoint binaries and full historical426-file workspace are
deliberately excluded. Original integrity manifests preserve their full bindings.
No secrets or API credentials are included. No further experiment is authorized.

CPU-only regression command with the pinned dependencies:
python -m unittest discover -s tests -p "test_stage_i_repetition*.py" -v
Do not execute the GPU runner from this review package.
"""
    write(bundle/'README_FIRST.md',readme)
    write(bundle/'CHATGPT_REVIEW_REQUEST.md',"""Independently assess technical v2 repair and actual output quality.
Confirm5 unchanged reuses +19 new attempts,24 unique cases and transparent lifetime
25-call accounting. Check EOS/cap/finished-beam proof, data/checkpoint provenance
and random20/targeted4 metrics separately. Distinguish literal sentence/word rules
from semantic loops, list paraphrases, filler and factual grounding. Review Core
coverage, contrasting opinions, caption/photo scope, missing sections and zeroF.
Teacher targets are comparison material, not independent ground truth. AI findings
are provisional; do not invent human confirmation or formally decide on behalf of
the researcher. Recommend the next review decision without executing experiments.
""")
    for p in (bundle/'README_FIRST.md',bundle/'CHATGPT_REVIEW_REQUEST.md'):included[p.name]=sha(p)
    write(bundle/'SHA256_MANIFEST.json',dict(sorted(included.items())))
    archive=ROOT/'outputs/stage_i/stage_i_repetition_guard_review_v2.zip'
    if archive.exists():raise ValueError('Archive exists; no overwrite')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle))
    write(out/'DELIVERY_RECEIPT.json',{'archive':str(archive.relative_to(ROOT)),'sha256':sha(archive),
        'bytes':archive.stat().st_size,'payload_files':len(included),'payload_manifest_excludes_itself':True,
        'archive_excludes_receipt_to_avoid_circular_hash':True})
    print((out/'DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))


if __name__=='__main__':build()
