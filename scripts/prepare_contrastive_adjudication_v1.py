"""Read-only derivation of locked TRAIN adjudication packets; no model imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
OUT = ROOT / 'outputs/stage_i/led_large_contrastive_adjudication_v1'
ZIP = ROOT / 'outputs/cloud_handoff/LED_LARGE_CONTRASTIVE_ADJUDICATION_V1_REVIEW.zip'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_evidence(positive, evidence):
    markers = {**positive['reviews'], **positive['captions']}
    assert markers[evidence['marker']][evidence['start_in_marker']:evidence['end_in_marker']] == evidence['quote']


def audit():
    manifest = read(PARENT / 'portable_review/SHA256_MANIFEST.json')['files']
    for rel, digest in manifest.items():
        assert sha(PARENT / 'portable_review' / rel) == digest, rel
    form = read(PARENT / 'RESEARCHER_ADJUDICATION_FORM_PENDING.json')
    positives, pairs, checks = [], [], 0
    for bid, item in form['positives'].items():
        path = PARENT / 'positives' / (bid + '.json')
        assert sha(path) == item['sha256'] and item['decision'] == 'PENDING'
        p = read(path)
        assert p['researcher_decision'] == 'PENDING' and not p['researcher_confirmed']
        assert p['corrected_labels'] == p['original_labels'][1:]
        assert p['original_labels'][0] == 0 and p['corrected_labels'][-1] == 2
        assert hashlib.sha256(p['positive_text'].encode()).hexdigest() == p['positive_utf8_sha256']
        for claim in p['claim_audit']:
            for e in claim['evidence']:
                check_evidence(p, e)
                checks += 1
        p['_file_sha256'] = sha(path)
        positives.append(p)
    by_id = {p['business_id']: p for p in positives}
    for i in range(1, 19):
        pid = f'P{i:02d}'
        path = PARENT / 'pairs' / (pid + '.json')
        assert sha(path) == form['pairs'][pid]['sha256']
        p = read(path)
        positive = by_id[p['business_id']]
        assert p['positive_text'] == positive['positive_text']
        assert p['positive_labels'] == positive['corrected_labels']
        assert p['researcher_decision'] == 'PENDING' and not p['accepted_for_training']
        edit = p['minimal_edit']
        assert p['positive_text'][edit['start']:edit['positive_end']] == edit['removed']
        assert p['negative_text'] == p['positive_text'][:edit['start']] + edit['inserted'] + p['positive_text'][edit['positive_end']:]
        for e in p['evidence']:
            check_evidence(positive, e)
        p['_file_sha256'] = sha(path)
        if i <= 16:
            # These strings are warnings only, never rules for accepting a negative.
            p['_wording_flags'] = sorted(set(re.findall(
                r'\b(?:consistently|unanimously|without|agree|all|always|never)\b',
                edit['inserted'], flags=re.I)))
            pairs.append(p)
    return positives, pairs, dict(parent_manifest_pass=len(manifest), positive_evidence_quotes=checks,
                                  all_pair_evidence_quotes=sum(len(read(PARENT/'pairs'/f'P{i:02d}.json')['evidence']) for i in range(1,19)))


def evidence_html(items, bid):
    return ''.join('<li><a href="#src-' + html.escape(bid + '-' + e['marker']) + '">' + html.escape(e['marker']) +
                   '</a> · offsets [' + str(e['start_in_marker']) + ', ' + str(e['end_in_marker']) +
                   ')<blockquote>' + html.escape(e['quote']) + '</blockquote></li>' for e in items)


def decision_html(key, kind):
    checks = (['I reviewed the COMPLETE positive, including every claim and attribution.',
               'I checked review consensus, disagreement, caption support and zero-caption compliance.'] if kind == 'positive' else
              ['I verified the edit against source evidence, not length or cue words alone.',
               'The edit causes a material defect; the positive is defensible and the change is isolated.'])
    return '<div class="decision" data-key="' + key + '"><label>Researcher decision <select><option>PENDING</option><option>ACCEPT</option><option>REJECT</option></select></label>' + ''.join(
        '<label class="check"><input type="checkbox">' + c + '</label>' for c in checks) + '<label>Evidence-based reason / uncertainty<textarea placeholder="Required for ACCEPT or REJECT. No automatic adjudication."></textarea></label></div>'


def render(positives, pairs):
    e = html.escape
    sections = []
    for idx, p in enumerate(positives, 1):
        bid = p['business_id']
        chunks = [f'<section id="case-{e(bid)}"><h2>{idx}. {e(p["name"])}</h2><p>TRAIN / Filtered · {e(bid)} · {len(p["captions"])} captions</p>',
                  '<h3>Complete immutable positive</h3><pre>' + e(p['positive_text']) + '</pre>',
                  '<p>SHA-256 of positive packet: <code>' + p['_file_sha256'] + '</code></p>',
                  '<p class="warn">Existing source-support judgments are model-assisted proposals. They are not independent researcher decisions.</p>']
        if p['warnings']:
            chunks.append('<pre class="warn">' + e(json.dumps(p['warnings'], ensure_ascii=False, indent=2)) + '</pre>')
        chunks.append('<details><summary>Complete claim-to-evidence audit (' + str(len(p['claim_audit'])) + ' claim groups)</summary>')
        for c in p['claim_audit']:
            chunks.append('<article><h4>' + e(c['claim_id']) + ' · ' + e(c['domain']) + '</h4><p>' + e(c['text']) + '</p><ul>' + evidence_html(c['evidence'], bid) + '</ul></article>')
        chunks.append('</details><details><summary>Full frozen model-visible reviews and condition-specific captions</summary>')
        for marker, text in {**p['reviews'], **p['captions']}.items():
            chunks.append('<article id="src-' + e(bid + '-' + marker) + '"><h4>' + e(marker) + '</h4><pre>' + e(text) + '</pre></article>')
        if not p['captions']:
            chunks.append('<p class="warn">ZERO Filtered captions. No visual assertion can be supported by these captions.</p>')
        chunks.append('</details>' + decision_html('positive:' + bid, 'positive'))
        for pair in (x for x in pairs if x['business_id'] == bid):
            edit = pair['minimal_edit']
            before = pair['positive_text']
            after = pair['negative_text']
            delta = len(pair['negative_labels']) - len(pair['positive_labels'])
            chunks.extend(['<article class="pair" id="' + pair['pair_id'] + '"><h3>' + pair['pair_id'] + ' · ' + e(pair['category']) + '</h3>',
                           '<p>Materiality argument (provisional): ' + e(pair['source_reason']) + '</p>',
                           '<p class="warn">Valid target tokens (headers + EOS included): ' + str(len(pair['positive_labels'])) + ' → ' + str(len(pair['negative_labels'])) + f' (Δ {delta:+d}). Absolute wording cues: ' + e(', '.join(pair['_wording_flags']) or 'none flagged') + '. A cue is not an acceptance criterion.</p>',
                           '<div class="columns"><div><h4>Before — immutable positive edit span</h4><pre>' + e(edit['removed'] or '[empty insertion point]') + '</pre></div><div><h4>After — locked negative edit span</h4><pre>' + e(edit['inserted'] or '[deleted]') + '</pre></div></div>',
                           '<p>Python character offsets in positive: [' + str(edit['start']) + ', ' + str(edit['positive_end']) + '). Minimal spans can start/end inside a word; read full context below.</p>',
                           '<details><summary>Full before/after with exact changed spans</summary><h4>Before</h4><pre>' + e(before[:edit['start']]) + '<mark>' + e(edit['removed']) + '</mark>' + e(before[edit['positive_end']:]) + '</pre><h4>After</h4><pre>' + e(after[:edit['start']]) + '<mark>' + e(edit['inserted']) + '</mark>' + e(after[edit['negative_end']:]) + '</pre></details>',
                           '<ul>' + evidence_html(pair['evidence'], bid) + '</ul>', decision_html('pair:' + pair['pair_id'], 'pair'), '</article>'])
        chunks.append('</section>')
        sections.append(''.join(chunks))
    data = dict(experiment='LED_LARGE_CONTRASTIVE_ADJUDICATION_V1',
                parent_form_sha256=sha(PARENT / 'RESEARCHER_ADJUDICATION_FORM_PENDING.json'),
                positives={p['business_id']: dict(sha256=p['_file_sha256']) for p in positives},
                pairs={p['pair_id']: dict(sha256=p['_file_sha256'], business_id=p['business_id'], category=p['category']) for p in pairs})
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    return HTML_HEAD + '<nav>' + ' · '.join('<a href="#case-' + e(p['business_id']) + '">' + e(p['name']) + '</a>' for p in positives) + '</nav>' + ''.join(sections) + '<script type="application/json" id="bindings">' + embedded + '</script><script>' + JAVASCRIPT + '</script></body></html>'


HTML_HEAD = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'none'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Contrastive V1 — Independent researcher adjudication</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1120px;margin:24px auto;padding:0 20px;color:#192c38;background:#f5f7f9}h1{font-size:27px}h2{color:#114b6c}section{background:white;padding:24px;margin:24px 0;border:1px solid #cbd5dc;border-radius:10px}article{padding:12px 0;border-top:1px solid #d5dee4}article.pair{margin-top:24px;border-top:3px solid #217389}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.55 system-ui;background:#f2f5f8;padding:14px;border-radius:6px}code{overflow-wrap:anywhere}blockquote{white-space:pre-wrap;border-left:3px solid #8ca6b5;padding:6px 12px;margin:8px 0}details{border:1px solid #beced8;border-radius:6px;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:600}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}.warn{background:#fff3cc;padding:12px;border-left:4px solid #aa7100}label{display:block;margin:10px 0}select,input,textarea,button{font:inherit}textarea{display:block;width:95%;min-height:70px}input[type=text]{width:90%;max-width:400px}.decision{background:#eaf2f6;padding:14px}.check{font-size:14px}button{padding:10px;margin:6px;border:1px solid #124f69;border-radius:5px;background:#114b6c;color:white;cursor:pointer}#toolbar{position:sticky;top:0;background:#edf4f7;padding:8px;z-index:1;border-bottom:2px solid #8aa7b5}#status{font-weight:bold}nav a{display:inline-block;padding:4px}mark{background:#ffdf91}a{color:#075173}@media(max-width:700px){.columns{grid-template-columns:1fr}#toolbar{position:static}}@media print{#toolbar,.decision{position:static}section{break-before:page}}
</style></head><body><h1>Contrastive Pair Qualification — Researcher adjudication</h1>
<p>Offline review only. Eight complete positives and the sixteen locked P01–P16 candidates. All decisions start PENDING. P17/P18 remain excluded. No replacement, rewriting or extra negatives.</p>
<div class="warn">Review all evidence independently. ACCEPT is never preselected. If any complete positive is unsupported, or fewer than 16 defensible pairs can be accepted, the contrastive branch is NO-GO. Pending review is not acceptance. Even all ACCEPT decisions grant <b>no GPU authorization</b>.</div>
<p>Known shortcut confounds: all 12 Core/balance negatives are shorter; six balance edits contain absolute/consensus cues. Mean token log-probability does not establish immunity to these shortcuts. Do not accept an omission solely because it is shorter: ask whether an important fact is materially lost and not adequately retained elsewhere.</p>
<p>Full pretrained LED/CUDA/FP16 numerical equivalence, AMP behavior, native gradient accumulation and GPU memory headroom remain <b>UNVERIFIED</b>. Original CE epoch 3 stays the current LED-large Filtered checkpoint. Completing primary RO/U/F/Shuffled comparisons has priority.</p>
<div id="toolbar"><span id="status" aria-live="polite"></span><br><button id="draft">Download draft JSON</button><button id="final">Download final adjudication JSON</button><p>No autosave: download a draft before closing this page. No network or execution commands.</p></div>
<label>Independent reviewer name <input id="reviewer" type="text" placeholder="Researcher enters their own name"></label>
<label class="check"><input type="checkbox" id="attest">I personally reviewed all eight complete positives and P01–P16 against the frozen model-visible evidence. My decisions are independent; no Teacher/target rewrite or replacement is requested.</label>
'''

JAVASCRIPT = r'''
'use strict';
const bindings=JSON.parse(document.getElementById('bindings').textContent);
const blocks=Array.from(document.querySelectorAll('.decision'));
function collect(){
 const result={experiment:bindings.experiment,decider_role:'researcher',independently_reviewed:false,
 parent_form_sha256:bindings.parent_form_sha256,positives:{},pairs:{},excluded_pair_ids:['P17','P18'],
 GPU_execution_approved:false,training_authorized:false,numerical_status:'UNVERIFIED',
 reviewer_name:document.getElementById('reviewer').value.trim()||null,reviewed_utc:null};
 for(const b of blocks){
  const [kind,key]=b.dataset.key.split(':');const dest=kind==='positive'?'positives':'pairs';
  result[dest][key]={...bindings[dest][key],decision:b.querySelector('select').value,
  notes:b.querySelector('textarea').value.trim()||null,
  confirmation_checks:Array.from(b.querySelectorAll('input[type=checkbox]')).map(x=>x.checked)};
 }
 return result;
}
function verdict(result,final){
 const positives=Object.values(result.positives),pairs=Object.values(result.pairs);
 if([...positives,...pairs].some(x=>x.decision==='REJECT'))return 'NO_GO_CONTRASTIVE_BRANCH';
 if(final&&pairs.filter(x=>x.decision==='ACCEPT').length<16)return 'NO_GO_CONTRASTIVE_BRANCH';
 if(positives.every(x=>x.decision==='ACCEPT')&&pairs.every(x=>x.decision==='ACCEPT'))return 'ADJUDICATION_COMPLETE_NO_GPU_AUTHORIZATION';
 return 'PENDING_INDEPENDENT_ADJUDICATION';
}
function update(){const r=collect();document.getElementById('status').textContent=
 Object.values(r.positives).filter(x=>x.decision==='ACCEPT').length+'/8 positives ACCEPT · '+
 Object.values(r.pairs).filter(x=>x.decision==='ACCEPT').length+'/16 pairs ACCEPT · '+verdict(r,false);}
function download(final){
 const r=collect();const entries=[...Object.values(r.positives),...Object.values(r.pairs)];
 if(entries.some(x=>x.decision!=='PENDING'&&(!x.notes||(x.decision==='ACCEPT'&&!x.confirmation_checks.every(Boolean))))){
 alert('Every decision requires an evidence-based reason. ACCEPT also requires both manual confirmation checks.');return;}
 if(final&&(!r.reviewer_name||!document.getElementById('attest').checked||entries.some(x=>x.decision==='PENDING'))){
 alert('Final adjudication requires your name, independent-review attestation and all 24 decisions. Otherwise download a draft.');return;}
 r.document_type=final?'RESEARCHER_ADJUDICATION_ONLY':'UNSIGNED_DRAFT';
 r.independently_reviewed=final;r.reviewed_utc=final?new Date().toISOString():null;
 r.exported_utc=new Date().toISOString();r.branch_status=verdict(r,final);
 r.authorization_notice='This is not a diagnostic, gate or training authorization. Fresh explicit diagnostic-only approval is required.';
 const url=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)+'\n'],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='CONTRASTIVE_V1_'+(final?'RESEARCHER_ADJUDICATION':'DRAFT')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
for(const b of blocks)b.addEventListener('input',update);
document.getElementById('draft').addEventListener('click',()=>download(false));
document.getElementById('final').addEventListener('click',()=>download(true));
document.addEventListener('click',event=>{const a=event.target.closest('a[href^="#src-"]');if(!a)return;
 const target=document.getElementById(a.getAttribute('href').slice(1));if(target){let parent=target.parentElement;while(parent){if(parent.tagName==='DETAILS')parent.open=true;parent=parent.parentElement;}}});
update();
'''


def main():
    assert not OUT.exists() and not ZIP.exists(), 'Versioned output exists; preserve it.'
    positives, pairs, checks = audit()
    originals = {str(p.relative_to(ROOT)): sha(p) for p in PARENT.rglob('*') if p.is_file()}
    OUT.mkdir(parents=True)
    for folder in ('positives', 'pairs', 'excluded'):
        (OUT / folder).mkdir()
    for p in positives:
        shutil.copyfile(PARENT/'positives'/(p['business_id']+'.json'), OUT/'positives'/(p['business_id']+'.json'))
    for i in range(1,19):
        shutil.copyfile(PARENT/'pairs'/f'P{i:02d}.json', OUT/('pairs' if i<=16 else 'excluded')/f'P{i:02d}.json')
    (OUT/'ADJUDICATE.html').write_text(render(positives,pairs),encoding='utf-8')
    shutil.copyfile(PARENT/'RESEARCHER_ADJUDICATION_FORM_PENDING.json',OUT/'ORIGINAL_PENDING_FORM_UNCHANGED.json')
    shutil.copyfile(PARENT/'NUMERICAL_EVIDENCE_AND_BLOCKERS.json',OUT/'INHERITED_NUMERICAL_LIMITS_UNCHANGED.json')
    bias = [dict(pair_id=p['pair_id'],category=p['category'],positive_tokens=len(p['positive_labels']),
                 negative_tokens=len(p['negative_labels']),delta=len(p['negative_labels'])-len(p['positive_labels']),
                 inserted_wording_flags=p['_wording_flags'],researcher_decision='PENDING') for p in pairs]
    write(OUT/'SHORTCUT_BIAS_AUDIT.json',dict(method='Exact existing corrected label lengths including headers/EOS; regex cue flags on inserted span, descriptive only.',
        shorter=sum(x['delta']<0 for x in bias),longer=sum(x['delta']>0 for x in bias),equal=sum(x['delta']==0 for x in bias),pairs=bias))
    write(OUT/'INITIAL_ADJUDICATION_PENDING.json',dict(status='PENDING_INDEPENDENT_ADJUDICATION',accepted_pairs=0,
        positive_decisions={p['business_id']:'PENDING' for p in positives},pair_decisions={p['pair_id']:'PENDING' for p in pairs},
        excluded_pairs=['P17','P18'],GPU_execution_approved=False,training_authorized=False,
        full_LED_CUDA_FP16_equivalence='UNVERIFIED',AMP='UNVERIFIED',native_CUDA_accumulation='UNVERIFIED',memory_headroom='UNVERIFIED'))
    (OUT/'README_FIRST_VI.md').write_text('''# Duyệt độc lập Contrastive V1

Giải nén ZIP và mở **ADJUDICATE.html** bằng Chrome/Edge. Không cần server, mạng hay GPU.
Duyệt 8 positive đầy đủ trước khi chấp nhận các cặp liên quan; mở phần claim/evidence và nguồn gốc khi cần.
Các liên kết REV/IMG mở đúng đoạn nguồn, trích dẫn giữ nguyên offset và văn bản.
P01–P16 khóa nguyên trạng. P17/P18 chỉ được lưu ở excluded/ để bảo toàn lịch sử, không được thay thế cặp bị loại.

Chọn ACCEPT/REJECT chỉ sau khi tự kiểm tra; mọi mục ban đầu PENDING. Ghi lý do và kiểm tra materiality,
đồng thuận/phản biện, giới hạn caption và nguồn. Quyết định cũ của model chỉ là đề xuất.
Không chấp nhận cặp chỉ vì negative ngắn hơn, hay chứa từ tuyệt đối. Core bị bỏ có thể còn hiện ở câu khác:
người duyệt phải xác nhận mất thông tin quan trọng thực sự, không chỉ rút gọn hoặc loại chi tiết dư thừa.

**Download draft JSON** lưu nháp trước khi đóng (không autosave, không tự nhập lại nháp).
**Download final adjudication JSON** yêu cầu đủ 24 quyết định, tên người duyệt và xác nhận độc lập.
JSON được tải riêng, không sửa artifact gốc. Trang không tự chấp nhận, chạy code hay cấp quyền GPU.
Nếu đóng trang, mở lại sẽ về PENDING; giữ JSON đã tải để đối chiếu, tránh mất ghi chú.

Nếu bất kỳ positive đầy đủ nào không có hỗ trợ, hoặc sau duyệt ít hơn 16 cặp bảo vệ được: NO-GO,
dừng nhánh contrastive; không thay positive, không viết lại target, không tạo negative bù.
Chưa duyệt xong là PENDING, không phải GO. Đủ 8+16 ACCEPT cũng chỉ hoàn thành adjudication:
vẫn cần phê duyệt diagnostic-only mới; không tự gate/train.

Full pretrained LED/CUDA/FP16 equivalence, AMP, native GPU accumulation và memory headroom: UNVERIFIED.
Giữ Original CE epoch3 làm checkpoint Filtered hiện hành. Ưu tiên hoàn tất RO/U/F/Shuffled chính.
Không inference, training, Teacher generation, validation/test access hoặc optimizer update trong gói này.
''',encoding='utf-8')
    (OUT/'RESEARCHER_REVIEW_REPORT.md').write_text('''# Locked adjudication scope

This package presents existing data, not new supervision or authorization. The original 58-entry portable
manifest was reverified. All eight full positives, sixteen locked candidates and two excluded controls
are copied byte-for-byte. All original artifacts and execution bindings remain unchanged.

## Decision rules

For each COMPLETE positive, assess every claim, attribution, consensus frequency, disagreement and
condition-specific visual statement using only model-visible frozen TRAIN reviews/captions. Evidence
quotes establish textual matches, not automatic semantic support. Source packets include full context.
For each pair, review the exact edited span, full before/after text and cited source excerpts. Decide
whether the negative creates a substantive error or omission and whether the positive is defensible.
Existing model-assisted support and salience arguments are not researcher-confirmed labels.

The fixed candidate inventory has six Core omissions, six disagreement-loss edits and four grounding edits.
All eight businesses have one to three locked candidates. Exactly sixteen defensible accepted pairs are
required, including at least four per category; no quota changes, replacements or new examples are allowed.
P17/P18 remain excluded regardless of their stored historical PENDING fields. Any unsupported positive or
failure to accept all sixteen means NO-GO. No evidence of that failure is inferred merely from pending review.

## Shortcut risks

Twelve negatives are shorter, three longer and one equal in corrected target token length. All twelve
Core/disagreement edits are shorter. All six disagreement negatives use consensus/absolutist cue wording
(e.g. consistently, unanimously, agree, without). Thus length, edit category and rhetorical wording are
confounded. Mean token log-probability normalization does not eliminate a learned shortcut or establish
factuality. P07/P12 minimal edit spans begin/end inside words; full before/after context is shown to avoid
misreading the minimal diff. Cue flags use a transparent regex and are not a comprehensive lexical audit.
Do not repair these biases by silently rewriting or replacing locked pairs. If they undermine defensibility,
record the concern and reject/stop. This 16-pair training-exposed convenience set cannot establish generalization.

## Stop boundary and baseline

No GPU diagnostic, gate, training, decoding or extra data generation is authorized. All full pretrained
LED/CUDA/FP16 numerical-equivalence, AMP, native GPU accumulation and memory-headroom claims remain
UNVERIFIED. Existing reduced-width CPU tests are inherited evidence only; no training tests were rerun here.
The interface exports reviewer-entered decisions separately; even all ACCEPT decisions cannot authorize
GPU execution. Keep Original CE epoch3 as current LED-large F; prioritize primary RO/U/F/Shuffled comparisons.
''',encoding='utf-8')
    checks.update(positives=8,locked_pairs=16,excluded_controls=2,accepted_pairs=0,all_decisions='PENDING',
                  GPU_operations=0,optimizer_updates=0,new_negatives=0,original_files_verified_unchanged=len(originals),
                  interface_test_status='Run separate interface regression tests; see CPU_INTERFACE_TESTS.txt')
    for rel,digest in originals.items():
        assert sha(ROOT/rel)==digest, rel
    write(OUT/'PRESERVATION_AND_EVIDENCE_CHECKS.json',checks)
    write(OUT/'PARENT_BINDINGS_UNCHANGED.json',dict(files=originals))
    shutil.copyfile(Path(__file__),OUT/'prepare_contrastive_adjudication_v1.py')
    # Packaging is performed only after focused UI tests, by --package.
    print(json.dumps(dict(output=str(OUT),checks=checks)))


def package():
    assert OUT.is_dir() and not ZIP.exists()
    assert (OUT/'CPU_INTERFACE_TESTS.txt').is_file()
    files={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    write(OUT/'SHA256_MANIFEST.json',dict(files=files))
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for name,digest in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    ZIP.with_suffix(ZIP.suffix+'.sha256').write_text(sha(ZIP)+'  '+ZIP.name+'\n',encoding='utf-8')
    print(json.dumps(dict(zip=str(ZIP),sha256=sha(ZIP),bytes=ZIP.stat().st_size,manifest_entries=len(files))))


if __name__=='__main__':
    import sys
    if '--package' in sys.argv:package()
    else:main()
