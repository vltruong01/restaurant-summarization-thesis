"""Offline 12-case researcher worksheets from the existing, unchanged 32-case pilot."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_result_review_v1'
PREP = ROOT / 'outputs/stage_i/teacher_target_redesign_pilot_v1/bundle'
REQUEST = Path('C:/Users/VUONGLOCTRUONG/.codex/attachments/113fbd87-717d-4b1b-82d2-2aeeff05e1f2/Pasted text.txt')
OUT = ROOT / 'outputs/stage_i/teacher_redesign_adjudication12_v1'
CONDITIONS = ('reviews_only', 'unfiltered', 'filtered')
PRIORITY = [
    ("P.F. Chang's", 'potential_loss_or_salience', 'Disagreement about Mongolian beef.', ['mongolian']),
    ('Teriyaki Madness', 'potential_loss_or_salience', 'Conflicting opinions on vegetable quality.', ['vegetab', 'cabbage']),
    ('Top Out Cafe', 'potential_loss_or_salience', 'Omission of repeatedly praised sticky wings.', ['sticky', 'wings']),
    ('Torch-Wood Cafe', 'potential_loss_or_salience', 'Recurring smoothie praise versus isolated blender incident.', ['smooth', 'blender', 'blade']),
    ('Calypso Queen Tropical Cruise', 'potential_loss_or_salience', 'Cruise/buffet/value information versus unverified counterfeit allegation.', ['counterfeit', 'buffet', 'cruise', 'value']),
    ('Backyard Bowls', 'potential_loss_or_salience', 'Repeated requests for later opening hours.', ['5pm', '5 pm', 'hours', 'later']),
    ("Domino's Pizza", 'potential_loss_or_salience', 'Recurring online-ordering problems versus isolated allegations.', ['online', 'app', 'alleg', 'rodent', 'tip']),
    ("Marsilio's Kitchen", 'potential_loss_or_salience', 'Possible overemphasis of a secondhand food-poisoning report.', ['poison', 'sick']),
    ('MOD Pizza', 'potential_benefit', 'Gluten-free option versus cross-contact limitations.', ['gluten', 'dairy', 'celiac']),
    ("Rubio's", 'potential_benefit', 'Vegan and allergy-related experiences.', ['vegan', 'allerg', 'avocado', 'cheese']),
    ('California Cafe', 'potential_benefit', 'Ingredient/allergen concern and historical-status handling.', ['wheat', 'allerg', 'closed']),
    ('Mystic Fish Seafood Grill & Bar', 'potential_benefit', 'Distinctive desserts and peak-hour reservation difficulties.', ['dessert', 'reserv']),
]
RUBRIC = {
    'A': 'Important Core information preservation',
    'B': 'Recurring and distinctive business-specific opinions',
    'C': 'Opinion balance, including disagreement on the same dish/aspect',
    'D': 'Source-grounded factual correctness',
    'E': 'Treatment of isolated serious incidents, allegations, hearsay and uncertainty',
    'F': 'Caption-supported visual information and usefulness',
    'G': 'Semantic nonredundancy, clarity and information density',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def th(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)


def esc(value):
    return html.escape(str(value), quote=True)


STYLE = '''<style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:24px auto;padding:0 18px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.target{padding:15px;background:#f3f5f7}td,th{border:1px solid #bbb;padding:8px;vertical-align:top}table{border-collapse:collapse;width:100%}
figure{display:inline-block;margin:10px;vertical-align:top}img{max-width:230px;max-height:180px}details{margin:14px 0}
.pending{border-left:5px solid #947500;padding:12px;background:#fff8da}article{border-top:2px solid #aaa;padding:16px 0}
@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>'''


def shell(title):
    return '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title>' + STYLE


class CheckHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.payloads = [], [], {}
        self.key = None

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'a':
            self.links.append(a.get('href', ''))
        if tag == 'img':
            self.links.append(a.get('src', ''))
        if tag == 'pre' and 'data-exact' in a:
            self.key = a['data-exact']
            self.payloads[self.key] = ''

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.key = None

    def handle_data(self, data):
        if self.key is not None:
            self.payloads[self.key] += data


def main():
    assert not OUT.exists(), 'Never overwrite an existing adjudication package'
    src_path = PREP / 'SOURCE_EVIDENCE.jsonl'
    a_path = AUDIT / 'stage_b_verified/locked_inputs/proposed_targets_STAGE_A_UNCHANGED.jsonl'
    old_path = AUDIT / 'stage_b_verified/locked_inputs/ORIGINAL_TARGETS_UNCHANGED.jsonl'
    b_path = AUDIT / 'stage_b_verified/BUSINESS_COMPARISONS_32.json'
    count_path = AUDIT / 'LOCAL_EXACT_LED_TOKEN_COUNTS_96.json'
    input_paths = [src_path, a_path, old_path, b_path, count_path, REQUEST,
                   PREP / 'PILOT_SAMPLE_MANIFEST.json', AUDIT / 'stage_b_verified/STAGE_B_REPORT_VI.md']
    original_hashes = {str(p): sha(p) for p in input_paths}
    sources, a_rows, old_rows, b_rows = jl(src_path), jl(a_path), jl(old_path), read(b_path)
    source = {r['business_id']: r for r in sources}
    a, old, b = [{r['business_id']: r for r in rows} for rows in (a_rows, old_rows, b_rows)]
    assert len(sources) == len(a_rows) == len(old_rows) == len(b_rows) == 32
    assert set(source) == set(a) == set(old) == set(b)
    assert all(r['split'] == 'train' for r in sources)
    counts = {(r['business_id'], r['condition']): r for r in read(count_path)}
    assert len(counts) == 96
    names = {r['business_name']: r['business_id'] for r in b_rows}
    assert len(names) == 32 and all(name in names for name, *_ in PRIORITY)
    selected = [names[name] for name, *_ in PRIORITY]
    assert len(set(selected)) == 12
    OUT.mkdir(parents=True)
    bundle = OUT / 'bundle'
    bundle.mkdir()
    # Preserve the 32-case pilot, including the other20; no resampling or output edits.
    preserved = {
        src_path: 'PRESERVED_PILOT_32/SOURCE_EVIDENCE.jsonl',
        a_path: 'PRESERVED_PILOT_32/proposed_targets_STAGE_A_UNCHANGED.jsonl',
        old_path: 'PRESERVED_PILOT_32/ORIGINAL_TARGETS_UNCHANGED.jsonl',
        b_path: 'PROVENANCE/STAGE_B_ASSISTANT_OBSERVATIONS_32_UNCHANGED.json',
        count_path: 'PRESERVED_PILOT_32/EXACT_LOCAL_TOKEN_COUNTS_96.json',
        PREP / 'PILOT_SAMPLE_MANIFEST.json': 'PRESERVED_PILOT_32/PILOT_SAMPLE_MANIFEST.json',
        REQUEST: 'PROVENANCE/RESEARCHER_REVIEW_DECISION_AND_REQUEST.txt',
        AUDIT / 'stage_b_verified/STAGE_B_REPORT_VI.md': 'PROVENANCE/STAGE_B_FIRST_PASS_REPORT_UNCHANGED.md',
    }
    for p, rel in preserved.items():
        cp(p, bundle / rel)
    case_data, annotations, expected_html, warning_quotes = [], [], {}, []
    copied_photos = {}
    for index, (name, group, focus, keywords) in enumerate(PRIORITY, 1):
        bid, cid = names[name], f'CASE_{index:02d}'
        s, prior = source[bid], b[bid]
        evidence = {r['evidence_id']: r for r in s['reviews']}
        evidence.update({c['evidence_id']: c for caps in s['captions'].values() for c in caps})
        candidates, relevant = [], set()
        for n, fact in enumerate(prior['facts'], 1):
            assert all(e in evidence for e in fact['evidence_ids'])
            relevant.update(e for e in fact['evidence_ids'] if '::REV_' in e)
            matches = {}
            for side, target_text in [('old', old[bid]['targets']['reviews_only']), ('new', a[bid]['targets']['reviews_only']['text'])]:
                value = fact[side + '_quote']
                matches[side] = value is None or value in target_text
                if not matches[side]:
                    warning_quotes.append({'case_id': cid, 'candidate': n, 'side': side, 'quote': value})
            candidates.append(dict(candidate_id=f'{cid}_INFO_{n:02d}', candidate_text= fact['fact'],
                ai_suggested_importance= fact['importance'], origin='STAGE_B_UNBLINDED_SELF_ASSESSMENT',
                researcher_confirmed=False, evidence_ids=fact['evidence_ids'],
                original_target_quote=fact['old_quote'], stage_a_target_quote=fact['new_quote'],
                target_quote_literal_checks=matches,
                prior_AI_coverage_not_adjudication={'original': fact['old_coverage'], 'stage_a': fact['new_coverage']}))
        # Source-only keyword routing for the requested concern; it does not assign importance.
        relevant.update(r['evidence_id'] for r in s['reviews'] if any(k in r['text'].lower() for k in keywords))
        exact_targets = {}
        for condition in CONDITIONS:
            measurement = counts[bid, condition]
            text_old, text_a = old[bid]['targets'][condition], a[bid]['targets'][condition]['text']
            assert measurement['old']['target_utf8_sha256'] == th(text_old)
            assert measurement['new']['target_utf8_sha256'] == th(text_a)
            frozen = s['exact_frozen_source_inputs'][condition]
            assert sha(PREP / frozen['file']) == frozen['utf8_sha256']
            cp(PREP / frozen['file'], bundle / frozen['file'])
            exact_targets[condition] = dict(original=text_old, stage_a=text_a,
                original_sha256=th(text_old), stage_a_sha256=th(text_a), measured_tokens=measurement)
            expected_html[cid + ':original:' + condition] = text_old
            expected_html[cid + ':stage_a:' + condition] = text_a
        for condition, caps in s['captions'].items():
            for cap in caps:
                pid = cap['photo_id']
                if cap['photo_available']:
                    src = PREP / cap['photo_file']
                    assert sha(src) == cap['original_photo_sha256']
                    if pid not in copied_photos:
                        cp(src, bundle / cap['photo_file'])
                        copied_photos[pid] = cap['original_photo_sha256']
        record = dict(case_id=cid, business_id=bid, business_name=name,
            original_pilot_index=prior['index'], original_selection_category=s['selection_category'],
            adjudication_priority_group=group, researcher_requested_focus=focus,
            full_source_record=s, exact_original_and_A_targets=exact_targets,
            relevant_review_ids=[r['evidence_id'] for r in s['reviews'] if r['evidence_id'] in relevant],
            information_candidates_UNCONFIRMED=candidates,
            AI_tradeoff_observations_UNCONFIRMED={k:prior[k] for k in ('balance','grounding','redundancy','conciseness','old_better','new_better','u','f')},
            observation_limits='All prior AI judgments are provisional; source quotes and complete targets are exact; no new researcher preference or Core label assigned.')
        case_data.append(record)
        annotation = dict(case_id=cid, business_id=bid, business_name=name, researcher_id=None,
            adjudication_status='UNREVIEWED', researcher_confirmed=False, confirmed_at=None,
            rubric_A_G={k:dict(criterion=v, original_rating_0_to_3=None, stage_a_rating_0_to_3=None,
                evidence_ids=[], researcher_notes=None) for k,v in RUBRIC.items()},
            information_candidates=[dict(candidate_id=c['candidate_id'], researcher_importance=None,
                original_coverage=None, stage_a_coverage=None, evidence_ids=c['evidence_ids'], notes=None,
                researcher_confirmed=False) for c in candidates],
            allegation_and_salience_decisions=[], additional_missing_information=[],
            condition_preferences=dict.fromkeys(CONDITIONS), overall_preference=None,
            preference_rationale=None, unresolved_uncertainty=None)
        annotations.append(annotation)
        expected_html.update({cid + ':review:' + r['evidence_id']:r['text'] for r in s['reviews']})
        expected_html.update({cid + ':caption:' + c['evidence_id']:c['model_visible_caption'] for caps in s['captions'].values() for c in caps})
        pieces = [shell(name), '<p><a href="../REVIEW.html">← 12 cases</a> · <a href="../RESEARCHER_ADJUDICATION_TABLE.json">Blank adjudication table</a></p>',
            f'<h1>{cid}: {esc(name)}</h1><p>TRAIN · original pilot case {prior["index"]} · {esc(s["selection_category"])} · {esc(bid)}</p>',
            '<p class="pending"><b>Researcher adjudication: PENDING.</b> All Core/Supporting candidates and AI observations below remain unconfirmed. No preference or outcome selected.</p>',
            f'<h2>Requested focus</h2><p>{esc(focus)}</p>',
            '<p>Read the complete source. Serious one-off reports can be relevant with attribution; neither severity nor brevity alone determines salience. Ratings and preference remain blank.</p>']
        for condition, pair in exact_targets.items():
            pieces.append(f'<h2>{esc(condition)}</h2><div class="pair">')
            for side in ('original','stage_a'):
                pieces.append(f'<section class="target"><h3>{esc(side.upper())}</h3><pre data-exact="{cid}:{side}:{condition}">{esc(pair[side])}</pre></section>')
            pieces.append('</div><table><tr><th>Exact pinned-tokenizer count</th><th>ORIGINAL</th><th>STAGE A</th></tr>')
            for key in ('original_tokenizer_BOS_content_EOS_tokens','corrected_training_content_EOS_label_tokens',
                        'review_body_standalone_no_special','visual_body_standalone_no_special'):
                m = pair['measured_tokens']
                pieces.append(f'<tr><td>{key}</td><td>{m["old"][key]}</td><td>{m["new"][key]}</td></tr>')
            pieces.append('</table>')
        pieces.append('<h2>Candidate information — not approved Core labels</h2><table><tr><th>Candidate / AI role</th><th>Exact target snippets</th><th>Direct review evidence</th></tr>')
        for c in candidates:
            anchors = '<br>'.join(f'<a href="#{esc(e)}">{esc(e)}</a>' for e in c['evidence_ids'])
            snippets = '<b>ORIGINAL:</b> ' + esc(c['original_target_quote']) + '<br><b>STAGE A:</b> ' + esc(c['stage_a_target_quote'])
            pieces.append(f'<tr id="{c["candidate_id"]}"><td>{esc(c["candidate_text"])}<br>AI: {esc(c["ai_suggested_importance"])}; researcher: UNCONFIRMED</td><td>{snippets}</td><td>{anchors}</td></tr>')
        pieces.append('</table><h2>Source reviews (complete original bodies)</h2><p>Priority routing is by cited evidence / requested-topic keywords, not an importance decision. All other selected reviews remain below.</p>')
        for priority in (True,False):
            pieces.append('<h3>' + ('Relevant to current adjudication / prior evidence' if priority else 'Other selected reviews — retained in full') + '</h3>')
            for r in s['reviews']:
                if (r['evidence_id'] in relevant) != priority:
                    continue
                e = r['evidence_id'];meta=r['included_metadata']
                pieces.append(f'<article id="{esc(e)}"><b>{esc(e)}</b> · review_id={esc(r["review_id"])} · stars={meta["stars"]} · date={esc(meta["date"])}<pre data-exact="{cid}:review:{esc(e)}">{esc(r["text"])}</pre></article>')
        pieces.append('<h2>Exact condition-visible captions and original photos</h2><p>Photos verify captions only. They cannot supply a new target claim absent from that condition. Do not join subjects/actions from different photos.</p>')
        for condition, caps in s['captions'].items():
            pieces.append(f'<h3>{condition}: {len(caps)} captions</h3>')
            if not caps:
                pieces.append('<p><b>ZERO CAPTIONS: this visual target must remain empty.</b></p>')
            for c in caps:
                e=c['evidence_id']
                pieces.append(f'<article id="{esc(e)}"><b>{esc(e)}</b> · photo_id={esc(c["photo_id"])}<pre data-exact="{cid}:caption:{esc(e)}">{esc(c["model_visible_caption"])}</pre>')
                if c['photo_available']:
                    pieces.append(f'<a href="../{esc(c["photo_file"])}"><img loading="lazy" src="../{esc(c["photo_file"])}" alt="Original {esc(c["photo_id"])}"></a>')
                else:
                    pieces.append('<p>Original photo unavailable; no substitute.</p>')
                pieces.append('</article>')
        pieces.append('<details><summary>Prior AI tradeoffs / visual concerns — provisional; inspect source independently</summary><pre>' + esc(json.dumps(record['AI_tradeoff_observations_UNCONFIRMED'], ensure_ascii=False, indent=2)) + '</pre></details>')
        pieces.append('<h2>Researcher worksheet</h2><table><tr><th>Criterion</th><th>ORIGINAL 0–3 / evidence</th><th>STAGE A 0–3 / evidence</th></tr>')
        for k,v in RUBRIC.items():
            pieces.append(f'<tr><td>{k}. {esc(v)}</td><td>UNREVIEWED</td><td>UNREVIEWED</td></tr>')
        pieces.append('</table><p>H. Overall preference: ORIGINAL / STAGE A / TIE / INCONCLUSIVE — <b>UNSELECTED</b>.</p><p>Record answers in the JSON table or the Markdown worksheet. No automatic scoring or scientific decision.</p></html>')
        page=bundle/f'cases/{cid}.html';page.parent.mkdir(exist_ok=True);page.write_text('\n'.join(pieces),encoding='utf-8')
        save(bundle / f'case_data/{cid}.json',record)
    save(bundle/'RESEARCHER_ADJUDICATION_TABLE.json',dict(status='PENDING_RESEARCHER_ADJUDICATION',
        preferences_allowed=['ORIGINAL','STAGE_A','TIE','INCONCLUSIVE'], optional_ratings=[0,1,2,3],
        no_automatic_weighted_outcome=True, cases=annotations))
    save(bundle/'PRIORITY_CASE_MANIFEST.json',dict(selected12=[dict(case_id=c['case_id'],business_id=c['business_id'],
        business_name=c['business_name'],original_pilot_index=c['original_pilot_index'],priority_group=c['adjudication_priority_group']) for c in case_data],
        other20_preserved_ids=[r['business_id'] for r in sources if r['business_id'] not in selected],pilot_membership_changed=False))
    rubric='''# Researcher adjudication rubric

Các trường trong RESEARCHER_ADJUDICATION_TABLE.json ban đầu đều chưa xác nhận.
Ghi researcher_id, dẫn evidence IDs và exact target/source spans trước khi quyết định.
A: bảo toàn Core quan trọng. B: ý kiến lặp lại/đặc trưng riêng doanh nghiệp.
C: cân bằng ý kiến, nhất là khen/chê cùng món hoặc cùng khía cạnh dịch vụ.
D: factual grounding theo nguồn. E: xử lý sự cố đơn lẻ, cáo buộc, hearsay và bất định.
F: visual hữu ích, đúng caption của điều kiện. G: ít lặp ý, rõ, mật độ thông tin.
H: ORIGINAL / STAGE A / TIE / INCONCLUSIVE, kèm lý do và vấn đề chưa phân định.

Nếu dùng điểm định hướng 0–3 (không bắt buộc):0 sai/thiếu nghiêm trọng;1 cần sửa lớn;
2 phần lớn hữu ích nhưng còn vấn đề;3 đáp ứng tốt tiêu chí. Không tự cộng thành điểm
GO. Không thưởng độ ngắn hoặc phạt chỉ vì ngoài guidance token. Không điền score
chỉ dựa vào số fact trong panel AI. Reviewer có thể thêm fact còn thiếu trong panel.

Mỗi candidate: researcher tự phân loại Core/Supporting/Exclude/Uncertain và ghi
coverage full/partial/missing/contradicted/uncertain cho hai target. Đây là các nhãn
mới cần người xác nhận, không kế thừa trạng thái duyệt từ D053 validation.
Một câu khen food chung không thay thế được ý quan trọng về món hoặc sự cố cụ thể.

Không coi mọi one-off là consensus. Cần phân biệt firsthand report, secondhand
hearsay, suy đoán và pattern có nhiều nguồn độc lập. Sự cố nghiêm trọng đơn lẻ vẫn
có thể đáng giữ nếu liên quan, có căn cứ và quy nguồn đúng; không mặc định xóa nó.
Attribution không tự giải quyết mức độ nổi bật hoặc tính phù hợp trong consensus.
Không suy tình trạng lịch sử thành hiện trạng hiện nay, hoặc trải nghiệm allergy
thành chứng nhận an toàn. Không thêm kiến thức web hay validation/test evidence.

Visual: chỉ dùng caption đúng U/F và photo mapping. Ảnh dùng để kiểm tra caption,
không thêm claims vắng khỏi caption. Không dựng quan hệ xuyên ảnh. Zero F phải có
visual body rỗng; đánh giá riêng chất lượng review để không nhầm rỗng đúng là Core tốt.

Giữ 32 doanh nghiệp là đơn vị pilot, RO/U/F chia sẻ canonical nên không phải96mẫu
độc lập. 12 ca được chọn có chủ đích cho adjudication, không đại diện ngẫu nhiên
cho cả32 hay2000train. Nhận xét AI và panel hậu nghiệm vẫn provisional cho đến
khi researcher duyệt; không tự chọn outcome hoặc cho phép training.
'''
    (bundle/'ADJUDICATION_RUBRIC.md').write_text(rubric,encoding='utf-8')
    worksheet='# Phiếu researcher — chưa xác nhận\n\n|Case|Business|A|B|C|D|E|F|G|H preference|Confirmed|\n|---|---|---|---|---|---|---|---|---|---|---|\n'
    for c in case_data:
        worksheet+=f'|{c["case_id"]}|{c["business_name"]}| | | | | | | | |false|\n'
    worksheet+='\nGhi evidence IDs, lý do, nhãn candidate và bất định vào JSON. Không có ô nào được tự chấm.\n'
    (bundle/'RESEARCHER_ADJUDICATION_TABLE.md').write_text(worksheet,encoding='utf-8')
    summary='''# Patterns across the original32 — provisional evidence observations

This summary preserves the original32-case membership. It is not a new full
independent semantic audit, approved Core labeling, preference vote or outcome.
The researcher's independent-review conclusion/12priorities is provided verbatim.
Its separate detailed32-case report was not supplied in this session; do not
claim it is included or invent additional independent adjudications.
Per-business observations below reuse the earlier, explicitly unconfirmed Stage B
self-assessment; they are prompts for source checking, not researcher decisions.

Recurring risks worth adjudicating: flattening same-dish disagreements (P.F.Chang's,
Teriyaki); omitting recurring signatures or use constraints (Top Out sticky wings,
Torch-Wood smoothie praise, Backyard hours, Domino online-ordering); assigning
scarce target space to isolated allegations/hearsay (Calypso, Domino, Marsilio).
Some A targets also drop positive value counterparts while retaining price criticism.

Potential source-supported gains to check: MOD retains dietary options plus a
reported facility limitation; Rubio's retains attributed dietary/allergy experiences;
California retains the wheat concern and attributes historical closure; Mystic
adds dessert/reservation specifics. These do not imply current verified policy,
allergy safety, or an overall win. Prior B also notes supporting information A loses.
Visual concision sometimes removes repetitive fragments but may also remove a
distinctive food/activity or select multiple generic scenes. More compact is not
automatically more useful. Inspect U/F independently and preserve zero-caption rules.

Measured descriptive lengths (content/EOS labels incl.headers): old->A means
RO227.09->115.13,U320.00->147.03,F294.59->145.25. A review body mean104.125,
25/32 below flexible110token guidance; this is a review cue, never an automatic
failure or instruction to pad. These numbers are not quality scores or training results.

## All32 preserved observations, not researcher-confirmed judgments

'''
    for r in b_rows:
        summary+=f'### {r["index"]:02d}. {r["business_name"]} — {r["business_id"]}\n\n'
        summary+='Potential loss / original advantage (prior AI): '+r['old_better']+'\n\n'
        summary+='Potential addition / A advantage (prior AI): '+r['new_better']+'\n\n'
        ids=list(dict.fromkeys(e for f in r['facts'] for e in f['evidence_ids']))
        summary+='Source evidence IDs: '+', '.join(ids)+'\n\n'
    (bundle/'SUMMARY_ORIGINAL_32_PROVISIONAL.md').write_text(summary,encoding='utf-8')
    memo='''# Decision memo — three available researcher outcomes; none selected

**Current state: researcher adjudication PENDING. Target replacement/training NOT approved.**

1. KEEP ORIGINAL targets. Appropriate if substantive information retention,
   disagreement synthesis or salience does not improve reliably enough, or remains
   uncertain. Preserve A as pilot evidence; no generation expansion or training follows.
2. REDESIGN TARGET POLICY before larger generation. Appropriate if adjudication
   identifies a systematic correctable supervision-policy issue (e.g. same-aspect
   contrast, recurring specificity, allegation salience, caption selection). First
   document the researcher-approved policy amendment; no target rewrite, repair,
   mass regeneration or new pilot is authorized by this package.
3. CONDUCT A SMALL CONTROLLED TRAINING EXPERIMENT only if adjudicated evidence
   supports it. This needs a separate concrete protocol and explicit approval,
   preserving matched initialization/architecture/data/optimization/checkpoint rules
   and the corrected label policy. Account for sample exposure and target-policy
   changes; do not assume target brevity predicts student quality. No GPU or new
   model experiment starts automatically, and this memo does not choose settings.

Adjudicate substantive tradeoffs first; ties/inconclusive are valid. Do not select
any outcome from scores, token reduction, labels copied from AI, or an automatic vote.
Record reviewer, evidence, unresolved issues and the researcher's explicit decision.
Keep all original target/split/input/checkpoint/history artifacts unchanged. The
stated research priority remains a defensible matched large RO/U/F/Shuffled
comparison; Shuffled would reuse matched F. No completed matched large family or
authorization is implied, and prior experimental decisions/deadlines are unchanged.
'''
    (bundle/'DECISION_MEMO_NO_OUTCOME_SELECTED.md').write_text(memo,encoding='utf-8')
    readme='''# 12-case researcher adjudication — offline preparation only

Open REVIEW.html, then the12 individual case sheets. Each has exact full ORIGINAL
and STAGE A RO/U/F targets, previously verified LED token counts, unshortened source
reviews, exact condition-visible captions, original photos and evidence anchors.
Relevant reviews are routed first; every selected review is retained in full.
Read ADJUDICATION_RUBRIC.md and complete RESEARCHER_ADJUDICATION_TABLE.json or .md.
All ratings, importance decisions, preferences and confirmations are initially blank.
Existing AI candidate importance/coverage/observations are clearly unconfirmed.
Expand prior AI tradeoffs only as needed; independently inspect the direct evidence.

Cases1–8 concern possible loss/salience; cases9–12 concern potential gains, exactly
as requested. No business was added or removed from the original32pilot. All32
sources, A drafts, original targets, sample membership and exact local counts are
preserved byte-for-byte under PRESERVED_PILOT_32/. Other20 have no new case sheet;
their source evidence and prior provisional observations remain available.
Only the12 priority cases'69 photos and36 exact frozen input text views are copied
into this compact package. The other20 photo/input attachments remain unchanged
in the complete original32pilot archive; their references in the preserved records
point to that original package, rather than new photo substitutes.

The independent-review decision/request supplied in chat is preserved verbatim
under PROVENANCE/. No separate detailed independent32-case review file was supplied.
Prior Stage B self-review is also preserved and explicitly distinguishable from
the researcher's independent-review declaration and future adjudication decisions.
No claim is made that new Core labels, ratings or outcomes have been approved.

Read SUMMARY_ORIGINAL_32_PROVISIONAL.md and DECISION_MEMO_NO_OUTCOME_SELECTED.md.
The memo presents KEEP ORIGINAL / REDESIGN POLICY / SMALL CONTROLLED TRAINING
as options only. Neither replacement nor training is authorized. No new targets,
Teacher calls, inference, training, held-out/validation content or decoding search.
Only the32training pilot source/target files were used; no D053 cases were loaded.
The original damaged A ZIP remains unchanged; its intact32drafts already match the
locked A copy used here. Original checkpoints and historical experiments are untouched.

SHA256_MANIFEST.json binds all files. VALIDATION_REPORT.json records text, token,
photo/evidence-link checks and empty researcher annotations. Stop for researcher review.
'''
    (bundle/'README_FIRST.md').write_text(readme,encoding='utf-8')
    index=[shell('Researcher adjudication: 12 training cases'),'<h1>12-case researcher adjudication</h1><p class="pending">PENDING: no outcome, preference, Core label or score assigned.</p>',
        '<p><a href="README_FIRST.md">README</a> · <a href="ADJUDICATION_RUBRIC.md">Rubric</a> · <a href="RESEARCHER_ADJUDICATION_TABLE.json">Blank JSON table</a> · <a href="RESEARCHER_ADJUDICATION_TABLE.md">Blank Markdown worksheet</a></p>',
        '<p>All32 pilot cases remain preserved. The12 cases are purposive adjudication priorities, not a new random quality sample.</p><ol>']
    for c in case_data:
        index.append(f'<li><a href="cases/{c["case_id"]}.html">{esc(c["business_name"])}</a> — {esc(c["researcher_requested_focus"])}</li>')
    index.append('</ol><p><a href="SUMMARY_ORIGINAL_32_PROVISIONAL.md">Provisional patterns across32</a> · <a href="DECISION_MEMO_NO_OUTCOME_SELECTED.md">Three outcomes — none selected</a></p></html>')
    (bundle/'REVIEW.html').write_text('\n'.join(index),encoding='utf-8')
    # Focused export validation: exact visible text, counts, complete reviews/captions, blank decisions.
    all_payloads, links_checked = {}, 0
    for page in [bundle/'REVIEW.html']+sorted((bundle/'cases').glob('*.html')):
        parsed=CheckHTML();parsed.feed(page.read_text(encoding='utf-8'))
        assert len(parsed.ids)==len(set(parsed.ids)), ('Duplicate HTML anchors',page)
        all_payloads.update(parsed.payloads)
        for href in parsed.links:
            if href.startswith('#'):
                assert href[1:] in parsed.ids, ('Unresolved evidence anchor',page,href)
            else:
                assert (page.parent/href).is_file(), ('Missing link/photo',page,href)
            links_checked+=1
    assert all_payloads==expected_html, 'HTML altered or omitted exact target/review/caption wording'
    assert all(not r['researcher_confirmed'] and r['overall_preference'] is None and
        all(v is None for v in r['condition_preferences'].values()) and
        all(x['original_rating_0_to_3'] is None and x['stage_a_rating_0_to_3'] is None for x in r['rubric_A_G'].values())
        for r in annotations)
    assert all(sha(p)==h for p,h in ((Path(k),v) for k,v in original_hashes.items()))
    for pid,h in copied_photos.items():
        assert sha(bundle/f'photos/{pid}.jpg')==h
    checks=dict(status='PASS_OFFLINE_ADJUDICATION_PREPARATION_ONLY',training_businesses_in_pilot=32,
        priority_cases=12,other20_preserved=True,priority_groups=dict(Counter(c['adjudication_priority_group'] for c in case_data)),
        exact_target_texts_visible=72,exact_token_pairs_reused=36,
        exact_full_reviews_visible=sum(len(c['full_source_record']['reviews']) for c in case_data),
        exact_captions_visible=sum(len(v) for c in case_data for v in c['full_source_record']['captions'].values()),
        original_photos_copied=len(copied_photos),all_HTML_payloads_exact=True,HTML_links_checked=links_checked,
        researcher_rows_unconfirmed=12,researcher_scores_assigned=0,researcher_preferences_assigned=0,
        original_inputs_rehashed_unchanged=True,prior_AI_target_quote_warnings=warning_quotes,
        independent_detailed32_report_separately_supplied=False,independent_review_decision_preserved_verbatim=True,
        new_targets_generated=0,targets_edited=0,model_API_GPU_calls=0,validation_test_content_access=False,
        scientific_outcome='NOT_SELECTED_RESEARCHER_ADJUDICATION_PENDING')
    save(bundle/'VALIDATION_REPORT.json',checks)
    save(bundle/'PROVENANCE/SOURCE_PROVENANCE.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        original_paths_and_SHA256=original_hashes,preserved_files={rel:sha(p) for p,rel in preserved.items()},
        label_policy='corrected content/EOS; exact previously measured counts reused; no re-tokenization or target changes'))
    manifest={p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    save(bundle/'SHA256_MANIFEST.json',dict(files=manifest))
    archive=ROOT/'outputs/cloud_handoff/TEACHER_REDESIGN_12_CASE_ADJUDICATION_V1.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(manifest)+1
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    download=Path('C:/Users/VUONGLOCTRUONG/Downloads')/archive.name
    assert not download.exists();cp(archive,download)
    save(OUT/'DELIVERY_RECEIPT.json',dict(archive=str(archive),download=str(download),sha256=sha(archive),
        bytes=archive.stat().st_size,manifest_files=len(manifest),CRC_and_SHA256='PASS',validation=checks))
    print(json.dumps(read(OUT/'DELIVERY_RECEIPT.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
