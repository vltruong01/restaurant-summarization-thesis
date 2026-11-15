"""Read-only export of already confirmed cases affected by provisional CLIP cutoff."""
from pathlib import Path
import base64
import html
import json
import yaml
from stage_f_verification_workflow import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT/'configs/stage_f_consistency_stability.yaml'
OUT = ROOT/yaml.safe_load(POLICY.read_text(encoding='utf-8'))['output_directory']


def build():
    report_path = OUT/'stability_report.json'
    labels_path = ROOT/'outputs/stage_f/ai_assisted_calibration_v1/latest_human_export.json'
    dataset_path = ROOT/'outputs/stage_f/validation_calibration_v1/human_check/annotation_dataset.json'
    before = {str(p): sha(p) for p in [report_path, labels_path, dataset_path]}
    report = read(report_path); labels = read(labels_path)
    cases = {c['case_id']: c for c in read(dataset_path)['cases']}
    confirmed = {c['case_id']: c for c in labels['cases']}
    threshold = report['full_fit']['threshold']
    negatives = [r for r in report['fit_cases'] if not r['positive']]
    rejected = [r for r in report['fit_cases'] if r['positive'] and r['decision_score'] < threshold]
    assert len(negatives) == report['counts']['negative'] and len(rejected) == report['full_fit']['fn']
    private=read(ROOT/'outputs/stage_f/validation_calibration_v1/human_check/private_calibration_scores.json')
    revised=[]
    for cid,h in confirmed.items():
        case=cases[cid]
        if case['kind']=='image_caption' and h['human_verified'] and h['human_final_label']['consistency']=='uncertain' and h.get('human_label_revision_history'):
            score=private[cid]['clip_score']
            revised.append({'case_id':cid,'photo_id':case['photo_ids'][0],'business_id':case['business_id'],
                            'caption':case['normalized_captions'][0],'raw_caption':case['raw_captions'][0],
                            'human_final_label':'uncertain','clip_score':score,'decision_score':round(float(score),6)})
    total=len(negatives)+len(rejected)+len(revised)
    cards = []; source_hashes = {}; exported_ids = []
    for group, rows in [('negative', negatives), ('rejected-positive', rejected),('revised-uncertain',revised)]:
        for index, r in enumerate(sorted(rows, key=lambda x:(x['decision_score'], x['case_id'])), 1):
            cid = r['case_id']; case = cases[cid]; human = confirmed[cid]
            assert human['human_verified'] is True
            assert human['human_final_label']['consistency'] == r['human_final_label']
            assert r['caption'] == case['normalized_captions'][0] and r['raw_caption'] == case['raw_captions'][0]
            assert r['photo_id'] == case['photo_ids'][0] and r['business_id'] == case['business_id']
            source = ROOT/case['image_paths'][0]
            source_hashes[str(source)] = sha(source)
            image = base64.b64encode(source.read_bytes()).decode('ascii')
            passed = r['decision_score'] >= threshold
            result = ('Không dùng case uncertain để fit threshold; score chỉ để đối chiếu' if group=='revised-uncertain' else 'CLIP giữ caption' if passed else 'CLIP loại caption')
            e = html.escape
            cards.append(f'''<article class="card" data-group="{group}" id="{e(cid)}">
<h2>{'Negative đã xác nhận' if group=='negative' else 'Uncertain do bạn vừa xác nhận sửa' if group=='revised-uncertain' else 'Positive bị CLIP loại'} · {index}</h2>
<img loading="lazy" src="data:image/jpeg;base64,{image}" alt="{e(r['photo_id'])}">
<p class="caption">{e(r['caption'])}</p>
<p><strong>Nhãn Consistency của bạn:</strong> {e(r['human_final_label'])} · <strong>Relevance:</strong> {e(human['human_final_label']['relevance'])}</p>
<p><strong>CLIP cosine:</strong> {r['clip_score']:.9f} · score quyết định: {r['decision_score']:.6f}<br>
<strong>{result}</strong> theo threshold tạm {threshold:.2f} (giữ nếu score ≥ threshold).</p>
<details><summary>Raw caption và IDs</summary><p>{e(r['raw_caption'])}</p>
<p>Case: <code>{e(cid)}</code><br>Photo: <code>{e(r['photo_id'])}</code><br>Business: <code>{e(r['business_id'])}</code></p></details></article>''')
            exported_ids.append(cid)
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage F — Audit Consistency đã xác nhận</title>
<style>body{font:16px system-ui;margin:24px auto;padding:0 18px;max-width:1200px;background:#f5f7fa;color:#182335}h1{font-size:26px}.notice{padding:16px;background:#fff1cc;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}.card{background:white;padding:18px;border-radius:10px}.card h2{font-size:18px}.card img{width:100%;height:320px;object-fit:contain;background:#eef1f4}.caption{font-size:18px;white-space:pre-wrap}code{overflow-wrap:anywhere}.toolbar{position:sticky;top:0;background:#f5f7fa;padding:14px 0}button{font:inherit;padding:10px 14px;margin-right:8px;cursor:pointer}button[aria-pressed=true]{background:#163f70;color:white}a{color:#163f70}</style>
<h1>Stage F — Audit CLIP trên nhãn Consistency đã xác nhận</h1>
<p class="notice"><strong>Chỉ xem, không gán lại nhãn.</strong> __TOTAL__ case đều đã được bạn xác nhận: __NNEG__ negative, __NPOS__ positive bị threshold tạm __THRESHOLD__ loại, và __NUNC__ case bạn đã sửa sang uncertain (loại khỏi fit). Không có case pending, không yêu cầu thêm nhãn. Score và threshold được hiển thị để audit sau calibration; threshold chưa freeze. Trang này không ghi file, không lưu browser draft, không chạy calibration hoặc chọn K.</p>
<p>So sánh ảnh và caption với quyết định CLIP. Caption đúng nhưng đơn giản vẫn có thể consistent; một chi tiết thị giác sai có thể inconsistent dù relevance vẫn positive. Các nhãn human dưới đây giữ nguyên.</p>
<p><a href="stability_report.md">Báo cáo stability</a> · <a href="stability_distributions.png">Biểu đồ</a></p>
<nav class="toolbar"><button data-show="negative" aria-pressed="true">__NNEG__ negative</button><button data-show="rejected-positive" aria-pressed="false">__NPOS__ positive bị loại</button><button data-show="revised-uncertain" aria-pressed="false">__NUNC__ uncertain vừa sửa</button><button data-show="all" aria-pressed="false">Tất cả __TOTAL__</button> <span id="count"></span></nav>
<main class="grid">''' + '\n'.join(cards) + '''</main>
<script>
const cards=[...document.querySelectorAll('.card')], buttons=[...document.querySelectorAll('button[data-show]')];
function show(group){let count=0;for(const card of cards){card.hidden=group!=='all'&&card.dataset.group!==group;if(!card.hidden)count++;}for(const button of buttons)button.setAttribute('aria-pressed',String(button.dataset.show===group));document.getElementById('count').textContent=count+' case';}
for(const button of buttons)button.onclick=()=>show(button.dataset.show);show('negative');
</script></html>'''
    for key,value in {'__TOTAL__':total,'__NNEG__':len(negatives),'__NPOS__':len(rejected),'__NUNC__':len(revised),'__THRESHOLD__':f'{threshold:.2f}'}.items():
        page=page.replace(key,str(value))
    target = OUT/'consistency_audit.html'; target.write_text(page, encoding='utf-8')
    assert page.count('src="data:image/jpeg;base64,') == total
    assert len(set(exported_ids)) == total
    assert before == {p: sha(p) for p in before}
    assert source_hashes == {p: sha(p) for p in source_hashes}
    write(OUT/'consistency_audit_export_report.json', {
        'status':'PASS_READ_ONLY_EXPORT', 'negative_cases':len(negatives), 'rejected_positive_cases':len(rejected),
        'user_revised_uncertain_cases':len(revised),'revised_uncertain_excluded_from_fit':True,
        'case_ids':exported_ids, 'source_sha256':before, 'source_images_sha256':source_hashes,
        'human_labels_unchanged':True, 'captions_and_IDs_exact':True,
        'images_embedded':total, 'threshold_frozen':False, 'K_selection_run':False,
        'pending_cases_included':False, 'original_images_unchanged':True,
        'html_sha256':sha(target), 'html_bytes':target.stat().st_size})
    print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'cases':total}))


if __name__ == '__main__': build()
