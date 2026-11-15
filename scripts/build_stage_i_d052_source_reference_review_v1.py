"""Build source-only D-049 reference verification worksheet. No model or API calls.

The exported candidates stay unchanged/unverified; the initial final-unit list is
empty. Only a researcher using the browser UI can confirm working reference units.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/stage_i/paired_lora_pilot_preparation_v1'
OUT = ROOT / 'outputs/stage_i/d052_source_reference_review_v1'
PHOTOS = ROOT / 'data/processed/yelp_2300_v1/photos'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'">
<title>D-052 · Source reference review</title>
<style>
:root{color-scheme:light;--bg:#f2f5f6;--ink:#19313c;--soft:#62747d;--line:#d9e2e7;--accent:#086975;--warn:#fff4d9;--ok:#e4f3e9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}header{background:#123845;color:white;padding:18px 24px}h1{font-size:23px;margin:0 0 7px}h2{font-size:19px}h3{font-size:16px}p{margin:9px 0}button,input,select,textarea{font:inherit}button{cursor:pointer;border:1px solid #aec0c8;background:white;color:var(--ink);border-radius:6px;padding:7px 11px}button.primary{background:var(--accent);color:white;border-color:var(--accent)}button:disabled{opacity:.45;cursor:not-allowed}input,textarea,select{border:1px solid #b8c8cf;border-radius:5px;padding:7px}input[type=text],textarea{width:100%}textarea{min-height:90px;line-height:1.45}small,.muted{color:var(--soft)}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:white;padding:12px 18px;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:4}.toolbar input[type=text]{width:175px}.toolbar .wide{flex:1;min-width:180px}.notice{padding:12px 18px;background:var(--warn);border-bottom:1px solid #ebd59f}.layout{display:grid;grid-template-columns:minmax(360px,1.05fr) minmax(400px,1fr);gap:18px;padding:18px;max-width:1750px;margin:auto}.panel{background:white;border:1px solid var(--line);border-radius:10px;padding:17px;min-width:0}.panel.sticky{align-self:start}.card{border:1px solid var(--line);border-radius:7px;padding:12px;margin:11px 0}.card.done{border-left:5px solid #438755}.card.pending{border-left:5px solid #d2a037}.wrap{white-space:pre-wrap;overflow-wrap:anywhere}.candidate-text{font-size:16px;font-weight:550;margin:9px 0}.controls{display:flex;gap:6px;flex-wrap:wrap}.tag{display:inline-block;background:#e9f0f2;border-radius:4px;padding:2px 7px;font-size:12px}.badge{padding:6px 9px;background:var(--ok);border-radius:5px}.photo{width:100%;max-height:380px;object-fit:contain;background:#ecf0f1;border-radius:5px}.source-review{max-height:570px;overflow:auto;padding:8px 0}.reference-list{padding:0;list-style:none}.reference-list li{padding:10px;border-bottom:1px solid var(--line)}dialog{border:1px solid #9eb1b9;border-radius:11px;width:min(900px,95vw);max-height:92vh;padding:20px}dialog::backdrop{background:#102d3c99}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field{display:block;margin:9px 0}.field label{display:block;font-weight:600;margin:4px 0}.checks{padding:11px;background:#eef4f5;border-radius:5px}.checks label{display:block;margin:7px 0}.error{color:#a62525;background:#fff0ef;padding:8px;border-radius:5px;white-space:pre-wrap}.success{color:#205d39;background:var(--ok);padding:8px;border-radius:5px}.hidden{display:none}details{margin:10px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.4 ui-monospace,Consolas,monospace}.right{margin-left:auto}.progress{font-size:13px;max-width:380px}.complete{background:var(--ok);padding:12px;border-radius:6px}a{color:var(--accent)}@media(max-width:900px){.layout{grid-template-columns:1fr}.toolbar{position:static}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Stage I · Xác minh reference từ nguồn</h1><div>24 validation businesses · 20 random + 4 targeted · nguồn F cố định · không hiển thị student outputs</div></header>
<div class="notice"><strong>Chỉ người dùng quyết định nhãn cuối.</strong> Candidate là gợi ý lấy từ teacher/caption, chưa phải ground truth. Đọc nguồn, tách thành các ý nguyên tử, xác nhận evidence và bổ sung các ý bị thiếu. Không cần coi mô tả đơn giản là lỗi. Trang hoạt động offline, không gửi dữ liệu đi đâu.</div>
<nav class="toolbar"><label>Người xác nhận <input type="text" id="confirmer" placeholder="Tên hoặc định danh"></label><button id="prev">←</button><select id="caseSelect" class="wide"></select><button id="next">→</button><span id="progress" class="progress"></span><button id="save">Tải progress JSON</button><button id="load">Nạp progress</button><input type="file" id="file" accept="application/json" hidden><button class="primary" id="final">Xuất reference đã xác nhận</button></nav>
<div id="storageNotice" class="notice hidden"></div><div id="message"></div>
<main class="layout"><section class="panel"><h2 id="businessTitle"></h2><div id="caseInfo"></div><details><summary>Quy tắc đánh giá đã cố định</summary><pre id="protocol"></pre></details><h3>Customer reviews · toàn văn đã chọn</h3><div id="reviews"></div><h3>Caption F và ảnh gốc</h3><div id="captions"></div></section>
<section class="panel sticky"><h2>Reference candidates <span class="muted">· chưa xác minh</span></h2><p class="muted">Accept / Edit mở cùng một form; chỉ lưu khi bạn xác nhận support, tính nguyên tử và các thuộc tính. Split tạo các bản nháp riêng, từng ý cần xác nhận. Uncertain / Exclude chỉ ghi lý do loại khỏi reference, không biến thành negative.</p><div class="controls"><button id="add" class="primary">+ Thêm ý bị thiếu từ nguồn</button><label><input type="checkbox" id="onlyPending"> Chỉ hiện candidate chưa xử lý</label></div><div id="drafts"></div><div id="candidates"></div><h3>Các reference units bạn đã xác nhận</h3><ul id="units" class="reference-list"></ul><div id="completeness"></div></section></main>
<dialog id="editor"><form id="unitForm"><h2 id="editorTitle">Xác nhận reference unit</h2><p class="muted">Mỗi unit nên là một ý có thể đánh giá độc lập. Sửa nội dung reference không thay đổi teacher target hay caption gốc.</p><div class="field"><label>Nội dung một ý nguyên tử</label><textarea id="unitText" required></textarea></div><div class="grid"><div class="field"><label>Nguồn được phép</label><select id="domain"><option value="review">Selected reviews</option><option value="visual">Supplied caption + ảnh</option></select></div><div class="field"><label>Aspect</label><input id="aspect" type="text" placeholder="food / service / value / atmosphere / visual…" required></div><div class="field"><label>Importance</label><select id="importance" required><option value="">Chọn sau khi đọc nguồn</option><option value="core">core · ý chính cần bảo toàn</option><option value="supporting">supporting · chi tiết bổ sung</option></select></div><div class="field"><label>Restaurant-specific?</label><select id="specific" required><option value="">Chọn sau khi đọc nguồn</option><option value="true">Có</option><option value="false">Không / mô tả chung</option></select></div><div class="field" id="polarityField"><label>Review polarity</label><select id="polarity"><option value="">Chọn sau khi đọc nguồn</option><option value="positive">positive</option><option value="negative">negative</option><option value="mixed">mixed</option><option value="neutral">neutral</option></select></div></div><div class="field"><label>Evidence JSON · chỉ giữ quotes hỗ trợ đúng ý này</label><textarea id="evidence" style="min-height:190px" spellcheck="false"></textarea><small>Review: {marker, review_id, quote}; visual: {marker, photo_id, quote}. Quote phải khớp nguyên văn nguồn.</small></div><div class="field"><label>Ghi chú người kiểm tra</label><textarea id="unitNotes" style="min-height:60px"></textarea></div><div class="checks"><label><input type="checkbox" id="atomic" required> Tôi đã kiểm tra unit chứa một ý có thể đánh giá độc lập.</label><label><input type="checkbox" id="support" required> Tôi đã đọc evidence; ý này được đúng nguồn hỗ trợ. Với visual, không thêm chi tiết chỉ có trong ảnh mà caption không nói.</label><label><input type="checkbox" id="humanConfirm" required> Tôi xác nhận đây là quyết định của mình; gợi ý teacher/AI chỉ là tài liệu hỗ trợ.</label></div><div id="editorError" class="error hidden"></div><div class="controls" style="margin-top:13px"><button type="button" id="cancelEdit">Hủy</button><button type="submit" class="primary">Lưu unit do tôi xác nhận</button></div></form></dialog>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('payload').textContent),KEY='stage_i_source_reference_'+DATA.bundle_id;
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clone=x=>JSON.parse(JSON.stringify(x)),NOW=()=>new Date().toISOString();
let S=clone(DATA.initial_working_state),current=0,context=null;
try{const saved=localStorage.getItem(KEY);if(saved){const value=JSON.parse(saved);if(value.bundle_id===DATA.bundle_id)S=value;}}catch(e){$('storageNotice').classList.remove('hidden');$('storageNotice').textContent='Trình duyệt không cho autosave trên file local. Hãy tải progress JSON thường xuyên.';}
function persist(){S.confirmed_by=$('confirmer').value.trim();S.last_saved_at=NOW();try{localStorage.setItem(KEY,JSON.stringify(S));}catch(e){$('storageNotice').classList.remove('hidden');$('storageNotice').textContent='Autosave không khả dụng. Hãy tải progress JSON để giữ kết quả.';}refreshProgress();}
function message(text,error=false){$('message').innerHTML='<div class="'+(error?'error':'success')+'" style="margin:12px 18px">'+esc(text)+'</div>';}
function person(){const p=$('confirmer').value.trim();if(!p)throw Error('Nhập tên/định danh người xác nhận trước.');return p;}
function caseNow(){return DATA.cases[current];}
function candidateById(id){return DATA.candidates.find(x=>x.unit_id===id);}
function caseById(id){return DATA.cases.find(x=>x.business_id===id);}
function invalidate(bid){S.business_reviews[bid]={human_verified:false,positive_negative_and_specificity_checked:false,confirmed_by:null,notes:S.business_reviews[bid]?.notes||''};S.researcher_confirmed_complete=false;}
function pendingCount(bid){return DATA.candidates.filter(x=>x.business_id===bid&&!(S.candidate_decisions[x.unit_id]?.human_verified)).length;}
function refreshProgress(){const n=Object.values(S.business_reviews).filter(x=>x.human_verified).length,pending=DATA.candidates.length-Object.values(S.candidate_decisions).filter(x=>x.human_verified).length;$('progress').textContent=n+'/24 businesses đầy đủ · '+S.units.length+' units đã xác nhận · '+pending+' candidate chưa xong';}
function render(){const c=caseNow();$('businessTitle').textContent=c.display_name+' · '+c.business_id;$('caseInfo').innerHTML='<span class="tag">'+esc(c.subset)+'</span> <span class="tag">'+c.reviews.length+' reviews</span> <span class="tag">'+c.captions.length+' captions</span><p class="muted">Chỉ nguồn validation cố định. REV/IMG_CAP/PHOTO mappings giữ nguyên.</p>';
$('reviews').innerHTML=c.reviews.map(r=>'<details id="'+esc(r.marker)+'"><summary>'+esc(r.marker)+' · '+esc(r.review_id)+'</summary><div class="wrap source-review">'+esc(r.text)+'</div></details>').join('');
$('captions').innerHTML=c.captions.length?c.captions.map(cap=>'<div class="card"><strong>'+esc(cap.caption_id)+'</strong> · PHOTO '+esc(cap.photo_id)+'<p class="wrap">'+esc(cap.text)+'</p><a href="'+esc(cap.image_file)+'" target="_blank" rel="noopener"><img loading="lazy" class="photo" src="'+esc(cap.image_file)+'" alt="Original photo '+esc(cap.photo_id)+'"></a></div>').join(''):'<div class="notice">Zero Filtered captions: không tạo visual reference unit bằng ảnh hoặc thông tin từ reviews.</div>';
$('drafts').innerHTML=S.drafts.filter(x=>x.business_id===c.business_id).map(d=>'<div class="card pending"><strong>Split draft · chưa xác nhận</strong><p class="wrap">'+esc(d.text)+'</p><button data-action="draft" data-id="'+esc(d.draft_id)+'">Mở và xác nhận riêng</button> <button data-action="dropDraft" data-id="'+esc(d.draft_id)+'">Bỏ bản nháp</button></div>').join('');
$('candidates').innerHTML=DATA.candidates.filter(x=>x.business_id===c.business_id&&(!$('onlyPending').checked||!S.candidate_decisions[x.unit_id]?.human_verified)).map(x=>{const d=S.candidate_decisions[x.unit_id],done=d?.human_verified;return '<article class="card '+(done?'done':'pending')+'"><span class="tag">'+esc(x.domain)+'</span> <small>'+esc(x.unit_id)+'</small><p class="candidate-text wrap">'+esc(x.text)+'</p><div class="muted">'+(done?'Bạn đã xử lý: '+esc(d.status)+' · '+esc(d.confirmed_by):'Gợi ý teacher/caption · human_verified=false')+'</div><details><summary>Evidence gợi ý (chưa phải xác nhận)</summary><pre>'+esc(JSON.stringify(x.evidence,null,2))+'</pre></details><div class="controls"><button data-action="accept" data-id="'+esc(x.unit_id)+'">Accept / Edit</button><button data-action="split" data-id="'+esc(x.unit_id)+'">Split</button><button data-action="uncertain" data-id="'+esc(x.unit_id)+'">Uncertain</button><button data-action="exclude" data-id="'+esc(x.unit_id)+'">Exclude</button></div></article>';}).join('');
$('units').innerHTML=S.units.filter(x=>x.business_id===c.business_id).map(x=>'<li><span class="tag">'+esc(x.domain)+' · '+esc(x.importance)+' · '+esc(x.polarity||'visual')+'</span><p class="wrap">'+esc(x.text)+'</p><small>'+esc(x.unit_id)+' · '+esc(x.confirmed_by)+'</small><div class="controls"><button data-action="editUnit" data-id="'+esc(x.unit_id)+'">Sửa và xác nhận lại</button><button data-action="removeUnit" data-id="'+esc(x.unit_id)+'">Loại unit</button></div></li>').join('');
const complete=S.business_reviews[c.business_id].human_verified;$('completeness').innerHTML='<div class="'+(complete?'complete':'card')+'"><h3>'+ (complete?'✓ Bạn đã xác nhận đầy đủ nguồn':'Xác nhận đầy đủ nguồn cho business này')+'</h3><p>Đọc toàn bộ selected reviews; kiểm tra các aspect chính, ý tích cực/tiêu cực và ý đối lập, chi tiết đặc trưng; thêm các ý quan trọng teacher bỏ sót. Xử lý hoặc giải thích việc loại candidates; không tạo negatives để cân bằng.</p><p>'+pendingCount(c.business_id)+' candidates chưa xử lý · '+S.drafts.filter(x=>x.business_id===c.business_id).length+' split drafts chưa xác nhận</p><button id="completeBusiness">'+(complete?'Bỏ xác nhận để rà lại':'Tôi xác nhận source references của business này đã đầy đủ')+'</button></div>';
$('caseSelect').value=current;$('prev').disabled=current===0;$('next').disabled=current===DATA.cases.length-1;refreshProgress();}
function openEditor(opts){const c=caseNow(),base=opts.unit||opts.candidate||opts.draft||{};context=opts;$('unitForm').reset();$('editorError').classList.add('hidden');$('editorTitle').textContent=opts.unit?'Sửa unit và xác nhận lại':opts.draft?'Xác nhận một split draft':'Tạo reference unit do bạn xác nhận';$('unitText').value=base.text||'';$('domain').value=base.domain||'review';$('aspect').value=opts.unit?.aspect||'';$('importance').value=opts.unit?.importance||'';$('specific').value=typeof opts.unit?.restaurant_specific==='boolean'?String(opts.unit.restaurant_specific):'';$('polarity').value=opts.unit?.polarity||'';$('evidence').value=JSON.stringify(base.evidence||[],null,2);$('unitNotes').value=opts.unit?.notes||'';updateDomain();$('editor').showModal();}
function updateDomain(){const visual=$('domain').value==='visual';$('polarityField').classList.toggle('hidden',visual);$('polarity').required=!visual;}
function validateUnit(u){const c=caseById(u.business_id);if(!c)throw Error('Unknown business.');if(!u.text?.trim()||!u.aspect?.trim())throw Error('Nội dung và aspect không được rỗng.');if(!['core','supporting'].includes(u.importance)||typeof u.restaurant_specific!=='boolean')throw Error('Chọn importance và restaurant specificity.');if(!['review','visual'].includes(u.domain))throw Error('Domain không hợp lệ.');if(u.domain==='review'&&!['positive','negative','mixed','neutral'].includes(u.polarity))throw Error('Chọn review polarity.');if(!Array.isArray(u.evidence)||!u.evidence.length)throw Error('Cần ít nhất một exact evidence quote.');for(const e of u.evidence){if(u.domain==='review'){const r=c.reviews.find(x=>x.marker===e.marker);if(!r||e.review_id!==r.review_id||!e.quote||!r.text.includes(e.quote))throw Error('REV/ID/quote không khớp selected reviews: '+(e.marker||'?'));}else{const cap=c.captions.find(x=>x.caption_id===e.marker);if(!cap||cap.photo_id!==e.photo_id||!e.quote||!cap.text.includes(e.quote))throw Error('IMG_CAP/PHOTO/quote không khớp supplied caption: '+(e.marker||'?'));}}if(!u.human_verified||!u.confirmed_by||u.support_status!=='supported')throw Error('Cần quyết định human-confirmed supported.');}
$('unitForm').addEventListener('submit',e=>{e.preventDefault();try{const who=person(),c=caseNow();if(!$('atomic').checked||!$('support').checked||!$('humanConfirm').checked)throw Error('Cần xác nhận đủ ba mục.');const u={unit_id:context.unit?.unit_id||c.business_id+'_D052_UNIT_'+String(S.next_unit_number++).padStart(4,'0'),business_id:c.business_id,domain:$('domain').value,text:$('unitText').value.trim(),support_status:'supported',importance:$('importance').value,aspect:$('aspect').value.trim(),restaurant_specific:$('specific').value==='true',evidence:JSON.parse($('evidence').value),human_verified:true,confirmed_by:who,confirmed_at:NOW(),atomicity_review:'User explicitly confirmed atomicity',notes:$('unitNotes').value.trim(),candidate_id:context.candidate?.unit_id||context.draft?.candidate_id||context.unit?.candidate_id||null,provenance:'User independently inspected source evidence and explicitly confirmed this reference unit; teacher/caption candidate was only an aid.'};if(u.domain==='review')u.polarity=$('polarity').value;validateUnit(u);if(context.unit)S.units=S.units.filter(x=>x.unit_id!==u.unit_id);S.units.push(u);if(context.draft){S.drafts=S.drafts.filter(x=>x.draft_id!==context.draft.draft_id);const id=context.draft.candidate_id;if(!S.drafts.some(x=>x.candidate_id===id))S.candidate_decisions[id]={status:'split',unit_ids:S.units.filter(x=>x.candidate_id===id).map(x=>x.unit_id),human_verified:true,confirmed_by:who,confirmed_at:NOW()};}else if(context.candidate){S.exclusion_log=S.exclusion_log.filter(x=>x.candidate_id!==context.candidate.unit_id);S.candidate_decisions[context.candidate.unit_id]={status:u.text===context.candidate.text?'accepted':'edited',unit_ids:[u.unit_id],human_verified:true,confirmed_by:who,confirmed_at:NOW()};}invalidate(c.business_id);persist();$('editor').close();render();}catch(err){$('editorError').textContent=err.message;$('editorError').classList.remove('hidden');}});
document.addEventListener('click',e=>{const b=e.target.closest('button[data-action]');if(!b)return;try{const a=b.dataset.action,id=b.dataset.id,candidate=candidateById(id);if(a==='accept'){const existing=S.units.filter(x=>x.candidate_id===id);if(existing.length===1)openEditor({unit:existing[0]});else if(existing.length>1)throw Error('Candidate ?? split: s?a t?ng unit trong danh s?ch ?? x?c nh?n.');else openEditor({candidate});}if(a==='editUnit')openEditor({unit:S.units.find(x=>x.unit_id===id)});if(a==='draft')openEditor({draft:S.drafts.find(x=>x.draft_id===id)});if(a==='split'){person();const answer=prompt('Nhập ít nhất hai ý nguyên tử, mỗi ý một dòng. Mỗi ý sẽ là bản nháp và cần xác nhận riêng. Không tự động sao chép nhãn final.',candidate.text);if(answer===null)return;const texts=answer.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);if(texts.length<2)throw Error('Split cần ít nhất hai ý.');if(S.units.some(x=>x.candidate_id===id))throw Error('Candidate này đã có units. Hãy sửa/loại các units hiện có trước khi Split lại.');if(S.drafts.some(x=>x.candidate_id===id))throw Error('Các split drafts đã tồn tại; hãy xử lý chúng trước.');S.drafts.push(...texts.map((text,i)=>({draft_id:id+'_SPLIT_DRAFT_'+NOW()+'_'+i,business_id:candidate.business_id,candidate_id:id,domain:candidate.domain,text,evidence:clone(candidate.evidence),human_verified:false,confirmed_by:null})));S.candidate_decisions[id]={status:'split_drafts_pending',human_verified:false,confirmed_by:null};invalidate(candidate.business_id);persist();render();}if(a==='exclude'||a==='uncertain'){const who=person();if(S.units.some(x=>x.candidate_id===id)||S.drafts.some(x=>x.candidate_id===id))throw Error('Hãy sửa/loại units và drafts liên quan trước khi loại candidate.');const reason=prompt('Ghi rõ lý do '+a+' sau khi bạn đọc nguồn. Candidate không được biến thành negative tự động.');if(reason===null)return;if(!reason.trim())throw Error('Cần lý do.');if(!confirm('Bạn xác nhận quyết định '+a+' này là quyết định của mình?'))return;const record={candidate_id:id,business_id:candidate.business_id,status:a,notes:reason.trim(),human_verified:true,confirmed_by:who,confirmed_at:NOW()};S.candidate_decisions[id]=record;S.exclusion_log=S.exclusion_log.filter(x=>x.candidate_id!==id);S.exclusion_log.push(record);invalidate(candidate.business_id);persist();render();}if(a==='dropDraft'){person();if(!confirm('Bỏ split draft này? Candidate sẽ chưa hoàn tất nếu còn thiếu quyết định.'))return;const d=S.drafts.find(x=>x.draft_id===id);S.drafts=S.drafts.filter(x=>x.draft_id!==id);S.candidate_decisions[d.candidate_id]={status:'pending',human_verified:false,confirmed_by:null};invalidate(d.business_id);persist();render();}if(a==='removeUnit'){person();if(!confirm('Loại unit đã xác nhận này và mở lại completeness?'))return;const u=S.units.find(x=>x.unit_id===id);S.units=S.units.filter(x=>x.unit_id!==id);if(u.candidate_id)S.candidate_decisions[u.candidate_id]={status:'pending',human_verified:false,confirmed_by:null};invalidate(u.business_id);persist();render();}}catch(err){message(err.message,true);}});
$('completeness').addEventListener('click',e=>{if(e.target.id!=='completeBusiness')return;try{const who=person(),c=caseNow();if(S.business_reviews[c.business_id].human_verified){invalidate(c.business_id);persist();render();return;}if(pendingCount(c.business_id)>0||S.drafts.some(x=>x.business_id===c.business_id))throw Error('Xử lý các candidates/split drafts trước; có thể dùng Exclude hoặc Uncertain kèm lý do, không tạo negative.');if(!S.units.some(x=>x.business_id===c.business_id&&x.domain==='review'))throw Error('Cần ít nhất một review reference unit.');for(const u of S.units.filter(x=>x.business_id===c.business_id))validateUnit(u);if(!confirm('Tôi đã đọc toàn bộ nguồn của business này, đã kiểm tra positive/negative/contrasts, aspects, restaurant specificity và các ý quan trọng bị thiếu. Tôi xác nhận reference đủ để đánh giá output sau này.'))return;S.business_reviews[c.business_id]={human_verified:true,positive_negative_and_specificity_checked:true,confirmed_by:who,confirmed_at:NOW(),notes:S.business_reviews[c.business_id]?.notes||''};persist();render();}catch(err){message(err.message,true);}});
function download(name,value){const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('save').onclick=()=>{persist();download('stage_i_source_reference_progress.json',S);};$('load').onclick=()=>$('file').click();$('file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const value=JSON.parse(await file.text());if(value.bundle_id!==DATA.bundle_id||value.case_inputs_sha256!==DATA.freeze_template.case_inputs_sha256||value.protocol_sha256!==DATA.freeze_template.protocol_sha256)throw Error('Progress thuộc nguồn/protocol khác.');if(!Array.isArray(value.units)||!Array.isArray(value.drafts)||!value.business_reviews||!value.candidate_decisions)throw Error('Progress schema không hợp lệ.');for(const u of value.units)validateUnit(u);if(!confirm('Nạp progress do bạn cung cấp vào trình duyệt này? Không thay đổi files nguồn.'))return;S=value;$('confirmer').value=S.confirmed_by||'';persist();render();message('Đã nạp progress; các xác nhận được giữ theo provenance trong JSON.');}catch(err){message(err.message,true);}e.target.value='';};
$('final').onclick=()=>{try{person();if(!DATA.cases.every(c=>S.business_reviews[c.business_id]?.human_verified&&S.business_reviews[c.business_id]?.positive_negative_and_specificity_checked))throw Error('Cần xác nhận đủ24 businesses trước khi freeze.');if(S.drafts.length||DATA.candidates.some(c=>!S.candidate_decisions[c.unit_id]?.human_verified))throw Error('Còn candidates/drafts chưa xử lý.');const seen=new Set();for(const u of S.units){validateUnit(u);if(seen.has(u.unit_id))throw Error('Duplicate unit ID.');seen.add(u.unit_id);}if(!confirm('Bạn xác nhận bộ reference nguồn đã hoàn tất trước khi xem/sinh student outputs mới? Bước xuất này chỉ ghi quyết định bạn đã xác nhận; Codex vẫn phải kiểm tra schema/hashes offline trước prediction.'))return;S.researcher_confirmed_complete=true;persist();const value={...clone(DATA.freeze_template),researcher_confirmed_complete:true,business_reviews:clone(S.business_reviews),units:clone(S.units),exclusion_log:clone(S.exclusion_log),candidate_decisions:clone(S.candidate_decisions),status:'USER_CONFIRMED_SOURCE_REFERENCE_PENDING_OFFLINE_VALIDATION',confirmed_by:person(),confirmed_at:NOW(),source_review_bundle_id:DATA.bundle_id};download('stage_i_source_reference_user_confirmed.json',value);message('Đã xuất reference do bạn xác nhận. Chưa có prediction nào được chạy; cần offline validation trước khi sử dụng.');}catch(err){message(err.message,true);}};
$('prev').onclick=()=>{if(current>0){current--;render();window.scrollTo(0,0);}};$('next').onclick=()=>{if(current<DATA.cases.length-1){current++;render();window.scrollTo(0,0);}};$('caseSelect').onchange=e=>{current=Number(e.target.value);render();window.scrollTo(0,0);};$('confirmer').onchange=persist;$('onlyPending').onchange=render;$('domain').onchange=updateDomain;$('add').onclick=()=>openEditor({});$('cancelEdit').onclick=()=>$('editor').close();
$('confirmer').value=S.confirmed_by||'';$('caseSelect').innerHTML=DATA.cases.map((c,i)=>'<option value="'+i+'">'+(i+1)+'/24 · '+esc(c.subset)+' · '+esc(c.display_name)+' · '+esc(c.business_id)+'</option>').join('');$('protocol').textContent=DATA.protocol;render();
</script></body></html>'''


def main():
    source_cases = rows(PREP / 'evaluation_inputs.jsonl')
    candidates = rows(PREP / 'reference_unit_candidates.jsonl')
    freeze = json.loads((PREP / 'reference_freeze_template.json').read_text(encoding='utf-8'))
    protocol = (PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md').read_text(encoding='utf-8')
    if len(source_cases) != 24 or len(candidates) != 226:
        raise ValueError('Frozen24 cases /226 candidates differ')
    if any(row.get('human_verified') is not False for row in candidates) or freeze['units']:
        raise ValueError('Preserve prepared unverified candidates; do not overwrite human labels')
    if freeze['case_inputs_sha256'] != sha(PREP / 'evaluation_inputs.jsonl') or freeze['protocol_sha256'] != sha(PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md'):
        raise ValueError('Frozen reference bindings differ')
    cases = []
    for row in source_cases:
        if row['condition'] != 'filtered' or row.get('split', 'validation') != 'validation':
            raise ValueError('Validation F source only')
        if sha_text(row['source']) != row['source_sha256']:
            raise ValueError('Source hash differs')
        review_parts = re.findall(r'\[(REV_\d+)\] (.*?)(?=\n\n\[REV_\d+\]|\n\n\[VISUAL_CONTEXT\]|\Z)', row['source'], flags=re.S)
        if len(review_parts) != len(row['selected_review_ids']):
            raise ValueError('Review mapping differs')
        caption_block = row['source'].split('[VISUAL_CONTEXT]', 1)[-1] if '[VISUAL_CONTEXT]' in row['source'] else ''
        caption_parts = re.findall(r'\[(IMG_CAP_\d+)\]\n(.*?)(?=\n\n\[IMG_CAP_\d+\]|\Z)', caption_block, flags=re.S)
        if len(caption_parts) != len(row['selected_photo_ids']):
            raise ValueError('Caption/photo mapping differs')
        caps = [{'caption_id': marker, 'text': text.strip(), 'photo_id': photo,
                 'image_file': 'images/' + photo + '.jpg'}
                for (marker, text), photo in zip(caption_parts, row['selected_photo_ids'])]
        for cap in caps:
            if not (PHOTOS / (cap['photo_id'] + '.jpg')).is_file():
                raise FileNotFoundError(cap['photo_id'])
        name = re.search(r'^name=(.*)$', row['source'], flags=re.M)
        cases.append({'business_id': row['business_id'], 'subset': row['subset'], 'split': 'validation',
                      'condition': 'filtered', 'display_name': name.group(1) if name else row['business_id'],
                      'source': row['source'], 'source_sha256': row['source_sha256'],
                      'reviews': [{'marker': marker, 'review_id': rid, 'text': text}
                                  for (marker, text), rid in zip(review_parts, row['selected_review_ids'])],
                      'captions': caps, 'selected_review_ids': row['selected_review_ids'],
                      'selected_photo_ids': row['selected_photo_ids']})
    bundle_id = sha_text(freeze['case_inputs_sha256'] + '|' + freeze['protocol_sha256'])
    initial = {'bundle_id': bundle_id, 'case_inputs_sha256': freeze['case_inputs_sha256'],
               'protocol_sha256': freeze['protocol_sha256'], 'confirmed_by': '',
               'researcher_confirmed_complete': False, 'business_reviews': freeze['business_reviews'],
               'units': [], 'drafts': [], 'candidate_decisions': {}, 'exclusion_log': [],
               'next_unit_number': 1, 'status': 'UNVERIFIED_SOURCE_ONLY_WORKSHEET'}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'images').mkdir()
    photo_ids = sorted({cap['photo_id'] for case in cases for cap in case['captions']})
    copies = {}
    for photo in photo_ids:
        source = PHOTOS / (photo + '.jpg')
        target = OUT / 'images' / (photo + '.jpg')
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise ValueError('Image copy differs')
        copies['images/' + photo + '.jpg'] = sha(target)
    # Preserve raw prepared candidates byte-for-byte; never promote them to labels.
    shutil.copyfile(PREP / 'reference_unit_candidates.jsonl', OUT / 'reference_unit_candidates_UNVERIFIED.jsonl')
    shutil.copyfile(PREP / 'reference_freeze_template.json', OUT / 'reference_freeze_template_UNVERIFIED.json')
    shutil.copyfile(PREP / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md', OUT / 'STAGE_I_PILOT_EVALUATION_PROTOCOL.md')
    write(OUT / 'source_cases.json', cases)
    write(OUT / 'initial_working_state_UNVERIFIED.json', initial)
    payload = {'bundle_id': bundle_id, 'cases': cases, 'candidates': candidates,
               'freeze_template': freeze, 'initial_working_state': initial, 'protocol': protocol}
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\u003c')
    (OUT / 'SOURCE_REFERENCE_REVIEW.html').write_text(HTML.replace('__PAYLOAD__', serialized), encoding='utf-8')
    (OUT / 'README_FIRST.md').write_text('''# Stage I source-reference worksheet — offline only

Open `SOURCE_REFERENCE_REVIEW.html` in a browser. It contains the fixed20 random and4 targeted validation F cases, full selected customer reviews, frozen captions and90 original photos. It deliberately excludes historical/new student outputs and teacher final targets. Teacher-derived candidate suggestions are explicitly unverified; they are not independent factual ground truth.

1. Enter your name/identifier. Read the complete source, not only suggested evidence.
2. Accept/Edit opens a unit form. Verify atomicity, support, aspect, polarity (reviews), importance and restaurant specificity. Keep exact evidence IDs/quotes. Only your explicit save creates a confirmed unit.
3. Split creates separate unverified drafts; confirm each independently. Add missing important source facts. Exclude/Uncertain requires your reason and never converts uncertainty to a negative label.
4. Confirm source completeness per business after checking positive/negative/contrasting opinions and omissions. Keep random/targeted roles unchanged.
5. Download progress JSON regularly; browser autosave is convenience, not a durable project artifact. Import it to resume or move browsers.
6. Only after all24 source checks are complete, export the user-confirmed reference JSON. Return it for offline schema/source/hash validation before any new student predictions. Exporting here does not authorize or launch inference.

All original226 candidate records remain byte-for-byte unchanged with `human_verified=false`; the initial final reference-unit list is empty. Browser decisions are separate working-state records. No human decisions or final units were generated by Codex. The source binding uses the original complete evaluation-input file hash without distributing its historical output fields.

Visual units must be supported by the caption supplied to LED. Photos allow independent inspection but do not authorize photo-only facts. A zero-caption source cannot acquire a visual reference from reviews or an unrelated image. Correct but simple descriptions are acceptable. Do not manufacture negative examples or add class-balance requirements.

No GPU/model/teacher calls, held-out access, accepted dataset modifications or scientific parameter changes occur in this worksheet. Internet/network connections are disabled by page policy. Files/relative image paths are portable together.
''', encoding='utf-8')
    report = {'status': 'SOURCE_REVIEW_WORKSHEET_PREPARED_NO_HUMAN_LABELS', 'businesses': len(cases),
              'random': sum(c['subset'] == 'random' for c in cases), 'targeted': sum(c['subset'] == 'targeted' for c in cases),
              'candidates': len(candidates), 'human_verified_candidates': 0, 'final_units': 0,
              'photos': len(photo_ids), 'review_count': sum(len(c['reviews']) for c in cases),
              'zero_caption_businesses': [c['business_id'] for c in cases if not c['captions']],
              'source_candidates_copy_exact': sha(PREP / 'reference_unit_candidates.jsonl') == sha(OUT / 'reference_unit_candidates_UNVERIFIED.jsonl'),
              'historical_or_new_student_outputs_included': False, 'teacher_final_targets_included': False,
              'teacher_candidate_provenance_preserved': True, 'held_out_data': False, 'GPU_or_teacher_calls': 0,
              'binding': {'case_inputs_sha256': freeze['case_inputs_sha256'], 'protocol_sha256': freeze['protocol_sha256']},
              'image_sha256': copies}
    write(OUT / 'EXPORT_REPORT.json', report)
    write(OUT / 'SHA256_MANIFEST.json', {'builder_sha256': sha(Path(__file__)),
          'files': {path.relative_to(OUT).as_posix(): sha(path) for path in sorted(OUT.rglob('*')) if path.is_file()}})
    print(json.dumps({'directory': str(OUT), **{k: report[k] for k in ('businesses','candidates','photos','human_verified_candidates','final_units')}}))


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    main()
