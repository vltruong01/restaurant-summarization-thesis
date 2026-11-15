'use strict';
const payload = JSON.parse(document.getElementById('review-data').textContent);
const cases = payload.businesses;
const images = payload.images;
const meta = payload.metadata;
const lengths = payload.target_lengths;
const noteKey = 'thesis_stage_g_pilot_notes_v1_' + payload.source_zip_sha256.slice(0,12);
let notes = {};
try { notes = JSON.parse(localStorage.getItem(noteKey) || '{}'); } catch (_) {}
const state = {index:0, split:'all', query:'', view:'overview', claim:0, condition:'unfiltered', photo:0, allReviews:false};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const current = () => cases[state.index];
const titles = {reviews_only:'Reviews-only',unfiltered:'Unfiltered',filtered:'Filtered'};
const statusNames = {unseen:'Chưa xem',read:'Đã xem',discuss:'Cần thảo luận'};
function saveNotes(){ try { localStorage.setItem(noteKey,JSON.stringify(notes)); } catch(_){ toast('Trình duyệt không lưu được. Hãy tải ghi chú để giữ lại.'); } }
function toast(text){ $('toast').textContent=text; $('toast').classList.add('visible'); setTimeout(()=>$('toast').classList.remove('visible'),2400); }
function setBusiness(index){ state.index=index; state.claim=0;state.photo=0;state.allReviews=false;render(); }
function renderSidebar(){
  const visible = cases.map((c,i)=>({c,i})).filter(({c}) => (state.split==='all'||c.split===state.split) && [meta[c.business_id].name,meta[c.business_id].city,c.business_id].join(' ').toLowerCase().includes(state.query));
  $('business-list').innerHTML=visible.map(({c,i})=>{const m=meta[c.business_id],s=notes[c.business_id]?.status||'unseen';return `<button class="business-item ${i===state.index?'active':''}" data-index="${i}" aria-current="${i===state.index}"><span class="dot">${s==='discuss'?'● Cần bàn':s==='read'?'✓':''}</span><strong>${esc(m.name)}</strong><small>${esc(m.city)} · ${c.split==='train'?'Train':'Validation'} ${c.zero_filtered_edge_case?'· 0 Filtered':''}</small></button>`;}).join('') || '<p class="subtle">Không tìm thấy business.</p>';
  $('business-list').querySelectorAll('[data-index]').forEach(b=>b.addEventListener('click',()=>setBusiness(Number(b.dataset.index))));
  $('read-progress').textContent=`${cases.filter(c=>['read','discuss'].includes(notes[c.business_id]?.status)).length} / 20 đã xem`;
}
function targetHTML(text){return esc(text).replace(/\[(REVIEW_CONSENSUS|VISUAL_EVIDENCE)\]/g,'<span class="section-marker">[$1]</span>');}
function renderOverview(){
  const c=current(),stats=lengths[c.business_id];
  $('overview-panel').innerHTML='<p class="section-help">So sánh nội dung ba targets. Phần review consensus được dùng chung; chỉ visual evidence thay đổi theo captions mỗi điều kiện.</p><div class="target-grid">'+Object.entries(c.targets).map(([condition,text])=>`<article class="card"><div class="card-head"><h3>${titles[condition]}</h3><span class="pill neutral">${stats[condition+'_LED_tokens']} LED tokens</span></div><div class="target-text">${targetHTML(text)}</div><button class="copy-target" data-copy="${condition}">Sao chép target</button></article>`).join('')+'</div>';
  $('overview-panel').querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',()=>copyText(c.targets[b.dataset.copy])));
  if(c.review_citation_metadata_diagnostics.length) $('overview-panel').insertAdjacentHTML('beforeend',`<details class="diagnostics"><summary>Ghi chú provenance: ${c.review_citation_metadata_diagnostics.length} ID do teacher chép khác mapping nguồn</summary><p>REV marker và span IDs vẫn định vị đúng nguồn. ID gốc do code xác định; raw ID vẫn được giữ riêng. Đây không phải bằng chứng tự động rằng claim đúng/sai.</p><pre>${esc(JSON.stringify(c.review_citation_metadata_diagnostics,null,2))}</pre></details>`);
}
async function copyText(text){try{await navigator.clipboard.writeText(text);toast('Đã sao chép target.');}catch(_){const area=document.createElement('textarea');area.value=text;document.body.appendChild(area);area.select();document.execCommand('copy');area.remove();toast('Đã sao chép target.');}}
function highlightedReview(review,claimIndex){
  const refs=current().review_source_resolution.filter(r=>r.claim_index===claimIndex&&r.review_id===review.review_id);
  const spans=refs.map(r=>[r.start_character,r.end_character]).sort((a,b)=>a[0]-b[0]);
  const merged=[];for(const [a,b] of spans){if(merged.length&&a<=merged[merged.length-1][1])merged[merged.length-1][1]=Math.max(b,merged[merged.length-1][1]);else merged.push([a,b]);}
  const chars=Array.from(review.text);let html='',position=0;
  for(const [a,b] of merged){html+=esc(chars.slice(position,a).join(''))+'<mark>'+esc(chars.slice(a,b).join(''))+'</mark>';position=b;}
  return html+esc(chars.slice(position).join(''));
}
function renderReviews(){
  const c=current(),claims=c.review_consensus_claims;
  const selected=claims[state.claim];
  const ids=new Set((selected?.support||[]).map(s=>s.review_id));
  const shown=state.allReviews?c.reviews:c.reviews.filter(r=>ids.has(r.review_id));
  $('reviews-panel').innerHTML=`<p class="section-help">Bấm một claim để xem các reviews được trích dẫn. Phần tô vàng là đúng span nguồn. Có thể mở toàn bộ ${c.reviews.length} reviews để kiểm tra ý kiến bị bỏ sót.</p><div class="review-layout"><div class="claim-list">${claims.map((cl,i)=>`<button class="claim-button ${i===state.claim?'active':''}" data-claim="${i}"><small>CLAIM ${i+1} · ${esc(cl.polarity)} · ${new Set(cl.support.map(s=>s.review_id)).size} reviews nguồn</small><p>${esc(cl.text)}</p></button>`).join('')}</div><div class="card evidence-panel"><div class="toolbar"><strong>${state.allReviews?'Tất cả reviews đã chọn':'Reviews hỗ trợ claim '+(state.claim+1)}</strong><button id="toggle-all-reviews">${state.allReviews?'Chỉ reviews được trích':'Xem tất cả '+c.reviews.length+' reviews'}</button></div>${shown.map(r=>{const support=(selected?.support||[]).filter(s=>s.review_id===r.review_id);return `<article class="evidence-review"><h3>${esc(r.review_marker)} <span class="pill neutral">${r.stars} ★</span></h3><div class="subtle mono">${esc(r.review_id)} · ${esc(r.date)}</div>${support.map(s=>`<blockquote class="quote">${esc(s.quote)}</blockquote>`).join('')}<details ${shown.length<=2?'open':''}><summary>Đọc toàn bộ review gốc ${support.length?'· có tô span được trích':''}</summary><p class="review-body">${highlightedReview(r,state.claim)}</p></details></article>`;}).join('')}</div></div>`;
  $('reviews-panel').querySelectorAll('[data-claim]').forEach(b=>b.addEventListener('click',()=>{state.claim=Number(b.dataset.claim);renderReviews();}));
  $('toggle-all-reviews').addEventListener('click',()=>{state.allReviews=!state.allReviews;renderReviews();});
}
function renderVisual(){
  const c=current(),condition=c.caption_conditions[state.condition];
  state.photo=Math.min(state.photo,Math.max(0,condition.captions.length-1));
  let content=`<div class="toolbar"><strong>Caption condition</strong>${['unfiltered','filtered'].map(k=>`<button data-condition="${k}" class="${state.condition===k?'active':''}">${titles[k]} · ${c.caption_conditions[k].captions.length} ảnh</button>`).join('')}</div>`;
  if(!condition.captions.length){content+='<div class="empty"><h2>0 Filtered captions</h2><p>Business này được giữ nguyên với visual evidence trống.<br>Không có ảnh/caption thay thế để ép đủ K.</p></div>';}
  else{
    content+=`<div class="photo-strip">${condition.captions.map((cap,i)=>`<button data-photo="${i}" class="photo-tile ${i===state.photo?'active':''}"><img src="${images[cap.photo_id]}" alt="Thumbnail ${esc(cap.caption_id)}"><strong>${esc(cap.caption_id)}</strong></button>`).join('')}</div>`;
    const cap=condition.captions[state.photo],assessment=condition.teacher_assessments[state.photo];
    const confirmations=c.human_confirmations.filter(r=>r.photo_id===cap.photo_id&&r.condition===state.condition);
    content+=`<div class="visual-layout"><div class="card photo-card"><button class="photo-open" id="open-image" aria-label="Mở ảnh lớn"><img src="${images[cap.photo_id]}" alt="Ảnh gốc ${esc(cap.photo_id)}"></button><div class="photo-meta"><span class="mono">${esc(cap.photo_id)}</span><button id="zoom-image">⤢ Xem ảnh lớn</button></div></div><div><div class="card"><div class="card-head"><h3>Caption BLIP-2</h3><span class="pill neutral">${esc(cap.caption_id)}</span></div><p class="caption-text">${esc(cap.caption)}</p><div class="subtle" style="font-size:12px">Đúng caption được cung cấp cho LED và teacher.</div></div><div class="claims-title">GPT teacher · candidate claims</div>${assessment.claims.map((cl,i)=>{const included=condition.serialized_visual_claims_with_all_evidence.some(g=>g.support.some(s=>s.photo_id===cap.photo_id&&s.caption_id===cap.caption_id&&s.raw_claim_index===i));const confirmed=confirmations.find(r=>r.claim===cl.text&&r.human_verified);return `<article class="visual-claim"><div class="card-head"><span class="pill ${cl.verdict}">${esc(cl.verdict)}</span><span class="subtle" style="font-size:11px">Claim ${i+1} · ${included?'Có trong target':'Không vào target'}</span></div><p><strong>${esc(cl.text)}</strong></p><blockquote>Caption support: “${esc(cl.caption_quote)}”</blockquote><p class="why">Teacher explanation: ${esc(cl.explanation)}</p>${confirmed?'<div class="human-note">Bạn đã xác nhận riêng claim này sau khi xem ảnh. Xác nhận không áp dụng cho toàn business.</div>':''}</article>`;}).join('')}${assessment.note?`<p class="subtle">Teacher note: ${esc(assessment.note)}</p>`:''}</div></div>`;
  }
  $('visual-panel').innerHTML=content;
  $('visual-panel').querySelectorAll('[data-condition]').forEach(b=>b.addEventListener('click',()=>{state.condition=b.dataset.condition;state.photo=0;renderVisual();}));
  $('visual-panel').querySelectorAll('[data-photo]').forEach(b=>b.addEventListener('click',()=>{state.photo=Number(b.dataset.photo);renderVisual();}));
  if(condition.captions.length){const cap=condition.captions[state.photo];const open=()=>{$('full-image').src=images[cap.photo_id];$('image-title').textContent=cap.caption_id+' · '+cap.photo_id;$('image-dialog').showModal();};$('open-image').addEventListener('click',open);$('zoom-image').addEventListener('click',open);}
}
function renderNotes(){
  const c=current(),record=notes[c.business_id]||{status:'unseen',text:''};
  $('notes-panel').innerHTML=`<div class="notes-grid"><div class="card"><h2>Ghi chú cho ${esc(meta[c.business_id].name)}</h2><p class="section-help">Đánh dấu tiến độ đọc; không phải xác nhận target đúng hoặc human label cuối.</p><div class="status-choice">${Object.entries(statusNames).map(([k,v])=>`<button data-status="${k}" class="${record.status===k?'active':''}">${v}</button>`).join('')}</div><label for="my-note" style="display:block;margin-top:20px;font-weight:600">Điểm cần bàn với GPT / Codex</label><textarea id="my-note" placeholder="Ví dụ: Unfiltered IMG_CAP_3, claim 2 — tôi không nhìn rõ chi tiết này…">${esc(record.text)}</textarea><span class="subtle" style="font-size:12px">Tự lưu trong trình duyệt. Nút “Tải ghi chú” xuất JSON để bạn gửi lại.</span></div><div class="card notes-help"><h3>Gợi ý khi đọc</h3><ol><li>Review consensus có giữ ý kiến trái chiều và dựa vào đúng reviews không?</li><li>Visual claim có được nói trong caption và nhìn thấy trong đúng ảnh không?</li><li>Claim đúng nhưng đơn giản: ghi riêng về giá trị thông tin, không tự gọi là sai.</li><li>Ghi condition, số claim, REV hoặc IMG_CAP để dễ đối chiếu.</li></ol><p class="section-help">Ghi chú ở đây không sửa dữ liệu gốc. Việc chốt nhãn hoặc method sẽ được xem lại cùng bạn.</p></div></div>`;
  $('my-note').addEventListener('input',()=>{notes[c.business_id]={...(notes[c.business_id]||{status:'unseen'}),text:$('my-note').value};saveNotes();});
  $('notes-panel').querySelectorAll('[data-status]').forEach(b=>b.addEventListener('click',()=>{notes[c.business_id]={...(notes[c.business_id]||{text:''}),status:b.dataset.status};saveNotes();renderNotes();renderSidebar();}));
}
function render(){
  const c=current(),m=meta[c.business_id];
  $('business-name').textContent=m.name;
  $('business-id').textContent=c.business_id;
  $('business-meta').textContent=(c.split==='train'?'TRAIN':'VALIDATION')+' · '+m.city+', '+m.state;
  $('business-position').textContent=(state.index+1)+' / 20';
  $('previous').disabled=state.index===0;$('next').disabled=state.index===cases.length-1;
  const stats=[[c.reviews.length,'Selected reviews'],[c.review_consensus_claims.length,'Review claims'],[c.caption_conditions.unfiltered.captions.length,'Unfiltered captions'],[c.caption_conditions.filtered.captions.length,'Filtered captions']];
  $('business-stats').innerHTML=stats.map(([n,label])=>`<div class="stat"><strong>${n}</strong><span>${label}</span></div>`).join('');
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));
  for(const key of ['overview','reviews','visual','notes'])$(key+'-panel').classList.toggle('hidden',key!==state.view);
  renderSidebar();
  if(state.view==='overview')renderOverview();else if(state.view==='reviews')renderReviews();else if(state.view==='visual')renderVisual();else renderNotes();
}
$('business-search').addEventListener('input',e=>{state.query=e.target.value.trim().toLowerCase();renderSidebar();});
document.querySelectorAll('[data-split]').forEach(b=>b.addEventListener('click',()=>{state.split=b.dataset.split;document.querySelectorAll('[data-split]').forEach(x=>x.classList.toggle('active',x===b));renderSidebar();}));
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;render();}));
$('previous').addEventListener('click',()=>setBusiness(Math.max(0,state.index-1)));
$('next').addEventListener('click',()=>setBusiness(Math.min(cases.length-1,state.index+1)));
$('close-image').addEventListener('click',()=>$('image-dialog').close());
$('image-dialog').addEventListener('click',e=>{if(e.target===$('image-dialog'))$('image-dialog').close();});
$('export-notes').addEventListener('click',()=>{
  const output={schema_version:'stage_g_personal_reading_notes_v1',source_zip_sha256:payload.source_zip_sha256,
    role:'user_reading_notes_for_discussion_not_final_annotations',scientific_outputs_modified:false,
    notes:cases.map(c=>({business_id:c.business_id,split:c.split,reading_status:notes[c.business_id]?.status||'unseen',note:notes[c.business_id]?.text||'',whole_business_human_verified:false}))};
  const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='stage_g_my_review_notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);toast('Đã tải ghi chú riêng; dữ liệu gốc không đổi.');
});
render();
