'use strict';
let actor=null,csrf='',cases=[],current=null,data=null,dirty=false,pending=false,timer=null,selection='',editSequence=0,saveInFlight=null;
const $=id=>document.getElementById(id);
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'';}
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-CSRF-Token':csrf}:{},body:body?JSON.stringify(body):undefined});const x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x;}
function draftKey(){return `stage_l_draft:${location.origin}:${actor.id}:${current}`;}
function draft(){return {review_consensus:$('reviewText').value,visual_evidence:$('visualText').value,review_claims:JSON.parse($('reviewClaims').value),visual_claims:JSON.parse($('visualClaims').value),notes:$('notes').value};}
function actionDraft(action){const value=draft();if(action==='submit'&&$('humanWritten').checked&&$('authorSources').checked&&$('authorBlind').checked){for(const claim of value.visual_claims)for(const link of claim.evidence)link.direct_image_viewed=true;}return value;}
function showPhoto(id){const p=data.source.photos.find(x=>x.photo_id===id);if(!p)return message('Không tìm thấy ảnh của business này.',true);$('photoTitle').textContent=p.photo_id+' · SHA256 '+p.sha256;$('photoLarge').src=p.url;$('photoDialog').showModal();$('claimComponent').value='visual';sourceOptions();$('sourceId').value=p.photo_id;}
function showReview(id){$('reviewsPanel').hidden=false;$('photosPanel').hidden=true;$('reviewSearch').value=id;renderReviews();$('reviewList').scrollIntoView?.({block:'nearest',behavior:'smooth'});}
function removeClaim(part,index){if(!actor||actor.role!=='author'||$('reviewText').readOnly)return;const node=$(part+'Claims'),claims=JSON.parse(node.value);claims.splice(index,1);node.value=JSON.stringify(claims,null,2);changed();message('Đã bỏ nhóm liên kết này, giữ nguyên câu chữ. Có thể thêm nguồn khác ở mục Thêm liên kết còn thiếu.');}
function renderValidation(errors,warnings){const awaiting=['NOT_STARTED','DRAFT','NEEDS_REVISION'].includes(data?.reference?.status);const filtered=errors.filter(e=>!(awaiting&&e==='visual claim requires a truthful direct-image inspection confirmation'));$('validation').textContent=[...filtered,...warnings].join('\n');}
function renderEvidence(){
 const panel=$('evidenceCards');panel.replaceChildren();if(!data)return;let groups=0,links=0,pendingPhotos=0;
 try{
  for(const part of ['review','visual']){
   const claims=JSON.parse($(part+'Claims').value),text=$(part==='review'?'reviewText':'visualText').value;
   if(!Array.isArray(claims))throw Error('Expected evidence array');
   claims.forEach((claim,index)=>{
    if(!claim||typeof claim.output_span!=='string'||!Array.isArray(claim.evidence))throw Error('Malformed evidence');
    groups++;const card=document.createElement('article');card.className='evidence-card';
    const title=document.createElement('strong');title.textContent=(part==='review'?'Review':'Visual')+' · '+(index+1);
    const span=document.createElement('p');span.className='claim-span';span.textContent=claim.output_span;card.append(title,span);
    if(!text.includes(claim.output_span)){const warning=document.createElement('p');warning.className='evidence-warning';warning.textContent='Câu chữ đã thay đổi: cần sửa hoặc bỏ liên kết này.';card.append(warning);}
    for(const link of claim.evidence){
     links++;const block=document.createElement('div');block.className='evidence-link';const id=document.createElement('code');id.textContent=link.review_id||link.photo_id||'Thiếu ID';block.append(id);
     if(part==='review'){const quote=document.createElement('blockquote');quote.textContent=link.quote||'';const button=document.createElement('button');button.textContent='Xem review gốc';button.onclick=()=>showReview(link.review_id);block.append(quote,button);}
     else{const photo=data.source.photos.find(p=>p.photo_id===link.photo_id);if(photo){const button=document.createElement('button'),img=document.createElement('img');img.className='evidence-photo';img.src=photo.url;img.alt='Ảnh nguồn '+photo.photo_id;img.loading='lazy';button.append(img);button.onclick=()=>showPhoto(photo.photo_id);block.append(button);}const status=document.createElement('small');status.textContent=link.direct_image_viewed===true?'Người viết đã xác nhận xem ảnh.':'Chờ người viết kiểm tra và xác nhận khi gửi.';if(link.direct_image_viewed!==true)pendingPhotos++;block.append(status);}
     if(link.note){const detail=document.createElement('details'),summary=document.createElement('summary'),note=document.createElement('p');summary.textContent='Ghi chú của liên kết';note.textContent=link.note;detail.append(summary,note);block.append(detail);}card.append(block);
    }
    if(!claim.evidence.length){const warning=document.createElement('p');warning.className='evidence-warning';warning.textContent='Chưa có nguồn cho ý này.';card.append(warning);}
    if(actor?.role==='author'&&!$('reviewText').readOnly){const remove=document.createElement('button');remove.textContent='Bỏ nhóm liên kết này';remove.onclick=()=>removeClaim(part,index);card.append(remove);}panel.append(card);
   });
  }
  $('evidenceSummary').textContent=groups?`${groups} nhóm ý · ${links} liên kết · ${pendingPhotos} liên kết ảnh chờ xác nhận. Đây là nguồn gợi ý để kiểm tra, không chứng minh toàn bộ bài đúng hoặc đủ bằng chứng.`:'Chưa có liên kết phù hợp với câu chữ hiện tại. Có thể thêm bằng công cụ bên dưới; không cần sửa JSON.';
 }catch(e){$('evidenceSummary').textContent='Dữ liệu liên kết cần kiểm tra ở mục nâng cao: '+e.message;}
 $('unlinkedClaims').replaceChildren();$('unlinkedPanel').hidden=true;
 const marker='EVIDENCE_SUGGESTIONS_SUMMARY: ';
 const line=$('notes').value.split('\n').findLast(x=>x.startsWith(marker));
 if(line){try{const summary=JSON.parse(line.slice(marker.length));const missing=[...(summary.unlinked_review_spans||[]).map(x=>'Review: '+x),...(summary.unlinked_visual_spans||[]).map(x=>'Visual: '+x)];for(const text of missing){const p=document.createElement('p');p.textContent=text;$('unlinkedClaims').append(p);}$('unlinkedPanel').hidden=!missing.length;}catch(e){/* Keep participant notes intact even if intake metadata was edited. */}}
}
function stash(){if(!current||!actor||!dirty)return;const raw={reviewText:$('reviewText').value,visualText:$('visualText').value,reviewClaims:$('reviewClaims').value,visualClaims:$('visualClaims').value,notes:$('notes').value};try{localStorage.setItem(draftKey(),JSON.stringify({version:data.version,raw}));}catch(e){message('Browser recovery storage unavailable. Save to server before navigating.',true);}}
function counts(){$('reviewWords').textContent=`${$('reviewText').value.trim().split(/\s+/).filter(Boolean).length} words`;$('visualWords').textContent=`${$('visualText').value.trim().split(/\s+/).filter(Boolean).length} words`;for(const part of ['review','visual']){try{$(part+'ClaimCount').textContent=`(${JSON.parse($(part+'Claims').value).length})`;}catch(e){$(part+'ClaimCount').textContent='(invalid JSON — local recovery retained)';}}renderEvidence();}
function changed(){editSequence++;dirty=true;counts();stash();$('saveState').textContent='Unsaved changes · browser recovery saved';clearTimeout(timer);timer=setTimeout(()=>save('save',true),1200);}
async function start(session){actor=session.actor;csrf=session.csrf;$('login').hidden=true;$('workspace').hidden=false;$('logout').hidden=false;$('identity').textContent=`${actor.id} · ${actor.role}`;await refreshCases();message('Source-only annotation. Choose a business; read all selected reviews and original photos.');}
async function refreshCases(){const x=await api('/api/cases');cases=x.cases;$('progress').textContent=Object.entries(x.progress).map(([s,n])=>s+': '+n).join(' · ');renderCases();}
function renderCases(){const q=$('caseSearch').value.toLowerCase(),status=$('statusFilter').value;$('caseList').replaceChildren();for(const c of cases){if(!`${c.name} ${c.business_id}`.toLowerCase().includes(q)||(status&&c.status!==status))continue;const b=document.createElement('button');b.textContent=`${c.name}\n${c.status} · ${c.reviews} reviews · ${c.photos} photos`;if(c.business_id===current)b.className='active';b.onclick=()=>openCase(c.business_id);$('caseList').append(b);}}
async function openCase(b){if(pending){message('A save is running. Wait for its receipt before navigating.',true);return;}if(dirty){const ok=await save('save',true);if(ok&&dirty){message('New edits arrived during saving. Wait for their saved receipt before navigating.',true);return;}if(!ok&&!confirm('Server save failed. Browser recovery is retained. Leave this business?'))return;}clearTimeout(timer);const x=await api('/api/case/'+encodeURIComponent(b));current=b;data=x;dirty=false;const r=x.reference;$('businessName').textContent=x.source.business_metadata.name||b;$('businessId').textContent=b;$('caseStatus').textContent=r.status;
for(const [id,k] of [['reviewText','review_consensus'],['visualText','visual_evidence'],['notes','notes']])$(id).value=r[k];for(const p of ['review','visual'])$(p+'Claims').value=JSON.stringify(r[p+'_claims'],null,2);
const editable=actor.role==='author'&&['NOT_STARTED','DRAFT','NEEDS_REVISION'].includes(r.status)&&(!r.author.human_id||r.author.human_id===actor.id);
for(const id of ['reviewText','visualText','notes','reviewClaims','visualClaims'])$(id).readOnly=!editable;
for(const id of ['save','submit','addClaim','useSelection'])$(id).disabled=!editable;
$('authorActions').hidden=!editable;$('checkerActions').hidden=!(actor.role==='checker'&&r.status==='SUBMITTED'&&r.author.human_id!==actor.id);
for(const el of document.querySelectorAll('.attestations input'))el.checked=false;$('checkerComment').value='';$('history').textContent=JSON.stringify({author:r.author,checker:r.checker,history:r.history},null,2);$('saveState').textContent='Server version '+x.version;renderValidation(x.errors,x.warnings);$('reviewSearch').value='';renderReviews();renderPhotos();sourceOptions();counts();renderCases();
const cached=localStorage.getItem(draftKey());if(editable&&cached){const d=JSON.parse(cached);if(confirm('An unsaved browser recovery draft exists. Restore it? Server version '+x.version+'; recovery base '+d.version+'.')){for(const [k,v] of Object.entries(d.raw))$(k).value=v;dirty=true;stash();counts();$('saveState').textContent='Recovered draft — review and Save';if(d.version!==x.version)message('Recovery was based on an older version. Compare the server history before saving; no automatic overwrite.',true);}else localStorage.removeItem(draftKey());}}
function renderReviews(){if(!data)return;const q=$('reviewSearch').value.toLowerCase();$('reviewList').replaceChildren();let n=0;for(const r of data.source.reviews){if(!`${r.review_id} ${r.text}`.toLowerCase().includes(q))continue;n++;const box=document.createElement('article');box.className='review';const id=document.createElement('code');id.textContent=r.review_id;const meta=document.createElement('div');meta.className='muted';meta.textContent=`${r.stars} stars · ${r.date}`;const text=document.createElement('p');text.textContent=r.text;const link=document.createElement('button');link.textContent='Link excerpt from this review';link.disabled=actor.role!=='author'||$('reviewText').readOnly;link.onmousedown=()=>{selection=window.getSelection().toString();};link.onclick=()=>{$('claimComponent').value='review';sourceOptions();$('sourceId').value=r.review_id;$('sourceQuote').value=selection;$('claimSpan').focus();};box.append(id,meta,text,link);$('reviewList').append(box);}$('reviewCounts').textContent=`Showing ${n} of ${data.source.reviews.length} frozen selected reviews. Search does not change the source scope.`;}
function renderPhotos(){$('photoGrid').replaceChildren();$('photoCounts').textContent=`${data.source.photos.length} original photos · all belong to this business`;for(const p of data.source.photos){const b=document.createElement('button'),img=document.createElement('img'),label=document.createElement('span');img.loading='lazy';img.src=p.url;img.alt='Original photo '+p.photo_id;label.textContent=p.photo_id;b.append(img,label);b.onclick=()=>{$('photoTitle').textContent=p.photo_id+' · SHA256 '+p.sha256;$('photoLarge').src=p.url;$('photoDialog').showModal();$('claimComponent').value='visual';sourceOptions();$('sourceId').value=p.photo_id;};$('photoGrid').append(b);}}
function sourceOptions(){if(!data)return;const visual=$('claimComponent').value==='visual';$('sourceId').replaceChildren();for(const r of data.source[visual?'photos':'reviews']){const o=document.createElement('option');o.value=r[visual?'photo_id':'review_id'];o.textContent=o.value;$('sourceId').append(o);}$('quoteLabel').hidden=visual;$('useSelection').hidden=visual;$('photoViewedLabel').hidden=!visual;$('photoViewed').checked=false;}
function addClaim(){try{const p=$('claimComponent').value,k=p+'Claims',span=$('claimSpan').value,source=$('sourceId').value;const text=$(p==='review'?'reviewText':'visualText').value;if(!span||!text.includes(span))throw Error('Claim span must occur exactly in your reference text.');let ev;if(p==='review'){const r=data.source.reviews.find(x=>x.review_id===source),quote=$('sourceQuote').value;if(!quote||!r.text.includes(quote))throw Error('Supply an exact excerpt from the selected review.');ev={review_id:source,quote};}else{if(!$('photoViewed').checked)throw Error('Inspect this original photo and confirm before linking it.');ev={photo_id:source,direct_image_viewed:true};}ev.note=$('evidenceNote').value;const claims=JSON.parse($(k).value);let c=claims.find(x=>x.output_span===span);if(!c){c={output_span:span,evidence:[]};claims.push(c);}c.evidence.push(ev);$(k).value=JSON.stringify(claims,null,2);changed();message('Evidence link added. Confirm semantic support with your checker.');}catch(e){message(e.message,true);}}
function save(action='save',quiet=false){
 if(pending){
  if(action!=='submit')return Promise.resolve(false);
  message('Đang lưu nháp; yêu cầu gửi checker sẽ tiếp tục sau khi lưu xong.');
  return (saveInFlight||Promise.resolve(false)).then(ok=>ok?(data.reference.status==='SUBMITTED'?true:save(action,quiet)):false);
 }
 saveInFlight=performSave(action,quiet);return saveInFlight;
}
async function performSave(action='save',quiet=false){
 if(!current||pending)return false;
 if(action==='save'&&!dirty)return true;
 if(actor.role!=='author')return false;
 clearTimeout(timer);pending=true;
 if(action==='submit')for(const id of ['reviewText','visualText','notes','reviewClaims','visualClaims'])$(id).readOnly=true;
 $('save').disabled=true;$('submit').disabled=action==='submit';
 const sequenceAtSave=editSequence;stash();
 try{
  if(action==='submit'){
   if(!['humanWritten','authorSources','authorBlind'].every(id=>$(id).checked))throw Error('Cần đủ ba xác nhận của người viết trước khi gửi checker.');
   const value=draft();if(!value.review_claims.length||!value.visual_claims.length)throw Error('Chưa có liên kết bằng chứng phù hợp cho cả hai phần. Mở mục Thêm liên kết còn thiếu; nội dung vừa sửa có thể không còn khớp bằng chứng cũ.');
  }
  const x=await api('/api/action/'+encodeURIComponent(current)+'/'+action,{expected_version:data.version,draft:actionDraft(action),attestations:{human_written:$('humanWritten').checked,sources_checked:$('authorSources').checked,no_predictions_or_teacher:$('authorBlind').checked}});
  data.version=x.version;data.reference=x.reference;dirty=editSequence!==sequenceAtSave;
  if(dirty){stash();timer=setTimeout(()=>save('save',true),1200);}else localStorage.removeItem(draftKey());
  $('saveState').textContent=(dirty?'New changes pending · saved base ':'Saved · server version ')+x.version;
  renderValidation(x.errors,x.warnings);
  if(!quiet)message(action==='submit'?'Submitted. A different human checker must verify this exact version.':'Draft saved with revision history.');
  await refreshCases();if(action==='submit'){pending=false;await openCase(current);}return true;
 }catch(e){message('Save failed: '+e.message+'\nBrowser recovery retained; do not clear browser storage.',true);return false;}
 finally{pending=false;if(action==='submit'&&data.reference.status!=='SUBMITTED')for(const id of ['reviewText','visualText','notes','reviewClaims','visualClaims'])$(id).readOnly=false;for(const id of ['submit','save'])$(id).disabled=$('reviewText').readOnly;}
}
async function check(action){try{const x=await api('/api/action/'+encodeURIComponent(current)+'/'+action,{expected_version:data.version,comment:$('checkerComment').value,attestations:{independent_check:$('independent').checked,sources_checked:$('checkerSources').checked,no_predictions_or_teacher:$('checkerBlind').checked,coverage_and_component_separation_checked:$('checkerCoverage').checked}});message(action==='approve'?'Approval recorded for this exact version.':'Feedback recorded. The author must revise and resubmit.');await refreshCases();await openCase(current);}catch(e){message(e.message,true);}}
$('login').onsubmit=async e=>{e.preventDefault();try{await start(await api('/api/login',{identity:$('loginId').value,password:$('password').value}));$('password').value='';}catch(e){message(e.message,true);}};
$('logout').onclick=async()=>{if(pending)return;if(dirty&&!await save('save'))return;await api('/api/logout',{});location.reload();};
for(const id of ['reviewText','visualText','reviewClaims','visualClaims','notes'])$(id).addEventListener('input',changed);
$('caseSearch').oninput=renderCases;$('statusFilter').onchange=renderCases;$('reviewSearch').oninput=renderReviews;$('claimComponent').onchange=sourceOptions;$('addClaim').onclick=addClaim;
$('useSelection').onmousedown=()=>{selection=window.getSelection().toString();};$('useSelection').onclick=()=>{if(selection)$('sourceQuote').value=selection;};
$('save').onclick=()=>save();$('submit').onclick=()=>save('submit');$('revise').onclick=()=>check('request_revision');$('reject').onclick=()=>check('reject');$('approve').onclick=()=>check('approve');
$('reviewsTab').onclick=()=>{$('reviewsPanel').hidden=false;$('photosPanel').hidden=true;};$('photosTab').onclick=()=>{$('reviewsPanel').hidden=true;$('photosPanel').hidden=false;};$('closePhoto').onclick=()=>$('photoDialog').close();
$('guide').onclick=async()=>{try{$('guideText').textContent=await api('/api/guidelines');$('guideDialog').showModal();}catch(e){message('Sign in to open the source-only writing guide.',true);}};$('closeGuide').onclick=()=>$('guideDialog').close();
window.addEventListener('beforeunload',e=>{if(dirty||pending){stash();e.preventDefault();e.returnValue='Unsaved draft';}});
api('/api/session').then(start).catch(()=>{});
