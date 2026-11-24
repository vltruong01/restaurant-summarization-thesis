'use strict';
(()=>{const D=window.TRIPADVISOR15_DATA,$=id=>document.getElementById(id);if(!D){$('error').hidden=false;$('error').textContent='Data could not load. Extract the entire package, including data/ and assets/.';return;}
const PRIVATE=D.edition==='private_complete',PHOTOS=PRIVATE||D.edition==='public_photos',SYSTEMS=D.systems,CRITERIA=D.criteria;
let index=0,photoIndex=0;
const E=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
const name=id=>SYSTEMS.find(s=>s.id===id).name;const num=(n,d=2)=>n===null||n===undefined?'NOT MEASURED':Number(n).toFixed(d);const signed=n=>(n>0?'+':'')+n.toFixed(2);
function table(headers,rows,classes=[]){const t=E('table'),head=E('thead'),tr=E('tr');headers.forEach((h,i)=>tr.append(E('th',classes[i]||'',h)));head.append(tr);t.append(head);const body=E('tbody');rows.forEach(row=>{const r=E('tr');row.forEach((v,i)=>{const c=E(i===0?'th':'td',classes[i]||'');if(i===0)c.scope='row';if(v instanceof Node)c.append(v);else c.textContent=v;r.append(c);});body.append(r);});t.append(body);return t;}
function model(r,id){return r.outputs.find(o=>o.system===id);}
// Recompute displayed private means directly from the 90 projected original records.
for(const r of D.restaurants)for(const o of r.outputs)if(PRIVATE)for(const c of CRITERIA)o.means[c.id]=o.ratings.reduce((a,x)=>a+x.scores[c.id],0)/o.ratings.length;
function aggregate(c,id){return D.restaurants.reduce((a,r)=>a+model(r,id).means[c],0)/D.restaurants.length;}
function tallies(c,id){const n={higher:0,equal:0,lower:0};for(const r of D.restaurants){const diff=model(r,'L').means[c]-model(r,id).means[c];n[diff>0?'higher':diff<0?'lower':'equal']++;}return n;}
function barChart(target,values,max,decimals){const wrap=$(target);wrap.replaceChildren();for(const v of values){const row=E('div','barRow'),track=E('div','barTrack'),bar=E('div','bar '+v.system);bar.style.width=(v.value/max*100)+'%';track.append(bar);row.append(E('span','',name(v.system)),track,E('strong','barValue',num(v.value,decimals)));wrap.append(row);}const axis=E('div','axis');axis.append(E('span','','0'),E('span','',String(max)));wrap.append(axis);}
function fillSelect(id){for(const c of CRITERIA){const o=E('option','',c.label);o.value=c.id;$(id).append(o);}}
function drawAggregate(){const c=$('chartCriterion').value;barChart('aggregateChart',SYSTEMS.map(s=>({system:s.id,value:aggregate(c,s.id)})),5,2);$('aggregateChart').setAttribute('aria-label',CRITERIA.find(x=>x.id===c).label+' mean ratings, axis 0 to 5');}
function drawRestaurantTable(){const c=$('restaurantCriterion').value;const rows=D.restaurants.map(r=>{const a=E('a','',r.name);a.href='#restaurant/'+encodeURIComponent(r.business_id);return[a,...SYSTEMS.map(s=>num(model(r,s.id).means[c])),signed(model(r,'L').means[c]-model(r,'C').means[c]),signed(model(r,'L').means[c]-model(r,'I').means[c])];});$('restaurantTable').replaceChildren(table(['Restaurant',...SYSTEMS.map(s=>s.name),'LED − Qwen-Caption','LED − Qwen-Image'],rows,['','numeric','numeric','numeric','numeric','numeric']));}
function drawOverview(){
 const counts=[['restaurants',D.cohort.restaurants],['selected reviews',D.cohort.selected_reviews],['selected photos',D.cohort.selected_photos],['original outputs',D.cohort.outputs],['V1 rating records',D.cohort.rating_records]];
 for(const [l,n]of counts){const e=E('div','stat');e.append(E('strong','',String(n)),E('span','',l));$('cohortCounts').append(e);}
 $('editionBadge').textContent=PRIVATE?'Local evidence edition':'Public photo edition';$('editionSummary').textContent=PRIVATE?'Complete source evidence · local review only':'Public photo edition · original selected photographs and captions';$('editionNotice').textContent=PRIVATE?'Complete local evidence for supervisor inspection. Sources and generated passages are included for restricted review; do not upload this folder to a public repository.':'This public edition includes the 73 exact selected photographs and their original/filtered BLIP-2 captions. Source review bodies, generated summary passages and individual evaluator records remain excluded. Aggregate results, restaurant means and output hashes are unchanged.';
 fillSelect('chartCriterion');fillSelect('restaurantCriterion');$('chartCriterion').value='visual_grounding_and_contribution';$('restaurantCriterion').value='usefulness';drawAggregate();drawRestaurantTable();
 $('aggregateTable').append(table(['Criterion',...SYSTEMS.map(s=>s.name)],CRITERIA.map(c=>[c.label,...SYSTEMS.map(s=>num(aggregate(c.id,s.id)))]),['','numeric','numeric','numeric']));
 $('pairwiseTable').append(table(['Criterion','LED vs Qwen-Caption: higher / equal / lower','LED vs Qwen-Image: higher / equal / lower'],CRITERIA.map(c=>[c.label,...['C','I'].map(id=>{const t=tallies(c.id,id);return[t.higher,t.equal,t.lower].join(' / ');})])));
 const gen=D.resources.generation;barChart('timeChart',gen.map(v=>({system:v.system,value:v.seconds})),400,3);barChart('memoryChart',gen.map(v=>({system:v.system,value:v.allocated_GiB})),20,3);
 $('generationTable').append(table(['System','Generation (s)','Peak allocated (GiB)','Calls / tokens','Measurement boundary'],gen.map(v=>[name(v.system),num(v.seconds,3),num(v.allocated_GiB,3),v.calls+' / '+v.tokens,v.definition]),['','numeric','numeric','','']));
 $('imageTimingNote').textContent='Qwen-Image’s '+num(D.resources.generation[2].seconds,3)+' s is a constructed sum: '+num(D.resources.qwen_shared_review_seconds,3)+' s reused Qwen Review + '+num(D.resources.qwen_image_visual_seconds,3)+' s direct-image Visual. The Visual-only peak is '+num(D.resources.qwen_image_visual_allocated_GiB,3)+' GiB; the full-workload memory entry is the maximum across the recorded components.';
 for(const v of gen){const p=E('p');p.append(E('strong','',name(v.system)+': '),document.createTextNode(v.hardware+'; '+v.precision+'. '+v.note));$('generationDefinitions').append(p);}
 $('preprocessingTable').append(table(['Stage','Elapsed (s)','Peak allocated (GiB)','Hardware','Scope'],D.resources.preprocessing.map(v=>[v.stage,num(v.seconds,3),num(v.allocated_GiB,3),v.hardware,v.scope]),['','numeric','numeric','','']));
}
function card(o,component){
 const e=E('section','modelCard system-'+o.system);e.dataset.system=o.system;
 e.append(E('h3','',name(o.system)),E('p','outputLabel','Original '+(component==='review'?'Review Consensus':'Visual Evidence')));
 if(o[component+'_text']!==null){const t=E('div','modelText',o[component+'_text']);t.dataset.output=o.system+'-'+component;e.append(t);}
 else e.append(E('p','withheldText','Original output text is withheld in this public edition. The complete local viewer retains the exact frozen text.'));
 const d=E('details','outputDigest');d.append(E('summary','','Frozen output SHA-256'),E('p','mono digest',o[component+'_sha256']));e.append(d);return e;
}
function publisherLinks(host,entry){
 const links=E('div','publisherLinks');
 for(const [i,url]of (entry?.publisher_pages||[]).entries()){
  const a=E('a','',i===0?'Open publisher source page':'Additional publisher source page');a.href=url;a.target='_blank';a.rel='noopener noreferrer';a.referrerPolicy='no-referrer';links.append(a);
 }
 if(!links.children.length)links.append(E('p','footnote','No publisher page recorded for this photo.'));
 host.append(links);
}
function evidenceUsage(host){
 const usage=E('ul','photoUsage');
 for(const [system,text]of [['L','LED F-S1000: filtered caption (text)'],['C','Qwen-Caption: filtered caption (text)'],['I','Qwen-Image: original photo (pixels)']])usage.append(E('li','systemText-'+system,text));
 host.append(usage);
}
function showSources(){
 const r=D.restaurants[index],entry=window.TRIPADVISOR15_PHOTO_INDEX?.restaurants.find(x=>x.business_id===r.business_id);
 $('sourceAlignment').textContent=r.review_count+' original selected reviews supply Review Consensus. No photo-derived evidence is supplied to the Review pass.';
 $('reviewCount').textContent=r.review_count+' reviews';$('photoCount').textContent=r.photo_count+' photos';
 $('sourceRestriction').hidden=PRIVATE;
 $('sourceRestriction').textContent='Public edition: source reviews, generated passages and individual evaluator scores remain withheld. The complete local edition retains these unchanged for restricted inspection.';
 $('galleryRestriction').hidden=PHOTOS;
 $('galleryRestriction').textContent='Photographs and BLIP-2 caption text are unavailable in this public deployment because redistribution rights have not been established. This is an explicit publication restriction, not a broken image loader. All '+r.photo_count+' selected photo IDs are listed below with recorded publisher-page links. The complete offline edition contains the actual photos and exact captions.';
 $('galleryStatus').textContent=PHOTOS?r.photo_count+' frozen original images · click a photograph to enlarge · captions shown verbatim':r.photo_count+' frozen source references · image bytes and caption text withheld · no external images loaded';
 $('photoGallery').replaceChildren();$('reviewList').replaceChildren();$('expandReviews').hidden=!PRIVATE;$('expandReviews').textContent='Expand all reviews';
 if(!entry||entry.photos.length!==r.photo_count){
  $('galleryStatus').textContent='Photo mapping unavailable. Extract the entire package, including data/photo-index.js. No substitute photos are used.';
  $('photoGallery').append(E('p','notice error','The selected-photo source index could not be loaded or matched.'));
 }else if(!PHOTOS){
  for(const p of entry.photos){
   const f=E('article','photoCard withheldPhotoCard');f.dataset.photoId=p.photo_id;
   f.append(E('div','publicationPlaceholder','Original photo withheld from public deployment'),E('p','mono photoId',p.photo_id),E('span','captionLabel','Original BLIP-2 caption'),E('p','caption','Caption withheld here; exact text is preserved in the complete offline edition.'));
   evidenceUsage(f);publisherLinks(f,p);$('photoGallery').append(f);
  }
 }else{
  for(const [i,p]of r.photos.entries()){
   const reference=entry.photos[i],f=E('article','photoCard originalPhotoCard');f.dataset.photoId=p.photo_id;
   if(reference.photo_id!==p.photo_id){f.append(E('p','notice error','Photo mapping mismatch. No substitute image loaded.'));$('photoGallery').append(f);continue;}
   const b=E('button','photoButton'),im=new Image();b.setAttribute('aria-label','Enlarge '+p.photo_id);b.dataset.photo=p.photo_id;im.alt=r.name+' · '+p.photo_id;im.src=p.src;
   im.onerror=()=>{b.replaceChildren(E('span','photoFailure','Original image is unavailable. Extract the entire offline package; no replacement is used.'));};b.append(im);b.onclick=()=>openPhoto(i);
   f.append(b,E('p','mono photoId',p.photo_id),E('p','nativeSize',p.width+' × '+p.height+' px · frozen original'));
   const raw=E('p','caption');raw.append(E('span','captionLabel','Original BLIP-2 caption'),E('span','captionText',p.original_caption));f.append(raw);
   const filtered=E('details','filteredCaptionDetails');filtered.append(E('summary','','Exact filtered caption supplied as input'),E('p','caption filteredCaption',p.filtered_caption));f.append(filtered);
   evidenceUsage(f);const enlarged=E('button','textButton enlargeLink','Enlarge original photo');enlarged.onclick=()=>openPhoto(i);f.append(enlarged);publisherLinks(f,reference);$('photoGallery').append(f);
  }
 }
 if(!PRIVATE){$('reviewList').append(E('div','accessMessage','Original review text is available in the complete local evidence viewer. No source text is embedded in this public page.'));return;}
 for(const [i,v]of r.reviews.entries()){
  const e=E('details','reviewItem');e.dataset.review=v.review_id;e.open=i===0;
  const summary=E('summary'),meta=E('span','reviewMeta');meta.append(E('strong','','Review '+(i+1)),E('span','',v.stars===null?'Rating unavailable':v.stars+' / 5 stars'),E('time','',v.date||'Date unavailable'));summary.append(meta);e.append(summary,E('p','mono reviewId',v.review_id),E('div','reviewText',v.text));$('reviewList').append(e);
 }
}

function expandReviews(){const reviews=[...document.querySelectorAll('.reviewItem')],open=reviews.some(e=>!e.open);reviews.forEach(e=>e.open=open);$('expandReviews').textContent=open?'Collapse reviews':'Expand all reviews';}
function meanScoreTable(r){
 const t=table(['Criterion',...SYSTEMS.map(s=>s.name)],CRITERIA.map(c=>[c.label,...SYSTEMS.map(s=>num(model(r,s.id).means[c.id]))]),['','numeric','numeric','numeric']);t.classList.add('meanScoreTable');
 for(const [i,s]of SYSTEMS.entries()){t.querySelectorAll('thead th')[i+1].classList.add('systemText-'+s.id);}
 for(const [i,c]of CRITERIA.entries())for(const [j,s]of SYSTEMS.entries())t.querySelectorAll('tbody tr')[i].querySelectorAll('td')[j].dataset.score=s.id+'-mean-'+c.id;
 return t;
}
function scoreTable(r){
 if(!PRIVATE)return table(['Criterion',...SYSTEMS.map(s=>s.name+' mean')],CRITERIA.map(c=>[c.label,...SYSTEMS.map(s=>num(model(r,s.id).means[c.id]))]),['','numeric','numeric','numeric']);
 const t=E('table','caseScoreTable'),head=E('thead'),a=E('tr'),metric=E('th','','Criterion');metric.rowSpan=2;metric.scope='col';a.append(metric);for(const s of SYSTEMS){const h=E('th','',s.name);h.colSpan=3;h.scope='colgroup';a.append(h);}const b=E('tr');for(const s of SYSTEMS)for(const label of ['A01','A02','Mean'])b.append(E('th','numeric'+(label==='Mean'?' mean':''),label));head.append(a,b);t.append(head);const body=E('tbody');for(const c of CRITERIA){const tr=E('tr'),label=E('th','',c.label);label.scope='row';tr.append(label);for(const s of SYSTEMS){const o=model(r,s.id);for(const who of ['A01','A02']){const td=E('td','numeric',num(o.ratings.find(x=>x.evaluator===who).scores[c.id]));td.dataset.score=s.id+'-'+who+'-'+c.id;tr.append(td);}const td=E('td','numeric mean',num(o.means[c.id]));td.dataset.individualMean=s.id+'-mean-'+c.id;tr.append(td);}body.append(tr);}t.append(body);return t;
}
function renderCase(){if($('photoDialog').open)$('photoDialog').close();photoIndex=0;const r=D.restaurants[index];$('restaurantSelect').value=String(index);$('caseName').textContent=r.name;$('caseId').textContent=r.business_id;$('casePosition').textContent='Restaurant '+(index+1)+' / '+D.restaurants.length;$('caseLink').href='#restaurant/'+encodeURIComponent(r.business_id);
 for(const id of ['previous','previousBottom'])$(id).disabled=index===0;for(const id of ['next','nextBottom'])$(id).disabled=index===14;
 showSources();$('reviewOutputs').replaceChildren(...r.outputs.map(o=>card(o,'review')));$('visualOutputs').replaceChildren(...r.outputs.map(o=>card(o,'visual')));$('caseScores').replaceChildren(meanScoreTable(r));$('individualScores').open=false;$('raterScores').replaceChildren(...(PRIVATE?[scoreTable(r)]:[]));$('raterRestriction').textContent=PRIVATE?'Original V1 ratings, unchanged. Mean scores above average these two records.':'Individual evaluator scores are withheld in this public edition; only the original two-rater means are published.';$('ratingDefinition').textContent=PRIVATE?'Original V1 ratings · mean of A01 and A02. Expand below to inspect individual scores. No scores are edited in this viewer.':'Two-rater arithmetic means from the original official V1 collection. Individual rater-level records are withheld in this public release.';
 document.title=r.name+' · TripAdvisor15';
}
function go(n){if(n<0||n>=15)return;location.hash='restaurant/'+encodeURIComponent(D.restaurants[n].business_id);}
function route(){const hash=decodeURIComponent(location.hash.slice(1)),caseId=hash.startsWith('restaurant/')?hash.slice(11):null;let view='restaurants';$('error').hidden=true;if(caseId){const i=D.restaurants.findIndex(r=>r.business_id===caseId);if(i>=0){index=i;view='restaurants';}else{$('error').hidden=false;$('error').textContent='Unknown restaurant identifier. Showing the first frozen restaurant; select a case above.';}}else if(hash==='overview'||hash==='results')view='overview';else if(hash==='restaurants')view='restaurants';else if(hash==='methods'||hash==='methodology')view='methods';
 $('overviewView').hidden=view!=='overview';$('restaurantView').hidden=view!=='restaurants';$('methodsView').hidden=view!=='methods';for(const a of document.querySelectorAll('[data-view]')){if(a.dataset.view===view)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');}if(view==='restaurants')renderCase();else document.title='TripAdvisor15 · LED and Qwen';window.scrollTo(0,0);
}
function openPhoto(i){photoIndex=i;renderPhoto();$('photoDialog').showModal();}
function renderPhoto(){const r=D.restaurants[index],p=r.photos[photoIndex];$('photoTitle').textContent=r.name+' · Photo '+(photoIndex+1)+' / '+r.photos.length;$('largePhoto').src=p.src;$('largePhoto').alt=p.photo_id;$('largePhoto').style.width=Math.max(p.width,Math.min(700,p.width*4))+'px';$('photoResolution').textContent='Original '+p.width+' × '+p.height+' pixels. Enlarged display only; image bytes are unchanged.';$('largeCaption').textContent='Original BLIP-2 caption: '+p.original_caption;$('largeFiltered').textContent='Exact filtered input caption: '+p.filtered_caption;$('photoDigest').textContent=p.photo_id+' · SHA-256 '+p.sha256;$('photoSource').href=p.source_url;$('photoPrevious').disabled=photoIndex===0;$('photoNext').disabled=photoIndex===r.photos.length-1;}
function photoMove(n){if(photoIndex+n>=0&&photoIndex+n<D.restaurants[index].photos.length){photoIndex+=n;renderPhoto();}}
function methods(){
 const m=D.model_identity;$('modelIdentity').append(table(['Model','Frozen identity'],[['LED F-S1000','Source-Separated V1 checkpoint-1000. Adapter SHA-256: '+m.LED.adapter_sha256+'. Backbone SHA-256: '+m.LED.backbone_sha256],['Qwen-Caption / Qwen-Image',m.Qwen.model_id+' · revision '+m.Qwen.revision+' · BF16 · thinking disabled.'],['BLIP-2',m.BLIP.model_id+' · revision '+m.BLIP.revision]]));
 const ul=E('ul');for(const text of D.resources.notes)ul.append(E('li','',text));$('resourceLimits').append(ul);$('provenanceTable').append(table(['Artifact','SHA-256 / provenance'],[['Original V1 rating export',D.provenance.original_v1_rating_sha256],['Frozen predictions',D.provenance.frozen_prediction_sha256],['Frozen blinded assignment mapping',D.provenance.assignment_key_sha256],['Selected review export',D.provenance.source_review_sha256],['Original V1 parent archive (recorded digest)',D.provenance.original_v1_parent_archive_recorded_sha256]]));
 $('publicationText').textContent=PRIVATE?'This full local edition is for restricted supervisor inspection. It contains source reviews, selected photos, captions and unchanged generated passages. Publication permission was not established. Do not deploy this folder publicly. A separately built public edition excludes those texts and images and excludes rater-level records.':'This public edition includes 73 selected original image files and their exact BLIP-2 captions, released at the researcher’s request. Images retain the exact frozen inference bytes and native dimensions; no replacement photos, upscaling or regenerated captions are used. Original review bodies, generated summary passages and individual evaluator records remain excluded. Summary hashes and all published numerical results are unchanged.';
}
for(const [i,r]of D.restaurants.entries()){const o=E('option','',(i+1)+'. '+r.name);o.value=String(i);$('restaurantSelect').append(o);}
$('restaurantSelect').onchange=e=>go(Number(e.target.value));for(const id of ['previous','previousBottom'])$(id).onclick=()=>go(index-1);for(const id of ['next','nextBottom'])$(id).onclick=()=>go(index+1);$('chartCriterion').onchange=drawAggregate;$('restaurantCriterion').onchange=drawRestaurantTable;$('expandReviews').onclick=expandReviews;$('photoClose').onclick=()=>$('photoDialog').close();$('photoPrevious').onclick=()=>photoMove(-1);$('photoNext').onclick=()=>photoMove(1);
document.addEventListener('keydown',e=>{if(!$('photoDialog').open)return;if(e.key==='ArrowLeft'){e.preventDefault();photoMove(-1);}if(e.key==='ArrowRight'){e.preventDefault();photoMove(1);}});window.addEventListener('hashchange',route);
document.querySelector('a[download]').onclick=e=>{e.preventDefault();const url=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)+'\n'],{type:'application/json;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='TRIPADVISOR15_'+(PRIVATE?'PRIVATE':'PUBLIC')+'_DATA.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);};
drawOverview();methods();route();
})();
