// Offline render/interaction checks; no browser visual QA, API, or annotations.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..'),folder=path.join(root,'outputs/stage_g/teacher_extended_pilot_d041_v1/preparation_review_ui_v1');
const html=fs.readFileSync(path.join(folder,'dist/index.html'),'utf8');
const source=fs.readFileSync(path.join(__dirname,'stage_g_d041_preparation_ui/app.js'),'utf8');
const text=html.split('<script id="review-data" type="application/json">')[1].split('</script>')[0];
const data=JSON.parse(text),nodes={},storage={},downloads=[];
function node(id){return nodes[id]||(nodes[id]={innerHTML:'',textContent:'',value:'',checked:false,
 classList:{add(){},remove(){},toggle(){}},showModal(){this.open=true},close(){this.open=false}})}
node('review-data').textContent=text;node('filter').value='all';node('search').value='';
const ctx={document:{getElementById:node,querySelectorAll:()=>[],createElement:()=>({click(){downloads.push(this.download)}})},
 localStorage:{getItem:k=>storage[k],setItem:(k,v)=>storage[k]=v},console,
 Blob:class{constructor(parts){this.parts=parts}},URL:{createObjectURL:()=> 'blob:offline-check',revokeObjectURL(){}},setTimeout:f=>f()};
vm.createContext(ctx);vm.runInContext(source,ctx);const run=s=>vm.runInContext(s,ctx);
assert(data.teacher_API_calls===0&&data.teacher_targets===0&&data.businesses.length===100);
let reviewChecks=0,photoChecks=0,sourceChecks=0,emptyChecks=0;
const uniqueImages=new Set();
for(let i=0;i<100;i++){
 const c=data.businesses[i];assert(c.teacher_targets===null&&c.human_verified===false);
 run(`choose(${i});tab='overview';panel()`);
 assert(node('content').innerHTML.includes(String(c.review_tokens)));
 assert(node('content').innerHTML.includes('chưa có'));
 run("tab='reviews';panel()");
 for(const r of c.reviews){
  assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(r.text)})`)));
  assert(node('content').innerHTML.includes(r.review_id));reviewChecks++;
 }
 for(const co of ['reviews_only','unfiltered','filtered']){
  run(`condition='${co}';tab='sources';panel()`);
  assert(node('content').innerHTML.includes(run(`esc(D.businesses[${i}].sources['${co}'].source)`)));
  assert(node('content').innerHTML.includes('không phải teacher target'));
  sourceChecks++;
 }
 for(const co of ['unfiltered','filtered']){
  const caps=c.caption_conditions[co].captions;
  assert(c.caption_conditions[co].teacher_assessments===null&&c.caption_conditions[co].human_final_labels===null);
  run(`condition='${co}';tab='photos';photoIndex=0;panel()`);
  if(!caps.length){assert(node('content').innerHTML.includes('0 Filtered captions'));emptyChecks++;continue}
  for(let j=0;j<caps.length;j++){
   run(`photoIndex=${j};photos()`);
   const cap=caps[j];assert(node('content').innerHTML.includes(cap.photo_id));
   assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(cap.caption)})`)));
   node('open-image').onclick();assert(node('zoom-image').src===data.images[cap.photo_id]);node('close-zoom').onclick();
   uniqueImages.add(cap.photo_id);photoChecks++;
  }
  node('condition').onchange({target:{value:co}});assert(run('condition')===co);
 }
}
run("choose(0);tab='protocol';panel()");assert(node('content').innerHTML.includes('gpt-6.1-sol'));
for(const [filter,count] of [['all',100],['manual25',25],['zero',8],['few',26],['train',80],['validation',20]]){
 node('filter').value=filter;node('filter').onchange();assert((node('list').innerHTML.match(/data-business=/g)||[]).length===count);
}
node('filter').value='all';node('search').value=data.businesses[0].business_id;node('search').oninput();assert((node('list').innerHTML.match(/data-business=/g)||[]).length===1);
node('search').value='';node('filter').value='all';
run("choose(0);tab='notes';panel()");node('note').value='Pre-generation discussion note';node('read').checked=true;node('save-note').onclick();
const saved=JSON.parse(storage.stage_g_d041_preparation_notes_v1)[data.businesses[0].business_id];
assert(saved.human_verified===false&&saved.stage==='pre_generation_input_discussion');
node('download').onclick();assert(downloads[0]==='stage_g_d041_preparation_my_notes.json');
assert(reviewChecks===2848&&sourceChecks===300&&photoChecks===815&&uniqueImages.size===659&&emptyChecks===8);
assert(!source.includes('fetch(')&&!source.includes('XMLHttpRequest'));
const report={status:'OFFLINE_RENDER_AND_INTERACTION_CHECKS_PASS',businesses:100,
 complete_review_render_checks:reviewChecks,source_render_checks:sourceChecks,caption_photo_render_checks:photoChecks,
 unique_original_image_mappings_checked:uniqueImages.size,zero_caption_views:emptyChecks,
 all_random25_zero_few_split_search_filters_pass:true,condition_selector_zoom_and_personal_notes_export_pass:true,
 teacher_targets_present:false,API_calls:0,human_labels_modified:false,actual_browser_visual_QA:false};
fs.writeFileSync(path.join(folder,'offline_runtime_check_report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
