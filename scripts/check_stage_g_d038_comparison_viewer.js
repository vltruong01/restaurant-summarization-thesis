// Offline render/interaction checks, not a substitute for a real-browser visual inspection.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'outputs/stage_g/pilot_comparison_ui_d038_v1/dist/index.html'),'utf8');
const source=fs.readFileSync(path.join(__dirname,'stage_g_comparison_ui/app.js'),'utf8');
const dataText=html.split('<script id="review-data" type="application/json">')[1].split('</script>')[0];
const data=JSON.parse(dataText),nodes={},storage={},downloads=[];
function node(id){return nodes[id]||(nodes[id]={id,innerHTML:'',textContent:'',value:'',checked:false,
 classList:{add(){},remove(){},toggle(){}},click(){if(this.onclick)this.onclick({target:this})},
 showModal(){this.open=true},close(){this.open=false}})}
node('review-data').textContent=dataText;node('filter').value='all';node('search').value='';
const ctx={document:{getElementById:node,querySelectorAll:()=>[],createElement:()=>({click(){downloads.push(this.download)}})},
 localStorage:{getItem:k=>storage[k],setItem:(k,v)=>storage[k]=v},console,
 Blob:class{constructor(parts){this.parts=parts}},URL:{createObjectURL:()=> 'blob:offline-check',revokeObjectURL(){}},setTimeout:f=>f()};
vm.createContext(ctx);vm.runInContext(source,ctx);
const run=s=>vm.runInContext(s,ctx);
let targetChecks=0,photos=0,claims=0,sourceQuotes=0;
for(let i=0;i<data.businesses.length;i++){
 run(`choose(${i})`);const c=data.businesses[i];
 for(const co of ['reviews_only','unfiltered','filtered']){
  run(`condition='${co}';tab='targets';panel()`);
  for(const v of ['original','revised'])if(c.versions[v].targets){
   const expected=run(`esc(D.businesses[${i}].versions.${v}.targets['${co}'])`);
   assert(node('content').innerHTML.includes(expected),`missing target ${c.business_id}/${co}/${v}`);targetChecks++;
  }
 }
 run("tab='reviews';panel()");
 for(const v of ['original','revised'])for(const cl of c.versions[v].review_consensus_claims){
  assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(cl.text)})`)));
  for(const s of cl.support||[]){assert(c.reviews.some(r=>r.review_marker===s.review_marker&&r.review_id===s.review_id));
   assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(s.quote)})`)));sourceQuotes++}
 }
 for(const co of ['unfiltered','filtered']){
  const caps=c.versions.original.caption_conditions[co].captions;
  run(`condition='${co}';tab='photos';photoIndex=0;panel()`);
  if(!caps.length){assert(node('content').innerHTML.includes('0 captions'));continue}
  for(let p=0;p<caps.length;p++){
   run(`photoIndex=${p};photos()`);assert(node('content').innerHTML.includes(caps[p].photo_id));
   node('open-image').onclick();assert(node('zoom-image').src===data.images[caps[p].photo_id]);node('close-zoom').onclick();
   for(const v of ['original','revised']){
    const cond=c.versions[v].caption_conditions[co];
    const a=run(`assessmentFor(D.businesses[${i}].versions.${v}.caption_conditions['${co}'],'${caps[p].photo_id}')`);
    if(cond.accepted_component===false)assert(node('content').innerHTML.includes('RAW PHẢN HỒI BỊ TỪ CHỐI'));
    for(const cl of a?.claims||[]){assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(cl.text)})`)));claims++}
   }photos++;
  }
 }
 run("tab='findings';panel()");for(const f of c.ai_second_opinion_findings)assert(node('content').innerHTML.includes(f.finding_id));
}
run("choose(0);tab='notes';panel()");node('note').value='Offline smoke note';node('read').checked=true;node('save-note').onclick();
assert(JSON.parse(storage.stage_g_d038_comparison_notes_v1)[data.businesses[0].business_id].human_verified===false);
node('download').onclick();assert(downloads[0]==='stage_g_d038_my_review_notes.json');
node('search').value='pWT3WAUvww4iqF-v2htBXA';node('search').oninput();assert(node('list').innerHTML.includes('BỊ CHẶN'));
node('search').value='';node('filter').value='blocked';node('filter').onchange();assert((node('list').innerHTML.match(/data-business=/g)||[]).length===1);
assert(targetChecks===117&&photos===164);
const report={status:'OFFLINE_RENDER_AND_INTERACTION_CHECKS_PASS',businesses:20,target_checks:targetChecks,
 condition_photo_checks:photos,visual_claim_occurrences_rendered:claims,source_quotes_checked:sourceQuotes,
 zero_caption_and_blocked_task_checks:true,zoom_search_notes_export_checks:true,
 actual_browser_visual_QA:false,teacher_API_calls:0,scientific_outputs_changed:false};
fs.writeFileSync(path.join(root,'outputs/stage_g/pilot_comparison_ui_d038_v1/offline_runtime_check_report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
