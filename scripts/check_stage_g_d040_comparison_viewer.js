// Exercise all comparison pairs and evidence views offline; not browser visual QA.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'outputs/stage_g/pilot_comparison_ui_d040_v1/dist/index.html'),'utf8');
const source=fs.readFileSync(path.join(__dirname,'stage_g_d040_comparison_ui/app.js'),'utf8');
const text=html.split('<script id="review-data" type="application/json">')[1].split('</script>')[0];
const data=JSON.parse(text),nodes={},storage={},downloads=[];
function node(id){return nodes[id]||(nodes[id]={id,innerHTML:'',textContent:'',value:'',checked:false,
 classList:{add(){},remove(){},toggle(){}},showModal(){this.open=true},close(){this.open=false}})}
node('review-data').textContent=text;node('filter').value='all';node('search').value='';
const ctx={document:{getElementById:node,querySelectorAll:()=>[],createElement:()=>({click(){downloads.push(this.download)}})},
 localStorage:{getItem:k=>storage[k],setItem:(k,v)=>storage[k]=v},console,
 Blob:class{constructor(parts){this.parts=parts}},URL:{createObjectURL:()=> 'blob:offline-check',revokeObjectURL(){}},setTimeout:f=>f()};
vm.createContext(ctx);vm.runInContext(source,ctx);
const run=s=>vm.runInContext(s,ctx);
assert(run('leftVersion')==='D039'&&run('rightVersion')==='D040');
let targets=0,photos=0,claims=0,quotes=0,failedViews=0,emptyViews=0;
const uniqueTargets=new Set();
const pairs=[['D039','D040']];
for(const [left,right] of pairs){
 for(let i=0;i<data.businesses.length;i++){
  run(`choose(${i});leftVersion='${left}';rightVersion='${right}';tab='targets';panel()`);
  const c=data.businesses[i];
  assert(node('content').innerHTML.includes('Bản bên trái')&&!node('content').innerHTML.includes('undefined'));
  for(const co of ['reviews_only','unfiltered','filtered']){
   run(`condition='${co}';tab='targets';panel()`);
   for(const v of [left,right])if(c.versions[v].targets){
    assert(node('content').innerHTML.includes(run(`esc(D.businesses[${i}].versions['${v}'].targets['${co}'])`)));
    uniqueTargets.add(`${i}/${v}/${co}`);targets++;
   }else{assert(v==='D038');assert(node('content').innerHTML.includes('D039 đã hoàn tất'))}
  }
  run("tab='reviews';panel()");
  assert(node('content').innerHTML.includes('id="left-version"'));
  for(const v of [left,right])for(const cl of c.versions[v].review_consensus_claims){
   assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(cl.text)})`)));
   for(const s of cl.support||[]){
    assert(c.reviews.some(r=>r.review_marker===s.review_marker&&r.review_id===s.review_id));
    assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(s.quote)})`)));quotes++;
   }
  }
  for(const co of ['unfiltered','filtered']){
   const caps=c.versions.D039.caption_conditions[co].captions;
   run(`condition='${co}';tab='photos';photoIndex=0;panel()`);
   if(!caps.length){assert(node('content').innerHTML.includes('0 captions'));emptyViews++;continue}
   for(let p=0;p<caps.length;p++){
    run(`photoIndex=${p};photos()`);assert(node('content').innerHTML.includes(caps[p].photo_id));
    node('open-image').onclick();assert(node('zoom-image').src===data.images[caps[p].photo_id]);node('close-zoom').onclick();
    for(const v of [left,right]){
     const cond=c.versions[v].caption_conditions[co];
     const a=run(`assessmentFor(D.businesses[${i}].versions['${v}'].caption_conditions['${co}'],'${caps[p].photo_id}')`);
     if(cond.accepted_component===false){assert(node('content').innerHTML.includes('RAW PHẢN HỒI BỊ TỪ CHỐI'));failedViews++}
     for(const cl of a?.claims||[]){assert(node('content').innerHTML.includes(run(`esc(${JSON.stringify(cl.text)})`)));claims++}
    }photos++;
   }
  }
  run("tab='findings';panel()");for(const f of c.ai_second_opinion_findings)assert(node('content').innerHTML.includes(f.finding_id));
 }
}
// Exercise the actual version-selector callbacks in all three evidence/target tabs.
for(const t of ['targets','reviews','photos']){
 run(`choose(0);tab='${t}';panel()`);
 node('left-version').onchange({target:{value:'D039'}});
 node('right-version').onchange({target:{value:'D040'}});
 assert(run('leftVersion')==='D039'&&run('rightVersion')==='D040');
}
run("choose(0);tab='notes';panel()");node('note').value='Offline comparison note';node('read').checked=true;node('save-note').onclick();
assert(JSON.parse(storage.stage_g_d040_comparison_notes_v1)[data.businesses[0].business_id].human_verified===false);
node('download').onclick();assert(downloads[0]==='stage_g_d040_my_review_notes.json');
node('search').value=data.businesses[0].business_id;node('search').oninput();assert((node('list').innerHTML.match(/data-business=/g)||[]).length===1);
node('search').value='';node('filter').value='validation';node('filter').onchange();assert((node('list').innerHTML.match(/data-business=/g)||[]).length===3);
assert(uniqueTargets.size===48&&targets===48&&failedViews===0&&emptyViews===1);
assert(!source.includes('???')&&!source.includes('B?n'));
const report={status:'OFFLINE_RENDER_AND_INTERACTION_CHECKS_PASS',businesses:8,comparison_pairs:1,
 unique_target_texts_checked:uniqueTargets.size,target_render_checks:targets,condition_photo_render_checks:photos,
 visual_claim_render_checks:claims,review_quote_render_checks:quotes,failed_history_views_checked:failedViews,
 zero_caption_views_checked:emptyViews,version_selector_zoom_search_notes_export_checks:true,
 actual_browser_visual_QA:false,API_calls:0,human_labels_or_teacher_outputs_changed:false};
fs.writeFileSync(path.join(root,'outputs/stage_g/pilot_comparison_ui_d040_v1/offline_runtime_check_report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
