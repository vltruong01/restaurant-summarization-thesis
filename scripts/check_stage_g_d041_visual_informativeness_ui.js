// Offline mock-DOM interaction checks. Synthetic in-memory drafts are not annotations.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const folder=path.resolve(__dirname,'../outputs/stage_g/d041_visual_informativeness_audit_v1');
const page=fs.readFileSync(path.join(folder,'index.html'),'utf8');
const embedded=page.match(/<script id="bundle" type="application\/json">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(embedded),script=page.match(/<script>([\s\S]*?)<\/script>/)[1];
assert(!script.includes('fetch(')&&!script.includes('XMLHttpRequest'));
const nodes={},storage={},handlers={},downloads=[];
function node(id){return nodes[id]||(nodes[id]={textContent:'',innerHTML:'',open:false,showModal(){this.open=true;},close(){this.open=false;}});}
node('bundle').textContent=embedded;
const context={document:{getElementById:node,addEventListener:(name,fn)=>handlers[name]=fn,
  createElement:()=>({click(){downloads.push(this.download);}})},
  localStorage:{getItem:k=>storage[k],setItem:(k,v)=>storage[k]=v,removeItem:k=>delete storage[k]},
  Blob:class{constructor(parts){this.parts=parts;}},URL:{createObjectURL:()=> 'blob:offline-check',revokeObjectURL(){}},
  Date,JSON,Map,console,setTimeout:f=>f(),confirm:()=>false,location:{reload(){}}};
vm.createContext(context);vm.runInContext(script,context);const run=s=>vm.runInContext(s,context);
assert(data.businesses.length===25&&Object.keys(data.images).length===168);
assert(data.annotation_template.rows.length===693);
let reviewChecks=0,captionChecks=0,factChecks=0,zeroChecks=0;
const content=node('cases').innerHTML,escape=s=>run(`esc(${JSON.stringify(s)})`);
for(const b of data.businesses){
  assert(content.includes(b.business_id)&&content.includes(escape(b.review_consensus)));
  for(const r of b.selected_reviews){assert(content.includes(escape(r.text))&&content.includes(r.review_id));reviewChecks++;}
  for(const cond of ['unfiltered','filtered']){
    const c=b.conditions[cond];assert(c.original_condition_target===b.review_consensus+'\n\n[VISUAL_EVIDENCE]\n'+c.visual_target_text);
    if(c.zero_caption){assert(c.captions.length===0&&c.retained_visual_facts.length===0&&c.visual_target_text==='');zeroChecks++;}
    for(const cap of c.captions){
      assert(content.includes(escape(cap.caption))&&content.includes(cap.photo_id));
      assert(data.images[cap.image_file].startsWith('data:image/jpeg;base64,'));
      assert(cap.human_assessment.informativeness===null&&cap.human_assessment.factual_support===null);
      const form=run(`form(${JSON.stringify(cap.audit_row_id)},'caption')`);
      assert(!form.includes('selected')&&!form.includes('checked'));captionChecks++;
      handlers.click({target:{dataset:{image:cap.image_file}}});
      assert(node('image-dialog').open&&node('zoom-image').src===data.images[cap.image_file]);
      node('close-image').onclick();assert(!node('image-dialog').open);
    }
    for(const fact of c.retained_visual_facts){
      assert(content.includes(escape(fact.text)));assert(fact.human_assessment.informativeness===null);
      for(const support of fact.support)assert(c.captions.some(cap=>cap.caption_id===support.caption_id&&cap.photo_id===support.photo_id));
      factChecks++;
    }
  }
}
assert(reviewChecks===719&&captionChecks===206&&factChecks===437&&zeroChecks===2);
assert(run('state.rows.every(r=>r.informativeness===null&&r.factual_support===null&&r.novelty_vs_reviews===null&&!r.human_verified)'));
const row=data.annotation_template.rows[0];
handlers.input({target:{dataset:{row:row.row_id,field:'chatgpt_second_opinion'},value:'Synthetic UI test placeholder; not a factual review.'}});
assert(run('state.rows[0].human_verified')===false&&run('state.rows[0].human_final_provenance')===null);
handlers.input({target:{dataset:{row:row.row_id,field:'notes'},value:'Synthetic in-memory UI note; not an annotation.'}});
assert(run('state.rows[0].human_verified')===false);
// Only an explicit checkbox event may change human-confirmation provenance.
handlers.input({target:{dataset:{row:row.row_id,field:'human_verified'},checked:true}});
assert(run('state.rows[0].human_verified')===true);
assert(run('state.rows[0].human_final_provenance').includes('User explicitly confirmed'));
handlers.input({target:{dataset:{row:row.row_id,field:'human_verified'},checked:false}});
assert(run('state.rows[0].human_final_provenance')===null);
node('export').onclick();assert(downloads[0]==='d041_visual_informativeness_annotations.json');
// All fake annotations exist only in this mock storage, not the artifact/template or browser.
assert(fs.readFileSync(path.join(folder,'index.html'),'utf8')===page);
const template=JSON.parse(fs.readFileSync(path.join(folder,'annotations_template.json'),'utf8'));
assert(template.rows.every(r=>r.informativeness===null&&r.factual_support===null&&!r.human_verified));
const report={status:'OFFLINE_RENDER_AND_INTERACTION_CHECKS_PASS',businesses:25,
  full_original_review_render_checks:reviewChecks,caption_photo_render_checks:captionChecks,
  target_fact_render_checks:factChecks,original_photo_zoom_checks:captionChecks,unique_original_images:168,
  zero_filtered_views:zeroChecks,blank_annotation_rows:693,JSON_draft_export_and_confirmation_provenance_checked:true,
  source_templates_or_targets_changed:false,API_calls:0,subjective_annotations_created:0,
  interaction_test_inputs_synthetic_in_memory_only:true,actual_browser_visual_QA:false};
fs.writeFileSync(path.join(folder,'offline_ui_check_report.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
