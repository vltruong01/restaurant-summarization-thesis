// TEST-ONLY in-memory browser fixture. Never open/click real human cases or save human labels.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class E{constructor(){this.children=[];this.value='';this.textContent='';}append(...items){this.children.push(...items);}replaceChildren(){this.children=[];}click(){this.onclick?.();}}
const elements=Object.fromEntries(['annotator','case','status','position','mode','previous','next','pending','export','import','consistency-policy'].map(id=>[id,new E()]));elements.mode.value='all';
const data={dataset_sha256:'TEST_ONLY',ai_artifact_sha256:'TEST_ONLY',verification_order_sha256:'TEST_ONLY',dataset_kind:'official_calibration',workflow:'AI_assisted_human_verification',selection:{round:0},active_case_ids:['TEST_IMAGE','TEST_PAIR'],images:{p:'data:image/jpeg;base64,TEST',q:'data:image/jpeg;base64,TEST'},cases:[
 {case_id:'TEST_IMAGE',name:'TEST_ONLY',kind:'image_caption',photo_ids:['p'],raw_captions:['test food'],normalized_captions:['test food.'],labels:{consistency:'',relevance:''},ai_suggested_label:{consistency:'consistent',relevance:'relevant'},ai_rationale:'TEST_ONLY'},
 {case_id:'TEST_PAIR',name:'TEST_ONLY',kind:'caption_pair',photo_ids:['p','q'],raw_captions:['test a','test b'],normalized_captions:['test a.','test b.'],labels:{redundancy:'',broad_topic:'',image_duplicate:''},ai_suggested_label:{redundancy:'same_information',broad_topic:'same_broad_visual_topic',image_duplicate:'distinct_content'},ai_rationale:'TEST_ONLY'}]};
data.secondary_review_sha256='TEST_ONLY_SECOND_REVIEW';
data.consistency_refinement={active:true,decision:'D-029'};
data.secondary_reviews={TEST_IMAGE:{consistency:'uncertain',relevance:'relevant',reason:'TEST_ONLY_SECOND_REVIEW'}};
const html=fs.readFileSync('scripts/stage_f_assisted_human_check_template.html','utf8');const source=html.split('<script>')[1].split('</script>')[0].replace('__DATA__',JSON.stringify(data));
const alerts=[];const ctx=vm.createContext({document:{getElementById:id=>elements[id],createElement:()=>new E()},localStorage:{getItem:()=>null,setItem:()=>{}},alert:m=>alerts.push(m),URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},Blob:function(){},setTimeout:()=>{}});vm.runInContext(source,ctx);
const exported=()=>JSON.parse(vm.runInContext('JSON.stringify(exported())',ctx));
function buttons(){return elements.case.children.filter(c=>c.onclick);}
function selects(){return elements.case.children.flatMap(c=>c.children).filter(c=>c.onchange);}
assert.equal(exported().cases.filter(c=>c.human_verified).length,0,'Showing AI suggestions must not verify');
assert.equal(elements['consistency-policy'].hidden,false,'D029 pause notice shown');
assert.ok(elements['consistency-policy'].textContent.includes('Không cần hoàn tất batch pending'),'Pending batch not required solely for negative gate');
buttons().find(b=>b.textContent.startsWith('Tôi xác nhận nhãn trùng')).onclick();assert.equal(alerts.length,1,'Annotator required');assert.equal(exported().cases[0].human_verified,false);
elements.annotator.value='TEST_ONLY';elements.annotator.oninput({target:elements.annotator});buttons().find(b=>b.textContent.startsWith('Tôi xác nhận nhãn trùng')).onclick();
assert.equal(exported().cases[0].human_verified,true);assert.equal(exported().cases[0].verification_action,'accepted_ai');
selects()[0].value='complementary_information';selects()[0].onchange();assert.equal(exported().cases[1].human_verified,false);
buttons().find(b=>b.textContent.startsWith('Tôi xác nhận nhãn do tôi')).onclick();assert.equal(exported().cases[1].verification_action,'corrected_ai');assert.equal(exported().cases[1].human_final_label.redundancy,'complementary_information');
elements.previous.onclick();selects()[0].value='inconsistent';selects()[0].onchange();assert.equal(exported().cases[0].human_verified,false,'Editing invalidates previous verification');
assert.equal(exported().cases[0].ai_suggested_label.consistency,'consistent','Human correction must not mutate AI suggestion');
buttons().find(b=>b.textContent.startsWith('Tôi xác nhận nhãn sau đối')).onclick();
assert.equal(exported().cases[0].human_final_label.consistency,'uncertain');
assert.equal(exported().cases[0].human_verified,true);
assert.equal(exported().cases[0].verification_action,'corrected_ai');
assert.equal(exported().cases[0].verification_source,'user_confirmed_after_chatgpt_comparison');
assert.equal(exported().cases[0].human_confirmation_provenance.final_label_author,'human_user');
assert.equal(exported().cases[0].human_confirmation_provenance.optional_comparison_aid,'ChatGPT');
assert.equal(exported().cases[0].human_confirmation_provenance.decision,'user_confirmed_final_label');
assert.equal(exported().annotation_roles.primary_annotator,'human_user');
assert.equal(exported().chatgpt_second_reviews.TEST_IMAGE.consistency,'uncertain');
assert.equal(exported().cases[0].secondary_review_sha256,'TEST_ONLY_SECOND_REVIEW');
assert.equal(exported().cases[0].ai_suggested_label.consistency,'consistent');
console.log('PASS_TEST_ONLY_UI_NO_AUTO_VERIFY_ACCEPT_CORRECT_EDIT_INVALIDATION_AI_PROVENANCE');
// A stale browser draft must not roll back corrected labels imported by the researcher.
const corrected=JSON.parse(JSON.stringify(data));corrected.imported_human_export_sha256='CORRECTED_SERVER_EXPORT';
corrected.imported_human_state={annotator:'TEST_ONLY',cases:[{case_id:'TEST_IMAGE',human_final_label:{consistency:'inconsistent',relevance:'relevant'},human_verified:true,verification_action:'corrected_ai',verified_at:'TEST_ONLY'}]};
function cachedContext(baseHash){const nodes=Object.fromEntries(Object.keys(elements).map(id=>[id,new E()]));nodes.mode.value='all';const saved={base_human_export_sha256:baseHash,annotator:'TEST_ONLY',rows:{TEST_IMAGE:{case_id:'TEST_IMAGE',human_final_label:{consistency:'consistent',relevance:'relevant'},human_verified:false,verification_action:'',verified_at:''}}};const local=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new E()},localStorage:{getItem:()=>JSON.stringify(saved),setItem:()=>{}},alert:()=>{},URL:{},Blob:function(){},setTimeout:()=>{}});vm.runInContext(html.split('<script>')[1].split('</script>')[0].replace('__DATA__',JSON.stringify(corrected)),local);return JSON.parse(vm.runInContext('JSON.stringify(exported())',local));}
const stale=cachedContext('OLD_EXPORT');assert.equal(stale.cases[0].human_final_label.consistency,'inconsistent');assert.equal(stale.cases[0].human_verified,true);
const current=cachedContext('CORRECTED_SERVER_EXPORT');assert.equal(current.cases[0].human_final_label.consistency,'consistent');assert.equal(current.cases[0].human_verified,false,'Intentional edits on the current imported version remain drafts');
console.log('PASS_TEST_ONLY_STALE_CACHE_CANNOT_OVERWRITE_CORRECTED_IMPORT_CURRENT_DRAFT_RETAINED');
// D-030 changes the notice, never auto-confirms annotations or hides their provenance.
const soft=JSON.parse(JSON.stringify(data));soft.CLIP_filtering_policy={mode:'soft_only'};
const softNodes=Object.fromEntries(Object.keys(elements).map(id=>[id,new E()]));softNodes.mode.value='all';
const softCtx=vm.createContext({document:{getElementById:id=>softNodes[id],createElement:()=>new E()},localStorage:{getItem:()=>null,setItem:()=>{}},alert:()=>{},URL:{},Blob:function(){},setTimeout:()=>{}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0].replace('__DATA__',JSON.stringify(soft)),softCtx);
assert.ok(softNodes['consistency-policy'].textContent.includes('D-030'));
assert.ok(softNodes['consistency-policy'].textContent.includes('không còn threshold'));
assert.equal(JSON.parse(vm.runInContext('JSON.stringify(exported())',softCtx)).cases.filter(c=>c.human_verified).length,0);
console.log('PASS_TEST_ONLY_D030_SOFT_NOTICE_NO_AUTO_VERIFICATION');
const exact=JSON.parse(JSON.stringify(data));exact.image_duplicate_policy={phash_role:'diagnostic_only'};
const exactNodes=Object.fromEntries(Object.keys(elements).map(id=>[id,new E()]));exactNodes.mode.value='all';
const exactCtx=vm.createContext({document:{getElementById:id=>exactNodes[id],createElement:()=>new E()},localStorage:{getItem:()=>null,setItem:()=>{}},alert:()=>{},URL:{},Blob:function(){},setTimeout:()=>{}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0].replace('__DATA__',JSON.stringify(exact)),exactCtx);
assert.ok(exactNodes['consistency-policy'].textContent.includes('D-031'));
assert.ok(exactNodes['consistency-policy'].textContent.includes('pHash is diagnostic-only'));
assert.equal(JSON.parse(vm.runInContext('JSON.stringify(exported())',exactCtx)).cases.filter(c=>c.human_verified).length,0);
console.log('PASS_TEST_ONLY_D031_NOTICE_NO_NEW_LABEL_BATCH_OR_AUTO_CONFIRMATION');
