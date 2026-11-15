// Bounded frontend state tests with fake DOM/network, never human data or scores.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const elements=new Map(),store=new Map();
function element(id){if(!elements.has(id))elements.set(id,{id,value:'',checked:false,hidden:false,disabled:false,textContent:'',elements:[],addEventListener(){}});return elements.get(id)}
const context={document:{getElementById:element},window:{addEventListener(){}},fetch:async()=>{throw Error('Synthetic bootstrap offline')},localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v),removeItem:k=>store.delete(k)},location:{reload(){}},confirm:()=>true,setTimeout,console};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'stage_n_human_web_v1/app.js'),'utf8'),context);
context.assert=assert;
vm.runInContext(`
globalThis.runTests=async()=>{
 const criteria=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution'];
 session={actor:{id:'A01'},csrf:'SYNTHETIC',rubric:{criteria:criteria.map(id=>({id}))}};
 current={alias:'SYNTHETIC_ALIAS',version:0,rating:{status:'NOT_STARTED'},visual_applicable:true};
 for(const c of criteria)$('score_'+c).value='';
 $('notes').value='SYNTHETIC persistence note';$('certainty').value='none';
 $('ratingForm').elements=[...criteria.map(c=>$('score_'+c)),$('notes'),$('save'),$('submit')];
 assert.strictEqual(Object.values(draft().scores).filter(v=>v===null).length,6);
 let resolve, calls=0;api=async()=>{calls++;return await new Promise(r=>resolve=r)};
 itemList=async()=>{};openItemAfterSave=async()=>{};
 let pending=save('save');assert.strictEqual(busy,true);
 assert($('ratingForm').elements.every(e=>e.disabled));assert($('open').disabled&&$('items').disabled);
 await save('save');assert.strictEqual(calls,1);
 resolve({version:1,rating:{status:'DRAFT'}});await pending;
 assert.strictEqual(current.version,1);assert.strictEqual(dirty,false);assert.strictEqual(busy,false);
 assert($('ratingForm').elements.every(e=>!e.disabled));assert(!$('open').disabled);
 api=async()=>{throw Error('Version conflict: synthetic stale draft')};await save('save');
 assert.strictEqual(current.version,1);assert.strictEqual(dirty,true);
 assert(JSON.parse(localStorage.getItem(localKey())).draft.evidence_notes==='SYNTHETIC persistence note');
 assert($('message').textContent.includes('Version conflict'));
 api=async()=>({version:2,rating:{status:'SUBMITTED'}});await save('submit');
 assert($('ratingForm').elements.every(e=>e.disabled));assert.strictEqual(current.rating.status,'SUBMITTED');
 api=async()=>({actor:{id:'A01',role:'rater'},rubric:{criteria:[]},live_scoring_released:false});rubric=()=>{};
 await loadSession();assert.strictEqual($('login').hidden,true);assert.strictEqual($('participant').hidden,false);
 assert.strictEqual($('declarations').hidden,true);assert.strictEqual($('work').hidden,true);
 assert($('eligibility').textContent.includes('Preparation access only'));
 api=async()=>({actor:{id:'A01',role:'rater'},rubric:{criteria:[]},live_scoring_released:true,direct_rating_access:true});
 await loadSession();assert.strictEqual($('participant').hidden,true);assert.strictEqual($('declarations').hidden,true);assert.strictEqual($('work').hidden,false);
 // Exercise the real navigation handler, mocking only the rendering surface.
 renderCurrentItem=()=>{dirty=false;$('visualOutput').textContent=current.visual_text??'No Visual component provided.'};
 current={alias:'SYNTHETIC_OLD',visual_text:null};dirty=false;busy=false;opening=false;
 api=async path=>({alias:path.split('/').pop(),visual_text:'[VISUAL_EVIDENCE]\\nSYNTHETIC distinct view'});
 $('items').value='SYNTHETIC_NEW';await $('items').onchange();
 assert.strictEqual(current.alias,'SYNTHETIC_NEW');assert.strictEqual($('items').value,current.alias);assert($('visualOutput').textContent.includes('distinct view'));
 dirty=true;confirm=()=>false;$('items').value='SYNTHETIC_CANCEL';await $('items').onchange();
 assert.strictEqual(current.alias,'SYNTHETIC_NEW');assert.strictEqual($('items').value,current.alias);assert.strictEqual(dirty,true);
 confirm=()=>true;dirty=false;
 let resolveNavigation;api=async()=>await new Promise(r=>resolveNavigation=r);
 const navigation=openItem('SYNTHETIC_PENDING');assert.strictEqual(opening,true);assert($('items').disabled&&$('open').disabled);
 await openItem('SYNTHETIC_TOO_FAST');assert.strictEqual(current.alias,'SYNTHETIC_NEW');
 resolveNavigation({alias:'SYNTHETIC_PENDING',visual_text:'SYNTHETIC pending output'});await navigation;
 assert.strictEqual(current.alias,'SYNTHETIC_PENDING');assert.strictEqual($('items').value,current.alias);assert.strictEqual(opening,false);
 api=async()=>{throw Error('SYNTHETIC navigation failure')};$('items').value='SYNTHETIC_ERROR';await $('items').onchange();
 assert.strictEqual(current.alias,'SYNTHETIC_PENDING');assert.strictEqual($('items').value,current.alias);assert(!$('items').disabled&&!$('open').disabled);
 assert($('message').textContent.includes('navigation failure'));
 return {status:'PASS',scope:'Fake DOM and synthetic data only',checks:['blank scores remain null','all controls blocked during save','duplicate in-flight save ignored','version advances on acknowledgement','conflict preserves local recovery','submitted item stays locked','preparation login displays rubric only and hides declarations and TEST scoring','authorized direct access opens work without declarations or coordinator step','dropdown change opens and renders the new item automatically','cancelled unsaved navigation restores the current selection','concurrent navigation blocked and selection matches loaded output','failed navigation retains current output and restores controls']};
};`,context);
context.runTests().then(x=>console.log(JSON.stringify(x,null,2))).catch(e=>{console.error(e);process.exitCode=1});
