/* Synthetic DOM contract tests, not a real-browser rendering test. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(id=''){this.id=id;this.value='';this.hidden=false;this.readOnly=false;this.disabled=false;this.checked=false;this.children=[];this.textContent='';this.className='';}
 addEventListener(type,fn){this['on'+type]=fn;} replaceChildren(){this.children=[];} append(...x){this.children.push(...x);if(this.id==='sourceId'&&x.length&&!this.value)this.value=x[0].value;} focus(){} showModal(){this.open=true;}close(){this.open=false;}
}
const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);};
const draft={business_id:'SYNTHETIC_DEMO',source_bundle_sha256:'demo',status:'NOT_STARTED',review_consensus:'',visual_evidence:'',review_claims:[],visual_claims:[],notes:'',history:[],author:{human_id:''},checker:{human_id:''}};
const source={business_metadata:{name:'Synthetic restaurant'},reviews:[{review_id:'R1',text:'The soup is good but service is slow.',stars:3,date:'2020'}],photos:[{photo_id:'P1',sha256:'demo',url:'/demo.jpg'}]};
let server={version:0,reference:structuredClone(draft)},saveDelay=null,requests=[];const storage=new Map();
const context={console,document:{getElementById:get,createElement:t=>new Element(),querySelectorAll:()=>[]},window:{addEventListener:()=>{},getSelection:()=>({toString:()=>''})},location:{origin:'http://127.0.0.1:9999',reload:()=>{}},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},confirm:()=>true,setTimeout:()=>1,clearTimeout:()=>{},fetch:async(path,opts={})=>{
 requests.push({path,opts});let x;
 if(path==='/api/session')return {ok:false,json:async()=>({error:'demo not logged in'})};
 if(path==='/api/cases')x={cases:[{business_id:'SYNTHETIC_DEMO',name:'Synthetic restaurant',status:server.reference.status,reviews:1,photos:1}],progress:{NOT_STARTED:1}};
 else if(path.startsWith('/api/case/'))x={source,reference:structuredClone(server.reference),version:server.version,errors:[],warnings:[]};
 else if(path.startsWith('/api/action/')){
  const body=JSON.parse(opts.body);if(saveDelay)await saveDelay.promise;
  server.version++;server.reference={...server.reference,...body.draft,status:path.endsWith('/submit')?'SUBMITTED':'DRAFT'};x={version:server.version,reference:structuredClone(server.reference),errors:[],warnings:[]};
 }else x={};return {ok:true,json:async()=>x};
}};
vm.createContext(context);vm.runInContext(fs.readFileSync('scripts/stage_l_reference_web_v1/app.js','utf8'),context);
const run=code=>vm.runInContext(code,context);
(async()=>{
 await run("start({actor:{id:'SYNTHETIC_AUTHOR',role:'author'},csrf:'demo'})");await run("openCase('SYNTHETIC_DEMO')");
 assert.equal(get('reviewList').children.length,1);assert.equal(get('photoGrid').children.length,1);
 get('reviewText').value='The soup is good.';get('visualText').value='A blue image.';get('claimComponent').value='review';run('sourceOptions()');get('sourceId').value='R1';get('claimSpan').value='soup is good';get('sourceQuote').value='soup is good';get('reviewClaims').value='[]';run('addClaim()');assert.equal(JSON.parse(get('reviewClaims').value)[0].evidence[0].review_id,'R1');
 get('claimComponent').value='visual';run('sourceOptions()');get('sourceId').value='P1';get('claimSpan').value='blue image';get('photoViewed').checked=true;get('visualClaims').value='[]';run('addClaim()');assert.equal(JSON.parse(get('visualClaims').value)[0].evidence[0].photo_id,'P1');
 assert.equal(await run("save('save')"),true);assert.equal(server.version,1);assert.equal(storage.size,0);
 assert.equal(get('evidenceCards').children.length,2);assert.match(get('evidenceSummary').textContent,/2 nhóm ý/);
 run("showReview('R1')");assert.equal(get('reviewSearch').value,'R1');assert.equal(get('reviewsPanel').hidden,false);
 get('visualClaims').value=JSON.stringify([{output_span:'blue image',evidence:[{photo_id:'P1',direct_image_viewed:false}]}]);run('counts()');
 run("showPhoto('P1')");assert.equal(get('photoDialog').open,true);assert.equal(JSON.parse(get('visualClaims').value)[0].evidence[0].direct_image_viewed,false);
 for(const id of ['humanWritten','authorSources','authorBlind'])get(id).checked=false;
 assert.equal(run("actionDraft('submit').visual_claims[0].evidence[0].direct_image_viewed"),false);
 for(const id of ['humanWritten','authorSources','authorBlind'])get(id).checked=true;
 assert.equal(run("actionDraft('save').visual_claims[0].evidence[0].direct_image_viewed"),false);
 assert.equal(run("actionDraft('submit').visual_claims[0].evidence[0].direct_image_viewed"),true);
 assert.equal(JSON.parse(get('visualClaims').value)[0].evidence[0].direct_image_viewed,false); // Payload construction never silently changes draft state.
 get('authorBlind').checked=false;assert.equal(run("actionDraft('submit').visual_claims[0].evidence[0].direct_image_viewed"),false);
 const savedClaims=get('reviewClaims').value,unchangedProse=get('reviewText').value;run("removeClaim('review',0)");assert.equal(JSON.parse(get('reviewClaims').value).length,0);assert.equal(get('reviewText').value,unchangedProse);get('reviewClaims').value=savedClaims;
 const notes=get('notes').value;get('notes').value='EVIDENCE_SUGGESTIONS_SUMMARY: '+JSON.stringify({unlinked_review_spans:['Unlinked candidate'],unlinked_visual_spans:[]});run('renderEvidence()');assert.equal(get('unlinkedPanel').hidden,false);assert.equal(get('unlinkedClaims').children[0].textContent,'Review: Unlinked candidate');get('notes').value=notes;
 for(const id of ['humanWritten','authorSources','authorBlind'])get(id).checked=false;
 get('notes').value='edit before request';run('changed()');let release;saveDelay={promise:new Promise(r=>release=r)};const pending=run("save('save',true)");await Promise.resolve();get('notes').value='edit during request';run('changed()');release();await pending;assert.equal(run('dirty'),true);assert.equal(storage.size,1);assert.equal(get('notes').value,'edit during request');saveDelay=null;
 await run("save('save',true)");assert.equal(server.reference.notes,'edit during request');assert.equal(run('dirty'),false);
 get('reviewClaims').value='not json';run('changed()');assert.equal(await run("save('save')"),false);assert.equal(storage.size,1);assert.match(get('message').textContent,/Save failed/);get('reviewClaims').value='[]';run('changed()');await run("save('save')");
 get('humanWritten').checked=true;get('authorSources').checked=true;get('authorBlind').checked=true;
 assert.equal(await run("save('submit')"),false);assert.equal(server.reference.status,'DRAFT');assert.match(get('message').textContent,/Chưa có liên kết/);
 get('reviewClaims').value=savedClaims;run('changed()');saveDelay={promise:new Promise(r=>release=r)};
 const saving=run("save('save',true)");assert.equal(get('submit').disabled,false);
 const submitting=run("save('submit')");assert.match(get('message').textContent,/yêu cầu gửi checker/);
 release();await saving;saveDelay=null;assert.equal(await submitting,true);assert.equal(get('reviewText').readOnly,true);assert.equal(server.reference.status,'SUBMITTED');
 assert.equal(requests.filter(x=>x.path.endsWith('/submit')).length,1);
 const request=requests.find(x=>x.path.includes('/api/action/'));assert.equal(request.opts.headers['X-CSRF-Token'],'demo');
 console.log('PASS: synthetic DOM loads exact review/photo controls; readable evidence cards/source buttons; removal preserves prose; unresolved candidates visible; photo confirmation only in explicit fully-attested submission payload (never opening/saving); evidence builder; server version/save receipt; in-flight edit retention; invalid JSON recovery; submit locks; CSRF header.');
 console.log('LIMITATION: no real-browser layout, image rendering, keyboard or visual screenshot test.');
})().catch(e=>{console.error(e);process.exitCode=1;});
