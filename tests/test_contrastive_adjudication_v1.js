'use strict';
// Dependency-free DOM fixture tests of the exact inline script, not browser layout tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'outputs/stage_i/led_large_contrastive_adjudication_v1');
const markup = fs.readFileSync(path.join(out, 'ADJUDICATE.html'), 'utf8');
const bindings = JSON.parse(markup.match(/<script type="application\/json" id="bindings">([\s\S]*?)<\/script>/)[1]);
const script = markup.match(/<script>([\s\S]*?)<\/script>/)[1];
let passed = 0;
function test(name, fn) { fn(); passed++; process.stdout.write('PASS ' + name + '\n'); }
function fixture() {
  const controls = {};
  for (const id of ['reviewer','attest','status','draft','final']) {
    controls[id] = {value:'', checked:false, textContent:'', addEventListener(){}};
  }
  controls.bindings = {textContent:JSON.stringify(bindings)};
  const blocks = [];
  for (const [kind, items] of [['positive',bindings.positives],['pair',bindings.pairs]]) {
    for (const key of Object.keys(items)) {
      const select = {value:'PENDING'}, textarea = {value:''}, checks = [{checked:false},{checked:false}];
      blocks.push({dataset:{key:kind+':'+key},select,textarea,checks,addEventListener(){},
        querySelector(s){return s==='select'?select:textarea},querySelectorAll(){return checks}});
    }
  }
  const alerts=[], downloads=[];
  const context = vm.createContext({document:{getElementById(id){return controls[id]},
    querySelectorAll(){return blocks},addEventListener(){},createElement(){return {click(){}}}},
    alert(msg){alerts.push(msg)},Blob:class {constructor(parts){this.text=parts.join('')}},
    URL:{createObjectURL(blob){downloads.push(JSON.parse(blob.text));return 'blob:fixture'},revokeObjectURL(){}},setTimeout(){}});
  vm.runInContext(script,context);
  return {context,blocks,controls,alerts,downloads,run(code){return vm.runInContext(code,context)}};
}
function decide(block,decision) {block.select.value=decision;block.textarea.value='Independent reason citing the source';for(const c of block.checks)c.checked=true;}
test('exactly eight positives and sixteen fixed pairs; excluded controls not selectable',()=>{
 assert.equal(Object.keys(bindings.positives).length,8);
 assert.deepEqual(Object.keys(bindings.pairs),Array.from({length:16},(_,i)=>'P'+String(i+1).padStart(2,'0')));
 assert.equal((markup.match(/<option>PENDING<\/option>/g)||[]).length,24);
 assert(!markup.includes('data-key="pair:P17"'));assert(!markup.includes('data-key="pair:P18"'));
 assert(!markup.includes('<option selected>ACCEPT'));
});
test('initial and reopened state is entirely PENDING, zero acceptance and no GPU approval',()=>{
 const f=fixture();const r=f.run('collect()');assert.equal(f.run('verdict(collect(),false)'),'PENDING_INDEPENDENT_ADJUDICATION');
 assert(Object.values(r.positives).every(x=>x.decision==='PENDING'));
 assert(Object.values(r.pairs).every(x=>x.decision==='PENDING'));
 assert.equal(r.GPU_execution_approved,false);assert.equal(r.training_authorized,false);
 assert.equal(r.numerical_status,'UNVERIFIED');assert.equal(r.reviewed_utc,null);
});
test('unsigned draft does not invent reviewer identity, acceptance or authorization',()=>{
 const f=fixture();f.run('download(false)');const r=f.downloads[0];assert.equal(r.document_type,'UNSIGNED_DRAFT');
 assert.equal(r.reviewer_name,null);assert.equal(r.independently_reviewed,false);assert.equal(r.reviewed_utc,null);
 assert.equal(r.GPU_execution_approved,false);assert.equal(r.branch_status,'PENDING_INDEPENDENT_ADJUDICATION');
});
test('final export is blocked while decisions remain pending',()=>{
 const f=fixture();f.controls.reviewer.value='Fixture reviewer';f.controls.attest.checked=true;f.run('download(true)');
 assert.equal(f.downloads.length,0);assert.equal(f.alerts.length,1);
});
test('ACCEPT requires manual confirmation and evidence-based notes',()=>{
 const f=fixture();f.blocks[0].select.value='ACCEPT';f.run('download(false)');assert.equal(f.downloads.length,0);
 f.blocks[0].textarea.value='A reason';f.run('download(false)');assert.equal(f.downloads.length,0);
 for(const c of f.blocks[0].checks)c.checked=true;f.run('download(false)');assert.equal(f.downloads.length,1);
});
test('any rejected positive or candidate declares NO-GO; no automatic replacement',()=>{
 for(const index of [0,8]){const f=fixture();decide(f.blocks[index],'REJECT');
 assert.equal(f.run('verdict(collect(),false)'),'NO_GO_CONTRASTIVE_BRANCH');f.run('download(false)');
 assert.equal(Object.keys(f.downloads[0].pairs).length,16);assert.deepEqual(Array.from(f.downloads[0].excluded_pair_ids),['P17','P18']);}
});
test('all accepted entries still require reviewer name and independent attestation',()=>{
 const f=fixture();for(const b of f.blocks)decide(b,'ACCEPT');f.run('download(true)');assert.equal(f.downloads.length,0);
 f.controls.reviewer.value='Fixture reviewer';f.run('download(true)');assert.equal(f.downloads.length,0);
 f.controls.attest.checked=true;f.run('download(true)');assert.equal(f.downloads.length,1);
 const r=f.downloads[0];assert.equal(r.branch_status,'ADJUDICATION_COMPLETE_NO_GPU_AUTHORIZATION');
 assert.equal(r.GPU_execution_approved,false);assert.equal(r.training_authorized,false);assert.equal(r.numerical_status,'UNVERIFIED');
 assert.equal(r.independently_reviewed,true);assert(r.reviewed_utc.endsWith('Z'));
});
test('final rejection remains NO-GO and preserves exact bound inventories',()=>{
 const f=fixture();for(const b of f.blocks)decide(b,'ACCEPT');decide(f.blocks.at(-1),'REJECT');
 f.controls.reviewer.value='Fixture reviewer';f.controls.attest.checked=true;f.run('download(true)');
 const r=f.downloads[0];assert.equal(r.branch_status,'NO_GO_CONTRASTIVE_BRANCH');
 for(const [id,p] of Object.entries(bindings.pairs))assert.equal(r.pairs[id].sha256,p.sha256);
});
test('all evidence links resolve to frozen source markers',()=>{
 const ids=new Set(Array.from(markup.matchAll(/\bid="([^"]+)"/g),m=>m[1]));
 assert.equal(ids.size,Array.from(markup.matchAll(/\bid="([^"]+)"/g)).length);
 const links=Array.from(markup.matchAll(/href="#([^"]+)"/g),m=>m[1]);assert(links.length>550);
 for(const id of links)assert(ids.has(id),'Unresolved evidence anchor: '+id);
});
test('source packets and pair copies remain byte-identical to parent',()=>{
 const parent=path.join(root,'outputs/stage_i/led_large_contrastive_pair_qualification_v1');
 for(const bid of Object.keys(bindings.positives))assert.deepEqual(fs.readFileSync(path.join(out,'positives',bid+'.json')),fs.readFileSync(path.join(parent,'positives',bid+'.json')));
 for(const pid of Object.keys(bindings.pairs))assert.deepEqual(fs.readFileSync(path.join(out,'pairs',pid+'.json')),fs.readFileSync(path.join(parent,'pairs',pid+'.json')));
});
test('bias report captures all twelve shorter Core/balance edits without treating flags as decisions',()=>{
 const r=JSON.parse(fs.readFileSync(path.join(out,'SHORTCUT_BIAS_AUDIT.json'),'utf8'));
 assert.equal(r.shorter,12);assert.equal(r.longer,3);assert.equal(r.equal,1);
 const review=r.pairs.filter(x=>x.category!=='grounding_error');assert.equal(review.length,12);assert(review.every(x=>x.delta<0));
 const balance=r.pairs.filter(x=>x.category==='same_aspect_disagreement_loss');assert.equal(balance.length,6);
 assert(balance.every(x=>x.inserted_wording_flags.length>0));assert(r.pairs.every(x=>x.researcher_decision==='PENDING'));
});
test('standalone HTML has no external assets or network API; no execution authorization controls',()=>{
 assert(markup.includes("connect-src 'none'"));assert(!markup.match(/<script[^>]+src=/));
 assert(!script.match(/\b(?:fetch|XMLHttpRequest|WebSocket|eval)\s*\(/));
 assert(!markup.includes('http://'));assert(!markup.includes('https://'));
 assert(markup.includes('UNVERIFIED'));assert(markup.includes('Original CE epoch 3'));
});
process.stdout.write('RESULT '+passed+' focused CPU/interface tests PASS.\nDOM fixture only; browser layout and file-download integration not verified by this test.\nNo model imports, GPU calls, optimizer updates or actual researcher decisions.\n');
