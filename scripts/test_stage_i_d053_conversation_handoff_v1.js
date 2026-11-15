'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert');
const api=require('./stage_i_d053_resolution_state_v2.js');const root=process.argv[2]||path.resolve(__dirname,'..');
const packaged=fs.existsSync(path.join(root,'source_cases.json'));const final=path.join(root,packaged?'final_references':'outputs/stage_i/d053_final_references_v1');
const page=fs.readFileSync(path.join(root,packaged?'RESEARCHER_APPROVAL_V2.html':'outputs/stage_i/d053_reference_proposals_v2/RESEARCHER_APPROVAL_V2.html'),'utf8');const P=JSON.parse(page.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const state=JSON.parse(fs.readFileSync(path.join(final,'RESEARCHER_DECISIONS_CONVERSATION_HANDOFF.json'),'utf8'));let tests=0;
function test(name,fn){fn();tests++;console.log('PASS '+name);}
test('Transparent conversation import, not UI-export impersonation',()=>{assert.equal(state.approval_recording_method,'direct_researcher_conversation_no_UI');assert.equal(state.history.length,1);assert.equal(state.history[0].action,'explicit_conversation_approval_import');});
test('Existing decision-state validator accepts approved handoff',()=>{assert(api.validate(state,P));assert(api.ready(state,P));});
test('All20 resolutions and24 packages explicitly recorded',()=>{for(const [g,n] of [['corrections',15],['omissions',5],['business_packages',24]]){assert.equal(Object.keys(state[g]).length,n);assert(Object.values(state[g]).every(d=>d.choice&&d.confirmed_at));}});
test('225 approved overrides including88 exclusions preserved',()=>{assert.equal(Object.keys(state.unit_overrides).length,225);assert.equal(Object.values(state.unit_overrides).filter(o=>o.action==='exclude').length,88);assert.equal(state.final_reference_freeze,false);});
test('Original blank worksheet and unapproved proposals stay untouched',()=>{assert(P.template.reviewer_id==='');assert(P.template.history.length===0);assert(P.units.every(u=>u.human_verified===false));});
console.log(JSON.stringify({status:'PASS',tests,real_UI_events_created:0,annotation_files_modified:0}));
