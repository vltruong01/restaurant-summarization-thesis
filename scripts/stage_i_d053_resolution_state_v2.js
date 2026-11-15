/* New decision worksheet only. Never edits original candidate labels/references. */
(function(root){
 'use strict';
 const clone=x=>JSON.parse(JSON.stringify(x));
 function equal(a,b){return JSON.stringify(a)===JSON.stringify(b);}
 function create(payload){return clone(payload.template);}
 function validate(s,p){
  if(s.schema_version!==p.template.schema_version||s.proposal_id!==p.version.proposal_id||!equal(s.proposal_files,p.version.files))throw Error('Wrong proposal version or file hashes');
  for(const group of ['corrections','omissions','business_packages']){
   if(!equal(Object.keys(s[group]||{}).sort(),Object.keys(p.template[group]).sort()))throw Error('Unknown or missing decision IDs');
   for(const d of Object.values(s[group])){
    const allowed=group==='corrections'?[null,'include_caption_safe_revision','exclude']:group==='omissions'?[null,'core','supporting','excluded']:[null,'approve_proposals_as_displayed'];
    if(!allowed.includes(d.choice)||typeof d.notes!=='string'||(d.choice&&!d.confirmed_at))throw Error('Invalid decision or missing timestamp');
   }
  }
  if(!s.unit_overrides||!Array.isArray(s.history)||typeof s.reviewer_id!=='string'||typeof s.researcher_attestation!=='boolean')throw Error('Invalid worksheet');
  const ids=new Set(p.units.map(u=>u.proposal_id));
  for(const [id,o] of Object.entries(s.unit_overrides)){
   if(!ids.has(id)||!['include','exclude','edit'].includes(o.action)||typeof o.notes!=='string'||!o.confirmed_at)throw Error('Invalid unit override');
   if(o.action==='edit'&&(typeof o.text!=='string'||!o.text.trim()))throw Error('Edited wording is empty');
   if(o.metadata){let m=o.metadata;if(!['core','supporting'].includes(m.importance)||!['useful_specific','useful_general','low_information'].includes(m.informativeness)||typeof m.aspect!=='string'||!m.aspect.trim()||typeof m.restaurant_specific!=='boolean')throw Error('Invalid metadata');
    let u=p.units.find(x=>x.proposal_id===id);if(u.domain==='visual'?m.polarity!==null:!['positive','negative','mixed','neutral'].includes(m.polarity))throw Error('Invalid polarity');}
  }
  if(s.final_reference_freeze!==false||s.new_human_confirmed_unit_count!==0)throw Error('This worksheet cannot freeze or create reference units');
  return true;
 }
 function ready(s,p){
  validate(s,p);return !!s.reviewer_id.trim()&&s.researcher_attestation&&['corrections','omissions','business_packages'].every(g=>Object.values(s[g]).every(x=>x.choice&&x.confirmed_at));
 }
 function identity(s){if(!s.reviewer_id.trim())throw Error('Enter your researcher identity before confirming');}
 function record(s,action,id,value){s.history.push({action,id,value:clone(value),researcher:s.reviewer_id,recorded_at:new Date().toISOString()});}
 function decide(s,p,group,id,choice,notes=''){
  identity(s);if(!Object.hasOwn(s[group]||{},id))throw Error('Unknown decision ID');
  const trial=clone(s);trial[group][id]={choice,notes,confirmed_at:choice?new Date().toISOString():null};validate(trial,p);
  s[group][id]=trial[group][id];record(s,group,id,s[group][id]);
  if(group!=='business_packages')for(const b of Object.values(s.business_packages)){b.choice=null;b.confirmed_at=null;}
 }
 function override(s,p,id,o){
  identity(s);const trial=clone(s);trial.unit_overrides[id]={...clone(o),confirmed_at:new Date().toISOString()};validate(trial,p);
  s.unit_overrides[id]=trial.unit_overrides[id];const bid=p.units.find(u=>u.proposal_id===id).business_id;
  s.business_packages[bid].choice=null;s.business_packages[bid].confirmed_at=null;record(s,'unit_override',id,s.unit_overrides[id]);
 }
 function approveAll(s,p){
  identity(s);if(!s.researcher_attestation)throw Error('Confirm that you reviewed all proposals/metadata/evidence changes first');
  if(!['corrections','omissions'].every(g=>Object.values(s[g]).every(d=>d.choice)))throw Error('Resolve the 20 scientific decisions first');
  for(const id of Object.keys(s.business_packages))decide(s,p,'business_packages',id,'approve_proposals_as_displayed',s.business_packages[id].notes);
 }
 function exportState(s,p,final=false){validate(s,p);if(final&&!ready(s,p))throw Error('Incomplete decisions: approval export blocked');return {...clone(s),export_kind:final?'RESEARCHER_DECISION_HANDOFF_NOT_FINAL_REFERENCES':'PROGRESS_ONLY',exported_at:new Date().toISOString()};}
 const api={create,validate,ready,decide,override,approveAll,exportState};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.D053Resolution=api;
})(typeof window!=='undefined'?window:globalThis);
