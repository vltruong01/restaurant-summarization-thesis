"""Offline revalidation of exact approved references; no model/API execution."""
from pathlib import Path
import argparse,hashlib,json
import jsonschema
from stage_i_d053_annotations import validate_annotations
from stage_i_d053_curated_references_v2 import validate_curated_annotations,freeze_curated_annotations
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_context(root):
 root=Path(root)
 if (root/'source_cases.json').exists():
  source=root;final=root/'final_references';prop=root/'reference_proposals';old=root/'researcher_finalization';schema=root/'schemas'
 else:
  final=root/'outputs/stage_i/d053_final_references_v1';prop=root/'outputs/stage_i/d053_reference_proposals_v2';old=root/'outputs/stage_i/d053_researcher_reference_finalization_v1';source=old/'source_bundle/stage_i_d053_reference_verification_chatgpt_bundle_v2';schema=root/'schemas'
 return {'root':root,'source':source,'final':final,'prop':prop,'old':old,'schema':schema,
  'state':read(final/'FINAL_REFERENCE_ANNOTATIONS.json'),'cases':read(source/'source_cases.json'),
  'candidates':[json.loads(s) for s in (source/'candidates_UNVERIFIED.jsonl').read_text(encoding='utf-8').splitlines()],
  'bindings':read(source/'bindings.json')['bindings'],'proposals':read(prop/'PROPOSED_ATOMIC_REFERENCE_UNITS.json')['units'],
  'curation':read(final/'incoming/extracted/CURATED_PRIORITY_RECOMMENDATIONS_NOT_APPROVED.json'),
  'approval':read(final/'RESEARCHER_APPROVAL_RECORD.json'),'prior':read(old/'RESEARCHER_APPROVED_DECISIONS.json'),
  'prior_annotations':read(old/'REFERENCE_ANNOTATIONS_DRAFT_NOT_FROZEN.json')['candidate_annotations']}
def arguments(c):return [c[k] for k in ['state','cases','candidates','bindings','proposals','curation','approval','prior','prior_annotations']]
def check(root):
 c=load_context(root);source=c['source'];final=c['final'];checks=[]
 for p,key in [(source/'source_cases.json','source_cases_sha256'),(source/'candidates_UNVERIFIED.jsonl','candidate_sha256'),(source/'D049_FROZEN_EVALUATION_PROTOCOL.md','protocol_sha256'),(source/'FROZEN_EVALUATION_CONFIG.json','evaluation_config_sha256')]:
  assert sha(p)==c['bindings'][key],p;checks.append(p.name)
 version=read(c['prop']/'PROPOSAL_VERSION.json');assert version['proposal_id']==c['curation']['source_proposal_id']
 for name,digest in version['files'].items():assert sha(c['prop']/name)==digest;checks.append(name)
 assert sha(c['old']/'RESEARCHER_APPROVED_DECISIONS.json')==c['approval']['prior_226_candidate_ledger_sha256']
 assert sha(final/'incoming/extracted/CURATED_PRIORITY_RECOMMENDATIONS_NOT_APPROVED.json')==c['approval']['curated_recommendations_file_sha256']
 archive=next((final/'incoming').glob('*.zip'));assert sha(archive)==c['approval']['input_archive_sha256']
 assert sha(final/'PRESERVED_226_PARENT_DECISIONS.json')==sha(c['old']/'RESEARCHER_APPROVED_DECISIONS.json')
 jsonschema.Draft202012Validator(read(source/'schemas/stage_i_d053_annotation_v1.schema.json')).validate(c['state'])
 jsonschema.Draft202012Validator(read(c['schema']/'stage_i_d053_curated_exclusion_extension_v2.schema.json')).validate(c['state']['curated_reference_policy'])
 result=validate_curated_annotations(*arguments(c));assert result['valid'],result
 assert freeze_curated_annotations(*arguments(c))==read(final/'FINAL_RESEARCHER_APPROVED_REFERENCES.json'),'Frozen reference bytes/content do not match accepted state and approval'
 for case in c['cases']:
  for cap in case['captions']:assert sha(source/cap['image_file'])==sha(final/'incoming/extracted'/cap['image_file'])
 original=validate_annotations(c['state'],c['cases'],c['candidates'],c['bindings'],final=True)
 assert not original['valid'] and set(original['errors'])==set(result['legacy_pending_resolved_only_by_explicit_all_child_exclusion'])
 manifest=final/'REFERENCE_FREEZE_MANIFEST.json'
 if (Path(root)/'source_cases.json').exists():assert manifest.exists(),'Missing frozen reference manifest'
 if manifest.exists():
  frozen_manifest=read(manifest)
  for row in frozen_manifest['files']:assert sha(final/row['path'])==row['sha256'] and (final/row['path']).stat().st_size==row['bytes'],row['path']
  code_dir=Path(root)/('code' if (Path(root)/'source_cases.json').exists() else 'scripts')
  for row in frozen_manifest['construction_and_validation_code']:assert sha(code_dir/Path(row['path']).name)==row['sha256'],row['path']
  assert sha(c['schema']/'stage_i_d053_curated_exclusion_extension_v2.schema.json')==frozen_manifest['supplemental_schema_sha256']
  assert sha(source/'code/stage_i_d053_annotations.py')==frozen_manifest['unchanged_original_validator_sha256']
 return {**result,'status':'FINAL_REFERENCES_PASS_V2_EXPLICIT_ADMISSIBILITY','unchanged_v1_final_status':'EXPECTED_BLOCK_24_SUPPORTED_PARENTS_ALL_CHILDREN_APPROVED_NON_SCORING','schema_v1_structural':'PASS','extension_schema_v2':'PASS','reference_freeze_manifest_checked':manifest.exists(),'source_and_proposal_binding_files_checked':checks,'photo_original_bytes_checked':90,'new_inference_or_training_authorized':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args();result=check(a.root)
 if a.output:
  with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['status','counts','schema_v1_structural','extension_schema_v2','legacy_original_pending_count']},indent=2))
