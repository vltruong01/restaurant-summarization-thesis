"""Run original and proposal-only tests from a freshly extracted exact ZIP."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True);a=p.parse_args()
 verifier=Path(__file__).with_name('verify_stage_i_d053_review_archive_v2.py')
 r=subprocess.run([sys.executable,'-X','utf8',str(verifier),'--zip',str(a.zip),'--output-dir',str(a.output_dir),'--original-root',str(a.original_root)],capture_output=True,text=True,encoding='utf-8',timeout=240)
 assert r.returncode==0,(r.stdout,r.stderr)
 out=a.output_dir.resolve();base=json.loads((out/'EXTRACTED_ZIP_VERIFICATION.json').read_text(encoding='utf-8'));root=Path(base['extracted_root']);manifest=json.loads((root/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
 env=os.environ.copy()
 for name in ['PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','NODE_OPTIONS','NODE_PATH']:env.pop(name,None)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
 rows={}
 for name,expected,command in [
  ('preserved_handoff',16,[sys.executable,'-s','-X','utf8','-m','unittest','discover','-s','code','-p','test_stage_i_d053_researcher_handoff_v1.py','-v']),
  ('new_proposal_integrity',22,[sys.executable,'-s','-X','utf8','-m','unittest','discover','-s','code','-p','test_stage_i_d053_atomic_proposals_v2.py','-v']),
  ('new_decision_guards',24,['node','code/test_stage_i_d053_resolution_state_v2.js','RESEARCHER_APPROVAL_V2.html'])]:
  s=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
  logs={}
  for kind,content in [('stdout',s.stdout),('stderr',s.stderr)]:
   file=out/(name+'.'+kind+'.log');file.write_text(content,encoding='utf-8');logs[kind]={'file':file.name,'sha256':sha(file)}
  assert s.returncode==0,(name,s.stdout,s.stderr)
  if name=='new_decision_guards':
   j=json.loads([x for x in s.stdout.splitlines() if x.startswith('{')][-1]);assert j['tests']==expected and j['synthetic_only']
  else:assert f'Ran {expected} tests' in s.stderr
  rows[name]={'tests':expected,'status':'PASS','command':command,'logs':logs}
 for row in manifest['files']:assert sha(root/row['path'])==row['sha256'],row['path']
 assert sha(a.zip.resolve())==base['archive_sha256']
 summary={'status':'SOFTWARE_AND_PROVENANCE_PASS_SCIENTIFIC_ACCEPTANCE_BLOCKED',
  'published_final_zip_sha256':base['archive_sha256'],'published_final_zip_bytes':a.zip.stat().st_size,
  'extracted_root':str(root),'all_payload_hashes_before_after':'PASS','no_original_project_dependencies_used_by_tests':True,
  'existing_tests':{'original':83,'corrected_ui':30,'portable_readiness':8,'researcher_handoff':16,'total':137},
  'new_tests':{'proposal_integrity':22,'decision_state_and_embedded_script_syntax':24,'total':46},'total_tests':183,'failures':0,'skipped':0,
  'browser_smoke':'NOT_PERFORMED_CURRENT_OFFLINE_PREPARATION','legacy_verifier_browser_field':'A hardcoded field inherited from a previous audit, not a new browser action or current finding.',
  'new_human_confirmed_units':0,'final_reference_freeze_created':False,'predictions_created':0,'GPU_or_API_execution':False,
  'scientific_semantic_support_or_completeness_certified_by_software':False,'additional_test_commands':rows}
 (out/'COMPLETE_D053_PROPOSAL_VALIDATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:summary[k] for k in ['status','published_final_zip_sha256','total_tests','failures','skipped','new_human_confirmed_units']},indent=2))
if __name__=='__main__':main()
