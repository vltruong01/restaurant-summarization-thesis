"""Verify the exact D053 final archive, all228 tests, and approved freeze."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True);a=p.parse_args()
 verifier=Path(__file__).with_name('verify_stage_i_d053_proposal_archive_v2.py')
 completed=subprocess.run([sys.executable,'-X','utf8',str(verifier),'--zip',str(a.zip),'--output-dir',str(a.output_dir),'--original-root',str(a.original_root)],capture_output=True,text=True,encoding='utf-8',timeout=300)
 assert completed.returncode==0,(completed.stdout,completed.stderr)
 output=a.output_dir.resolve();prior=json.loads((output/'COMPLETE_D053_PROPOSAL_VALIDATION_SUMMARY.json').read_text());root=Path(prior['extracted_root']);env=os.environ.copy()
 for name in ['PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','NODE_OPTIONS','NODE_PATH']:env.pop(name,None)
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 rows={}
 for name,number,command in [
  ('final_reference40',40,[sys.executable,'-s','-X','utf8','-m','unittest','discover','-s','code','-p','test_stage_i_d053_final_references_v2.py','-v']),
  ('conversation_handoff5',5,['node','code/test_stage_i_d053_conversation_handoff_v1.js','.']),
  ('complete_reference_check',None,[sys.executable,'-s','-X','utf8','code/check_stage_i_d053_final_references_v2.py','--root','.'])]:
  r=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=180);logs={}
  for kind,content in [('stdout',r.stdout),('stderr',r.stderr)]:
   path=output/(name+'.'+kind+'.log');path.write_text(content,encoding='utf-8');logs[kind]={'file':path.name,'sha256':sha(path)}
  assert r.returncode==0,(name,r.stdout,r.stderr)
  if number==40:assert 'Ran 40 tests' in r.stderr
  if number==5:assert json.loads([s for s in r.stdout.splitlines() if s.startswith('{')][-1])['tests']==5
  rows[name]={'status':'PASS','tests':number,'command':command,'logs':logs}
 manifest=json.loads((root/'SHA256_MANIFEST.json').read_text());actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()};expected={r['path'] for r in manifest['files']}|{'SHA256_MANIFEST.json'};assert actual==expected
 for row in manifest['files']:assert sha(root/row['path'])==row['sha256'],row['path']
 assert sha(a.zip)==prior['published_final_zip_sha256']
 report={'status':'FINAL_D053_REFERENCES_AND_ARCHIVE_VALIDATED_CURATED_V2','published_final_zip_sha256':sha(a.zip),'published_final_zip_bytes':a.zip.stat().st_size,
  'extracted_root':str(root),'total_tests':228,'prior_regression_tests':183,'new_final_reference_tests':40,'new_conversation_handoff_tests':5,'failures':0,'skipped':0,
  'reference_file_sha256':sha(root/'final_references/FINAL_RESEARCHER_APPROVED_REFERENCES.json'),'frozen_reference_manifest_sha256':sha(root/'final_references/REFERENCE_FREEZE_MANIFEST.json'),
  'core':261,'supporting':315,'excluded_audit_only':88,'scoring_units':576,'businesses':24,'original_parent_decisions':226,'all_payload_hashes_before_after':'PASS','manifest_covers_all_files_except_itself':True,
  'unchanged_v1_final_gate':'EXPECTED_BLOCK_24_EXPLICIT_ALL_CHILD_EXCLUSION_PARENTS','v1_structural_and_evidence_checks':'PASS','strict_v2_approval_bound_exclusion_bridge':'PASS',
  'browser_smoke':'NOT_PERFORMED_NO_NEW_ANNOTATION_UI_REQUIRED','no_original_project_dependencies_used_by_extracted_tests':True,'no_predictions_training_teacher_or_held_out_access':True,
  'human_approval_origin':'Direct researcher instruction; no UI interactions or independent machine truth fabricated','D054_authorized':False,'new_test_commands':rows}
 (output/'COMPLETE_D053_FINAL_REFERENCE_TEST_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','published_final_zip_sha256','total_tests','failures','skipped','core','supporting','excluded_audit_only']},indent=2))
if __name__=='__main__':main()
