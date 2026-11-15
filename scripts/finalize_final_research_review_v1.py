"""Verify, preserve and package derived research discussion. No original mutations."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.metadata,json,math,re,shutil,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'outputs/final_research_analysis_20261106_v1'
P=TASK/'PACKAGE'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def command(args):
 r=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],capture_output=True,text=True,encoding='utf8',check=True)
 return dict(command=[str(x) for x in args],exit_code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
def manifest():
 files=[dict(file=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f!=P/'PACKAGE_MANIFEST.json']
 save(P/'PACKAGE_MANIFEST.json',dict(format='final_research_review_manifest_v1',hash='SHA-256',excludes='This manifest itself; archive hash is in the external sidecar',files=files))
def archive(path):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for f in sorted(P.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(P).as_posix())
def extract_checked(path,dest):
 assert not dest.exists();dest.mkdir()
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names))
  assert z.testzip() is None
  for name in names:
   target=(dest/name).resolve();assert target.is_relative_to(dest.resolve()) and not name.endswith('/')
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
   assert sha(target)==sha(P/name),name
 return len(names)
def main():
 final=TASK/'FINAL_RESEARCH_RESULTS_AND_DISCUSSION_REVIEW.zip';assert not final.exists(),'Do not overwrite a released review archive'
 s=P/'SCRIPTS';s.mkdir(exist_ok=True)
 scripts={'verify_integrated_research_statistics_v1.py':'verify_integrated_research_statistics_v1.py','verify_review_package.py':'verify_final_research_review_package_v1.py','render_final_research_figures_v1.py':'render_final_research_figures_v1.py','build_final_research_discussion_v1.py':'build_final_research_discussion_v1.py','prepare_final_research_analysis_v1.py':'prepare_final_research_analysis_v1.py','finalize_final_research_review_v1.py':'finalize_final_research_review_v1.py'}
 for name,source in scripts.items():shutil.copy2(ROOT/'scripts'/source,s/name)
 (P/'requirements.txt').write_text('numpy==1.26.4\nmatplotlib=='+importlib.metadata.version('matplotlib')+'\n',encoding='utf8')
 inv=read(P/'SOURCE_INVENTORY.json');preserved=[]
 for row in inv['archives']:
  source=ROOT/row['original_archive'];h=sha(source);assert h==row['sha256']
  preserved.append(dict(file=row['original_archive'],sha256=h,status='UNCHANGED'))
 for row in inv['documents']:
  source=Path(row['original_file']);h=sha(source);assert h==row['sha256']
  preserved.append(dict(file=row['original_file'],sha256=h,status='UNCHANGED'))
 for row in inv['member_copies']:
  assert sha(P/row['package_file'])==row['sha256']
  if 'original_file' in row:assert sha(ROOT/row['original_file'])==row['sha256']
 for row in inv['protected_excluded_artifacts']:
  source=ROOT/row['file'];h=sha(source);assert h==row['sha256']
  preserved.append(dict(file=row['file'],sha256=h,status='UNCHANGED_AND_EXCLUDED_FROM_ANALYSIS'))
 save(P/'SOURCE_PRESERVATION.json',dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),protected=preserved,source_copies_verified=len(inv['member_copies']),scope='Selected inputs and already recorded protected artifacts,not a new exhaustive hash of the whole project or model binaries',original_references_predictions_ratings_and_Word_written=False))
 # Cross-check every rounded report table cell and selected inferential assertions against frozen-derived values.
 pairs=read(P/'VERIFIED_TABLES/P_PAIRED_COMPARISONS.json');primary=[r for r in pairs if r['left']=='LED F original' and r['right']=='Qwen-Caption']
 assert sum(r['holm_p_within_family']<.05 for r in primary)==5
 visual=next(r for r in pairs if r['left']=='Qwen-Caption' and r['criterion']=='visual_grounding_and_contribution')
 assert math.isclose(visual['holm_p_within_family'],.1478271484375) and visual['raw_exact_signflip_p']<.05
 means=read(P/'VERIFIED_TABLES/P_DESCRIPTIVES.json');assert len(means)==54
 text=(P/'FINAL_RESULTS_ANALYSIS.md').read_text(encoding='utf8')
 for row in means:
  if row['panel']=='COMBINED':assert f"{row['mean']:.3f} ± {row['sample_SD']:.3f}" in text
 for stage in ['N','O']:
  agreed=read(P/f'VERIFIED_TABLES/{stage}_AGREEMENT_VERIFIED.json');values=[r['quadratic_weighted_kappa'] for r in agreed if r['quadratic_weighted_kappa'] is not None]
  if stage=='N':assert round(min(values),3)==.460 and round(max(values),3)==.952
  else:assert round(min(values),3)==0 and round(max(values),3)==.904
 params=read(P/'SOURCES/TRAINING/LORA_TRAINABLE_PREFLIGHT.json')['scope']
 assert round(100*params['trainable_parameters']/params['frozen_backbone_parameters'],3)==1.069
 native=read(P/'SOURCES/TRAINING/F_S1000_COMPLETION_NATIVE.json')['completion']
 assert round(native['seconds'],3)==7465.886 and round(native['peak_allocated_bytes']/2**30,3)==3.414 and round(native['peak_reserved_bytes']/2**30,3)==5.189
 # Machine checks never turn qualitative annotations into independent confirmations.
 with (P/'VERIFIED_TABLES/P_PARTICIPANT_FLAG_COUNTS.csv').open(encoding='utf8') as stream:flags=list(csv.DictReader(stream))
 assert sum(int(r['serious_flags']) for r in flags)==17 and sum(int(r['nonempty_comments']) for r in flags)==90
 link_count=0;json_count=0;csv_count=0
 for file in P.rglob('*'):
  if not file.is_file():continue
  assert file.suffix.lower() not in {'.db','.sqlite','.sqlite3','.pt','.safetensors','.jpg','.jpeg'}
  if file.suffix=='.json':read(file);json_count+=1
  if file.suffix=='.csv':
   with file.open(encoding='utf-8-sig',newline='') as stream:
    rows=list(csv.reader(stream));assert rows and all(len(r)==len(rows[0]) for r in rows);csv_count+=1
  if file.parent==P and file.suffix=='.md':
   for relative in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf8')):
    assert not relative.startswith(('http:','https:','file:'))
    assert (file.parent/relative).is_file(),(file,relative);link_count+=1
  if file.suffix in {'.json','.csv','.md','.py','.txt'}:
   content=file.read_text(encoding='utf-8-sig')
   assert not re.search(r'\b(?:hf_[a-zA-Z0-9]{24,}|sk-(?:proj-)?[a-zA-Z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b',content),'Potential credential pattern: '+str(file)
 report=dict(status='PASS_NUMERICAL_AND_PACKAGE_CHECKS_WITH_DISCLOSED_LIMITATIONS',checked_at_utc=datetime.now(timezone.utc).isoformat(),statistics=read(P/'VERIFIED_TABLES/VERIFICATION_RESULTS.json'),P_collection_bindings_and_journal=read(P/'VERIFIED_TABLES/P_ORIGINAL_COLLECTION_VERIFICATION.json'),original_source_preservation='PASS',json_files_parsed=json_count,csv_files_parsed=csv_count,root_report_links_checked=link_count,static_figures_visually_inspected=['FIGURES/HUMAN_MEANS_SEPARATE_STAGES.png','FIGURES/P_PRIMARY_PAIRED_INTERVALS.png'],credential_pattern_scan='PASS within documented recognizable-token and excluded-file checks;not a universal security certification',external_ChatGPT_analysis_archive='NOT PROVIDED / NOT VERIFIED;authorized independent reproduction used instead',photo_semantic_audit='NOT PERFORMED;participant claims not automatically confirmed',human_identity_authentication='NOT ESTABLISHED',new_model_calls=0,new_ratings=0,portable_replay='NOT_YET_RUN')
 save(P/'VALIDATION_REPORT.json',report);manifest()
 probe=TASK/'PORTABILITY_PROBE_V2.zip';assert not probe.exists();archive(probe)
 extracted=TASK/'PORTABILITY_PROBE_V2_EXTRACTED';count=extract_checked(probe,extracted)
 rootcheck=command([extracted/'SCRIPTS/verify_review_package.py','--package',extracted])
 replay_dir=TASK/'PORTABLE_FULL_STATISTICAL_REPLAY'
 replay=command([extracted/'SCRIPTS/verify_integrated_research_statistics_v1.py','--package',extracted,'--output',replay_dir])
 compared=[]
 for original in (P/'VERIFIED_TABLES').glob('*.json'):
  new=replay_dir/original.name
  if new.exists():assert read(new)==read(original),original.name;compared.append(original.name)
 assert len(compared)>=12
 report['portable_replay']=dict(status='PASS',probe_archive_members_checked=count,package_verifier=rootcheck,independent_statistics_replay=replay,numerical_JSON_tables_equal=compared,scope='Full CPU arithmetic replay from extracted small package,no original model or dataset dependency')
 save(P/'VALIDATION_REPORT.json',report);manifest();archive(final)
 final_extracted=TASK/'FINAL_ARCHIVE_EXTRACTED';count=extract_checked(final,final_extracted)
 finalcheck=command([final_extracted/'SCRIPTS/verify_review_package.py','--package',final_extracted])
 digest=sha(final);(TASK/(final.name+'.sha256')).write_text(digest+'  '+final.name+'\n',encoding='ascii')
 receipt=dict(status='PASS',archive=final.name,bytes=final.stat().st_size,sha256=digest,archive_reopened_and_extracted=True,all_members_byte_matched_to_packaged_files=count,final_extracted_package_verifier=finalcheck,portable_full_statistical_replay='PASS',source_preservation='PASS',no_new_experiments=True)
 save(TASK/'DELIVERY_VERIFICATION.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
