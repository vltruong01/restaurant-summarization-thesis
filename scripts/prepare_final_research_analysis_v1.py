"""Copy only authorized frozen evidence to a separate final-analysis workspace."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'outputs/final_research_analysis_20261106_v1'
P=DEST/'PACKAGE'
ARCHIVES={
'M':('outputs/stage_m/final_test100_evaluation_review_20261101_v1/STAGE_M_FINAL_TEST100_EVALUATION_REVIEW.zip',None),
'N':('outputs/stage_n/final_human_statistical_analysis_20261102_v1/STAGE_N_FINAL_HUMAN_STATISTICAL_REVIEW.zip',None),
'O':('outputs/stage_o/final_human_statistical_analysis_20261106_v1/STAGE_O_FINAL_STATISTICAL_REVIEW.zip',None),
'P':('outputs/stage_p/v1_original_recovered_20261106_v1/STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip','e2f3715d522f1d9505075f2f79c0d394880cc9dbb8462579b99c41ca82f28fc5')}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 assert not DEST.exists(),'Use a new final-analysis folder; never overwrite prior results'
 (P/'SOURCES').mkdir(parents=True);receipts=[];copies=[]
 selections={
 'M':['RESULTS_ORIGINAL/AGGREGATE_METRICS.csv','RESULTS_ORIGINAL/PAIRED_COMPARISONS.csv','RESULTS_ORIGINAL/PAIRED_COMPARISONS.json','RESULTS_ORIGINAL/PER_BUSINESS_COMPONENT_METRICS.csv','RESULTS_ORIGINAL/STRUCTURE_REPETITION_LENGTH.md','RESULTS_ORIGINAL/STAGE_M_PRIMARY_REPORT.md','RESULTS_ORIGINAL/COMPLETION.json','RESULTS_ORIGINAL/RESULT_INTEGRITY_VERIFICATION.json','SUMMARY/METHODOLOGY.md','REFERENCE_PROVENANCE/AI_ASSISTANCE_AND_LIMITATIONS.md','PROTOCOL/AUTOMATIC_EVALUATION_PROTOCOL.md','PROTOCOL/METRIC_LOCK.json','PROTOCOL/MODEL_AND_DECODER_LOCK.json','PROTOCOL/SHUFFLED_POLICY.json','PROTOCOL/COHORT100_FREEZE.json'],
 'N':['ANALYSIS_CONFIG.json','DATA/VERIFIED_ANALYSIS_INPUT.json','DATA/ORIGINAL_RATINGS_240.json','TABLES/DESCRIPTIVE_STATISTICS.json','TABLES/AGREEMENT.json','TABLES/PAIRED_COMPARISONS.json','METHODOLOGY.md','PROVENANCE/RESEARCHER_ACCEPTANCE_AS_RECORDED.json','PROVENANCE/MODEL_ASSISTED_AUDIT_SUMMARY.json','PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md','PROVENANCE/RUBRIC.json'],
 'O':['ANALYSIS_CONFIG.json','COLLECTION/STAGE_O_TRIPADVISOR15_RATINGS_FINAL_90.json','DATA/ORIGINAL_PRIVATE_ITEMS_90.json','DATA/ORIGINAL_OUTPUTS_45.json','TABLES/DESCRIPTIVES.json','TABLES/AGREEMENT.json','TABLES/PAIRED_COMPARISONS.json','METHODOLOGY_AND_LIMITATIONS.md','SUMMARY.md','COLLECTION/PROVENANCE/RUBRIC.json','COLLECTION/PROVENANCE/TWO_RATER_PROTOCOL_AMENDMENT.md'],
 'P':None}
 for stage,(relative,expected) in ARCHIVES.items():
  archive=ROOT/relative;h=sha(archive)
  if expected:assert h==expected
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None
   names=z.namelist();assert len(names)==len(set(names))
   manifest=json.loads(z.read('PACKAGE_MANIFEST.json'));data=manifest.get('files',manifest.get('members'))
   entries=[dict(file=name,**info) for name,info in data.items()] if isinstance(data,dict) else data
   for r in entries:
    raw=z.read(r['file']);assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],r['file']
   chosen=names if stage=='P' else selections[stage]
   for name in chosen:
    if name.endswith('/'):continue
    target=(P/'SOURCES'/stage/name).resolve();assert target.is_relative_to((P/'SOURCES'/stage).resolve())
    target.parent.mkdir(parents=True,exist_ok=True);raw=z.read(name);target.write_bytes(raw)
    copies.append(dict(original_archive=relative,member=name,package_file=target.relative_to(P).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  receipts.append(dict(stage=stage,original_archive=relative,sha256=h,archive_members=len(names),manifest_members_checked=len(entries),crc='PASS',member_hashes='PASS'))
 for name in ['STAGE_P_EFFICIENCY_AUDIT.md','EFFICIENCY_COMPARISON.md','MISSING_MEASUREMENTS.md','EFFICIENCY_MEASUREMENTS.csv','VERIFIED_FACTS.json','SOURCE_PRESERVATION.json']:
  source=ROOT/'outputs/stage_p/fair_efficiency_audit_20261106_v1/PACKAGE'/name
  target=P/'SOURCES/EFFICIENCY'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  copies.append(dict(original_file=source.relative_to(ROOT).as_posix(),package_file=target.relative_to(P).as_posix(),sha256=sha(source),bytes=source.stat().st_size))
 # Preserve exact research-question wording and reference organization,not the Word file.
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};documents=[]
 for source,tag in [(ROOT/'thesis_proposal_v16.docx','APPROVED_PROPOSAL'),(Path(r'C:\Users\VUONGLOCTRUONG\Downloads\Thesis_Research_Progress_v3.docx'),'STYLE_REFERENCE')]:
  with zipfile.ZipFile(source) as z:
   tree=ET.fromstring(z.read('word/document.xml'));paragraphs=[]
   for i,para in enumerate(tree.findall('.//w:body/w:p',ns)):
    text=''.join(n.text or '' for n in para.findall('.//w:t',ns));style=para.find('w:pPr/w:pStyle',ns)
    if text.strip():paragraphs.append(dict(paragraph_index=i,text=text,style_id=style.get('{'+ns['w']+'}val') if style is not None else ''))
   if tag=='APPROVED_PROPOSAL':
    index=next(i for i,r in enumerate(paragraphs) if r['text']=='Research Questions');selection=paragraphs[index:index+6]
   else:selection=[r for r in paragraphs if any(r['text'].startswith(str(i)+'.') for i in range(12,22)) or any(r['text'].startswith(str(i)+'.'+str(j)) for i in range(12,22) for j in range(1,6))]
   save(P/'SOURCES'/tag/'DOCUMENT_EXTRACT.json',dict(source_file=str(source),source_sha256=sha(source),paragraphs=selection,scope='Read-only semantic/style extraction;not a rendered Word layout validation'))
   documents.append(dict(role=tag,original_file=str(source),sha256=sha(source)))
   if tag=='APPROVED_PROPOSAL':save(P/'SOURCES/RESEARCH_QUESTIONS.json',[r['text'] for r in selection if r['text'].startswith(('To what extent','Does filtering','How computationally'))])
 # Actual LoRA scope and one accepted1000-update training segment;no weight loading.
 preflight=ROOT/'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1/downloaded/RUN/LINUX_TRAINABLE_LOAD_PREFLIGHT.json'
 native=ROOT/'outputs/stage_i/led_large_source_separated_v1_two_epoch_preparation_v1/MILESTONE_1000_NATIVE_STATUS.json'
 for source,name in [(preflight,'LORA_TRAINABLE_PREFLIGHT.json'),(native,'F_S1000_COMPLETION_NATIVE.json')]:
  target=P/'SOURCES/TRAINING'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  copies.append(dict(original_file=source.relative_to(ROOT).as_posix(),package_file=target.relative_to(P).as_posix(),sha256=sha(source),bytes=source.stat().st_size))
 train=ROOT/'outputs/cloud_handoff/LED_LARGE_RO_U_CE3_DIRECT750_TRAINING_RESULTS_REVIEW_V1.zip'
 with zipfile.ZipFile(train) as z:
  for name in ['closeout/TRAINING_COMPLETION_REPORT.md','frozen_expected/INITIALIZATION_HASHES.json']:
   target=P/'SOURCES/TRAINING'/Path(name).name;raw=z.read(name);target.write_bytes(raw)
   copies.append(dict(original_archive=train.relative_to(ROOT).as_posix(),member=name,package_file=target.relative_to(P).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 receipts.append(dict(stage='Training provenance',original_archive=train.relative_to(ROOT).as_posix(),sha256=sha(train),verification='Selected provenance members copied;weights not read'))
 # Source-input hashes only;never export current/exploratory ratings into official data.
 protected=[]
 for file in [ROOT/'outputs/stage_p/three_system_human_evaluation_live_20261106_v1/STATE/ratings.sqlite3',ROOT/'outputs/stage_p/completed_human_evaluation_handoff_20261106_v3/STAGE_P_COMPLETED_HUMAN_EVALUATION_REVIEW.zip']:
  protected.append(dict(file=file.relative_to(ROOT).as_posix(),sha256=sha(file),purpose='Preservation hash only;not an analysis input'))
 save(P/'SOURCE_INVENTORY.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),archives=receipts,member_copies=copies,documents=documents,protected_excluded_artifacts=protected,
  official_P_source_sha256=ARCHIVES['P'][1],exploratory_revisions_included=False,external_ChatGPT_statistical_ZIP_provided=False,
  authorization='Researcher clarified independent reproduction from recovered original V1,without requiring externally created statistical ZIP.'))
 print(json.dumps(dict(status='PREPARED',root=str(P),archive_checks=receipts,official_P_source=ARCHIVES['P'][1],member_copies=len(copies)),indent=2))

if __name__=='__main__':main()
