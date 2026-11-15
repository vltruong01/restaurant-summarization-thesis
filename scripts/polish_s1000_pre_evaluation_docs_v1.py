"""Readability-only spacing in new reports/current documentation. Never touches scientific artifacts."""
from pathlib import Path
import re,json,hashlib,difflib
R=Path(__file__).resolve().parents[1];O=R/'outputs/pre_evaluation/s1000_readiness_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=re.compile(r'https?://\S+|`[^`]*`|“[^”]*”|[0-9a-f]{32,}|\d{4}-\d\d-\d\dT[0-9:.]+(?:Z|[+-][0-9:]+)|(?:S1000|S500|RO750|U750|CE3|TEST100|TRAIN2000|VALIDATION200|BLIP-2|FP16|FP32|SHA256|BERTScore|LoRA|Qwen2\.5-VL(?:-7B-Instruct)?)')
def prose(t):
 t=re.sub(r'(?<=\d)(?=[A-Za-z])',' ',t)
 t=re.sub(r'(?<=[A-Za-z])(?=\d)',' ',t)
 t=re.sub(r',(?=[A-Za-z])',', ',t)
 t=re.sub(r'(?<=[a-z])(TEST|TRAIN|VALIDATION|GPU|CPU|BOS|EOS)',r' \1',t)
 t=re.sub(r'\b(TEST|TRAIN|VALIDATION|GPU|CPU|CLIP|pHash|BOS|EOS)(?=[a-z])',r'\1 ',t)
 t=re.sub(r'\b(RO|U|F|S|TEST|TRAIN|VALIDATION|Teacher|Review|Visual)(?=(?:inputs?|outputs?|captions?|checkpoint|tasks?|pass|source|reference|gold|adapters?|data\b|Review|Visual|Teacher|labels?|body|decode|model))',r'\1 ',t)
 replacements={'Suses':'S uses','Sreuses':'S reuses','Uuses':'U uses','Fuses':'F uses','ROhas':'RO has','Uhas':'U has','Fhas':'F has','FandS':'F and S','Schanges':'S changes','Fcaption':'F caption','FReview':'F Review','TeacherTRAIN':'Teacher TRAIN','TeacherINPUT':'Teacher INPUT','STEP':'STEP','Source-SeparatedS':'Source-Separated S','S1000is':'S1000 is','S1000has':'S1000 has','TESTsources':'TEST sources','RO-S':'RO-S'}
 for a,b in replacements.items():t=t.replace(a,b)
 t=re.sub(r'\bS\s+1000\b','S1000',t)
 t=re.sub(r'\b(\d+)\s+e([+-]\d+)\b',r'\1e\2',t)
 t=re.sub(r'\bFP\s+(16|32)\b',r'FP\1',t)
 t=t.replace('v 4.2','v4.2').replace('Fannotations','F annotations').replace('RO/Unew','RO/U new').replace('TRAINsentence','TRAIN sentence').replace('referencecreation','reference creation')
 return t
def polish(s):
 out=[];pos=0
 for m in protected.finditer(s):out.extend([prose(s[pos:m.start()]),m.group()]);pos=m.end()
 out.append(prose(s[pos:]));result=''.join(out)
 result=re.sub(r'(S1000|TEST100|TRAIN2000|VALIDATION200|FP16|FP32)(?=[a-z])',r'\1 ',result)
 return result
names=['thesis_sdd_docs/THESIS_PROGRESS.md','thesis_sdd_docs/CODEX_THESIS_IMPLEMENTATION_PLAN.md','thesis_sdd_docs/CHATGPT_NEW_CONVERSATION_HANDOFF.md','thesis_sdd_docs/LED_LARGE_READINESS_AND_EXPERIMENT_PROPOSAL_V1.md','THESIS_PROGRESS.md']
for p in list(O.glob('*.md'))+[R/n for n in names]:p.write_text(polish(p.read_text(encoding='utf8')),encoding='utf8')
changes=json.loads((O/'EVIDENCE/DOCUMENTATION_CHANGE_MANIFEST.json').read_text())
for x in changes:
 p=R/x['path'];x['after_sha256']=sha(p);before=(O/x['original_snapshot']).read_text(encoding='utf8');after=p.read_text(encoding='utf8')
 (O/x['diff']).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=x['path']+' (before)',tofile=x['path']+' (after)')),encoding='utf8')
(O/'EVIDENCE/DOCUMENTATION_CHANGE_MANIFEST.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf8')
print('Readable spacing applied only to new prose/current documentation; quotations/hashes/URLs preserved')
