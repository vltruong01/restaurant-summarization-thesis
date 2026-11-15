"""Reproducible TRAIN/VALIDATION lexical screening; no TEST, model calls or semantic auto-judge."""
from pathlib import Path
import json,hashlib,re,unicodedata,collections,difflib,statistics,csv
R=Path(__file__).resolve().parents[1];O=R/'outputs/pre_evaluation/s1000_readiness_v1';E=O/'EVIDENCE';M=R/'outputs/stage_i/led_large_ro_u_source_separated_s1000_matched_v1';D=M/'COMPARISON'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def hs(x):return hashlib.sha256(x.encode()).hexdigest()
def save(n,x):(E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def words(s):return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*",unicodedata.normalize('NFKC',s).casefold())
def body(s):return re.sub(r'^\[(?:REVIEW_CONSENSUS|VISUAL_EVIDENCE)\]\s*','',s).strip()
def sentences(s):return [x.strip() for x in re.split(r'(?<=[.!?])\s+',body(s)) if x.strip()]
def ngrams(s,n):
 t=words(s);return [tuple(t[i:i+n]) for i in range(max(0,len(t)-n+1))]
trainfile=R/'outputs/stage_g/teacher_production_d042_v1/guarded_production6_retry_v1/dataset/train_reviews_only.jsonl'
train=[json.loads(l) for l in trainfile.read_text(encoding='utf8').splitlines() if l.strip()];assert len(train)==2000 and all(r['split']=='train' for r in train)
source=read(D/'EXACT_VALIDATION200_SOURCES.json');source={s['business_id']:s for s in source}
trainids={r['business_id'] for r in train};assert not trainids&set(source)
train_index={'review':collections.defaultdict(set),'U_visual':collections.defaultdict(set),'F_visual':collections.defaultdict(set)};train_openings=collections.Counter();train_bindings={str(trainfile.relative_to(R)):sha(trainfile)}
for r in train:
 for s in sentences(r['target']):
  t=words(s)
  if len(t)>=8:train_index['review'][tuple(t)].add(r['business_id'])
  if len(t)>=6:train_openings[' '.join(t[:6])]+=1
for arm,condition in [('U','unfiltered'),('F','filtered')]:
 path=trainfile.parent/f'train_{condition}.jsonl';train_bindings[str(path.relative_to(R))]=sha(path)
 rr=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]
 assert len(rr)==2000 and {r['business_id'] for r in rr}==trainids
 for r in rr:
  target=r['target'].split('\n\n[VISUAL_EVIDENCE]\n',1)[1]
  for sent in sentences(target):
   t=words(sent)
   if len(t)>=8:train_index[arm+'_visual'][tuple(t)].add(r['business_id'])
pred={a:read(M/f'RECOVERY_R1/DOWNLOADED/VALIDATION_RESULTS/{a}/predictions.json') for a in ['RO','U']}
pred['F']=read(R/'outputs/stage_i/led_large_filtered_validation200_selection_v1/MERGED/V1_1000/predictions.json')
phrases=['selected reviewers frequently praise','reviewers commonly describe','service is often described','value opinions are mixed','food opinions are mixed','service experiences vary','the atmosphere is often described']
rows=[];sentence_details=[];openings={};recurrent={};lexical={}
for a,pp in pred.items():
 bytask=collections.defaultdict(list)
 for p in pp:
  bid=p['business_id'];s=source[bid]
  for task,g in p['passes'].items():
   target=s['common_original_review_reference'] if task=='review' else s['conditions'][a]['original_visual_reference']
   src=s['exact_common_review_source'] if task=='review' else '\n'.join(x['exact_caption'] for x in s['conditions'][a]['captions'])
   out=body(g['text']);ref=body(target);ss=sentences(g['text']);tt=sentences(target);refnorm=' '.join(words(ref));srctris=set(ngrams(src,3));outtris=ngrams(out,3)
   exact=near=cross=0;eligible=0
   for sent in ss:
    w=words(sent)
    if len(w)<8:continue
    eligible+=1;norm=' '.join(w);same=(' '+norm+' ') in (' '+refnorm+' ')
    scores=[difflib.SequenceMatcher(None,w,words(t),autojunk=False).ratio() for t in tt if len(words(t))>=6]
    best=max(scores,default=0.0);ne=(not same and best>=.85);training=sorted(train_index['review' if task=='review' else a+'_visual'].get(tuple(w),set()))
    exact+=same;near+=ne;cross+=bool(training)
    sentence_details.append(dict(condition=a,business_id=bid,task=task,output_span=sent,normalized_exact_contiguous_in_own_teacher=bool(same),near_sentence_similarity=best,near_flag=ne,
     exact_normalized_sentence_in_TRAIN_teacher_business_ids=training,semantic_error_verified_by_this_screen=False))
   r=dict(condition=a,business_id=bid,task=task,output_sha256=g['text_sha256'],teacher_sha256=hs(target),source_sha256=hs(src),body_words=len(words(out)),sentences=len(ss),eligible_sentences_8plus_words=eligible,
    exact_raw_full_teacher_body=out==ref,exact_normalized_full_teacher_body=words(out)==words(ref),normalized_exact_own_teacher_sentences=exact,near_own_teacher_sentences=near,exact_sentence_matches_any_TRAIN_teacher=cross,
    output_trigram_fraction_also_in_source=sum(t in srctris for t in outtris)/len(outtris) if outtris else None,
    style_phrase_hits={v:int(v in ' '.join(words(out))) for v in phrases},judgment='Lexical screen only; source support and harm not established')
   rows.append(r);bytask[task].append(out)
 for task,texts in bytask.items():
  key=a+'_'+task;op=collections.Counter();ng=collections.Counter();sent=collections.Counter()
  for text in texts:
   op.update(' '.join(words(s)[:6]) for s in sentences(text) if len(words(s))>=6)
   ng.update(set(' '.join(x) for x in ngrams(text,6)))
   sent.update(set(' '.join(words(s)) for s in sentences(text) if len(words(s))>=8))
  openings[key]=op.most_common(20);recurrent[key]=dict(sixgrams_business_document_frequency=ng.most_common(30),sentences_business_document_frequency=[(x,n) for x,n in sent.most_common(30) if n>1])
  lexical[key]=dict(unique_raw_bodies=len(set(texts)),distinct_unigrams=len(set(t for x in texts for t in words(x))),word_count=sum(len(words(x)) for x in texts))
summary={}
for key in openings:
 a,task=key.split('_',1);rr=[x for x in rows if x['condition']==a and x['task']==task];n=len(rr)
 summary[key]=dict(outputs=n,eligible_sentences=sum(x['eligible_sentences_8plus_words'] for x in rr),normalized_exact_own_teacher_sentences=sum(x['normalized_exact_own_teacher_sentences'] for x in rr),near_own_teacher_sentences=sum(x['near_own_teacher_sentences'] for x in rr),exact_raw_full_teacher_bodies=sum(x['exact_raw_full_teacher_body'] for x in rr),exact_normalized_full_teacher_bodies=sum(x['exact_normalized_full_teacher_body'] for x in rr),exact_sentence_matches_any_TRAIN_teacher=sum(x['exact_sentence_matches_any_TRAIN_teacher'] for x in rr),
  source_trigram_overlap_mean=statistics.mean(x['output_trigram_fraction_also_in_source'] for x in rr if x['output_trigram_fraction_also_in_source'] is not None),style_phrase_output_counts={p:sum(x['style_phrase_hits'][p] for x in rr) for p in phrases},**lexical[key])
save('MIMICRY_ALL_VALIDATION_SCREEN.json',rows);save('MIMICRY_SENTENCE_MATCHES.json',sentence_details);save('MIMICRY_BOILERPLATE.json',dict(TRAIN_teacher_top_sixword_sentence_openings=train_openings.most_common(30),validation_output_openings=openings,cross_business_recurrence=recurrent))
save('MIMICRY_SUMMARY.json',dict(status='COMPLETE_LEXICAL_SCREEN_NOT_SEMANTIC_CERTIFICATION',TRAIN_businesses=2000,TRAIN_teacher_style_phrase_business_counts={p:sum(p in ' '.join(words(r['target'])) for r in train) for p in phrases},VALIDATION_businesses=200,assembled_outputs=600,components_screened=len(rows),summary=summary,TRAIN_reference_files=train_bindings,TRAIN_match_scope='Common original Review targets; own-condition Visual targets indexed separately for U/F',
 methods=dict(normalization='NFKC/casefold, regex alphanumeric words with internal apostrophe/hyphen; markers excluded',exact='Own-teacher normalized contiguous sentence, minimum8words; raw/normalized full-body equality separately',near='Best sentence token SequenceMatcher autojunk=False ratio>=0.85 and not normalized-exact; heuristic, not paraphrase judge',boilerplate='Per-condition/task document frequency of6grams and8plus-word exact normalized sentences',source_overlap='Generated word trigrams present in exact model-visible review source or own captions; not factuality'),
 limitations=['VALIDATION teacher targets were not training examples; same-business similarity does not prove memorization of validation answers.','TRAIN teacher exact sentence overlaps can be generic style/consensus; they do not independently prove harmful copying.','Common captions naturally cause near-exact Visual rendering; that is not itself harmful mimicry.','No TEST input/reference/output used here, no student/teacher inference, no retuning.']))
issues=read(D/'CLAIM_LEVEL_ISSUES.json');examples=[]
for a,bid in [('U','NRl1WVY_HJNilsMWaVMQPg'),('U','cPDX2Y4H0fks-Qj5tfBduA'),('RO','s4nlPf6kXz97uk0yerqmHg'),('F','5wlbureqTc6gsaB_y-OkOA'),('F','am5qD6LFjkV8f_4MLp8Img')]:
 s=source[bid];p=next(x for x in pred[a] if x['business_id']==bid)
 example_issues=[x for x in issues if x['condition']==a and x['business_id']==bid and x['confirmed']]
 examples.append(dict(condition=a,business_id=bid,name=s['name'],raw_output=p['combined_text'],teacher_review=s['common_original_review_reference'],teacher_visual=None if a=='RO' else s['conditions'][a]['original_visual_reference'],
  confirmed_prior_source_grounded_findings=example_issues,provenance='Bounded previously documented model-assisted source review, not an automatically inferred error from similarity',
  similarity=[r for r in rows if r['condition']==a and r['business_id']==bid]))
save('MIMICRY_SELECTED_EXAMPLES.json',examples)
print(json.dumps(summary,indent=2))
