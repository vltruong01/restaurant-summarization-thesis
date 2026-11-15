"""CPU-only evidence review of the completed locked CAD pilot. No model generation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path(__file__).resolve().parents[1];P=R/'outputs/stage_i/led_large_cad_v1'
E=R/'outputs/stage_i/led_large_cad_v1_vast_execution_v1';RUN=E/'downloaded/run'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
 with (E/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 cases=read(P/'TRAIN_SOURCE_ONLY_CASES.json');byid={c['business_id']:c for c in cases}
 results=read(RUN/'PAIRED_OUTPUTS.json');assert len(results)==12
 completion=read(RUN/'COMPLETION.json');assert completion['generation_calls']==37 and completion['optimizer_updates']==0
 assert read(E/'downloaded/operations/POSTFLIGHT_PRESERVATION.json')['status']=='PASS'
 assert (E/'LIFECYCLE_KEEP_INSTANCE.json').exists(),'Latest keep-instance preference must be recorded'
 # CPU re-decode of all37 raw returned sequences, immutable source token check.
 from transformers import LEDTokenizerFast
 ws=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
 pc=read(ws/'configs/stage_i_led_large_decoder_policy_v2.json')
 tok=LEDTokenizerFast.from_pretrained(ws/pc['source_tokenizer'],local_files_only=True)
 returned=0
 for case,result in zip(cases,results):
  assert case['business_id']==result['business_id'] and case['source_sha256']==result['source_sha256']
  assert tok.encode(case['source'],add_special_tokens=True)==case['input_ids']
  assert result['source_input_ids_sha256']==hashlib.sha256(json.dumps(case['input_ids']).encode()).hexdigest()
  for method in ['vanilla','alpha0','cad']+(['cad_repeat'] if 'cad_repeat' in result else []):
   row=result[method];ids=row['raw_token_ids'];assert tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==row['text']
   assert row['RNG']['transition_exact'] and row['RNG']['restoration_exact'];assert not row['errors']
   returned+=1
  assert result['vanilla']['raw_token_ids']==result['alpha0']['raw_token_ids']
 assert returned==37
 save('RAW_OUTPUT_AND_SOURCE_CPU_AUDIT.json',dict(status='PASS',source_retokenizations=12,raw_sequences_redecoded=37,
      alpha0_exact_pairs=12,first_case_CAD_exact_repeat=results[0]['cad']['raw_token_ids']==results[0]['cad_repeat']['raw_token_ids'],
      held_out_or_validation_access=False,pretrained_model_loaded=False,new_generations=0))
 quotes=[]
 def quote(i,marker,needle=None):
  case=cases[i];table=case['captions'] if marker.startswith('IMG_CAP_') else case['reviews']
  text=table[marker];q=text if needle is None else needle;start=text.index(q)
  assert text[start:start+len(q)]==q
  key=f'Q{len(quotes)+1:02}'
  quotes.append(dict(id=key,business_id=case['business_id'],source_sha256=case['source_sha256'],marker=marker,
      start=start,end=start+len(q),quote=q,offset_basis='exact marker-body Unicode codepoints'))
  return key
 annotations=[]
 def row(i,gain,core,balance,visual,repetition,verdict,evidence):
  annotations.append(dict(business_id=cases[i]['business_id'],name=cases[i]['source'].split('name=',1)[1].split('\n')[0],
      caption_count=len(cases[i]['captions']),potential_gain=gain,Core_coverage=core,opinion_balance=balance,
      visual_grounding=visual,semantic_repetition_and_usability=repetition,provisional_verdict=verdict,evidence_ids=evidence,
      researcher_confirmed=False))
 row(0,'CAD adds salient coffee praise and the specific hot-chocolate complaint.',
     'Vanilla omits coffee; CAD restores it. Both still miss the recurring homemade-tortilla/salsa identity and later chicken-quality complaint.',
     'Egg-blandness caveat survives. Unlike the LOCAL partial output, Vast CAD does not explicitly transfer it to the scramble. Do not reuse that local criticism here.',
     'No captions. Vanilla invents frozen taco/bell/chicken-strip photo relations. CAD gives five ingredient assertions in VISUAL_EVIDENCE; review mentions cannot validate them as visual evidence.',
     'Repeated The taco contains template remains; not a zero-caption repair.', 'MIXED_REVIEW_GAIN_ZERO_CAPTION_FAILURE',
     [quote(0,'REV_19','I come back for the amazing coffee options.'),quote(0,'REV_9','It is neither hot nor terribly chocolaty.'),
      quote(0,'REV_3',"The eggs weren't even seasoned with salt"),quote(0,'REV_30',"I'm chewing on a taco bell style chicken strip in my breakfast taco.")])
 row(1,'Adds pizza-quality and wait/complaint-handling caveats; removes duplicated inattentive wording.',
     'More meaningful opinion detail, but named pizza varieties remain absent from both outputs.',
     'REV20 supports mediocre/cardboard-rubbery pizza; lunch special $3.99 is in REV2. The unsupported pricey/expensive consensus from LOCAL CAD is absent in this Vast output.',
     'Both fail zero-caption compliance: pizza photo versus pizza-slice photo.',
     'Generic aspects remain; source-grounded review benefit does not establish overall clean output.', 'MIXED_REVIEW_GAIN_ZERO_CAPTION_FAILURE',
     [quote(1,'REV_20','I got a cardboard/rubbery taste to the first slices of pizza.'),
      quote(1,'REV_2','lunch special $3.99, cheese slice (cut into two) and a drink.'),
      quote(1,'REV_14','White Chicken was definitely the crowd favorite.')])
 row(2,'Adds cut/side/ambience specifics and a salad-bar caveat.',
     'Meat is central. CAD drops the recurring dry/overcooked/tough meat complaints present in vanilla.',
     'More details do not compensate for loss of same-aspect positive/negative meat synthesis.',
     'Both assert visuals despite zero captions. CAD adds interior/tables/chairs and cheese-bread meal relations. Bacon-wrapped ribs are unsupported by the selected bacon evidence, which names filet/chicken.',
     'Ribs/ribs-wrapped repetition remains. Local rice-with-bacon error is absent here and is not counted against Vast CAD.', 'REGRESSION_CORE_DISAGREEMENT_AND_GROUNDING',
     [quote(2,'REV_5','All our meat was over cooked and chewy.'),quote(2,'REV_15','The filet was also overcooked and tough.'),
      quote(2,'REV_18','Steak is very bland and dry.'),quote(2,'REV_3','filet wrapped in bacon'),
      quote(2,'REV_13','fillet mignon and chicken wrapped in bacon'),quote(2,'REV_29','Cheese bread is to die for.')])
 row(3,'Adds a general food-quality caveat and more differentiated service/value wording.',
     'CAD drops signature brisket from the review list, present in vanilla and explicitly praised in REV1. New named dishes do not guarantee better Core selection.',
     'Mixed food/service/value is useful, but not all newly chosen details were exhaustively claim-audited.',
     'New woman making chocolate pudding on a drink is unsupported: CAP1 says chocolate DRINK with purple flower; CAP3 woman making a drink. Group moves from a TABLE in CAP2 to a bar. Vanilla already has sitting-woman/drink-on-drink errors; CAD does not repair them reliably.',
     'Removes some literal adjective duplication but adds a clear visual construction error.', 'REGRESSION_VISUAL_GROUNDING_WITH_REVIEW_TRADEOFF',
     [quote(3,'IMG_CAP_1'),quote(3,'IMG_CAP_2'),quote(3,'IMG_CAP_3'),quote(3,'REV_1','mouth watering for their brisket.')])
 row(4,'Visual section is shorter and preserves all five caption themes without the many visible/is-on micro-sentences.',
     'Coffee and ambience remain generic; useful baked-food particulars and working/noise tradeoff remain underrepresented.',
     'Coffee decline is supported; watered-down specificity is not established by that complaint. CAD repeats staff/relaxing adjectives and pluralizes criticism of furniture seen in one review.',
     'Caption-grounded visual composition/concision is improved; no clear new visual relation error identified in the selected captions.',
     'Helpful/helpful and relaxed/relaxed persist. Concise visuals alone are not a substantive overall quality GO.', 'MIXED_VISUAL_CONCISION_REVIEW_CONCERNS',
     [quote(4,'IMG_CAP_5'),quote(4,'REV_15','Not only has the quality of the coffee gone down'),
      quote(4,'REV_18','But lately it is WAY too loud.'),quote(4,'REV_19','most of the furniture is for sale (for astronomical prices)')])
 row(5,'Adds onion rings and service-order/wait details; restores caption-supported fries and removes tray-in-front-of-sign fusion.',
     'Some useful business-specific coverage beyond generic burger quality, though tempura green beans/tuna alternatives are still absent.',
     'New dry/underseasoned onion-ring caveat misbinds negative burger/chicken descriptions. Selected onion-ring reviews are favorable; REV4 specifically says the PATTY needed seasoning.',
     'Chicken tenders ON the sandwich remains unsupported: CAP1 lists tenders, fries and a sandwich on a tray, not sandwich filling.',
     'Fewer inattentive duplicates, but two source-attribution errors remain.', 'MIXED_USEFUL_ADDITIONS_ASPECT_MISBINDING',
     [quote(5,'IMG_CAP_1'),quote(5,'IMG_CAP_2'),quote(5,'REV_2','The onion rings are AWESOME.'),
      quote(5,'REV_4','The patty needed seasoning, and the burger was lacking sauce!')])
 row(6,'Adds quiche/Paris-Brest names, but many additions are not usable faithful synthesis.',
     'Paris-Brest becomes sandwiches; fresh pastries/croissants are generalized as imported. Missing take-away/no-seating identity despite strong source detail.',
     'Positive croissant praise appears after although; quiche inherits not-overly-sugary from sweets. A corrected/wrong-business closed-Sunday review becomes felt forced to leave.',
     'Pastries-on-display fusion with the independent cakes caption remains; masked woman still omitted. Some sentence merging is not factual repair.',
     'Repeated pricey-but-quality statements and malformed atmosphere-as-food list reduce usability.', 'REGRESSION_FACT_BINDING_AND_USABILITY',
     [quote(6,'REV_4','The eclairs, lemon tarts and Paris-Brest are outstanding.'),
      quote(6,'REV_9','The quiche was delightfully flavorful and light; the sweets were as you would imagine--delicious and not overly sugary.'),
      quote(6,'REV_13','I will post this review to the right organization'),quote(6,'IMG_CAP_4'),quote(6,'IMG_CAP_5')])
 row(7,'Adds food-quality disagreement and missed-buzzer/reservation-handling detail.',
     'Review specificity modestly improves but dish/portion identity remains generic.',
     'Bland sides/dry chicken and seating confusion have source support; repeated seasoning words add little.',
     'Severe new fusion: sushi is placed on a GRILL and meat/vegetables cook on TOP OF THE SUSHI. CAP1 grill foods and CAP3 sushi-on-black-plate are separate. Vanilla wrongly puts sushi on a bar; CAD worsens rather than resolves the relation error.',
     'Lively repeated three times; quick and pricey repeated. Significant regression despite useful review caveats.', 'REGRESSION_SEVERE_VISUAL_RELATION_AND_REPETITION',
     [quote(7,'IMG_CAP_1'),quote(7,'IMG_CAP_2'),quote(7,'IMG_CAP_3'),
      quote(7,'REV_7','It was very bland and didn\'t have much flavor.'),quote(7,'REV_16',"I dont care for the chicken, it gets a little dry sometimes"),
      quote(7,'REV_3',"didn't ring the buzzers they gave us")])
 row(8,'Adds smokehouse/cheese-fries/beer/Wednesdays detail and attempts to separate ribs from wings.',
     'Greater length has real source topics but also redundant/misassigned topics; not evidence of better Core synthesis.',
     'Rubbery belongs to TURKEY BURGER(REV26), not ribs; disliked cheese is FRIES(REV27), not wings. CAD creates false same-aspect negative contrasts.',
     'Sauce sentence repeats exactly, bottles/counter repeats semantically, and sandwich holds fries misstates CAP3 plate-with-sandwich-and-fries.',
     'Large expansion and exact/semantic repetition markedly reduce usability.', 'REGRESSION_ASPECT_GROUNDING_AND_REPETITION',
     [quote(8,'REV_26','the turkey burger was tough and rubbery'),
      quote(8,'REV_27','Ordered wings which were good and fries with cheese sauce but not a fan or the cheese they used.'),
      quote(8,'REV_6','half priced wing days on Wednesdays'),quote(8,'IMG_CAP_1'),quote(8,'IMG_CAP_3')])
 row(9,'Adds value disagreement and chicken-nacho specificity; compresses visual wording.',
     'Some improvement over generic wings/service; no evidence of comprehensive Core coverage.',
     'Expensive mediocre food has direct source support; several service/food positives and negatives remain.',
     'Mostly supported caption themes. CAD drops BEER from three glasses, and both omit their tray. No new serious visual relation error identified here.',
     'Loud/loud duplication and generic competitive wording remain. Modest local benefit, not validated overall gain.', 'MODEST_LOCAL_GAIN_WITH_LIMITATIONS',
     [quote(9,'REV_16','The prices are ridiculous for completely mediocre food.'),quote(9,'IMG_CAP_4'),quote(9,'IMG_CAP_2')])
 row(10,'Adds distinctive seafood-paella/calamari/scallop names and restores the caption-supported plate-with-fork description.',
     'Named-dish coverage improves, though cooking-quality nuances and strong wine-list identity are still incomplete.',
     'Broad price/service contrast retained; positive seafood descriptions are supported by selected reviews.',
     'Persistent dessert-with-lime error: CAP2 garnishes a DRINK. Persistent group-at-bar error: CAP1 places people at a TABLE. Shorter visual prose does not cure either.',
     'More concise and specific in places, but not reliably grounded.', 'MIXED_SPECIFICITY_GAIN_PERSISTENT_VISUAL_ERRORS',
     [quote(10,'REV_7','The seafood paella was ridiculously delicious.'),quote(10,'REV_25',"world's best calamari app"),
      quote(10,'IMG_CAP_1'),quote(10,'IMG_CAP_2'),quote(10,'IMG_CAP_3')])
 row(11,'Adds butter chicken, curry goat, chicken tikka masala and palak paneer; concise supported visual section.',
     'Distinctive-dish coverage improves; CAD loses explicit buffet identity present in vanilla. Cold buffet and naan-quality details remain omitted.',
     'Average/mediocre caveat is broader than the important dish/temperature disagreements. Neither is a complete balanced synthesis.',
     'Both align with the two generic captions; CAD combines their short descriptions without an identified serious new visual error.',
     'Affordable/affordable remains. Specificity gain alone is insufficient for overall GO.', 'MIXED_DISH_GAIN_BUFFET_AND_NEGATIVE_DETAIL_LOSS',
     [quote(11,'REV_14','Butter chicken is out of this world'),quote(11,'REV_30','Curry goat and butter chicken have been my got to dishes'),
      quote(11,'REV_17','The naan is often burnt and the curries are lacking flavor.'),
      quote(11,'REV_19','The rice and four of the other dishes I chose were literally cold.'),quote(11,'IMG_CAP_2')])
 save('SOURCE_EVIDENCE_QUOTES.json',quotes)
 save('PROVISIONAL_CASE_ANNOTATIONS.json',dict(assessor='implementation author; model-assisted,unblinded',
      scope='all24 final texts,all captions,source passages relevant to changed claims;not exhaustive independent all-claim adjudication',
      researcher_confirmed_labels=0,cases=annotations))
 cross=[]
 local=R/'outputs/stage_i/led_large_cad_v1_local_execution_v1/run/cases'
 for i,result in enumerate(results[:3]):
  old=read(local/f'{i:02d}.json')
  cross.append(dict(business_id=result['business_id'],vanilla_raw_exact=old['vanilla']['raw_token_ids']==result['vanilla']['raw_token_ids'],
      CAD_raw_exact=old['cad']['raw_token_ids']==result['cad']['raw_token_ids'],CAD_text_exact=old['cad']['text']==result['cad']['text']))
 save('CROSS_HOST_LOCAL_VAST_COMPARISON.json',dict(cases=cross,
      environments='Local Windows RTX3050 driver596.36 vs Linux RTX3090 driver595.91.07;locked primary versions/code/data same',
      cause='Not isolated;no extra numerical/GPU experiments authorized',
      interpretation='Within-host alpha0 and first-case deterministic repeat pass;do not claim bitwise cross-host CAD determinism',
      local_failure_evidence_sha256=sha(R/'outputs/cloud_handoff/LED_LARGE_CAD_V1_LOCAL_RESOURCE_STOP_REVIEW.zip')))
 decision=dict(recommendation='NO_GO',scope='Do not adopt/promote this fixed alpha0.3 CAD policy for the main thesis comparison',
      technical_status='PASS_12_CASES_37_CALLS',researcher_quality_decision='PENDING',
      basis=['All3zero-caption CAD visual sections remain unsupported','Severe cross-caption relations in Tujague and Kobe',
             'New aspect-binding errors in Habit/Little Gourmand/Whistlers','Loss of recurring meat disagreement in Terra Mar',
             'Review-specificity/concision gains exist in several cases but coexist with important regressions'],
      limitations=['12 previously examined stratified TRAIN cases,not generalization','Unblinded model-assisted assessment,not researcher-confirmed',
           'Targeted source-linked review,not exhaustive independent atomic-claim annotation','Some CAD outputs differ across local/Vast hosts despite unchanged primary bindings;cause unisolated'],
      no_universal_CAD_ineffectiveness_claim=True,no_tuning_or_additional_technique_authorized=True,
      primary_Filtered_checkpoint='Original LED-large CE epoch3 unchanged',RO_U_F_Shuffled_protocol_unchanged=True,
      contrastive='Final researcher NO_GO_CLOSED',instance_policy='KEEP_RUNNING;destroy/stop authorization revoked by newest user message')
 save('FINAL_DECISION.json',decision)
 lines=['# CAD alpha0.3: source-grounded TRAIN comparison','',
      '**Technical PASS; provisional NO-GO for adopting this fixed CAD policy. Independent researcher decision remains PENDING.**',
      'All12 pairs were reviewed. No candidate rewrite, zero-caption safeguard, Teacher/reference/photo/validation evidence or selection is used.',
      'Assessor is the implementation author/model-assisted/unblinded. This is targeted source-linked analysis, not an exhaustive independent atomic-claim study.',
      '', '| Business | Useful changes | Grounding / Core / repetition concerns | Provisional assessment |', '|---|---|---|---|']
 for a in annotations:lines.append('| '+a['name']+' | '+a['potential_gain']+' | '+a['visual_grounding']+' '+a['opinion_balance']+' | '+a['provisional_verdict']+' |')
 for a in annotations:
  lines.extend(['','## '+a['name'],'',f'ID: `{a["business_id"]}`; Filtered captions: {a["caption_count"]}',
       '', '**Core:** '+a['Core_coverage'],'','**Repetition/usability:** '+a['semantic_repetition_and_usability'],
       '', '**Evidence IDs:** '+', '.join(a['evidence_ids'])])
 lines+=['','## Exact source excerpts','']
 for q in quotes:lines.append(f'- **{q["id"]} / {q["business_id"]} / {q["marker"]} [{q["start"]}:{q["end"]}]**: {q["quote"]}')
 lines+=['','## Interpretation','',
      'CAD restores some useful dish names and caveats, but source-word association does not enforce predicate/argument binding or modality attribution. The observed errors are consistent with that limitation; they do not prove a universal causal mechanism.',
      'All3zero-caption cases still assert visual facts. Do not credit review-derived visual assertions as grounded. No safeguard was applied to either method.',
      'No loss/ROUGE/summary-length metric is used as the substantive decision. Exact sentence repetition can miss lively,lively,lively and unsupported relationships.',
      'Cross-host differences are preserved separately. Local partial-review complaints that no longer occur in the Vast outputs were explicitly not carried forward.',
      'No broad CAD ineffectiveness claim, no checkpoint promotion, no rerun/alpha search/new technique. Keep Original CE3 and primary matched RO/U/F/Shuffled protocol. Stop for independent review.']
 (E/'CASE_LEVEL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (E/'README_FIRST_VI.md').write_text('''# Chỉ cần gửi ZIP này cho GPT review

CAD đã hoàn tất trên RTX3090:12/12 cặp,24 output khoa học,12 alpha0 kiểm tra và
1 lần lặp CAD; tổng37 generate calls. Không training/backward/optimizer.
Alpha0 khớp vanilla12/12; case đầu CAD lặp chính xác. Headroom tối thiểu80.10%.
Toàn bộ146 file export đã tải về và xác minh SHA256. Base/adapter giữ nguyên.

Khuyến nghị model-assisted: NO-GO cho việc áp dụng policy CAD alpha0.3 hiện tại.
Có chi tiết review tốt hơn, nhưng còn lỗi visual zero-caption và các lỗi liên kết
thực thể/aspect mới. Không kết luận CAD vô dụng nói chung. Researcher chưa chấm.

Đọc CASE_LEVEL_REVIEW.md, ALL_PAIRED_OUTPUTS.md, EXECUTION_REPORT.md và
FINAL_DECISION.json. Tất cả nguồn TRAIN,raw tokens,RNG/cache/logit/memory receipts,
exact code/config hashes và closure Contrastive có trong gói. Giữ nguyên evidence
lượt local bị thiếu VRAM; code/dữ liệu/alpha/beam không đổi,không che giấu retry.

Theo yêu cầu mới nhất: GIỮ instance57481997,không destroy/stop. Service pilot đã
EXITED,autostart=false,autorestart=false. Không có công việc GPU tiếp theo.
''',encoding='utf-8')
 heading='## CAD V1 Vast:12-pair technical PASS, provisional adoption NO-GO; instance retained (2026-10-24)\n\n'
 text=heading+('Researcher explicitly authorized a new RTX3090 host execution after local resource stop. Same CAD V1 config/runner/helper/preflight bindings,alpha0.3,beam4,Original CE3. '
      '121 transferred files verified;10 CPU +1 CUDA/FP16 reduced LED test PASS. One fresh run,37 generate calls,24 final scientific outputs on12 locked TRAIN cases,12 exact alpha0 identities and first-case CAD exact repeat. '
      '0 optimizer/backward,weights unchanged,80.10% memory headroom,333.4s main subprocess.146 exported files downloaded/hash-verified. '
      'Provisional source-linked review:local specificity/concision gains but serious relation/aspect errors,semantic repetition,and3/3zero-caption visual failures. '
      'NO-GO for adopting current fixed CAD policy;researcher decision pending,no universal method-ineffectiveness claim. Repeated TRAIN exposure and cross-host CAD differences disclosed. '
      'Contrastive remains closed;Original CE3 and main RO/U/F/Shuffled unchanged. Newest user instruction revokes destroy:keep instance57481997;no further lifecycle/GPU actions.\n\n')
 for rel in ['DECISIONS.md','THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md']:
  path=R/rel;old=path.read_bytes();assert heading.encode() not in old;path.write_bytes(text.encode()+old)
 print(json.dumps(dict(source_quotes=len(quotes),case_reviews=len(annotations),recommendation=decision['recommendation'],cross_host=cross),indent=2))
if __name__=='__main__':main()
