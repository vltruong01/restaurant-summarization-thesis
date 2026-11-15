"""Source-grounded partial review, postflight disk hashes and final stop receipt."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,sys
R=Path(__file__).resolve().parents[1];P=R/'outputs/stage_i/led_large_cad_v1'
E=R/'outputs/stage_i/led_large_cad_v1_local_execution_v1'
WS=R/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE'
CP=R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/final_evidence/runs/F_large_decoder_policy_v2/train/checkpoint-750'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    with (E/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    sys.path.insert(0,str(R/'scripts'))
    from run_led_large_cad_v1 import preflight
    c=preflight(WS,CP)
    assert read(E/'PROCESS_EXIT.json')['return_code']==1
    fail=read(E/'run/FAILURE_STOP_NO_RETRY.json')
    assert fail['generation_calls']==11 and fail['complete_pairs']==3
    assert 'headroom below 15%' in fail['error']
    files={str((CP/n).relative_to(R)):sha(CP/n) for n in read(CP/'checkpoint_integrity.json')['files']}
    files.update({str((WS/rel).relative_to(R)):sha(WS/rel) for rel in c['model_files']})
    write('POSTFLIGHT_ORIGINAL_DISK_INTEGRITY.json',dict(status='PASS_UNCHANGED_DISK_BYTES',files=files,
        parent_python_files_verified=len(c['parent_code_bindings']),runtime_sources_verified=len(c['runtime_source_hashes']),
        live_parameter_hash_after_abort='NOT_COMPLETED_PROCESS_EXITED_AFTER_RESOURCE_STOP',
        no_optimizer_or_backward_executed=True,weights_or_precision_fallback=False))
    cases=read(P/'TRAIN_SOURCE_ONLY_CASES.json');quotes=[]
    requests={
        0:[('REV_3',"The eggs weren't even seasoned with salt"),('REV_3','I ordered 2 bacon and egg tacos'),
           ('REV_9','It is neither hot nor terribly chocolaty.'),('REV_19',"I come back for the amazing coffee options."),
           ('REV_22','scrambled eggs with onions, peppers and cheese.'),
           ('REV_30',"I'm chewing on a taco bell style chicken strip in my breakfast taco.")],
        1:[('REV_20','The pizza was ok, but definitely nothing out of this world.'),
           ('REV_20','I got a cardboard/rubbery taste to the first slices of pizza.'),
           ('REV_14','White Chicken was definitely the crowd favorite.'),
           ('REV_17','worth the wait for inexpensive but good pasta or pizza!'),
           ('REV_12','reasonable prices.'),('REV_24',"The price on the website didn't match what it says in the store, but they matched it.")],
        2:[('REV_5','All our meat was over cooked and chewy.'),('REV_15','The filet was also overcooked and tough.'),
           ('REV_18','Steak is very bland and dry.'),('REV_13','The rice was cooked perfectly, the beans cooked with bacon'),
           ('REV_29','Cheese bread is to die for.'),('REV_21','their chocolate moose/moose cake is so good!')]
    }
    for i,items in requests.items():
        for marker,quote in items:
            text=cases[i]['reviews'][marker];start=text.index(quote)
            assert text[start:start+len(quote)]==quote
            quotes.append(dict(id=f'Q{len(quotes)+1:02}',business_id=cases[i]['business_id'],source_sha256=cases[i]['source_sha256'],
                marker=marker,quote=quote,start=start,end=start+len(quote),offset_basis='exact reviews[marker] Unicode codepoints'))
    write('SOURCE_EVIDENCE_QUOTES.json',quotes)
    annotations=[dict(business_id=cases[0]['business_id'],assessment='MIXED_NO_OVERALL_GO',
        review_gain='Adds salient repeatedly praised coffee and specific hot-chocolate criticism (Q03,Q04)',
        disagreement='Retains egg-blandness caveat, but incorrectly transfers it to scramble: Q01/Q02 concern bacon/egg tacos',
        visual='Both fail zero-caption compliance; CAD creates five unsupported ingredient sentences in VISUAL_EVIDENCE',
        repetition='Vanilla repeats next-to template; CAD repeats The taco contains five times; not a semantic repetition solution',
        omissions='Both miss important homemade tortilla/salsa identity and later chicken-quality disagreement',
        facts='Ingredients occur in reviews(Q05/Q06) but do not support a visual claim or their co-location'),
      dict(business_id=cases[1]['business_id'],assessment='MIXED_NEW_UNSUPPORTED_PRICE_DISAGREEMENT',
        review_gain='Adds pizza-quality caveat and waits/complaint handling; REV20 explicitly supports mediocre/cardboard/rubbery pizza (Q07,Q08)',
        disagreement='CAD asserts others call it pricey/expensive; selected price evidence Q10/Q11/Q12 supports inexpensive/reasonable and a corrected price mismatch, not that asserted opposing consensus',
        visual='Both fail zero-caption compliance: photo shows pizza vs pizza slices',
        repetition='Removes literal inattentive,inattentive repetition, but remains generic',
        omissions='Neither gives useful named varieties such as White Chicken (Q09), BBQ chicken, Garden Pesto; length is not Core coverage',
        facts='Some improvement in food/opinion detail does not compensate for unverified price polarity and continuing unsupported visuals'),
      dict(business_id=cases[2]['business_id'],assessment='REGRESSION_MIXED_OPINIONS_AND_GROUNDING',
        review_gain='Adds rodizio cuts/sides/salad specifics',
        disagreement='Drops repeated negative meat-quality side present in vanilla; Q13/Q14/Q15 explicitly report overcooked,chewy,tough,bland,dry meat',
        visual='Both fail zero-caption compliance. CAD adds tables/chairs,cheese bread and chocolate moose cake as visual evidence although captions are absent',
        repetition='Some list repetition persists: ribs then ribs wrapped in bacon',
        omissions='More dish words accompany loss of central positive/negative meat contrast',
        facts='CAD extends bacon from beans to rice despite Q16 specifying beans cooked with bacon; Q17/Q18 are review facts,not photo evidence')]
    write('PROVISIONAL_CASE_ANNOTATIONS.json',dict(assessor='implementation author, model-assisted, unblinded',
        researcher_confirmed_labels=0,review_scope='3 complete zero-caption pairs; all12 source packets retained; nine captioned pairs NOT GENERATED',
        cases=annotations,quality_review_pending=True,unsupported_price_verdict='provisional source review,not an automated keyword entailment score'))
    partial=(E/'run/calls/03_vanilla.json')
    mem=read(partial)['memory'];head=mem['headroom_fraction']
    write('FINAL_DECISION.json',dict(recommendation='INCONCLUSIVE',reason='LOCAL_RESOURCE_STOP_INCOMPLETE12CASE_PILOT',
        technical_implementation='10 CPU + explicit CUDA reduced LED/FP16 test PASS;3 real-model alpha0 pairs exact;first-case CAD exact repeat',
        quality_findings='No complete quality gate;3zero-caption pairs all retain unsupported visuals;local Core gains coexist with disagreement/grounding regressions',
        generation_calls=11,returned_sequences=11,complete_pairs=3,unpaired_vanilla=1,
        planned_but_not_generated=['Tujague CAD/alpha0']+[x['business_id'] for x in cases[4:]],
        memory_stop=mem,threshold_min_headroom=.15,actual_headroom=head,optimizer_updates=0,
        full24scientific_outputs=False,automatic_retry=False,training=False,
        future_GPU_attempt='Requires fresh explicit authorization; no local resume/retry or automatic transfer',
        current_Filtered_checkpoint='Original CE epoch3 unchanged',primary_RO_U_F_Shuffled_protocol_unchanged=True,
        contrastive='FINAL_RESEARCHER_NO_GO_CLOSED',researcher_quality_decision='PENDING'))
    lines=['# Provisional source-grounded partial CAD comparison','',
        '**INCONCLUSIVE: pilot stopped for local VRAM headroom;3/12 pairs completed.**',
        'Assessor: implementation author/model-assisted/unblinded. Researcher-confirmed labels:0.',
        'All three completed cases have ZERO Filtered captions. Nine captioned pairs were not generated.',
        'They cannot establish CAD effectiveness, overall12-case quality, or captioned-case grounding.',
        '', '| Case | Core / disagreement | Visual / regressions | Provisional verdict |',
        '|---|---|---|---|',
        '| Tempo | Adds important coffee coverage and hot-chocolate criticism; transfers bacon/egg-taco criticism to scramble | Five unsupported visual ingredient statements; source facts are not visual support | Mixed |',
        '| Mesa | Adds pizza-quality and service caveats; invents an opposing pricey/expensive consensus | Both assert pizza photos with no captions; important named varieties still omitted | Mixed with new grounding concern |',
        '| Terra Mar | More specific cuts/sides, but drops recurring negative meat-quality side | Rice/bacon misbinding; adds review-derived bread/cake as visuals despite zero captions | Regression |','',
        '## Exact source links','']
    for q in quotes:lines.append(f'- **{q["id"]} / {q["business_id"]} / {q["marker"]} [{q["start"]}:{q["end"]}]**: {q["quote"]}')
    lines+=['','## Interpretation limits','',
        'Three of three completed vanilla and CAD visual sections remain unsupported under zero captions. This is an observed partial count, not a population rate.',
        'More specific wording does not ensure evidence entailment or preservation of disagreement. CAD boosts source-associated words without enforcing modality attribution; the observed review-to-visual leakage is consistent with that limitation, not proof of a universal mechanism.',
        'No caption safeguard, rewriting, alpha tuning or additional technique was applied.',
        'Full source packets and all11 returned sequences (including alpha0/repeat and unpaired Tujague vanilla) remain available. Tujague vanilla passed EOS/RNG but failed memory guard afterward; no CAD comparison is fabricated.',
        'The12 cases were selected/examined in prior TRAIN pilots; this is not independent generalization. No D053/Teacher/validation/photo evidence was used.',
        'Do not promote CAD, reopen Contrastive, modify the primary thesis study or infer a broad scientific CAD NO-GO. Stop for independent review.']
    (E/'CASE_LEVEL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (E/'README_FIRST_VI.md').write_text(f'''# CAD V1: đọc trước

Đã chốt Contrastive NO-GO theo quyết định researcher; giữ nguyên lịch sử FAIL.
CAD đã triển khai,10 test CPU và test CUDA/FP16 thu nhỏ đều PASS. Chạy thật
trên RTX3050 Laptop6GB:3/12 cặp hoàn tất,11 chuỗi trả về tổng cộng. Alpha0
khớp vanilla3/3; CAD ca đầu lặp lại chính xác. Không train/optimizer/backward.

Dừng ở vanilla ca4(Tujague) do headroom còn{100*head:.2f}%, dưới15%. Không OOM
được ghi nhận, không tự giảm beam/precision, không retry. GPU đã giải phóng.
Kết luận CAD: **INCONCLUSIVE do pilot chưa đủ12 ca**, không phải quality GO.
Ba ca zero-caption vẫn có visual unsupported; CAD có cả lợi ích lẫn lỗi mới.

Xem CASE_LEVEL_REVIEW.md, ALL_PAIRED_OUTPUTS.md, ALL_RETURNED_CALLS.md,
EXECUTION_REPORT.md, FINAL_DECISION.json. Toàn bộ9 cặp captioned còn thiếu;
không có bảng chấm giả cho các ca chưa chạy. Bản sao checkpoint/base và toàn
bộ Contrastive logs/checkpoints vẫn giữ nguyên, có SHA256 trong closure receipt.

Muốn chạy lại trên GPU rộng hơn cần phê duyệt mới cho một lần thử khác.
Không tự resume/retry. Không thuê máy, thay model hay train thêm trong lượt này.
''',encoding='utf-8')
    heading='## Contrastive CLOSED; CAD local inference resource stop (2026-10-24)\n\n'
    text=(heading+'Researcher finalized Contrastive NO-GO for the bounded branch, not universal scientific ineffectiveness. '
        '113 full backup files and original CE3 were reverified. CAD alpha0.3 raw-logit LED beam4 implementation: '
        '10 CPU tests plus1 explicit CUDA/FP16 reduced-model test PASS; all12 source IDs reproduce exactly. '
        'Single local RTX3050 6GB run returned11 sequences,3 completed zero-caption pairs,3 exact alpha0 identities '
        'and first-case deterministic CAD repeat. Vanilla case4 completed but resource guard stopped with '
        f'{head*100:.2f}% headroom(<15%); no OOM recorded,no retry/fallback,0 optimizer/backward. '
        'CAD scientific decision INCONCLUSIVE(incomplete pilot);3/3 observed zero-caption cases retain unsupported visuals, '
        'with mixed review gains and new grounding/disagreement regressions. Nine captioned pairs ungenerated. '
        'Original CE3 and RO/U/F/Shuffled unchanged. Stop for independent review; no additional GPU authorization inferred.\n\n')
    for rel in ['DECISIONS.md','THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md']:
        path=R/rel;old=path.read_bytes()
        assert heading.encode() not in old
        path.write_bytes(text.encode()+old)
    print('CLOSED: resource-stop, no retry, original disk hashes PASS')
if __name__=='__main__':main()
