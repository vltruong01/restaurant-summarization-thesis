"""Serialize the manually reviewed second disjoint sample; never amend human ratings."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from s1000_final_eval_core_v1 import read, digest, file_hash

ROOT=Path('outputs/stage_n/random40_rating_quality_audit_round2_20261102_v1')
PRIOR=Path('outputs/stage_n/random40_rating_quality_audit_20261102_v1')
STUDY=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')

# Authored after inspecting every new sampled output/comment/score. Source
# ordinals resolve to exact selected review and photo IDs in the frozen sources.
CASES=[
('R01','BROADLY_REASONABLE','Review giữ food/service/value và bối cảnh chật/xuống cấp có nguồn; lặp inattentive/run down rõ. Điểm5 faithfulness và4 coverage có thể biện hộ.',[2,3,5,9,10,15,20],[],['run down, cramped, or run down.']),
('R02','BROADLY_REASONABLE','Năm ảnh chỉ tacos/bowl/khách và không gian, không có cảnh burgers/fries/menu condiments. Visual1 và chỉ rõ lỗi cảnh là hợp lý; phần Review được đánh giá riêng.',[2,8,13,18,30],[1,2,3,4,5],['There are hamburgers and french fries.']),
('R03','BROADLY_REASONABLE','Lặp fish merely okay và Killer là thật, cùng bỏ các food negatives. Mức3 faithfulness có thể tranh luận: R8 gọi món ăn fresh/tasty rồi liệt kê Mojo fries, nên không nên coi mọi liên hệ freshness với fries là hallucination chắc chắn. Ghi chú này mang tính giải thích thận trọng, không phải lỗi điểm xác định.',[2,8,13,18,27,30],[],['one reviewer found the fish merely okay and another found the fish merely okay.']),
('R04','BROADLY_REASONABLE','Nguồn có sandwich/nachos/food-service trái chiều, beer/coffee và lựa chọn chay bị lược. Ảnh có sandwich/brick exterior/dim diners; laptop với POS chưa chắc và người chấm đã ghi uncertainty, nên Visual4 có thể biện hộ.',[2,3,10,11,14,18,19,22,24,25],[2,3,4,5,6,8,10,13],['A laptop is on the counter.']),
('R05','BROADLY_REASONABLE','Bar/khách/bottles và phụ nữ nhìn lên khớp ảnh; menu-next-to-food không rõ, Visual3 có căn cứ. Review có thiếu toilet/cold/dry và lặp inattentive. Faithfulness5 hơi rộng vì câu đầu gộp tính từ gà/jambalaya, nhưng không đủ để buộc đổi điểm.',[1,9,16,26,27],[1,6,11,14,18,26,27],['The menu is next to a plate of food.']),
('R06','BROADLY_REASONABLE','Khác với Visual sai ở mục Starbucks cũ của A02, output này mô tả đúng drinks/green umbrellas/diners/counter. Nhiều câu generic people/restaurant/tables lặp nên2 redundancy có thể biện hộ; Review bỏ order errors và ý khen workspace.',[1,2,4,7,10,19,21,23,24,26],[1,2,3,4,5,6],['There are three Starbucks drinks.','The coffee shop has green umbrellas.']),
('R07','BROADLY_REASONABLE','R13 nói over cooked hot dogs và cheese not melted, không phải undercooked/underseasoned. Bỏ slush/ranch và mô tả Visual từ bộ ảnh khác được chỉ đúng; điểm thấp có cơ sở.',[2,4,6,12,13,16,19,24],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],['undercooked hot dogs, underseasoned cheese','The plate has the words "tuna salad deluxe" on it.']),
('R08','BROADLY_REASONABLE','Nhận ra attribution fries/mac đúng: người nói fries decent khen mac, người khác mới chê mac. Lặp brisket nhiều câu rõ; score3/4/2 có thể biện hộ.',[1,10,14,20,27],[],['one reviewer found them merely decent and disliked the mac and cheese.']),
('R09','BROADLY_REASONABLE','Đọc đủ15 reviews và11 ảnh: các ý mixed food/service/value có nguồn, bỏ order mistakes và vệ sinh là thật. Bag/white-plate burger/breakfast/orange juice đều có trong ảnh; Visual5 không lệch nguồn.',[1,2,3,4,5,6,9,10,12,13,14,15],[1,2,3,4,5,6,8,9,10,11],['A glass of orange juice is on the table.']),
('R10','BROADLY_REASONABLE','Gulf đối diện mới có limited food/services; Wawa khó ra vào do footprint/traffic/pump blocking. Review2 và Visual1 cho thành phần áp dụng nhưng rỗng là đúng cách dùng rubric.',[1,2,3,5,7,9,11,15,18],[],['because of its limited food or drink options.']),
('R11','BROADLY_REASONABLE','Những claim về món/dips/portions/service/patio có nguồn; giá overpriced và drive-through bị bỏ. Faithfulness5 và coverage4 hợp lý, không bắt mọi omission thành sai thực tế.',[2,4,7,14,16,17,18,22,26,30],[],['slow service, order mistakes, or order problems.']),
('R12','CALIBRATION_DISCUSSION','Nhận xét xác định đúng ingredients/patio và thừa nhận same-plate transitions. Tuy nhiên câu The plate nối từ grill/fries/meat sang cake/nuts tạo quan hệ sai giữa hai ảnh. Visual4 có thể hơi rộng so với lỗi fusion; nên trao đổi cách cho3 hay4, không tự sửa. Redundancy2 có căn cứ vì fries lặp nguyên ý và generous lặp.',[4,7,16,17,18,22,26,30],[3,4,5,6,8],['The plate contains french fries.','The plate contains a piece of cake.','The plate contains nuts.']),
('R13','BROADLY_REASONABLE','Cà phê brewed reasonably well không tự chứng minh food reasonably priced; thiếu early closing/waits có nguồn. Hình eggs/gravy/ice-cream sandwich/macarons/coffee/flowers phần lớn đúng; plate on top of bowl chưa được chứng minh rõ, Visual4 hợp lý.',[4,5,9,20,21,28],[1,2,3,4,5,6,7,8],['The plate is on top of a bowl of food.']),
('R14','BROADLY_REASONABLE','Các claim food/service/value có nguồn nhưng bỏ các món cụ thể và lượng nhiều được nhắc lặp. Faithfulness5/coverage3 phân biệt được đúng nội dung với thiếu thông tin. Cần nhớ ghi chú cleanliness là tổng hợp source check, không suy từ ảnh.',[1,2,4,5,7,12,19,22,23,24],[],['Several reviewers describe the restaurant as convenient, quick, or clean.']),
('R15','BROADLY_REASONABLE','Đọc lại toàn bộ R3: người viết định mua Bobbie nhưng cuối cùng nhận nhầm Cole Turkey và nói wrong sandwich just okay. Người chấm phân biệt chi tiết này và Bobbie disappointment ở R23 đúng. Review4/coverage4 có cơ sở.',[2,3,7,12,14,15,18,21,23,27],[],['while others describe it as merely okay or disappointing.']),
('R16','BROADLY_REASONABLE','Đọc đủ21 reviews và17 ảnh: không có ảnh nội thất/chalkboard/many colorful lights; food/value không có cơ sở rõ cho expensive. Visual1/faithfulness3 và bỏ birria/salsa có cơ sở.',[1,2,6,7,8,9,12,15,18,19,21],list(range(1,18)),['The restaurant has a chalkboard menu. The restaurant has many colorful lights.']),
('R17','BROADLY_REASONABLE','Đọc đủ30 reviews và12 ảnh: không có chicken/tomato salad glass-bowl hay broccoli/mushroom plate như Visual; shrimp/oysters/bread/diners mới là cảnh thật. Review bỏ oyster-quality và food-preparation negatives; score1 Visual hợp lý.',[1,2,3,7,9,10,15,18,19,21,22,23,24,28],list(range(1,13)),['The salad is in a glass bowl.']),
('R18','BROADLY_REASONABLE','Ghi chú phân biệt chữ Tuna Salad Deluxe ở poster với chữ trên plate đúng. Salad/pita có ảnh, Review thiếu menu sáng tạo/food negatives và delicious lặp rõ; Visual4/coverage3 có thể biện hộ.',[2,3,8,12,14,16],[1,4,5,6,7,8],['The plate has the words "tuna salad deluxe" on it.']),
('R19','BROADLY_REASONABLE','Dùng lại đối chiếu35 ảnh: không xác lập egg plate/person holding menu/sign picture of sandwich. Lỗi bland chips sang bread và long waits thiếu bằng chứng được nhận ra đúng. Visual1/review3 có căn cứ.',[1,3,5,18,20,21,28],list(range(1,36)),['An egg is on the plate of food.','the bread as tough, bland, or otherwise disappointing.']),
('R20','CALIBRATION_DISCUSSION','Không có drink rõ trong8 ảnh, nên có căn cứ trừ Visual. Tuy nhiên một ảnh menu có chicken/romaine/tomatoes trên đĩa nền trắng có hoa, đồng thời ghi Flatbread Pizza; nó không chứng minh salad nhưng hỗ trợ một phần thuộc tính. Mức2 so với3 cần thống nhất cách cân phần đúng/sai và loại món. Ghi chú có uncertainty, chưa phải lỗi nguồn rõ phải buộc đổi.',[2,3,8,12,14,16],[1,3,4,5,6,7,8],['The salad contains chicken.','A drink is on a table.']),
('R21','BROADLY_REASONABLE','Reasonably priced coffee không có cơ sở từ reasonably brewed; omissions early closing/waits/cookie specifics đúng. Visual ice-cream/flowers/coffee/window có thật;4 so với5 là khác nhau hợp lý về contribution, không xác định điểm sai.',[1,4,5,9,20,21,24,28],[6,7,8],['the coffee as reasonably priced or good']),
('R22','BROADLY_REASONABLE','Bắt đúng phủ định và đối tượng greasy/oily; nguồn nói no oil hoặc so với đối thủ. Các phàn nàn thực về soggy/overcooked/onion/bland và thiếu vegan/bowls có thật. Faithfulness2/usefulness2 có thể biện hộ.',[2,7,11,12,16,17,18,19,20,30],[],['some describe it as greasy, oily, or lacking flavor.']),
('R23','BROADLY_REASONABLE','Gom coffee với cold meatballs/unmelted cheese là sai nhãn khía cạnh; lặp hoagie/amazing rõ. Nguồn R1 chỉ nói breakfast sandwiches ready, nên lưu ý không đồng nhất availability với lời khen chất lượng là tốt.',[1,2,3,5,7,9,11,13,15,16,18],[],['Coffee experiences vary: some reviews report cold coffee, cold meatballs and unmelted cheese']),
('R24','CALIBRATION_DISCUSSION','Omission beer/local taps/vegetarian và duplicate inattentive được chỉ đúng. Riêng Non-Redundancy2 chỉ với một cụm lặp cục bộ trong đoạn ngắn có thể hơi nghiêm so với anchor3 noticeable local repetition. Nên thống nhất mức độ, không coi chênh1 điểm là lỗi khách quan.',[2,3,10,11,14,18,19,22,23,24,25,29],[],['slow, inattentive, or inattentive service.']),
('R25','BROADLY_REASONABLE','Bỏ positive workspace/cleanliness và order errors là thật. Atmosphere thường nhỏ/lạnh/loud làm quá mức một số nguồn, nên faithfulness4/coverage3 có thể biện hộ.',[1,2,4,7,10,17,19,20,21,23,24,26],[],['The atmosphere is often described as small, cramped, loud, cold, or unwelcoming']),
('R26','BROADLY_REASONABLE','Tám ảnh không có steak white-plate/wine scene và ghế không xanh như Visual. Score1 và cờ exact span có cơ sở; Review mixed food/service/value/music đúng nhưng thiếu brunch/noise.',[1,4,6,8,12,16,18,19],[1,2,3,4,5,6,7,8],['A steak is sitting on a white plate.']),
('R27','BROADLY_REASONABLE','Chín ảnh không có pizza/wings-grill/taco-pickle/checkered scene. Lặp barbecue sauce và thêm long waits/inattentive counter service không được hỗ trợ rõ; review3/redundancy2/visual1 có cơ sở.',[1,2,8,9,10,14,18,20,22,23,27],list(range(1,10)),['The photo shows a close-up of a pizza.','long waits, inattentive counter service']),
('R28','BROADLY_REASONABLE','Bộ12 ảnh không có outdoor patio umbrellas, many TVs hay drinks group on bar như output. Visual1 là lỗi cảnh lớn; omission Blue Duck fries/noise/BYOB trong Review có nguồn.',[2,4,9,19,25,26,27],[1,2,3,4,5,6,7,8,9,10,11,12],['Many televisions are visible. People are sitting at the bar.']),
('R29','BROADLY_REASONABLE','R19 chê cơ chế trả trước nhưng khen tacos succulent/phenomenal, không nói không đáng giá hay expensive. Người chấm bắt đúng chuyển nghĩa và bỏ service waits/birria/salsa; faithfulness2/usefulness2 có thể biện hộ.',[1,2,6,7,8,9,12,15,18,19,21],[],['others describe the restaurant as expensive or say the food was not worth the upfront cost.']),
('R30','BROADLY_REASONABLE','Nhận xét đúng source imbalance không gian: nhiều người thích đọc/làm việc, sạch, trong khi output tập trung negative seating; recurrent order errors cũng bị bỏ. Faithfulness4/coverage3 có cơ sở.',[1,2,4,7,10,17,19,20,21,23,24,26],[],['The atmosphere is often described as small, cramped, loud, cold, or unwelcoming']),
('R31','BROADLY_REASONABLE','R22 đặt live music ở gần Frenchman street chứ không xác nhận trong Coop’s. Bathroom/cold-dry qualifications bị bỏ, crispy/juicy gộp chicken/jambalaya. Faithfulness3/coverage3 hợp lý.',[1,9,16,18,22,26,27,29],[],['with live music and regulars enjoying the atmosphere.']),
('R32','BROADLY_REASONABLE','Ảnh slaw-sandwich, diners dưới art wall, drink/sunglasses và plate/cutlery hiện diện. Grounding5 có thể biện hộ; Review thiếu Bird & Pig/vegetarian/beer-coffee được nêu đúng.',[2,3,5,10,11,18,19,21,22,23,29],[2,4,6,9,13,15],['The drink is next to a pair of sunglasses.']),
('R33','BROADLY_REASONABLE','Đối chiếu thêm R23 xác nhận nguyên văn burger burned and inedible, nên không được nhầm đây là ghép từ R7 burned food. Review có các món/food-service-value trái chiều; bỏ positive recovery và bottomless fries là vấn đề coverage, mức5/4 có thể biện hộ.',[2,3,4,5,6,7,10,11,14,15,22,23,24,30],[],['though one review reports a burned, inedible burger.']),
('R34','BROADLY_REASONABLE','Đúng là R7 chỉ liệt kê French toast như món thường gặp, không khen cụ thể; nguồn áp đảo negative waits/understaffing nên mixed có thể giảm nhẹ. Người chấm đã ghi giới hạn và trừ faithfulness4; coverage4 hơi rộng nhưng không buộc đổi.',[1,2,3,7,8,12,15,19,20,24,25,28,29],[],['including pancakes, French toast, eggs, hash browns, and bacon']),
('R35','BROADLY_REASONABLE','Dùng lại kiểm tra5 ảnh Jimmy Hula’s: group long table/sign-food/shrimp-rice/interior/two tacos-chips đều có. Review còn thiếu fish-taco/Mojo nhưng mixed themes có nguồn;5 Visual có căn cứ.',[2,5,8,13,16,18,27,28,30],[1,2,3,4,5],['A bowl contains shrimp and rice.']),
('R36','BROADLY_REASONABLE','Đọc đủ30 reviews và34 ảnh: nhấn slow/inattentive service thiếu nguồn rõ khi R5 là pickup chưa sẵn nhưng được rush fix; ảnh có tacos/dumplings/interior/bao nhưng không xác lập người đàn ông ngồi bar. Review3/Visual3 là cách đánh giá có căn cứ.',[1,2,4,5,10,16,17,20,21,22,23,29],[3,4,6,7,10,12,14,17,18,19,21,28,29],['A man is sitting at a bar.']),
('R37','BROADLY_REASONABLE','Wings/Corona đúng ảnh,5 Visual có thể biện hộ cho nội dung hẹp nhưng đúng và hữu ích. Beer mentions tốt/very decent chứ không mediocre; người chấm đã trừ faithfulness4 và bắt homey lặp.',[3,14,18,19,23],[1],['A bottle of corona beer is on a table.','one reviewer found the beer selection mediocre.']),
('R38','BROADLY_REASONABLE','Salad ảnh thể hiện tomato/mushroom chứ không thấy chicken rõ; cảnh broccoli/mushrooms/drinks có. Visual3 có cơ sở nhưng nên giữ loại đồ uống/ingredient ở mức hình thực sự xác lập, không suy chắc cocktail versus wine chỉ từ màu/chân ly.',[1,6,13,14,15,16,19,22,27,30],[1,2,3,4,5,6,7,8],['The salad contains chicken.']),
('R39','BROADLY_REASONABLE','Cùng lỗi Wawa/Gulf và coffee-meatball như R10 được A01 nhận ra. Visual áp dụng nhưng rỗng được chấm1 thay vì N/A là đúng quy tắc;3 versus4 coverage không phải sai nguồn.',[1,2,3,5,7,9,11,15,18],[],['hard to get in and out of because of its limited food or drink options.']),
('R40','BROADLY_REASONABLE','Ảnh stools, sandwich/pickle/slaw, waffles/potatoes/fries/dip khớp. Review chung nhưng bỏ đặc trưng/noise/BYOB nên3 coverage/usefulness có thể biện hộ, không cần hạ faithfulness chỉ vì omission.',[2,4,9,19,25,26,27],[3,5,6,8,9,10,12],['The sandwich contains coleslaw.','The plates include waffles.']),
]


def rows_at(path):
    c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:return [dict(x) for x in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]
    finally:c.close()


def main():
    manifest=read(ROOT/'SAMPLE_MANIFEST.json');previous=read(PRIOR/'SAMPLE_MANIFEST.json')
    old_aliases={x['alias'] for x in previous['records']};items={x['audit_id']:x for x in read(ROOT/'AUDIT_ITEMS.json')}
    sources=read(ROOT/'SOURCES.json');index=read(ROOT/'SOURCE_INDEX.json')
    assert len(items)==len(CASES)==40 and len({x['alias'] for x in items.values()})==40
    assert not old_aliases & {x['alias'] for x in items.values()}
    keys=read(STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json');table={x['alias']:x for x in keys}
    snapshot=rows_at(ROOT/'PRIVATE_SNAPSHOT.sqlite3')
    for rater in ['A01','A02']:
        pool=sorted([x for x in snapshot if x['rater']==rater and x['alias'] not in old_aliases],key=lambda x:table[x['alias']]['position'])
        assert len(pool)==100
        sampled={x['alias'] for x in random.Random('20261102:round2:'+rater).sample(pool,20)}
        assert sampled=={x['alias'] for x in items.values() if x['rater']==rater}
    findings=[]
    for q,verdict,note,revord,phord,spans in CASES:
        x=items[q];s=sources[x['business_id']];text=x['review_output']+'\n'+(x['visual_output'] or '')
        for span in spans:assert span in text,(q,span)
        findings.append({'audit_id':q,'rater':x['rater'],'position':x['position'],'business_id':x['business_id'],'business_name':x['business_name'],
            'alias':x['alias'],'original_version':x['version'],'original_scores':x['rating']['scores'],'original_comment':x['rating']['evidence_notes'],
            'assessment':verdict,'model_assisted_audit_note_vi':note,'exact_output_spans':spans,
            'review_evidence':[{'ordinal':n,'review_id':s['reviews'][n-1]['review_id'],'full_text_sha256':hashlib.sha256(s['reviews'][n-1]['text'].encode()).hexdigest()} for n in revord],
            'photo_evidence':[{'ordinal':n,'photo_id':s['photos'][n-1]['photo_id'],'sha256':s['photos'][n-1]['sha256']} for n in phord],
            'scores_changed':False})
    evidence=[('R07',13,'it again was just over cooked hot dogs'),('R10',1,'Gas is almost always $0.05/g cheaper at the Gulf across the street, but it has very limited food/services.'),
        ('R29',19,'wherein you pay upfront for the food, which was less appealing for me'),('R33',23,'the burger I got because it was burned and inedible.'),
        ('R31',22,'Its near frenchman street with a lot of live music as well.')]
    lookup={x['audit_id']:x for x in findings}
    for q,n,quote in evidence:
        r=sources[items[q]['business_id']]['reviews'][n-1];assert quote in r['text'],(q,quote)
        lookup[q].setdefault('exact_source_quotes',[]).append({'review_id':r['review_id'],'quote':quote})
    (ROOT/'ITEM_LEVEL_AUDIT.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    reused_photos=0;new_photos=0
    prior_sources=read(PRIOR/'SOURCES.json');cfg=read(STUDY/'CONFIG.json')
    for entry in index:
        s=sources[entry['business_id']];assert digest(s)==cfg['sources'][entry['business_id']]['canonical_sha256']
        if entry['source_evidence_reused_from_round1']:
            assert s==prior_sources[entry['business_id']];reused_photos+=len(s['photos'])
        else:new_photos+=len(s['photos'])
        for p in s['photos']:assert file_hash(STUDY/'PHOTOS'/(p['photo_id']+'.jpg'))==p['sha256']==p['original_source_sha256']
    # Verify the shared event journal from the consistent snapshot without invoking
    # any server, save, reopen or synthetic grading against genuine records.
    c=sqlite3.connect((ROOT/'PRIVATE_SNAPSHOT.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    versions={};states={};chain=None;events=[]
    try:
        assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        for row in c.execute('SELECT * FROM events ORDER BY seq'):
            e=json.loads(row['event_json']);item=table[e['alias']]
            blank={'scores':{criterion:'N/A' if criterion=='visual_grounding_and_contribution' and item['visual_text'] is None else None for criterion in ['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']},
                'evidence_notes':'','serious_grounding_issue':False,'issue_span':'','evidence_source_ids':[],'issue_certainty':'none','status':'NOT_STARTED','submitted_at_utc':''}
            old=states.get(e['alias'],blank)
            assert row['event_hash']==digest(e) and e['previous_event_sha256']==chain
            assert e['version']==versions.get(e['alias'],0)+1 and e['rater']==item['annotator'] and old['status']!='SUBMITTED'
            assert e['prior_state_sha256']==digest(old)
            states[e['alias']]=e['payload'];versions[e['alias']]=e['version'];chain=row['event_hash'];events.append(e)
        assert len(states)==len(snapshot)==240
        for row in snapshot:assert states[row['alias']]==json.loads(row['payload']) and versions[row['alias']]==row['version']
    finally:c.close()
    unchanged=rows_at(STUDY/'PRIVATE_STATE/ratings.sqlite3')==snapshot
    counts={r:dict(Counter(x['assessment'] for x in findings if x['rater']==r)) for r in ['A01','A02']}
    correction_check=read(ROOT/'PRIOR_SAMPLE_CHANGE_CHECK.json')
    report={'format':'stage_n_random40_round2_rating_quality_audit_v1','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'sample_records':40,'per_rater':20,'overlap_with_round1_aliases':0,'unique_businesses':25,'source_reviews_in_materials':692,
        'source_scope':'All40 new outputs/comments/scores inspected. Seven new businesses: all174 selected review transcripts read;18 businesses reuse the previous bounded source review with targeted full-review follow-ups. This is not exhaustive verification of all692 reviews/every claim.',
        'photos_newly_viewed_as_labelled_contact_sheets':new_photos,'photo_source_checks_reused_from_round1':reused_photos,
        'individual_photo_reopened':'oaflWoReL4uCmAqIMyG12w','all294_photo_sha256_bindings':'PASS',
        'model_identity_hidden_in_review_materials':True,'reviewer_blind_to_human_scores':False,'classification_counts':counts,
        'new_confirmed_material_source_mismatch_found':[],'calibration_questions':['R12','R20','R24'],
        'original_corrections_user_report':'User reports A02 corrected on original local site.',
        'original_corrections_storage_status':'NOT_OBSERVED: round1 sampled rows unchanged, including A02 positions1/3, versions2/1 and previous comments. Original port8766 process confirmed to use this workspace. No saved amendment seen.',
        'recorded_prior_sample_changes':correction_check['changes'],
        'sqlite_and_full_event_chain':'PASS','event_count':len(events),'journal_head_sha256':chain,
        'scores_changed_by_agent':False,'reopened_records':False,'live_rows_equal_snapshot_after_audit':unchanged,
        'full_agreement_or_model_comparison_run':False,
        'limitations':['Bounded model-assisted plausibility audit, not an independent human gold standard or authorship authentication.',
            'Record-level disjointness does not imply disjoint businesses or independent output text; sources/text may recur across participants.',
            'Adjacent rating choices and contribution/coverage judgments can differ reasonably; calibration questions do not mandate replacement scores.',
            'Do not extrapolate a population correctness rate or claim all240 ratings verified.',
            'The two previously identified A02 issues remain unresolved in the recorded database until the actual saved amendments can be verified.'],
        'sample_manifest_sha256':file_hash(ROOT/'SAMPLE_MANIFEST.json'),'item_level_audit_sha256':file_hash(ROOT/'ITEM_LEVEL_AUDIT.json')}
    (ROOT/'AUDIT_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# Kiểm tra mẫu thứ hai —20 mục mới mỗi người','',
        '**Mẫu mới không trùng:** A01 thêm20 và A02 thêm20 mục, loại toàn bộ40 alias đã kiểm tra lần1 trước khi lấy mẫu, seed `20261102:round2`. Tổng hai lượt đã xem40 mục khác nhau/người (80 bản ghi), không phải80 business.','',
        '**Kết luận trong40 mục mới:** chưa thấy thêm sai lệch nguồn nghiêm trọng, rõ như mục Starbucks cũ. A01 có19 mục nhìn chung có thể biện hộ và1 câu hỏi calibration; A02 có18 mục nhìn chung có thể biện hộ và2 câu hỏi calibration. Đây không phải tỷ lệ chính xác hoặc bằng chứng một người tốt hơn người kia.','',
        '| Người | Mẫu mới | Nhìn chung có thể biện hộ | Trao đổi cách áp dụng rubric | Lỗi nguồn lớn mới được xác định |',
        '|---|---:|---:|---:|---:|','| A01 |20|19|1|0|','| A02 |20|18|2|0|','',
        '## Các điểm nên trao đổi, không tự sửa','',
        '- **A01 mục029 — Gamers Guild, R20:** điểm Visual2 có thể hơi nghiêm. Không có drink rõ, nhưng một số thuộc tính chicken/romaine/tomatoes và đĩa sáng màu xuất hiện ở ảnh menu. Ảnh này đồng thời ghi Flatbread Pizza nên không tự xác lập đây là salad. Nên thống nhất cách cân phần đúng/sai thay vì kết luận2 chắc chắn sai.',
        '- **A02 mục096 — Med Cuisine, R12:** các vật thể phần lớn đúng, nhưng The plate nối đĩa grill/fries/meat sang cake/nuts của ảnh khác. Visual4 có thể hơi rộng; cần thống nhất3 hay4 cho lỗi quan hệ/ghép cảnh.',
        '- **A02 mục004 — The Bricks, R24:** lặp inattentive là thật; Non-Redundancy2 cho một cụm lặp cục bộ có thể nghiêm so với mốc3. Không buộc đổi vì chênh một điểm.','',
        '## Ví dụ có bằng chứng cho việc chấm có cơ sở','',
        '- A01 mục032 — eegee’s, R07: nguồn nói hot dogs over cooked và cheese not melted; output biến thành undercooked/underseasoned. Nhận xét phát hiện đúng cả hai.',
        '- A02 mục013 và A01 mục033 — Wawa, R10/R39: limited food/services thuộc trạm Gulf đối diện; access khó do giao thông/footprint/pump blocking. Cả hai chỉ đúng gán nhầm và chấm Visual áp dụng nhưng rỗng là1.',
        '- A01 mục076 — La Mexicana, R29: nhận xét trả tiền upfront less appealing không đồng nghĩa món đắt/không đáng giá; người chấm chỉ đúng chuyển nghĩa.',
        '- A02 mục007 — Denny’s, R09: bag/food/burger/fries/breakfast/orange juice đều có ảnh, nên Visual5 có căn cứ.',
        '- A01 mục024 — Red Robin, R33: review `mgmw6Ei93UHGLB5qsDwDEQ` thực sự có burger burned and inedible. Không được coi claim này là suy diễn chỉ vì review khác nói chung chung burned food.','',
        '## Bản sửa của hai mục cũ','',
        '**Chưa xác minh được bản sửa đã lưu.** Người dùng xác nhận A02 sửa trên trang local gốc. Tuy nhiên, đã kiểm tra command line tiến trình phục vụ8766: nó đang dùng chính WORKSPACE hiện tại; database vẫn giữ A02 mục001 version2/Visual4 và ghi chú fries/red stools/milkshake, mục003 version1 và ghi chú speakeasy-like không có nguồn. Không thấy bản ghi hay sự kiện sửa mới; không thể ghi hai vấn đề đã được giải quyết. Không tự sửa hoặc mở khóa thay người chấm.','',
        '## Phạm vi và giới hạn','',
        '25 business,692 reviews có trong tài liệu nguồn,294 ảnh liên quan. Lần này xem mới113 ảnh qua bảng có ID và đọc đủ174 reviews của7 business mới; dùng lại nguồn/ảnh đã kiểm tra của18 business (181 ảnh, hash/source nguyên vẹn), đọc lại một số review đầy đủ để giải quyết claim cụ thể. Mở riêng ảnh menu Gamers Guild để kiểm tra tên món/chicken. Không nói đã kiểm chứng lại từng câu của cả692 reviews hoặc mọi ảnh ở độ phân giải gốc.',
        '', 'Tất cả40 output/nhận xét/điểm mới đã được đọc. Tên mô hình không nằm trong tài liệu kiểm tra, nhưng reviewer đã thấy điểm nên không phải audit mù điểm. Nhận định có hỗ trợ mô hình, không chứng nhận danh tính, việc xem nguồn thực tế, AI use hoặc tính đúng của cả240 ratings.',
        '', f"SQLite và toàn bộ{len(events)} sự kiện: PASS;240 bản ghi không đổi trong quá trình audit: {unchanged}. Không chạy agreement/so sánh mô hình, không sửa hoặc điền điểm.",
        '', '## Từng mục','']
    for f in findings:
        lines += [f"### {f['audit_id']} — {f['rater']} mục{f['position']:03d} — {f['business_name']}",'',
            '**'+f['assessment']+'** — '+f['model_assisted_audit_note_vi'],'',
            'Điểm gốc: '+json.dumps(f['original_scores'],ensure_ascii=False),'',
            'Review IDs: '+', '.join(x['review_id'] for x in f['review_evidence'])+'.',
            'Photo IDs: '+(', '.join(x['photo_id'] for x in f['photo_evidence']) or 'Visual N/A hoặc output rỗng/kiểm tra text không cần ảnh')+'.','']
        for evidence in f.get('exact_source_quotes',[]):lines += ['> '+evidence['quote'],'Nguồn: `'+evidence['review_id']+'`','']
    (ROOT/'BAO_CAO_KIEM_TRA_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'classification_counts':counts,'overlap_round1':0,'new_photos':new_photos,'reused_photos':reused_photos,
        'confirmed_new_material_mismatches':0,'prior_corrections':'NOT_OBSERVED','all240_rows_unchanged':unchanged},indent=2))


if __name__=='__main__':main()
