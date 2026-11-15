"""Write the model-assisted, source-grounded review performed in this session.

This script validates exact citations and serializes the manually authored findings;
it does not grade outputs, infer participant identity, or modify study scores.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
from collections import Counter

from s1000_final_eval_core_v1 import read, digest, file_hash

ROOT=Path('outputs/stage_n/random40_rating_quality_audit_20261102_v1')
STUDY=Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')

# Findings were authored after reading the sampled outputs/ratings and source
# materials. Ordinals below resolve to immutable source IDs, never to model IDs.
# Q07/Q24 are demonstrable evidence problems; Q16/Q23 are calibration questions.
FINDINGS=[
('Q01','BROADLY_REASONABLE','Nhận ra đúng việc ghép ý của hai người thành một và lặp lời khen brisket; điểm4/4/3 phù hợp mức lỗi và độ bao quát.',[1,10,14,27],[],['one reviewer found them merely decent and disliked the mac and cheese.']),
('Q02','BROADLY_REASONABLE','Điểm faithfulness4 và coverage/usefulness3 có lý: nguồn có bánh cứng/thịt ít và enoteca/cheese/gelato nổi bật; long waits chưa có chứng cứ trực tiếp rõ.',[1,3,5,20,21,28],[],['some reviewers report long waits or order problems.']),
('Q03','BROADLY_REASONABLE','Giữ Review tốt nhưng thiếu các món bị chê/BYOB. Visual3 là mức có thể biện hộ: sandwich, stools và chicken-waffle hiện diện; lá giống hành xanh, tên parsley không chắc. Không xác nhận thành phần sốt chỉ qua ảnh.',[4,9,19,25,26,27],[1,2,3,7,8,9],['The fries are garnished with parsley.']),
('Q04','BROADLY_REASONABLE','Cụm family-style fried chicken lặp ba lần và thiếu burgers là thật; Visual2 có căn cứ vì bộ ảnh chỉ rõ một ly riêng, không có cảnh hai đồ uống trên bàn. Speakeasy-like thực sự có nguồn, nên faithfulness5 ở mục này không mâu thuẫn với bằng chứng đó.',[4,17,19,21,24],[2,3,5],['family-style fried chicken, family-style fried chicken, and family-style fried chicken.','Two drinks are sitting on a table next to each other.']),
('Q05','BROADLY_REASONABLE','Visual1 và cờ cảnh sai có cơ sở: sau khi xem28 ảnh không thấy ba martini trên bàn; không được suy nhà hàng có bàn ăn ban công chỉ từ ảnh mặt tiền có ban công. Review có lẫn tính từ của gà/jambalaya và thiếu vệ sinh.',[1,9,26,29],list(range(1,29)),['There are three different types of martinis on the table.']),
('Q06','BROADLY_REASONABLE','Nhận xét và điểm nhìn chung hợp lý: BBQ/beer/happy-hour/sports có nguồn, ảnh có cánh gà/Corona, burger/onion-rings/knife và beer. Mức4 hay5 cho cách trình bày có thể khác nhau giữa người chấm.',[1,2,3,4,7,11,19],[1,2,3,5],['A bottle of corona beer is on the table.']),
('Q07','RECHECK_MATERIAL_SCORE','Cần A02 xem lại Visual4/5 và câu nhận xét xác nhận fries/red stools/milkshake. Sáu ảnh nguồn Starbucks chỉ có nội/ngoại thất và đồ uống Starbucks; không thấy fries topped with cheese/bacon, ghế đỏ hay milkshake với chocolate sprinkles. Đây là lệch nguồn rõ, không phải chỉ khác nhau một điểm. Không suy ra nguyên nhân hoặc gian lận.',[10,21,24],list(range(1,7)),['A plate holds french fries.','The restaurant has red stools.','The milkshake has chocolate sprinkles.']),
('Q08','BROADLY_REASONABLE','Đúng là chips bị chê bland chứ không phải bread. Visual có giá rượu, người bên cửa sổ, salmon với topping đỏ và rót bia; sauce được người chấm để ở mức không chắc. Điểm3/3/4 hợp lý.',[3,5,18,21,28],[2,3,6,16,17,33],['the bread as tough, bland, or otherwise disappointing.','Salmon is covered in red sauce.']),
('Q09','BROADLY_REASONABLE','Nguồn khen giá phần lớn và có phàn nàn add-ons, không rõ cơ sở cho some expensive. Bỏ gaming/món đặc trưng và lặp từ là thật. Chips nằm cạnh sandwich trong ảnh, nên Visual3 có thể biện hộ.',[3,8,11,12,14],[1,3,4,6,8],['the food expensive or not worth the cost.','The sandwich has chips on it.']),
('Q10','BROADLY_REASONABLE','Bar/khách/bottles và người phụ nữ nhìn lên có trong ảnh. Menu-next-to-food khó xác lập nên người chấm giữ thận trọng; điểm4 không phải cam kết mọi chi tiết đều đúng. Các thiếu sót Review nêu có nguồn.',[1,9,16,26,27],[1,6,11,14,18,26,27],['The menu is next to a plate of food.']),
('Q11','BROADLY_REASONABLE','Visual1 phù hợp: không thấy enchilada hoặc green drink trên đĩa trong8 ảnh. Lỗi reasonably priced coffee cũng đúng là đổi ý brewed reasonably well thành giá.',[4,5,9,20,28],list(range(1,9)),['The plate contains an enchilada.','the coffee as reasonably priced or good']),
('Q12','BROADLY_REASONABLE','Review tổng hợp đúng sự trái chiều đồ ăn/phục vụ/giá và cảnh quan; việc chưa nêu brunch/noise/side charges là thiếu thông tin chứ không đảo nội dung. Điểm5 faithfulness,4 coverage hợp lý.',[1,2,4,6,8,12,16,18,19],[],['Food opinions are mixed:']),
('Q13','BROADLY_REASONABLE','Ảnh sticks dài bị gọi tater tots và cheeseburger fries không rõ bằng chứng; Visual2 có lý. Ghi chú sạch sẽ cần đọc đúng phạm vi: có nguồn nói clean service, nhưng không xác lập một drive-through well maintained.',[2,5,9,10,15,20],[1,2,3,4,5,6],['Tater tots are sitting on a black plate.','clean, fast, or well maintained.']),
('Q14','BROADLY_REASONABLE','Faithfulness5 vẫn có thể đi cùng coverage/usefulness3: các ý chung đúng nhưng bỏ salad/dressing, Beef Tony/tortellini và ý kiến về sốt ngọt. Không cần ép hạ faithfulness chỉ vì thiếu ý.',[1,5,6,13,14,15,16,19,22,27],[],['Selected reviewers frequently praise the food,']),
('Q15','BROADLY_REASONABLE','Lỗi Visual lớn được chỉ đúng: bộ ảnh có wings, đĩa thịt/rau và quầy bar, không có jar of caramel hoặc sandwich/cup of soup. Review thiếu hệ thống khó dùng/phòng nóng và poor value drinks chưa rõ nguồn.',[17,22,23,25,27],[1,7,9,11],['The tray contains a jar of caramel.','The plate contains a cup of soup.']),
('Q16','CALIBRATION_DISCUSSION','Các quan sát sign/bread/group/cake và nghi vấn chicken trong salad là hợp lý. Riêng Non-Redundancy2 có thể quá nghiêm nếu chỉ dựa trên lặp cấu trúc câu: phần lớn câu vẫn bổ sung thuộc tính mới. Nên thống nhất lặp ý với lặp chủ ngữ; chưa đủ căn cứ tuyên bố điểm2 sai chắc chắn.',[1,13,14,15,16,30],[1,2,5,6,7,8],['The salad contains chicken.','The people are posing for a picture.']),
('Q17','BROADLY_REASONABLE','Nhận ra gán nhạc live của khu Frenchman street cho quán có cơ sở. Phàn nàn toilet lặp nhiều nguồn và cold/dry food bị bỏ. Faithfulness3/coverage3 hợp lý.',[1,9,18,22,26,27],[],['with live music and regulars enjoying the atmosphere.']),
('Q18','BROADLY_REASONABLE','Coverage/usefulness3 hợp lý vì bỏ bread và nhiều đặc trưng market/enoteca. Ảnh charcuterie/bread/cheese/sandwich cầm tay/wine đều có; Visual5 có thể biện hộ, câu wine lặp được trừ ở tiêu chí riêng.',[1,3,5,20,21],[3,5,11,12,15,18,27],['The wine rack is filled with many bottles of wine.']),
('Q19','BROADLY_REASONABLE','Visual2/cờ người không có trong ảnh là đúng sau khi xem cả6 ảnh food-only. Review3 coverage cho thiếu understaffing và độ nặng phản hồi tiêu cực hợp lý. Faithfulness5 chỉ nên hiểu là các claim nêu có nguồn, không chứng nhận tính đại diện hoàn hảo.',[2,3,8,12,15,20,24],[1,2,3,4,5,6],['A group of people is sitting at a table in a restaurant.']),
('Q20','BROADLY_REASONABLE','Người chấm bắt đúng scene fusion: cake/nuts thuộc đĩa tráng miệng, còn onions/tomatoes/fries ở đĩa grill khác. Visual3 cho hỗn hợp chi tiết đúng nhưng quan hệ sai hợp lý. Repetition generous/leftovers rõ.',[4,7,16,17,18,22,26,30],[3,4,6,8],['The plate contains a piece of cake.','The plate contains onions.']),
('Q21','BROADLY_REASONABLE','Review khẳng định chung có nguồn nhưng lược bỏ mắng nhân viên/slow service/mayo. Faithfulness5 và coverage/usefulness3 là cách phân biệt đúng giữa lỗi và omission; flavorful lặp nhẹ.',[1,3,7,12,21,22,23,29],[],['fresh, flavorful, flavorful, or excellent.']),
('Q22','BROADLY_REASONABLE','Ảnh3 là sandwich thịt thái lát chứ không thấy xúc xích hot dog; nhận xét bắt đúng sai tên món. Các cảnh police car/exterior/three drinks đúng, nên Visual3 thay vì hạ mọi thứ về1 là hợp lý.',[7,12,21,22,23],[1,2,3,4,5,6],['The photo shows a hot dog.']),
('Q23','CALIBRATION_DISCUSSION','Visual5 có căn cứ từ6 ảnh. Điểm Non-Redundancy2 cần trao đổi vì nhiều contains/shown là các thuộc tính khác nhau, không tự động là lặp ý nghiêm trọng. Coverage4 cũng hơi rộng khi phản hồi thiếu nhân sự/chờ lâu áp đảo chưa được biểu đạt rõ; khác biệt3–4 là calibration, không kết luận sai khách quan.',[1,2,3,8,15,20,25,28],[1,2,3,4,5,6],['The hamburger contains bacon.','The hamburger contains cheese.','The hamburger contains an egg.']),
('Q24','RECHECK_SOURCE_COMMENT','Cần sửa hoặc giải thích nhận xét speakeasy-like not supported: review i1Iq_i2JoVSzlAeB04AJpA nói nguyên văn very old world, speakeasy kind of look. Thiếu burgers và lặp family-style là đúng. Riêng lỗi ghi chú này chưa chứng minh faithfulness4 phải đổi, vì điểm tổng có thể có lý do khác.',[4,17,19,21,24],[2,3,5],['speakeasy-like']),
('Q25','BROADLY_REASONABLE','Tách tiêu chí tốt: claim thực duy nhất về lời khen Bobbie được nguồn hỗ trợ, nhưng output sụp thành vòng lặp, nên coverage/coherence/non-redundancy/usefulness đều1. Ảnh khay sandwich có thật. Faithfulness5 không tự làm output tốt.',[2,4,7,13,15,20,23],[1,2,3,4,5],['Selected reviewers frequently praise the sandwiches, including Bobbie,']),
('Q26','BROADLY_REASONABLE','Visual1/cờ steak có cơ sở:8 ảnh chủ yếu nhà hàng/quầy bar/khách và cocktails; không thấy steak trên đĩa, ghế không xanh như output. Review đúng các ý chung nhưng thiếu brunch/noise.',[1,4,6,12,16,18,19],[1,2,3,4,5,6,7,8],['A steak is sitting on a white plate.','The dining room has blue chairs.']),
('Q27','BROADLY_REASONABLE','Visual có burger nhưng không có breakfast plate/orange juice;1 là mức có thể biện hộ cho sai cảnh lớn. Review có burger/chicken-teriyaki/kids-cheese/refills/food/service/value được nguồn xác nhận.',[2,3,4,5,6,7,14,15,22],[1,2,3,4,5,6,7,8,9],['A plate of breakfast food is on a table. A glass of orange juice is on the table.']),
('Q28','BROADLY_REASONABLE','Lỗi oily/greasy gán sai đối tượng/phủ định là thật và quan trọng; faithfulness2 có thể biện hộ, không cần trùng mức1 của người khác. Burger/potato salad/man/three tacos/two sauces có ảnh hỗ trợ Visual5.',[2,7,11,12,16,17,18,19,20,30],[1,2,3,4,5,6],['some describe it as greasy, oily, or lacking flavor.']),
('Q29','BROADLY_REASONABLE','Bộ9 ảnh có rice/noodles/chicken/shrimp, không có cảnh plate of wings + Corona. Visual1 và nguồn chung của Review được chấm riêng hợp lý. Coverage3 vì Review rất chung có thể biện hộ.',[1,2,4,5,7,12,19,22,23],[1,2,3,4,5,6,7,8,9],['There is a plate of wings. A bottle of corona beer is on a table.']),
('Q30','BROADLY_REASONABLE','Chỉ đúng gán mayo của Georgioso/Boise Cheese Steak sang pastrami. Sáu ảnh không có khay nhiều sandwich, nên Visual1 có cơ sở dù source review có kể catering: không được lấy review làm bằng chứng ảnh.',[2,7,8,10,12,17,23,24],[1,2,3,4,5,6],['though one reviewer found the sandwich overly smothered in mayonnaise.','A large tray of sandwiches is displayed on a table.']),
('Q31','BROADLY_REASONABLE','Faithfulness3 vì ghép room prices với uncomfortable và thêm difficulty getting attention là hợp lý. Food/rooms/value có nguồn; thiếu song/remote/AC là khoảng trống. Coverage/usefulness4 hơi hào phóng nhưng trong khoảng chủ quan có thể biện hộ.',[1,5,8,17,21,22,23,24,25],[],['the karaoke room prices high or uncomfortable.']),
('Q32','BROADLY_REASONABLE','Nguồn nói location cũ too intense và location này clean/minimalist nên nhận xét gán nhầm đúng. 21 ảnh không xác lập pasta plate, người nữ chuẩn bị món hay3 pizza nhỏ; Visual1 có cơ sở. Một ảnh salad có chicken không đủ chứng minh cả cảnh pasta/chicken.',[13,15,18,19,27,29],list(range(1,22)),['The plate contains pasta. The plate contains chicken.','Three small pizzas are visible.']),
('Q33','BROADLY_REASONABLE','No sushi/chopsticks trong5 ảnh là lỗi cảnh rõ; Visual1 hợp lý. Source khen beer good/very decent, không rõ nguồn cho mediocre beer selection; người chấm trừ faithfulness4 có căn cứ.',[3,14,18,19],[1,2,3,4,5],['There is a tray of sushi.','one reviewer found the beer selection mediocre.']),
('Q34','BROADLY_REASONABLE','Greasy/oily đảo lời khen no oil hoặc nhận xét về đối thủ; faithfulness1 và usefulness2 phản ánh lỗi trung tâm của quán vegan/healthy. Coverage3 có thể biện hộ vì vẫn bao quát price/staff/atmosphere. Không cần xem mức1 so với2 của A02 là sai chỉ vì lệch1 điểm.',[2,7,11,16,17,18,19,20,30],[],['some describe it as greasy, oily, or lacking flavor.']),
('Q35','BROADLY_REASONABLE','Ảnh red-background burger, tall fruit drink/strawberries và soup/salad trên đĩa đều có. Visual5 và Non-Redundancy3 phân biệt grounding với cách diễn đạt. Bỏ good recovery/refill variability là thiếu chi tiết có nguồn.',[4,5,7,10,14,15,22,23],[1,2,6,8],['The hamburger is on a red background.','Strawberries are next to the drink.']),
('Q36','BROADLY_REASONABLE','Cách gọi sushi là hibachi/grill food không chuẩn và bỏ ý overcooked/showmanship có nguồn. Visual quesadillas/person holding bowl/tray drink không được5 ảnh hỗ trợ. Điểm1 Visual hợp lý; không nên khái quát rằng ảnh không có chicken/noodles vì thực tế có.',[2,3,10,12,16,17,18,24],[1,2,3,4,5],['including sushi, rolls, sashimi, and other dishes.','There is a plate of quesadillas.']),
('Q37','BROADLY_REASONABLE','Review có món/portions/patio/service; lặp generous/leftovers/order problems và omission drive-through/price có nguồn. Không có người đang làm salad trong bếp ở8 ảnh; Visual1 có thể biện hộ cho mô tả cảnh sai.',[2,4,7,14,16,17,18,22,26,30],[1,2,3,4,5,6,7,8],['A person is preparing a salad on a plate in a kitchen.']),
('Q38','BROADLY_REASONABLE','Các claim BBQ/bland/service/sports/beers đều có nguồn; bỏ happy hour/dog/patio được nêu đúng. Non-Redundancy3 cho bland, bland hơi nghiêm nhưng không mâu thuẫn rubric đến mức cần đổi.',[1,2,3,7,11,12,14,18,23,27],[],['bland, bland, or lacking flavor.']),
('Q39','BROADLY_REASONABLE','Cùng lỗi reasonably brewed -> reasonably priced được bắt đúng. Dessert sandwich/coffee/flowers/window seating có ảnh.5 Visual có thể biện hộ vì phạm vi mô tả hẹp nhưng đúng; chưa có bằng chứng chấm sai.',[1,4,5,9,20,21,24,28],[6,7,8],['the coffee as reasonably priced or good','An ice cream sandwich is visible.']),
('Q40','BROADLY_REASONABLE','Cả5 cảnh trong Visual đều có ảnh tương ứng và không ghép quan hệ sai rõ. Review cân bằng nhưng thiếu fish tacos/Mojo fries/beer/surfer detail; mức5 faithfulness và4 coverage hợp lý.',[2,5,8,13,16,18,27,28,30],[1,2,3,4,5],['A bowl contains shrimp and rice.','Two tacos and chips are in a basket on a wooden table.']),
]


def main():
    items={x['audit_id']:x for x in read(ROOT/'AUDIT_ITEMS.json')}
    sources=read(ROOT/'SOURCES.json');index=read(ROOT/'SOURCE_INDEX.json')
    assert len(FINDINGS)==len(items)==40 and {x[0] for x in FINDINGS}==set(items)
    full_read_tags={'B01','B03','B04','B05','B06','B07','B08','B09','B10','B13','B15','B18','B21','B22'}
    findings=[]
    for q,verdict,note,review_indices,photo_indices,spans in FINDINGS:
        item=items[q];src=sources[item['business_id']];combined=item['review_output']+'\n'+(item['visual_output'] or '')
        for span in spans:assert span in combined,(q,span)
        review_evidence=[]
        for ordinal in review_indices:
            r=src['reviews'][ordinal-1]
            review_evidence.append({'review_ordinal':ordinal,'review_id':r['review_id'],'full_text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),
                'source_location':'SOURCES.json -> '+item['business_id']+' -> reviews['+str(ordinal-1)+']'})
        photo_evidence=[]
        for ordinal in photo_indices:
            p=src['photos'][ordinal-1];photo_evidence.append({'photo_ordinal':ordinal,'photo_id':p['photo_id'],'sha256':p['sha256']})
        finding={'audit_id':q,'rater':item['rater'],'original_position':item['position'],'alias':item['alias'],'business_id':item['business_id'],
            'business_name':item['business_name'],'original_scores':item['rating']['scores'],'original_comment':item['rating']['evidence_notes'],
            'assessment':verdict,'model_assisted_audit_note_vi':note,'exact_output_spans':spans,'review_evidence':review_evidence,'photo_evidence':photo_evidence,
            'score_changed':False,'independently_authenticated_human_judgment':False}
        findings.append(finding)
    lookup={x['audit_id']:x for x in findings}
    lookup['Q07']['confirmed_audit_problem']={'criterion':'visual_grounding_and_contribution','recorded_score':4,
        'exact_rater_comment_span':'Visual: accurately describes fries, restaurant with red stools, milkshake;',
        'source_observation':'All six original source photographs reviewed; interior/exterior seating/counters and Starbucks drinks. No depicted cheese/bacon fries or red-stool milkshake scene.',
        'requested_action':'Ask A02 to revisit original position1 against the six original photographs and explain the mismatch. Do not change the score automatically.'}
    for q in ['Q07']:
        assert lookup[q]['confirmed_audit_problem']['exact_rater_comment_span'] in items[q]['rating']['evidence_notes']
    source=next(s for s in sources.values() if s['business_metadata']['name']=="Gallagher's Restaurant")
    review=source['reviews'][18];quote='The decor inside is great, very old world, speakeasy kind of look.  I loved it.'
    assert quote in review['text']
    gallagher_review_id=review['review_id'];gallagher_quote=quote
    lookup['Q24']['confirmed_audit_problem']={'exact_rater_comment_span':"'Speakeasy-like' atmosphere is not supported by the reviews",
        'review_id':review['review_id'],'exact_source_quote':quote,
        'requested_action':'Ask A02 to correct/clarify the unsupported criticism in the comment. A faithfulness score of4 is not automatically invalidated by this comment error.'}
    assert lookup['Q24']['confirmed_audit_problem']['exact_rater_comment_span'] in items['Q24']['rating']['evidence_notes']
    # Exact excerpts underpinning key positive checks; references preserve full original text.
    excerpts=[('Q01',1,'The fries are decent but nothing as spectacular as the meat and Mac and cheese.'),
        ('Q01',27,"I think I was most disappointed by the mac and cheese."),
        ('Q17',22,'Its near frenchman street with a lot of live music as well.'),
        ('Q28',7,'Way better than Maoz. No greasy, oily food!!'),
        ('Q34',7,'Way better than Maoz. No greasy, oily food!!'),
        ('Q30',23,'Friendly staff and fresh bread but beware of the "italian" sandwich (Georgioso) which is a few slices of meat and cheese absolutely SMOTHERED in mayo.'),
        ('Q32',15,"A lifetime ago I went to the original storefront and was so deeply turned off by how intensely they embraced the Serial Killer pun.")]
    for q,ordinal,quote in excerpts:
        source=sources[items[q]['business_id']];review=source['reviews'][ordinal-1];assert quote in review['text'],(q,quote)
        lookup[q].setdefault('additional_exact_source_quotes',[]).append({'review_id':review['review_id'],'quote':quote})
    (ROOT/'ITEM_LEVEL_AUDIT.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    stats={r:dict(Counter(f['assessment'] for f in findings if f['rater']==r)) for r in ['A01','A02']}
    photo_count=0
    for s in sources.values():
        assert digest(s)==read(STUDY/'CONFIG.json')['sources'][s['business_id']]['canonical_sha256']
        for p in s['photos']:
            assert file_hash(STUDY/'PHOTOS'/(p['photo_id']+'.jpg'))==p['sha256']==p['original_source_sha256'];photo_count+=1
    # Compare all genuine rows against the consistent snapshot; never writes to live DB.
    def stored(path):
        c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        try:return [dict(r) for r in c.execute('SELECT * FROM ratings ORDER BY rater,alias')]
        finally:c.close()
    assert stored(ROOT/'PRIVATE_SNAPSHOT.sqlite3')==stored(STUDY/'PRIVATE_STATE/ratings.sqlite3')
    report={'format':'stage_n_random40_rating_quality_audit_v1','checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'Model-assisted audit of recorded rating plausibility and source comments, not new human ratings or full agreement/model-comparison analysis.',
        'sample_records':40,'records_per_rater':20,'distinct_businesses':23,'photos_viewed_as_contact_sheets':photo_count,
        'original_photos_reopened_individually':['BrtkXhN0cgy6P1Ar1mKt6Q','nLbOmodyy4Z_PwA-HFnIIA','rBp6n5HsfMPbUm3GwAhd9A'],
        'source_reading':{'full_review_transcripts_read_for_business_tags':sorted(full_read_tags),
            'selected_sentence_and_context_excerpts_read_for_other_business_tags':[x['tag'] for x in index if x['tag'] not in full_read_tags],
            'reviews_in_source_materials':sum(len(s['reviews']) for s in sources.values()),
            'not_a_claim_of_exhaustive_sentence_by_sentence_verification_of_all_668_reviews_or_all_output_claims':True},
        'sampling':'See SAMPLE_MANIFEST.json; separate deterministic random20 of120 per rater; no outcome-based replacement.',
        'model_identity_hidden_in_review_materials':True,'reviewer_blind_to_rater_scores':False,
        'classification_counts':stats,'confirmed_evidence_problems':['Q07','Q24'],'calibration_questions':['Q16','Q23'],
        'human_scores_modified':False,'live_database_rows_unchanged':True,'source_and_all_230_photo_bindings':'PASS',
        'analysis_or_winner_selection_run':False,
        'conclusion':'Most sampled judgments are defensible. A02 position1 has a material Visual-source mismatch; position3 has a false source-absence criticism. These specific cases need participant review; the sample neither certifies all240 ratings nor proves dishonesty/AI authorship.',
        'limitations':['AI-assisted audit judgments are not an independent human gold standard; adjacent1-5 scores can be reasonable disagreements.',
            '40 ratings are not40 independent businesses; recurring business/output records overlap. No population accuracy percentage or rater ranking inferred.',
            'Source checking used14 full business review transcripts and excerpts for9 others; all230 photo files were inspected through labelled thumbnail sheets with3 full-file reopenings, not exhaustive high-resolution inspection.',
            'Account/timestamp records do not independently establish who read the sources or whether AI was used.',
            'No ratings were reopened/replaced. Any accepted amendment requires the participant to review it with a documented reason, not automatic score correction.'],
        'sample_manifest_sha256':file_hash(ROOT/'SAMPLE_MANIFEST.json'),'item_level_audit_sha256':file_hash(ROOT/'ITEM_LEVEL_AUDIT.json')}
    (ROOT/'AUDIT_SUMMARY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# Kiểm tra ngẫu nhiên20 mục A01 và20 mục A02','',
        '**Kết luận:** phần lớn nhận xét/điểm trong mẫu có cơ sở. Không nên xác nhận toàn bộ panel đã hoàn hảo: có2 vấn đề nguồn cụ thể cần A02 xem lại, và2 điểm cần thống nhất cách hiểu rubric. Không sửa điểm gốc.','',
        '| Người chấm | Mẫu | Nhìn chung có thể biện hộ | Cần trao đổi cách áp dụng rubric | Cần xem lại bằng chứng |',
        '|---|---:|---:|---:|---:|','| A01 |20|20|0|0|','| A02 |20|16|2|2|','',
        'Các nhóm trên là đánh giá định tính của audit bằng mô hình, không phải tỷ lệ chính xác được xác nhận hoặc kết luận người nào giỏi hơn.','',
        '## Hai trường hợp cần A02 xem lại','',
        '**1. Starbucks — A02 mục001, Q07.** Visual được chấm4/5 và ghi rằng fries/red stools/milkshake khớp ảnh. Sáu ảnh nguồn thực tế là Starbucks (nội/ngoại thất, quầy/ghế và đồ uống); không có cảnh fries cheese/bacon hay ghế đỏ và milkshake như output. Đây là vấn đề bằng chứng rõ. Cần hỏi người chấm đã đối chiếu đúng bộ ảnh chưa; chưa xác định nguyên nhân và không kết luận gian lận.',
        '', 'Ảnh đối chiếu: `PHOTO_SHEETS/B22_1.jpg`; full ID/hashes có trong `SOURCE_INDEX.json` và `SOURCES.json`.','',
        '**2. Gallagher’s — A02 mục003, Q24.** Nhận xét nói speakeasy-like không được reviews hỗ trợ. Nhưng review `'+gallagher_review_id+'` nói:',
        '', '> '+gallagher_quote,'',
        'Phê bình thiếu burgers và lặp family-style là có cơ sở; riêng phê bình speakeasy cần sửa/giải thích. Chưa đủ lý do tự đổi điểm faithfulness4 chỉ từ một lỗi trong nhận xét.','',
        '## Hai trường hợp nên trao đổi rubric','',
        '- A02 mục106 — Mungo’s, Q16: Non-Redundancy2; nhiều câu lặp chủ ngữ nhưng bổ sung thuộc tính khác nhau. Có thể hơi nghiêm nếu đánh đồng lặp cấu trúc với lặp ý.',
        '- A02 mục070 — IHOP, Q23: Non-Redundancy2 có câu hỏi tương tự; coverage4 cũng cần cân nhắc việc nguồn áp đảo bởi chờ lâu/thiếu nhân sự. Đây là hiệu chỉnh cách hiểu tiêu chí, chưa phải lỗi điểm khách quan.','',
        '## Những gì mẫu làm tốt','',
        '- A01 PlantPure, Q34 và A02 PlantPure, Q28: nhận ra greasy/oily bị đảo phủ định hoặc gán từ đối thủ sang business đang xét. Mức1 và2 đều có thể biện hộ.',
        '- A02 Deli George, Q30: bắt đúng mayo của Georgioso/Boise Cheese Steak bị gán sang pastrami, và không lấy review catering làm bằng chứng khay sandwich trong ảnh.',
        '- A01 DuBa, Q15; A02 Coop’s, Q05; A01 Hong Kong Express, Q29: phát hiện các cảnh/đồ vật sai ảnh rõ ràng.',
        '- A02 Capriotti’s, Q25: chấm rất thấp coverage/coherence/redundancy/usefulness khi output lặp Bobbie hàng loạt, nhưng không hạ mọi tiêu chí một cách máy móc.','',
        '## Phạm vi và giới hạn','',
        'Seed20261102; chọn riêng20/120 mỗi người trước khi xem nội dung. Mẫu bao gồm23 business,668 reviews trong tài liệu nguồn và230 ảnh riêng. Đã đọc toàn bộ transcripts reviews của14 business, đọc trích đoạn/đối chiếu ngữ cảnh ở9 business còn lại; xem tất cả230 ảnh qua bảng ảnh có ID, mở riêng3 ảnh Starbucks. Không tuyên bố đã kiểm chứng từng câu của toàn bộ668 reviews hoặc mọi chi tiết ở độ phân giải gốc.',
        '', 'Reviewer đã thấy điểm/nhận xét nên audit không mù điểm; tên điều kiện/mô hình không được đưa vào tài liệu kiểm tra. Không thể suy danh tính, việc xem nguồn thực tế hay việc dùng AI chỉ từ chất lượng văn phong và log nộp.',
        '', 'Đây là rà soát có hỗ trợ mô hình, không thay thế hai người chấm, không sửa/chấm đè và không chọn mô hình thắng. Toàn bộ240 bản ghi trên trang gốc được so sánh với snapshot và không đổi.','',
        '## Kết quả từng mục','']
    for f in findings:
        lines += [f"### {f['audit_id']} — {f['rater']} mục{f['original_position']:03d} — {f['business_name']}",'',
            '**'+f['assessment']+'** — '+f['model_assisted_audit_note_vi'],'',
            'Điểm gốc: '+json.dumps(f['original_scores'],ensure_ascii=False),
            '', 'Review IDs đối chiếu: '+', '.join(x['review_id'] for x in f['review_evidence'])+'.',
            'Photo IDs đối chiếu: '+(', '.join(x['photo_id'] for x in f['photo_evidence']) or 'Visual N/A hoặc không cần ảnh để xem xét vấn đề nêu')+'.','']
        for evidence in f.get('additional_exact_source_quotes',[]):lines += ['> '+evidence['quote'],'Nguồn: `'+evidence['review_id']+'`','']
    (ROOT/'BAO_CAO_KIEM_TRA_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'classification_counts':stats,'confirmed_problem_ids':['Q07','Q24'],'calibration_ids':['Q16','Q23'],'photos_verified':photo_count,'ratings_unchanged':True},indent=2))


if __name__=='__main__':main()
