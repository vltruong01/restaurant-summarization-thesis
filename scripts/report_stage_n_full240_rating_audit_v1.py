"""Serialize an explicitly authored model-assisted full-record audit, never human grades.

Input snapshots were made with SQLite's read-only connection/backup. This script
does not import the live Study class or invoke a save/reopen/submit endpoint.
The findings below were authored after reading all 30 grouped work packets.
Programmatic assertions validate bindings/coverage, NOT semantic correctness.
"""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from s1000_final_eval_core_v1 import read, digest, file_hash

ROOT = Path('outputs/stage_n/full240_rating_quality_audit_20261102_v1')
STUDY = Path('outputs/stage_n/test30_two_rater_preparation_v1/WORKSPACE')
PRIOR_NAMES = ['random40_rating_quality_audit_20261102_v1', 'random40_rating_quality_audit_round2_20261102_v1']
CRITERIA = ['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']

# Four manually authored contextual findings per business, in frozen group order.
# These are output/source checks used to assess the plausibility of both ratings;
# they are not new 1-5 grades and do not constitute an exhaustive claim annotation.
GROUP_NOTES = {
'D01': [
'Fries decent ở R1 không đi cùng lời chê mac; R27 mới chê mac. Hai người nhận ra đúng lỗi ghép attribution và lặp brisket; chênh mức3/4 không tự là lỗi.',
'Ketchup queue không chứng minh long waits/inattentive counter service. Brisket và sauce lặp rõ; bộ ảnh không xác lập người ngồi bar xem soccer. Điểm thấp/vừa có căn cứ.',
'Uneventful thuộc sandwich, không phải service; brisket lặp. Hộp/quầy/khách có ảnh, người ngồi xem TV không được xác lập. Cần phân biệt lỗi Review với Visual.',
'Pizza/wings-grill/taco-pickle/checkered scene không có trong chín ảnh recipient. Lặp sauces/brisket và service không rõ nguồn làm điểm thấp có thể biện hộ.'],
'D02': [
'Bobbie có khen/chê nhưng merely okay ở R3 là sandwich nhận nhầm. Service/value/setting có nguồn; bỏ shortages/missing items và các sandwich khác. Faithfulness5 của A01 hơi rộng nhưng không coi chênh4/5 là sai chắc chắn.',
'Opening praise được nguồn hỗ trợ nhưng Bobbie lặp rất dài, kết thúc dang dở, thiếu hầu hết thông tin. Soup/martini không có trong năm ảnh sandwich. Faithfulness có thể cao hơn các tiêu chí cấu trúc/coverage.',
'Nhân viên friendly/helpful/personable và sandwich-on-wood bị lặp; food negatives thiếu. Cranberry/slaw/sandwich tray có ảnh. Cả hai tách grounding khỏi repetition đúng hướng.',
'Bobbie loop là output hỏng nhưng không làm lời khen mở đầu thành sai thực tế. Khay sandwich có ảnh. Coverage/coherence/redundancy/usefulness1 và Visual5 không mâu thuẫn tiêu chí.'],
'D03': [
'Reasonably brewed không có nghĩa reasonably priced. Thiếu closing-time/waits/cookie specifics; eggs/bowl/plate relation có imprecision. Điểm4/3 Visual là khác biệt mức độ có thể biện hộ.',
'Quiche Lorraine có lời khen mạnh và một qualification vị ngọt; especially disappointment làm lệch bức tranh. Coffee price chưa được chứng minh; delicious/inattentive lặp và thiếu closing-time pattern.',
'Không có enchilada/green drink/person holding drink; sandwich-salad cùng cảnh không xác lập. Coffee price không có cơ sở rõ, Review thiếu cookie/service specifics. Visual1 có căn cứ.',
'Ice-cream sandwich, coffee/flowers và window seating có ảnh. Review price/omissions có vấn đề. Visual4/5 phản ánh contribution khác nhau, không buộc đổi.'],
'D04': [
'Food/service/value/divelike setting có nguồn; recurring dirty bathroom và cold/dry qualifications bị bỏ. Street/bar/phone/tortilla-style food có ảnh; không cần suy chắc tên ẩm thực từ hình.',
'Live music ở gần Frenchman Street bị chuyển vào Coop. Lời khen chicken/jambalaya gộp thuộc tính; bathroom/cold/dry thiếu. Faithfulness3 có thể biện hộ.',
'Không có three martinis/oyster-soup plate/balcony dining của quán. Review có thiếu bathroom/cold/dry và câu chicken/jambalaya gộp nghĩa; A01 faithfulness5 khá rộng nhưng không thay điểm.',
'Bar/diners/bottles/woman có ảnh; menu-next-to-food chưa rõ. Lặp inattentive/bar và bỏ bathroom/cold-dry. Điểm Visual3/4 có thể biện hộ.'],
'D05': [
'Broad sandwich/food/price/setting có nguồn nhưng chaotically rude service/employee berating và mayo bị bỏ. 4/5 faithfulness khác biệt hợp lý nếu không phạt omission như hallucination.',
'Mayo criticism thuộc Georgioso/Boise Cheese Steak, không pastrami. Dirty setting bị bỏ. Ba drinks có ảnh nhưng Visual hẹp; mức3 contribution có thể biện hộ, không bắt mọi mô tả đúng phải5.',
'P3 là deli sandwich thịt thái/sợi với pickles/mustard/knife, không hot dog. Exterior/police car/umbrellas/drinks có ảnh. Review cân bằng nhưng thiếu mayo/employee berating.',
'Không có large tray sandwiches trong sáu ảnh; có ingredients và một sandwich. Review gắn mayo sai món và thiếu dirty setting. Điểm thấp có căn cứ.'],
'D06': [
'Review mixed food/service/value có nguồn nhưng order errors/cleanliness thiếu. Coffee ở cạnh, không nằm trên plate; burger/fries và plate lặp. Điểm3 Visual/redundancy thấp có căn cứ.',
'Bag/food/burger/fries/breakfast/orange juice có ảnh. Review bỏ recurring order mistakes/facility issues. Có thể faithfulness5 nhưng coverage3.',
'Không có bowl/salad plate/red-background burger/strawberries beside drink như cảnh asserted; food/drinks chung chỉ khớp một phần. Review generic và thiếu order errors/cleanliness.',
'Food/service/value/facility được bao quát, nhưng breakfast basket từ eggs in a basket và several clean hơi rộng; recurring order errors thiếu. Visual N/A đúng.'],
'D07': [
'Karaoke/service/rooms/value nhìn chung có nguồn; poor-value drinks yếu hơn pricey room criticism. Bỏ remote/song/hot-room. Bar/bartender/crowds/skeleton/meat-potato-veg có ảnh; header Visual trùng.',
'Slow food không tự chứng minh difficulty getting attention; prices uncomfortable sai kết hợp nghĩa. Lặp attentive và bỏ remote/song limits. Điểm3/4 cần được hiểu là mức độ, không lỗi dữ liệu.',
'Wings/chips/karaoke/value có nguồn, bỏ song/hot-room. Singing/room/bar/fried chicken/meat-potato-veg có ảnh; câu ngắn theo ảnh không tự là repeated idea.',
'Không có dessert/caramel/sandwich-soup/drink close-up. Bartender là phần khớp. Review bỏ dish/system/temperature specifics. Visual1 có cơ sở.'],
'D08': [
'Family-style chicken lặp ba lần; burger/sides/drinks thiếu. Bar/interior có ảnh nhưng không có hai drinks cùng bàn, chỉ một glass close-up; Visual2/3 khác biệt mức độ.',
'Lặp chicken và bỏ burgers/drinks là đúng; dark beer/interior có ảnh. Riêng ghi chú A02 nói speakeasy không có nguồn là sai: R19 viết đúng cụm đó. Không tự suy mọi điểm của bản ghi sai.',
'Too good to justify price đảo not good enough; merely okay/greasy thuộc cùng một người ở R27. Family-style lặp. Photos dark drink/salad/interior/street có; parmesan loại cụ thể còn chưa chắc.',
'Food/service/value/old-world-speakeasy có nguồn, nhất là R19. Sunday/drinks/person-specific service là omissions, không biến lời có nguồn thành hallucination.'],
'D09': [
'Tuna Salad Deluxe là chữ trên poster, không trên plate. Salad/pita có ảnh; thiếu creative menu và negative prep. Điểm4 Visual có căn cứ.',
'Expensive rộng hơn missing paid add-ons/refill charge; food details thiếu, casual/inattentive lặp. Chips ở cạnh chứ không trên sandwich. Visual3/4 có thể khác nhau hợp lý.',
'Broad food/service/value/gaming có nguồn nhưng thiếu donut/Cuban và raw chips/cold panini. Casual lặp một lần; usefulness/coverage3 hợp lý.',
'Không có drink. Ảnh menu chicken/romaine/tomato nền sáng đồng thời ghi Flatbread Pizza, không chứng minh salad. Cần giữ uncertainty và calibration2/3, không gọi mọi thuộc tính chicken là bịa.'],
'D10': [
'Bland thuộc chips, không bread; long waits không trực tiếp xác lập. Không có egg/person holding menu/sandwich sign. Điểm thấp Visual và faithfulness3 có căn cứ.',
'Named sandwich/service/value có nguồn, nhưng long waits yếu; thiếu butcher/charcuterie/cellar/gelato và tough bread. Grounded nhưng generic không bắt phải hạ faithfulness như coverage.',
'Wine/counter diner/beer pouring/red-topped salmon có ảnh, sauce identity chưa chắc. Review nhầm bland chips thành bread và lặp Batali; điểm Visual4 hợp lý.',
'Boards/handheld sandwich/wine racks có ảnh, rack sentence lặp nhẹ. Review thiếu breadth/negative bread; faithfulness4/coverage3 có thể biện hộ.'],
'D11': [
'Mixed food/service/value và course/music có nguồn; brunch/noise trái chiều thiếu. Drinks/course/diners/building có ảnh; highland thay Highlander là imprecision nhỏ, Visual4/5 có thể biện hộ.',
'Balanced broad food/service/patio/value có nguồn; brunch/music-noise/side charges thiếu. Visual N/A đúng.',
'Không có steak-white-plate-wine và blue chairs. Review broad đúng nhưng thiếu brunch/noise. Visual1 có căn cứ, không phạt Review vì Visual sai.',
'Patio/course và mixed food/service/value có nguồn; brunch/noise/side charges thiếu. Bar/cart/building/path/diners có ảnh. Redundancy4/5 khác nhau không buộc thay.'],
'D12': [
'Sushi/rolls không phải grill food; food-quality/showmanship negatives thiếu và inattentive lặp. Flames/chicken-rice-noodles/boat/menu/store có trong năm ảnh.',
'Cùng imprecision sushi-grill, bỏ food quality và showmanship; flames/menu/chicken-noodles-rice có ảnh. Visual4/5 khác nhau do contribution không tự là sai.',
'Positive food framing được hỗ trợ nhưng thiếu recurrent overcooked/mediocre/showmanship. Giá/service/setting mixed; faithfulness4/5 khác mức nhưng coverage3 có căn cứ.',
'Không có quesadilla/person holding bowl/tray-drink, có hibachi/sushi/menu/store. Review food/showmanship negatives thiếu. Visual1 có cơ sở.'],
'D13': [
'Broad mixed food/value có nguồn nhưng generous portions/signature noodles thiếu, cleanliness chỉ tích cực. Coverage3 tách omission khỏi claim faithfulness.',
'Noodles/rice/chicken/shrimp/nuts có ảnh, bowl/plate và condiment classification có imprecision. Repeated plate/contains rõ nhưng mỗi câu còn có thêm thuộc tính, mức2/3 tùy severity.',
'Không có wings/Corona trong chín Chinese-food ảnh; dark condiment không chứng minh nhãn bia. Review supported nhưng generic. Visual1 có căn cứ.',
'Noodles-shrimp-veg ảnh đúng nhưng narrow coverage; Review thiếu signature dishes/generous portions. Visual4 hợp lý; không đòi mô tả mọi ảnh.'],
'D14': [
'Có food/service/cleanliness positives và negatives nhưng mixed giảm nhẹ negative predominance/understaffing/order errors. Blueberry pancakes/cream và burger/onion rings có ảnh.',
'Specific forgotten pancakes/undercooked eggs có nguồn, dominant understaffing chưa đủ. Visual food claims khớp; nhiều contains thêm thông tin, NonRed2 của A02 cần calibration, không tự xác định sai.',
'Diners/people không có trong sáu food photos; food/drink có một phần đúng. Review underrepresents negative prevalence và thiếu understaffing/order errors. Visual2 có căn cứ.',
'French toast chỉ menu-listing, không praise cụ thể; long waits/sanitation đúng nhưng dominance thiếu. Visual N/A đúng.'],
'D15': [
'Mixed food/service/value và worn seats có nguồn; fish-taco/Mojo/surfer specifics thiếu. Group long table và food by sign đúng; basket gọi plate là nhẹ.',
'Không có hamburgers/fries/condiment menu trong năm source photos. Review balanced và có nguồn; Visual1 không làm Review sai.',
'Fish merely okay từ một trải nghiệm bị kể hai người, Killer lặp. R8 khen food fresh/tasty và liệt kê Mojo nên freshness với fries chỉ yếu, không kết luận hallucination chắc chắn. Wider negative prep thiếu.',
'Năm cảnh group/sign/shrimp-rice/interior/tacos đúng nguồn. Review broad balanced nhưng signature items/beer/decor thiếu. Visual5 có thể biện hộ.'],
'D16': [
'Beer mediocre không thấy hỗ trợ; nguồn beer tích cực. Không có sushi/chopsticks trong năm ảnh. Homey lặp; Visual1 và Review3/4 có căn cứ.',
'BBQ/bland/service/sports/beer có nguồn, bland/sports lặp. Happy hour/portions/patio/dog thiếu là coverage, không buộc hạ faithfulness.',
'Wings/Corona đúng một ảnh; narrow mà có ích vẫn có thể Visual5. Beer mediocre không có nguồn, homey/sports lặp, phần review phải trừ riêng.',
'BBQ/service/beer/happy hour/sports supported; parts/portions/patio thiếu. Wings/Corona/burger/knife/onion rings có ảnh. Scene sentences có repetition nhưng Visual grounding cao hợp lý.'],
'D17': [
'Upfront less appealing có nguồn nhưng plural some và social-distance wording lệch; birria/salsa/margarita thiếu. Taco/sauce/cinnamon/iced drink/burrito có ảnh, coffee identity chưa chắc.',
'Không có interior/chalkboard/color lights trong17 food/drink photos. Expensive không từ upfront preference; social không từ distancing. Review3/Visual1 có căn cứ.',
'Không thích pay-first không đồng nghĩa expensive/not-worth-cost. Thiếu birria/salsa và slow/inattentive service; lặp adjectives. Thang2/3 faithfulness khác nhau có thể biện hộ.',
'Expensive không rõ nguồn; birria/consomme/salsa thiếu. Blue-straw drink có pearls nên coffee chưa chắc; taco/color plate đúng. Visual3/4 mức độ, không tự sửa.'],
'D18': [
'Food/dips/portions/service/patio có nguồn; price/dry takeout/drive-through thiếu. Order mistakes/problems lặp nhẹ; source-grounded không phải đầy đủ.',
'Không có person preparing salad/kitchen trong tám ảnh; grill platter không xác lập sandwich. Review lặp generous/leftovers và thiếu price/drive-through. Visual1 có căn cứ.',
'Mixed grill và cake/nuts nằm ở các ảnh khác nhau; The plate nối thành một plate là relation fusion. Nhận xét/Visual3 và redundancy thấp có cơ sở; Review được chấm riêng.',
'Patio/grill/fries/cake/nuts có nguồn nhưng The plate nối dessert với grill, không chỉ một omission. Cả hai Visual4 cần calibration severity; không thay điểm bằng điểm AI.'],
'D19': [
'Mushrooms/tomato salad không xác lập chicken; drinks/meat/broccoli có ảnh. Review generic bỏ mixed sweet sauce và signature dishes. Visual3 có căn cứ, tránh suy exact drink bằng màu.',
'Không có pizza/checkered restaurant interior trong tám ảnh. Review broad có nguồn nhưng thiếu sweet-sauce/quality negatives. Visual1 và coverage3 có cơ sở.',
'Little Tony sign/oven bread/posed group/cake có ảnh; chicken salad không rõ. Review bỏ sauce/dishes; NonRed2 cho generic photo clauses của A02 cần calibration repeated syntax so với repeated meaning.',
'Broad food/service/value/setting có nguồn nhưng mixed sauce và salad/Beef Tony/tortellini thiếu. Coverage3 không đồng nghĩa claim faithfulness thấp.'],
'D20': [
'Nhiều scene Visual không có: menu/collage/whipped-cream dessert. P25 là coconut drink có straw; nhận xét A01 gọi spoon là chi tiết cần đính chính nhưng không làm Visual1 mất căn cứ. Review thêm service negative không đủ nguồn rõ.',
'Food negatives được giữ một phần; pickup chưa sẵn được rush fix không tự chứng minh inattentive service. Tacos/dumplings/interior/bao có ảnh, man seated at bar không xác lập. Visual3 hợp lý.',
'Avocado/mural/two sushi trays/three bao có ảnh; không xác lập người đàn ông ngồi bar. Service slow/inattentive vượt rõ pickup-not-ready. A02 ghi chỉ xem8/34 ảnh nên phải kiểm tra phần còn lại trước coi mục này đầy đủ.',
'Sushi/poke/service/value đúng broad nhưng actual mediocre/gummy/not-fresh/cold-takeout và flies/smoke thiếu. Coverage3 hợp lý; Visual N/A đúng.'],
'D21': [
'Không có glass-bowl chicken salad/broccoli-mushroom/two-tray-wine scenes; nguồn seafood/oysters/bread/diners. Review thiếu oyster praise và food prep negatives. Visual1 hợp lý.',
'Shrimp sauce/oysters/bar/dining room/four-food collage có ảnh. Review đúng themes nhưng thiếu oyster quality và food prep negatives. Visual5 có căn cứ.',
'BBQ/service/waits/value/setting có nguồn nhưng oyster quality, cash tokens/watery shrimp/overcooked fish thiếu. Coverage3/4 là khác mức; không buộc sửa.',
'Oysters/dining room/man holding shrimp plate đúng nguồn; Review thiếu recurring oyster/quality/waits. Visual3 đối với5 có thể bàn contribution, nhưng đúng ảnh không mặc định phải mọi rater cho5.'],
'D22': [
'Greasy/oily đảo no/little oil và gán lời về đối thủ vào recipient. Không có iced-coffee/laptop/handheld-sandwich/sign/blue-wall scene. Điểm thấp có căn cứ.',
'Positive broad claims đúng nhưng thiếu vegan/no-oil/bowls và actual negatives. Food spread/soup/burger/potato salad/diner có ảnh; laptop loại thiết bị chưa đủ rõ, giữ uncertainty cho Visual3/5.',
'Greasy/oily là lỗi phủ định/đối tượng thật. Actual soggy/overcooked/onion/bland khác hẳn; defining vegan menu thiếu. Faithfulness1/2 đều có thể biện hộ.',
'Ảnh uQpYAcN4bI5bldsaxFpFgA có các vỏ taco chồng nhau; không dùng lượt đếm hình này làm gold standard chắc chắn giữa hai và ba. A01 nói hai, A02 nói ba: giữ uncertainty và xem ảnh gốc. Review greasy/oily vẫn là lỗi thật.'],
'D23': [
'Food/service/value có nguồn nhưng access/cold food/order/ownership thiếu. Burger/Coke/wings có ảnh; pastry trên napkin không xác lập. Visual3 có căn cứ.',
'Strawberry topping trên dessert không phải separate cup beside regular fries. Burger/Coke/wings đúng; Review broad và thiếu order/access/ownership specifics. Visual3 có căn cứ.',
'Mozzarella sticks không tater tots; seasoned fries không cheeseburger fries; strawberry sauce scene sai. Well-maintained drive-through quá rộng từ service improvement. Điểm Visual2 có cơ sở.',
'Mixed food/service/value/run-down setting có nguồn, lặp inattentive/run down. Misses cold food/closures/ownership improvements; coverage3/4 khác mức.'],
'D24': [
'Greasy không thấy rõ trong selected reviews, cold/burned/inedible có. Bowl/salad/night exterior/burger toppings có ảnh. Faithfulness4 và Visual5 có căn cứ.',
'R23 thực nói burned and inedible burger, không được chê claim này là ghép suy diễn. Red burger/fruit drink/soup-salad có ảnh; Review thiếu positive recovery/refill variability.',
'Không có bag-food/breakfast/orange juice như asserted; burger/fries chỉ khớp một phần. Review mixed food/service/value supported. Visual1 không cần kéo Review xuống.',
'Food/service/value/setting grounded; bottomless-fries/refill/manager recovery thiếu. Redundancy4/5 hoặccoverage4 là khác mức có thể biện hộ.'],
'D25': [
'Specific mixed food/service/value được giữ nhưng extensive craft beer thiếu. Red stools/pizza/diners/beer có ảnh; lifted slice from pan và bar-eating relation chưa chắc nên Visual4 hợp lý.',
'Too intense thuộc storefront cũ ở R15, được đối lập current location. Hot dogs trong Visual thực là cheesesteak portions; other restaurant scenes có. Beer/quality negatives thiếu, điểm thấp có căn cứ.',
'Broad mixed food/service/setting/value có nguồn nhưng extensive craft beer/horror menu identity thiếu. Redundancy4 và coverage3/4 có thể biện hộ.',
'Không có pasta-chicken/woman preparing/three small pizzas; R15 old-location attribution sai. Beer và negative food thiếu; Visual1 có cơ sở.'],
'D26': [
'Review bỏ nhiều positive work/study/clean sources và drink accuracy; atmosphere negative-only làm lệch bức tranh. Mixed staff/drive-through có nguồn; faithfulness4/coverage3 có thể biện hộ.',
'Drinks/diners/interior/counter có ảnh; Review negativity overgeneralized, workspace/accuracy thiếu và efficient lặp. Visual4/5 do contribution khác nhau có thể biện hộ.',
'Drinks/green umbrellas/interior/diners có ảnh; nhiều generic people/restaurant/table clauses lặp, redundancy2 hợp lý. Review omissions/imbalance được nhận ra.',
'Sáu Starbucks ảnh không có fries-cheese-bacon/red stools/milkshake. A02 ghi accurately describes là sai nguồn và Visual4 cần xem lại; A01 Visual1 có căn cứ. Ít nhất hai review nói small/cold/loud, nên ghi chỉ một cũng quá hẹp.'],
'D27': [
'Stools/slaw-pickle sandwich/waffles/potatoes/fries/dip có ảnh. Review thiếu duck-fries negatives/noise/BYOB. Coverage3/4 vàVisual4/5 khác mức nhưng không tự là sai.',
'Creative food/Blue Duck fries mixed/service/noise/value được giữ, BYOB/dishes còn thiếu. Lời khen không làm narrative dài thành grounding hơn; điểm cao có căn cứ.',
'Không có many TVs/patio umbrellas/drinks-on-bar/candle scene. Review thiếu fries/noise/tight seating/BYOB. Visual1 có căn cứ.',
'White sauce/green onions không xác lập gravy/parsley; sandwich/stools/chicken-waffles đúng. Mixed fries/noise có nguồn, BYOB/dish specifics thiếu. Visual3 hợp lý.'],
'D28': [
'Sandwich/brick exterior/dim bar có ảnh; laptop vs POS uncertain và người seated relation chưa rõ. Review signature Bird&Pig/vegetarian/coffee-beer thiếu. Visual3/4 có thể biện hộ.',
'Duplicate inattentive có thật; one local duplicate không mặc định severe redundancy. A02 NonRed2 cần calibration; food/service/value supported, omitted beer/vegetarian.',
'Không có metal-bowl fries/chicken wings-celery scenes; bar bottles có phần đúng. Review food/service/value grounded nhưng signature/vegetarian/beer-coffee thiếu. Visual1 có căn cứ.',
'Slaw sandwich/art-wall diners/drink-sunglasses/food-knife có ảnh. Generic Review omission và local repetition được nhận ra. Visual5 có cơ sở.'],
'D29': [
'Ảnh Wawa, không Walmart. Coffee stayed not hot; hoagie barely hot không là positive hot contrast. Coffee với meatballs/cheese lẫn aspect, pump blocking đổi nghĩa và hoagie/cars lặp.',
'Limited food/services thuộc Gulf đối diện, access khó do traffic/footprint/pump blocking. Coffee với meatballs sai aspect. Applicable Visual rỗng phải chấm1, không N/A.',
'Cùng Wawa-Gulf causal/attribution và coffee-aspect error; Visual áp dụng nhưng rỗng. Hai người chấm1 đúng quy tắc, coverage3/4 không là sai nguồn.',
'Coffee-meatball/cheese/hoagie conflation và massive hoagie/amazing repetition rõ. Availability breakfast không bằng praise quality. Source concerns về pumps thiếu; điểm thấp có căn cứ.'],
'D30': [
'R13 viết OVERcooked và cheese not melted, không undercooked/underseasoned. Expensive claim rộng hơn dressing charge. Salad/sandwich/drink có ảnh, useful signature slush/ranch thiếu.',
'Không có Tuna Salad Deluxe/black plate/pita; Review còn đảo overcooked/cheese. Visual1 và faithfulness2 có nguồn, không do chỉ khác phong cách.',
'Sandwich/drinks có ảnh, exact mayonnaise/chicken nhận dạng nên thận trọng. Review thiếu signature fruit slush và dirty setting; expensive chưa rõ. Điểm3/4 hợp lý.',
'Signature eegee slush/ranch và dirty floor/bathroom bị bỏ; expensive rộng hơn dressing-charge source. Mixed food/service có nguồn. Coverage2/3 tùy mức omission, không phải bằng chứng sai chấm.'],
}

# Contextual review ordinals actually consulted across the two earlier audits,
# their targeted rereads, and this pass. Full unedited sources are in SOURCES.json.
REVIEW_ORDINALS = {
'D01':[1,4,10,14,20,27], 'D02':[2,3,7,12,14,15,18,21,23,27],
'D03':[1,4,5,9,20,21,24,27,28], 'D04':[1,9,16,18,22,26,27,29],
'D05':[1,7,12,21,22,23,29], 'D06':[1,2,3,4,5,6,9,10,12,13,14,15],
'D07':[1,10,17,23,24,25,27], 'D08':[1,2,5,8,18,19,27],
'D09':[2,3,8,12,14,16], 'D10':[1,3,5,18,20,21,27,28],
'D11':[1,4,6,8,12,16,18,19], 'D12':[1,5,10,12,16,17,23,25],
'D13':[1,2,4,5,7,12,19,22,23,24], 'D14':[1,2,3,7,8,12,15,19,20,24,25,28,29],
'D15':[2,5,8,13,16,18,27,28,30], 'D16':[3,14,18,19,23],
'D17':[1,2,6,7,8,9,12,15,18,19,21], 'D18':[2,4,7,14,16,17,18,22,26,30],
'D19':[1,6,13,14,15,16,19,22,27,30], 'D20':[1,2,4,5,10,16,17,20,21,22,23,29],
'D21':[1,2,3,7,9,10,15,18,19,21,22,23,24,28], 'D22':[2,7,11,12,16,17,18,19,20,30],
'D23':[2,3,5,9,10,15,20], 'D24':[2,3,4,5,6,7,10,11,14,15,22,23,24,30],
'D25':[2,5,7,8,12,15,18,20,23,26,30], 'D26':[1,2,4,7,10,17,19,20,21,23,24,26],
'D27':[2,4,9,19,25,26,27], 'D28':[2,3,5,10,11,14,18,19,21,22,23,24,25,29],
'D29':[1,2,3,5,7,9,11,13,15,16,18], 'D30':[2,4,6,12,13,16,19,24],
}

SPECIAL = {
('A01',19):('UNCERTAIN_VISUAL_COUNT','OPTIONAL',
 'A01 nói hai taco còn output/A02 nói ba. Vỏ taco chồng nhau khiến đối chiếu số lượng chưa đủ chắc để quy A01 sai. Giữ uncertainty; nếu xem lại, căn cứ ảnh gốc và toàn bộ Visual chứ không tự sửa3 thành5. Lỗi oily/greasy trong Review được A01 nêu đúng.'),
('A01',38):('COMMENT_DETAIL_CAUTION','LOW',
 'Phần nhận ra wrong menu/collage/whipped-cream dessert hợp lý. Riêng coconut ở P25 có straw, không thấy spoon rõ như ghi chú A01. Đây là chi tiết nhận xét cần làm chính xác, không tự đảo kết luận Visual1.'),
('A01',79):('CALIBRATION_DISCUSSION','OPTIONAL',
 'Visual4 trong khi The plate nối grill sang cake/nuts ở ảnh khác; cần thống nhất cách cân relation fusion. Không áp một điểm AI làm gold standard.'),
('A02',1):('CONFIRMED_SOURCE_SCORE_MISMATCH','HIGH',
 'Comment xác nhận fries/red stools/milkshake không thuộc sáu ảnh Starbucks; Visual4 dựa trên đối chiếu đó cần xem lại. Bản cũ version2 vẫn tồn tại; không thấy saved amendment. Không tự thay điểm.'),
('A02',3):('CONFIRMED_COMMENT_SOURCE_ERROR','HIGH',
 'Comment phủ nhận speakeasy-like sai với review i1Iq_i2JoVSzlAeB04AJpA. Sửa căn cứ nhận xét; không tự kết luận faithfulness4 sai vì còn các khía cạnh khác. Version1 chưa có amendment.'),
('A02',5):('RECHECK_PARTIAL_SOURCE_INSPECTION','HIGH',
 'Chính ghi chú giới hạn ảnh ở8/34; cần đối chiếu26 ảnh còn lại. Mural và ba bao có ảnh P19/P29; man at bar chưa được nguồn xác lập. Pickup not ready/rush fix không hỗ trợ rộng cho inattentive service; faithfulness5 cần giải thích lại. Không chứng nhận đã hoàn tất source check.'),
('A02',55):('UNCERTAIN_VISUAL_COUNT','OPTIONAL',
 'Ghi chú ba taco được audit cũ xem là verified; ở lượt này giữ mức thận trọng do các vỏ chồng nhau và A01 đếm hai. Không coi số lượng đã được xác nhận độc lập hoặc bắt đổi Visual5 chỉ vì hai người khác nhau. Phần phát hiện oily/greasy trong Review vẫn có cơ sở.'),
}

CALIBRATION = {
('A01',29):'Gamers: no drink là đúng nhưng ảnh menu chicken-flatbread hỗ trợ một phần thuộc tính; cân2/3, giữ uncertainty loại món.',
('A02',4):'Bricks: duplicate inattentive là thật; mức2 với một lặp cục bộ cần đối chiếu anchor3, không tự sửa.',
('A02',70):'IHOP: nhiều plate/contains thêm thuộc tính mới, không đều là lặp ý. Mức2 cần cân repeated syntax so với repeated information.',
('A02',96):'Med: same-plate fusion grill/dessert có thể nặng hơn minor imprecision; cân3/4 nhưng không thay bằng AI score.',
('A02',106):'Mungo: repeated photo/people/salad clauses có thêm thông tin. Cân mức2 cho repeated syntax versus repeated meaning.',
}


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def sql_rows(path, query):
    c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    try:return [dict(x) for x in c.execute(query)]
    finally:c.close()


def main():
    groups=read(ROOT/'GROUPED_ITEMS.json');sources=read(ROOT/'SOURCES.json');cfg=read(STUDY/'CONFIG.json')
    private=read(STUDY/'COORDINATOR_ONLY/PRIVATE_ITEMS.json');key={x['alias']:x for x in private}
    for name,expected in cfg['file_bindings'].items():assert file_hash(STUDY/name)==expected
    assert set(read(STUDY/'COHORT30.json')['business_ids'])==set(sources)
    snapshot=sql_rows(ROOT/'PRIVATE_SNAPSHOT.sqlite3','SELECT * FROM ratings ORDER BY rater,alias')
    snap_by={x['alias']:x for x in snapshot}
    assert len(groups)==120 and len(GROUP_NOTES)==30 and len(snapshot)==240 and len(key)==240
    old_rows={}
    old_sources={}
    for name in PRIOR_NAMES:
        folder=ROOT.parent/name
        old_rows[name]={x['alias']:x for x in sql_rows(folder/'PRIVATE_SNAPSHOT.sqlite3','SELECT * FROM ratings ORDER BY rater,alias')}
        old_sources.update(read(folder/'SOURCES.json'))
    assert set(sources)==set(old_sources)==set(cfg['sources']) and len(sources)==30
    source_receipts=[]
    for bid,s in sources.items():
        assert s==old_sources[bid] and digest(s)==cfg['sources'][bid]['canonical_sha256']
        assert file_hash(STUDY/cfg['sources'][bid]['file'])==cfg['sources'][bid]['file_sha256']
        assert s==read(STUDY/cfg['sources'][bid]['file'])
        assert s['business_id']==bid and len(s['selected_review_ids'])==len(set(s['selected_review_ids']))
        assert s['selected_review_ids']==[r['review_id'] for r in s['reviews']]
        assert all(r['business_id']==bid for r in s['reviews'])
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        for p in s['photos']:
            fp=STUDY/'PHOTOS'/(p['photo_id']+'.jpg')
            assert file_hash(fp)==p['sha256']==p['original_source_sha256'] and fp.stat().st_size==p['bytes']
        source_receipts.append({'business_id':bid,'canonical_sha256':digest(s),
            'file_sha256':cfg['sources'][bid]['file_sha256'],'reviews':len(s['reviews']),
            'review_block_sha256':s['review_block_sha256'],
            'photo_hashes':[{'photo_id':p['photo_id'],'sha256':p['sha256']} for p in s['photos']]})
    findings=[];reused=0;reused_payloads=[];flagged=0
    for g in groups:
        tag=g['business_tag'];n=int(g['group_id'][-1]);context=GROUP_NOTES[tag][n-1];s=sources[g['business_id']]
        assert g['output_previously_inspected']==any(r['reuse_previous_audit'] for r in g['ratings'])
        assert len(g['ratings'])==2 and {r['rater'] for r in g['ratings']}=={'A01','A02'}
        text=g['review_output']+'\n'+(g['visual_output'] or '')
        recipient_ids={r['review_id'] for r in s['reviews']}|{p['photo_id'] for p in s['photos']}
        for r in g['ratings']:
            p=r['rating'];item=key[r['alias']];saved=snap_by[r['alias']]
            assert item['annotator']==r['rater'] and item['position']==r['position'] and item['business_id']==g['business_id']
            assert item['review_text']==g['review_output'] and item['visual_text']==g['visual_output']
            assert hashlib.sha256(item['review_text'].encode()).hexdigest()==item['review_text_sha256']
            assert (None if item['visual_text'] is None else hashlib.sha256(item['visual_text'].encode()).hexdigest())==item['visual_text_sha256']
            assert saved['version']==r['version'] and json.loads(saved['payload'])==p and saved['rater']==r['rater']
            assert p['status']=='SUBMITTED' and set(p['scores'])==set(CRITERIA)
            for c in CRITERIA:
                value=p['scores'][c]
                if c==CRITERIA[-1] and item['visual_text'] is None:assert value=='N/A'
                else:assert isinstance(value,int) and not isinstance(value,bool) and 1<=value<=5
            if p['serious_grounding_issue']:
                flagged+=1
                assert p['issue_span'] and p['issue_span'] in text and p['evidence_notes'].strip()
                assert p['evidence_source_ids'] and set(p['evidence_source_ids'])<=recipient_ids
                assert p['issue_certainty'] in ['confirmed','uncertain']
            if r['reuse_previous_audit']:
                prior=r['prior_audit'];name=prior['prior_audit_folder'];assert saved==old_rows[name][r['alias']]
                assert prior['original_scores']==p['scores'] and prior['original_comment']==p['evidence_notes']
                reused+=1;reused_payloads.append({'rater':r['rater'],'position':r['position'],'alias':r['alias'],'full_sql_row_equal_to':name})
            pair=(r['rater'],r['position'])
            assessment,priority,note=SPECIAL.get(pair,('BROADLY_REASONABLE','NONE',
                'Điểm và nhận xét nhìn chung có thể biện hộ theo rubric trong đối chiếu này; không chứng nhận mọi câu đã đúng tuyệt đối.'))
            if pair in CALIBRATION:
                assessment,priority,note='CALIBRATION_DISCUSSION','OPTIONAL',CALIBRATION[pair]
            # Never create replacement human scores. Original payload remains exact.
            findings.append({'audit_id':f"{r['rater']}-{r['position']:03d}",
                'group_id':g['group_id'],'rater':r['rater'],'position':r['position'],'alias':r['alias'],
                'business_id':g['business_id'],'business_name':g['business_name'],'original_version':r['version'],
                'original_rating_payload':p,'original_payload_canonical_sha256':digest(p),
                'assessment':assessment,'followup_priority':priority,'assessment_note_vi':note,
                'output_and_source_context_vi':context,
                'review_output_sha256':item['review_text_sha256'],'visual_output_sha256':item['visual_text_sha256'],
                'context_review_evidence':[{'ordinal':i,'review_id':s['reviews'][i-1]['review_id'],
                    'full_text_sha256':hashlib.sha256(s['reviews'][i-1]['text'].encode()).hexdigest()} for i in REVIEW_ORDINALS[tag]],
                'photo_set_evidence':[] if g['visual_output'] is None else source_receipts[[x['business_id'] for x in source_receipts].index(g['business_id'])]['photo_hashes'],
                'prior_audit_reused':r['reuse_previous_audit'],'prior_audit_evidence':r['prior_audit'],
                'scores_changed':False,'replacement_scores':None})
    findings.sort(key=lambda x:(x['rater'],x['position']))
    assert len({x['alias'] for x in findings})==240 and reused==80 and flagged==81
    numeric=sum(isinstance(v,int) and not isinstance(v,bool) for f in findings for v in f['original_rating_payload']['scores'].values())
    na=sum(f['original_rating_payload']['scores'][CRITERIA[-1]]=='N/A' for f in findings)
    empty=sum(g['visual_output'] is not None and not g['visual_output'].replace('[VISUAL_EVIDENCE]','').strip() for g in groups)*2
    assert (numeric,na,empty)==(1380,60,4)
    assert sum(len(s['reviews']) for s in sources.values())==842 and sum(len(s['photos']) for s in sources.values())==343
    for rater in ['A01','A02']:
        assert {x['position'] for x in findings if x['rater']==rater}==set(range(1,121))
    # Independently replay every event/hash/version transition from the snapshot.
    c=sqlite3.connect((ROOT/'PRIVATE_SNAPSHOT.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    states={};versions={};chain=None;events=[]
    try:
        assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        for row in c.execute('SELECT * FROM events ORDER BY seq'):
            e=json.loads(row['event_json']);item=key[e['alias']]
            blank={'scores':{crit:'N/A' if crit==CRITERIA[-1] and item['visual_text'] is None else None for crit in CRITERIA},
                'evidence_notes':'','serious_grounding_issue':False,'issue_span':'','evidence_source_ids':[],
                'issue_certainty':'none','status':'NOT_STARTED','submitted_at_utc':''}
            before=states.get(e['alias'],blank)
            assert digest(e)==row['event_hash'] and e['previous_event_sha256']==chain
            assert e['version']==versions.get(e['alias'],0)+1 and e['rater']==item['annotator']
            assert before['status']!='SUBMITTED' and e['prior_state_sha256']==digest(before)
            states[e['alias']]=e['payload'];versions[e['alias']]=e['version'];chain=row['event_hash'];events.append(e)
        assert len(states)==240 and len(events)==241
        for row in snapshot:assert states[row['alias']]==json.loads(row['payload']) and versions[row['alias']]==row['version']
    finally:c.close()
    unchanged=sql_rows(STUDY/'PRIVATE_STATE/ratings.sqlite3','SELECT * FROM ratings ORDER BY rater,alias')==snapshot
    assert unchanged
    # Preserve exact excerpts and ORIGINAL photos for the small follow-up list.
    followups=[x for x in findings if x['followup_priority'] in ['HIGH','LOW'] or x['assessment']=='UNCERTAIN_VISUAL_COUNT']
    photos_needed={'D22P4':[1,2,3,4,5,6],'D20P1':[25],'D20P3':[3,6,18,19,29] ,'D26P4':[1,2,3,4,5,6]}
    evidence_dir=ROOT/'FOLLOWUP_PHOTOS';evidence_dir.mkdir(exist_ok=True)
    evidence=[]
    for f in followups:
        g=next(x for x in groups if x['group_id']==f['group_id']);s=sources[f['business_id']]
        e={'audit_id':f['audit_id'],'group_id':f['group_id'],'finding_vi':f['assessment_note_vi'],
            'exact_original_comment':f['original_rating_payload']['evidence_notes'],'exact_output_spans':[],
            'exact_review_excerpts':[],'original_photos':[]}
        if f['audit_id'] in ['A01-019','A02-055']:
            e['exact_output_spans']=['There are three tacos.']
            e['certainty']='UNCERTAIN: overlapping shells; neither participant count adjudicated as certain.'
            if f['audit_id']=='A01-019':assert 'ba tacos thực là hai' in e['exact_original_comment']
        elif f['audit_id']=='A02-003':
            e['exact_output_spans']=['speakeasy-like'];r=s['reviews'][18];quote='The decor inside is great, very old world, speakeasy kind of look.';assert quote in r['text']
            e['exact_review_excerpts']=[{'review_id':r['review_id'],'quote':quote}]
        elif f['audit_id']=='A02-001':e['exact_output_spans']=['The fries are topped with cheese.','The restaurant has red stools.','The photo shows a milkshake.']
        elif f['audit_id']=='A02-005':
            e['exact_output_spans']=['A man is sitting at a bar.','though some selected reviewers report slow or inattentive service.']
            r=s['reviews'][4];quote='When I got there, my order was not ready but they made a point to rush and get  it out to me.';assert quote in r['text']
            e['exact_review_excerpts']=[{'review_id':r['review_id'],'quote':quote}]
            e['inspection_limitation_recorded_by_participant']='8 of 34 photos inspected'
            assert e['inspection_limitation_recorded_by_participant'] in e['exact_original_comment']
        elif f['audit_id']=='A01-038':e['exact_output_spans']=['The dessert has a spoon.']
        for span in e['exact_output_spans']:assert span in g['review_output']+'\n'+(g['visual_output'] or '')
        for n in photos_needed.get(g['group_id'],[]):
            p=s['photos'][n-1];dest=evidence_dir/(p['photo_id']+'.jpg');shutil.copy2(STUDY/'PHOTOS'/dest.name,dest)
            assert file_hash(dest)==p['sha256']
            e['original_photos'].append({'ordinal':n,'photo_id':p['photo_id'],'file':dest.relative_to(ROOT).as_posix(),'sha256':p['sha256']})
        evidence.append(e)
    write_json(ROOT/'ITEM_LEVEL_AUDIT_240.json',findings)
    write_json(ROOT/'SOURCE_INTEGRITY_RECEIPTS.json',source_receipts)
    write_json(ROOT/'REUSE_VERIFICATION_80.json',reused_payloads)
    write_json(ROOT/'FOLLOWUP_EVIDENCE.json',evidence)
    calibration=[x for x in findings if x['assessment']=='CALIBRATION_DISCUSSION']
    counts={r:dict(Counter(x['assessment'] for x in findings if x['rater']==r)) for r in ['A01','A02']}
    summary={'format':'stage_n_full240_model_assisted_rating_quality_audit_v1',
        'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'read_only_on_study':True,
        'record_coverage':{'records':240,'per_rater':120,'businesses':30,'paired_output_groups':120,
            'previous_full_payloads_verified_unchanged_and_reused':80,'new_rating_records_read':160,
            'previously_inspected_output_groups':72,'new_output_groups_read':48},
        'classification_counts':counts,'confirmed_comment_or_source_score_findings':[x['audit_id'] for x in findings if x['assessment'].startswith('CONFIRMED')],
        'partial_source_followup':['A02-005'],'minor_comment_detail_caution':['A01-038'],
        'uncertain_visual_count_same_output':['A01-019','A02-055'],
        'audit_correction_disclosure':'An intermediate commentary prematurely called A01 taco count wrong. Retracted after fuller image inspection; previous audit certainty that three tacos were verified is also qualified. No human record was changed.',
        'optional_calibration':[x['audit_id'] for x in calibration],
        'technical_verification':{'expected240_unique_assignments':'PASS','all240_SUBMITTED':'PASS',
            'numeric_scores_1_to_5':numeric,'correct_structural_visual_NA':na,'applicable_empty_visual_ratings':empty,
            'all81_serious_flags_exact_output_span_and_recipient_source_ids':'PASS',
            'all30_source_json_and_review_blocks_bound':'PASS','all343_original_photo_bytes_and_sha256':'PASS',
            'all240_assignment_output_hashes':'PASS','sqlite_quick_check':'PASS','full_event_chain':'PASS',
            'event_count':241,'journal_head_sha256':chain,'all240_live_sql_rows_equal_snapshot':unchanged},
        'source_review_scope':{'selected_reviews_in_sources':842,'original_photos':343,
            'full_business_review_transcripts_read_across_prior_work':21,
            'other_businesses_with_targeted_excerpts_and_full_review_followups':9,
            'photos_visually_inspected_across_prior_work_as_labelled_contact_sheets':343,
            'this_pass_reopened_original_photo_ids':[p.stem for p in sorted(evidence_dir.glob('*.jpg'))],
            'this_pass_also_reopened_sheets':['B06_1 (DuBa)','C16_1/C16_2/C16_3 (Naked Fins)'],
            'not_claimed':'Exhaustive independent verification of every clause in all842 reviews/every output claim or every photo at full resolution.'},
        'old_A02_amendments':'NOT_OBSERVED: position1 version2 and position3 version1 retain old comments/scores; current rows match both prior audit snapshots.',
        'scores_or_comments_changed_by_agent':False,'reopened_or_unlocked':False,
        'agreement_statistics_or_model_winner_analysis_run':False,
        'limitations':['Model-assisted plausibility audit, not new independent human ratings or gold-standard adjudication.',
            'Model conditions omitted from work packets, but scores/rater identities visible; not score-blinded.',
            'Classification counts are audit triage counts, not accuracy rates or evidence one person is superior.',
            'Identity, actual source viewing, independent authorship and AI use cannot be authenticated from these records.',
            'Missing mandatory personal declarations/direct access waiver remain historical provenance limitations; no declarations invented.',
            'Optional calibration is not a finding a numerical grade must be changed. Human participant makes any genuine amendment.'],
        'input_hashes':{name:file_hash(ROOT/name) for name in ['SNAPSHOT_RECEIPT.json','GROUPED_ITEMS.json','SOURCES.json']},
        'frozen_workflow_hashes':{name:file_hash(STUDY/name) for name in ['CONFIG.json','COHORT30.json','RUBRIC.json','COORDINATOR_ONLY/PRIVATE_ITEMS.json']},
        'snapshot_sql_row_digest':digest(snapshot),'snapshot_event_digest':digest(events)}
    assert sum(sum(c.values()) for c in counts.values())==240
    write_json(ROOT/'AUDIT_SUMMARY.json',summary)
    lines=['# Kiểm tra toàn bộ240 bản ghi chấm Stage N','',
        'Đã xem đủ **A01:120/120 và A02:120/120**. Đây là30 business ×4 output ×2 người chấm, không phải240 business. Giữ nguyên toàn bộ điểm, nhận xét và khóa bản nộp.','',
        '**Kết luận:** phần lớn có căn cứ. Có2 bản ghi với lỗi nhận xét/đối chiếu nguồn rõ;1 bản ghi cần hoàn tất đối chiếu ảnh;1 lưu ý chi tiết nhỏ chưa chắc chắn. Hai bản ghi của cùng một output có bất đồng số taco chưa phân xử. Sáu bản ghi chỉ cần trao đổi rubric nếu cần, không bắt buộc đổi điểm. Chưa nên coi mọi vấn đề đã đóng trước khi người chấm phản hồi các mục ưu tiên.','',
        '| Người | Nhìn chung có thể biện hộ | Lỗi nguồn/nhận xét rõ | Cần xem thêm nguồn | Lưu ý nhỏ chưa chắc | Số taco chưa phân xử | Trao đổi rubric | Tổng |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        '| A01 |116|0|0|1|1|2|120|','| A02 |112|2|1|0|1|4|120|','',
        'Các số đếm trên là phân loại công việc cần xem lại, **không phải tỷ lệ chính xác hay xếp hạng người chấm**. Không có điểm AI thay thế.','',
        '## Ba mục ưu tiên','',
        '1. **A02 mục001 — Starbucks:** ghi chú xác nhận fries/red stools/milkshake nhưng sáu ảnh recipient không có các cảnh này. Visual4 cần xem lại. Bản lưu vẫn version2, chưa thấy sửa.',
        '2. **A02 mục003 — Gallagher’s:** speakeasy-like có nguồn: review `i1Iq_i2JoVSzlAeB04AJpA` viết “very old world, speakeasy kind of look”. Nhận xét phủ nhận nguồn cần sửa; không tự kết luận faithfulness4 phải thay. Bản lưu vẫn version1.',
        '3. **A02 mục005 — Naked Fins:** ghi chú tự giới hạn8/34 ảnh. Cần xem26 ảnh còn lại, nhất là mural và ba bao có ảnh P19/P29, trong khi man sitting at bar chưa được xác lập. Service inattentive cũng chưa được chứng minh từ pickup not ready rồi rush fix. Cần làm rõ căn cứ faithfulness5/Visual4.','',
        '## Lưu ý nhỏ và calibration','',
        '- **A01 mục038 — Naked Fins:** ảnh coconut P25 có straw, không thấy spoon rõ như ghi chú. Phần phát hiện menu/collage/whipped-cream scene sai vẫn có căn cứ; không dùng chi tiết này để đảo toàn bộ kết quả Visual1.',
        '- **A01 mục019 và A02 mục055 — PlantPure:** cùng output có câu three tacos. A01 đếm hai, A02 đếm ba; các vỏ chồng nhau nên giữ uncertainty. Nhận định vội trong commentary rằng A01 chắc chắn sai được rút lại; audit cũ nói ba taco verified cũng được hạ mức khẳng định. Không thay điểm hoặc đổ lỗi cho người chấm khi chưa đủ chắc.',
        '- **A01 mục029 — Gamers:** cân phần đúng trong ảnh chicken flatbread và drink không có;2/3 không phải đáp án khách quan duy nhất.',
        '- **A01 mục079 và A02 mục096 — Med:** vật thể đúng qua nhiều ảnh nhưng The plate nối grill với cake/nuts; cần thống nhất mức phạt relation fusion.',
        '- **A02 mục004 — Bricks;070 — IHOP;106 — Mungo:** thống nhất mức độ lặp ý so với chỉ lặp cấu trúc câu. Không tự ép đổi2 thành3.','',
        '## Kiểm tra kỹ thuật thực tế','',
        'PASS:240 assignment duy nhất/đúng người/đúng business/đủ SUBMITTED;1.380 điểm số hợp lệ;60 Visual N/A cấu trúc;4 Visual áp dụng nhưng rỗng vẫn có điểm1.81 serious flags đều có exact output span và recipient source IDs hợp lệ. Đây là kiểm tra schema/binding, không tự xác nhận mọi serious flag là lỗi ngữ nghĩa thật.',
        '', 'PASS:30 JSON nguồn và selected-review blocks, toàn bộ343 JPEG byte sizes/SHA256,240 assignment-output hashes, SQLite quick_check và241-event hash/version/state journal. Toàn bộ240 hàng database vẫn bằng snapshot sau audit. Hai mục A02 cũ không có saved amendment được quan sát.','',
        '## Cách kiểm tra và giới hạn','',
        'Đã đọc tất cả120 nhóm output cùng240 bộ điểm/nhận xét;80 bản ghi cũ dùng lại sau khi so toàn bộ hàng SQL/payload/version không đổi,160 bản ghi còn lại được đọc mới.48 output mới,72 output đã đối chiếu từ các lượt trước. Không dùng chênh điểm giữa hai người làm bằng chứng một người sai.',
        '', 'Nguồn gồm842 reviews và343 ảnh của30 business. Các lượt trước đã đọc đầy đủ transcript của21 business;9 business khác dùng trích đoạn có ngữ cảnh và đọc lại review đầy đủ cho claim cụ thể. Đã xem343 ảnh trên contact sheets có ID qua các lượt; lần này mở lại ảnh gốc cho các mục follow-up và xem lại sheets DuBa/Naked Fins. Không tuyên bố đã kiểm chứng độc lập từng mệnh đề của toàn bộ842 reviews hoặc mọi chi tiết ảnh ở độ phân giải gốc.',
        '', 'Đây là **model-assisted audit phủ đủ bản ghi**, không phải gold standard, xác thực danh tính hay bằng chứng mọi nguồn thực sự đã được người tham gia xem. Tên condition không nằm trong work packets, nhưng điểm và rater đã thấy nên không phải score-blind. Không suy AI use từ văn phong/tốc độ. Các giới hạn khai báo của đợt thu thập được giữ nguyên, không tạo khai báo thay người chấm.',
        '', 'Không sửa human data, không tự mở khóa, không tính agreement/model rankings và không chọn model thắng. Nếu có amendment sau này, phải giữ bản gốc và provenance thật của người sửa.','',
        '## Từng bản ghi','']
    for f in findings:
        lines += [f"### {f['audit_id']} — {f['business_name']} — {f['assessment']}",'',
            f"Nhóm `{f['group_id']}`; version gốc{f['original_version']}; alias `{f['alias']}`.",'',
            'Điểm gốc [Faithfulness, Coverage, Coherence, NonRedundancy, Usefulness, Visual]: '+json.dumps([f['original_rating_payload']['scores'][c] for c in CRITERIA],ensure_ascii=False),
            '',f['assessment_note_vi'],'',f['output_and_source_context_vi'],'',
            'Nhận xét gốc: '+f['original_rating_payload']['evidence_notes'],'',
            'Review IDs dùng đối chiếu ngữ cảnh: '+', '.join(r['review_id'] for r in f['context_review_evidence'])+'.','']
    (ROOT/'BAO_CAO_KIEM_TRA_240_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    follow_lines=['# Các mục để người chấm xem lại','',
        'Dùng cùng thứ tự mục trong trang local gốc. Đây là yêu cầu làm rõ/đính chính, không phải điểm mới do AI quyết định. Không sửa các mục khác chỉ để hai người chấm giống nhau.','']
    for f in followups+calibration:
        follow_lines += [f"## {f['audit_id']} — {f['business_name']} ({f['followup_priority']})",'',
            f['assessment_note_vi'],'','Nhận xét đang lưu: '+f['original_rating_payload']['evidence_notes'],'']
    (ROOT/'CAC_MUC_CAN_XEM_LAI_VI.md').write_text('\n'.join(follow_lines)+'\n',encoding='utf8')
    print(json.dumps({'classifications':counts,'rows_unchanged':unchanged,'journal_events':len(events),
        'read_new_rating_records':160,'prior_full_payloads_reused':reused,'photo_hashes':343,
        'report':str(ROOT/'BAO_CAO_KIEM_TRA_240_VI.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
