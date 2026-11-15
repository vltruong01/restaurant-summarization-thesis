"""Create separate synthetic rubric practice, never read or score TEST outputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
READY=ROOT/'outputs/stage_n/test30_two_rater_preparation_v1'
OUT=ROOT/'outputs/stage_n/practice_and_guidance_v1'

GUIDE='''# Hướng dẫn dùng Stage N — chấm chất lượng output

## 1. Công việc cần làm

Stage N là đọc nguồn và chấm chất lượng bản tóm tắt của mô hình. Mỗi người chấm
120 mục: 30 business, mỗi business có 4 output. Hai người A01/A02 có tổng cộng
240 bản chấm. Bạn chấm điểm theo 6 tiêu chí và có thể ghi nhận xét; không cần
viết lại Review/Visual, tạo reference mới, hay chọn APPROVE/REQUEST_REVISION
như công việc checker ở Stage L.

Hiện chỉ hoàn tất chuẩn bị. Chấm thật vẫn khóa; chưa có tài khoản/mật khẩu thật
hoặc phiên chấm nào được mở bởi bước chuẩn bị này. Cần supervisor duyệt sửa đổi
hai người chấm và researcher cho phép bắt đầu trước khi mở chấm thật.

## 2. Người điều phối cần làm sau khi đã được duyệt

Thư mục dùng bản thật:
`outputs/stage_n/test30_two_rater_preparation_v1/`.

1. Giữ ZIP đầy đủ và thư mục COORDINATOR_ONLY cho người điều phối. Trong đó có
   khóa tên mô hình và bằng chứng riêng; không gửi toàn bộ cho người chấm.
2. Tạo ba tài khoản A01, A02 và COORDINATOR bằng các lệnh getpass trong README.
   Mỗi người có mật khẩu riêng ít nhất 12 ký tự; không dùng tài khoản author01/
   checker01 của Stage L. Chạy lại tạo tài khoản không đổi mật khẩu có sẵn.
3. Chỉ khi đã có phê duyệt thật, dùng LIVE_RELEASE_TEMPLATE.json để lập bản
   WORKSPACE/LIVE_RELEASE.json với hai phê duyệt, thời gian thực tế và đúng hashes.
   Không đánh dấu đã duyệt thay supervisor. Bản template hiện để false.
4. Chạy LAUNCH_LOCAL_REVIEW.cmd, rồi mở http://127.0.0.1:8766.
   Đây là địa chỉ trên chính máy đang chạy ứng dụng; người ở máy khác không thể
   mở localhost của bạn. Không tự đưa ứng dụng lên internet. Có thể dùng lần
   lượt hai browser profile riêng trên cùng máy, đăng xuất khi đổi người.
5. Mỗi người đăng nhập, đọc rubric và khai báo consent, đã hiểu công việc,
   từng xem output/kết quả tự động hay chưa, đã làm Stage L hay chưa, xung đột
   liên quan. Khai đúng thực tế; ưu tiên người chưa xem kết quả và không làm Stage L.
6. Người điều phối đăng nhập COORDINATOR, xem khai báo, ghi ELIGIBLE/INELIGIBLE
   và lý do. Chỉ người đủ điều kiện mới xem và chấm output được.

Tài khoản, mật khẩu và khai báo phải do người thật thực hiện sau bước duyệt;
file hướng dẫn này không cấp quyền mở chấm thật.

## 3. Một lượt chấm của A01/A02

1. Đăng nhập tài khoản của mình. Danh sách Your items là thứ tự riêng của bạn.
2. Chọn mục rồi bấm Open selected item. Bạn sẽ thấy nguồn bên trái và output
   ẩn danh bên phải. Không đoán tên mô hình hoặc trao đổi điểm với người còn lại.
3. Đọc selected reviews, chú ý ý chính, ý kiến trái chiều, sao và thời gian.
   Xem từng ảnh gốc; có thể mở ảnh ra tab riêng. Review/photo IDs giúp ghi bằng chứng.
4. Đọc phần Review Consensus và Visual Evidence của output đang được hiển thị.
5. Chọn điểm 1–5 cho từng tiêu chí áp dụng. 1 là kém, 3 là trung bình theo rubric,
   5 là tốt; 2 và 4 nằm giữa. Không mặc định mọi mục đều 3 hoặc đều 5.
6. Ghi nhận xét nếu cần. Nếu báo lỗi grounding nghiêm trọng, chọn ô tương ứng,
   chép đúng đoạn output có vấn đề, nhập IDs nguồn hỗ trợ việc kiểm tra, giải
   thích ngắn và chọn Uncertain hoặc Confirmed. Không cần lập bảng bằng chứng
   dài cho mọi câu; các ô này bắt buộc khi báo lỗi grounding nghiêm trọng.
7. Save draft để lưu nháp nếu chưa xong. Có thể đóng/mở lại và tiếp tục.
8. Khi đã thực sự chấm xong mọi tiêu chí áp dụng, bấm Submit and lock this item,
   xác nhận nộp. Mục sẽ khóa và tiến độ tăng; sau đó chọn mục kế tiếp.
   Không có nút tự nộp tất cả. Nếu đã nộp nhầm, báo người điều phối; không sửa DB.

## 4. Sáu tiêu chí — nhớ ngắn gọn

| Tiêu chí | Chấm phần nào | Câu hỏi thực hành |
| --- | --- | --- |
| Review Faithfulness | Chỉ Review | Có nói đúng reviews, đúng người/ý kiến/phủ định không? |
| Aspect Coverage | Chỉ Review | Có giữ ý chính, tích cực/tiêu cực/trái chiều đáng chú ý không? |
| Coherence | Các phần có trong output | Có dễ hiểu, mạch lạc, nhất quán không? |
| Non-Redundancy | Các phần có trong output | Có tránh lặp câu/lặp ý không? Điểm cao là ít lặp. |
| Usefulness | Chỉ Review | Có thông tin cụ thể, hữu ích, đại diện trải nghiệm khách không? |
| Visual Grounding and Contribution | Visual so với ảnh gốc recipient | Mô tả có căn cứ ở ảnh, đúng cảnh/quan hệ và hữu ích không? |

Đọc các mốc 1/3/5 đầy đủ trong giao diện trước khi chọn điểm. Bảng này chỉ là
cách nhớ, không thay rubric gốc.

Nếu Visual được xác định là không áp dụng, để N/A mà ứng dụng cung cấp. Nếu
Visual áp dụng nhưng output rỗng, vẫn cần tự chấm chất lượng theo rubric; đó
không phải ô đánh giá bị thiếu. Không tự sửa output hoặc bỏ business khó.
Chỉ dùng ảnh của business đang chấm, không mượn chi tiết từ ảnh/business khác.

## 5. Sau khi cả hai hoàn tất

Người điều phối kiểm tra đủ 120 SUBMITTED/người rồi xuất final complete export.
Ứng dụng yêu cầu đủ 240 mục và audit hợp lệ. Chỉ sau đó mới chạy phân tích thống
kê trên điểm thật. Điểm mẫu do GPT tạo không nhập vào ứng dụng và không tính vào
240 bản chấm; tài khoản mẫu hoặc khai báo giả không được dùng thay người thật.

## 6. Gửi gì cho GPT để học cách chấm?

Gửi **STAGE_N_GPT_PRACTICE_ONLY.zip** và nhờ GPT làm theo
GPT_PRACTICE_PROMPT_EN.md. Nếu GPT không đọc được ZIP, đính kèm riêng prompt,
PRACTICE_CASE.json, RUBRIC.json và hai ảnh trong PRACTICE_IMAGES/.

Bộ này có một business hoàn toàn giả và hai hình minh họa kỹ thuật, không phải
ảnh nhà hàng thật. Có bốn output mẫu để tập phân biệt đúng/sai, thiếu ý, Visual
không áp dụng và Visual rỗng nhưng áp dụng. Đây không phải output của mô hình
thật, bài chấm thật hay đáp án chuẩn. GPT giải thích cách suy nghĩ để người chấm
học phương pháp; không học một thứ tự thắng/thua hoặc sao chép điểm mẫu.

Không dùng ZIP chuẩn bị thật có private mapping để nhờ GPT chấm mẫu TEST30 cho
hai người chấm. Nếu người chấm đã xem output TEST hoặc bảng kết quả thật, phải
khai báo việc đó; luyện tập bằng bộ giả không xóa lịch sử tiếp xúc trước đó.
'''

PROMPT='''# Stage N — Synthetic scoring demonstration for learning only

Please teach me how to use the attached six-criterion rubric by working through
the SYNTHETIC_PRACTICE_CAFE_001 case. This is an isolated teaching exercise, not
genuine TEST30 data, an independent human evaluation, or a gold-standard answer.
Do not claim to act as A01/A02 or create any genuine participant submission.

Read PRACTICE_CASE.json, RUBRIC.json and BOTH individually attached PNG images
in PRACTICE_IMAGES/. The PNGs are simple technical illustrations, NOT original
restaurant photographs. Use only the supplied fictional reviews and visible
illustrations; do not infer real-world identity, food quality, ingredients or
scene continuity. If you cannot actually open either image, say so and leave
image-grounding judgments unscored rather than inventing an inspection.

The four practice outputs have NO mapping to real experimental model identities.
Do not guess RO/U/F/S, predict a winner, or import knowledge of previous results.
Do not rewrite the outputs or create improved summaries. Demonstrate judging
the exact strings provided, including the empty applicable Visual in X04.

For each output, in the supplied practice order:

1. Explain in a few sentences which important source facts/opinions are covered,
   missed or distorted. Distinguish repeated opinions from a single experience.
2. Give a suggested illustrative score for each of the six rubric criteria, with
   a brief reason tied to the original 1/3/5 anchors. Scores 2/4 are intermediate.
   Review Faithfulness, Aspect Coverage and Usefulness judge REVIEW only.
   Coherence and Non-Redundancy judge all available output components.
   Visual Grounding and Contribution judges only the visible recipient evidence.
   Use the explicit visual_applicable field: X03 is N/A, not a low score;
   X04 is applicable but empty and still needs a reasoned illustrative score.
3. For any serious factual/grounding defect, show the exact output span, the
   relevant SYNTHETIC_REVIEW or SYNTHETIC_IMAGE IDs, and a short explanation.
   Keep confirmed contradictions/unsupported assertions separate from uncertainty.
   Do not label every omission a factual error. Do not turn absence of one view
   into proof that an object could never exist elsewhere in a real restaurant.
4. Finish each example with one practical lesson a novice rater can apply.

At the end, show ONE sample of the optional note fields for an output with a
serious grounding defect: exact span, recipient source IDs, short explanation,
and Uncertain/Confirmed. Label it SYNTHETIC TEACHING EXAMPLE. Do not produce an
application-importable submission, approval, final export, or real-rater JSON.

Keep explanations understandable and reasonably concise. Suggested points are
for teaching, not an objective answer key or reliability evidence. Emphasize that
real participants must inspect genuine sources and make their own judgments;
they must not copy AI scores into the actual study. Respond in Vietnamese,
retaining criterion names, source IDs and quoted English output spans exactly.
'''

def main():
    before=hashlib.sha256((READY/'STAGE_N_HUMAN_EVALUATION_TEST30_READY.zip').read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True);images=OUT/'PRACTICE_IMAGES';images.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for i in [1,2]:
        im=Image.new('RGB',(800,520),'#f7fafb');d=ImageDraw.Draw(im)
        d.rectangle((0,0,800,56),fill='#173f48')
        d.text((20,18),'SYNTHETIC PRACTICE ILLUSTRATION — NOT A REAL PHOTO',fill='white',font=font)
        d.line((40,430,760,430),fill='#75858b',width=3)
        if i==1:
            d.rectangle((145,270,590,315),fill='#b58459',outline='#67472d',width=3)
            d.rectangle((175,315,195,430),fill='#75543b');d.rectangle((545,315,565,430),fill='#75543b')
            d.rounded_rectangle((225,200,290,270),radius=8,fill='#c44545',outline='#762f2f',width=3)
            d.arc((280,211,321,254),-90,90,fill='#a23232',width=8)
            d.rectangle((625,205,704,320),fill='#537fc7',outline='#254f8e',width=3)
            d.rectangle((613,320,715,345),fill='#537fc7',outline='#254f8e',width=3)
            d.line((625,345,625,430),fill='#254f8e',width=8);d.line((704,345,704,430),fill='#254f8e',width=8)
        else:
            d.rectangle((140,235,655,280),fill='#b5babe',outline='#5e666e',width=3)
            d.rectangle((165,280,628,430),fill='#d9dce0',outline='#5e666e',width=3)
            d.polygon([(509,194),(568,194),(557,234),(520,234)],fill='#b66c43',outline='#673f28')
            d.line((539,194,539,116),fill='#317340',width=7)
            d.ellipse((490,127,537,153),fill='#4d9760');d.ellipse((542,113,586,139),fill='#4d9760')
            d.ellipse((502,159,540,180),fill='#4d9760');d.ellipse((540,149,577,174),fill='#4d9760')
        d.text((25,468),f'Source ID: SYNTHETIC_IMAGE_0{i}',font=font,fill='#173f48')
        d.text((25,495),'Separate view. Do not assume the two illustrations depict one continuous scene.',font=small,fill='#465b63')
        im.save(images/f'SYNTHETIC_IMAGE_0{i}.png')
    good='[REVIEW_CONSENSUS]\nSeveral fictional reviewers praise the soup or bread, although one reports lukewarm soup. Service experiences differ: two describe friendly treatment, while others mention a wait or inattentive service. One visitor considers the price high. These accounts suggest mixed experiences rather than uniformly positive service.'
    case={'format':'stage_n_practice_only_v1','synthetic':True,'not_for_real_rating_import':True,'business_id':'SYNTHETIC_PRACTICE_CAFE_001',
        'business_name':'Fictional Practice Cafe','source_type':'Invented reviews and technical illustrations, not Yelp or TEST evidence',
        'reviews':[
            {'review_id':'SYNTHETIC_REVIEW_01','stars':5,'date':'2000-01-01','text':'The tomato soup was comforting and the bread was crisp. Our server was friendly. We waited about twenty minutes before being seated on Saturday.'},
            {'review_id':'SYNTHETIC_REVIEW_02','stars':4,'date':'2000-01-02','text':'I enjoyed the soup and fresh bread. The person at the counter was friendly, and my takeaway order was ready quickly.'},
            {'review_id':'SYNTHETIC_REVIEW_03','stars':2,'date':'2000-01-03','text':'My soup arrived lukewarm. It was hard to get the server’s attention, and the meal seemed expensive for what I received.'},
            {'review_id':'SYNTHETIC_REVIEW_04','stars':3,'date':'2000-01-04','text':'The bread was good. Our group waited roughly twenty-five minutes for the food. The room was noisy during our visit.'}],
        'images':[{'image_id':f'SYNTHETIC_IMAGE_0{i}','file':f'PRACTICE_IMAGES/SYNTHETIC_IMAGE_0{i}.png','is_original_restaurant_photo':False} for i in [1,2]],
        'practice_outputs':[
            {'practice_id':'PRACTICE_X01','review_text':good,'visual_applicable':True,'visual_text':'[VISUAL_EVIDENCE]\nOne illustration shows a red mug on a brown table and a blue chair. A separate illustration shows a counter with a potted plant.'},
            {'practice_id':'PRACTICE_X02','review_text':'[REVIEW_CONSENSUS]\nAll reviewers agree that service is always fast and attentive. Every visitor receives hot soup, and parking is free. The cafe is the cheapest option in town. Service is always fast and attentive.',
             'visual_applicable':True,'visual_text':'[VISUAL_EVIDENCE]\nA blue chair stands beside a potted plant on the same outdoor terrace, overlooking the ocean. A bowl of fresh salad is served on the table.'},
            {'practice_id':'PRACTICE_X03','review_text':'[REVIEW_CONSENSUS]\nSome visitors enjoy the food, and opinions about service differ.','visual_applicable':False,'visual_text':None},
            {'practice_id':'PRACTICE_X04','review_text':good,'visual_applicable':True,'visual_text':'[VISUAL_EVIDENCE]\n'}]}
    (OUT/'PRACTICE_CASE.json').write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    shutil.copyfile(READY/'WORKSPACE/RUBRIC.json',OUT/'RUBRIC.json')
    (OUT/'HUONG_DAN_STAGE_N_VI.md').write_text(GUIDE,encoding='utf8')
    (OUT/'GPT_PRACTICE_PROMPT_EN.md').write_text(PROMPT,encoding='utf8')
    (OUT/'README.md').write_text('''# Stage N synthetic practice only

Send this ZIP to GPT with: “Please follow GPT_PRACTICE_PROMPT_EN.md and demonstrate
how to score the attached fictional case. This is synthetic practice only.”

If ZIP reading fails, attach the prompt, PRACTICE_CASE.json, RUBRIC.json and both
PNG files separately. Vietnamese practical instructions are in
HUONG_DAN_STAGE_N_VI.md. No real TEST source, output, condition key, human score,
account, password, approval or release is present. The PNGs are simple technical
illustrations, not photographs. Do not import illustrative GPT scores into the
real application or report them as independent human evaluation. This exercise
teaches rubric use; it does not establish calibration, blindness or reliability.
''',encoding='utf8')
    data=json.loads((OUT/'PRACTICE_CASE.json').read_text(encoding='utf8'))
    assert data['synthetic'] and data['business_id'].startswith('SYNTHETIC_') and len(data['practice_outputs'])==4
    assert all(r['review_id'].startswith('SYNTHETIC_') for r in data['reviews'])
    assert data['practice_outputs'][2]['visual_text'] is None and data['practice_outputs'][3]['visual_applicable'] and not data['practice_outputs'][3]['visual_text'].removeprefix('[VISUAL_EVIDENCE]\n')
    for i in data['images']:
        with Image.open(OUT/i['file']) as im:im.verify()
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.zip','.sha256'] and p.name not in ['MANIFEST.json','VERIFICATION_RECEIPT.json'])
    manifest={'synthetic_practice_only':True,'files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    zpath=OUT/'STAGE_N_GPT_PRACTICE_ONLY.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,OUT/'MANIFEST.json']:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for r in manifest['files']:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    assert before==hashlib.sha256((READY/'STAGE_N_HUMAN_EVALUATION_TEST30_READY.zip').read_bytes()).hexdigest()
    checksum=hashlib.sha256(zpath.read_bytes()).hexdigest()
    (OUT/(zpath.name+'.sha256')).write_text(checksum+'  '+zpath.name+'\n',encoding='utf8')
    receipt={'status':'PASS','zip_bytes':zpath.stat().st_size,'zip_sha256':checksum,'synthetic_cases':1,'practice_outputs':4,'illustrations':2,'real_TEST_data_included':False,'real_scores_or_approvals_created':False,'original_ready_ZIP_unchanged_sha256':before,'all_members_hashed_and_CRC_checked':True}
    (OUT/'VERIFICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
