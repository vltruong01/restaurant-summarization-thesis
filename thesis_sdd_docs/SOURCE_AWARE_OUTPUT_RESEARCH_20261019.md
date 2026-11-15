# Nghiên cứu và áp dụng kiểm soát output dựa trên nguồn

Ngày19/10/2026. Đã thực hiện một thử nghiệm CPU trên24output guard đã lưu.
**Chưa đạt cải thiện chất lượng sử dụng; không khuyến nghị thay pipeline chính.**
Đây là nhận xét AI dựa trên bằng chứng, không phải quyết định GO/NO-GO của researcher.

## Kỹ thuật đã nghiên cứu và thực sự áp dụng

1. **Đối chiếu NLI theo đơn vị câu.** SummaC cho thấy granularity của source và
   hypothesis rất quan trọng. Thử nghiệm dùng cửa sổ câu review với vị trí nguồn
   chính xác; visual chỉ được so với một caption riêng. Đây là phương pháp lấy
   ý tưởng từ sentence-level NLI, không phải tái hiện SummaC-Conv đã huấn luyện.
   [Laban et al.,2022](https://aclanthology.org/2022.tacl-1.10/).
2. **Khử dư thừa ngữ nghĩa và MMR.** MiniLM tạo embedding để truy hồi và đo tương
   đồng; hai câu chỉ được đánh dấu trùng nghĩa khi thêm điều kiện entailment
   hai chiều. MMR cân bằng độ liên quan nguồn và độ trùng với câu đã chọn.
   Các ngưỡng là preset thăm dò, không phải thông số tối ưu lấy từ bài báo.
   [Carbonell & Goldstein,1998](https://doi.org/10.1145/290941.291025).
3. **Ràng buộc cấu trúc và zero-caption.** Chuẩn hóa wrapper một cách công khai;
   không cho giữ visual khi đầu vào không có caption. Không sửa từ trong câu,
   không điền thông tin từ Teacher hoặc D053 references, không giả định đây là
   năng lực mới mà LED học được.

Falke et al. đã chỉ ra rằng NLI ngoài miền không mặc nhiên đủ tốt để xếp hạng
summary đúng/sai. Vì vậy score được lưu như dự đoán của bộ kiểm tra, không phải
nhãn verified hay nhãn người. [Falke et al.,2019](https://aclanthology.org/P19-1213/).

Hai scorer phụ được đóng băng:all-MiniLM-L6-v2 revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 và nli-deberta-v3-small revision
e9890682d9e4279b7ae6d0fcfb435a43206280ec. Mean-pooling có attention mask theo
[model card gốc](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
NLI weights được tải công khai, không dùng API key; LFS SHA256/Git blob IDs
được đối chiếu trước khi dùng. Không thay backbone hoặc adapter của LED.

## Thiết kế và giới hạn

Một config duy nhất,24business validation F theo thứ tự cũ:20random+4targeted.
Các ca này đã được xem nhiều lần, nên là tập development cho thử nghiệm mới;
không coi đây là đánh giá độc lập hay kết quả held-out. Không đổi ngưỡng sau khi
xem score. Tất cả câu giữ lại là substring nguyên vẹn của raw prediction.
Mỗi quyết định giữ/bỏ có premise, markerREV/IMG_CAP, PHOTO ID khi có và logits.
Teacher targets, verdicts và approved references bị loại khỏi input của selector.

Giới hạn preset:review entailment>=.8,visual>=.9; margin tương ứng.6/.8;
cosine>=.85 và mutual-entailment>=.9 cho dedup;MMRlambda=.65. Không tuyên bố
những ngưỡng này đã được hiệu chuẩn. Không cắt ngầm token:window/cặp quá dài
bị ghi nhận hoặc loại trước forward; giữ nguyên toàn bộ nguồn để kiểm tra.

## Kết quả thực tế

- Hoàn tất24/24 trong447.76giây;1021cặp NLI,527batchforward của scorerCPU.
- Không sinh thêm bằng LED, không training, không GPU, không Teacher API/test.
- Giữ66câu raw,68binding/span nguồn chính xác. Không viết lại nội dung câu.
- **17/24 phần review rỗng;23/24 giữ ít hơn3câu review.** Trung bình chỉ.42câu
  review và2.33câu visual. Không thể gọi đây là summary đầy đủ, hữu ích.
- Cả24có wrapper chuẩn do serializer. Metric cấu trúc cũ chỉ đạt6/24 vì còn
  yêu cầu nội dung không rỗng. Không được báo24/24 là decoder đã học tốt hơn.
- Zero-caption đúng1/1 do quy tắc hậu xử lý. Đây là một invariant hữu ích,
  không phải bằng chứng rằng checkpoint đã học cách xử lý zero-caption.
- NLI vẫn chấp nhận câu **There is a donut sitting next to a donut.** với score
  entailment.9886207. Nguồn IMG_CAP_1 của gcQSAQIx-KQOdf5E8HF6Lw ghi
  *two cups of coffee and a donut sitting next to each other.* PHOTO
  9YRB33hF9nyyv4C8xckbJg đã được AI xem trực tiếp:hai cốc và một donut.
  Đây là ví dụ lỗi đếm/quan hệ lọt qua model;score cao không phải xác suất đúng.

Không suy ra17review rỗng nghĩa là mọi câu raw của17ca đều sai. Bộ kiểm tra có
thể loại quá mức những câu tổng hợp nhiều ý kiến; truy hồi/miền dữ liệu/ngữ cảnh
có thể góp phần. Đây là các giải thích còn cần kiểm chứng, không phải nguyên
nhân đã được chứng minh. Không hạ ngưỡng hoặc ghép lại câu bằng tay để che lỗi.

## Quyết định kỹ thuật được đề nghị

Không đưa hard-NLI-filter này vào bản chính. Nó giảm một số biểu hiện không
được nguồn hỗ trợ nhưng đánh đổi bằng mất thông tin nặng và vẫn có false positive.
Không biến việc xóa câu thành bằng chứng cải thiện mô hình, factual accuracy
hoặc Core coverage. Các references giữ nguyên; chưa có điểm Core xác nhận mới.

Nếu researcher muốn mở một can thiệp ở cấp học, hai tài liệu đáng xem là
[Unlikelihood, Welleck et al.,2020](https://arxiv.org/abs/1908.04319) và
[DITTO, Xu et al.,2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/148c0aeea1c5da82f4fa86a09d4190da-Paper-Conference.pdf).
Chúng tác động vào objective học chống lặp, khác với hậu xử lý; không chứng minh
sẽ giải quyết grounding/Core trong dataset này. Chưa triển khai/chạy objective
mới hoặc chọn lossweight/schedule. Không dùng24validation failures làm training
negatives. Nếu được duyệt sau, negative views phải lấy từ train riêng và giữ
positive Teacher labels nguyên vẹn; phương pháp RO/U/F phải được kiểm soát công bằng.

Không tự mở thêm training, inference variant hay tối ưu threshold. Giữ deadline
D06022/10 và D05925/10;D051strict replayFAIL và mọi artifact khoa học cũ không đổi.

## Artifact và kiểm chứng

Input/config/protocol:outputs/stage_i/source_consistency_preparation_v1.
Raw scores/derived outputs:outputs/stage_i/source_consistency_postprocess_v1.
Report/HTML:outputs/stage_i/source_consistency_report_v1_1.
ZIP:outputs/stage_i/stage_i_source_consistency_review_v1_1.zip.
SHA256:44ef73923eb4820407404469882d13ede01cbd83988f7e3b29a6987c67d207c4.
153payload hashes,24casecards/48panels/90ảnh và12test mới sau giải nén PASS;
91test hồi quy guard cũ cũng PASS. Software/integrityPASS không phải qualityGO.
