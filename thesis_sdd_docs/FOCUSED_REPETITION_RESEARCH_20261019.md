# Nghiên cứu tập trung sau D-060: lặp câu và lặp từ/cụm

Trạng thái: **nghiên cứu và audit offline hoàn tất; chưa có thử nghiệm mới được duyệt**.
Không chạy model, GPU, generation, training hoặc Teacher API. Không thay dữ liệu,
checkpoint, decoder chính thức, định nghĩa metric cũ, references hoặc nhãn người.
Đây không phải quyết định GO/NO-GO D-060 hay gia hạn thời gian tối ưu.

## Kết luận thực tế

Khuyến nghị một hướng duy nhất cho bước thử tiếp theo, nếu được researcher duyệt:
**một chính sách giải mã chống lặp có nhận biết section và đơn vị từ/cụm**, dùng
checkpoint hiện có. Thử khả thi trên đúng24validation F trước; không train thêm,
không thử nhiều checkpoint/hyperparameter và không tự áp dụng production/test.

Chính sách cụ thể phải qua kiểm tra tokenizer/beam/marker offline trước khi có
hash-bound GPU approval. Tài liệu này là khuyến nghị nghiên cứu, không phải runner
đã được kiểm định hay config giải mã đã đóng băng. Không thể hứa giải quyết triệt
để chất lượng tóm tắt bằng một tham số chống lặp.

## 1. Bằng chứng hiện có thay đổi cách hiểu vấn đề

144 records đã lưu được đọc lại;12nhóm model/subset khớp hoàn toàn metric đã công bố.
Random20 và targeted4 tách riêng. Các con số sau chỉ thuộc random20 Filtered:

| Mốc | Output lặp nguyên câu | Output có cặp từ lexical giống nhau liên tiếp |
|---|---:|---:|
| Historical F |15/20|19/20|
| Diagnostic B: beam4 + no-repeat4 |0/20|20/20|
| D-058 B epoch5 |8/20|14/20|
| D-059 Expanded epoch3 |3/20|12/20|
| D-059 Expanded epoch5 |8/20|13/20|
| D-060 RO-to-F epoch3 |17/20|16/20|

Diagnostic B đã loại lặp nguyên câu quan sát được trên20ca này. Đây không phải
bảo đảm cho mọi câu, tokenizer hoặc business; nó vẫn lặp từ/cụm và có lỗi grounding.
Lượt D-059/D-060 dùng decoder baseline đã đóng băng, không mặc nhiên dùng bộ chặn B.

Hàm words() hiện hành lowercase, bỏ dấu câu/markers rồi ghép chuỗi từ toàn output.
Vì vậy metric tính cả “attentive, attentive”, “service. Service…” và “Bang Bang”.
Nó không tính một số lỗi khác như “reasonable or reasonable”. Không được gọi16/20
là16output đều có lỗi từ dính liền kiểu “the the”. Metric này cũng không tự xác
định cặp lặp nào là lỗi ngôn ngữ.

Audit bổ sung giữ nguyên metric cũ: D-060epoch3 có16/20output với cặp lexical
trong cùng câu; chỉ1/20 có từ giống nhau phân cách bằng whitespace thuần.
Đó là “Bang Bang Shrimp”, business p7-1xG891NdGrcYBMOvxpw, được reviews hỗ trợ.
Tuy nhiên output lặp nguyên tên món ba lần trong cùng danh sách vẫn là dư thừa:

> including the Bang Bang Shrimp, Bang Bang Shrimp, and the Bang Bang Shrimp.

Cần giữ phần tên “Bang Bang”, đồng thời tránh liệt kê lại cả món. Bỏ hết từ lặp
hoặc cho phép tất cả cụm có trong source đều là quy tắc quá thô.

Ví dụ D-060 cùng business:

> slow, slow, or slow service

> contemporary, contemporary, contemporary, or roomy

Đây là các lỗi thực tế cần nhắm đến. Cặp chỉ nằm ở ranh giới câu là2output trong
historicalF nhưng không làm biến mất vấn đề16/20của D-060. Không hồi tố thay số cũ.

Diagnostic B còn có xung đột cấu trúc đã được chứng minh: sau một header sai
VISUAL_CONTEXT/VISUAL_CONSENSUS, no-repeat4 chặn tiền tố token dùng chung của
VISUAL_EVIDENCE; ba output xuất hiện VISUAL__EVIDENCE. Bộ chặn không được coi
markers là văn xuôi thông thường. Đây là xung đột sau một lỗi header có sẵn,
không chứng minh bộ chặn gây ra lỗi header ban đầu.

Ca zero-caption j3kGicU3mPG_ciIKG1Vuyg trong B vẫn sinh bagels/cream-cheese không
có caption. B cóEOS trước cap trong cả72outputs nhưng vẫn có lỗi nội dung. Vì
vậy zero lặp câu/EOS hợp lệ không tương đương summary đúng, đầy đủ và hữu ích.

## 2. Những gì tài liệu gốc hỗ trợ

**Unlikelihood.** Welleck et al., ICLR2020, bổ sung tín hiệu âm cho các continuation
không mong muốn ở mức token/sequence; sequence UL dùng continuation do model sinh.
Đây là phương pháp học trực tiếp chống lặp, không phải tăng LoRA rank hay thêm
epoch CE. Bối cảnh chính là language modeling, không phải LED opinion summarization.
[Bài gốc](https://arxiv.org/html/1908.04319v2).

**Summarization với UL.** Nair & Singh, RANLP-SRW2021, nghiên cứu UL trong Transformer
encoder–decoder trên CNN/DailyMail. Kết quả hỗ trợ khả năng giảm lặp, không cung cấp
bảo đảm factuality cho restaurant summarization của ta.
[Bài gốc](https://aclanthology.org/2021.ranlp-srw.18/).

**DITTO.** Xu et al., NeurIPS2022, phân tích sentence-loop self-reinforcement và tạo
negative views bằng lặp câu trong summary, giữ document. Thử summarization dùng
BART-large/CNN-DailyMail và đồng thời dùng beam5,tri-gram blocking. Kết quả không
phải bằng chứng rằng chỉ đổi loss sẽ loại hết lặp trên LED-LoRA. Không tự sao chép
lambda hay lịch học của bài báo.
[Bài gốc](https://arxiv.org/html/2206.02369v2).

**Decoder hiện hành.** Mã Transformers4.51.3 chặn n-gram trên token IDs và đặt score
của token bị chặn thành âm vô cùng. Nó không hiểu từ lexical, tên riêng hay câu.
Tài liệu cảnh báo bigram blocking có thể làm mất việc nhắc lại tên như New York.
Soft repetition_penalty vẫn có thể để token bị phạt thắng; không đảm bảo hết lặp.
[Mã chính thức đúng version](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/logits_process.py).

Coverage của pointer-generator cần thành phần attention/loss khác, không phải một
flag của LED. Contrastive search có thêm lựa chọn và chi phí hidden-state/candidate,
không có bảo đảm cho checkpoint này. Không ưu tiên hai hướng đó cho lần cuối.
[Coverage gốc](https://aclanthology.org/P17-1099/).

## 3. Đề xuất một lần thử, với điều kiện thực hiện rõ ràng

Mục tiêu là **can thiệp giải mã chung**, không tuyên bố model đã học tốt hơn.
Không chọn D-059 chỉ vì số lặp thấp nhất rồi trộn F đó với RO/U cũ trong so sánh chính.
Đề xuất dùng historicalF thuộc bộ RO/U/F đã matched cho pilot khả thi24cases;
baseline và DiagnosticB F đã có được dùng nguyên trạng làm đối chiếu.

Hợp đồng cần hoàn thiện và kiểm thử offline trước bất kỳ generation mới nào:

1. Phân biệt nội dung với section markers; giữ đúng marker tokenizer, không để
   lịch sử n-gram cấm một marker hợp lệ. Không tự rename header sai để cho PASS.
2. Xử lý lặp nội dung ở mức câu hoàn chỉnh và mục từ/cụm trong danh sách, gồm dấu
   phẩy/and/or. Chặn token IDs lặp đơn giản không đủ. BPE, chữ hoa/thường và từ nhiều
   token phải được xử lý đúng; không thêm một bộ NER/model mới mà chưa khai báo.
3. Giữ các trường hợp lặp hợp lệ trong tên/đơn vị ngữ pháp, ví dụ tên món đã có
   trong nguồn. Ngoại lệ không được hardcode businessID/đáp án của24cases hoặc
   dựa vào held-out references. Cơ chế tổng quát cần được đặc tả, chưa được chứng minh.
4. Không chỉ chặn dấu chấm cuối một câu đã lặp: model có thể thêm filler để né.
   Phải kiểm tra near-repeat/semantic repetition và báo thất bại nếu có hiện tượng đó.
5. Kiểm tra từng beam từ decoder prefix thực tế, beam reorder, EOS, cap và trường
   hợp mọi đường bị chặn. Khi không thể đáp ứng, báo lỗi/không hoàn tất; không tự
   nới bộ chặn, ép câu thay thế, xóa nội dung hoặc gọi model khác.
6. Giữ raw predictions, cấu hình và trace token bị can thiệp. Không xóa câu lặp
   hậu kỳ rồi báo đó là cải thiện học của model. Nếu có hậu xử lý, phải là một
   phương pháp được duyệt riêng với cả raw và processed metrics.

Các phép thử âm tối thiểu dùng token/text cố định, không model forward: header
chia sẻ tiền tố; Bang Bang/tên nhiều token; “slow, slow, or slow”; “fish, fish tacos”;
“service. Service…”; câu trùng được nối thêm filler; đổi beam; empty visual;
EOS/cap; toàn bộ scores bị chặn. Các phép thử này **chưa được triển khai/chạy**.
Đây là điểm chưa giải quyết, không được đánh dấu decoder-ready.

Nếu phần offline này đạt yêu cầu, đề xuất **một config duy nhất,24outputs F** trên
fixed20random+4targeted, beam4 và các tham số còn lại giữ nguyên, không sweep.
GPU execution vẫn cần duyệt config/runner/hashes. Không có ủy quyền mới ở đây.

Đánh giá cả giảm lặp lẫn Core/opinion/grounding/visual usefulness/completion.
Giữ tên hợp lệ; không coi Supporting omissions là lỗi tự động; không bịa visual
cho zero-caption. AI assessments phải provisional. Nếu chỉ đổi lặp thành filler,
hallucination hoặc mất Core, loại phương án dù metric lặp bằng0.

Nếu được researcher xác nhận có ích, một phương pháp giải mã chung phải được
duyệt và áp dụng đồng nhất cho RO/U/F/Shuffled trước held-out. Không thay riêng F.
Không tự chạy các điều kiện đó hay tuyên bố giữ nguyên decoder chính thức khi
thực tế đã thay_logits processing. Đây là thay đổi methodology cần khai báo.

## 4. Tại sao chưa chọn thêm training

UL/DITTO là hướng có lý do khoa học nếu mục tiêu là sửa phân phối model, nhưng
đòi loss, negative-candidate/negative-view policy, trọng số loss, schedule và kiểm
tra matchedRO/U/F mới. Không có giá trị nào đã được xác minh cho dataset này.
Đây không phải thay đổi một flag hay một lần CE continuation nữa. Trong cửa sổ
defense hiện tại, rủi ro mở thêm nhánh training cao hơn một can thiệp inference
có giới hạn. Không tự triển khai hai hướng song song hoặc làm parameter search.

Không có bằng chứng đủ để hứa “triệt để” cả lặp bề mặt, semantic repetition,
grounding và thông tin riêng business. Một bộ lọc có thể đảm bảo một quy tắc
chuỗi hẹp bằng construction; không thể từ đó suy ra model đã tóm tắt tốt hơn.

## 5. Giới hạn thời gian và bằng chứng

Giữ nguyên deadline D-0602026-10-22T07:11:22.846767+00:00 (22/10,14:11:22 ICT),
và deadline D-05925/10không thay. Nghiên cứu này không reset đồng hồ và không mở
thêm thời gian training. Nếu không có phương án đã kiểm định và chất lượng được
xác nhận trong giới hạn, dừng tối ưu và hoàn thành RO/U/F/Shuffled/human evaluation.

Audit: outputs/stage_i/repetition_research_20261019_v1/ gồm144record flags,12nhóm
thống kê, hash nguồn và ghi nhận zero model/GPU/API calls. Định nghĩa metric cũ
vẫn ở scripts/stage_i_decoding_diagnostics.py, không sửa. Các artifactD053–D060
và D051strict replayFAIL được giữ nguyên. Đánh giá tên món ở trên là nhận xét
AI dựa trên review đã có, không tạo thêm human-confirmed label.
