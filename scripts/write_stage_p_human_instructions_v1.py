"""Write evaluator/coordinator instructions; no credentials or ratings included."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/stage_p/three_system_human_evaluation_ready_20261106_v1';P=D/'PACKAGE';APP=P/'EVALUATOR_APP'
def main():
    guide='''# Independent restaurant-output review — Windows guide

This is a NEW session:15 restaurants,three anonymous outputs per restaurant,45 items per evaluator. Do not transfer or consult previous session scores. No rewriting of summaries is required.

## Start and sign in

1. Extract the entire evaluator ZIP onto a writable local Windows10/11 x64 folder. Do not run inside the ZIP. Python is bundled; no installation,GPU,internet or administrator privileges are needed.
2. Before distribution,the coordinator reviews the package using START_PREVIEW.cmd. This is read-only:forms cannot save or submit. Do not use it to collect scores.
3. After review,the coordinator deliberately starts START_COLLECTION.cmd. The app opens http://127.0.0.1:8768/. Keep its console open; closing it stops the server. This does not start another research/model experiment.
4. On first launch,the console automatically displays accounts A01,A02 and COORDINATOR with distinct memorable passwords. Record them privately then give each evaluator only their own credentials. Passwords are stored using salted PBKDF2 hashes,not plaintext files. Later starts preserve accounts,passwords and work; passwords are not printed again. There is no manual registration or participant-approval gate.
5. Use only your assigned account. A01 and A02 each have their own45 items,random order and saved work. Do not inspect or exchange the other evaluator's judgments. If working on separate computers,keep each extracted folder and return only your own JSON export. The study ID/frozen assignment mapping remains the same. localhost links open only on the computer running the app.

## Every item

1. Select an item; it opens automatically. Check the item heading and restaurant ID. Open selected item reopens it if needed.
2. Read the selected source reviews,including positive,negative and mixed opinions. Judge only the current restaurant's sources.
3. Browse the original photos. The first24 are just the first page. Use Next/Previous,page number,photos-per-page or an exact photo ID. Click a thumbnail to enlarge; use arrow keys or Previous/Next in the viewer. Do not infer taste,price,service quality or reviewer satisfaction from pixels.
4. Read the anonymous Review Consensus and Visual Evidence separately. Do not use reviews to establish details that are not visible in the Visual source images.
5. Score each of the six criteria1–5;higher is better. Use the unchanged anchors printed under each control.2 and4 are intermediate;criteria need not receive the same score.
6. Save draft to pause. Submit and lock this item only after finishing;wait for the success message. Submitted items cannot be revised without contacting the coordinator. Save before closing the page or console.

## What each criterion judges

| Criterion | Scope |
| --- | --- |
| Review Faithfulness | Review only:source support,correct polarity and attribution |
| Aspect Coverage | Review only:important recurring positives,negatives,mixed views and specific details |
| Coherence | Both components:readability and logical organization |
| Redundancy avoidance / Non-Redundancy | Both components:unnecessary repeated ideas or sentences |
| Usefulness | Review only:specific,representative,useful source-grounded customer experience |
| Visual Grounding and Contribution | Visual against original photos:observable support and useful scene coverage |

All outputs in this session have both components,so all six scores are required;do not enter N/A. If an applicable component looks empty,judge it under the existing rubric. Identical text may recur in another item:assess each item,without penalizing similarity between different items. This is different from repetition within one output.

General comments may be blank. For a serious grounding defect,record the exact problematic output span,relevant current-restaurant review/photo IDs,a short explanation and whether the finding is confirmed or uncertain. A suspicion is not automatically a confirmed factual error.

## Pause,resume and return results

Use the SAME extracted folder. Scores and journal persist in STATE/ratings.sqlite3;do not delete STATE or replace WORKSPACE. After saving,you may close the console and start again later with START_COLLECTION.cmd. This preserves the session and all accounts;no old scores are imported.

Download my progress produces your own partial backup. At Submitted45/45,click Download my completed results. A01 returns NEW_SESSION_A01_RESULTS_FINAL.json;A02 returns NEW_SESSION_A02_RESULTS_FINAL.json. Send only that JSON to the coordinator. Do not edit the export manually or send the account database. If incomplete,the app rejects a final export instead of filling missing records.

Keep a copy of the entire extracted application folder while it is stopped,especially STATE. JSON progress downloads provide an additional content backup but are not a mechanism to reset or silently overwrite submissions. Report problems to the coordinator.
'''
    (APP/'READ_ME_FIRST_EN.md').write_text(guide,encoding='utf8')
    for rater in ['A01','A02']:
        text=f'''# Instructions for evaluator {rater}

Your account: **{rater}**. The coordinator supplies your newly generated password privately. Previous study passwords do not automatically carry over. URL on the host computer: **http://127.0.0.1:8768/**.

You have **45 items from15 restaurants**. Read reviews,inspect the paginated original photo gallery,read both output sections,then score all six criteria using the printed1/3/5 anchors (2/4 are intermediate). Save draft to pause;Submit and lock when done. All Visual scores apply in this session. Work independently and do not consult previous scores or the other evaluator's work. Identical text in a different item is not within-output repetition.

On **Submitted45/45**,download **NEW_SESSION_{rater}_RESULTS_FINAL.json** and return it to the coordinator. Use Download my progress for your own partial backup. Do not edit JSON. Detailed workflow and rubric scopes are in READ_ME_FIRST_EN.md. No model names,benchmarks,previous scores or reference summaries are supplied.
'''
        (APP/('INSTRUCTIONS_'+rater+'_EN.md')).write_text(text,encoding='utf8')
    vi='''HƯỚNG DẪN NHANH — PHIÊN CHẤM MỚI

1. Giải nén toàn bộ thư mục trên Windows64-bit. Không cần cài Python hoặc thuê GPU.
2. Người điều phối xem trước bằng START_PREVIEW.cmd (chỉ đọc). Sau khi xem xét mới dùng START_COLLECTION.cmd để mở phiên chấm.
3. Lần đầu mở,cửa sổ console hiện tài khoản A01,A02,COORDINATOR và mật khẩu mới. Ghi lại riêng;chỉ đưa mỗi người mật khẩu của họ. Không có bước đăng ký hay chờ duyệt. Những lần mở sau không đổi mật khẩu và không xóa bài chấm.
4. Mỗi người chấm45 mục. Đọc reviews,xem ảnh qua các trang,đối chiếu Review Consensus và Visual Evidence. Chấm đủ6 tiêu chí từ1–5 theo rubric đang hiển thị. Không lấy điểm phiên cũ để điền.
5. Save draft để nghỉ;Submit and lock khi xong. Đợi báo lưu thành công. Không xóa STATE;giữ nguyên thư mục để làm tiếp.
6. Khi đạt Submitted45/45,bấm Download my completed results. Gửi NEW_SESSION_A01_RESULTS_FINAL.json hoặc NEW_SESSION_A02_RESULTS_FINAL.json tùy tài khoản. Không tự sửa JSON.
7. Hướng dẫn đầy đủ bằng tiếng Anh trong READ_ME_FIRST_EN.md và INSTRUCTIONS_A01_EN.md / INSTRUCTIONS_A02_EN.md. http://127.0.0.1:8768/ chỉ hoạt động trên máy đang chạy ứng dụng.
'''
    (APP/'BAT_DAU_VI.txt').write_text(vi,encoding='utf8')
    master='''# Stage P — new concurrent three-system human evaluation

REVIEW PACKAGE ONLY. No real ratings have been collected or imported. The original Stage O90-record collection remains unchanged and separate.

## Scope and fairness

15 frozen TripAdvisor restaurants ×3 systems ×2 evaluators =90 NEW records,45 per evaluator. Systems are original accepted Stage O LED F-S1000 (not RTX3090 benchmark reruns),frozen Qwen-Caption,and frozen Qwen-Image. All three are re-rated in the new session;old LED scores are not substituted. All15 Qwen Review components are identical across variants. Each combined output still receives all six assessments,but those repeated Review texts do not create independent generated content. Interpretation of Review-only score differences must acknowledge this.

Independent random seeds/order per evaluator;randomized three-system order within each restaurant. Position balance is exact:each system occurs5 times in each within-restaurant position per evaluator. Neutral C1/C2/C3 codes are local slot positions and do not identify a stable model. Aliases differ across raters. True model mapping and seeds are in COORDINATOR_ONLY,never in EVALUATOR_APP. Model styles may nevertheless be recognizable;returning evaluators' prior Stage O exposure cannot be erased by a new login/session. Actual participant identity and source inspection are not independently authenticated.

## Review and distribution

1. Inspect VALIDATION_REPORT.md and the frozen data/bindings. Review EVALUATOR_APP with START_PREVIEW.cmd. The package has no STATE database,passwords,collection-open marker or submitted scores.
2. Keep this master ZIP,COORDINATOR_ONLY and STATISTICS private. **Do not send this master ZIP to evaluators**:it contains the identity key for later unblinding.
3. A separate adjacent STAGE_P_BLINDED_EVALUATOR_WINDOWS.zip contains ONLY EVALUATOR_APP and is ready to distribute after your review. It excludes model identities,previous ratings,benchmarks and diagnostics. The two evaluators may use the same host with separate accounts,or independent extracted copies with the same frozen assignments. Do not collect the same account concurrently on multiple copies.
4. The evaluator package bundles official Windows x64 embeddable Python3.11.9 and license. No installed Python,GPU,internet or paid API is needed. The default URL is127.0.0.1:8768,so the old Stage O8767 application is not replaced.
5. After review,the operator intentionally runs START_COLLECTION.cmd. It records only that the operator opened this session;it does not invent supervisor approval,participant declarations or ratings. First launch automatically provisions A01/A02/COORDINATOR and prints distinct memorable passwords once,with salted PBKDF2 hashing in STATE. No plaintext credential file is created. Give each rater their own account/password privately.

The exact existing RUBRIC.json is byte-identical. All247 selected reviews and5,143 original photo renditions are shared across conditions;the full gallery,not just73 model-selected photos,is retained. Photo pages and zoom are inherited from the working application. Each output's original text is preserved,including header placement and wording. Do not retype or repair predictions.

## Return and analyze real ratings later

Receive NEW_SESSION_A01_RESULTS_FINAL.json and NEW_SESSION_A02_RESULTS_FINAL.json. Do not import old Stage O exports. Validate using a Python3.11+ interpreter:

```text
python STATISTICS/validate_stage_p_human_exports_v1.py --package . --exports path/to/NEW_SESSION_A01_RESULTS_FINAL.json path/to/NEW_SESSION_A02_RESULTS_FINAL.json
python STATISTICS/analyze_stage_p_human_ratings_v1.py --package . --exports path/to/NEW_SESSION_A01_RESULTS_FINAL.json path/to/NEW_SESSION_A02_RESULTS_FINAL.json --output NEW_STATISTICAL_RESULTS
```

Validation uses standard library only (the bundled EVALUATOR_APP/PYTHON/python.exe also works). Analysis additionally requires the pinned NumPy version in STATISTICS/requirements.txt;it does not load models or call an API. The coordinator's original repository CPU analysis environment can run it. Missing,incomplete,wrong-study,duplicate,mismatched or altered exports are rejected;no missing score is filled.

The frozen ANALYSIS_PLAN.json specifies all six criteria,means/SD/medians/distributions,per-rater results,QWK/exact agreement,and restaurant-paired contrasts. Bootstrap20,000 whole restaurant clusters retains both raters/all systems;exact exploratory2^15 sign-flip tests and Holm adjustment within12 primary/6 additional contrasts. Pointwise95% intervals are not multiplicity-adjusted. Two evaluators and15 convenience restaurants do not justify treating90 scores as independent replicates. Report comments/flags verbatim;unverified flags are not independently confirmed factual errors. No composite winner or final quality conclusion is produced now.

## Layout

- EVALUATOR_APP/:blinded portable app,all original evidence,new blank session.
- COORDINATOR_ONLY/:true assignment key,random seeds,position-balance ledger,45 frozen outputs,bindings and frozen analysis plan.
- STATISTICS/:export validation and future analysis scripts.
- VALIDATION/:actual software-test results,synthetic-only screenshots,integrity and preservation receipts. No test rating exports or test databases.
- REPRODUCIBILITY/:build/test scripts and original request.

Actual tests use isolated invented SYNTHETIC restaurants/outputs only. They do not count as human evaluation. No GPU inference,training,caption regeneration,old rating changes or production scoring occurred. STOP for researcher review before distribution.
'''
    (P/'README.md').write_text(master,encoding='utf8')
    stats=P/'STATISTICS';stats.mkdir(exist_ok=True)
    for name in ['validate_stage_p_human_exports_v1.py','analyze_stage_p_human_ratings_v1.py']:shutil.copy2(ROOT/'scripts'/name,stats/name)
    import importlib.metadata
    (stats/'requirements.txt').write_text('numpy=='+importlib.metadata.version('numpy')+'\n',encoding='ascii')
    p=P/'COORDINATOR_ONLY/BINDINGS.json';b=json.loads(p.read_text());b['analysis_plan_sha256']=hashlib.sha256((P/'COORDINATOR_ONLY/ANALYSIS_PLAN.json').read_bytes()).hexdigest();p.write_text(json.dumps(b,indent=2)+'\n',encoding='utf8')
    print('Wrote clear operator/rater guides; no passwords,old ratings or example production scores.')
if __name__=='__main__':main()
