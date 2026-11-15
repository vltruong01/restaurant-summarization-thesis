"""Build the read-only Vietnamese inspection notebook using standard JSON."""

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    cell = {"cell_type": kind, "id": f"inspection-{len(cells):03d}", "metadata": {},
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        compile(source, f"notebook-cell-{len(cells)}", "exec")
        cell.update({"execution_count": None, "outputs": []})
    cells.append(cell)


add("markdown", """
# Khám phá và kiểm tra dữ liệu Yelp cho thesis

Notebook giúp xem file, schema, review, metadata ảnh và ảnh gốc; kiểm tra dữ liệu thiếu, ID, rating, ngày, ngôn ngữ và độ dài token. **Chế độ mặc định đọc mẫu giới hạn**, đủ để mở và chạy nhanh trên máy local. Phần audit toàn bộ ở cuối chỉ chạy khi bật cờ.

Notebook chỉ quan sát dữ liệu. Phần 11 đọc kết quả audit/cohort đã được Stage B tạo riêng; không tự chọn lại businesses hoặc thay đổi split. Mẫu đầu file không phải mẫu ngẫu nhiên/đại diện; số liệu mẫu không được suy rộng thành thống kê toàn Yelp. Caption trong `photos.json` là caption nguồn/người dùng, không phải caption BLIP-2.

Chọn kernel **Python 3 (ipykernel)** đang có pandas, orjson, Pillow và PyYAML. Nếu một thư viện tùy chọn chưa có, phần tương ứng sẽ báo bỏ qua. Xem `README.md` để biết đường dẫn và hướng dẫn chạy.
""")
add("markdown", """
## 1. Cấu hình lần xem dữ liệu

Sửa các giới hạn dưới đây rồi chạy từ trên xuống. Bật audit toàn bộ sẽ đọc file review khoảng 5,3 GB; không nạp toàn bộ review vào RAM. Kiểm tra duplicate review ID toàn bộ dùng SQLite tạm trên đĩa, có thể mất thêm thời gian và dung lượng. Các cờ này không lọc hoặc thay đổi raw data.
""")
add("code", """
from pathlib import Path
import sys
import json
import importlib.util
import yaml
import pandas as pd
from IPython.display import display, Image as NotebookImage

BUSINESS_SAMPLE_SIZE = 5_000
REVIEW_SAMPLE_SIZE = 10_000
PHOTO_SAMPLE_SIZE = 5_000
AUXILIARY_SAMPLE_SIZE = 100
LANGUAGE_SAMPLE_SIZE = 50
TOKEN_SAMPLE_SIZE = 100
IMAGE_PREVIEW_COUNT = 6
BUSINESS_ID_OVERRIDE = None  # Có thể nhập business_id để xem các dòng có trong mẫu hiện tại.
RUN_FULL_AUDIT = False
CHECK_FULL_REVIEW_DUPLICATES = False

# Tìm root dù kernel chạy từ notebooks/ hay từ project root.
PROJECT_ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                     if (p / 'configs/project.yaml').is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Mở notebook bên trong thư mục yelp_dataset.')
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from data_inspection import (
    resolve_data_root, sample_jsonl, schema_table, sample_quality, is_restaurant,
    photo_inventory, inspect_photo_sample, full_core_audit,
)
config = yaml.safe_load((PROJECT_ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
DATA_ROOT = resolve_data_root(PROJECT_ROOT, config)
paths = {key: DATA_ROOT / config['paths'][key]
         for key in ['business_jsonl', 'review_jsonl', 'photo_jsonl', 'photo_directory']}
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 120)
display(pd.DataFrame([{'project_root': str(PROJECT_ROOT), 'data_root': str(DATA_ROOT),
                      'python': sys.executable, 'mode': 'QUICK / PREFIX SAMPLES',
                      'full_audit_enabled': RUN_FULL_AUDIT}]))
""")
add("markdown", """
## 2. File inventory và dữ liệu gốc

Yelp dùng JSONL: mỗi dòng là một JSON object. Không gọi `json.load()` hoặc `pd.read_json()` trên cả file review/user lớn. Bảng kích thước dưới đây đọc metadata filesystem; chưa đếm tổng số record.
""")
add("code", """
inventory_rows = []
for path in sorted(DATA_ROOT.glob('yelp_academic_dataset_*.json')) + [paths['photo_jsonl']]:
    exists = path.is_file()
    inventory_rows.append({'file': str(path.relative_to(DATA_ROOT)), 'exists': exists,
                           'size_GiB': round(path.stat().st_size / 1024**3, 3) if exists else None})
display(pd.DataFrame(inventory_rows))
assert all(paths[key].is_file() for key in ['business_jsonl', 'review_jsonl', 'photo_jsonl'])
assert paths['photo_directory'].is_dir()
photo_files, photo_file_inventory = photo_inventory(paths['photo_directory'])
display(pd.DataFrame([photo_file_inventory]))
print('Số file ảnh là thống kê toàn thư mục filename; chưa kiểm tra toàn bộ ảnh có đọc được hay không.')
""")
add("markdown", """
## 3. Đọc mẫu business, review và photo metadata

Các bảng là **mẫu prefix**: những record hợp lệ đầu file trong giới hạn đã đặt. Log đọc mẫu báo riêng dòng trống, JSON lỗi và JSON không phải object. Cột `complete_scan=False` có nghĩa là chưa đọc toàn file.
""")
add("code", """
business_records, business_sample_log = sample_jsonl(paths['business_jsonl'], BUSINESS_SAMPLE_SIZE)
review_records, review_sample_log = sample_jsonl(paths['review_jsonl'], REVIEW_SAMPLE_SIZE)
photo_records, photo_sample_log = sample_jsonl(paths['photo_jsonl'], PHOTO_SAMPLE_SIZE)
business_df = pd.DataFrame(business_records)
review_df = pd.DataFrame(review_records)
photo_df = pd.DataFrame(photo_records)
sample_logs = pd.DataFrame([dict(dataset=name, **log) for name, log in
                           [('business', business_sample_log), ('review', review_sample_log),
                            ('photo', photo_sample_log)]])
display(sample_logs.drop(columns=['path', 'error_examples']))
display(business_df[['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories']].head(8))
display(review_df[['review_id', 'business_id', 'stars', 'date', 'text']].head(5))
display(photo_df[['photo_id', 'business_id', 'label', 'caption']].head(8))
""")
add("code", """
for name, records in [('business', business_records), ('review', review_records), ('photo', photo_records)]:
    print()
    print(f'Schema của mẫu {name}: {len(records):,} records')
    display(schema_table(records))
""")
add("markdown", """
## 4. Kiểm tra chất lượng mẫu

Duplicate bên dưới chỉ được kiểm tra **trong mẫu**. Stars review phải là 1–5; stars trung bình của business có thể là số lẻ, nên không áp quy tắc review lên business. Ngày lỗi và text trống được báo để quan sát, không tự động loại khỏi dataset.
""")
add("code", """
for kind, records, id_field in [('business', business_records, 'business_id'),
                               ('review', review_records, 'review_id'),
                               ('photo', photo_records, 'photo_id')]:
    print(f'QC mẫu {kind}')
    display(sample_quality(records, kind, id_field))
review_dates = pd.to_datetime(review_df['date'], errors='coerce')
review_lengths = review_df['text'].map(lambda value: len(value) if isinstance(value, str) else 0)
display(pd.DataFrame([{'sample_reviews': len(review_df),
                      'earliest_date_in_sample': review_dates.min(),
                      'latest_date_in_sample': review_dates.max(),
                      'mean_characters': round(review_lengths.mean(), 1),
                      'median_characters': review_lengths.median()}]))
print('Mô tả độ dài text của từng review trong mẫu, chưa phải độ dài bộ 30 review/business.')
""")
add("code", """
star_distribution = review_df['stars'].value_counts(dropna=False).sort_index()
label_distribution = photo_df['label'].fillna('<missing>').value_counts()
display(star_distribution.rename('sample_review_count').to_frame())
display(label_distribution.rename('sample_photo_count').to_frame())
if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    star_distribution.plot.bar(ax=axes[0], color='#2563eb', rot=0)
    axes[0].set(title='Review stars — prefix sample', xlabel='Stars', ylabel='Sample records')
    label_distribution.plot.bar(ax=axes[1], color='#0d9488', rot=25)
    axes[1].set(title='Photo labels — prefix sample', xlabel='Label', ylabel='Sample records')
    fig.tight_layout()
    plt.show()
else:
    print('Matplotlib chưa có; các bảng phân bố bên trên vẫn dùng được.')
""")
add("markdown", """
## 5. Liên kết business–review–photo và xem một nhà hàng

Ảnh gắn với business qua `business_id`, không có mapping trực tiếp với `review_id`. Business vắng trong một mẫu prefix chưa thể được kết luận là orphan ID. Kiểm tra referential integrity toàn cục nằm trong audit đầy đủ ở cuối.

Ví dụ bên dưới chỉ hiển thị các review/photo đã đọc vào mẫu. Số lượng hiển thị không phải toàn bộ dữ liệu của nhà hàng đó.
""")
add("code", """
restaurant_df = business_df[business_df.apply(lambda row: is_restaurant(row.to_dict()), axis=1)].copy()
review_counts_in_sample = review_df.groupby('business_id').size().rename('reviews_in_current_sample')
photo_counts_in_sample = photo_df.groupby('business_id').size().rename('photos_in_current_sample')
business_examples = restaurant_df[['business_id', 'name', 'city', 'stars', 'review_count']].merge(
    review_counts_in_sample, on='business_id', how='left').merge(
    photo_counts_in_sample, on='business_id', how='left')
business_examples[['reviews_in_current_sample', 'photos_in_current_sample']] = business_examples[
    ['reviews_in_current_sample', 'photos_in_current_sample']].fillna(0).astype(int)
business_examples = business_examples.sort_values(
    ['photos_in_current_sample', 'reviews_in_current_sample'], ascending=False)
display(business_examples.head(12))
print('review_count là metadata nguồn; hai cột *_in_current_sample chỉ đếm các dòng trong mẫu.')
EXAMPLE_BUSINESS_ID = BUSINESS_ID_OVERRIDE or (
    business_examples.iloc[0]['business_id'] if len(business_examples) else None)
if EXAMPLE_BUSINESS_ID:
    print('Business đang xem:', EXAMPLE_BUSINESS_ID)
    display(business_df[business_df['business_id'] == EXAMPLE_BUSINESS_ID])
    display(review_df[review_df['business_id'] == EXAMPLE_BUSINESS_ID][
        ['review_id', 'stars', 'date', 'text']].sort_values('date').head(12))
    display(photo_df[photo_df['business_id'] == EXAMPLE_BUSINESS_ID].head(12))
""")
add("markdown", """
## 6. Kiểm tra và xem ảnh gốc

Kiểm tra dưới đây chỉ mở tối đa số ảnh preview đã đặt. Không suy ra tất cả ảnh đều hợp lệ; không thực hiện duplicate-content detection hoặc lọc ảnh của thí nghiệm.
""")
add("code", """
example_photos = [row for row in photo_records if row.get('business_id') == EXAMPLE_BUSINESS_ID]
preview_records = example_photos if example_photos else photo_records
photo_preview_qc = inspect_photo_sample(preview_records, paths['photo_directory'], photo_files, IMAGE_PREVIEW_COUNT)
display(photo_preview_qc)
for row in photo_preview_qc.to_dict('records'):
    if row['readable']:
        print(f"photo_id={row['photo_id']} | business_id={row['business_id']} | label={row['label']}")
        display(NotebookImage(filename=str(paths['photo_directory'] / row['local_file']), width=420))
""")
add("markdown", """
## 7. Ngôn ngữ và token review trong mẫu

Đây là chẩn đoán thăm dò. `langdetect` có thể sai, nhất là text ngắn; chưa dùng kết quả này để chốt eligibility hoặc lọc review. Cách áp dụng English-only vào Stage B/C vẫn cần thống nhất.

Token được tính bằng asset tokenizer LED đã xuất trong A1, không tải model hoặc gọi Hugging Face API. Độ dài dưới đây là **từng review**, chưa phải input business-level hoặc ngân sách token cuối cùng.
""")
add("code", """
language_rows = []
if importlib.util.find_spec('langdetect'):
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = config['reproducibility']['seed']
    for row in review_records[:LANGUAGE_SAMPLE_SIZE]:
        text = row.get('text', '')
        if not isinstance(text, str) or len(text.strip()) < 20:
            language = 'unknown/short'
        else:
            try:
                language = detect(text[:2000])
            except Exception:
                language = 'unknown/error'
        language_rows.append({'review_id': row.get('review_id'), 'language': language,
                              'text_preview': text[:180] if isinstance(text, str) else ''})
    language_df = pd.DataFrame(language_rows)
    display(language_df['language'].value_counts().rename('sample_reviews').to_frame())
    display(language_df.head(10))
else:
    print('langdetect chưa có; bỏ qua chẩn đoán ngôn ngữ.')

tokenizer_file = PROJECT_ROOT / config['tokenizer']['assets_directory'] / 'tokenizer.json'
token_rows = []
if tokenizer_file.is_file() and importlib.util.find_spec('tokenizers'):
    from tokenizers import Tokenizer
    led_tokenizer = Tokenizer.from_file(str(tokenizer_file))
    for row in review_records[:TOKEN_SAMPLE_SIZE]:
        text = row.get('text', '')
        text = text if isinstance(text, str) else ''
        token_rows.append({'review_id': row.get('review_id'),
                           'led_tokens': len(led_tokenizer.encode(text, add_special_tokens=True).ids)})
    token_df = pd.DataFrame(token_rows)
    display(token_df['led_tokens'].describe(percentiles=[.5, .9, .95]).to_frame())
    display(token_df.sort_values('led_tokens', ascending=False).head(8))
    print('Tokenizer revision:', config['tokenizer']['revision'])
else:
    print('Thiếu asset/tokenizers; chạy Stage A1 hoặc dùng kernel có tokenizers để xem phần này.')
""")
add("markdown", """
## 8. Các file phụ và báo cáo sơ bộ đã có

User, tip và check-in không phải input chính của thiết kế hiện tại, nhưng có thể xem schema bằng mẫu nhỏ. Các CSV dưới đây là **artifact cũ**, không phải kết quả tính lại trong notebook. `table_3x.csv` dùng pool sơ bộ khác với điều kiện cuối; đọc cùng `eligibility_thresholds.csv` để tránh nhầm số lượng.
""")
add("code", """
auxiliary_samples = {}
for name in ['user', 'tip', 'checkin']:
    path = DATA_ROOT / f'yelp_academic_dataset_{name}.json'
    if path.is_file():
        records, log = sample_jsonl(path, AUXILIARY_SAMPLE_SIZE)
        auxiliary_samples[name] = records
        print(f'{name}: {len(records)} prefix records; complete_scan={log["complete_scan"]}')
        display(schema_table(records))
        preview = pd.DataFrame(records).head(2).copy()
        for column in preview.columns:
            preview[column] = preview[column].map(lambda value: str(value)[:160])
        display(preview)

for relative in ['processed/preliminary_stats/table_3x.csv',
                 'processed/preliminary_stats/eligibility_thresholds.csv',
                 'processed/language_stats/table_3_2_language_row.csv',
                 'processed/language_stats/restaurant_language_distribution.csv']:
    path = DATA_ROOT / relative
    if path.is_file():
        print('Báo cáo có sẵn:', relative)
        display(pd.read_csv(path))
""")
add("markdown", """
## 9. Audit toàn bộ core files — chủ động bật khi cần

Đặt `RUN_FULL_AUDIT=True` ở đầu rồi chạy lại cell dưới. Audit đọc đầy đủ business, photo metadata và review theo streaming; kiểm tra ID, rating, date, text, file ảnh có trên máy và tính eligibility **gốc**: Restaurants, loại accommodation, ≥10 raw review records và ≥5 ảnh có file local.

Không lọc tiếng Anh, không kiểm tra nội dung mọi ảnh, không loại duplicate ảnh, không chọn 2.300 businesses và không tạo train/validation/test. Count review metadata của business không được dùng thay cho số dòng review thực tế. Nếu muốn kiểm tra duplicate review ID toàn bộ, bật thêm `CHECK_FULL_REVIEW_DUPLICATES=True`; SQLite tạm có thể cần hàng trăm MB và được dọn sau khi chạy.

So sánh 10.197 chỉ giúp đối chiếu thống kê sơ bộ. Parse errors/duplicate ID/data-quality flags phải được xem cùng kết quả trước khi dùng cho Stage B.
""")
add("code", """
full_audit_result = None
if RUN_FULL_AUDIT:
    full_audit_result = full_core_audit(
        paths['business_jsonl'], paths['review_jsonl'], paths['photo_jsonl'], paths['photo_directory'],
        min_reviews=config['study']['minimum_reviews'], min_photos=config['study']['minimum_local_photos'],
        track_review_duplicates=CHECK_FULL_REVIEW_DUPLICATES, progress=True,
    )
    display(pd.DataFrame([{
        'scope': full_audit_result['scope'],
        'business_records': full_audit_result['sources']['business']['records'],
        'review_records': full_audit_result['sources']['review']['records'],
        'photo_metadata_records': full_audit_result['sources']['photo']['records'],
        'eligible_count_original_rule': full_audit_result['eligible_count'],
        'matches_preliminary_10197': full_audit_result['eligible_count'] == 10197,
        'parse_errors': full_audit_result['parse_errors'],
        'counts_need_review': full_audit_result['counts_need_review'],
        'review_duplicate_ids_checked': full_audit_result['review_duplicate_ids_checked'],
        'duplicate_review_occurrences': full_audit_result['review_duplicate_id_occurrences'],
        'elapsed_seconds': full_audit_result['elapsed_seconds'],
    }]))
    display(pd.DataFrame(full_audit_result['sources']).T.drop(columns=['path', 'error_examples']))
    display(pd.Series(full_audit_result['quality_flags'], name='count').to_frame())
    display(pd.DataFrame(full_audit_result['eligible_businesses']).head(20))
else:
    print('FULL AUDIT CHƯA CHẠY. Kết quả phía trên là mẫu prefix và báo cáo đã có.')
    print('Đổi RUN_FULL_AUDIT=True khi muốn kiểm tra toàn bộ; raw data giữ nguyên.')
""")
add("markdown", """
## 10. Những gì notebook này đã và chưa kiểm chứng

- Xem inventory, schema, mẫu review/photo, một ví dụ business và một số ảnh gốc.
- Quick QC chỉ áp dụng trong phạm vi mẫu; prefix không đại diện cho phân bố toàn dataset.
- Số filename ảnh là inventory toàn thư mục, không phải số ảnh đạt Stage D.
- Nếu full audit chưa bật, notebook không tự tái xác minh pool 10.197 từ raw files; xem báo cáo Stage B đã lưu ở phần 11.
- Notebook không tự tạo cohort hoặc khóa split. Stage C chọn review tiếng Anh, token budget, K và teacher cần spec riêng.

Bước tiếp theo là dùng các quan sát để hoàn thiện spec Stage B/C, sau đó triển khai và kiểm chứng theo acceptance criteria. Không lấy giá trị từ mẫu prefix làm ngưỡng nghiên cứu.
""")
add("markdown", """
## 11. Kết quả xác minh và dataset 2.300 businesses đã lưu

Phần này đọc **báo cáo full scan đã được chạy riêng**, không quét lại 7 triệu review. Baseline metadata có hai photo IDs lặp với label khác nhau; báo cáo reviewed gate kiểm tra ảnh duy nhất và xác nhận cả hai businesses đó nằm ngoài pool đủ điều kiện. Xem cả baseline và reviewed report để hiểu cảnh báo.

Dataset Stage B chứa toàn bộ review và ảnh nguồn của 2.300 businesses. Chưa lọc tiếng Anh hoặc chọn tối đa 30 review; chưa có BLIP-2 captions/targets. Mỗi tập thí nghiệm là một business; số dòng review bên dưới là số review nguồn. Chỉ đọc manifest và vài dòng review mẫu, không sửa dataset.
""")
add("code", """
saved_reports = {}
for name in ['eligibility_audit', 'eligibility_verified', 'validation_report']:
    report_path = PROJECT_ROOT / 'outputs/cohort' / f'{name}.json'
    if report_path.is_file():
        saved_reports[name] = json.loads(report_path.read_text(encoding='utf-8'))
if 'eligibility_verified' in saved_reports:
    verified = saved_reports['eligibility_verified']
    display(pd.DataFrame([{key: verified[key] for key in [
        'gate_passed', 'eligible_count', 'raw_review_records', 'local_unique_photos',
        'duplicate_metadata_occurrences', 'eligible_ids_unchanged_from_baseline'
    ]}]))
    print('Đây là báo cáo đã lưu từ full scan/unique-photo verification riêng, không phải kết quả prefix.')
else:
    print('Chưa có reviewed Stage B eligibility report.')

cohort_config = yaml.safe_load((PROJECT_ROOT / 'configs/cohort.yaml').read_text(encoding='utf-8'))
cohort_path = PROJECT_ROOT / cohort_config['output_directory']
if (cohort_path / 'dataset_manifest.json').is_file():
    cohort_manifest = json.loads((cohort_path / 'dataset_manifest.json').read_text(encoding='utf-8'))
    split_df = pd.read_csv(cohort_path / 'business_split.csv')
    assert len(split_df) == 2300 and split_df['business_id'].is_unique
    assert split_df['split'].value_counts().to_dict() == {'train': 2000, 'validation': 200, 'test': 100}
    display(pd.DataFrame(cohort_manifest['split_summary']).T)
    display(pd.read_csv(cohort_path / 'cohort_businesses.csv').head(10))
    display(pd.read_csv(cohort_path / 'photo_manifest.csv', nrows=10))
    for split_name in ['train', 'validation', 'test']:
        source_reviews, source_sample_log = sample_jsonl(cohort_path / 'reviews' / f'{split_name}.jsonl', 3)
        preview = pd.DataFrame(source_reviews)[['review_id', 'business_id', 'stars', 'date', 'text']]
        preview['text'] = preview['text'].str.slice(0, 300)
        print(f'{split_name}: 3 dòng review nguồn đầu file, không đại diện cho toàn split')
        display(preview)
    if 'validation_report' in saved_reports:
        acceptance = saved_reports['validation_report']
        print('Independent validation PASS:', acceptance['validation_passed'])
        display(pd.DataFrame(acceptance['checks']))
    else:
        print('Dataset đã xuất; independent validation chưa có báo cáo.')
else:
    print('Chưa có dataset Stage B hoàn tất tại', cohort_path)
""")

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.7", "file_extension": ".py",
                      "mimetype": "text/x-python", "pygments_lexer": "ipython3"},
}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/01_yelp_data_inspection.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {destination} ({len(cells)} cells)")
