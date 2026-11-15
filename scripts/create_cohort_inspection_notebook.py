"""Build a Vietnamese, read-only overview of the immutable Stage B cohort."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    cell = {'cell_type': kind, 'id': f'cohort-overview-{len(cells):03d}', 'metadata': {},
            'source': source.splitlines(keepends=True)}
    if kind == 'code':
        compile(source, f'cohort-cell-{len(cells)}', 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', """
# Kiểm tra English eligibility trước khi lấy 2.300 businesses

Notebook kiểm tra **toàn bộ pool gốc 10.197 businesses** trước khi đề xuất cohort 2.300. Phần 1A–1C kiểm tra ≥10 English reviews và ≥5 unique local photos, so sánh cohort v1, rồi đề xuất split mới. Các phần 2–9 giữ thống kê **cohort nguồn v1 để đối chiếu**, không phải thống kê dataset English v2 đã được xây dựng.

**Một tập thí nghiệm = một business.** Review ở đây là toàn bộ review nguồn, chưa lọc tiếng Anh hoặc chọn tối đa 30 review. Caption trong photo metadata là caption Yelp, chưa phải BLIP-2. Notebook không thay đổi dataset/split và không chọn review, ảnh hoặc ngưỡng nghiên cứu.

English gate quét toàn bộ file review gốc nhưng chỉ gọi detector đến khi chứng minh business có ≥10 English reviews; business không đạt được kiểm tra hết review. **Count English của business đạt có thể là cận dưới, không phải tổng chính xác.** Cache/report cho phép chạy lại nhanh. Sau đó đọc toàn bộ review của cohort v1 để đối chiếu thống kê; không nạp toàn bộ corpus text vào RAM. Full image validation vẫn thuộc Stage D.

Mở bằng kernel **Python 3 (ipykernel)** có pandas, numpy, matplotlib, orjson, Pillow và PyYAML. Chạy **Run All**. Các bảng/biểu đồ đã chạy được lưu ngay trong notebook; báo cáo phụ nằm ở `outputs/cohort_inspection/`.
""")
add('markdown', """
## 1. Cấu hình

`TOP_N` giới hạn số dòng trong bảng xếp hạng, không giới hạn dữ liệu thống kê. Nhập `BUSINESS_ID_TO_VIEW` để xem một business cụ thể; mặc định xem business nhiều review nhất trong train. Thay ID, chạy lại cell cấu hình này rồi cell phần 8 để xem business khác. Phần thống kê toàn cohort chỉ cần chạy một lần mỗi session.
""")
add('code', """
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import os
import sys
import json
import hashlib
import time
import orjson
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from IPython.display import display

TOP_N = 15
BUSINESS_ID_TO_VIEW = None  # Ví dụ: '-1B9pP_CrRBJYPICE5WbRA'
REVIEW_PREVIEW_N = 8
PHOTO_PREVIEW_N = 6
DATASET_DIR_OVERRIDE = None  # Chỉ cần khi package được đặt ở nơi khác.

PROJECT_ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                     if (p / 'configs/cohort.yaml').is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Mở notebook bên trong thư mục project yelp_dataset.')
cohort_config = yaml.safe_load((PROJECT_ROOT / 'configs/cohort.yaml').read_text(encoding='utf-8'))
DATASET_DIR = (Path(DATASET_DIR_OVERRIDE).expanduser().resolve() if DATASET_DIR_OVERRIDE
               else (PROJECT_ROOT / cohort_config['output_directory']).resolve())
REPORT_DIR = PROJECT_ROOT / 'outputs/cohort_inspection'
assert not REPORT_DIR.resolve().is_relative_to(DATASET_DIR), 'Output phải nằm ngoài dataset.'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = REPORT_DIR / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)
SPLIT_ORDER = ['train', 'validation', 'test']
COLORS = {'train': '#2563eb', 'validation': '#f59e0b', 'test': '#16a34a'}
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 130)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
assert TOP_N > 0 and REVIEW_PREVIEW_N > 0 and PHOTO_PREVIEW_N > 0
assert (DATASET_DIR / 'dataset_manifest.json').is_file(), DATASET_DIR
print('Dataset:', DATASET_DIR)
print('Thống kê: toàn cohort; review được đọc bằng streaming.')

def scan_jsonl(path, stats=None):
    stats = stats if stats is not None else {}
    digest = hashlib.sha256()
    count = 0
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            record = orjson.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'JSON không phải object: {path}:{number}')
            count += 1
            yield record
    stats.update(records=count, sha256=digest.hexdigest(), complete_scan=True)

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)
""")
add('markdown', """
## 1A. Xét English eligibility trên toàn bộ pool trước sampling

Reuse settings của script language sơ bộ: langdetect 1.0.9, seed 42, strip text, <20 ký tự => unknown, classify tối đa 2.000 ký tự, chỉ label `en` được tính. Không thêm confidence cutoff. Đây là nhận diện tự động, chưa phải language labels được con người kiểm chứng.

Lần đầu có thể mất vài phút với tám CPU workers. Worker/cache nằm trong module riêng để chạy được trên Windows. Cache được ghi theo batch để có thể tiếp tục sau gián đoạn. Kết quả đã hoàn tất chỉ được dùng lại khi source/config/artifact checks khớp. Không chỉnh sửa dataset v1 hoặc canonical split.
""")
add('code', """
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from english_eligibility import run_english_gate
english_config = yaml.safe_load((PROJECT_ROOT / 'configs/english_eligibility.yaml').read_text(encoding='utf-8'))
print('English eligibility policy:')
display(pd.DataFrame([{
    'min_English_reviews': english_config['minimum_english_reviews'],
    'min_local_photos': english_config['minimum_local_photos'],
    **english_config['language_detector'],
}]))
english_gate_report = run_english_gate(PROJECT_ROOT)
ENGLISH_REPORT_DIR = PROJECT_ROOT / english_config['execution']['cache_directory']
display(pd.DataFrame([{key: english_gate_report[key] for key in [
    'baseline_businesses', 'english_eligible_businesses', 'english_ineligible_businesses',
    'reviews_language_assessed', 'all_businesses_decided', 'all_reviews_language_classified',
    'current_cohort_ineligible_businesses', 'old_cohort_businesses_removed', 'new_cohort_businesses_added',
]}]))
print(english_gate_report['scope_note'])
""")
add('markdown', """
## 1B. Pool đạt/không đạt và ảnh hưởng lên cohort v1

`english_review_count_lower_bound` là số English reviews thực sự đã phát hiện. Chỉ khi `english_count_is_exact=True` mới là tổng chính xác. **Mọi business không đạt phải được kiểm tra hết review.** Những reviews chưa assessed của business đã đạt không được coi là non-English.

Không dùng phân bố ngôn ngữ của phần đã assessed để suy ra tỷ lệ tiếng Anh của toàn corpus: assessment dừng khi đủ điều kiện. Các kiểm tra photo ở đây là ID/file presence/size và số ảnh, không phải readability/content QC.
""")
add('code', """
english_decisions = pd.read_csv(ENGLISH_REPORT_DIR / 'business_english_decisions.csv')
english_passed = pd.read_csv(ENGLISH_REPORT_DIR / 'english_eligible_businesses.csv')
english_failed = pd.read_csv(ENGLISH_REPORT_DIR / 'english_ineligible_businesses.csv')
v1_comparison = pd.read_csv(ENGLISH_REPORT_DIR / 'current_cohort_comparison.csv')
assert len(english_decisions) == 10197 and english_decisions['business_id'].is_unique
assert english_passed['english_review_count_lower_bound'].ge(10).all()
assert english_passed['local_photo_records'].ge(5).all()
assert english_failed['english_review_count_lower_bound'].lt(10).all()
assert english_failed['english_count_is_exact'].all()
assert english_failed['reviews_language_assessed'].eq(english_failed['raw_review_records']).all()
assert set(english_passed['business_id']).isdisjoint(set(english_failed['business_id']))
assert set(english_passed['business_id']) | set(english_failed['business_id']) == set(english_decisions['business_id'])
project_config = yaml.safe_load((PROJECT_ROOT / 'configs/project.yaml').read_text(encoding='utf-8'))
from data_inspection import resolve_data_root
raw_data_root = resolve_data_root(PROJECT_ROOT, project_config)
baseline_ids = set(english_decisions['business_id'])
pool_names = pd.DataFrame([
    {key: row.get(key) for key in ['business_id', 'name', 'city', 'state']}
    for row in scan_jsonl(raw_data_root / project_config['paths']['business_jsonl'])
    if row['business_id'] in baseline_ids
])
assert pool_names['business_id'].is_unique and len(pool_names) == len(english_decisions)
print('Businesses không đạt trong toàn pool — số English tại đây là exact:')
display(english_failed.merge(pool_names, on='business_id', validate='one_to_one'))
print('Businesses thuộc cohort v1 nhưng không đạt English eligibility:')
v1_failures = v1_comparison.loc[~v1_comparison['eligible_english']]
display(v1_failures.merge(pool_names, on='business_id', validate='one_to_one'))
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(['Pool gốc', 'Đạt English gate', 'Không đạt'],
              [len(english_decisions), len(english_passed), len(english_failed)],
              color=['#64748b', '#16a34a', '#dc2626'])
ax.bar_label(bars, padding=3)
ax.set(ylabel='Số businesses', title='Eligibility hoàn tất trước khi lấy 2.300 businesses')
ax.set_ylim(0, len(english_decisions) * 1.12)
save_figure(fig, 'english_eligibility_gate.png')
""")
add('markdown', """
## 1C. Đề xuất lại 2.300 businesses sau English gate

Sorted qualified IDs → fresh `random.Random(42).sample(..., 2300)` → 2.000/200/100 theo selection position. Đây là **proposed manifest**, không tự xây dựng/ghi đè package v1 hay split đã có. Nếu IDs/split thay đổi, bước sau phải tạo dataset vật lý phiên bản mới rồi validate trước khi dùng Stage C.

Các số review trong bảng đề xuất là **review nguồn**, không phải số English tổng hoặc số ≤30 đã chọn. Exact English corpus và representative review selection cho cohort cuối vẫn là bước tiếp theo.
""")
add('code', """
import random
proposed_split = pd.read_csv(ENGLISH_REPORT_DIR / 'proposed_business_split.csv')
proposed_order = proposed_split.sort_values('selection_index')
expected_draw = random.Random(42).sample(sorted(english_passed['business_id']), 2300)
assert len(proposed_split) == 2300 and proposed_split['business_id'].is_unique
assert proposed_order['business_id'].tolist() == expected_draw
assert proposed_order['selection_index'].tolist() == list(range(2300))
assert proposed_split['split'].value_counts().to_dict() == {'train': 2000, 'validation': 200, 'test': 100}
assert proposed_split['english_review_count_lower_bound'].ge(10).all()
assert proposed_split['local_photo_records'].ge(5).all()
assert all(row.split == ('train' if row.selection_index < 2000 else 'validation' if row.selection_index < 2200 else 'test')
           for row in proposed_split.itertuples())
proposed_summary = proposed_split.groupby('split').agg(
    businesses=('business_id', 'size'), raw_source_reviews=('raw_review_records', 'sum'),
    original_local_photos=('local_photo_records', 'sum'),
).reindex(SPLIT_ORDER)
print('Manifest đề xuất:', ENGLISH_REPORT_DIR / 'proposed_business_split.csv')
display(proposed_summary)
display(proposed_split.merge(pool_names, on='business_id', validate='one_to_one').head(15))
print('Physical dataset rebuilt:', english_gate_report['creates_physical_dataset'])
print('Cohort v1 không đạt English gate:', english_gate_report['current_cohort_ineligible_businesses'])
print('IDs bị thay khỏi đề xuất mới:', english_gate_report['old_cohort_businesses_removed'])
print('IDs mới vào đề xuất:', english_gate_report['new_cohort_businesses_added'])
print('Phần 2–9 bên dưới chỉ phân tích cohort NGUỒN v1 để đối chiếu.')
""")
add('markdown', """
## 2. Đọc manifest và inventory ảnh

**Từ phần này trở đi: cohort nguồn v1 lịch sử, không phải proposed English-qualified cohort.**

Đọc các bảng business/split/photo, kiểm tra IDs và liên kết split, đối chiếu file ảnh thực tế. Counts `review_count` trong business metadata không được dùng thay cho số review của package. Review sẽ được đếm lại ở phần 3.
""")
add('code', """
manifest = json.loads((DATASET_DIR / 'dataset_manifest.json').read_text(encoding='utf-8'))
checksum_inventory = json.loads((DATASET_DIR / 'checksums.json').read_text(encoding='utf-8'))
business_df = pd.read_csv(DATASET_DIR / 'cohort_businesses.csv', keep_default_na=False)
split_df = pd.read_csv(DATASET_DIR / 'business_split.csv', keep_default_na=False)
photo_df = pd.read_csv(DATASET_DIR / 'photo_manifest.csv', keep_default_na=False)
business_records = list(scan_jsonl(DATASET_DIR / 'businesses.jsonl'))
original_businesses = {row['business_id']: row for row in business_records}
assert business_df['business_id'].is_unique and split_df['business_id'].is_unique
assert set(business_df['business_id']) == set(split_df['business_id']) == set(original_businesses)
assert len(business_records) == len(original_businesses) == 2300
assert business_df['split'].value_counts().to_dict() == {'train': 2000, 'validation': 200, 'test': 100}
membership = split_df.set_index('business_id')['split'].to_dict()
assert business_df.set_index('business_id')['split'].to_dict() == membership
assert photo_df['photo_id'].is_unique and photo_df['relative_path'].is_unique
assert photo_df['business_id'].map(membership).eq(photo_df['split']).all()
photo_files = {}
with os.scandir(DATASET_DIR / 'photos') as entries:
    for entry in entries:
        if entry.is_file():
            photo_files[f'photos/{entry.name}'] = entry.stat().st_size
assert set(photo_files) == set(photo_df['relative_path'])
assert photo_df['relative_path'].map(photo_files).eq(photo_df['bytes']).all()
for relative in ['cohort_businesses.csv', 'business_split.csv', 'photo_manifest.csv',
                 'businesses.jsonl', 'dataset_manifest.json']:
    actual_hash = hashlib.sha256((DATASET_DIR / relative).read_bytes()).hexdigest()
    assert actual_hash == checksum_inventory[relative]['sha256'], relative
display(pd.DataFrame([{
    'dataset': manifest['dataset_version'], 'businesses': len(business_df),
    'reviews_trong_manifest': manifest['raw_reviews'], 'photo_files_thực_tế': len(photo_files),
    'photo_size_MiB': round(sum(photo_files.values()) / 1024**2, 2),
    'seed': manifest['seed'], 'lọc_tiếng_Anh': manifest['english_filter_applied'],
    'đã_chọn_review': manifest['review_selection_applied'],
}]))
display(business_df.head(8))
""")
add('markdown', """
## 3. Đếm lại toàn bộ review và photo metadata

Đọc đủ ba file review của cohort. Kiểm tra review ID trùng, business/split joins và SHA-256 của file đã đọc. Tính rating, độ dài **ký tự** và khoảng thời gian review cho mỗi business. Độ dài ký tự không phải số token LED. Phần này không thực hiện language detection hoặc cắt review.
""")
add('code', """
started = time.perf_counter()
review_counts = Counter()
rating_counts = Counter()
ratings_by_business = defaultdict(Counter)
years_by_split = {split: Counter() for split in SPLIT_ORDER}
character_totals = Counter()
first_date, last_date = {}, {}
review_ids = set()
quality = Counter({'duplicate_review_ids': 0, 'invalid_join': 0, 'empty_text': 0,
                   'invalid_stars': 0, 'invalid_date': 0})
review_scan_reports = {}
for split in SPLIT_ORDER:
    stats = {}
    for row in scan_jsonl(DATASET_DIR / 'reviews' / f'{split}.jsonl', stats):
        bid, rid = row.get('business_id'), row.get('review_id')
        quality['duplicate_review_ids'] += rid in review_ids
        review_ids.add(rid)
        quality['invalid_join'] += membership.get(bid) != split
        review_counts[bid] += 1
        text = row.get('text')
        quality['empty_text'] += not isinstance(text, str) or not text.strip()
        character_totals[bid] += len(text) if isinstance(text, str) else 0
        stars = row.get('stars')
        if isinstance(stars, bool) or stars not in (1, 2, 3, 4, 5):
            quality['invalid_stars'] += 1
        else:
            rating_counts[int(stars)] += 1
            ratings_by_business[bid][int(stars)] += 1
        date = row.get('date')
        try:
            datetime.fromisoformat(date)
        except (ValueError, TypeError):
            quality['invalid_date'] += 1
        else:
            first_date[bid] = min(first_date.get(bid, date), date)
            last_date[bid] = max(last_date.get(bid, date), date)
            years_by_split[split][int(date[:4])] += 1
    relative = f'reviews/{split}.jsonl'
    assert stats['sha256'] == checksum_inventory[relative]['sha256'], relative
    review_scan_reports[split] = stats
    print(f'{split}: đã đọc đủ {stats["records"]:,} reviews; SHA-256 khớp.')
assert not any(quality.values()), dict(quality)
assert len(review_ids) == sum(review_counts.values())
assert set(review_counts) == set(membership)
del review_ids  # Không cần giữ 601k IDs trong RAM sau kiểm tra.

metadata_photo_counts = Counter()
metadata_photo_ids = set()
for split in SPLIT_ORDER:
    stats = {}
    for row in scan_jsonl(DATASET_DIR / 'photo_metadata' / f'{split}.jsonl', stats):
        bid, pid = row['business_id'], row['photo_id']
        assert membership.get(bid) == split and pid not in metadata_photo_ids
        metadata_photo_ids.add(pid)
        metadata_photo_counts[bid] += 1
    assert stats['sha256'] == checksum_inventory[f'photo_metadata/{split}.jsonl']['sha256']
assert metadata_photo_ids == set(photo_df['photo_id'])
photo_counts = photo_df.groupby('business_id').size()
stats_df = business_df.copy()
stats_df['reviews'] = stats_df['business_id'].map(review_counts)
stats_df['photos'] = stats_df['business_id'].map(photo_counts)
stats_df['mean_review_stars'] = stats_df['business_id'].map(
    {bid: sum(star * n for star, n in counts.items()) / review_counts[bid]
     for bid, counts in ratings_by_business.items()})
stats_df['first_review_date'] = pd.to_datetime(stats_df['business_id'].map(first_date))
stats_df['last_review_date'] = pd.to_datetime(stats_df['business_id'].map(last_date))
stats_df['review_span_years'] = (stats_df['last_review_date'] - stats_df['first_review_date']).dt.days / 365.25
stats_df['mean_review_characters'] = stats_df['business_id'].map(character_totals) / stats_df['reviews']
stats_df['metadata_business_review_count'] = stats_df['business_id'].map(
    {bid: row.get('review_count') for bid, row in original_businesses.items()})
assert stats_df['reviews'].eq(stats_df['raw_review_records']).all()
assert stats_df['photos'].eq(stats_df['local_photo_records']).all()
assert stats_df['photos'].eq(stats_df['business_id'].map(metadata_photo_counts)).all()
assert stats_df['reviews'].min() >= 10 and stats_df['photos'].min() >= 5
assert stats_df['reviews'].sum() == manifest['raw_reviews'] == 601493
assert stats_df['photos'].sum() == manifest['local_photos'] == len(photo_files) == 30027
print(f'Full cohort scan hoàn tất: {time.perf_counter() - started:.2f}s')
display(pd.Series(dict(quality), name='số_lỗi').to_frame())
""")
add('markdown', """
## 4. Tổng quan, theo split và các mức phân vị

Mean/median/min/max và p90/p95/p99 được tính trên **số review/ảnh mỗi business**. Rating trung bình mỗi business và số review gốc trong business metadata được giữ riêng để tránh nhầm với số dòng review thực tế.
""")
add('code', """
split_summary = stats_df.groupby('split').agg(
    businesses=('business_id', 'size'), reviews=('reviews', 'sum'), photos=('photos', 'sum'),
    mean_reviews=('reviews', 'mean'), median_reviews=('reviews', 'median'),
    min_reviews=('reviews', 'min'), max_reviews=('reviews', 'max'),
    mean_photos=('photos', 'mean'), median_photos=('photos', 'median'),
    min_photos=('photos', 'min'), max_photos=('photos', 'max'),
).reindex(SPLIT_ORDER)
for split in SPLIT_ORDER:
    expected = manifest['split_summary'][split]
    assert int(split_summary.loc[split, 'businesses']) == expected['businesses']
    assert int(split_summary.loc[split, 'reviews']) == expected['raw_reviews']
    assert int(split_summary.loc[split, 'photos']) == expected['local_photos']
display(pd.DataFrame([{'businesses': len(stats_df), 'reviews': int(stats_df['reviews'].sum()),
                      'photos': int(stats_df['photos'].sum()),
                      'mean_reviews_per_business': stats_df['reviews'].mean(),
                      'mean_photos_per_business': stats_df['photos'].mean()}]).round(2))
display(split_summary.round(2))
distribution_summary = stats_df[['reviews', 'photos']].describe(percentiles=[.25, .5, .75, .9, .95, .99]).T
display(distribution_summary.round(2))
display(stats_df[['mean_review_stars', 'mean_review_characters', 'review_span_years']].describe().T.round(2))
""")
add('markdown', """
## 5. Nhà hàng nhiều/ít review và ảnh nhất

Mỗi bảng hiển thị tối đa `TOP_N` businesses. Bảng extrema ghi **giá trị min/max và số business đồng hạng**; nếu có nhiều đồng hạng, bảng top chỉ hiển thị một phần theo business ID. Biến `fewest_review_businesses`, `fewest_photo_businesses`, `most_review_businesses`, `most_photo_businesses` chứa toàn bộ nhóm đồng hạng.
""")
add('code', """
rank_columns = ['business_id', 'name', 'city', 'state', 'split', 'reviews', 'photos',
                'mean_review_stars', 'first_review_date', 'last_review_date']
extrema_rows = []
for column in ['reviews', 'photos']:
    for direction, value in [('minimum', stats_df[column].min()), ('maximum', stats_df[column].max())]:
        extrema_rows.append({'metric': column, 'extreme': direction, 'value': int(value),
                             'businesses_at_extreme': int(stats_df[column].eq(value).sum())})
extrema_df = pd.DataFrame(extrema_rows)
display(extrema_df)
fewest_review_businesses = stats_df.loc[stats_df['reviews'].eq(stats_df['reviews'].min()), rank_columns].sort_values('business_id')
most_review_businesses = stats_df.loc[stats_df['reviews'].eq(stats_df['reviews'].max()), rank_columns].sort_values('business_id')
fewest_photo_businesses = stats_df.loc[stats_df['photos'].eq(stats_df['photos'].min()), rank_columns].sort_values('business_id')
most_photo_businesses = stats_df.loc[stats_df['photos'].eq(stats_df['photos'].max()), rank_columns].sort_values('business_id')
rankings = {}
for column, ascending, title in [
    ('reviews', False, 'NHIỀU REVIEW NHẤT'), ('reviews', True, 'ÍT REVIEW NHẤT'),
    ('photos', False, 'NHIỀU ẢNH NHẤT'), ('photos', True, 'ÍT ẢNH NHẤT'),
]:
    ranking = stats_df.sort_values([column, 'business_id'], ascending=[ascending, True])
    rankings[f'{column}_{"min" if ascending else "max"}'] = ranking[rank_columns].head(TOP_N)
    print(f'{title} — top {TOP_N}')
    display(rankings[f'{column}_{"min" if ascending else "max"}'].round({'mean_review_stars': 2}))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, column, title in zip(axes, ['reviews', 'photos'], ['10 businesses nhiều review nhất', '10 businesses nhiều ảnh nhất']):
    top = stats_df.sort_values([column, 'business_id'], ascending=[False, True]).head(10).iloc[::-1]
    y = np.arange(len(top))
    bars = ax.barh(y, top[column], color=[COLORS[s] for s in top['split']])
    ax.set_yticks(y, [f'{name[:35]} ({city})' for name, city in zip(top['name'], top['city'])], fontsize=8)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlim(0, top[column].max() * 1.16)
    ax.set_xlabel('Số review' if column == 'reviews' else 'Số ảnh')
    ax.set_title(title)
save_figure(fig, 'top_businesses.png')
""")
add('markdown', """
## 6. Phân bố review và ảnh

Histogram và scatter sử dụng trục log để nhìn được cả businesses nhỏ và lớn. Các nhóm số lượng là cách trình bày thống kê, không phải ngưỡng lọc hoặc tiêu chí chọn human evaluation.
""")
add('code', """
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, column, title, color in [(axes[0, 0], 'reviews', 'Review / business', '#2563eb'),
                               (axes[0, 1], 'photos', 'Ảnh / business', '#16a34a')]:
    values = stats_df[column]
    bins = np.geomspace(values.min(), values.max() * 1.01, 28)
    ax.hist(values, bins=bins, color=color, edgecolor='white')
    ax.set_xscale('log')
    ax.axvline(values.median(), color='#dc2626', linestyle='--', label=f'Median = {values.median():g}')
    ax.set_xlabel(f'{title} (trục log)')
    ax.set_ylabel('Số businesses')
    ax.legend()
for split in SPLIT_ORDER:
    part = stats_df.loc[stats_df['split'].eq(split)]
    axes[1, 0].scatter(part['reviews'], part['photos'], s=15, alpha=.5, label=split, color=COLORS[split])
axes[1, 0].set(xscale='log', yscale='log', xlabel='Review / business (log)', ylabel='Ảnh / business (log)')
axes[1, 0].legend()
axes[1, 0].set_title('Review và ảnh của từng business')
axes[1, 1].boxplot([stats_df.loc[stats_df['split'].eq(s), 'reviews'] for s in SPLIT_ORDER], tick_labels=SPLIT_ORDER)
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Review / business theo split')
axes[1, 1].set_ylabel('Số review (log)')
save_figure(fig, 'review_photo_distributions.png')

review_groups = pd.cut(stats_df['reviews'], [0, 30, 100, 300, 1000, np.inf],
                       labels=['≤30', '31–100', '101–300', '301–1.000', '>1.000'])
photo_groups = pd.cut(stats_df['photos'], [0, 5, 10, 20, 50, np.inf],
                      labels=['≤5', '6–10', '11–20', '21–50', '>50'])
for label, groups in [('Nhóm số review', review_groups), ('Nhóm số ảnh', photo_groups)]:
    counts = groups.value_counts(sort=False)
    print(label)
    display(pd.DataFrame({'businesses': counts, 'percent_businesses': (counts / len(stats_df) * 100).round(2)}))
""")
add('markdown', """
## 7. Rating, năm review, loại ảnh và địa lý

Rating và năm được tính trên tất cả review nguồn trong package. Năm đầu/cuối có thể chưa đủ cả năm; xem first/last review date để hiểu phạm vi thời gian. Loại ảnh lấy từ metadata gốc Yelp. Bảng state/city mô tả cohort hiện tại.
""")
add('code', """
rating_df = pd.DataFrame({'stars': range(1, 6), 'reviews': [rating_counts[n] for n in range(1, 6)]})
rating_df['percent_reviews'] = (rating_df['reviews'] / rating_df['reviews'].sum() * 100).round(2)
year_df = pd.DataFrame(years_by_split).fillna(0).astype(int).sort_index()
year_df.index.name = 'year'
label_df = photo_df['label'].value_counts().rename_axis('label').rename('photos').to_frame()
label_df['percent_photos'] = (label_df['photos'] / len(photo_df) * 100).round(2)
display(rating_df)
display(year_df)
display(label_df)
state_df = stats_df.groupby('state').agg(businesses=('business_id', 'size'), reviews=('reviews', 'sum'), photos=('photos', 'sum')).sort_values('businesses', ascending=False)
city_df = stats_df.groupby(['state', 'city']).agg(businesses=('business_id', 'size'), reviews=('reviews', 'sum'), photos=('photos', 'sum')).sort_values('businesses', ascending=False)
print('Theo state'); display(state_df)
print('Top 15 city'); display(city_df.head(15))
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bars = axes[0].bar(rating_df['stars'], rating_df['reviews'], color='#f59e0b')
axes[0].bar_label(bars, fontsize=8)
axes[0].set(xlabel='Stars', ylabel='Số review', title='Rating của toàn bộ review nguồn')
axes[0].set_ylim(0, rating_df['reviews'].max() * 1.15)
for split in SPLIT_ORDER:
    axes[1].plot(year_df.index, year_df[split], label=split, color=COLORS[split])
axes[1].set(xlabel='Năm', ylabel='Số review', title='Review theo năm và split')
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1].legend()
bars = axes[2].bar(label_df.index, label_df['photos'], color='#16a34a')
axes[2].bar_label(bars, fontsize=8)
axes[2].set(xlabel='Nhãn gốc Yelp', ylabel='Số ảnh', title='Photo labels')
axes[2].set_ylim(0, label_df['photos'].max() * 1.15)
save_figure(fig, 'rating_time_photo_labels.png')
""")
add('markdown', """
## 8. Xem chi tiết một business

Nhập `BUSINESS_ID_TO_VIEW` ở phần 1, chạy lại cell cấu hình rồi cell dưới. Cell đọc review của split tương ứng và chỉ giữ text của business đang xem. Review preview sắp xếp theo ngày; đây là xem dữ liệu, chưa phải thuật toán chọn review Stage C. Các ảnh preview chỉ giới hạn bởi `PHOTO_PREVIEW_N`; dataset vẫn giữ toàn bộ ảnh.
""")
add('code', """
chosen_bid = BUSINESS_ID_TO_VIEW
if chosen_bid is None:
    chosen_bid = stats_df.loc[stats_df['split'].eq('train')].sort_values(['reviews', 'business_id'], ascending=[False, True]).iloc[0]['business_id']
if chosen_bid not in membership:
    raise ValueError('Business ID không thuộc dataset; lấy ID từ các bảng phần 5.')
chosen = stats_df.loc[stats_df['business_id'].eq(chosen_bid)].iloc[0]
display(chosen.to_frame(name='value'))
print('Business ID đang xem:', chosen_bid)
print('Categories:', original_businesses[chosen_bid].get('categories'))
chosen_reviews = [row for row in scan_jsonl(DATASET_DIR / 'reviews' / f'{membership[chosen_bid]}.jsonl')
                  if row['business_id'] == chosen_bid]
assert len(chosen_reviews) == chosen['reviews']
detail_reviews = pd.DataFrame(chosen_reviews)
detail_reviews['date'] = pd.to_datetime(detail_reviews['date'])
detail_reviews = detail_reviews.sort_values(['date', 'review_id'])
print(f'{len(detail_reviews):,} review nguồn — preview {min(REVIEW_PREVIEW_N, len(detail_reviews))} review gần nhất:')
with pd.option_context('display.max_colwidth', 700):
    display(detail_reviews[['review_id', 'stars', 'date', 'text']].tail(REVIEW_PREVIEW_N))
print('Rating của business này:')
display(detail_reviews['stars'].value_counts().sort_index().rename('reviews').to_frame())
chosen_photos = photo_df.loc[photo_df['business_id'].eq(chosen_bid)].sort_values('photo_id')
print(f'{len(chosen_photos):,} ảnh gốc — bảng metadata:')
display(chosen_photos[['photo_id', 'label', 'relative_path', 'bytes']])
preview_photos = chosen_photos.head(PHOTO_PREVIEW_N)
columns = min(3, len(preview_photos))
rows = int(np.ceil(len(preview_photos) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False)
preview_results = []
for ax in axes.ravel():
    ax.axis('off')
for ax, (_, row) in zip(axes.ravel(), preview_photos.iterrows()):
    path = (DATASET_DIR / row['relative_path']).resolve()
    assert path.is_relative_to(DATASET_DIR)
    result = {'photo_id': row['photo_id'], 'readable_in_preview': False, 'width': None, 'height': None, 'error': ''}
    try:
        with Image.open(path) as image:
            image.load()
            result.update(readable_in_preview=True, width=image.width, height=image.height)
            thumbnail = image.convert('RGB')
            thumbnail.thumbnail((800, 600))
            ax.imshow(thumbnail)
        ax.set_title(f'{row["label"]} | {row["photo_id"]}', fontsize=8)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        ax.text(.5, .5, result['error'], ha='center', va='center', wrap=True, transform=ax.transAxes)
    preview_results.append(result)
save_figure(fig, 'business_photo_preview.png')
display(pd.DataFrame(preview_results))
print('Chỉ các ảnh preview được decode; chưa kiểm tra readability của toàn bộ 30.027 ảnh.')
""")
add('markdown', """
## 9. Báo cáo đã lưu và phạm vi kiểm tra

Bảng tổng hợp và các biểu đồ được lưu ngoài dataset. Notebook kiểm tra toàn bộ review file hashes, counts, ID uniqueness/joins; photo metadata hashes, file presence/size và counts; small manifest hashes. Không đọc/hash lại toàn bộ bytes ảnh, không full image validation, không chọn English reviews/30 reviews/token budget. Báo cáo Stage B độc lập vẫn là bằng chứng full-package/source comparison trước đó.
""")
add('code', """
stats_df.to_csv(REPORT_DIR / 'business_statistics.csv', index=False)
split_summary.to_csv(REPORT_DIR / 'split_statistics.csv')
extrema_df.to_csv(REPORT_DIR / 'extrema.csv', index=False)
for name, frame in rankings.items():
    frame.to_csv(REPORT_DIR / f'top_{name}.csv', index=False)
summary_report = {
    'dataset': manifest['dataset_version'], 'scope': 'FULL Stage B cohort; all review and photo metadata records',
    'businesses': len(stats_df), 'reviews': int(stats_df['reviews'].sum()), 'photos': len(photo_df),
    'actual_photo_files': len(photo_files), 'source_counts_match_manifests': True,
    'review_quality_flags': dict(quality),
    'review_files_hash_verified': SPLIT_ORDER, 'photo_metadata_files_hash_verified': SPLIT_ORDER,
    'review_count_distribution': {k: float(v) for k, v in stats_df['reviews'].describe(percentiles=[.5, .9, .95, .99]).items()},
    'photo_count_distribution': {k: float(v) for k, v in stats_df['photos'].describe(percentiles=[.5, .9, .95, .99]).items()},
    'extrema': extrema_df.to_dict(orient='records'),
    'preview_business_id': chosen_bid, 'preview_images': preview_results,
    'full_image_integrity_checked': False, 'full_image_byte_hashes_rechecked': False,
    'english_filter_or_review_selection_applied': False,
}
(REPORT_DIR / 'overview_summary.json').write_text(json.dumps(summary_report, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
print('Các thống kê và biểu đồ đã lưu tại:', REPORT_DIR)
acceptance_path = PROJECT_ROOT / 'outputs/cohort/validation_report.json'
if acceptance_path.is_file():
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    print('Báo cáo Stage B full-package/source validation đã lưu:', acceptance['validation_passed'])
    print('Đây là kết quả validation riêng trước đó, không phải full-image hash scan vừa chạy trong notebook.')
""")

notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.7', 'file_extension': '.py',
                      'mimetype': 'text/x-python', 'pygments_lexer': 'ipython3'},
}, 'nbformat': 4, 'nbformat_minor': 5}
destination = ROOT / 'notebooks/02_yelp_2300_dataset_overview.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Created {destination} ({len(cells)} cells)')
