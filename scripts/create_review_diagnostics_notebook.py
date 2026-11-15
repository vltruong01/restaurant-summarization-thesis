"""Generate the Stage C1 English corpus/rating/time/LED diagnostics notebook."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    cell = {'cell_type': kind, 'id': f'stage-c-{len(cells):03d}', 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        compile(source, f'stage-c-cell-{len(cells)}', 'exec')
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', """
# Stage C1 — Review tiếng Anh và token LED cho 2.300 businesses

Notebook đọc kết quả **nhận diện ngôn ngữ đầy đủ trên tất cả review của cohort**, khác với English business gate trước đó chỉ chứng minh ≥10 English reviews. Hiển thị counts chính xác, review bị loại, phân bố rating/thời gian, token lengths và những vấn đề cần chốt trước chọn ≤30 review.

Cohort/split giữ nguyên. Bộ `yelp_2300_english_v1` là **toàn bộ pool review English**, chưa phải input ≤30 review hay dataset train LED. Các phần tổng quan test chỉ kiểm tra/mô tả dữ liệu; các bảng phục vụ quyết định budget dùng **train/validation**, không dùng test để chốt ngưỡng.

Mở bằng Python 3 (ipykernel), **Run All**. Pipeline C1 có cache: lần đầu có thể mất 10–20 phút trên CPU; khi report đã hoàn tất, chỉ đọc/kiểm tra artifacts. Không cần GPU, torch hoặc tải weights. Toàn bộ text được giữ nguyên; language detector probe mới được strip/cắt 2.000 ký tự.
""")
add('markdown', """
## 1. Cấu hình và kết quả pipeline

`BUSINESS_ID_TO_VIEW` và `EXAMPLE_ROWS` chỉ dùng xem dữ liệu, không phải tham số nghiên cứu. Các quy tắc chọn review/budget còn TBD được hiển thị rõ ở phần 8.
""")
add('code', """
from pathlib import Path
import sys, json, hashlib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import orjson

BUSINESS_ID_TO_VIEW = None
EXAMPLE_ROWS = 8
PROJECT_ROOT = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
                     if (p / 'configs/review_diagnostics.yaml').is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Mở notebook bên trong project yelp_dataset.')
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from review_diagnostics import run_diagnostics
cfg = yaml.safe_load((PROJECT_ROOT / 'configs/review_diagnostics.yaml').read_text(encoding='utf-8'))
SOURCE_DIR = PROJECT_ROOT / cfg['dataset_directory']
ENGLISH_DIR = PROJECT_ROOT / cfg['english_pool_directory']
REPORT_DIR = PROJECT_ROOT / cfg['output_directory']
FIGURE_DIR = REPORT_DIR / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SPLITS = ['train', 'validation', 'test']
COLORS = {'train': '#2563eb', 'validation': '#f59e0b', 'test': '#16a34a'}
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 150)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
completion = run_diagnostics(PROJECT_ROOT)
display(pd.DataFrame([{key: completion[key] for key in [
    'raw_reviews', 'english_reviews', 'non_english_reviews', 'unknown_reviews',
    'businesses', 'minimum_english_reviews_per_business', 'all_review_language_counts_exact',
    'selection_ready', 'overall_stage_c_complete',
]}]))
print('English review pool:', ENGLISH_DIR)
print('Đây là C1; chưa chọn tối đa 30, chưa khóa source token budget.')

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)
""")
add('markdown', """
## 2. Metadata và đối chiếu counts/hashes

Chỉ nạp **metadata**, không nạp toàn bộ review text. Số `led_tokens_without_special` tính trên nguyên văn review, không truncation/padding. `led_tokens_with_special` thêm BOS/EOS cho review đơn; không cộng BOS/EOS lặp lại khi xây một input business.
""")
add('code', """
review_meta = pd.read_csv(REPORT_DIR / 'review_language_token_metadata.csv', keep_default_na=False)
business_stats = pd.read_csv(REPORT_DIR / 'business_review_statistics.csv', keep_default_na=False)
source_split = pd.read_csv(SOURCE_DIR / 'business_split.csv')
english_split = pd.read_csv(ENGLISH_DIR / 'business_split.csv')
assert source_split.equals(english_split)
assert len(review_meta) == completion['raw_reviews'] == 601493 and review_meta['review_id'].is_unique
assert len(business_stats) == 2300 and business_stats['business_id'].is_unique
assert business_stats['english_reviews'].ge(10).all()
assert review_meta['business_id'].map(source_split.set_index('business_id')['split']).eq(review_meta['split']).all()
assert review_meta['led_tokens_with_special'].eq(review_meta['led_tokens_without_special'] + completion['special_tokens_per_single_review']).all()
for relative, expected in completion['english_pool_artifact_sha256'].items():
    with (ENGLISH_DIR / relative).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, relative
english_meta = review_meta.loc[review_meta['language'].eq('en')].copy()
excluded_meta = review_meta.loc[~review_meta['language'].eq('en')].copy()
assert len(english_meta) == completion['english_reviews']
assert len(english_meta) + len(excluded_meta) == len(review_meta)
assert english_meta.groupby('business_id').size().sort_index().equals(
    business_stats.set_index('business_id')['english_reviews'].sort_index().rename(None))
display(review_meta.head(10))
display(business_stats.head(10))
""")
add('markdown', """
## 3. Ngôn ngữ — counts chính xác toàn cohort

Counts là labels của detector đã chạy đầy đủ, chưa phải ground truth con người. `unknown` gồm text quá ngắn/không có đặc trưng ngôn ngữ; không tự gộp unknown thành tiếng Anh. Khác với gate cũ, các counts ở đây đều exact cho cohort 2.300.
""")
add('code', """
language_table = review_meta.groupby(['split', 'language']).size().unstack(fill_value=0).reindex(SPLITS)
split_table = pd.DataFrame(completion['split_summary']).T.reindex(SPLITS)
display(split_table)
display(language_table)
overall_language = review_meta['language'].value_counts().rename_axis('language').rename('reviews').to_frame()
overall_language['percent_reviews'] = (overall_language['reviews'] / len(review_meta) * 100).round(4)
display(overall_language)
display(business_stats[['raw_reviews', 'english_reviews', 'non_english_reviews', 'unknown_reviews']].describe(percentiles=[.5, .9, .95, .99]).T.round(2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
kept = split_table['english_reviews'] / split_table['raw_reviews'] * 100
bars = axes[0].bar(kept.index, kept, color=[COLORS[s] for s in kept.index])
axes[0].bar_label(bars, fmt='%.3f%%', padding=3)
axes[0].set(ylim=(0, 108), ylabel='% review được gán en', title='English giữ lại theo split')
excluded_counts = excluded_meta['language'].value_counts().head(12)
axes[1].barh(excluded_counts.index[::-1], excluded_counts.values[::-1], color='#dc2626')
axes[1].set(xlabel='Số review', title='Labels không giữ lại (top 12)')
save_figure(fig, 'exact_language_counts.png')
""")
add('markdown', """
## 4. Rating và thay đổi khi giới hạn tiếng Anh

So sánh phân bố sao của raw pool và English pool giúp chốt rating population. Các bảng gợi ý phương pháp bên dưới dùng train/validation. Quyết định population chưa được notebook tự chốt.
""")
add('code', """
allowed_raw = review_meta.loc[review_meta['split'].isin(['train', 'validation'])]
allowed_en = english_meta.loc[english_meta['split'].isin(['train', 'validation'])]
rating_comparison = pd.DataFrame({
    'raw_reviews': allowed_raw['stars'].value_counts().reindex(range(1, 6), fill_value=0),
    'english_reviews': allowed_en['stars'].value_counts().reindex(range(1, 6), fill_value=0),
})
rating_comparison['raw_percent'] = rating_comparison['raw_reviews'] / len(allowed_raw) * 100
rating_comparison['english_percent'] = rating_comparison['english_reviews'] / len(allowed_en) * 100
rating_comparison['percentage_point_change'] = rating_comparison['english_percent'] - rating_comparison['raw_percent']
display(rating_comparison.round(4))
raw_stars = pd.crosstab(allowed_raw['business_id'], allowed_raw['stars']).reindex(columns=range(1, 6), fill_value=0)
en_stars = pd.crosstab(allowed_en['business_id'], allowed_en['stars']).reindex(index=raw_stars.index, columns=range(1, 6), fill_value=0)
raw_proportions = raw_stars.div(raw_stars.sum(axis=1), axis=0)
en_proportions = en_stars.div(en_stars.sum(axis=1), axis=0)
rating_shift = ((raw_proportions - en_proportions).abs().sum(axis=1) / 2).rename('rating_total_variation')
display(rating_shift.describe(percentiles=[.5, .9, .95, .99]).to_frame())
display(business_stats.merge(rating_shift, left_on='business_id', right_index=True).sort_values('rating_total_variation', ascending=False).head(10))
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(5)
ax.bar(x - .18, rating_comparison['raw_percent'], width=.36, label='Raw train/validation', color='#64748b')
ax.bar(x + .18, rating_comparison['english_percent'], width=.36, label='English train/validation', color='#2563eb')
ax.set(xticks=x, xticklabels=range(1, 6), xlabel='Stars', ylabel='% reviews', title='Rating population — chỉ train/validation')
ax.legend()
save_figure(fig, 'rating_population_comparison.png')
""")
add('markdown', """
## 5. Thời gian và số review English mỗi business

Không lựa chọn strata ở phần này. Ngày đầu/cuối và year histogram mô tả toàn bộ English evidence; first/last years có thể chỉ có dữ liệu một phần năm. English corpus giữ nguyên source order; review inputs sau này phải được sắp xếp theo ngày sau selection.
""")
add('code', """
english_meta['year'] = english_meta['date'].str[:4].astype(int)
year_counts = english_meta.groupby(['year', 'split']).size().unstack(fill_value=0).reindex(columns=SPLITS, fill_value=0)
display(year_counts)
business_stats['first_english_review_date'] = pd.to_datetime(business_stats['first_english_review_date'])
business_stats['last_english_review_date'] = pd.to_datetime(business_stats['last_english_review_date'])
business_stats['english_span_years'] = (business_stats['last_english_review_date'] - business_stats['first_english_review_date']).dt.days / 365.25
display(business_stats[['english_reviews', 'english_span_years']].describe(percentiles=[.5, .9, .95, .99]).T.round(2))
print('Businesses có <=30 English reviews:', int(business_stats['english_reviews'].le(30).sum()))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for split in SPLITS:
    axes[0].plot(year_counts.index, year_counts[split], label=split, color=COLORS[split])
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].set(xlabel='Năm', ylabel='English reviews', title='English reviews theo năm'); axes[0].legend()
v = business_stats['english_reviews']
axes[1].hist(v, bins=np.geomspace(v.min(), v.max()*1.01, 28), color='#16a34a', edgecolor='white')
axes[1].set(xscale='log', xlabel='English reviews/business (log)', ylabel='Businesses', title='Cần chọn từ pool lớn như thế nào?')
axes[1].axvline(30, color='#dc2626', linestyle='--', label='Tối đa 30 sau selection'); axes[1].legend()
save_figure(fig, 'english_time_and_business_counts.png')
""")
add('markdown', """
## 6. LED token lengths — nguyên văn từng review

Mọi review được encode đầy đủ bằng tokenizer LED đã pin. Base count không có special tokens. Histogram giúp nhìn phần đuôi dài; token lengths của review đơn **không phải exact length của input business có markers**. Các bảng xem budget ở đây chỉ dùng train/validation.
""")
add('code', """
token_summary = english_meta.groupby('split')['led_tokens_without_special'].describe(percentiles=[.5, .9, .95, .99]).reindex(SPLITS)
display(token_summary.round(2))
display(allowed_en['led_tokens_without_special'].describe(percentiles=[.5, .9, .95, .99]).to_frame())
print('Scientific input identity:', completion.get('identity_sha256'))
scientific = json.loads((REPORT_DIR / 'scientific_identity.json').read_text(encoding='utf-8'))
print('LED tokenizer revision:', scientific['identity']['tokenizer_revision'])
print('BOS/EOS thêm cho một review đơn:', completion['special_tokens_per_single_review'])
display(allowed_en.sort_values('led_tokens_without_special', ascending=False).head(15)[
    ['review_id', 'business_id', 'split', 'stars', 'date', 'led_tokens_without_special']])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
v = allowed_en['led_tokens_without_special']
axes[0].hist(v, bins=np.geomspace(max(1, v.min()), v.max()*1.01, 35), color='#2563eb', edgecolor='white')
axes[0].set(xscale='log', xlabel='Tokens/review (log)', ylabel='Review count', title='LED lengths — train/validation English')
axes[1].boxplot([allowed_en.loc[allowed_en['split'].eq(s), 'led_tokens_without_special'] for s in ['train', 'validation']], tick_labels=['train', 'validation'])
axes[1].set(yscale='log', ylabel='Tokens/review (log)', title='Lengths theo split được phép dùng chốt budget')
save_figure(fig, 'english_led_token_lengths.png')

# Các business <=30 review có thể vẫn dài hơn budget; đây là SUM review đơn,
# chưa phải marked input và chưa tự chọn/drop review.
small_pools = business_stats.loc[business_stats['split'].isin(['train', 'validation']) & business_stats['english_reviews'].le(30)]
display(small_pools[['business_id', 'name', 'split', 'english_reviews', 'english_token_sum_individual_reviews']].sort_values('english_token_sum_individual_reviews', ascending=False).head(15))
print('Ngay cả pool <=30 vẫn cần kiểm tra exact assembled token length trước khi freeze.')
""")
add('markdown', """
## 7. Xem các review không giữ lại và một business

Examples để chẩn đoán language errors: label tự động không thay cho manual audit. Chỉ nạp text của một số IDs/business; không đọc toàn corpus vào RAM. Các examples chẩn đoán mặc định lấy train/validation.
""")
add('code', """
excluded_examples = excluded_meta.loc[excluded_meta['split'].isin(['train', 'validation'])].sort_values(['language', 'review_id']).groupby('language', sort=True).head(2).head(24)
example_ids = set(excluded_examples['review_id'])
texts = {}
for split in ['train', 'validation']:
    wanted = set(excluded_examples.loc[excluded_examples['split'].eq(split), 'review_id'])
    if not wanted: continue
    with (SOURCE_DIR / 'reviews' / f'{split}.jsonl').open('rb') as stream:
        for line in stream:
            row = orjson.loads(line)
            if row['review_id'] in wanted:
                texts[row['review_id']] = row['text']
                if wanted <= texts.keys(): break
example_table = excluded_examples.copy()
example_table['text_preview'] = example_table['review_id'].map(texts).str.slice(0, 700)
display(example_table[['review_id', 'business_id', 'split', 'language', 'language_reason', 'text_preview']])
chosen_bid = BUSINESS_ID_TO_VIEW
if chosen_bid is None:
    chosen_bid = business_stats.loc[business_stats['split'].eq('train')].sort_values(['english_reviews', 'business_id'], ascending=[False, True]).iloc[0]['business_id']
if chosen_bid not in set(business_stats['business_id']):
    raise ValueError('Business ID không thuộc cohort.')
chosen = business_stats.loc[business_stats['business_id'].eq(chosen_bid)].iloc[0]
display(chosen.to_frame(name='value'))
chosen_meta = english_meta.loc[english_meta['business_id'].eq(chosen_bid)].sort_values(['date', 'review_id'])
display(chosen_meta[['review_id', 'stars', 'date', 'led_tokens_without_special']].head(EXAMPLE_ROWS))
display(chosen_meta[['review_id', 'stars', 'date', 'led_tokens_without_special']].tail(EXAMPLE_ROWS))
print('Đây là full English pool của business, chưa chọn 30 và chưa cắt theo budget.')
""")
add('markdown', """
## 8. Các quyết định C2 chưa tự áp dụng

Rating population, calendar/quantile strata, quota rounding, complete-review removal và token allocations phải được ghi rõ trước khi freeze. Không dùng test/outputs mô hình để chọn rules. Một optional preference question đã được gửi trong lúc C1 chạy; notebook không giả định người dùng đã chọn.

Sau khi chốt C2: chọn ≤30 đại diện rating/thời gian, chronological ordering, xây đúng text có `[REV_i]`, đếm exact LED input tokens với metadata/markers/caption reserve, lưu review IDs/text/order cố định dùng chung cả ba model. Không cắt giữa review.
""")
add('code', """
pending = yaml.safe_load((PROJECT_ROOT / 'configs/research_pending.yaml').read_text(encoding='utf-8'))
pending_rows = []
for section in ['language_policy', 'review_selection', 'token_budget']:
    for key, value in pending[section].items():
        if value is None:
            pending_rows.append({'section': section, 'parameter': key, 'status': 'TBD; chưa áp dụng'})
display(pd.DataFrame(pending_rows))
print('Selected <=30 manifest exists:', False)
print('Stage C complete:', completion['overall_stage_c_complete'])
acceptance_path = REPORT_DIR / 'validation_report.json'
if acceptance_path.is_file():
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    display(pd.DataFrame([{key: acceptance[key] for key in [
        'validation_passed', 'raw_reviews', 'english_reviews', 'non_english_reviews', 'unknown_reviews',
        'businesses', 'token_samples_reencoded', 'language_samples_reclassified',
    ]}]))
else:
    print('C1 export hoàn tất; independent validation report chưa có.')
print('CSV/reports/figures:', REPORT_DIR)
""")

nb = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.7', 'file_extension': '.py',
                      'mimetype': 'text/x-python', 'pygments_lexer': 'ipython3'},
}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'notebooks/03_english_review_selection_diagnostics.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(f'Created {path} ({len(cells)} cells)')
