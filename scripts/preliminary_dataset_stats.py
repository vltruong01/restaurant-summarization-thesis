import os
import math
import random
from collections import Counter, defaultdict

import orjson
import numpy as np
import pandas as pd

from tqdm import tqdm
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\thesis_proposal\yelp_dataset\data"

BUSINESS_FILE = os.path.join(
    BASE,
    "yelp_academic_dataset_business.json"
)

REVIEW_FILE = os.path.join(
    BASE,
    "yelp_academic_dataset_review.json"
)

PHOTO_META = os.path.join(
    BASE,
    "yelp_photos",
    "photos.json"
)

PHOTO_DIR = os.path.join(
    BASE,
    "yelp_photos",
    "photos"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "processed",
    "preliminary_stats"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMPORTANT:
#
# FIRST RUN:
# use 1 review + 1 photo.
#
# This lets us inspect the actual distributions BEFORE
# deciding the final eligibility threshold.
#
# SECOND RUN:
# you can change these to 10 reviews + 3 photos if the
# distributions support the rule used in your pilot.
# ============================================================

MIN_REVIEWS = 1
MIN_PHOTOS = 1

MAX_REVIEWS_PER_BUSINESS = 30

TOKEN_LIMIT = 5000

SEED = 42

random.seed(SEED)
DetectorFactory.seed = SEED


# ============================================================
# CATEGORY DEFINITIONS
# ============================================================

# IMPORTANT:
# Do NOT use "Hotels & Travel" alone.
# It also contains taxis, airports, transportation, etc.

HOTEL_KEYS = {
    "Hotels",
    "Resorts",
    "Bed & Breakfast",
    "Hostels",
    "Vacation Rentals",
    "Guest Houses",
}

RESTAURANT_KEYS = {
    "Restaurants"
}


# Thresholds we want to inspect after the first run.
# This lets us objectively decide eligibility.

REVIEW_THRESHOLDS = [1, 5, 10, 15, 20, 30]

PHOTO_THRESHOLDS = [1, 3, 5, 10, 15, 20]


# ============================================================
# JSONL READER
# ============================================================

def iter_jsonl(path, desc):
    """
    Yelp JSON files are JSONL:
    one JSON object per line.

    This reader avoids loading the entire dataset into RAM.
    """

    file_size = os.path.getsize(path)

    with open(path, "rb") as f:

        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=desc
        ) as pbar:

            for line in f:

                pbar.update(len(line))

                if not line.strip():
                    continue

                yield orjson.loads(line)


# ============================================================
# BUSINESS CATEGORY
# ============================================================

def classify_business(categories):

    if not categories:
        return None

    cats = {
        x.strip()
        for x in categories.split(",")
        if x.strip()
    }

    # Hotel checked first.
    # A hotel can sometimes also contain Restaurant category.

    if cats.intersection(HOTEL_KEYS):
        return "Hotels/Accommodation"

    if cats.intersection(RESTAURANT_KEYS):
        return "Restaurants"

    return None


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language_safe(text):

    if text is None:
        return "unknown"

    text = text.strip()

    if len(text) < 20:
        return "unknown"

    try:
        # Enough text for reliable language identification.
        # No need to pass an entire huge review.
        return detect(text[:2000])

    except Exception:
        return "unknown"


# ============================================================
# DISTRIBUTION HELPER
# ============================================================

def describe_distribution(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "Std": np.nan,
            "Min": np.nan,
            "Q1": np.nan,
            "Median": np.nan,
            "Q3": np.nan,
            "P90": np.nan,
            "P95": np.nan,
            "Max": np.nan,
        }

    return {
        "N": len(values),
        "Mean": np.mean(values),
        "Std": np.std(values),
        "Min": np.min(values),
        "Q1": np.percentile(values, 25),
        "Median": np.median(values),
        "Q3": np.percentile(values, 75),
        "P90": np.percentile(values, 90),
        "P95": np.percentile(values, 95),
        "Max": np.max(values),
    }


# ============================================================
# PLANNED DATASET SPLIT
# ============================================================

def planned_split(n):

    if n < 3:
        return n, 0, 0

    # Proposal currently expects roughly 100 held-out businesses.
    # But do not take 100 if the eligible subset is tiny.

    test_n = min(
        100,
        max(1, round(n * 0.10))
    )

    remaining = n - test_n

    val_n = max(
        1,
        round(remaining * 0.10)
    )

    if val_n >= remaining:
        val_n = max(0, remaining - 1)

    train_n = remaining - val_n

    return train_n, val_n, test_n


# ============================================================
# STEP 1
# LOAD HOSPITALITY BUSINESSES
# ============================================================

print("\n====================================================")
print("STEP 1 - LOAD HOSPITALITY BUSINESSES")
print("====================================================\n")

business_domain = {}
business_name = {}

for obj in iter_jsonl(
    BUSINESS_FILE,
    "Reading businesses"
):

    domain = classify_business(
        obj.get("categories")
    )

    if domain is None:
        continue

    bid = obj["business_id"]

    business_domain[bid] = domain

    business_name[bid] = obj.get(
        "name",
        ""
    )


restaurant_businesses = sum(
    domain == "Restaurants"
    for domain in business_domain.values()
)

hotel_businesses = sum(
    domain == "Hotels/Accommodation"
    for domain in business_domain.values()
)


print("\nCandidate businesses by category")
print("----------------------------------")
print(
    f"Restaurants:          "
    f"{restaurant_businesses:,}"
)

print(
    f"Hotels/Accommodation: "
    f"{hotel_businesses:,}"
)


# ============================================================
# STEP 2
# CHECK DOWNLOADED PHOTOS
# ============================================================

print("\n====================================================")
print("STEP 2 - INDEX DOWNLOADED PHOTOS")
print("====================================================\n")


available_photo_ids = set()

for entry in tqdm(
    os.scandir(PHOTO_DIR),
    desc="Indexing JPG files"
):

    if not entry.is_file():
        continue

    filename = entry.name.lower()

    if not filename.endswith(
        (".jpg", ".jpeg")
    ):
        continue

    photo_id = os.path.splitext(
        entry.name
    )[0]

    available_photo_ids.add(photo_id)


print(
    "\nDownloaded image files:",
    f"{len(available_photo_ids):,}"
)


# ============================================================
# STEP 3
# COUNT PHOTOS PER BUSINESS
# ============================================================

print("\n====================================================")
print("STEP 3 - COUNT PHOTOS")
print("====================================================\n")


photo_count = Counter()


for obj in iter_jsonl(
    PHOTO_META,
    "Reading photos.json"
):

    bid = obj.get("business_id")

    if bid not in business_domain:
        continue

    photo_id = obj.get("photo_id")

    # Count only images that actually exist locally.
    if photo_id not in available_photo_ids:
        continue

    photo_count[bid] += 1


businesses_with_photo = {
    bid
    for bid in business_domain
    if photo_count[bid] >= 1
}


print(
    "\nHospitality businesses "
    "with >= 1 downloaded photo:",
    f"{len(businesses_with_photo):,}"
)


# ============================================================
# STEP 4
# COUNT REVIEWS
# ============================================================

print("\n====================================================")
print("STEP 4 - COUNT REVIEWS")
print("====================================================\n")


review_count = Counter()

star_count = defaultdict(Counter)


for obj in iter_jsonl(
    REVIEW_FILE,
    "Reading reviews - pass 1"
):

    bid = obj.get("business_id")

    if bid not in businesses_with_photo:
        continue

    review_count[bid] += 1

    try:

        stars = int(
            round(
                float(
                    obj.get("stars", 0)
                )
            )
        )

        stars = max(
            1,
            min(5, stars)
        )

        star_count[bid][stars] += 1

    except Exception:
        pass


candidate_ids = {
    bid
    for bid in businesses_with_photo
    if review_count[bid] >= 1
}


print(
    "\nBusinesses with >=1 review "
    "AND >=1 photo:",
    f"{len(candidate_ids):,}"
)


# ============================================================
# STEP 5
# THRESHOLD ANALYSIS
# ============================================================

print("\n====================================================")
print("STEP 5 - ELIGIBILITY THRESHOLD ANALYSIS")
print("====================================================\n")


threshold_rows = []


for domain in [
    "Restaurants",
    "Hotels/Accommodation"
]:

    domain_ids = [
        bid
        for bid in candidate_ids
        if business_domain[bid] == domain
    ]

    for min_reviews in REVIEW_THRESHOLDS:

        for min_photos in PHOTO_THRESHOLDS:

            n = sum(

                review_count[bid] >= min_reviews
                and
                photo_count[bid] >= min_photos

                for bid in domain_ids
            )

            threshold_rows.append({

                "Domain": domain,

                "Minimum reviews":
                    min_reviews,

                "Minimum photos":
                    min_photos,

                "Eligible businesses":
                    n,

            })


threshold_df = pd.DataFrame(
    threshold_rows
)


# ============================================================
# CURRENT ELIGIBILITY
# ============================================================

eligible_ids = {

    bid
    for bid in candidate_ids

    if review_count[bid] >= MIN_REVIEWS
    and photo_count[bid] >= MIN_PHOTOS

}


print(
    f"\nCurrent rule: "
    f">= {MIN_REVIEWS} reviews, "
    f">= {MIN_PHOTOS} photos"
)

print(
    "Eligible businesses:",
    f"{len(eligible_ids):,}"
)


# ============================================================
# STEP 6
# SAMPLE UP TO 30 REVIEWS PER BUSINESS
# ============================================================

print("\n====================================================")
print("STEP 6 - SELECT REVIEWS FOR TOKEN/LANGUAGE ANALYSIS")
print("====================================================\n")


# For preliminary profiling:
#
# keep a deterministic reservoir sample within each star rating.
#
# This preserves rating diversity without keeping every review
# in RAM.
#
# Final model preprocessing can later add the exact
# date-stratification rule described in Section 3.2.


samples = defaultdict(
    lambda: defaultdict(list)
)

seen = defaultdict(
    lambda: defaultdict(int)
)


# Allocate approximately 30 / 5 = 6 reservoir slots
# initially per star.
#
# Afterwards we combine them and keep at most 30.
#
# This is preliminary profiling, not final training selection.

PER_STAR_LIMIT = math.ceil(
    MAX_REVIEWS_PER_BUSINESS / 5
)


rng = random.Random(SEED)


for obj in iter_jsonl(
    REVIEW_FILE,
    "Reading reviews - pass 2"
):

    bid = obj.get("business_id")

    if bid not in eligible_ids:
        continue

    try:

        stars = int(
            round(
                float(
                    obj.get("stars", 0)
                )
            )
        )

        stars = max(
            1,
            min(5, stars)
        )

    except Exception:

        stars = 0


    seen[bid][stars] += 1

    bucket = samples[bid][stars]


    record = {

        "review_id":
            obj.get("review_id"),

        "stars":
            stars,

        "date":
            obj.get("date", ""),

        "text":
            obj.get("text", "") or ""

    }


    if len(bucket) < PER_STAR_LIMIT:

        bucket.append(record)

    else:

        j = rng.randrange(
            seen[bid][stars]
        )

        if j < PER_STAR_LIMIT:

            bucket[j] = record


# ============================================================
# STEP 7
# LED TOKENIZER + LANGUAGE DISTRIBUTION
# ============================================================

print("\n====================================================")
print("STEP 7 - LED TOKENIZATION")
print("====================================================\n")


print(
    "Loading tokenizer: "
    "allenai/led-base-16384"
)


tokenizer = AutoTokenizer.from_pretrained(
    "allenai/led-base-16384"
)


business_rows = []

language_counter = defaultdict(
    Counter
)


for bid in tqdm(
    sorted(eligible_ids),
    desc="Analyzing eligible businesses"
):

    selected = []


    for star_bucket in samples[bid].values():

        selected.extend(
            star_bucket
        )


    # Ensure at most 30.
    #
    # Sort chronologically afterwards,
    # consistent with Section 3.2.

    if len(selected) > MAX_REVIEWS_PER_BUSINESS:

        rng2 = random.Random(
            f"{SEED}_{bid}"
        )

        selected = rng2.sample(
            selected,
            MAX_REVIEWS_PER_BUSINESS
        )


    selected.sort(
        key=lambda x: x["date"]
    )


    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    domain = business_domain[bid]


    for review in selected:

        lang = detect_language_safe(
            review["text"]
        )

        language_counter[domain][lang] += 1


    # --------------------------------------------------------
    # Token length
    # --------------------------------------------------------

    review_text = "\n".join(
        review["text"]
        for review in selected
    )


    token_ids = tokenizer.encode(

        review_text,

        add_special_tokens=False,

        truncation=False

    )


    token_count = len(
        token_ids
    )


    business_rows.append({

        "business_id":
            bid,

        "business_name":
            business_name.get(
                bid,
                ""
            ),

        "domain":
            domain,

        "total_reviews":
            review_count[bid],

        "available_photos":
            photo_count[bid],

        "selected_reviews":
            len(selected),

        "selected_review_tokens":
            token_count,

        "over_5000_tokens":
            token_count > TOKEN_LIMIT,

    })


business_df = pd.DataFrame(
    business_rows
)


# ============================================================
# DISTRIBUTION TABLE
# ============================================================

distribution_rows = []


for domain in [
    "Restaurants",
    "Hotels/Accommodation"
]:

    d = business_df[
        business_df["domain"] == domain
    ]

    for variable, column in [

        (
            "Reviews per business",
            "total_reviews"
        ),

        (
            "Photos per business",
            "available_photos"
        ),

        (
            "Selected review tokens per business",
            "selected_review_tokens"
        ),

    ]:

        stats = describe_distribution(
            d[column].values
        )

        distribution_rows.append({

            "Domain":
                domain,

            "Statistic":
                variable,

            **stats

        })


distribution_df = pd.DataFrame(
    distribution_rows
)


# ============================================================
# LANGUAGE TABLE
# ============================================================

language_rows = []


for domain, counts in language_counter.items():

    total = sum(
        counts.values()
    )

    for language, n in counts.most_common():

        language_rows.append({

            "Domain":
                domain,

            "Language":
                language,

            "Reviews":
                n,

            "Percentage":
                100 * n / total
                if total
                else 0,

        })


language_df = pd.DataFrame(
    language_rows
)


# ============================================================
# BUILD MAIN TABLE 3.X
# ============================================================

table_rows = []


for domain in [
    "Restaurants",
    "Hotels/Accommodation"
]:

    d = business_df[
        business_df["domain"] == domain
    ]


    n_businesses = len(d)


    train_n, val_n, test_n = planned_split(
        n_businesses
    )


    total_reviews = int(
        d["total_reviews"].sum()
    )


    total_photos = int(
        d["available_photos"].sum()
    )


    mean_reviews = (
        d["total_reviews"].mean()
        if n_businesses
        else np.nan
    )


    median_reviews = (
        d["total_reviews"].median()
        if n_businesses
        else np.nan
    )


    mean_photos = (
        d["available_photos"].mean()
        if n_businesses
        else np.nan
    )


    median_photos = (
        d["available_photos"].median()
        if n_businesses
        else np.nan
    )


    mean_tokens = (
        d["selected_review_tokens"].mean()
        if n_businesses
        else np.nan
    )


    median_tokens = (
        d["selected_review_tokens"].median()
        if n_businesses
        else np.nan
    )


    p90_tokens = (
        d["selected_review_tokens"].quantile(0.90)
        if n_businesses
        else np.nan
    )


    pct_over_5000 = (

        100
        * d["over_5000_tokens"].mean()

        if n_businesses
        else np.nan
    )


    english_rows = language_df[

        (language_df["Domain"] == domain)
        &
        (language_df["Language"] == "en")

    ]


    if len(english_rows):

        english_pct = float(
            english_rows[
                "Percentage"
            ].iloc[0]
        )

    else:

        english_pct = 0.0


    table_rows.append({

        "Domain":
            domain,

        "Eligible businesses":
            n_businesses,

        "Total reviews":
            total_reviews,

        "Total photos":
            total_photos,

        "Mean reviews/business":
            mean_reviews,

        "Median reviews/business":
            median_reviews,

        "Mean photos/business":
            mean_photos,

        "Median photos/business":
            median_photos,

        "Mean selected-review tokens":
            mean_tokens,

        "Median selected-review tokens":
            median_tokens,

        "P90 selected-review tokens":
            p90_tokens,

        "% > 5,000 review tokens":
            pct_over_5000,

        "English selected reviews (%)":
            english_pct,

        "Expected train":
            train_n,

        "Expected validation":
            val_n,

        "Expected test":
            test_n,

    })


table_3x_df = pd.DataFrame(
    table_rows
)


# ============================================================
# SAVE OUTPUT
# ============================================================

print("\n====================================================")
print("SAVING RESULTS")
print("====================================================\n")


excel_file = os.path.join(

    OUTPUT_DIR,

    "preliminary_yelp_statistics.xlsx"

)


with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:

    table_3x_df.to_excel(
        writer,
        sheet_name="Table_3X",
        index=False
    )

    threshold_df.to_excel(
        writer,
        sheet_name="Threshold_Check",
        index=False
    )

    distribution_df.to_excel(
        writer,
        sheet_name="Distributions",
        index=False
    )

    language_df.to_excel(
        writer,
        sheet_name="Languages",
        index=False
    )

    business_df.to_excel(
        writer,
        sheet_name="Business_Detail",
        index=False
    )


table_3x_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "table_3x.csv"
    ),

    index=False

)


threshold_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "eligibility_thresholds.csv"
    ),

    index=False

)


print("\n================ TABLE 3.X =================\n")

print(
    table_3x_df.to_string(
        index=False
    )
)


print("\n============================================")
print("DONE")
print("============================================")

print(
    "\nExcel output:\n",
    excel_file
)

print(
    "\nCheck sheet 'Threshold_Check' "
    "before deciding the final "
    "MIN_REVIEWS / MIN_PHOTOS."
)