import os
import math
import random
from collections import Counter, defaultdict

import orjson
import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory


# ============================================================
# 1. CONFIGURATION
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
    "language_stats_final"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FINAL THESIS SETTINGS
# ============================================================

MIN_REVIEWS = 10
MIN_PHOTOS = 5

MAX_REVIEWS_PER_BUSINESS = 30

SEED = 42

random.seed(SEED)

# Make langdetect deterministic
DetectorFactory.seed = SEED


# ============================================================
# IMPORTANT:
# SAME HOTEL / ACCOMMODATION DEFINITION AS PRELIMINARY ANALYSIS
# ============================================================

HOTEL_KEYS = {
    "Hotels",
    "Resorts",
    "Bed & Breakfast",
    "Hostels",
    "Vacation Rentals",
    "Guest Houses",
}


# ============================================================
# 2. JSONL READER
# ============================================================

def iter_jsonl(path, description):
    """
    Efficiently read Yelp JSONL files without
    loading the whole file into RAM.
    """

    total_size = os.path.getsize(path)

    with open(path, "rb") as f:

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=description
        ) as pbar:

            for line in f:

                pbar.update(len(line))

                if not line.strip():
                    continue

                try:
                    yield orjson.loads(line)

                except Exception:
                    continue


# ============================================================
# 3. LANGUAGE DETECTION
# ============================================================

def detect_language_safe(text):
    """
    Detect language safely.

    Reviews shorter than 20 characters are marked
    as unknown.

    Only first 2000 characters are needed.
    """

    if text is None:
        return "unknown"

    text = text.strip()

    if len(text) < 20:
        return "unknown"

    try:

        return detect(
            text[:2000]
        )

    except Exception:

        return "unknown"


# ============================================================
# 4. ALLOCATE STAR-RATING QUOTA
# ============================================================

def allocate_rating_quota(
    star_counter,
    max_reviews=30
):
    """
    Select up to MAX_REVIEWS_PER_BUSINESS while
    approximately preserving the original
    1-5 star distribution.
    """

    total_reviews = sum(
        star_counter.values()
    )

    if total_reviews == 0:
        return {}

    target = min(
        max_reviews,
        total_reviews
    )

    # Fractional expected quota
    raw_quota = {

        star:
        target * count / total_reviews

        for star, count
        in star_counter.items()

    }

    # Initial floor allocation
    quota = {

        star:
        min(
            count,
            int(
                math.floor(
                    raw_quota[star]
                )
            )
        )

        for star, count
        in star_counter.items()

    }

    remaining = (
        target
        - sum(
            quota.values()
        )
    )

    # Largest remainder method
    ranking = sorted(

        star_counter.keys(),

        key=lambda star:
        raw_quota[star]
        - math.floor(
            raw_quota[star]
        ),

        reverse=True
    )

    while remaining > 0:

        changed = False

        for star in ranking:

            if remaining == 0:
                break

            if (
                quota[star]
                < star_counter[star]
            ):

                quota[star] += 1

                remaining -= 1

                changed = True

        if not changed:
            break

    return quota


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

required_files = [
    BUSINESS_FILE,
    REVIEW_FILE,
    PHOTO_META,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nCannot find:\n{path}"
        )


if not os.path.isdir(PHOTO_DIR):

    raise FileNotFoundError(
        f"\nCannot find photo directory:\n{PHOTO_DIR}"
    )


# ============================================================
# STEP 1
# CLASSIFY BUSINESSES
#
# IMPORTANT:
# HOTEL/ACCOMMODATION IS CHECKED FIRST.
#
# A business having both:
# Restaurants + Hotels
#
# will NOT be included in the restaurant subset.
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1 - CLASSIFY RESTAURANT BUSINESSES")
print("=" * 70)


restaurant_ids = set()
hotel_ids = set()

restaurant_names = {}


for obj in iter_jsonl(
    BUSINESS_FILE,
    "Reading business dataset"
):

    categories = obj.get(
        "categories"
    )

    if not categories:
        continue


    category_set = {

        x.strip()

        for x in categories.split(",")

        if x.strip()

    }


    business_id = obj.get(
        "business_id"
    )

    if not business_id:
        continue


    # --------------------------------------------------------
    # HOTEL / ACCOMMODATION FIRST
    # --------------------------------------------------------

    if category_set.intersection(
        HOTEL_KEYS
    ):

        hotel_ids.add(
            business_id
        )

        continue


    # --------------------------------------------------------
    # RESTAURANTS
    # --------------------------------------------------------

    if "Restaurants" in category_set:

        restaurant_ids.add(
            business_id
        )

        restaurant_names[
            business_id
        ] = obj.get(
            "name",
            ""
        )


print(
    "\nRestaurant-category businesses "
    "(after accommodation exclusion):",
    f"{len(restaurant_ids):,}"
)

print(
    "Hotel/accommodation businesses:",
    f"{len(hotel_ids):,}"
)


# ============================================================
# STEP 2
# INDEX ACTUAL LOCALLY AVAILABLE PHOTOS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2 - INDEX LOCALLY AVAILABLE PHOTOS")
print("=" * 70)


available_photo_ids = set()


with os.scandir(
    PHOTO_DIR
) as entries:

    for entry in tqdm(
        entries,
        desc="Indexing image files"
    ):

        if not entry.is_file():
            continue

        filename = (
            entry.name.lower()
        )

        if not filename.endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):
            continue

        photo_id = os.path.splitext(
            entry.name
        )[0]

        available_photo_ids.add(
            photo_id
        )


print(
    "\nLocally available photos:",
    f"{len(available_photo_ids):,}"
)


# ============================================================
# STEP 3
# COUNT LOCALLY AVAILABLE PHOTOS PER RESTAURANT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3 - COUNT PHOTOS PER RESTAURANT")
print("=" * 70)


photo_count = Counter()


for obj in iter_jsonl(
    PHOTO_META,
    "Reading photos.json"
):

    business_id = obj.get(
        "business_id"
    )

    if business_id not in restaurant_ids:
        continue


    photo_id = obj.get(
        "photo_id"
    )

    if not photo_id:
        continue


    # Only count photos actually present locally
    if photo_id not in available_photo_ids:
        continue


    photo_count[
        business_id
    ] += 1


restaurants_with_enough_photos = {

    business_id

    for business_id
    in restaurant_ids

    if photo_count[
        business_id
    ] >= MIN_PHOTOS

}


print(
    f"\nRestaurants with >= {MIN_PHOTOS} photos:",
    f"{len(restaurants_with_enough_photos):,}"
)


# ============================================================
# STEP 4
# COUNT REVIEWS AND STAR DISTRIBUTIONS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4 - COUNT REVIEWS AND STAR DISTRIBUTIONS")
print("=" * 70)


review_count = Counter()

star_counts = defaultdict(
    Counter
)


for obj in iter_jsonl(
    REVIEW_FILE,
    "Review dataset - pass 1"
):

    business_id = obj.get(
        "business_id"
    )

    # Businesses already failing photo criterion can be ignored
    if (
        business_id
        not in restaurants_with_enough_photos
    ):
        continue


    review_count[
        business_id
    ] += 1


    try:

        star = int(
            round(
                float(
                    obj.get(
                        "stars",
                        0
                    )
                )
            )
        )

        star = max(
            1,
            min(
                5,
                star
            )
        )

        star_counts[
            business_id
        ][
            star
        ] += 1

    except Exception:

        continue


# ============================================================
# STEP 5
# FINAL ELIGIBLE RESTAURANTS
# ============================================================

eligible_ids = {

    business_id

    for business_id
    in restaurants_with_enough_photos

    if review_count[
        business_id
    ] >= MIN_REVIEWS

}


print("\n")
print("=" * 70)
print("FINAL ELIGIBILITY CHECK")
print("=" * 70)

print(
    f"Minimum reviews: {MIN_REVIEWS}"
)

print(
    f"Minimum locally available photos: {MIN_PHOTOS}"
)

print()

print(
    "ELIGIBLE RESTAURANTS:",
    f"{len(eligible_ids):,}"
)


# ============================================================
# IMPORTANT VALIDATION
# ============================================================

EXPECTED_ELIGIBLE_RESTAURANTS = 10197


if (
    len(eligible_ids)
    != EXPECTED_ELIGIBLE_RESTAURANTS
):

    print("\n")
    print("!" * 70)
    print("WARNING")
    print("!" * 70)

    print(
        "\nExpected eligible restaurants:",
        f"{EXPECTED_ELIGIBLE_RESTAURANTS:,}"
    )

    print(
        "Current result:",
        f"{len(eligible_ids):,}"
    )

    print(
        "\nDO NOT use the final English percentage "
        "in the thesis until this count is resolved."
    )

else:

    print("\n")
    print("=" * 70)

    print(
        "SUCCESS: Eligible restaurant count matches "
        "Table 3.2 = 10,197"
    )

    print("=" * 70)


# ============================================================
# STEP 6
# BUILD UP-TO-30 REVIEW QUOTAS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6 - BUILD REVIEW-SELECTION QUOTAS")
print("=" * 70)


quota_by_business = {}


for business_id in eligible_ids:

    quota_by_business[
        business_id
    ] = allocate_rating_quota(

        star_counts[
            business_id
        ],

        MAX_REVIEWS_PER_BUSINESS

    )


# ============================================================
# STEP 7
# SECOND REVIEW PASS
#
# RESERVOIR SAMPLING WITHIN EACH STAR GROUP
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7 - SELECT UP TO 30 REVIEWS PER BUSINESS")
print("=" * 70)


rng = random.Random(
    SEED
)


samples = {

    business_id: {

        star: []

        for star, quota
        in quota_by_business[
            business_id
        ].items()

        if quota > 0

    }

    for business_id
    in eligible_ids

}


seen = {

    business_id:
    Counter()

    for business_id
    in eligible_ids

}


for obj in iter_jsonl(
    REVIEW_FILE,
    "Review dataset - pass 2"
):

    business_id = obj.get(
        "business_id"
    )

    if business_id not in eligible_ids:
        continue


    try:

        star = int(
            round(
                float(
                    obj.get(
                        "stars",
                        0
                    )
                )
            )
        )

        star = max(
            1,
            min(
                5,
                star
            )
        )

    except Exception:

        continue


    quota = quota_by_business[
        business_id
    ].get(
        star,
        0
    )


    if quota <= 0:
        continue


    seen[
        business_id
    ][
        star
    ] += 1


    sample_list = samples[
        business_id
    ][
        star
    ]


    review_record = {

        "review_id":
        obj.get(
            "review_id",
            ""
        ),

        "stars":
        star,

        "date":
        obj.get(
            "date",
            ""
        ),

        "text":
        obj.get(
            "text",
            ""
        ) or ""

    }


    # --------------------------------------------------------
    # Reservoir sampling
    # --------------------------------------------------------

    if len(sample_list) < quota:

        sample_list.append(
            review_record
        )

    else:

        j = rng.randrange(

            seen[
                business_id
            ][
                star
            ]

        )

        if j < quota:

            sample_list[
                j
            ] = review_record


# ============================================================
# STEP 8
# LANGUAGE DETECTION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8 - DETECT LANGUAGE OF SELECTED REVIEWS")
print("=" * 70)


language_counter = Counter()

business_language_rows = []

selected_review_rows = []


for business_id in tqdm(
    sorted(
        eligible_ids
    ),
    desc="Detecting review languages"
):

    selected_reviews = []


    for star_samples in samples[
        business_id
    ].values():

        selected_reviews.extend(
            star_samples
        )


    # Chronological ordering, consistent with methodology
    selected_reviews.sort(
        key=lambda x: x[
            "date"
        ]
    )


    business_language_counter = Counter()


    for review in selected_reviews:

        language = detect_language_safe(
            review[
                "text"
            ]
        )


        language_counter[
            language
        ] += 1


        business_language_counter[
            language
        ] += 1


        selected_review_rows.append({

            "business_id":
            business_id,

            "business_name":
            restaurant_names.get(
                business_id,
                ""
            ),

            "review_id":
            review[
                "review_id"
            ],

            "stars":
            review[
                "stars"
            ],

            "date":
            review[
                "date"
            ],

            "language":
            language,

        })


    total_selected = sum(
        business_language_counter.values()
    )

    english_selected = (
        business_language_counter.get(
            "en",
            0
        )
    )


    business_language_rows.append({

        "business_id":
        business_id,

        "business_name":
        restaurant_names.get(
            business_id,
            ""
        ),

        "total_reviews":
        review_count[
            business_id
        ],

        "available_photos":
        photo_count[
            business_id
        ],

        "selected_reviews":
        total_selected,

        "english_selected_reviews":
        english_selected,

        "english_percentage":
        (
            100
            * english_selected
            / total_selected
        )
        if total_selected > 0
        else 0,

    })


# ============================================================
# STEP 9
# FINAL LANGUAGE STATISTICS
# ============================================================

total_selected_reviews = sum(
    language_counter.values()
)


english_reviews = language_counter.get(
    "en",
    0
)


unknown_reviews = language_counter.get(
    "unknown",
    0
)


non_english_reviews = (
    total_selected_reviews
    - english_reviews
)


english_percentage = (

    100
    * english_reviews
    / total_selected_reviews

    if total_selected_reviews > 0

    else 0

)


print("\n")
print("=" * 70)
print("FINAL LANGUAGE STATISTICS")
print("=" * 70)


print(
    "Eligible restaurants:       ",
    f"{len(eligible_ids):,}"
)

print(
    "Total selected reviews:     ",
    f"{total_selected_reviews:,}"
)

print(
    "English selected reviews:   ",
    f"{english_reviews:,}"
)

print(
    "Non-English + unknown:      ",
    f"{non_english_reviews:,}"
)

print(
    "Unknown reviews:            ",
    f"{unknown_reviews:,}"
)


print(
    "\nENGLISH-LANGUAGE SELECTED REVIEWS:"
)

print(
    f"{english_percentage:.2f}%"
)


# ============================================================
# STEP 10
# FULL LANGUAGE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)


language_rows = []


for language, count in (
    language_counter.most_common()
):

    percentage = (

        100
        * count
        / total_selected_reviews

        if total_selected_reviews > 0

        else 0

    )


    language_rows.append({

        "Language":
        language,

        "Selected reviews":
        count,

        "Percentage":
        percentage,

    })


language_df = pd.DataFrame(
    language_rows
)


if not language_df.empty:

    print(
        language_df.to_string(
            index=False,
            formatters={
                "Percentage":
                lambda x:
                f"{x:.4f}%"
            }
        )
    )


# ============================================================
# STEP 11
# THESIS TABLE 3.2 VALUE
# ============================================================

table_3_2_language = pd.DataFrame(
    [
        {
            "Statistic":
            "English-language selected reviews",

            "Value":
            f"{english_percentage:.2f}%"
        }
    ]
)


print("\n")
print("=" * 70)
print("VALUE FOR THESIS TABLE 3.2")
print("=" * 70)


if (
    len(eligible_ids)
    == EXPECTED_ELIGIBLE_RESTAURANTS
):

    print(
        "\nEnglish-language selected reviews | "
        f"{english_percentage:.2f}%"
    )

else:

    print(
        "\nWARNING: eligible count is not 10,197."
    )

    print(
        "Do not copy this percentage into the thesis yet."
    )


# ============================================================
# STEP 12
# SAVE CSV OUTPUTS
# ============================================================

print("\n")
print("=" * 70)
print("SAVING OUTPUT FILES")
print("=" * 70)


language_csv = os.path.join(
    OUTPUT_DIR,
    "restaurant_language_distribution.csv"
)


business_csv = os.path.join(
    OUTPUT_DIR,
    "restaurant_business_language_stats.csv"
)


selected_reviews_csv = os.path.join(
    OUTPUT_DIR,
    "selected_review_languages.csv"
)


table_row_csv = os.path.join(
    OUTPUT_DIR,
    "table_3_2_language_row.csv"
)


excel_file = os.path.join(
    OUTPUT_DIR,
    "restaurant_language_statistics_final.xlsx"
)


language_df.to_csv(
    language_csv,
    index=False
)


pd.DataFrame(
    business_language_rows
).to_csv(
    business_csv,
    index=False
)


pd.DataFrame(
    selected_review_rows
).to_csv(
    selected_reviews_csv,
    index=False
)


table_3_2_language.to_csv(
    table_row_csv,
    index=False
)


# ============================================================
# STEP 13
# SAVE EXCEL
# ============================================================

with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:


    table_3_2_language.to_excel(
        writer,
        sheet_name="Table_3_2_Value",
        index=False
    )


    language_df.to_excel(
        writer,
        sheet_name="Language_Distribution",
        index=False
    )


    pd.DataFrame(
        business_language_rows
    ).to_excel(
        writer,
        sheet_name="Business_Language",
        index=False
    )


    pd.DataFrame(
        selected_review_rows
    ).to_excel(
        writer,
        sheet_name="Selected_Reviews",
        index=False
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nMain Excel file:"
)

print(
    excel_file
)


if (
    len(eligible_ids)
    == EXPECTED_ELIGIBLE_RESTAURANTS
):

    print("\n")
    print("=" * 70)
    print("FINAL VALUE TO USE IN THESIS")
    print("=" * 70)

    print(
        "\nEligible restaurant businesses | "
        f"{len(eligible_ids):,}"
    )

    print(
        "English-language selected reviews | "
        f"{english_percentage:.2f}%"
    )

else:

    print("\n")
    print("=" * 70)
    print("RESULT REQUIRES CHECKING")
    print("=" * 70)

    print(
        "\nExpected 10,197 eligible restaurants, "
        f"but found {len(eligible_ids):,}."
    )