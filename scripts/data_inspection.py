"""Read-only Yelp inspection; diagnostics never select the experimental cohort."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import os
import sqlite3
import tempfile
import time

import orjson
import pandas as pd

ACCOMMODATION_CATEGORIES = {
    "Hotels", "Resorts", "Bed & Breakfast", "Hostels", "Vacation Rentals", "Guest Houses"
}
CORE_FIELDS = {
    "business": ["business_id", "categories"],
    "review": ["review_id", "business_id", "text", "stars", "date"],
    "photo": ["photo_id", "business_id", "label"],
}


def find_project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "configs/project.yaml").is_file():
            return candidate
    raise FileNotFoundError("Không tìm thấy configs/project.yaml; mở notebook trong thư mục project.")


def resolve_data_root(root, config):
    path = Path(os.environ.get("THESIS_DATA_ROOT", config["paths"]["data_root"]))
    return (path if path.is_absolute() else Path(root) / path).resolve()


def stream_jsonl(path, max_records=None, stats=None, progress=False):
    """Stream JSON objects, report invalid rows, and keep at most 10 error examples."""
    path = Path(path)
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive or None.")
    stats = stats if stats is not None else {}
    stats.update({"path": str(path), "file_bytes": path.stat().st_size,
                  "lines_read": 0, "bytes_read": 0, "records": 0,
                  "blank_lines": 0, "malformed_json": 0, "non_object_json": 0,
                  "error_examples": [], "complete_scan": False})
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=stats["file_bytes"], desc=path.name, unit="B", unit_scale=True)
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                stats["lines_read"] += 1
                stats["bytes_read"] += len(line)
                if bar is not None:
                    bar.update(len(line))
                if not line.strip():
                    stats["blank_lines"] += 1
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    stats["malformed_json"] += 1
                    if len(stats["error_examples"]) < 10:
                        stats["error_examples"].append({"line": line_number, "error": "malformed JSON"})
                    continue
                if not isinstance(record, dict):
                    stats["non_object_json"] += 1
                    if len(stats["error_examples"]) < 10:
                        stats["error_examples"].append({"line": line_number, "error": "JSON is not an object"})
                    continue
                stats["records"] += 1
                yield record
                if max_records is not None and stats["records"] >= max_records:
                    break
        stats["complete_scan"] = stats["bytes_read"] == stats["file_bytes"]
    finally:
        if bar is not None:
            bar.close()


def sample_jsonl(path, limit):
    stats = {}
    records = list(stream_jsonl(path, max_records=limit, stats=stats))
    return records, stats


def categories_of(record):
    value = record.get("categories")
    return {part.strip() for part in value.split(",") if part.strip()} if isinstance(value, str) else set()


def is_restaurant(record):
    categories = categories_of(record)
    return "Restaurants" in categories and not categories.intersection(ACCOMMODATION_CATEGORIES)


def valid_id(value):
    return isinstance(value, str) and bool(value) and value.strip() == value


def valid_review_stars(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value in (1, 2, 3, 4, 5)


def valid_date(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def schema_table(records):
    frame = pd.DataFrame(records)
    return pd.DataFrame([
        {"field": column, "pandas_dtype": str(frame[column].dtype),
         "python_types": dict(Counter(type(value).__name__ for value in frame[column])),
         "missing_or_null": int(frame[column].isna().sum()),
         "missing_or_null_pct": round(100 * frame[column].isna().mean(), 2)}
        for column in frame.columns
    ])


def sample_quality(records, kind, id_field):
    required = CORE_FIELDS.get(kind, [id_field])
    ids = [row.get(id_field) for row in records if valid_id(row.get(id_field))]
    counts = Counter(ids)
    checks = [
        {"check": "records inspected", "count": len(records)},
        {"check": "invalid/missing primary ID", "count": sum(not valid_id(row.get(id_field)) for row in records)},
        {"check": "duplicate ID occurrences after first (in sample)", "count": sum(count - 1 for count in counts.values())},
    ]
    for field in required:
        checks.append({"check": f"missing field: {field}", "count": sum(field not in row for row in records)})
    if kind == "review":
        checks += [
            {"check": "empty/non-string review text", "count": sum(not isinstance(row.get("text"), str) or not row["text"].strip() for row in records)},
            {"check": "invalid review stars (must be 1–5)", "count": sum(not valid_review_stars(row.get("stars")) for row in records)},
            {"check": "invalid/missing date", "count": sum(not valid_date(row.get("date")) for row in records)},
        ]
    elif kind == "photo":
        checks += [
            {"check": "empty/non-string source caption (informational, not BLIP-2 QC)",
             "count": sum(not isinstance(row.get("caption"), str) or not row["caption"].strip() for row in records)},
            {"check": "empty/non-string photo label",
             "count": sum(not isinstance(row.get("label"), str) or not row["label"].strip() for row in records)},
        ]
    elif kind == "business":
        checks.append({"check": "empty/non-string categories",
                       "count": sum(not isinstance(row.get("categories"), str) or not row["categories"].strip() for row in records)})
    return pd.DataFrame(checks)


def photo_inventory(directory):
    """Index filenames only; do not decode or filter the image pool."""
    filenames = {}
    extensions = Counter()
    duplicate_stems = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                continue
            photo_id = Path(entry.name).stem
            extensions[suffix] += 1
            if photo_id in filenames:
                duplicate_stems += 1
            else:
                filenames[photo_id] = entry.name
    return filenames, {"image_files": sum(extensions.values()), "unique_photo_stems": len(filenames),
                       "duplicate_stems": duplicate_stems, "extensions": dict(extensions),
                       "scope": "filenames only; not image integrity/duplicate-content validation"}


def inspect_photo_sample(records, directory, filenames, limit=6):
    from PIL import Image
    results = []
    for record in records[:limit]:
        photo_id = record.get("photo_id")
        filename = filenames.get(photo_id) if valid_id(photo_id) else None
        result = {"photo_id": photo_id, "business_id": record.get("business_id"),
                  "label": record.get("label"), "local_file": filename,
                  "readable": False, "width": None, "height": None, "error": None}
        if filename is None:
            result["error"] = "file not found in image inventory"
        else:
            try:
                with Image.open(Path(directory) / filename) as image:
                    result["width"], result["height"] = image.size
                    image.verify()
                result["readable"] = True
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
        results.append(result)
    return pd.DataFrame(results)


def full_core_audit(business_path, review_path, photo_path, photo_directory,
                    min_reviews=10, min_photos=5, track_review_duplicates=False, progress=True):
    """Full streaming diagnostic of original raw-count eligibility; no English filtering or splits."""
    started = time.perf_counter()
    sources = {}
    quality = Counter()
    business_ids, restaurant_ids = set(), set()
    for business in stream_jsonl(business_path, stats=sources.setdefault("business", {}), progress=progress):
        for field in CORE_FIELDS["business"]:
            quality[f"business_missing_{field}"] += field not in business
        bid = business.get("business_id")
        if not valid_id(bid):
            quality["business_invalid_id"] += 1
            continue
        if bid in business_ids:
            quality["business_duplicate_id_occurrences"] += 1
            continue
        business_ids.add(bid)
        if is_restaurant(business):
            restaurant_ids.add(bid)

    filenames, inventory = photo_inventory(photo_directory)
    photo_counts, photo_labels = Counter(), Counter()
    photo_ids = set()
    for photo in stream_jsonl(photo_path, stats=sources.setdefault("photo", {}), progress=progress):
        for field in CORE_FIELDS["photo"]:
            quality[f"photo_missing_{field}"] += field not in photo
        pid, bid = photo.get("photo_id"), photo.get("business_id")
        if not valid_id(pid):
            quality["photo_invalid_id"] += 1
            continue
        if pid in photo_ids:
            quality["photo_duplicate_id_occurrences"] += 1
        photo_ids.add(pid)
        if not valid_id(bid):
            quality["photo_invalid_business_id"] += 1
        elif bid not in business_ids:
            quality["photo_unknown_business_id"] += 1
        if pid not in filenames:
            quality["photo_missing_local_file"] += 1
        elif valid_id(bid) and bid in restaurant_ids:
            photo_counts[bid] += 1
        label = photo.get("label")
        photo_labels[str(label) if label is not None else "<missing>"] += 1
    quality["local_image_stems_without_photo_metadata"] = len(set(filenames) - photo_ids)

    review_counts, star_counts = Counter(), Counter()
    duplicate_reviews = None
    with ExitStack() as stack:
        connection = None
        if track_review_duplicates:
            temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="yelp-review-id-audit-"))
            connection = sqlite3.connect(Path(temporary) / "review_ids.sqlite")
            stack.callback(connection.close)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE ids (review_id TEXT PRIMARY KEY) WITHOUT ROWID")
            duplicate_reviews = 0
        for review in stream_jsonl(review_path, stats=sources.setdefault("review", {}), progress=progress):
            for field in CORE_FIELDS["review"]:
                quality[f"review_missing_{field}"] += field not in review
            bid, rid = review.get("business_id"), review.get("review_id")
            if not valid_id(bid):
                quality["review_invalid_business_id"] += 1
            elif bid not in business_ids:
                quality["review_unknown_business_id"] += 1
            if valid_id(bid) and bid in restaurant_ids:
                review_counts[bid] += 1
            if not valid_id(rid):
                quality["review_invalid_id"] += 1
            elif connection is not None:
                inserted = connection.execute("INSERT OR IGNORE INTO ids VALUES (?)", (rid,)).rowcount
                duplicate_reviews += inserted == 0
                if sources["review"]["records"] % 50000 == 0:
                    connection.commit()
            text = review.get("text")
            quality["review_empty_or_non_string_text"] += not isinstance(text, str) or not text.strip()
            stars = review.get("stars")
            if not valid_review_stars(stars):
                quality["review_invalid_stars"] += 1
            else:
                star_counts[int(stars)] += 1
            quality["review_invalid_date"] += not valid_date(review.get("date"))
        if connection is not None:
            connection.commit()

    eligible = [
        {"business_id": bid, "raw_review_records": review_counts[bid], "local_photo_records": photo_counts[bid]}
        for bid in sorted(restaurant_ids)
        if review_counts[bid] >= min_reviews and photo_counts[bid] >= min_photos
    ]
    parse_errors = sum(info["malformed_json"] + info["non_object_json"] for info in sources.values())
    quality_keys = [
        "business_invalid_id", "business_duplicate_id_occurrences", "photo_invalid_id",
        "photo_duplicate_id_occurrences", "photo_invalid_business_id", "photo_unknown_business_id",
        "photo_missing_local_file", "review_invalid_business_id", "review_unknown_business_id",
        "review_invalid_id", "review_empty_or_non_string_text", "review_invalid_stars", "review_invalid_date",
    ]
    for key in quality_keys:
        quality.setdefault(key, 0)
    return {
        "scope": "FULL core-file scan; original raw-review-count eligibility; no English filtering or image-integrity filtering",
        "sources": sources, "image_inventory": inventory,
        "unique_business_ids": len(business_ids), "restaurant_business_ids_after_accommodation_exclusion": len(restaurant_ids),
        "minimum_raw_reviews": min_reviews, "minimum_local_photos": min_photos,
        "eligible_count": len(eligible), "eligible_businesses": eligible,
        "parse_errors": parse_errors, "quality_flags": dict(quality),
        "photo_labels": dict(photo_labels), "review_star_counts": dict(star_counts),
        "review_duplicate_ids_checked": track_review_duplicates,
        "review_duplicate_id_occurrences": duplicate_reviews,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "counts_need_review": bool(parse_errors or (duplicate_reviews or 0) or any(quality[key] for key in [
            "business_invalid_id", "business_duplicate_id_occurrences", "business_missing_categories",
            "photo_invalid_id", "photo_duplicate_id_occurrences", "photo_invalid_business_id",
            "photo_unknown_business_id", "review_invalid_business_id", "review_unknown_business_id", "review_invalid_id",
        ])),
        "language_eligibility_policy": "unresolved; this audit does not implement Stage B/C policy",
    }
