"""Deterministic text/evidence diagnostics; no model execution or automatic factual judge."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import re
import statistics

CONDITIONS = ("reviews_only", "unfiltered", "filtered")
VISUAL = "[VISUAL_EVIDENCE]"
REVIEW = "[REVIEW_CONSENSUS]"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sections(text):
    parts = text.split(VISUAL, 1)
    return parts[0].replace(REVIEW, "").strip(), parts[1].strip() if len(parts) == 2 else ""


def sentences(text):
    clean = re.sub(r"\[[A-Z_]+\]", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]


def normalized(text):
    return " ".join(text.lower().split())


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", re.sub(r"\[[A-Z_]+\]", "", text).lower())


def text_diagnostics(text):
    # Exact same repeated-sentence definition as the accepted cloud-return audit.
    ss = [normalized(s) for s in sentences(text)]
    cc = Counter(ss)
    duplicate = sum(n - 1 for n in cc.values())
    ww = words(text)
    adjacent = [(i, ww[i]) for i in range(len(ww) - 1) if ww[i] == ww[i + 1]]
    return {"exact_repeated_sentence": duplicate > 0,
            "sentence_count": len(ss), "duplicate_sentence_instances": duplicate,
            "duplicate_sentence_fraction": duplicate / len(ss) if ss else 0.0,
            "consecutive_repeated_word": bool(adjacent), "consecutive_repeated_word_pairs": len(adjacent),
            "consecutive_repeated_word_pair_fraction": len(adjacent) / max(1, len(ww) - 1)}


def structure(text, condition, caption_count):
    review, visual = sections(text)
    review_header = text.startswith(REVIEW + "\n") and text.count(REVIEW) == 1
    if condition == "reviews_only":
        compliant = review_header and VISUAL not in text and bool(review)
    else:
        compliant = (review_header and text.count(VISUAL) == 1
                     and "\n\n[VISUAL_EVIDENCE]\n" in text and bool(review)
                     and (bool(visual) or caption_count == 0))
    unexpected = re.findall(r"\[[A-Z_]+\]", text)
    compliant = bool(compliant and all(m in (REVIEW, VISUAL) for m in unexpected))
    zero = condition == "filtered" and caption_count == 0
    return {"section_headers_compliant": compliant,
            "empty_visual_required": zero,
            "empty_visual_compliant": bool(compliant and not visual) if zero else None,
            "visual_text_without_caption_input": bool(visual) if zero else False}


def termination(token_ids, max_length=768):
    if not token_ids:
        raise ValueError("Missing generated token IDs")
    # LED decoder-start=EOS=2. The initial decoder-start is NOT an EOS termination.
    eos = [i for i, v in enumerate(token_ids) if i > 0 and v == 2]
    first = eos[0] if eos else None
    return {"generated_tokens_including_decoder_start": len(token_ids),
            "decoder_start_correct": token_ids[0] == 2,
            "generated_bos_after_start": len(token_ids) > 1 and token_ids[1] == 0,
            "eos_after_decoder_start": first is not None,
            "first_eos_position": first,
            "eos_before_length_cap": first is not None and first < max_length - 1,
            "cap_reached": len(token_ids) >= max_length,
            "eos_at_cap_may_be_forced": first == max_length - 1,
            "nonpadding_tokens_after_first_eos": any(t != 1 for t in token_ids[first + 1:]) if first is not None else False}


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(min=min(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=percentile(.90), p95=percentile(.95), max=max(values))


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot silently report empty/missing experiment results")
    n = len(rows)
    result = {"outputs": n,
              "generated_length": distribution([r["generated_tokens_including_decoder_start"] for r in rows])}
    for key in ("exact_repeated_sentence", "consecutive_repeated_word", "cap_reached",
                "eos_after_decoder_start", "eos_before_length_cap", "eos_at_cap_may_be_forced",
                "section_headers_compliant", "visual_text_without_caption_input"):
        count = sum(bool(r[key]) for r in rows)
        result[key] = {"count": count, "denominator": n, "rate": count / n}
    zero = [r for r in rows if r["empty_visual_required"]]
    result["zero_caption_empty_visual_compliance"] = {
        "applicable": len(zero), "compliant": sum(bool(r["empty_visual_compliant"]) for r in zero),
        "rate": sum(bool(r["empty_visual_compliant"]) for r in zero) / len(zero) if zero else None}
    for key in rows[0]:
        if key.startswith("teacher_agreement_"):
            vals = [r[key] for r in rows if r[key] is not None]
            result[key] = {"macro_mean": statistics.mean(vals) if vals else None, "applicable": len(vals)}
    return result


def captions(record):
    block = record["source"].split("[VISUAL_CONTEXT]", 1)[-1] if "[VISUAL_CONTEXT]" in record["source"] else ""
    parts = re.findall(r"\[(IMG_CAP_\d+)\]\n(.*?)(?=\n\n\[IMG_CAP_\d+\]|\Z)", block, flags=re.S)
    photos = record["selected_photo_ids"]
    if len(parts) != len(photos):
        raise ValueError("Caption/photo mapping differs")
    return [{"caption_id": key, "text": text.strip(), "photo_id": photo}
            for (key, text), photo in zip(parts, photos)]


def lexical_jaccard(a, b):
    aa, bb = set(words(a)), set(words(b))
    return len(aa & bb) / len(aa | bb) if aa | bb else 0.0


def qualitative_records(case, prediction, experiment):
    review, visual = sections(prediction["text"])
    rs = sentences(review)
    review_rows = []
    for i, claim in enumerate(case["shared_review_evidence"]["claims"]):
        best = max(rs, key=lambda s: lexical_jaccard(claim["text"], s), default="")
        review_rows.append({"business_id": case["business_id"], "condition": case["condition"],
                            "subset": case["subset"], "experiment": experiment, "teacher_claim_index": i,
                            "teacher_claim": claim, "student_review_text": review,
                            "lexical_candidate_student_sentence": best,
                            "lexical_similarity_is_not_entailment": True,
                            "human_coverage": None, "human_contrast_preserved": None,
                            "human_supported_by_reviews": None, "human_notes": "", "human_verified": False})
    caps = captions(case)
    teacher_claims = case["visual_claims_with_evidence"]
    visual_rows = []
    for i, claim in enumerate(sentences(visual)):
        candidate = max(caps, key=lambda c: lexical_jaccard(claim, c["text"]), default=None)
        exact = [c for c in teacher_claims if normalized(c["text"]) == normalized(claim)]
        visual_rows.append({"business_id": case["business_id"], "condition": case["condition"],
                            "subset": case["subset"], "experiment": experiment, "student_sentence_index": i,
                            "student_sentence": claim, "caption_pool": caps,
                            "lexical_candidate_caption": candidate, "exact_teacher_claim_matches": exact,
                            "no_caption_evidence_available": not caps,
                            "human_caption_support": None, "human_factual_classification": None,
                            "human_notes": "", "human_verified": False})
    return review_rows, visual_rows


def rouge_backend():
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        return scorer, {"status": "AVAILABLE_CPU_TEXT_METRIC", "package": "rouge-score",
                        "version": importlib.metadata.version("rouge-score"), "use_stemmer": False,
                        "scope": "Teacher-target agreement only; not factual correctness"}
    except ImportError:
        return None, {"status": "NOT_INSTALLED_NO_AUTO_INSTALL", "scope": "No metric values fabricated"}


def evaluate(cases, predictions, output, experiment):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    expected = {(c["condition"], c["business_id"]) for c in cases}
    indexed = {(p["condition"], p["business_id"]): p for p in predictions}
    if len(indexed) != len(predictions) or set(indexed) != expected:
        raise ValueError("Missing, extra or duplicate predictions; partial results cannot be summarized as complete")
    scorer, rouge_info = rouge_backend()
    rows, review_worksheet, visual_worksheet = [], [], []
    for case in cases:
        pred = indexed[(case["condition"], case["business_id"])]
        row = {"business_id": case["business_id"], "condition": case["condition"], "subset": case["subset"],
               "experiment": experiment, "caption_count": len(case["selected_photo_ids"]),
               **text_diagnostics(pred["text"]), **termination(pred["token_ids"]),
               **structure(pred["text"], case["condition"], len(case["selected_photo_ids"]))}
        ref_review, ref_visual = sections(case["target"])
        pred_review, pred_visual = sections(pred["text"])
        for block, ref, hypothesis in (("full", case["target"], pred["text"]),
                                       ("review", ref_review, pred_review), ("visual", ref_visual, pred_visual)):
            scores = scorer.score(ref, hypothesis) if scorer is not None and ref.strip() else None
            for metric in ("rouge1", "rouge2", "rougeL"):
                for stat in ("precision", "recall", "fmeasure"):
                    row[f"teacher_agreement_{block}_{metric}_{stat}"] = getattr(scores[metric], stat) if scores else None
        rr, vr = qualitative_records(case, pred, experiment)
        review_worksheet.extend(rr)
        visual_worksheet.extend(vr)
        rows.append(row)
    summary = {"experiment": experiment, "status": "EXISTING_BASELINE_ONLY" if experiment == "baseline" else "GENERATED_DIAGNOSTIC_RESULTS",
               "automatic_metrics_not_human_ground_truth": True, "rouge": rouge_info,
               "BERTScore": {"status": "NOT_COMPUTED", "reason": "No installed pinned scorer/model; no model download or inference authorized in offline preparation"},
               "qualitative_assessment": "Human coverage/contrast/caption-support/factual labels remain blank; lexical candidates are navigation aids, not verification",
               "subsets": {role: {condition: aggregate([r for r in rows if r["subset"] == role and r["condition"] == condition])
                                  for condition in CONDITIONS} for role in ("random", "targeted")}}
    write_json(output / "summary.json", summary)
    write_csv(output / "per_case_metrics.csv", rows)
    for name, data in (("review_coverage_worksheet.jsonl", review_worksheet), ("visual_support_worksheet.jsonl", visual_worksheet)):
        with (output / name).open("x", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "comparison_cases.json", [{"business_id": c["business_id"], "condition": c["condition"],
                "subset": c["subset"], "source": c["source"], "teacher": c["target"],
                "baseline": c["baseline_prediction"]["text"], "diagnostic": indexed[(c["condition"], c["business_id"])]["text"],
                "captions": captions(c), "review_evidence": c["shared_review_evidence"],
                "teacher_visual_evidence": c["visual_claims_with_evidence"]} for c in cases])
    esc = html.escape
    pieces = ["<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
              "<meta http-equiv=Content-Security-Policy content=\"default-src 'none'; style-src 'unsafe-inline'\">",
              "<title>Stage I decoding comparison</title><style>body{font:15px system-ui;margin:24px;background:#f4f6fa}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:450px;overflow:auto;font:14px/1.5 system-ui}.grid{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:16px}article{background:white;padding:16px}summary{cursor:pointer}details{margin:18px 0}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>",
              f"<h1>{esc(experiment)}: existing-source comparison</h1><p>Random20 and targeted4 are separate. Teacher agreement is not factual truth. No original-photo verification or human-final annotations are implied. Baseline mode shows the saved prediction twice, without new generation.</p>"]
    for c in cases:
        pred = indexed[(c["condition"], c["business_id"])]
        pieces.append(f"<details><summary>{esc(c['subset'])} / {esc(c['business_id'])} / {esc(c['condition'])}</summary><div class=grid>")
        for label, text in (("Teacher", c["target"]), ("Saved baseline", c["baseline_prediction"]["text"]), (experiment, pred["text"])):
            pieces.append(f"<article><h3>{esc(label)}</h3><pre>{esc(text)}</pre></article>")
        pieces.append(f"</div><details><summary>Complete source / REV and IMG_CAP IDs</summary><pre>{esc(c['source'])}</pre></details></details>")
    (output / "COMPARISON.html").write_text("\n".join(pieces) + "\n</html>\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--experiment", choices=("baseline", "greedy_A", "beam_ngram_B"), required=True)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = read_jsonl(args.preparation / "cases.jsonl")
    if args.experiment == "baseline":
        predictions = [c["baseline_prediction"] for c in cases]
    else:
        if args.predictions_root is None:
            parser.error("An existing completed predictions root is required")
        predictions = [p for c in CONDITIONS for p in read_jsonl(args.predictions_root / c / "predictions.jsonl")]
    summary = evaluate(cases, predictions, args.output, args.experiment)
    print(json.dumps({"status": summary["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
