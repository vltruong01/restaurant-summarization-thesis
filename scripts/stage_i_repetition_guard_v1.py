"""Narrow, auditable surface constraints. No model imports or output rewriting.

This is NOT a semantic/factuality filter. Parsing and exception limits are part
of the versioned policy. Each beam is evaluated from its actual token prefix.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

MARKERS = ("[REVIEW_CONSENSUS]", "[VISUAL_EVIDENCE]")
WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
SENTENCE_END = re.compile(r"[.!?]+")
ITEM_END = re.compile(r",|;|\b(?:and|or)\b", re.IGNORECASE)
ABBREVIATIONS = frozenset(("dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "inc", "co", "no"))
GRAMMATICAL_REDUPLICATIONS = frozenset(("had", "that"))


def _sentence_ends(text):
    for end in SENTENCE_END.finditer(text):
        if end.group() == ".":
            left, right = text[:end.start()], text[end.end():]
            if left[-1:].isdigit() and right[:1].isdigit():
                continue
            word = re.search(r"([^\W_]+)$", left)
            if word and (word.group().casefold() in ABBREVIATIONS or
                         (len(word.group()) == 1 and word.group().isupper())):
                continue
        yield end


def _words(text):
    return tuple(m.group().casefold().replace("\u2019", "'") for m in WORD.finditer(text))


def _sections(text):
    """Canonical markers select a history, not a fresh history on each occurrence.

    Bracketed syntax, including an unfinished marker, is excluded from prose.
    Unknown/repeated markers are reported, never corrected or forced.
    """
    histories = {"preamble": "", **{m: "" for m in MARKERS}}
    active = "preamble"
    pos = 0
    for m in re.finditer(r"\[REVIEW_CONSENSUS\]|\[VISUAL_EVIDENCE\]", text):
        histories[active] += text[pos:m.start()] + "\n"
        active = m.group()
        pos = m.end()
    histories[active] += text[pos:]
    return {k: re.sub(r"\[[A-Z_]+(?:\]|$)", " ", v) for k, v in histories.items() if v.strip()}


@lru_cache(maxsize=128)
def _source_pairs(source):
    """Verbatim source reduplications, NOT verified names or semantic evidence."""
    spans = list(WORD.finditer(source))
    pairs = {}
    run = 1
    run_start = 0
    for a, b in zip(spans, spans[1:]):
        word = a.group().casefold().replace("\u2019", "'")
        gap = source[a.end():b.start()]
        if word == b.group().casefold().replace("\u2019", "'") and gap.isspace() and "\n" not in gap and "\r" not in gap:
            if run == 1:
                run_start = a.start()
            run += 1
            if run > pairs.get(word, {}).get("source_max_run", 0):
                pairs[word] = {"source_quote": source[run_start:b.end()],
                               "source_start": run_start, "source_end": b.end(),
                               "source_max_run": run}
        else:
            run = 1
    return pairs


def _item_words(text):
    words = list(_words(text))
    while words and words[0] in ("a", "an", "the", "including"):
        words.pop(0)
    return tuple(words)


def analyze(text, source="", finalize=False):
    """Analyze ONLY completed surface events; EOS finalizes the final fragment.

    No prefix ban: 'The fish was good, but ...' remains a legitimate contrast.
    Extensions of a prior sentence are flagged, not silently called errors.
    Literal reduplication across spaces is distinct from completed list items.
    """
    if not isinstance(text, str) or not isinstance(source, str):
        raise ValueError("text/source must be strings")
    violations, flags, exceptions = [], [], []
    source_pairs = _source_pairs(source)
    all_sections = _sections(text)
    for section, body in all_sections.items():
        spans = list(WORD.finditer(body))
        output_run = 1
        for a, b in zip(spans, spans[1:]):
            gap = body[a.end():b.start()]
            # Newline/paragraph boundaries are not literal local stutters.
            closed = b.end() < len(body) or finalize
            word = a.group().casefold().replace("\u2019", "'")
            if closed and word == b.group().casefold().replace("\u2019", "'") and gap and gap.isspace() and "\n" not in gap and "\r" not in gap:
                output_run += 1
                item = {"rule": "adjacent_word", "section": section,
                        "text": body[a.start():b.end()], "word": word,
                        "start": a.start(), "end": b.end(), "output_run": output_run}
                if word in GRAMMATICAL_REDUPLICATIONS and output_run == 2:
                    flags.append({**item, "rule": "grammatical_reduplication_not_blocked",
                                  "interpretation": "grammatical_or_stutter_ambiguous_requires_review"})
                elif word in source_pairs and output_run <= source_pairs[word]["source_max_run"]:
                    exceptions.append({**item, **source_pairs[word],
                                       "basis": "verbatim_source_pair_not_verified_name"})
                else:
                    violations.append(item)
            else:
                output_run = 1

        # Conservative mechanical punctuation with explicit abbreviation and
        # decimal exemptions. Not a claim of linguistically perfect parsing.
        chunks, pos = [], 0
        for end in _sentence_ends(body):
            chunks.append((body[pos:end.start()], True))
            pos = end.end()
        if body[pos:].strip():
            chunks.append((body[pos:], bool(finalize)))
        completed = []
        for chunk, closed in chunks:
            current = _words(chunk)
            if closed and current and current in completed:
                violations.append({"rule": "duplicate_sentence", "section": section,
                                   "text": chunk.strip(), "normalized_words": list(current)})
            for old in completed:
                if len(old) >= 3 and len(current) > len(old) and current[:len(old)] == old:
                    flags.append({"rule": "phrase_extension_after_repeat", "section": section,
                                  "prior_words": list(old), "text": chunk.strip(),
                                  "interpretation": "possible_filler_or_legitimate_contrast_requires_review"})
            if closed and current:
                completed.append(current)

            # Only finished items are blocked. A shared first noun does not
            # prohibit a distinct item, e.g. 'fish, fish tacos'. First-clause
            # suffix matching handles 'service was slow, slow, or slow'.
            items, left = [], 0
            for sep in ITEM_END.finditer(chunk):
                items.append((chunk[left:sep.start()], True))
                left = sep.end()
            items.append((chunk[left:], closed))
            prior_items = []
            for item_text, item_closed in items:
                item = _item_words(item_text)
                if item_closed and item:
                    match = next((old for old in prior_items
                                  if len(old) >= len(item) and old[-len(item):] == item
                                  and (len(old) == len(item) or not set(old) & {"not", "no", "never", "without"})), None)
                    if match is not None:
                        violations.append({"rule": "duplicate_list_item", "section": section,
                                           "text": item_text.strip(), "normalized_words": list(item),
                                           "prior_item_words": list(match)})
                    prior_items.append(item)

    for marker in re.findall(r"\[[^\]\n]*(?:\]|$)", text):
        if marker not in MARKERS:
            flags.append({"rule": "unexpected_marker", "marker": marker})
    for marker in MARKERS:
        if text.count(marker) > 1:
            flags.append({"rule": "repeated_marker", "marker": marker})
    return {"violations": violations, "flags": flags,
            "source_reduplication_exceptions": exceptions}


class RepetitionGuardLogitsProcessor:
    """Lazy score-frontier filtering for the pinned beam4 implementation.

    Evaluate candidates down to eight legal options per row, including all
    raw-score ties; lower tails are NOT pruned or renormalized. An unevaluated
    candidate cannot enter global top8 in exact arithmetic. FP32 beam-score
    addition can collapse inequalities: every next prefix AND final output is
    checked and an admitted violation fails closed (no retry/fallback).

    The runner pins Transformers' beam selection source. This is not a generic
    sampling/greedy processor. No mutable row-index history survives reordering.
    """
    def __init__(self, tokenizer, source, num_beams=4, trace_path=None):
        if num_beams != 4 or tokenizer.eos_token_id != 2:
            raise ValueError("Only approved beam4/LED EOS2 supported")
        self.tokenizer = tokenizer
        self.source = source
        self.num_beams = num_beams
        self.frontier = 2 * num_beams
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.calls = 0
        self.tested_candidates = 0
        self.blocked_candidates = 0
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self.decode = lru_cache(maxsize=2048)(self._decode)
        self.last_scores_by_prefix = {}
        self.selected_continuations_checked = 0

    def _decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)

    def _emit(self, record):
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def validate_final(self, ids):
        ids = tuple(int(i) for i in ids)
        report = analyze(self.decode(ids), self.source, finalize=True)
        if report["violations"]:
            self._emit({"event": "FAIL_CLOSED_FINAL", "token_ids": list(ids), "analysis": report})
            raise ValueError("Guard violation in final sequence; stop without rewrite/fallback")
        return report

    def __call__(self, input_ids, scores):
        import torch  # CPU tensor tests are possible; no model/device side effect.
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0] or scores.shape[0] != self.num_beams:
            raise ValueError("Expected exactly one beam4 batch")
        if torch.isnan(scores).any().item() or torch.isposinf(scores).any().item():
            raise ValueError("NaN/+Inf scores; stop")
        result = scores.clone()
        cpu = scores.detach().float().cpu()
        prefixes = input_ids.detach().cpu().tolist()
        self.calls += 1
        for row, prefix_list in enumerate(prefixes):
            prefix = tuple(prefix_list)
            before = analyze(self.decode(prefix), self.source, finalize=False)
            if before["violations"]:
                self._emit({"event": "FAIL_CLOSED_PREFIX", "token_ids": prefix_list, "analysis": before})
                raise ValueError("An invalid beam prefix was admitted; stop (including FP32 tie risk)")
            finite = torch.isfinite(cpu[row])
            if not finite.any().item():
                raise ValueError("All scores masked; no fallback")
            order = torch.argsort(cpu[row], descending=True, stable=True).tolist()
            legal, cutoff, rejected, tested = 0, None, [], []
            for token in order:
                score = float(cpu[row, token])
                if score == float("-inf"):
                    break
                if legal >= self.frontier and score < cutoff:
                    break
                candidate = prefix + (token,)
                assessment = analyze(self.decode(candidate), self.source,
                                     finalize=(token == self.tokenizer.eos_token_id))
                self.tested_candidates += 1
                tested.append(token)
                if assessment["violations"]:
                    result[row, token] = -float("inf")
                    self.blocked_candidates += 1
                    rejected.append({"token_id": token, "original_score": score,
                                     "violations": assessment["violations"]})
                else:
                    legal += 1
                    if legal >= self.frontier:
                        cutoff = score
            if legal == 0:
                self._emit({"event": "FAIL_CLOSED_NO_LEGAL_TOKEN", "row": row,
                            "prefix_ids": prefix_list, "rejected": rejected})
                raise ValueError("No finite legal continuation; no fallback")
            if rejected:
                self._emit({"event": "MASK", "call": self.calls, "row_at_this_step_only": row,
                            "prefix_ids": prefix_list, "source_sha256": self.source_sha256,
                            "tested_token_ids": tested, "legal_frontier_count": legal,
                            "rejected": rejected})
        # Ephemeral actual-prefix keys, never beam-row identity. Used only by
        # the independent top8 safety hook before HF updates/finalizes beams.
        self.last_scores_by_prefix = {}
        for i, p in enumerate(prefixes):
            key = tuple(p)
            available = torch.isfinite(result[i].detach()).cpu()
            previous = self.last_scores_by_prefix.get(key)
            self.last_scores_by_prefix[key] = available if previous is None else (previous | available)
        return result


class SelectedCandidateSafetyCheck:
    """Validate actual top8 before HF beam update/finalization; never repair.

    Prevents FP32 accumulation ties from admitting an untested forbidden tail.
    Pinned HF also presents -Inf backup slots when few finite options exist;
    these impossible transitions are ignored, not counted as legal outputs.
    """
    def __init__(self, guard):
        self.guard = guard

    def __call__(self, input_ids, scores=None, **kwargs):
        import torch
        rows = input_ids.detach().cpu().tolist()
        if len(rows) != self.guard.frontier:
            raise ValueError("Pinned HF top8 selected-continuation hook changed")
        for ids in rows:
            prefix, token = tuple(ids[:-1]), ids[-1]
            known = self.guard.last_scores_by_prefix.get(prefix)
            if known is None:
                raise ValueError("Selected continuation not bound to an actual processed prefix")
            if not known[token].item():
                continue
            report = analyze(self.guard.decode(tuple(ids)), self.guard.source,
                             finalize=token == self.guard.tokenizer.eos_token_id)
            self.guard.selected_continuations_checked += 1
            if report["violations"]:
                self.guard._emit({"event": "FAIL_CLOSED_SELECTED_CONTINUATION", "token_ids": ids,
                                  "analysis": report, "possible_FP32_frontier_tie": True})
                raise ValueError("Selected continuation violates guard; stop before beam update")
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
