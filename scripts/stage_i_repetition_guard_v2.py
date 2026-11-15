"""Narrow terminal beam repair; all nonterminal v1 behavior is unchanged.

Only the forced-EOS step at prefix length 767 receives row-wise handling.
No forbidden EOS is unmasked, no text is rewritten, and no finished-beam
selection arguments or return values are modified. Imports do not load models.
"""
from __future__ import annotations

import math

from stage_i_repetition_guard_v1 import (
    RepetitionGuardLogitsProcessor as _V1Guard,
    SelectedCandidateSafetyCheck,
    analyze,
)


class NoCompliantCompletion(ValueError):
    """Expected terminal infeasibility, distinct from integrity/runtime errors."""


class RepetitionGuardLogitsProcessor(_V1Guard):
    """V1 verbatim except at the pinned max_length=768 forced-EOS position."""

    def __init__(self, tokenizer, source, num_beams=4, trace_path=None):
        super().__init__(tokenizer, source, num_beams=num_beams, trace_path=trace_path)
        self.terminal_seen = False
        self.terminal_calls = 0
        self.terminal_row_decisions = []

    @property
    def terminal_rows_pruned(self):
        """Observed rejected terminal rows; no mutation or extra policy."""
        return sum(item['event'] == 'TERMINAL_EOS_ROW_REJECTED'
                   for item in self.terminal_row_decisions)

    def __call__(self, input_ids, scores):
        # Deliberately delegate even failure behavior outside the incident's
        # exact terminal position. This is not generic dead-beam pruning.
        if input_ids.ndim != 2 or input_ids.shape[1] != 767:
            return super().__call__(input_ids, scores)

        import torch
        if scores.ndim != 2 or input_ids.shape[0] != 4 or scores.shape[0] != 4 or scores.shape[1] <= 2:
            raise ValueError('Terminal repair requires exactly one beam4 score batch')
        if torch.isnan(scores).any().item() or torch.isposinf(scores).any().item():
            raise ValueError('Terminal NaN/+Inf score; no repair')
        finite = torch.isfinite(scores)
        if not (finite.sum(dim=1).eq(1).all().item() and finite[:, 2].all().item()
                and scores[:, 2].eq(0).all().item()):
            raise ValueError('Terminal contract requires only forced EOS2 finite at score zero')
        if self.terminal_seen:
            raise ValueError('Terminal step invoked twice; no retry')

        result = scores.clone()
        prefixes = input_ids.detach().cpu().tolist()
        self.calls += 1
        # Validate the entire input first: prior admitted violations are an
        # unexpected integrity failure, not an excuse to prune that beam.
        for prefix in prefixes:
            prior = analyze(self.decode(tuple(prefix)), self.source, finalize=False)
            if prior['violations']:
                self._emit({'event': 'FAIL_CLOSED_TERMINAL_PRIOR_PREFIX',
                            'token_ids': prefix, 'analysis': prior})
                raise ValueError('Invalid prior terminal prefix; no repair')

        self.terminal_seen = True
        self.terminal_calls += 1
        self.terminal_row_decisions = []
        for row, prefix in enumerate(prefixes):
            assessment = analyze(self.decode(tuple(prefix) + (2,)), self.source, finalize=True)
            blocked = bool(assessment['violations'])
            self.tested_candidates += 1
            if blocked:
                result[row, 2] = -float('inf')
                self.blocked_candidates += 1
            decision = {
                'event': 'TERMINAL_EOS_ROW_REJECTED' if blocked else 'TERMINAL_EOS_ROW_ALLOWED',
                'call': self.calls, 'row_at_this_step_only': row,
                'prefix_length': 767, 'prefix_ids': prefix,
                'token_id': 2, 'original_score': 0.0,
                'source_sha256': self.source_sha256,
                'EOS_unmasked_or_bypassed': False,
                'analysis': assessment,
            }
            self.terminal_row_decisions.append(decision)
            self._emit(decision)

        self.last_scores_by_prefix = {}
        for row, prefix in enumerate(prefixes):
            key = tuple(prefix)
            available = torch.isfinite(result[row].detach()).cpu()
            previous = self.last_scores_by_prefix.get(key)
            self.last_scores_by_prefix[key] = available if previous is None else (previous | available)
        # Even zero legal terminal rows is passed through unchanged to pinned
        # HF MaxLength finalization. The observer rejects initial placeholders,
        # but may accept an earlier truly finished, compliant hypothesis.
        return result


TerminalRepetitionGuardLogitsProcessor = RepetitionGuardLogitsProcessor


class FinishedBeamObserver:
    """Observe pinned HF finalization without changing its computation.

    Latest detached tensor references stay on their existing device during the
    loop. Only verify_returned transfers them to CPU after generation.
    """
    method_name = '_update_finished_beams'

    def __init__(self, model, guard):
        self.model = model
        self.guard = guard
        self.calls = 0
        self.last_snapshot = None
        self._active = False
        self._wrapper = None

    def __enter__(self):
        if self._active:
            raise ValueError('Finished-beam observer already active')
        original = getattr(self.model, self.method_name, None)
        if not callable(original):
            raise ValueError('Pinned finished-beam method absent')
        self._had_override = self.method_name in getattr(self.model, '__dict__', {})
        self._previous_override = getattr(self.model, '__dict__', {}).get(self.method_name)
        self._original = original

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError('Pinned finished-beam return tuple changed')
            sequences, scores, indices, finished = result
            if (tuple(sequences.shape) != (1, 4, 768) or tuple(scores.shape) != (1, 4)
                    or tuple(finished.shape) != (1, 4) or indices.ndim != 3
                    or tuple(indices.shape[:2]) != (1, 4)):
                raise ValueError('Pinned finished-beam tensor shapes changed')
            import torch
            if finished.dtype != torch.bool:
                raise ValueError('Pinned finished-state dtype changed')
            self.calls += 1
            self.last_snapshot = {
                'sequences': sequences.detach(), 'scores': scores.detach(),
                'indices': indices.detach(), 'finished': finished.detach(),
                'cur_len': kwargs.get('cur_len'),
            }
            return result  # Preserve original tuple identity and tensor values.

        self._wrapper = observe
        setattr(self.model, self.method_name, observe)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        binding_ok = getattr(self.model, self.method_name, None) is self._wrapper
        try:
            if self._had_override:
                setattr(self.model, self.method_name, self._previous_override)
            else:
                delattr(self.model, self.method_name)
        finally:
            self._active = False
        if not binding_ok:
            raise ValueError('Finished-beam observer binding changed unexpectedly')
        return False

    @staticmethod
    def _sequence(ids, name):
        if hasattr(ids, 'detach'):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        if not isinstance(ids, (list, tuple)) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(name + ' must be exactly one integer token sequence')
        ids = list(ids)
        if len(ids) > 768 or ids[0] != 2 or any(i < 0 for i in ids):
            raise ValueError(name + ' length/start/token binding changed')
        return ids

    def _completed_path(self, ids, name):
        ids = self._sequence(ids, name)
        eos = next((i for i in range(1, len(ids)) if ids[i] == 2), None)
        if eos is None:
            raise ValueError(name + ' lacks a generated EOS beyond decoder-start')
        pad = self.guard.tokenizer.pad_token_id
        if any(token != pad for token in ids[eos + 1:]):
            raise ValueError(name + ' contains non-PAD tokens after generated EOS')
        return tuple(ids[:eos + 1]), eos

    def verify_returned(self, returned_ids):
        if self.last_snapshot is None or self.calls == 0:
            raise ValueError('No actual finished-beam snapshot; no completion proof')
        snapshot = self.last_snapshot
        # This is intentionally the sole CPU transfer of observed loop tensors.
        scores = snapshot['scores'].detach().cpu()
        finished = snapshot['finished'].detach().cpu()
        score = float(scores[0, 0])
        if math.isnan(score) or score == float('inf'):
            raise ValueError('Unexpected numerical finished-beam score')
        if not bool(finished[0, 0]) or not math.isfinite(score):
            if self.guard.terminal_seen:
                raise NoCompliantCompletion('No compliant finished hypothesis; terminal placeholders are not outputs')
            raise ValueError('No genuinely finished best hypothesis before terminal step')
        actual_best = snapshot['sequences'][0, 0].detach().cpu().tolist()
        expected_path, eos_position = self._completed_path(actual_best, 'Observed finished best')
        returned_path, returned_eos = self._completed_path(returned_ids, 'Returned best')
        if returned_path != expected_path or returned_eos != eos_position:
            raise ValueError('Returned best is not bound to actual finished-beam selection')
        analysis = self.guard.validate_final(list(returned_path))
        return {
            'actual_finished_beam_verified': True,
            'actual_normalized_sequence_score': score,
            'is_sent_finished': True, 'generated_EOS_position': eos_position,
            'decoder_start_not_counted_as_completion': True,
            'completion_origin': 'terminal_max_length_EOS' if eos_position == 767 else 'earlier_finished_EOS',
            'diagnostic_length_cap_reached': eos_position == 767,
            'terminal_guard_seen': self.guard.terminal_seen,
            'terminal_rows_rejected': sum(item['event'] == 'TERMINAL_EOS_ROW_REJECTED'
                                          for item in self.guard.terminal_row_decisions),
            'observed_finished_update_calls': self.calls,
            'raw_output_not_rewritten': True,
            'final_guard_analysis': analysis,
        }
