"""Synthetic CPU checks only; no model loading or optimizer updates."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
import torch.nn.functional as F
import led_large_contrastive_objective_spec_v1 as spec


class MathematicalContract(unittest.TestCase):
    def sample(self):
        gen = torch.Generator(device='cpu').manual_seed(71)
        logits = torch.randn(5, 9, generator=gen, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([4, 6, 7, 2, -100])
        return logits, labels

    def test_zero_pair_weight_is_original_ce_and_gradient(self):
        logits, labels = self.sample()
        s = spec.sequence_score(logits, labels)
        a = spec.objective(s, s.detach(), weight=0.0)
        b = F.cross_entropy(logits, labels, ignore_index=-100)
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))
        ga = torch.autograd.grad(a, logits, retain_graph=True)[0]
        gb = torch.autograd.grad(b, logits)[0]
        self.assertTrue(torch.allclose(ga, gb, atol=1e-12, rtol=1e-12))

    def test_padding_is_excluded(self):
        logits, labels = self.sample()
        original = spec.sequence_score(logits, labels)
        changed = logits.detach().clone(); changed[-1] = 10000.0
        self.assertTrue(torch.equal(original.detach(), spec.sequence_score(changed, labels)))

    def test_eos_is_supervised(self):
        logits, labels = self.sample()
        g = torch.autograd.grad(-spec.sequence_score(logits, labels), logits)[0]
        self.assertGreater(g[3].abs().sum().item(), 0.0)
        self.assertEqual(g[4].abs().sum().item(), 0.0)

    def test_identical_scores_have_finite_constant_pair_term(self):
        s = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
        value = spec.objective(s, s)
        self.assertTrue(torch.isfinite(value))
        self.assertAlmostEqual(torch.autograd.grad(value, s)[0].item(), -1.0)

    def test_ranking_direction(self):
        sp = torch.tensor(-1.0, requires_grad=True)
        sn = torch.tensor(-0.8, requires_grad=True)
        gp, gn = torch.autograd.grad(spec.objective(sp, sn), (sp, sn))
        self.assertLess(gp.item(), -1.0)
        self.assertGreater(gn.item(), 0.0)

    def test_large_gap_is_finite_and_coefficients_bounded(self):
        for a, b in [(-10000., 0.), (0., -10000.)]:
            sp = torch.tensor(a, requires_grad=True); sn = torch.tensor(b, requires_grad=True)
            value = spec.objective(sp, sn)
            grads = torch.autograd.grad(value, (sp, sn))
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(all(torch.isfinite(g) for g in grads))
            w = spec.replay_coefficient(sp, sn).item()
            self.assertGreaterEqual(w, 0.0); self.assertLessEqual(w, 0.100001)

    def test_sequential_surrogate_equals_joint_gradient(self):
        pos, labels = self.sample(); neg = pos.detach().clone().requires_grad_(True)
        other = labels.clone(); other[1] = 3
        sp = spec.sequence_score(pos, labels); sn = spec.sequence_score(neg, other)
        direct = torch.autograd.grad(spec.objective(sp, sn), (pos, neg), retain_graph=True)
        w = spec.replay_coefficient(sp, sn)
        streamed_pos = torch.autograd.grad(-(1+w)*sp, pos)[0]
        streamed_neg = torch.autograd.grad(w*sn, neg)[0]
        self.assertTrue(torch.allclose(direct[0], streamed_pos, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(direct[1], streamed_neg, atol=1e-12, rtol=1e-12))

    def test_eight_pair_accumulation_identity(self):
        values = [torch.tensor([-1.0-i/10, -0.8], dtype=torch.float64, requires_grad=True) for i in range(8)]
        full = sum(spec.objective(v[0], v[1]) for v in values)/8
        direct = torch.autograd.grad(full, values)
        for v, expected in zip(values, direct):
            w = spec.replay_coefficient(v[0], v[1])
            got = torch.autograd.grad((-(1+w)*v[0]+w*v[1])/8, v)[0]
            self.assertTrue(torch.allclose(got, expected, atol=1e-12, rtol=1e-12))

    def test_rng_replayed_synthetic_dropout_matches_joint(self):
        # Synthetic CPU dropout only: this is not proof of LED/CUDA equivalence.
        old = torch.get_rng_state()
        try:
            torch.manual_seed(11)
            w = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
            x = torch.arange(4, dtype=torch.float64)
            yp = torch.tensor([3]); yn = torch.tensor([4])
            def score(y):
                logits = F.dropout(x, p=.25, training=True) @ w
                return spec.sequence_score(logits[None, :], y)
            before_pos = torch.get_rng_state().clone()
            sp = score(yp); before_neg = torch.get_rng_state().clone()
            sn = score(yn); after_pair = torch.get_rng_state().clone()
            expected = torch.autograd.grad(spec.objective(sp, sn), w)[0]
            torch.set_rng_state(before_pos)
            with torch.no_grad(): scout_sp = score(yp); scout_sn = score(yn)
            coeff = spec.replay_coefficient(scout_sp, scout_sn)
            torch.set_rng_state(before_pos); replay_sp = score(yp)
            spec.assert_replayed_score(replay_sp, scout_sp)
            first = torch.autograd.grad(-(1+coeff)*replay_sp, w)[0]
            torch.set_rng_state(before_neg); replay_sn = score(yn)
            spec.assert_replayed_score(replay_sn, scout_sn)
            second = torch.autograd.grad(coeff*replay_sn, w)[0]
            self.assertTrue(torch.allclose(expected, first+second, atol=1e-12, rtol=1e-12))
            self.assertTrue(torch.equal(torch.get_rng_state(), after_pair))
        finally:
            torch.set_rng_state(old)

    def test_bad_replay_fails(self):
        with self.assertRaises(ValueError):
            spec.assert_replayed_score(torch.tensor(-2.0), torch.tensor(-1.9))

    def test_invalid_or_all_ignored_stops(self):
        logits, labels = self.sample()
        with self.assertRaises(ValueError): spec.sequence_score(logits, torch.full_like(labels, -100))
        bad = logits.detach().clone(); bad[0, 0] = float('nan')
        with self.assertRaises(ValueError): spec.sequence_score(bad, labels)


if __name__ == '__main__':
    unittest.main()
