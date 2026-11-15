"""Prospectively fixed analysis helpers; this module has no dataset/GPU reader.
Current tests use synthetic numbers/text only, not actual model outputs.
"""
import hashlib
import hmac
import json
import random
import numpy as np

CONDITIONS = ('RO', 'U', 'F', 'Shuffled')
CRITERIA = ('review_faithfulness', 'aspect_coverage', 'coherence', 'non_redundancy',
            'usefulness', 'visual_grounding_and_contribution')
CONTRASTS = (('F', 'U'), ('F', 'Shuffled'), ('U', 'RO'), ('F', 'RO'))


def require(ok, message):
    if not ok: raise ValueError(message)


def paired_statistics(values, *, replicates=5000, seed=42):
    """Rows=businesses; columns=RO,U,F,Shuffled. Paired percentile CIs.

    The seed/count/contrast order are fixed, not user-tunable study parameters.
    Input is precomputed scores, never source text or reference payloads.
    """
    require(replicates == 5000 and seed == 42, 'Analysis lock differs')
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == 4 and len(x) >= 2 and np.isfinite(x).all(),
            'Complete finite matched business matrix required; no pairwise dropping')
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(x), size=(replicates, len(x)))
    output = []
    for a, b in CONTRASTS:
        delta = x[:, CONDITIONS.index(a)] - x[:, CONDITIONS.index(b)]
        samples = delta[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975], method='linear')
        output.append(dict(contrast=a+'-'+b, mean_difference=float(delta.mean()),
                           CI95=[float(lower), float(upper)], businesses=len(x),
                           primary=(a, b) == ('F', 'U'), simultaneous_CI=False))
    return dict(conditions=list(CONDITIONS), contrasts=output, seed=seed, replicates=replicates,
                resampling_unit='business; retain all conditions and all annotators within sampled business',
                p_values_computed=False, multiplicity='Pointwise CIs; no family-wise significance claim')


def human_means(ratings, *, visual=False):
    """Business x condition x 3 independent raters. RO visual must be NaN (N/A).
    Return matched scalar means without pretending N/A is zero.
    """
    x = np.asarray(ratings, dtype=float)
    require(x.ndim == 3 and x.shape[1:] == (4, 3), 'Expected business x four conditions x three raters')
    applicable = x[:, 1:, :] if visual else x
    require(np.isfinite(applicable).all() and ((applicable >= 1) & (applicable <= 5)).all()
            and (applicable == np.floor(applicable)).all(), 'Missing/invalid actual human rating')
    if visual:
        require(np.isnan(x[:, 0, :]).all(), 'RO visual must remain N/A')
        out = np.full(x.shape[:2], np.nan)
        out[:, 1:] = applicable.mean(axis=2)
        return out
    return x.mean(axis=2)


def visual_statistics(values):
    """Supplementary visual contrasts; RO is structurally N/A, never zero."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and x.shape[1] == 4 and len(x) >= 2 and np.isnan(x[:, 0]).all()
            and np.isfinite(x[:, 1:]).all(), 'Visual matrix requires RO=N/A and complete U/F/Shuffled')
    rng = np.random.Generator(np.random.PCG64(42))
    indices = rng.integers(0, len(x), size=(5000, len(x)))
    result = []
    for a, b in (('F', 'U'), ('F', 'Shuffled')):
        delta = x[:, CONDITIONS.index(a)] - x[:, CONDITIONS.index(b)]
        ci = np.quantile(delta[indices].mean(axis=1), [.025, .975], method='linear')
        result.append(dict(contrast=a+'-'+b, mean_difference=float(delta.mean()), CI95=ci.tolist(),
                           businesses=len(x), supplementary=True))
    return dict(contrasts=result, RO='N/A', resampling_unit='business', seed=42, replicates=5000)


def ordinal_agreement(rater_by_unit, alpha_function):
    x = np.asarray(rater_by_unit, dtype=float)
    require(x.ndim == 2 and x.shape[0] == 3, 'Exactly three independent annotators required')
    # Exclude only structural N/A units (e.g. RO visual), never an inconvenient rating.
    structural_na = np.isnan(x).all(axis=0)
    x = x[:, ~structural_na]
    if x.shape[1] == 0:
        return dict(alpha=None, status='NOT_APPLICABLE', units=0, structural_NA=int(structural_na.sum()))
    require(np.isfinite(x).all() and ((x >= 1) & (x <= 5)).all()
            and (x == np.floor(x)).all(), 'Incomplete/out-of-domain agreement data')
    if len(np.unique(x)) == 1:
        return dict(alpha=None, status='UNDEFINED_NO_VARIATION', units=x.shape[1], structural_NA=int(structural_na.sum()))
    value = alpha_function(reliability_data=x, value_domain=np.arange(1, 6), level_of_measurement='ordinal')
    require(np.isfinite(value), 'Unexpected nonfinite agreement statistic')
    return dict(alpha=float(value), status='COMPUTED', units=x.shape[1], structural_NA=int(structural_na.sum()))


def separate_annotation_packets(businesses, secret, *, caption_counts=None):
    """Return a PRIVATE admin ledger and separately serializable public packets.

    Production secret is generated once by the researcher/admin, kept private,
    never replaced after seeing outputs. Fixtures use an explicitly synthetic key.
    """
    require(len(businesses) == 30 and len(set(businesses)) == 30, 'Exactly 30 unique sampled businesses')
    require(isinstance(secret, bytes) and len(secret) >= 32, 'Private >=256-bit key required')
    caption_counts = caption_counts or {}
    private, packets = {}, {}
    for annotator in ('A01', 'A02', 'A03'):
        order = [(b, c) for b in sorted(businesses) for c in CONDITIONS]
        derived = hmac.new(secret, ('presentation:'+annotator).encode(), hashlib.sha256).digest()
        random.Random(int.from_bytes(derived, 'big')).shuffle(order)
        tasks = []
        for i, (bid, condition) in enumerate(order):
            value = json.dumps([annotator, bid, condition], separators=(',', ':')).encode()
            alias = hmac.new(secret, value, hashlib.sha256).hexdigest()[:24]
            require(alias not in private, 'Alias collision')
            count = 0 if condition == 'RO' else caption_counts.get((bid, condition), 1)
            require(type(count) is int and 0 <= count <= 5, 'Condition-caption count invalid')
            private[alias] = dict(business_id=bid, condition=condition, annotator=annotator, caption_count=count)
            tasks.append(dict(alias=alias, position=i+1,
                              scores={k: 'N/A' if condition == 'RO' and k.startswith('visual_') else None for k in CRITERIA},
                              evidence_notes='', reviewer_attestation=None,
                              diagnostic_flags=dict(core_omission=None, lost_disagreement=None,
                                                    unsupported_visual_claim=None, wrong_visual_relation=None),
                              summary_artifact=None, source_packet=None))
        packets[annotator] = dict(annotator=annotator, tasks=tasks, task_count=120,
                                  status='BLANK_SYNTHETIC_FIXTURE_NOT_COMPLETED_RATINGS')
    return dict(status='PRIVATE_ADMIN_ONLY', condition_key=private, key_id=hashlib.sha256(secret).hexdigest()), packets


def bertscore_constructor_kwargs(model_directory):
    # Exactly the accepted scorer settings, with the API keyword verified in the
    # pinned 0.3.13 wheel. Calling this function does not import/load BERTScore.
    return dict(model_type=str(model_directory), num_layers=17, batch_size=1, lang='en', device='cpu',
                idf=False, rescale_with_baseline=False, use_fast_tokenizer=False)
