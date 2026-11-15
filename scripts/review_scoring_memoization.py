"""Task-local exact memoization; no scoring, discovery or tie-rule changes."""
from contextlib import contextmanager
import budget_aware_review_selection as base
import run_length_balanced_review_pilot as stage4
import run_shared_pool_review_pilot as stage41
import run_source_length_tie_pilot as stage42
import budget_aware_review_selection_v4_2 as secondary
from token_length_cache import TokenLengthCache


@contextmanager
def memoized_review_scoring():
    original_init=base.Evaluator.__init__
    original_jsd=base.jsd
    original_secondary_jsd=secondary.jsd
    assert original_jsd is original_secondary_jsd
    originals=[module.TokenLengthCache for module in (stage4,stage41,stage42)]
    fast={};exact={};token_cache=[];context=[];distances={}
    def cached_jsd(a,b):
        key=(tuple(a),tuple(b))
        if key not in distances:distances[key]=original_jsd(a,b)
        return distances[key]
    def tokenizer_factory(tokenizer):
        if not token_cache:token_cache.append(TokenLengthCache(tokenizer))
        return token_cache[0]
    def initialize(ev,population,cfg,tokenizer,bid):
        original_init(ev,population,cfg,tokenizer,bid)
        # Reuse only when all inputs determining cached scores are identical.
        key=(bid,tuple((r['review_id'],r['stars'],r['date'],r['text']) for r in ev.population),cfg['search']['rating_weight'],cfg['search']['temporal_weight'],tuple(ev.full_time),tuple(sorted(ev.length_cost.items())),tuple(sorted(cfg['format'].items())))
        if context:assert context[0]==key,'Attempted cross-population or cross-rule memoization'
        else:context.append(key)
        ev.fast_cache=fast;ev.exact_cache=exact
    base.Evaluator.__init__=initialize
    base.jsd=cached_jsd
    secondary.jsd=cached_jsd
    for module in (stage4,stage41,stage42):module.TokenLengthCache=tokenizer_factory
    try:yield
    finally:
        base.Evaluator.__init__=original_init
        base.jsd=original_jsd
        secondary.jsd=original_secondary_jsd
        for module,original in zip((stage4,stage41,stage42),originals):module.TokenLengthCache=original
