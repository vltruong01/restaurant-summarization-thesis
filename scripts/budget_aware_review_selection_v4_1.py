"""Controlled v4.1 correction: shared exact candidate pool and v3 incumbents."""
import copy
from budget_aware_review_selection import nondominated
from budget_aware_review_selection_v4 import Evaluator,choose_candidate,priority


def discovery_config(cfg):
    result=copy.deepcopy(cfg)
    result['algorithm_version']=cfg['sampling_and_tie_algorithm_version']
    return result


def search_population(population,cfg,tokenizer,bid,extra_feasible=None):
    rules=discovery_config(cfg);ev=Evaluator(population,rules,tokenizer,bid)
    seeds,rescue=ev.seed_candidates()
    for label,items in extra_feasible or []:
        seeds.append(ev.evaluate([r['review_id'] for r in items],{'type':'baseline_candidate','label':label}))
    shared={c['ids']:c for c in seeds};statistics={}
    budgets=sorted(cfg['budget']['exploratory_review_token_scenarios'])
    for budget in budgets:
        feasible=[c for c in seeds if c['tokens']<=budget]
        rescued=not feasible and rescue['tokens']<=budget
        if rescued:feasible=[rescue]
        local=[]
        if feasible:
            tie=lambda c:priority(c['ids'],rules,bid,'anchor')
            ranks=[sorted(feasible,key=lambda c:(c['weighted_cost'],-c['count'],tie(c),c['ids'])),
                   sorted(feasible,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids'])),
                   sorted(feasible,key=lambda c:(c['temporal_jsd'],c['rating_jsd'],-c['count'],tie(c),c['ids']))]
            anchors=[]
            for rank in ranks:
                if rank[0]['ids'] not in [v['ids'] for v in anchors]:anchors.append(rank[0])
            for anchor in anchors[:cfg['search']['local_starts']]:
                proposals,iterations=ev.improve(anchor,budget)
                for c in proposals:shared.setdefault(c['ids'],c)
                local.append({'start_ids':list(anchor['ids']),'iterations':iterations})
        statistics[budget]={'rescue_used':rescued,'local_starts':local}
    choices={};frontiers={};sensitivity=[]
    ev.shared_candidates=list(shared.values())
    for budget in budgets:
        pool=[c for c in ev.shared_candidates if c['tokens']<=budget]
        ordinary_seed_fits=any(c['tokens']<=budget for c in seeds)
        if not ordinary_seed_fits and rescue['tokens']<=budget and rescue['ids'] not in {c['ids'] for c in pool}:pool.append(rescue)
        frontier=nondominated(pool);chosen=choose_candidate(frontier,rules,bid)
        choices[budget]=chosen;frontiers[budget]=frontier
        for tolerance in cfg['search']['tolerance_sensitivity']:
            c=choose_candidate(frontier,rules,bid,tolerance)
            sensitivity.append({'review_budget':budget,'tolerance_per_metric':tolerance,'selected_review_ids':list(c['ids']) if c else [],
                                'count':c['count'] if c else 0,'tokens':c['tokens'] if c else None,
                                'rating_jsd':c['rating_jsd'] if c else None,'temporal_jsd':c['temporal_jsd'] if c else None,
                                'status':'feasible' if c else 'budget_infeasible'})
    return ev,choices,frontiers,sensitivity,statistics
