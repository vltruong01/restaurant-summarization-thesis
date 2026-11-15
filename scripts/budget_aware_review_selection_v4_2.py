"""Preserve primary representation; secondary source BODY-distribution matching."""
from bisect import bisect_right
from collections import Counter
import random
from budget_aware_review_selection import nondominated
from budget_aware_review_selection_v4 import choose_candidate, priority
from budget_aware_review_selection_v4_1 import discovery_config, search_population as primary_search
from review_selection_pilot import jsd, stable_seed


def length_profile(lengths, bins=4):
    values=sorted(lengths.values())
    cuts=[values[(len(values)-1)*q//bins] for q in range(1,bins)]
    assigned={rid:bisect_right(cuts,value) for rid,value in lengths.items()}
    counts=Counter(assigned.values())
    return cuts,assigned,[counts[i] for i in range(bins)]


def signature(ev,ids):
    stars=Counter(int(ev.by_id[rid]['stars']) for rid in ids)
    times=Counter(ev.bins[rid] for rid in ids)
    return (len(ids),tuple(stars[i] for i in range(1,6)),tuple(times[i] for i in range(5)))


def length_score(ids,profile):
    _,assigned,source=profile
    counts=Counter(assigned[rid] for rid in ids)
    return jsd(source,[counts[i] for i in range(len(source))])


def secondary_choice(group,profile,rules,bid):
    return min(group,key=lambda c:(length_score(c['ids'],profile),priority(c['ids'],rules,bid,'choice'),c['ids'])) if group else None


def refine(ev,initial,group,profile,rules,budget):
    current=initial;feasible={c['ids']:c for c in group};trace=[];evaluated=[]
    assigned=profile[1];opts=ev.cfg['secondary_length_balance']
    for iteration in range(opts['refinement_iterations']):
        proposals={};present=set(current['ids'])
        for outgoing in current['ids']:
            cell=(int(ev.by_id[outgoing]['stars']),ev.bins[outgoing])
            for length_bin in range(len(profile[2])):
                available=sorted((rid for rid in ev.cells[cell] if rid not in present and assigned[rid]==length_bin),key=ev.position.__getitem__)
                if not available:continue
                rng=random.Random(stable_seed(rules['seed'],rules['algorithm_version'],ev.bid,'secondary_length',budget,iteration,outgoing,length_bin))
                incoming=available[rng.randrange(len(available))]
                ids=ev.canonical([rid for rid in current['ids'] if rid!=outgoing]+[incoming])
                proposals[ids]={'out':outgoing,'in':incoming}
        ranked=sorted(proposals,key=lambda ids:(length_score(ids,profile),priority(ids,rules,ev.bid,'secondary_proposal',budget,iteration),ids))
        shortlist=ranked[:opts['exact_proposals_per_iteration']];iteration_candidates=[current]
        for ids in shortlist:
            c=ev.evaluate(ids,{'type':'secondary_same_cell_proposal','budget':budget,'iteration':iteration,'operation':proposals[ids],'before_ids':list(current['ids'])})
            assert signature(ev,c['ids'])==signature(ev,initial['ids'])
            evaluated.append(c)
            if c['tokens']<=budget:feasible[c['ids']]=c;iteration_candidates.append(c)
        best=secondary_choice(iteration_candidates,profile,rules,ev.bid)
        improved=length_score(best['ids'],profile)<length_score(current['ids'],profile)-1e-12
        trace.append({'iteration':iteration,'structural_proposals':len(proposals),'exact_proposals':len(shortlist),'unevaluated_not_certified':len(proposals)-len(shortlist),'before_ids':list(current['ids']),'best_ids':list(best['ids']),'accepted':improved,'length_jsd_before':length_score(current['ids'],profile),'length_jsd_best':length_score(best['ids'],profile),'tokens_before':current['tokens'],'tokens_best':best['tokens']})
        if not improved:break
        current=best
    return secondary_choice(list(feasible.values()),profile,rules,ev.bid),list(feasible.values()),trace,evaluated


def search_population(population,cfg,tokenizer,bid,incumbents=(),retained_candidates=()):
    extra=[(label,[r for r in population if r['review_id'] in set(ids)]) for label,ids in incumbents]
    ev,_,_,_,primary_stats=primary_search(population,cfg,tokenizer,bid,extra)
    rules=discovery_config(cfg)
    pool={c['ids']:c for c in ev.shared_candidates}
    for ids in retained_candidates:
        c=ev.evaluate(ids,{'type':'retained_accepted_candidate'})
        pool.setdefault(c['ids'],c)
    ordinary=list(pool.values())
    body={rid:len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for rid,r in ev.by_id.items()}
    profile=length_profile(body,cfg['secondary_length_balance']['length_bins'])
    choices={};frontiers={};groups={};refinements={};evaluations={};sensitivity=[]
    for budget in sorted(cfg['budget']['exploratory_review_token_scenarios']):
        fits=[c for c in ordinary if c['tokens']<=budget]
        if not fits:
            rescue=min(ev.by_id,key=lambda rid:(ev.length_cost[rid],ev.position[rid],rid))
            c=ev.evaluate([rescue],{'type':'minimum_cost_singleton_rescue'})
            if c['tokens']<=budget:fits=[c]
        frontier=nondominated(fits);control=choose_candidate(frontier,rules,bid)
        group=[c for c in frontier if signature(ev,c['ids'])==signature(ev,control['ids'])] if control else []
        tie=secondary_choice(group,profile,rules,bid)
        final,expanded,trace,evaluated=refine(ev,tie,group,profile,rules,budget) if tie else (None,[],[],[])
        choices[budget]={'v4_2_primary_control':control,'v4_2_existing_length_tie':tie,'v4_2_length_balanced':final}
        frontiers[budget]=frontier;groups[budget]=expanded;refinements[budget]=trace;evaluations[budget]=evaluated
        for tolerance in cfg['search']['tolerance_sensitivity']:
            c=choose_candidate(frontier,rules,bid,tolerance)
            g=[x for x in frontier if signature(ev,x['ids'])==signature(ev,c['ids'])] if c else []
            s=secondary_choice(g,profile,rules,bid)
            sensitivity.append({'review_budget':budget,'primary_tolerance':tolerance,'scope':'existing_candidates_only_no_refinement','control_ids':list(c['ids']) if c else [],'secondary_ids':list(s['ids']) if s else []})
    return {'choices':choices,'frontiers':frontiers,'equivalent_groups':groups,'refinement_traces':refinements,'secondary_evaluations':evaluations,'primary_discovery_statistics':primary_stats,'primary_candidates':ordinary,'candidate_audit':ev.audit,'body_lengths':body,'length_cutpoints':profile[0],'source_length_histogram':profile[2],'sensitivity':sensitivity}
