"""Deterministic bounded complete-review search; only exact LED lengths certify fit."""
from collections import Counter, defaultdict
import copy
import random
from review_selection_pilot import ordered, hamilton, rank_segments, select_initial, stable_seed, fixed_time_bins, formatted_input, jsd


def nondominated(candidates, numerical=1e-12):
    result=[]
    for b in candidates:
        dominated=False
        for a in candidates:
            if a is b: continue
            no_worse=(a['rating_jsd']<=b['rating_jsd']+numerical and a['temporal_jsd']<=b['temporal_jsd']+numerical and a['count']>=b['count'])
            better=(a['rating_jsd']<b['rating_jsd']-numerical or a['temporal_jsd']<b['temporal_jsd']-numerical or a['count']>b['count'])
            if no_worse and better: dominated=True; break
        if not dominated: result.append(b)
    return result


def choose_candidate(candidates,cfg,tolerance=None):
    if not candidates: return None
    rule=cfg['search']
    rt=rule['comparable_rating_jsd_tolerance'] if tolerance is None else tolerance
    tt=rule['comparable_temporal_jsd_tolerance'] if tolerance is None else tolerance
    anchor=min(candidates,key=lambda c:(c['weighted_cost'],c['rating_jsd'],c['temporal_jsd'],-c['count'],c['tokens'],c['ids']))
    comparable=[c for c in candidates if c['rating_jsd']<=anchor['rating_jsd']+rt+1e-12 and c['temporal_jsd']<=anchor['temporal_jsd']+tt+1e-12]
    return min(comparable,key=lambda c:(-c['count'],c['weighted_cost'],c['tokens'],c['ids']))


class Evaluator:
    def __init__(self,population,cfg,tokenizer,bid):
        self.population=ordered(population); self.cfg=cfg; self.tokenizer=tokenizer; self.bid=bid
        self.by_id={r['review_id']:r for r in self.population}
        self.position={r['review_id']:i for i,r in enumerate(self.population)}
        self.bins,self.full_time=fixed_time_bins(self.population,cfg['diagnostics']['temporal_rank_bins'])
        self.full_stars=Counter(int(r['stars']) for r in self.population)
        self.fast_cache={}; self.exact_cache={}; self.audit=[]
        self.cells=defaultdict(list); self.length_cost={}
        for r in self.population:
            rid=r['review_id']; cell=(int(r['stars']),self.bins[rid]); self.cells[cell].append(rid)
            # Ranking cost only; every feasible/proposed subset is encoded as one full block.
            self.length_cost[rid]=len(tokenizer.encode(formatted_input([r],cfg),add_special_tokens=False).ids)
        for cell in self.cells:
            self.cells[cell].sort(key=lambda rid:(self.length_cost[rid],self.position[rid],rid))

    def canonical(self,ids):
        ids=tuple(ids)
        if len(ids)!=len(set(ids)) or any(rid not in self.by_id for rid in ids): raise ValueError('Invalid candidate IDs')
        return tuple(sorted(ids,key=self.position.__getitem__))

    def fast(self,ids):
        ids=self.canonical(ids)
        if not ids: raise ValueError('Empty candidate is not a valid input')
        if ids not in self.fast_cache:
            stars=Counter(int(self.by_id[rid]['stars']) for rid in ids)
            times=Counter(self.bins[rid] for rid in ids)
            r=jsd([self.full_stars[s] for s in range(1,6)],[stars[s] for s in range(1,6)])
            t=jsd(self.full_time,[times[i] for i in range(len(self.full_time))])
            self.fast_cache[ids]={'ids':ids,'count':len(ids),'rating_jsd':r,'temporal_jsd':t,
                'weighted_cost':self.cfg['search']['rating_weight']*r+self.cfg['search']['temporal_weight']*t,
                'ranking_length_estimate':sum(self.length_cost[rid] for rid in ids)}
        return self.fast_cache[ids]

    def evaluate(self,ids,origin,path=None):
        ids=self.canonical(ids)
        if not ids or len(ids)>self.cfg['selection']['maximum_reviews']: raise ValueError('Invalid candidate size')
        if ids not in self.exact_cache:
            info=dict(self.fast(ids))
            text=formatted_input([self.by_id[rid] for rid in ids],self.cfg)
            info['tokens']=len(self.tokenizer.encode(text,add_special_tokens=True).ids)
            self.exact_cache[ids]=info
        info=dict(self.exact_cache[ids])
        info['origin']=origin; info['path']=copy.deepcopy(path or [])
        self.audit.append({'origin':origin,'ids':list(ids),'count':info['count'],'tokens':info['tokens'],
                           'rating_jsd':info['rating_jsd'],'temporal_jsd':info['temporal_jsd'],
                           'weighted_cost':info['weighted_cost'],'exact_evaluation':True})
        return info

    def seed_candidates(self):
        ids_all=[r['review_id'] for r in self.population]
        result=[]
        # Exact single-review cost uses index 1 for every review. This seed
        # guarantees that a rare-rating review is not overlooked when it is
        # the only complete original that fits the budget.
        cheapest=min(ids_all,key=lambda rid:(self.length_cost[rid],self.position[rid],rid))
        result.append(self.evaluate([cheapest],{'type':'seed','variant':'minimum_cost_singleton','k':1}))
        for k in range(1,min(len(ids_all),self.cfg['selection']['maximum_reviews'])+1):
            quotas=hamilton(self.full_stars,k)
            random_ids=[]; low_ids=[]
            for star in range(1,6):
                group=[r for r in self.population if int(r['stars'])==star]
                for segment_index,segment in enumerate(rank_segments(group,quotas[star])):
                    rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'seed',k,star,segment_index))
                    random_ids.append(segment[rng.randrange(len(segment))]['review_id'])
                    low_ids.append(min(segment,key=lambda r:(self.length_cost[r['review_id']],r['date'],r['review_id']))['review_id'])
            joint=[]
            for step in range(k):
                offers=[next(rid for rid in group if rid not in joint) for group in self.cells.values() if any(rid not in joint for rid in group)]
                rid=min(offers,key=lambda rid:(self.fast([*joint,rid])['weighted_cost'],self.length_cost[rid],self.position[rid],rid))
                joint.append(rid)
            for variant,ids in (('rating_time_random',random_ids),('rating_time_low_cost',low_ids),('joint_distribution_low_cost',joint)):
                result.append(self.evaluate(ids,{'type':'seed','variant':variant,'k':k}))
        return result

    def incoming(self,current,iteration):
        current=set(current)
        if len(self.population)<=30: return [r['review_id'] for r in self.population if r['review_id'] not in current]
        result=set()
        for cell,group in sorted(self.cells.items()):
            available=[rid for rid in group if rid not in current]
            if not available: continue
            result.update(available[:self.cfg['search']['shortest_available_per_star_time_cell']])
            rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'incoming',iteration,*cell))
            result.update(rng.sample(available,min(len(available),self.cfg['search']['seeded_available_per_star_time_cell'])))
        return sorted(result,key=self.position.__getitem__)

    def improve(self,start,budget):
        current=start; evaluated=[]; statistics=[]
        for iteration in range(self.cfg['search']['local_iterations']):
            ids=current['ids']; present=set(ids); incoming=self.incoming(ids,iteration)
            proposals={}
            def offer(new_ids,operation):
                key=self.canonical(new_ids)
                if not key or key==ids or key in proposals: return
                fast=self.fast(key)
                proposals[key]=(fast,operation)
            if len(ids)>1:
                for rid in ids: offer([v for v in ids if v!=rid],{'type':'delete','out':[rid],'in':[]})
            for rid in incoming:
                if len(ids)<self.cfg['selection']['maximum_reviews']:
                    offer([*ids,rid],{'type':'add','out':[],'in':[rid]})
                for removed in ids:
                    offer([v for v in ids if v!=removed]+[rid],{'type':'replace','out':[removed],'in':[rid]})
            values=list(proposals.values())
            quality=lambda pair:(pair[0]['weighted_cost'],-pair[0]['count'],pair[0]['ranking_length_estimate'],pair[0]['ids'])
            count=lambda pair:(-pair[0]['count'],pair[0]['weighted_cost'],pair[0]['ranking_length_estimate'],pair[0]['ids'])
            comparable=[pair for pair in values if pair[0]['rating_jsd']<=current['rating_jsd']+self.cfg['search']['comparable_rating_jsd_tolerance']+1e-12 and pair[0]['temporal_jsd']<=current['temporal_jsd']+self.cfg['search']['comparable_temporal_jsd_tolerance']+1e-12]
            cheap=lambda pair:(pair[0]['ranking_length_estimate'],pair[0]['weighted_cost'],-pair[0]['count'],pair[0]['ids'])
            limit=self.cfg['search']['exact_proposals_per_ranking']
            shortlist={}
            for ranking in (sorted(values,key=quality)[:limit],sorted(values,key=count)[:limit],sorted(comparable,key=cheap)[:limit]):
                for fast,operation in ranking: shortlist[fast['ids']]=operation
            feasible=[current]
            for key,operation in sorted(shortlist.items()):
                move={'iteration':iteration,'operation':operation,'before_ids':list(ids),'after_ids':list(key),
                      'tokens_before':current['tokens'],'rating_jsd_before':current['rating_jsd'],'temporal_jsd_before':current['temporal_jsd']}
                candidate=self.evaluate(key,{'type':'local_proposal','budget':budget,'iteration':iteration,'operation':operation,'before_ids':list(ids)},[*current['path'],move])
                move.update(tokens_after=candidate['tokens'],rating_jsd_after=candidate['rating_jsd'],temporal_jsd_after=candidate['temporal_jsd'])
                # The path stored on a first evaluation must contain both pre/post evidence.
                if candidate['path'] and candidate['path'][-1].get('after_ids')==list(key):
                    candidate['path'][-1].update(move)
                evaluated.append(candidate)
                if candidate['tokens']<=budget: feasible.append(candidate)
            next_candidate=choose_candidate(nondominated(feasible),self.cfg)
            statistics.append({'iteration':iteration,'structural_moves_scored':len(proposals),'exact_proposals':len(shortlist),
                               'unevaluated_not_certified':len(proposals)-len(shortlist),'accepted_ids':list(next_candidate['ids'])})
            if next_candidate['ids']==ids: break
            current=next_candidate
        return evaluated,statistics


def search_population(population,cfg,tokenizer,bid,extra_feasible=None):
    ev=Evaluator(population,cfg,tokenizer,bid)
    seeds=ev.seed_candidates()
    for label,items in (extra_feasible or []):
        seeds.append(ev.evaluate([r['review_id'] for r in items],{'type':'baseline_candidate','label':label}))
    results={}; sensitivity=[]; all_frontiers={}; search_stats={}
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        feasible=[c for c in seeds if c['tokens']<=budget]
        if feasible:
            anchors=[]
            ranks=[sorted(feasible,key=lambda c:(c['weighted_cost'],-c['count'],c['tokens'],c['ids'])),
                   sorted(feasible,key=lambda c:(-c['count'],c['weighted_cost'],c['tokens'],c['ids'])),
                   sorted(feasible,key=lambda c:(c['temporal_jsd'],c['rating_jsd'],-c['count'],c['tokens'],c['ids']))]
            for rank in ranks:
                if rank[0]['ids'] not in [v['ids'] for v in anchors]: anchors.append(rank[0])
            stat=[]
            for anchor in anchors[:cfg['search']['local_starts']]:
                proposals,iterations=ev.improve(anchor,budget)
                feasible.extend(c for c in proposals if c['tokens']<=budget)
                stat.append({'start_ids':list(anchor['ids']),'iterations':iterations})
            unique={c['ids']:c for c in feasible}
            frontier=nondominated(list(unique.values()))
            chosen=choose_candidate(frontier,cfg)
        else: chosen=None; frontier=[]; stat=[]
        results[budget]=chosen; all_frontiers[budget]=frontier; search_stats[budget]=stat
        for tolerance in cfg['search']['tolerance_sensitivity']:
            choice=choose_candidate(frontier,cfg,tolerance)
            sensitivity.append({'review_budget':budget,'tolerance_per_metric':tolerance,'selected_review_ids':list(choice['ids']) if choice else [],
                                'count':choice['count'] if choice else 0,'tokens':choice['tokens'] if choice else None,
                                'rating_jsd':choice['rating_jsd'] if choice else None,'temporal_jsd':choice['temporal_jsd'] if choice else None,
                                'status':'feasible' if choice else 'budget_infeasible'})
    return ev,results,all_frontiers,sensitivity,search_stats
