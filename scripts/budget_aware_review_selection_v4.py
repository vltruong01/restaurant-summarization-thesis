"""Length-stratified/uniform candidates, token-neutral ties; exploratory only."""
from collections import defaultdict
import random
from budget_aware_review_selection import Evaluator as BaseEvaluator, nondominated
from review_selection_pilot import hamilton, rank_segments, stable_seed


def priority(ids, cfg, bid, *parts):
    return stable_seed(cfg['seed'], cfg['algorithm_version'], bid, *parts, list(ids))


def choose_candidate(candidates, cfg, bid, tolerance=None):
    if not candidates: return None
    rt = cfg['search']['comparable_rating_jsd_tolerance'] if tolerance is None else tolerance
    tt = cfg['search']['comparable_temporal_jsd_tolerance'] if tolerance is None else tolerance
    tie = lambda c: priority(c['ids'], cfg, bid, 'choice')
    anchor = min(candidates, key=lambda c: (c['weighted_cost'], c['rating_jsd'], c['temporal_jsd'], -c['count'], tie(c), c['ids']))
    eligible = [c for c in candidates if c['rating_jsd'] <= anchor['rating_jsd']+rt+1e-12 and c['temporal_jsd'] <= anchor['temporal_jsd']+tt+1e-12]
    return min(eligible, key=lambda c: (-c['count'], c['weighted_cost'], tie(c), c['ids']))


class Evaluator(BaseEvaluator):
    def seed_candidates(self):
        result = []; maximum = min(len(self.population), self.cfg['selection']['maximum_reviews'])
        cheapest = min(self.by_id, key=lambda rid: (self.length_cost[rid], self.position[rid], rid))
        rescue = self.evaluate([cheapest], {'type':'seed','variant':'minimum_cost_singleton_rescue','k':1})
        for k in range(1, maximum+1):
            quotas = hamilton(self.full_stars, k)
            segments = [(star, index, segment) for star in range(1,6)
                        for index, segment in enumerate(rank_segments([r for r in self.population if int(r['stars'])==star], quotas[star]))]
            for draw in range(self.cfg['search']['uniform_rating_time_seed_draws']):
                ids = []
                for star, index, segment in segments:
                    rng = random.Random(stable_seed(self.cfg['seed'], self.cfg['algorithm_version'], self.bid, 'uniform_seed', k, draw, star, index))
                    ids.append(segment[rng.randrange(len(segment))]['review_id'])
                result.append(self.evaluate(ids, {'type':'seed','variant':'rating_time_uniform','draw':draw,'k':k}))
            for draw in range(self.cfg['search']['length_stratum_seed_draws']):
                ids=[]; strata=self.cfg['search']['length_rank_strata']
                for star,index,segment in segments:
                    pool=sorted(segment,key=lambda r:(self.length_cost[r['review_id']],self.position[r['review_id']],r['review_id']))
                    bins=min(strata,len(pool))
                    parts=[pool[i*len(pool)//bins:(i+1)*len(pool)//bins] for i in range(bins)]
                    group=parts[(k+star+index+draw)%len(parts)]
                    rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'length_stratum_seed',k,draw,star,index))
                    ids.append(group[rng.randrange(len(group))]['review_id'])
                result.append(self.evaluate(ids,{'type':'seed','variant':'rating_time_rotating_length_strata','draw':draw,'k':k}))
            for draw in range(self.cfg['search']['joint_uniform_seed_draws']):
                cells={}
                for cell,group in sorted(self.cells.items()):
                    group=sorted(group,key=self.position.__getitem__)
                    rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'joint_uniform',k,draw,*cell));rng.shuffle(group)
                    cells[cell]=group
                offsets=defaultdict(int);ids=[]
                for step in range(k):
                    offers=[group[offsets[cell]] for cell,group in cells.items() if offsets[cell]<len(group)]
                    rid=min(offers,key=lambda rid:(self.fast([*ids,rid])['weighted_cost'],priority(self.canonical([*ids,rid]),self.cfg,self.bid,'joint',k,draw,step),rid))
                    ids.append(rid);offsets[(int(self.by_id[rid]['stars']),self.bins[rid])]+=1
                result.append(self.evaluate(ids,{'type':'seed','variant':'joint_distribution_uniform','draw':draw,'k':k}))
        return result,rescue

    def incoming(self,current,iteration):
        present=set(current)
        if len(self.population)<=30:return [r['review_id'] for r in self.population if r['review_id'] not in present]
        result=set();strata=self.cfg['search']['length_rank_strata']
        for cell,group in sorted(self.cells.items()):
            available=[rid for rid in group if rid not in present]
            if not available:continue
            uniform=sorted(available,key=self.position.__getitem__)
            rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'incoming_uniform',iteration,*cell))
            result.update(rng.sample(uniform,min(len(uniform),self.cfg['search']['uniform_incoming_per_cell'])))
            bins=min(strata,len(available))
            for i in range(bins):
                part=available[i*len(available)//bins:(i+1)*len(available)//bins]
                rng=random.Random(stable_seed(self.cfg['seed'],self.cfg['algorithm_version'],self.bid,'incoming_length_stratum',iteration,*cell,i))
                result.update(rng.sample(part,min(len(part),self.cfg['search']['incoming_per_length_stratum'])))
        return sorted(result,key=self.position.__getitem__)

    def improve(self,start,budget):
        current=start;evaluated=[];statistics=[]
        for iteration in range(self.cfg['search']['local_iterations']):
            ids=current['ids'];proposals={}
            def offer(new_ids,operation):
                key=self.canonical(new_ids)
                if not key or key==ids or key in proposals:return
                proposals[key]=(self.fast(key),operation)
            if len(ids)>1:
                for rid in ids:offer([v for v in ids if v!=rid],{'type':'delete','out':[rid],'in':[]})
            for rid in self.incoming(ids,iteration):
                if len(ids)<self.cfg['selection']['maximum_reviews']:offer([*ids,rid],{'type':'add','out':[],'in':[rid]})
                for removed in ids:offer([v for v in ids if v!=removed]+[rid],{'type':'replace','out':[removed],'in':[rid]})
            values=list(proposals.values())
            tie=lambda pair:priority(pair[0]['ids'],self.cfg,self.bid,'proposal',budget,iteration)
            quality=lambda pair:(pair[0]['weighted_cost'],-pair[0]['count'],tie(pair),pair[0]['ids'])
            count=lambda pair:(-pair[0]['count'],pair[0]['weighted_cost'],tie(pair),pair[0]['ids'])
            same_count=[pair for pair in values if pair[0]['count']==len(ids)]
            limit=self.cfg['search']['exact_proposals_per_ranking'];shortlist={}
            for ranking in (sorted(values,key=quality)[:limit],sorted(values,key=count)[:limit],sorted(same_count,key=quality)[:limit]):
                for fast,operation in ranking:shortlist[fast['ids']]=operation
            feasible=[current]
            for key,operation in sorted(shortlist.items()):
                move={'iteration':iteration,'operation':operation,'before_ids':list(ids),'after_ids':list(key),
                      'tokens_before':current['tokens'],'rating_jsd_before':current['rating_jsd'],'temporal_jsd_before':current['temporal_jsd']}
                candidate=self.evaluate(key,{'type':'local_proposal','budget':budget,'iteration':iteration,'operation':operation,'before_ids':list(ids)},[*current['path'],move])
                move.update(tokens_after=candidate['tokens'],rating_jsd_after=candidate['rating_jsd'],temporal_jsd_after=candidate['temporal_jsd'])
                candidate['path'][-1].update(move);evaluated.append(candidate)
                if candidate['tokens']<=budget:feasible.append(candidate)
            chosen=choose_candidate(nondominated(feasible),self.cfg,self.bid)
            statistics.append({'iteration':iteration,'structural_moves_scored':len(proposals),'exact_proposals':len(shortlist),
                               'unevaluated_not_certified':len(proposals)-len(shortlist),'accepted_ids':list(chosen['ids'])})
            if chosen['ids']==ids:break
            current=chosen
        return evaluated,statistics


def search_population(population,cfg,tokenizer,bid,extra_feasible=None):
    ev=Evaluator(population,cfg,tokenizer,bid);seeds,rescue=ev.seed_candidates()
    for label,items in extra_feasible or []:
        seeds.append(ev.evaluate([r['review_id'] for r in items],{'type':'baseline_candidate','label':label}))
    results={};frontiers={};sensitivity=[];stats={}
    for budget in cfg['budget']['exploratory_review_token_scenarios']:
        feasible=[c for c in seeds if c['tokens']<=budget]
        rescue_used=not feasible and rescue['tokens']<=budget
        if rescue_used:feasible=[rescue]
        if feasible:
            tie=lambda c:priority(c['ids'],cfg,bid,'anchor')
            ranks=[sorted(feasible,key=lambda c:(c['weighted_cost'],-c['count'],tie(c),c['ids'])),
                   sorted(feasible,key=lambda c:(-c['count'],c['weighted_cost'],tie(c),c['ids'])),
                   sorted(feasible,key=lambda c:(c['temporal_jsd'],c['rating_jsd'],-c['count'],tie(c),c['ids']))]
            anchors=[]
            for rank in ranks:
                if rank[0]['ids'] not in [v['ids'] for v in anchors]:anchors.append(rank[0])
            local=[]
            for anchor in anchors[:cfg['search']['local_starts']]:
                proposals,iterations=ev.improve(anchor,budget);feasible.extend(c for c in proposals if c['tokens']<=budget)
                local.append({'start_ids':list(anchor['ids']),'iterations':iterations})
            unique={c['ids']:c for c in feasible};frontier=nondominated(list(unique.values()))
            chosen=choose_candidate(frontier,cfg,bid)
        else:chosen=None;frontier=[];local=[]
        results[budget]=chosen;frontiers[budget]=frontier;stats[budget]={'rescue_used':rescue_used,'local_starts':local}
        for tolerance in cfg['search']['tolerance_sensitivity']:
            choice=choose_candidate(frontier,cfg,bid,tolerance)
            sensitivity.append({'review_budget':budget,'tolerance_per_metric':tolerance,'selected_review_ids':list(choice['ids']) if choice else [],
                                'count':choice['count'] if choice else 0,'tokens':choice['tokens'] if choice else None,
                                'rating_jsd':choice['rating_jsd'] if choice else None,'temporal_jsd':choice['temporal_jsd'] if choice else None,
                                'status':'feasible' if choice else 'budget_infeasible'})
    return ev,results,frontiers,sensitivity,stats
