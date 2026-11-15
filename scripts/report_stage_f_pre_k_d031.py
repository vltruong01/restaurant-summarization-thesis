"""Return all160-business support distributions; no K comparison, choice or grid change."""
import numpy as np
from stage_f_verification_workflow import read,write,sha
from stage_f_d031_protocol import ROOT,BASE,STATE,CONFIG,require_validated_amendment,assert_complete_hard_thresholds

STAGES=['basic_eligible_captions','after_exact_duplicate_collapse','after_relevance_filtering',
        'after_semantic_redundancy_reduction','coarser_semantic_groups']
CANDIDATES=[3,5,8,10]


def summarize(businesses):
    rows=[]
    for r in businesses:
        assert r['role']=='K_study'
        row={'business_id':r['business_id'],'raw_photos':r['raw_pool_count'],
             'basic_eligible_captions':r['basic_pool_count'],
             'after_exact_duplicate_collapse':r['after_exact_duplicate_count'],
             'after_relevance_filtering':r['after_relevance_count'],
             'after_semantic_redundancy_reduction':r['post_redundancy_count'],
             'coarser_semantic_groups':r['available_reference_groups'],
             'reference_groups_before_redundancy':r['reference_group_count_before_redundancy']}
        values=[row[k] for k in STAGES]
        assert values==sorted(values,reverse=True) and min(values)>=0
        assert row['raw_photos']>=values[0]
        rows.append(row)
    assert len({r['business_id'] for r in rows})==len(rows)
    summary={}
    for key in STAGES+['reference_groups_before_redundancy']:
        values=np.array([r[key] for r in rows],dtype=int)
        assert len(values)>0
        summary[key]={'businesses':len(rows),'total':int(values.sum()),'min':int(values.min()),
                      'mean':float(values.mean()),'median':float(np.median(values)),
                      'p90':float(np.quantile(values,.90)),'p95':float(np.quantile(values,.95)),
                      'max':int(values.max()),'zero_businesses':int((values==0).sum()),
                      'businesses_below_K':{str(k):int((values<k).sum()) for k in CANDIDATES}}
    tail=[r['business_id'] for r in rows if r['coarser_semantic_groups']>10]
    return {'status':'D031_160_BUSINESS_PREK_DISTRIBUTION_REVIEW_REQUIRED_NO_K_SELECTION',
            'decision':'D-031','businesses':rows,'summary':summary,'primary_stage_columns':STAGES,
            'K_candidates':CANDIDATES,'K_selected':None,'K_sensitivity_run':False,
            'quantiles':'numpy_linear','groups_above_max_K_businesses':tail,
            'grid_adequacy_review_needed':bool(tail),'grid_inadequacy_proven':False,'grid_changed':False,
            'review_note':'Counts are support diagnostics. A groups>10 tail is flagged, not proof of inadequacy. Return for review before K sensitivity; no silent new K.',
            'group_denominator':'coarser groups among kept semantic representatives; reference groups formed before redundancy are additionally reported',
            'zero_businesses_included':True,'original_businesses_excluded':0}


def markdown(r):
    lines=['# D-031 pre-K distribution —160K-study businesses','',
           '**Fixed filtering applied after amendment validation. No K sensitivity comparison or K selection has run.**','',
           'Exact normalized-pixel duplicates collapse deterministically; pHash is diagnostic-only. CLIP remains soft-only. '
           'Caption thresholds are relevance0.08, redundancy0.59 and coarser grouping0.46. All160original K-study businesses, including empty filtered pools, remain in the report.','',
           '| Stage | Total | Mean | Median | p90 | p95 | Max | Zero businesses | <3 | <5 | <8 | <10 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in STAGES:
        s=r['summary'][key];b=s['businesses_below_K']
        lines.append(f"| {key} | {s['total']} | {s['mean']:.3f} | {s['median']:.2f} | {s['p90']:.2f} | {s['p95']:.2f} | {s['max']} | {s['zero_businesses']} | {b['3']} | {b['5']} | {b['8']} | {b['10']} |")
    raw=sum(b['raw_photos'] for b in r['businesses']);s=r['summary']
    lines+=['',f"Raw photos {raw}; Basic failures {raw-s['basic_eligible_captions']['total']}; exact duplicates collapsed {s['basic_eligible_captions']['total']-s['after_exact_duplicate_collapse']['total']}; relevance rejects {s['after_exact_duplicate_collapse']['total']-s['after_relevance_filtering']['total']}; semantic-redundancy removals {s['after_relevance_filtering']['total']-s['after_semantic_redundancy_reduction']['total']}.",'',
            'The group metric is the unchanged coverage denominator: groups formed on the relevance-eligible pool, then counted among retained semantic representatives. Original reference-group counts before reduction are also available in JSON. Unfiltered remains all Basic-eligible captions with the unchanged fixed-seed ranking. No fixed top-K captions or final source inputs are created here.','',
            '## Grid support and stop for review','',
            f"{len(r['groups_above_max_K_businesses'])}businesses have more than10available coarser groups. This is a review flag, not permission to extend K={{3,5,8,10}} or proof that the grid is inadequate.",'',
            'Counts belowK are descriptive availability statistics, not business exclusion rules. If a business has fewer than the eventual K usable captions, keep its available captions; do not duplicate/fabricate/relax thresholds to force K. Zero-group businesses remain in empty-rate reports and have no defined coverage denominator.','',
            'Stop now and review this distribution before the sensitivity comparison. The existing validation-only rule, same maximumK/caption budget across conditions and approximate5000total-source planning target remain unchanged. Caption/source budgets and complete source serialization are still TBD; this count report does not validate source lengths or authorize budget changes.','',
            'No held-out test data, train/test production caption generation, GPTteacher or LED training. No human labels or old pHash/CLIP analyses changed.','']
    return '\n'.join(lines)


def main():
    require_validated_amendment();frozen=read(BASE/'frozen_thresholds.json');assert_complete_hard_thresholds(frozen)
    pools=read(BASE/'fixed_filter_pools.json')
    assert pools['threshold_manifest_sha256']==sha(BASE/'frozen_thresholds.json') and pools['decision']=='D-031'
    assert pools['roles_processed']==['K_study'] and pools['business_count']==160
    result=summarize(pools['businesses']);assert len(result['businesses'])==160
    result['threshold_manifest_sha256']=sha(BASE/'frozen_thresholds.json')
    result['fixed_pools_sha256']=sha(BASE/'fixed_filter_pools.json')
    result['protocol_sha256']=sha(CONFIG)
    for path in [STATE/'pre_k_distribution.json',BASE/'K_study_distribution.json']:
        if path.exists():assert read(path)==result
        else:write(path,result)
    (STATE/'pre_k_distribution.md').write_text(markdown(result),encoding='utf-8')
    print('D031_PREK_REPORT_READY_RETURN_FOR_REVIEW_NO_K_SENSITIVITY')
    return result


if __name__=='__main__':main()
