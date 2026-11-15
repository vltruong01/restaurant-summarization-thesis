"""Record actual29 full-original reads and report residual diagnostic conclusions."""
from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_c/review_selection_residual_audit_v1'


def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',*['| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    validation=json.loads((OUT/'validation_report.json').read_text(encoding='utf-8'));assert validation['validation_passed']
    scopes=json.loads((OUT/'qualitative_read_scope.json').read_text(encoding='utf-8'));reviews=json.loads((OUT/'qualitative_reviews_to_read.json').read_text(encoding='utf-8'))
    notes={
        '9tXwyr5Yqu4WnuCLOUPtrQ':'Both versions have detailed food, atmosphere and opening-period experiences. V4-only examples include an opening-night visit, margarita/price comments and a long wait with good service. V4.1-only examples include a substantial forgotten-order/closing-time complaint and a detailed positive taco/drink visit. This is a missed feasible quantitative candidate, not evidence that every v4.1 review is poor. There are only2 v4.1-only originals, so the actual read scope is5 rather than6.',
        'TIGOlkeBefuEC4TxfueJEA':'Both versions include useful detail. V4-only examples cover rum/food/music/noise, a drinks-only visit and a critical decor/service visit. V4.1-only examples cover soft opening/rum/food, decor changes and a later group visit with food/price/dietary limitations. Restoring the dominating30-review set is justified by measured rating/time/count, not by declaring the longer narratives semantically superior.',
        '6a4gLLFSgr-Q6CZXDLzBGQ':'Removed short reviews include specific oyster/bacon and mac-and-cheese opinions as well as generic repeat-visit praise. Added originals describe initial service problems, dishes/beer and noise/reservations/staff. One also discusses the neighbouring Butcher shop, illustrating that more text can introduce evidence with uncertain business relevance. Same-cell replacements keep JSD/count, but exact date-span coverage changes0.672→0.651.',
        'IkY2ticzHEn4QFn8hQLSWg':'Removed short comments contain concrete greasy/small-steak criticism and generic hype judgments. Added originals elaborate on customer-reported discriminatory service, food value/greasiness and a vegetarian-order problem. These are reviewers claims, not verified facts. Longer replacements add particular experiences, but no source-wide aspect-frequency or truth claim follows. Same-cell replacement count/JSD/date span unchanged.',
        'TcNZXteosegb1RO4O5hREw':'Removed concise praise includes po-boys/atmosphere and a specific catfish opinion. Added originals mention historical oyster availability, oyster serving days/price/parking and queue/side/beer detail. A longer review is not a present-day fact source or necessarily a better consensus sample. Rating/time-bin JSD and count stay fixed while exact date-span coverage changes0.721→0.747.'
    }
    for c in scopes:
        assert {r['review_id'] for r in reviews if r['business_id']==c['business_id']}==set(c['read_review_ids'])
        c['observations']=notes[c['business_id']];c['status']='assistant_read_complete'
    report={'date':'2026-10-06','read_reviews':len(reviews),'businesses':len(scopes),'cases':scopes,
            'sampling':'First3 chronological v4-only/v4.1-only originals in two residual cases (Taco only has2 v4.1-only); three removed/added originals in each of three lowest-token exact30 probes. All29 full original texts actually read.',
            'limitations':'Purposive assistant inspection, not formal human evaluation, exhaustive source reading, aspect recall, semantic diversity scoring or proof of generated-summary quality. Source middle50% BODY length is a diagnostic probe, not a quality filter.'}
    assert len(reviews)==29
    (OUT/'qualitative_review_notes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    pools=pd.read_csv(OUT/'diagnostic_pool_summary.csv');ties=pd.read_csv(OUT/'objective_equivalence_audit.csv');residual=pd.read_csv(OUT/'residual_case_diagnostics.csv')
    probes=json.loads((OUT/'same_histogram_counterfactuals.json').read_text(encoding='utf-8'))
    group=ties.groupby('review_budget').agg(businesses=('business_id','size'),median_equivalent_sets=('exact_histogram_equivalent_sets','median'),longer_same_histogram_exists=('longer_same_histogram_exists','sum'),median_selected_midrank=('chosen_body_midrank_in_group','median')).reset_index()
    lines=['# Residual coverage and rating/time/count equivalence audit — 2026-10-06',
           '**Both remaining v4-dominates-v4.1 cases are explained by omitted candidates and recover under unchanged ranking. Adding those candidates alone leaves substantial short-review shift. The current primary objectives do not identify content richness or aspect coverage.**',
           'Scope:same80 accepted train/validation businesses,40,407 unchanged detector-English originals.720 cached diagnostic choices,240 original equivalence audits, two residual4500 cases and three lowest-block exactly30 cases. Not a fresh v4.2 search/expanded2200 run/final2300 manifest. No objective/source/allocation/filter/model/API changes.',
           '## Missing feasible candidates',table(residual),
           'Taco Grande4500: v4.1=28/4349; accepted v4=29/4238 with better rating/time JSD. Rhone Rum Bar4500: v4.1=29/4485; v4=30/4403 with better rating/time. Both accepted v4 sets were missing from the v4.1 shared pool. Adding each and reranking with the same rule recovers29/30 and removes domination. No budget increase, exclusion or review truncation is needed.',
           '## Retention-only counterfactuals on all80',table(pools.round(6)),
           'All three diagnostic pools have0 v3/v4 domination,0<10/<5 and0 adjacent count decreases in this sample. Incumbent-only median BODY ratios0.493/0.707/0.761 versus freshv4.1=0.457/0.498/0.503; short20 fractions11.34%/9.78%/7.82% versus source1.05%. All ordinary old-v4 evaluations give ratios0.493/0.731/0.770 and short20=11.34%/9.70%/6.90%. These are cached reuse effects, not actual future pilot outcomes or proof that bias is resolved. Quality-anchor/candidate changes can have nonmonotonic count in other data; no guarantee is imposed.',
           '## Same primary objectives can select different evidence',table(group.round(6)),
           'At3500,63/80 businesses have a longer complete set in the current frontier with exactly the same selected count/star/time-bin histograms. Counts58/57 at4000/4500. Median equivalence-group sizes17/24/25; selected BODY midrank median0.5 each budget. Hash ties are token-neutral, but that descriptive median is not a statistical proof of an unbiased selector. The found candidate pools, token feasibility, count and strict weighted-cost ranking can still produce source-length shifts. Do not attribute every short selection to an explicit shortest-token tie rule.',
           'For all240 cases, exact count/cost finalists include the actual chosen candidate. Before acceptance, an arithmetic reconstruction (.5*(r+t) versus cached .5*r+.5*t) caused some false empty finalist counts; comparator repaired to cached evaluation cost and all240 membership assertions added. Prior diagnostic predecessor preserved; scientific baseline selections unchanged. Histograms/count/token/JSD checks and core conclusions are independent of that correction.',
           '## Same-cell diagnostic replacements, not new model inputs',
           table(pd.DataFrame([{k:v for k,v in x.items() if k in ['name','replacements','tokens_before','tokens_after','body_mean_before','body_mean_after','date_span_before','date_span_after']} for x in probes]).round(6)),
           'Three seeded same-star/same-source-time-bin replacements in each case preserve exactly30 originals and rating/time JSD: Geno1191→1376, Parkway1259→1517, Cochon1279→1549, all under3500. Removed/added IDs, complete originals and selected IDs stored separately. Source Q25–Q75 length range is a diagnostic construction only; no minimum-length eligibility gate, semantic objective, longest-review preference or final inputs. Only3 replacements were probed, so no claim that the full remaining set becomes source-like or semantically representative.',
           'Exact chronological date span can decrease/increase even when rank-bin histograms and temporal JSD stay fixed: Cochon0.672→0.651, Parkway0.721→0.747. Do not equate fixed-bin JSD equality with identical dates or all notions of temporal coverage.',
           '## Full-original reading',report['sampling'],report['limitations'],
           table(pd.DataFrame(scopes)[['name','observations']]),
           '## What to do next',
           'Candidate protection is a concrete correction that preserves the current research objectives: retain accepted v3 and v4 feasible incumbents and discovery candidates. A future fresh pilot can test that retention. It does not by itself resolve the concern about useful content representation; even objective-identical sets differ in length and detail.',
           'For the next methodological decision, state what additional evidence should distinguish these sets. Keeping rating/time/count alone requires reporting the content/length limitation and evaluating aspect/opinion coverage separately. A secondary source-length-balance criterion only among equally representative, equally sized feasible candidates would remain non-semantic but changes the present tie methodology; it must be proposed/reviewed before implementation. This audit does not adopt it, specify a new distance/threshold/weight, optimize semantic diversity or change source allocation. Length is not a quality label, and blindly maximizing tokens/choosing longest reviews is unsupported.',
           f"## Verification\n\nIndependent{validation['independent_original_diagnostic_equivalence_checks']} original/diagnostic/equivalence checks, {validation['unique_blocks_fresh_encoded']} unique blocks freshly encoded; count/complete boundaries/budget/exact histogram/base2JSD/length diagnostics checked. All accepted sources/scientific caches/config/scripts hashes unchanged before/after. Final diagnostic validation{validation['elapsed_seconds']}s. Notebook00 section9.3 contains tables/full texts/current-status handoff. No final freeze/training/GPU rental."]
    (OUT/'residual_review_report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('Recorded29 originals in5 purposive cases; generated report.')


if __name__=='__main__':main()
