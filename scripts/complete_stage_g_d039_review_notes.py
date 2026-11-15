"""Complete AI discussion notes after reading the full literal audit; no model/label changes."""
import copy
import json
from pathlib import Path
from prepare_stage_g_teacher_pilot import ROOT, read, file_sha, immutable

NEW=ROOT/'outputs/stage_g/teacher_pilot_v5_focused_d039_v1'


def main():
    if (NEW/'review_note_completion_provenance.json').exists():
        raise FileExistsError('Completed notes already preserved; new version required')
    machine=read(NEW/'machine_comparison.json');report=read(NEW/'comparison_report.json')
    draft=NEW/'report_draft_v1';draft.mkdir(exist_ok=True)
    old_hash={}
    for name in ['comparison_report.json','comparison_report.md']:
        src=NEW/name;old_hash[name]=file_sha(src);(draft/name).write_bytes(src.read_bytes())
    new_findings=[]
    for fid,bid,pid,classification,reason in [
        ('GLASS_WATER_LITERAL_CONFLICT','PRrWtQ3rdINngwESZNefVQ','Mry2XDZYS_rKCGy5b9ZNJw',
         'new_exact_verdict_conflict_uncertain_ordinary_interpretation',
         'Exact claim The plates are next to a glass of water. is U uncertain/F verified on identical photo/caption. A partially visible transparent vessel makes ordinary water/glass interpretation plausible but incomplete; do not force a verdict or classify a reasonable simple description as definitely false. D038 both accepted related paraphrases, while original had a semantic uncertainty difference. This is a new literal match/conflict, not a first-ever water ambiguity.'),
        ('WOODEN_TABLE_INTERPRETATION','nv59KN19fEQax9RL0lUdkw','RmeIk6J22rLDWAkxJT_gtg',
         'new_semantic_verdict_difference_possible_excessive_strictness',
         'D039 U verifies wooden table, F says uncertain because wood may be only a serving base and surrounding tabletop covered. Same image/caption but paraphrased claim, not exact-literal group. Original/D038 both verify the ordinary wooden-table description. Flag this potential overprecision regression for researcher judgment; do not automatically reject ordinary compatible scenes.')]:
        rows={v:[r for r in data['visual_claims'] if r['business_id']==bid and r['photo_id']==pid]
              for v,data in machine['versions'].items()}
        new_findings.append({'finding_id':fid,'business_id':bid,'photo_id':pid,'classification':classification,
            'reason':reason,'evidence':rows,'AI_second_opinion':True,'human_verified':False})
    report['findings'].extend(new_findings)
    for f in report['findings']:
        if f['finding_id']=='TWO_DRINKS_CONCERN':
            f['classification']='unsupported_count_possible_regression_review_needed'
            f['reason']+=' Original verified two drinks, D038 marked count uncertain; D039 returns to verification. This is a concerning regression relative to D038, not a newly introduced caption.'
        if f['finding_id']=='ZERO_CONTROL_REVIEW_CONTRAST':
            f['reason']+=' D039 actually retains these qualified contrasts in mixed breakfast-taco/salsa/coffee/service/atmosphere claims; it does not turn single-source frozen-chicken or chocolate complaints into recurring consensus. Positive tortillas, Texas Scramble and value remain.'
    report['exact_cross_condition_current_summary']={'matched_groups':25,'conflicts':2,
        'conflicting_claims':['The pasta dish contains green beans.','The plates are next to a glass of water.'],
        'counts_are_not_factual_error_rates':True}
    (NEW/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=(NEW/'comparison_report.md').read_text(encoding='utf-8')
    md=md.replace('Green-bean contradiction remains; background pizza/posture/food-presence semantic differences are separately listed.',
                  'Two exact contradictions: green beans and glass of water. Background pizza/posture/food-presence/wooden-table semantic differences are separately listed.')
    md+='\n## Final full-audit discussion additions\n\nD039 has **2 conflicts among25exact shared groups**: green beans and glass of water. This is not an error rate and does not establish that the filtered condition is factually better.\n'
    for f in new_findings:
        md+=f"\n### {f['finding_id']} — {f['business_id']}\n\n**{f['classification']}** — {f['reason']}\n\nPhoto `{f['photo_id']}`.\n"
        for version,rows in f['evidence'].items():
            md+=f'\n**{version}**\n'
            for r in rows:
                md+=f"- {r['condition']} / [{r['caption_id']}] / [{r['photo_marker']}] / claim{r['claim_index']+1} / **{r['verdict']}**: {r['text']}\n  - Caption quote: `{r['caption_quote']}`\n  - Teacher explanation: {r['explanation']}\n"
    md+='\nTwo-drinks count: original verified, D038 uncertain, D039 verified again. Marked a concerning count regression, awaiting user inspection; do not manufacture a definitive negative from partial visibility. The zero-caption review control preserves all noted qualified minority complaints as well as main praise after full30-review inspection.\n'
    (NEW/'comparison_report.md').write_text(md,encoding='utf-8')
    immutable(NEW/'review_note_completion_provenance.json',{'role':'AI_review_note_completion_only',
        'draft_comparison_sha256':old_hash,
        'final_comparison_sha256':{n:file_sha(NEW/n) for n in old_hash},
        'notes_script_sha256':file_sha(Path(__file__)), 'new_findings':[f['finding_id'] for f in new_findings],
        'teacher_outputs_labels_or_method_changed':False,'API_calls':0,
        'reason':'Full claim inspection exposed second literal conflict and ordinary-scene semantic difference after initial report generation; preserve report draft and add explicit notes.'})
    print(json.dumps({'findings':len(report['findings']),'literal_conflicts':2,'teacher_outputs_changed':False}))


if __name__=='__main__':main()
