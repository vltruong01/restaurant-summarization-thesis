"""Add verified output flags/permutation interpretation; never changes predictions."""
from pathlib import Path
import json,csv,collections,shutil,zipfile,tempfile,difflib,datetime
from s1000_final_eval_core_v1 import *
R=Path(__file__).resolve().parents[1];O=R/'outputs/stage_k/s1000_test100_phase1_v1';E=O/'EVIDENCE';D=O/'ANALYSIS';Q=O/'REVIEW_PACKAGE';P=R/'outputs/final_evaluation/s1000_preparation_v1'
def cp(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);require(file_hash(p)==file_hash(q),'Copy SHA')
def append(p,s):
    with p.open('a',encoding='utf8') as f:f.write('\n\n'+s.strip()+'\n')
def main():
    require(not (E/'VISUAL_OUTPUT_PERMUTATION_AND_FLAGGED_CASES.json').exists(),'Already added; do not repeat')
    old=read(O/'PACKAGE_INTEGRITY_REPORT.json');require(old['sha256']=='7fa46308282b77eebaa0bdb8f9314912d23b9be339e9ce8300ab9d41962a4fe4','Unexpected original package')
    cp(O/'PACKAGE_INTEGRITY_REPORT.json',E/'PACKAGE_INTEGRITY_BEFORE_DIAGNOSTIC_ADDENDUM.json')
    f={(r['business_id'],r['task']):r for r in read(O/'EXECUTION/F/raw_predictions.json')};s={(r['business_id'],r['task']):r for r in read(O/'EXECUTION/S/raw_predictions.json')}
    mappings=read(P/'TEST100_SHUFFLED_INPUT_MANIFEST.json')['mappings'];proof=[]
    for m in mappings:
        donor=m['donor_id'] or m['recipient_id'];rr=s[m['recipient_id'],'visual'];ff=f[donor,'visual']
        require(rr['text']==ff['text'] and rr['token_ids']==ff['token_ids'],'Visual donor output differs')
        proof.append(dict(recipient_id=m['recipient_id'],donor_id=m['donor_id'],active_intervention=m['visual_source_changed'],S_visual_text_sha256=rr['text_sha256'],F_donor_visual_text_sha256=ff['text_sha256'],raw_tokens_equal=True,
            S_was_new_generation=rr['independent_generation']))
    metrics=list(csv.DictReader((D/'TEST100_REFERENCE_FREE_METRICS.csv').open(encoding='utf8')))
    flags=[{k:r[k] for k in ['condition','business_id','task','output_sha256','raw_generated_tokens','malformed','foreign_component_header','at_max_length','repeated_3gram_fraction']} for r in metrics if r['malformed']=='True' or r['at_max_length']=='True']
    require(len(flags)==4 and len({r['business_id'] for r in flags})==2,'Unexpected flagged cases')
    write(E/'VISUAL_OUTPUT_PERMUTATION_AND_FLAGGED_CASES.json',dict(status='VERIFIED_FROM_ACTUAL_RAW_IDS_AND_TEXT',Visual_pairs_checked=100,exact_S_equals_F_donor=100,permutation=proof,
        flagged_logical_components=flags,unique_independent_F_review_cases=2,corresponding_S_reviews_are_exact_reuse=True,predictions_modified=False,new_inference_calls=0,
        interpretation='Caption-only Visual + same F checkpoint + one-to-one whole-block donors yields permuted F outputs. Equal aggregate reference-free statistics cannot establish recipient grounding.'))
    findings='''## Verified diagnostics that require explicit interpretation

There are TWO DIFFERENT flagged F Review cases, both preserved verbatim and then reused in S:

- `oP7An13EMBE_IZSqLzQ_bQ`:130 raw tokens; the Review output contains the foreign `[VISUAL_EVIDENCE]` marker. It is a section-format contamination flag, not proof that captions entered the caption-free Review input.
- `ymE6ka6y6J-z2y2rX8QH3Q`:768 raw tokens, at the frozen generation cap, repeated-trigram fraction0.968503937. This is a severe surface repetition/cap case. Forced final EOS does not establish natural completeness. It is NOT the foreign-marker case.

These are2 independent F outputs and4 logical flagged F/S components because S Review is reused; do not count them as4 independent generations. No output was repaired, replaced or retried for quality. RO/U Review have no such flags. F/S each have2 empty Visual bodies, corresponding exactly to their2 zero-caption inputs; no Visual content was fabricated. All700 logical components end in EOS, but1 independent F Review is cap-terminated.

Direct raw-token/text comparison establishes S Visual equals the corresponding F DONOR Visual for100/100 businesses (98 newly generated S Visual calls and2 unchanged empty reuses). Thus S Visual outputs are an exact permutation of F Visual outputs. Global length/repetition and donor-caption lexical-overlap distributions are identical by construction of this caption-only, one-to-one intervention. This DOES NOT show invariance to wrong captions, equal recipient photo grounding, or failure of the negative control. The intervention changes which recipient receives each Visual description. Recipient-specific independent references/photos are needed to assess the resulting grounding loss. Per-business length/repetition changes are still descriptive; their global mean difference is constrained to zero by the permutation.

U Visual mean repeated-trigram fraction is0.122986 versus F0.060546. In the input-defined same-caption-count27-business subset, paired U-minus-F mean is0.038697 (3.87 percentage points),95% paired bootstrap interval[0.019983,0.058818]. This is a redundancy finding only; same caption count does not equate content, ancestry or supervision and does not prove better factuality/usefulness. No model selection follows.
'''
    for name in ['TEST100_REFERENCE_FREE_COMPARISON.md','TEST100_GENERATION_INTEGRITY_REPORT.md','TEST100_FINAL_INFERENCE_REPORT.md']:append(O/name,findings)
    append(O/'THESIS_SDD_DOCS_UPDATE_SUMMARY.md','Additional current notes explicitly preserve2 independent F Review format/cap problems (reused in S) and verify exact F-to-S donor Visual permutation. Identical global Visual diagnostics are therefore structural, not a grounding outcome. No predictions or scientific settings changed.')
    cp(E/'DOCUMENTATION_CHANGE_MANIFEST.json',E/'DOCUMENTATION_CHANGE_MANIFEST_BEFORE_DIAGNOSTIC_ADDENDUM.json')
    changes=read(E/'DOCUMENTATION_CHANGE_MANIFEST.json')
    note='''Verified diagnostic addendum:2 distinct F Review outputs are flagged (one foreign Visual marker, one repetition/cap at768 tokens), both retained and reused in S; no quality retry. S Visual matches its F donor output exactly in100/100 token/text comparisons, so aggregate F/S Visual length/repetition distributions are structurally permuted, not evidence of equivalent recipient grounding. U-minus-F trigram repetition in the input-defined same-count27 subset is0.038697 (95% paired CI0.019983..0.058818), a descriptive redundancy finding only. Predictions/metrics/source settings are unchanged.'''
    for c in changes:
        p=R/c['path'];append(p,note);c['after_sha256']=file_hash(p);c['diagnostic_addendum']=True
        if c['path'].startswith('thesis_sdd_docs/'):
            before=(O/c['snapshot']).read_text(encoding='utf8');after=p.read_text(encoding='utf8');dp=O/'DOCUMENTATION_DIFFS'/(p.name+'.patch')
            dp.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=p.name+' before StageK',tofile=p.name+' current')),encoding='utf8')
    (E/'DOCUMENTATION_CHANGE_MANIFEST.json').write_text(json.dumps(changes,indent=2)+'\n',encoding='utf8')
    for p in O.glob('*.md'):cp(p,Q/p.name)
    for p in E.iterdir():
        if p.is_file():cp(p,Q/'EVIDENCE'/p.name)
    for c in changes:
        if c['path'].startswith('thesis_sdd_docs/'):cp(R/c['path'],Q/'UPDATED_DOCUMENTATION'/c['path'])
    cp(Path(__file__),Q/'SCRIPTS'/Path(__file__).name)
    # Original predictions and quantitative CSV remain byte-identical.
    require(file_hash(D/'TEST100_RO_U_F_S_PREDICTIONS.jsonl')==read(D/'GENERATION_INTEGRITY_RECEIPT.json')['predictions_file_sha256'],'Prediction mutation')
    require(file_hash(D/'TEST100_REFERENCE_FREE_METRICS.csv')==read(D/'GENERATION_INTEGRITY_RECEIPT.json')['metrics_file_sha256'],'Metric table mutation')
    mf=[dict(filename=p.relative_to(Q).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(Q.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.csv']
    with (Q/'PACKAGE_MANIFEST.csv').open('w',encoding='utf8',newline='') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=['filename','bytes','sha256']);w.writeheader();w.writerows(mf)
    z=O/old['archive'];history=O/'STAGE_K_TEST100_PHASE1_REVIEW_BEFORE_DIAGNOSTIC_ADDENDUM.zip'
    require(z.resolve().parent==history.resolve().parent==O.resolve() and not history.exists(),'Archive history path validation')
    require(file_hash(z)==old['sha256'],'Original ZIP changed');z.rename(history)
    with zipfile.ZipFile(z,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as a:
        for p in sorted(Q.rglob('*')):
            if p.is_file():a.write(p,p.relative_to(Q).as_posix())
    with zipfile.ZipFile(z) as a:
        require(a.testzip() is None and len(a.namelist())==len(set(a.namelist()))==len(mf)+1,'ZIP CRC')
        with tempfile.TemporaryDirectory(prefix='stage_k_addendum_check_',dir=O) as tmp:
            t=Path(tmp);a.extractall(t)
            for r in mf:require(file_hash(t/r['filename'])==r['sha256'] and (t/r['filename']).stat().st_size==r['bytes'],'Member SHA')
            records=[json.loads(l) for l in (t/'TEST100_RO_U_F_S_PREDICTIONS.jsonl').read_text(encoding='utf8').splitlines()]
            require(len(records)==400 and len({(r['business_id'],r['condition']) for r in records})==400,'Record frame')
            for p in t.rglob('*.json'):read(p)
    final={**old,'sha256':file_hash(z),'bytes':z.stat().st_size,'payload_files':len(mf),'zip_members':len(mf)+1,'diagnostic_addendum_verified':True,
        'unique_independent_F_Review_flagged_cases':2,'S_Visual_equals_F_donor_100_of_100':True,'previous_verified_archive_preserved':history.name,
        'predictions_and_metric_tables_unchanged':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (O/'PACKAGE_INTEGRITY_REPORT.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf8');(O/(z.name+'.sha256')).write_text(final['sha256']+'  '+z.name+'\n',encoding='utf8')
    print(json.dumps(final,indent=2))
if __name__=='__main__':main()
