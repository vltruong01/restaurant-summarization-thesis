"""Prepare locked Stage N inputs; never creates accounts or ratings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from s1000_final_eval_core_v1 import read, digest, file_hash
from stage_n_human_evaluation_v1 import CRITERIA, RATERS

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'outputs/final_evaluation/s1000_preparation_v1'
SOURCE = ROOT/'outputs/stage_l/test100_human_references_v1/SOURCE_PACKAGE'

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

AMENDMENT = '''# Stage N: prospective two-rater protocol amendment

Status: PREPARATION ONLY. Supervisor approval and subsequent researcher release
for live scoring are outstanding. No actual human ratings have been collected by
this preparation. The historical three-rater protocol and all three original
presentation designs remain preserved in COORDINATOR_ONLY/HISTORICAL.

The researcher requested two actual human participants A01 and A02 because of
limited personnel. Each rates all frozen 30 TEST businesses and all four frozen
conditions: 120 items per rater, 240 records. A03 is not recruited and has no new
assignment or synthetic replacement. Original A01/A02 aliases and randomized
positions are reused byte-for-value; no cohort resampling or outcome-dependent
reordering is performed. Seed 20261028 freezes membership; seed 42 freezes the
independent per-rater presentation order, as in the historical implementation.

The original six five-point criteria and anchors remain unchanged. Review
Faithfulness, Aspect Coverage AND Usefulness judge Review Consensus only. This
is the more specific frozen rule, not a newly broadened usefulness criterion.
Coherence and Non-Redundancy judge all available generated components. Visual
Grounding and Contribution judges U/F/S against ORIGINAL RECIPIENT photos, not
donor photos or captions. RO Visual is N/A, never zero or one. An applicable
empty Visual remains a quality assessment requiring a human 1-5 score; no score
is filled automatically. Zero-F-caption cases remain; intervention applicability
does not remove their generated component from assessment. All defective outputs
and identical F/S Review texts remain verbatim. References/Teacher/automatic
scores must not be used as factual substitutes for the selected reviews/photos.

Quadratic-weighted Cohen kappa replaces three-rater ordinal alpha as the primary
agreement statistic, separately by condition and criterion (30 paired units).
Optional pooled kappa is supplementary and may be affected by condition mixture.
Use the fixed 1-5 category scale. Undefined expected disagreement is reported
null with a reason. No consensus ratings or imputation replace genuine responses.
Optional ordinal alpha is not implemented in this two-rater revision.

Require 240 genuine submitted records before analysis. Report means, medians,
distributions, per-rater summaries, valid/missing/N/A counts and actual comments.
For F-RO/F-U/F-S average the TWO scores within each business, then preserve the
30-business pairing. No RO Visual comparison. Retain the original PCG64 seed
20261028, 20,000 paired business bootstrap draws, percentile 95% CIs and 50,000
sign-flip draws. These human comparisons remain EXPLORATORY, with no new
confirmatory claim or multiplicity-adjusted winner selection. Agreement is not
semantic accuracy; two raters give limited agreement precision and cannot
establish population-wide assessor reliability.

Require consent, rubric familiarity, prior output/score exposure, Stage L
involvement, and conflicts declarations, followed by a documented coordinator
eligibility review. Prefer participants uninvolved in Stage L and unexposed to
model results. Software records declarations, not independently authenticated
identity, authorship, source inspection, or blindness. RO structure and repeated
F/S Review texts may reveal identities. Model/output-budget differences and
recipient versus model-visible source availability limit causal interpretation.

Do not release scoring merely by starting the web server. The real workspace
requires a separately approved LIVE_RELEASE.json bound to this amendment and
CONFIG.json. The distributed template is false/unapproved. Synthetic tests use
only SYNTHETIC_* business IDs in separate temporary workspaces; test scores are
never human results. This amendment must be approved BEFORE real collection.
'''

README = '''# Stage N TEST30 preparation

**Coordinator review package; do not distribute this ZIP to raters.** It contains
private condition mappings and frozen prediction provenance. Raters use only
the authenticated localhost interface. No accounts, real sessions, consent,
eligibility decisions or real ratings are bundled. Live scoring remains locked.

## Contents

- WORKSPACE/SOURCES and PHOTOS: exact selected reviews and individual original
  recipient JPEGs for the frozen 30 businesses. No human references or metrics.
- WORKSPACE/COORDINATOR_ONLY: original historical design and private derived key,
  native component receipts, shuffled provenance; NEVER share with raters.
- WORKSPACE/RUBRIC.json and TWO_RATER_PROTOCOL_AMENDMENT.md: exact anchors and
  documented pending two-rater change. Blank 240-row schema has no human scores.
- scripts: local web app, preparer, two-rater analyzer, verification/tests and
  Stage L/core helper dependencies. Runtime: Python 3.11+; standard library for
  the web app; numpy for bootstrap analysis. No GPU/API/model download is used.
- REPORTS: actual preparation checks and bounded synthetic technical results.

## Before live use (human actions; not performed by this preparation)

1. Read and obtain supervisor approval of the two-rater amendment, and explicit
   researcher authorization to open scoring. Select two real participants.
2. On a trusted local machine, from package root create each account with its
   own password (12+ characters). Commands prompt privately; no plaintext file:

       python scripts/stage_n_human_evaluation_v1.py add-user --workspace WORKSPACE --id A01 --role rater
       python scripts/stage_n_human_evaluation_v1.py add-user --workspace WORKSPACE --id A02 --role rater
       python scripts/stage_n_human_evaluation_v1.py add-user --workspace WORKSPACE --id COORDINATOR --role coordinator

   Repeating account creation preserves existing passwords and records.
3. Only after approvals, copy LIVE_RELEASE_TEMPLATE.json to
   WORKSPACE/LIVE_RELEASE.json, record actual approval time and both approvals.
   The required config/amendment hashes are precomputed in the template. This is
   an operator release record, not independent proof of supervisor identity.
4. Launch locally (never bind externally):

       python scripts/stage_n_human_evaluation_v1.py serve --workspace WORKSPACE

   Open http://127.0.0.1:8766. Each participant logs in separately, reads rubric,
   and submits truthful declarations. Coordinator reviews eligibility and
   records reasons/limitations before that rater can view or score outputs.
5. Raters work independently in their own order, inspect original reviews/photos,
   select scores without metric guidance, Save draft, resume, and Submit when
   complete. Submit locks the item; it does not auto-submit the next item.
   Partial drafts are allowed; applicable scores must be complete at submission.
   Serious grounding defects require exact output span, source IDs, explanation
   and uncertain/confirmed distinction. No unsupported score is prefilled.
6. Coordinator alone can download mapping/partial/final export. Final export
   requires all 240 submitted records, eligible declarations and event integrity.
   Back up PRIVATE_STATE separately; never include it in public review packages.

       python scripts/stage_n_human_evaluation_v1.py backup --workspace WORKSPACE --destination PRIVATE_BACKUP.sqlite3
       python scripts/stage_n_human_evaluation_v1.py export --workspace WORKSPACE --final --destination REAL_RATINGS_FINAL.json
       python scripts/analyze_stage_n_two_raters_v1.py --workspace WORKSPACE --ratings REAL_RATINGS_FINAL.json --automatic-completion WORKSPACE/COORDINATOR_ONLY/STAGE_M_COMPLETION.json --output REAL_ANALYSIS

Do not run the last analysis on the blank schema. Approval release and complete
real records are required. Local machine/OS access to the package is trusted;
HTTP rater isolation does not protect against someone reading coordinator files
directly. Local browser recovery stores only that account's current draft on the
same origin; use separate OS/browser profiles on shared computers, and log out.
This is a single-machine localhost service, not a production internet service.

## Independent verification

Run `python scripts/verify_stage_n_package_v1.py --root .` after extraction.
It hashes every listed file, source/photo and private prediction binding without
creating a real rating database. Technical test results are not human findings.
Review REPORTS for actual checks, limitations and remaining approvals. No Stage
K/L/M file, model output, reference or historical protocol is modified.
'''

def prepare(output):
    if output.exists(): raise ValueError('Use a fresh preparation directory')
    ws=output/'WORKSPACE'; private=ws/'COORDINATOR_ONLY'; hist=private/'HISTORICAL'
    hist.mkdir(parents=True); (ws/'PHOTOS').mkdir(); (ws/'SOURCES').mkdir()
    bindings={}; protected={}
    def preserve(p, destination):
        h=file_hash(p); protected[str(p.relative_to(ROOT))]=h
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p,destination)
        assert file_hash(destination)==h
        return h
    for name in ['HUMAN_EVALUATION_PROTOCOL.md','HUMAN_EVALUATION_COHORT30.json','PRIVATE_HUMAN_DESIGN_KEY.csv','HUMAN_RATING_TEMPLATE_DO_NOT_DISTRIBUTE_YET.csv','COHORT100_FREEZE.json','MODEL_AND_DECODER_LOCK.json','METRIC_LOCK.json','AUTOMATIC_EVALUATION_PROTOCOL.md','TEST100_SHUFFLED_INPUT_MANIFEST.json']:
        preserve(PREP/name,hist/name)
    preserve(ROOT/'scripts/analyze_s1000_human_ratings_prepared_v1.py',hist/'analyze_s1000_human_ratings_prepared_v1.py')
    preserve(SOURCE/'ANNOTATION_PACKAGE_MANIFEST.json',private/'SOURCE_PACKAGE_MANIFEST.json')
    preserve(ROOT/'outputs/stage_m/human_primary_scored_v1/COMPLETION.json',private/'STAGE_M_COMPLETION.json')
    cohort=read(PREP/'HUMAN_EVALUATION_COHORT30.json')['business_ids']; assert len(cohort)==len(set(cohort))==30
    legacy=list(csv.DictReader((PREP/'PRIVATE_HUMAN_DESIGN_KEY.csv').open(encoding='utf8')))
    key=[r for r in legacy if r['annotator'] in RATERS]; assert len(key)==240
    sources={}; reviews=0; photos={}
    for b in cohort:
        p=SOURCE/'SOURCES'/f'{b}.json'; s=read(p)
        preserve(p,ws/'SOURCES'/p.name); assert s['split']=='test' and s['business_id']==b
        assert s['selected_review_ids']==[r['review_id'] for r in s['reviews']]
        assert all(r['business_id']==b and 'date' in r and 'stars' in r for r in s['reviews'])
        assert hashlib.sha256(s['selected_review_block'].encode()).hexdigest()==s['review_block_sha256']
        sources[b]={'file':'SOURCES/'+p.name,'canonical_sha256':digest(s),'file_sha256':file_hash(p)};reviews+=len(s['reviews'])
        for photo in s['photos']:
            p=ROOT/photo['path'];h=file_hash(p)
            assert h==photo['sha256']==photo['original_source_sha256'] and p.stat().st_size==photo['bytes']
            if photo['photo_id'] not in photos: preserve(p,ws/'PHOTOS'/p.name)
            photos[photo['photo_id']]=h
    native={};receipts=[]
    for c in ['RO','U','F','S']:
        p=ROOT/'outputs/stage_k/s1000_test100_phase1_v1/EXECUTION'/c/'raw_predictions.json'
        protected[str(p.relative_to(ROOT))]=file_hash(p)
        for r in read(p):
            if r['business_id'] not in cohort:continue
            assert r['condition']==c and hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
            native[c,r['business_id'],r['task']]=r
            receipts.append({'native_file':str(p.relative_to(ROOT)),'native_file_sha256':file_hash(p),'native_record':r})
    assert len(native)==210
    items=[]
    for r in key:
        b,c=r['business_id'],r['condition'];rv=native[c,b,'review'];vi=native.get((c,b,'visual'))
        items.append({**r,'position':int(r['position']),'review_text':rv['text'],'review_text_sha256':rv['text_sha256'],
            'visual_text':vi['text'] if vi else None,'visual_text_sha256':vi['text_sha256'] if vi else None})
    for rater in RATERS:
        ordered=sorted([(b,c) for b in cohort for c in ['RO','U','F','S']],key=lambda v:digest([42,'presentation',rater,*v]))
        actual=sorted([r for r in items if r['annotator']==rater],key=lambda r:r['position'])
        assert [(r['business_id'],r['condition']) for r in actual]==ordered
        assert all(r['alias']==digest([42,'blind',rater,i,r['business_id'],r['condition']])[:20] for i,r in enumerate(actual,1))
    assert all(native['F',b,'review']['text']==native['S',b,'review']['text'] for b in cohort)
    put(private/'PRIVATE_ITEMS.json',items);put(private/'NATIVE_COMPONENT_RECEIPTS.json',receipts)
    put(ws/'COHORT30.json',read(PREP/'HUMAN_EVALUATION_COHORT30.json'))
    # Parse the historical table verbatim; preserve its exact anchors.
    table=[line.split('|')[1:-1] for line in (PREP/'HUMAN_EVALUATION_PROTOCOL.md').read_text(encoding='utf8').splitlines() if line.startswith('| ')][2:]
    assert len(table)==6
    rubric={'scale':'1-5, higher is better; 2 and 4 are intermediate anchors','criteria':[]}
    for index,(criterion,cells) in enumerate(zip(CRITERIA,table)):
        name,a1,a3,a5=[v.strip() for v in cells]
        scope='Review Consensus only' if index in [0,1,4] else 'All available generated components' if index in [2,3] else 'Visual Evidence against original recipient photos; absent component N/A'
        rubric['criteria'].append({'id':criterion,'name':name,'scope':scope,'anchors':{'1':a1,'3':a3,'5':a5}})
    put(ws/'RUBRIC.json',rubric);(ws/'TWO_RATER_PROTOCOL_AMENDMENT.md').write_text(AMENDMENT,encoding='utf8')
    (output/'README.md').write_text(README,encoding='utf8')
    with (ws/'BLANK_RATING_EXPORT_240.csv').open('w',encoding='utf8',newline='') as f:
        fields=['alias','annotator','position','business_id',*CRITERIA,'evidence_notes','status','submitted_at_utc'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in items:w.writerow({k:r[k] for k in ['alias','annotator','position','business_id']}|{CRITERIA[-1]:'N/A' if r['condition']=='RO' else '', 'status':'NOT_STARTED'})
    for p in [private/'PRIVATE_ITEMS.json',ws/'RUBRIC.json',ws/'COHORT30.json',ws/'TWO_RATER_PROTOCOL_AMENDMENT.md',private/'NATIVE_COMPONENT_RECEIPTS.json']:
        bindings[p.relative_to(ws).as_posix()]=file_hash(p)
    put(ws/'CONFIG.json',{'format':'stage_n_test30_two_raters_v1','synthetic':False,'expected_records':240,'expected_businesses':30,'sources':sources,'file_bindings':bindings})
    put(output/'LIVE_RELEASE_TEMPLATE.json',{'supervisor_approved':False,'researcher_live_scoring_approved':False,'approved_at_utc':'','rater_ids':RATERS,
        'amendment_sha256':file_hash(ws/'TWO_RATER_PROTOCOL_AMENDMENT.md'),'config_sha256':file_hash(ws/'CONFIG.json')})
    report={'status':'PASS','scope':'Preparation integrity only, not human evaluation','businesses':30,'raters':RATERS,'items_per_rater':120,'expected_records':240,
        'native_components':len(native),'selected_reviews':reviews,'individual_original_photos':len(photos),'historical_orders_and_aliases_reused':True,
        'blank_numeric_cells':1380,'structural_na_cells':60,'real_accounts_created':0,'real_rating_records':0,'supervisor_approval':'PENDING','live_scoring':'LOCKED',
        'checks':{'frozen_cohort':'PASS','native_text_hashes':'PASS','review_and_photo_bindings':'PASS','recipient_evidence_only':'PASS','deterministic_original_A01_A02_orders':'PASS',
            'identical_F_S_Review_retained':'PASS','six_original_anchors':'PASS','historical_files_preserved':'PASS'},
        'protected_input_sha256':protected}
    put(output/'REPORTS/PREPARATION_INTEGRITY.json',report)
    assert all(file_hash(ROOT/p)==h for p,h in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_input_sha256'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);prepare(p.parse_args().output)
