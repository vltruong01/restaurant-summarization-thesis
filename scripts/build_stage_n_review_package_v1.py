"""Package already prepared Stage N evidence; never runs a real study or metrics."""
import csv
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from s1000_final_eval_core_v1 import read, file_hash
from audit_stage_n_preparation_v1 import audit
from prepare_stage_n_test30_v1 import README

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'outputs/stage_n/test30_two_rater_preparation_v1'

def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def text(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value,encoding='utf8')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-validated-tests',action='store_true');args=parser.parse_args()
    ws=ROOT/'WORKSPACE';reports=ROOT/'REPORTS';protected={}
    if args.reuse_validated_tests:
        prior={r['path']:r['sha256'] for r in read(ROOT/'PACKAGE_MANIFEST.json')['files']}
        # Only document/packager edits may reuse the already recorded code tests.
        for p in (PROJECT/'scripts').glob('*stage_n*'):
            if p.is_file() and p.name!='build_stage_n_review_package_v1.py':assert file_hash(p)==prior['scripts/'+p.name]
        for p in (PROJECT/'scripts/stage_n_human_web_v1').glob('*'):assert file_hash(p)==prior['scripts/stage_n_human_web_v1/'+p.name]
    for folder in ['outputs/stage_l/approved_test100_final_export_v1','outputs/stage_m/human_primary_scored_v1']:
        for p in (PROJECT/folder).rglob('*'):
            if p.is_file():protected[p.relative_to(PROJECT).as_posix()]=file_hash(p)
    protected.update(read(reports/'PREPARATION_INTEGRITY.json')['protected_input_sha256'])
    start=dt.datetime.now(dt.timezone.utc).isoformat()
    scripts=ROOT/'scripts';scripts.mkdir(exist_ok=True)
    names=['stage_n_human_evaluation_v1.py','prepare_stage_n_test30_v1.py','analyze_stage_n_two_raters_v1.py','audit_stage_n_preparation_v1.py','verify_stage_n_package_v1.py','test_stage_n_human_evaluation_v1.py','test_stage_n_ui_v1.cjs','stage_n_synthetic_browser_demo_v1.py','build_stage_n_review_package_v1.py','stage_l_reference_app_v1.py','s1000_final_eval_core_v1.py']
    for name in names:shutil.copyfile(PROJECT/'scripts'/name,scripts/name)
    shutil.copytree(PROJECT/'scripts/stage_n_human_web_v1',scripts/'stage_n_human_web_v1',dirs_exist_ok=True)
    (ROOT/'DOCUMENTATION').mkdir(exist_ok=True)
    shutil.copyfile(PROJECT/'thesis_sdd_docs/STAGE_N_TEST30_TWO_RATER_PREPARATION.md',ROOT/'DOCUMENTATION/STAGE_N_TEST30_TWO_RATER_PREPARATION.md')
    # Correct a compatibility statement: hashlib.file_digest requires Python 3.11.
    readme=README.replace('Python 3.10+','Python 3.11+')
    readme+='''
## Verification scope and limitations

The real cohort contains 842 selected reviews and 343 individual original JPEGs.
There are 210 native components (30 RO Review + 3 x 60 Review/Visual), reused
across both raters as 240 item assignments. Exactly ONE of the two zero-F-caption
TEST100 cases is in the frozen TEST30; no extra case was added or excluded.
See REPORTS/TECHNICAL_VALIDATION.md and FROZEN_BINDING_AUDIT.json for actual checks.

The Windows Chrome smoke used a temporary SYNTHETIC-only workspace: login,
recipient review/photo display, empty applicable Visual, blank scores, draft
save and reload/resume were observed. Full six-score submission, declaration
approval and coordinator export were tested through the HTTP/backend fixtures,
not all clicked manually in Chrome. It was not a usability study. The temporary
service was stopped; no real study server, accounts or rating DB was created.

After the browser smoke, save controls were hardened to disable inputs/navigation
during an in-flight save. The final frontend state test checks this behavior,
duplicate-click suppression, version conflicts/recovery and submitted locking.
No browser screenshot containing desktop/account details is included.

Technical test dependencies: Pillow and numpy; app itself uses the Python 3.11+
standard library. Analysis uses numpy. Node.js is needed only for the optional
frontend state test. Existing local versions are recorded in REPRODUCIBILITY.json.
No package installation was performed. The preparer/detailed project audit need
the original repository inputs; the portable package verifier and app work from
this extracted directory. Do not rerun preparation over this frozen workspace.

Stage L/M completion receipts are coordinator context, not evidence of participant
independence. They are never served to raters. Account creation and release are
deliberate future coordinator actions after approvals. Real participants then
declare truthfully; the coordinator must assess exposure and conflicts.
'''
    text(ROOT/'README.md',readme)
    rubric=read(ws/'RUBRIC.json');guide='''# Participant scoring instructions

Read selected original reviews and recipient photos for each business before
scoring its anonymous output. Work independently; do not discuss scores or view
other raters' progress, condition maps, references, Teacher targets, automatic
results or model rankings. Use only the displayed recipient evidence. You may
open an individual photo in its own tab to inspect it at original resolution.

The same business recurs in your independently randomized item order. Identical
Review text may recur. Score the displayed item independently; do not infer which
model should be better. Lack of a Visual component may make structural masking
imperfect, and this limitation is recorded rather than hidden.

## Five-point rubric (original anchors)

All applicable scores are 1-5, higher is better. Scores 2 and 4 lie between the
printed anchors. No default score is selected. The scopes below are mandatory.

'''
    for c in rubric['criteria']:
        guide+='### '+c['name']+'\n\nScope: '+c['scope']+'.\n\n'
        guide+='\n'.join('- **'+k+'**: '+c['anchors'][k] for k in ['1','3','5'])+'\n\n'
    guide+='''## Special cases and evidence notes

A genuinely absent Visual component is structural N/A, not a poor score. An empty
but applicable Visual is still a human quality judgment, not an unanswered rating.
No caption or output repair is allowed. Photos are the ORIGINAL RECIPIENT photos;
donor photos never justify claims about the recipient. Do not assume every photo
must be described, or judge missing source captions as missing human annotation.

For serious grounding defects, enter an exact output span, recipient review/photo
IDs, an explanation, and whether the finding is uncertain or confirmed. General
readability comments are optional. Do not label an uncertain inference as a
confirmed hallucination. Do not use low/high ratings to communicate a desired
condition ranking. Save incomplete drafts, resume later, and Submit only when all
applicable criteria have been genuinely judged. Submit locks that item, preventing
accidental overwrites. Contact the coordinator about a genuine post-submit error;
there is no silent edit/reopen endpoint. Any later correction needs an explicitly
authorized, traceable amendment rather than manual database editing.

Consent, task familiarity, previous exposure, Stage L involvement and conflicts
must be declared truthfully. A coordinator reviews eligibility before output
access. Software cannot certify identity, independence or actual source inspection.
Do not put names, contact details or passwords in free-text comments.
'''
    text(ROOT/'RATER_SCORING_INSTRUCTIONS.md',guide)
    text(ROOT/'STATISTICAL_ANALYSIS_PLAN.md','''# Two-rater analysis prepared before data collection

Status: prospective amendment, supervisor approval pending; no genuine scores
available. The original 3-rater analysis is preserved as historical evidence.
The new analyzer expects all 240 genuine submitted assignments, with exact
alias/rater/business/condition bindings. It does not run on a blank CSV or a
synthetic final export through its production CLI. Missing/duplicate/misaligned
records and invalid scores are rejected, not dropped or imputed.

Descriptive output: 24 condition/criterion cells, with RO Visual explicitly N/A;
23 applicable cells each contain 60 scores (30 businesses x 2 raters). Report
means, medians, category distributions, per-rater summaries, valid/missing counts,
and 60 structural N/A entries overall. Missing applicable scores are zero in a
complete accepted panel, not silently omitted. Preserve original participant
notes and evidence spans verbatim for later qualitative interpretation; no
model-generated conclusions or automatic winner are produced.

Primary agreement: quadratic-weighted Cohen kappa per condition/criterion,
23 cells, 30 paired units each, categories fixed to 1..5. The formula is
1 - observed squared disagreement / expected squared disagreement from the
two marginal frequency distributions; division by 16 cancels in the ratio.
Zero expected disagreement is null with a reason, not invented perfect agreement.
Missing complete pairs are not imputed. Supplementary pooled kappa is provided
per criterion, clearly labelled as mixing conditions (not additional independent
businesses). No optional ordinal alpha is used in this revision.
Primary implementation reference:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
The code is independently implemented and tested against hand-computable cases;
it does not require sklearn at runtime. Agreement does not prove semantic accuracy.

Exploratory F-RO/F-U/F-S: 17 applicable comparisons (5 Review/all-component
criteria x 3 contrasts, plus Visual F-U/F-S). Average the TWO rater scores within
each business, then sample the 30 matched businesses with replacement, preserving
condition pairing. Retain original PCG64 seed 20261028, 20,000 bootstrap draws,
percentile 95% CIs, and 50,000 two-sided Monte Carlo sign-flip draws with +1
correction. Raw exploratory p values are not prespecified confirmatory tests or
a multiplicity-adjusted winner claim. Sign-flip assumptions are not guaranteed
by the fixed shuffled permutation; shared donors and identical F/S Review output
also constrain interpretation. No three-rater or 240-row pseudoreplication.

The two-rater amendment changes agreement and within-business averaging only;
it preserves the frozen cohort, conditions, scales, rubric scopes and original
business bootstrap settings. The older analyzer's automatic receipt field
`conditions == 4` is incompatible with the actual Stage M receipt, which records
`condition_records == 400` and `components == 700`; the NEW analyzer validates
the actual schema and exact receipt binding, without changing the old script.

Two-rater agreement has limited precision; no reliability CI or ordinal alpha
is claimed. Optional follow-up analyses require explicit review. Exported
declarations and recorded approvals cannot independently authenticate humans.
''')
    text(ROOT/'LIMITATIONS_AND_REMAINING_ACTIONS.md','''# Review boundaries and remaining human actions

- Supervisor approval of the documented 3-to-2-rater amendment is PENDING.
- Researcher authorization to start live scoring is PENDING. The package has no
  LIVE_RELEASE.json; the template contains false approvals and an empty time.
- Recruit exactly two real humans A01/A02. Prefer people who were not Stage L
  authors/checkers and have not seen outputs, identities or automatic scores.
  Record exposure/conflicts honestly and document the coordinator's eligibility
  decision. Declarations cannot prove independence or identity.
- Accounts are planned and isolated in software, but no real user/password,
  declaration, eligibility decision, rating or submission was created here.
  Provision passwords after review using the private getpass setup commands.
- Private condition keys and prior source/model receipts are for the coordinator
  only. Distribute the local interface, not this entire review ZIP, to raters.
  Local OS/filesystem access is trusted; raters with direct access to coordinator
  files can defeat blinding. Use separate browser/OS profiles on shared machines.
- HTTP sessions use random HttpOnly/SameSite=Strict cookies, CSRF checks, same
  Origin/Host enforcement, eight-hour expiry and login throttling. The listener
  binds only 127.0.0.1. It is not designed for public hosting or remote recruitment.
- RO structural absence, repeated F/S Review text, output budgets, prior model
  ancestry, frozen shuffled dependence and known cross-split source-image reuse
  remain design limitations. No added causal claim or checkpoint selection.
- An empty applicable output remains quality evidence, not missing human work;
  the single zero-F-caption case in TEST30 and original defects are retained.
- Real analysis requires all 240 genuine submissions. All recorded numerical
  test scores existed only in temporary SYNTHETIC_* workspaces. Test logs never
  constitute human results, independent inspection, usability validation, or a
  third rater. Browser testing was bounded, not exhaustive.
- No Stage K/L/M artifact was modified, no automatic evaluation rerun, and no
  GPU, paid API, training, reference generation or new research stage started.

STOP: review this package and the amendment before provisioning/opening scoring.
''')
    text(ROOT/'AUTHENTICATION_AND_SETUP.md','''# Local participant setup (after researcher/supervisor approval)

Follow the account and release commands in README.md. Exactly A01/A02 have rater
roles and COORDINATOR has coordinator permissions. Passwords require 12+ characters
and are salted PBKDF2-HMAC-SHA256 (200,000 iterations); setup uses getpass twice,
not plaintext command arguments. Existing users are never overwritten by setup.
No credentials or private state are bundled. Never share coordinator credentials.

The startup script binds http://127.0.0.1:8766 only. A real rater cannot log in
without a valid prospective release. Starting a locked server does not authorize
scoring. The release must bind CONFIG.json and TWO_RATER_PROTOCOL_AMENDMENT.md,
actual approval time and exactly [A01,A02]. A false/missing/mismatched release
continues to block real access. File-level coordinator access is trusted.

After release each actual rater completes declarations. In the coordinator tab,
review declarations and record ELIGIBLE or INELIGIBLE with a reason and exposure
limitations. The app will not silently infer independence or approve eligibility.
Raters then work in separate sessions, in their own frozen order. Progress and
rating requests are owner-scoped. Submit rejects stale versions and duplicate
submissions; valid drafts and atomic event history are preserved in SQLite.

For recovery, first use the CLI backup command (SQLite consistent backup), then
verify the event journal using the CLI verify command. Do not reset or overwrite
submitted ratings. Any real post-submit amendment requires a separately approved
traceable procedure; no automatic reopen or score change is implemented here.
Local recovery drafts are account-and-alias keyed. Log out when finished; use
separate OS/browser profiles when sharing a device. Never copy PRIVATE_STATE or
genuine participant comments into a public package without appropriate review.
''')
    text(ROOT/'LAUNCH_LOCAL_REVIEW.cmd','@echo off\ncd /d "%~dp0"\npython scripts/stage_n_human_evaluation_v1.py serve --workspace WORKSPACE\npause\n')
    text(ROOT/'TEST_DEPENDENCIES.txt','Python >=3.11\nApp: standard library only\nAnalysis: numpy\nSynthetic tests: numpy, Pillow\nOptional frontend state tests: Node.js\nNo dependency installation was performed for this preparation.\n')
    key=read(ws/'COORDINATOR_ONLY/PRIVATE_ITEMS.json')
    with (ROOT/'ANONYMIZED_PRESENTATION_AUDIT.csv').open('w',encoding='utf8',newline='') as f:
        fields=['annotator','position','alias','business_id','review_text_sha256','visual_text_sha256','visual_applicable'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in key:writer.writerow({k:r[k] for k in fields[:-1]}|{'visual_applicable':r['visual_text'] is not None})
    put(reports/'FROZEN_BINDING_AUDIT.json',audit(ROOT,PROJECT))
    # Genuine workspace is never instantiated: verify its prepared state read-only.
    assert not (ws/'LIVE_RELEASE.json').exists() and not (ws/'PRIVATE_STATE').exists()
    run=[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s',str(scripts),'-p','test_stage_n_human_evaluation_v1.py','-v']
    if not args.reuse_validated_tests:
        unit=subprocess.run(run,cwd=ROOT,encoding='utf8',capture_output=True);text(reports/'SYNTHETIC_UNIT_HTTP_TESTS.txt',unit.stdout+unit.stderr);assert unit.returncode==0
        js=subprocess.run(['node',str(scripts/'test_stage_n_ui_v1.cjs')],cwd=ROOT,encoding='utf8',capture_output=True);text(reports/'SYNTHETIC_FRONTEND_STATE_TESTS.txt',js.stdout+js.stderr);assert js.returncode==0
    else:
        from types import SimpleNamespace
        assert 'Ran 17 tests' in (reports/'SYNTHETIC_UNIT_HTTP_TESTS.txt').read_text(encoding='utf8')
        assert (reports/'SYNTHETIC_UNIT_HTTP_TESTS.txt').read_text(encoding='utf8').rstrip().endswith('OK')
        assert json.loads((reports/'SYNTHETIC_FRONTEND_STATE_TESTS.txt').read_text(encoding='utf8'))['status']=='PASS'
        unit=SimpleNamespace(returncode=0);js=SimpleNamespace(returncode=0)
    check=subprocess.run(['node','--check',str(scripts/'stage_n_human_web_v1/app.js')],capture_output=True,encoding='utf8');assert check.returncode==0
    put(reports/'BROWSER_SMOKE_OBSERVATION.json',{'status':'PASS_BOUNDED_SYNTHETIC_SMOKE','browser':'Existing Windows Google Chrome','automation':'computer-use skill; native sky API',
        'url':'http://127.0.0.1:8767','fixture':'Separate temporary SYNTHETIC_BUSINESS_000 only; generated one solid-color test JPEG; no genuine TEST inputs',
        'observed_checks':['Synthetic-only banner','A01 temporary fixture login','Independent own-item progress 0/4','Selected review ID, stars and date','Recipient photo displayed with its ID',
            'Anonymous original fixture output','Empty applicable Visual explicitly retained','All six scores initially Choose/blank','Synthetic note saved as DRAFT','Page reload retains session and DRAFT','Reopen restores exact synthetic note; all six scores remain blank'],
        'not_manually_browser_tested':['Complete six-score submission and locking','Coordinator eligibility and export screens','All120 items per real rater','Multi-user usability or independent human participation'],
        'additional_final_code_checks':'Final frontend state test covers in-flight control disabling, duplicate-click suppression, conflict recovery and submitted locking; backend HTTP fixtures cover auth/CSRF/owner isolation',
        'real_study_started':False,'screenshots_included':False,'passwords_included':False,'temporary_service_stopped':True})
    put(reports/'REPRODUCIBILITY.json',{'prepared_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
        'numpy':importlib.metadata.version('numpy'),'Pillow':importlib.metadata.version('Pillow'),
        'node':subprocess.run(['node','--version'],capture_output=True,text=True).stdout.strip(),
        'backend_tests':17,'backend_exit_code':unit.returncode,'frontend_state_exit_code':js.returncode,'javascript_syntax_exit_code':check.returncode,
        'test_data':'Isolated synthetic fixture only; no numerical human results produced','scoring_release':'LOCKED; no actual accounts/database created',
        'source_counts':{'businesses':30,'reviews':842,'individual_photos':343,'native_components':210,'planned_raters':2,'expected_rating_records':240}})
    # Technical checks do not authorize live release or claim actual human independence.
    text(reports/'TECHNICAL_VALIDATION.md','''# Stage N technical validation

PASS means the named technical check was actually performed, not that humans have
participated or judged model quality. No real accounts, ratings or release exist.

| Requirement | Actual evidence | Result |
| --- | --- | --- |
| 1. Exact TEST30 | Cohort recomputed from original TEST100 seeded ranking; exact original IDs | PASS |
| 2. 120 each / 240 | Balanced 30 x 4 per A01/A02; exact original assignment subset; blank cells checked | PASS |
| 3. Prediction/recipient map | 210 full native records including token IDs equal original K files; hashes match M receipt | PASS |
| 4. Reviews/photos | 842 selected reviews, original IDs/stars/dates; 343 individual original JPEG hashes; source-manifest canonical binding | PASS |
| 5. Independent orders | Both original seed42 orders/aliases reproduced; historical A03 retained only in old design | PASS |
| 6. Hidden model IDs | Public API allowlisted payload; no coordinator map/export for raters; native Chrome anonymous display | PASS (structural limits remain) |
| 7. Six criteria/1-5 | Original anchors copied verbatim; invalid/missing/boolean scores rejected on submit; no prefills | PASS |
| 8. RO Visual N/A | Blank schema 60 structural N/A; backend rejects numeric RO Visual | PASS |
| 9. Empty outputs | Original empty applicable component retained; still requires human score, zero-F case included | PASS |
| 10. Save/resume/submit | Temp DB reopen, submit lock, stale version + simultaneous writes; Chrome draft save/reload; frontend in-flight guards | PASS (bounded tests) |
| 11. Account separation | Hash/idempotence, A03 denied, other-rater item access/writes and coordinator exports denied | PASS |
| 12. Export/decoding | Isolated full synthetic final export; incomplete final denied; mappings decode correctly | PASS |
| 13. K/L/M unchanged | Preparation input hashes plus final before/after frozen artifact snapshot | PASS |

17 backend/unit/HTTP tests passed; see SYNTHETIC_UNIT_HTTP_TESTS.txt. Frontend state
tests and JS syntax check passed. The latter use fake DOM/network, not a browser.
Actual Chrome observation scope is separately listed in BROWSER_SMOKE_OBSERVATION.
Portable ZIP extraction/member-hash verification is separately recorded after build.

Quadratic kappa is tested on known perfect/reversed/nontrivial agreement and
undefined constant-category cases; complete240 synthetic analysis checks the
23 applicable agreement cells, 17 comparisons, 30-business pairing and 60 N/A.
Tests never enter real workspace storage. No genuine human statistic is reported.

LIMITATIONS: no full real-rater usability run, independent identity authentication,
supervisor approval, or proof of source inspection. Local trusted coordinator/OS
access can reveal mappings. Blank forms do not count as ratings. No live study was
opened. Inspect README and the prospective amendment before authorizing collection.
''')
    changed=[p for p,h in protected.items() if file_hash(PROJECT/p)!=h];assert not changed,changed
    put(reports/'FROZEN_K_L_M_UNCHANGED.json',{'status':'PASS','snapshot_started_at_utc':start,'verified_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'Initial preparation hashes retained; additional immutable final Stage L exports/approval ledger and Stage M results hashed before and after bounded final package checks',
        'files':protected,'changed_files':changed,'stage_k_l_m_rerun':False})
    assert not (ws/'PRIVATE_STATE').exists() and not (ws/'LIVE_RELEASE.json').exists()
    # Only this fresh Stage N folder is packaged; no private state/credentials.
    excluded={'PACKAGE_MANIFEST.json','PACKAGE_INTEGRITY_RECEIPT.json'}
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in excluded and p.suffix not in ['.zip','.sha256','.pyc'] and '__pycache__' not in p.parts]
    assert all('PRIVATE_STATE' not in p.parts and p.suffix not in ['.sqlite3','.db'] for p in files)
    manifest={'format':'stage_n_review_package_manifest_v1','scope':'Coordinator-only review; no human scoring release','files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(files)]}
    put(ROOT/'PACKAGE_MANIFEST.json',manifest)
    # Verify copied code from an actually extracted ZIP, not just the source tree.
    archive=ROOT/'STAGE_N_HUMAN_EVALUATION_TEST30_READY.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in [*sorted(files),ROOT/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    with tempfile.TemporaryDirectory(prefix='stage_n_verified_extraction_') as tmp:
        extract=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;names=z.namelist();assert len(names)==len(set(names))
            assert all((extract/n).resolve().is_relative_to(extract.resolve()) for n in names)
            z.extractall(extract)
        for r in manifest['files']:assert file_hash(extract/r['path'])==r['sha256']
        portable=subprocess.run([sys.executable,'-X','utf8','-B',str(extract/'scripts/verify_stage_n_package_v1.py'),'--root',str(extract)],cwd=extract,encoding='utf8',capture_output=True)
        assert portable.returncode==0,portable.stdout+portable.stderr
        no_db=not (extract/'WORKSPACE/PRIVATE_STATE').exists();assert no_db
    receipt={'status':'PASS','zip_file':archive.name,'zip_bytes':archive.stat().st_size,'zip_sha256':file_hash(archive),'zip_members':len(names),
        'crc_test':'PASS','reopened_extracted_every_manifest_member_sha256':'PASS','portable_verifier':json.loads(portable.stdout),
        'no_private_state_or_live_release_in_package':True,'real_human_scoring_started':False,
        'manifest_sha256':file_hash(ROOT/'PACKAGE_MANIFEST.json'),'frozen_K_L_M_after_packaging_unchanged':all(file_hash(PROJECT/p)==h for p,h in protected.items())}
    assert receipt['frozen_K_L_M_after_packaging_unchanged']
    put(ROOT/'PACKAGE_INTEGRITY_RECEIPT.json',receipt)
    text(ROOT/(archive.name+'.sha256'),receipt['zip_sha256']+'  '+archive.name+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
