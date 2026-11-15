"""Validate explicit, source-bound imports without inventing human actions."""
import hashlib,json
from pathlib import Path
CRITERIA=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']
EXTRA=['evidence_notes','serious_grounding_issue','issue_span','evidence_source_ids','issue_certainty','status','submitted_at_utc']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def payload(row):return dict(scores={c:row[c] for c in CRITERIA},**{k:row[k] for k in EXTRA})
def load_bound(root,sha,cache):
    if sha not in cache:
        assert isinstance(sha,str) and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
        raw=(Path(root)/(sha+'.json')).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        cache[sha]=json.loads(raw.decode('utf-8-sig'))
    return cache[sha]
def valid_import_event(event,prior,imports,cache):
    try:
        assert event['action']=='import_a02_portable_revision' and event['rater']=='A02'
        provenance=event['import_provenance'];assert provenance['operator_action']=='researcher_authorized_file_import'
        source=load_bound(imports,provenance['source_file_sha256'],cache)
        baseline=load_bound(imports,provenance['baseline_file_sha256'],cache)
        assert source['format']=='stage_p_a02_portable_revision_v1' and source['final'] is True and source['rater']=='A02'
        assert source['records_sha256']==digest(source['records']) and len(source['records'])==45
        assert baseline['records_sha256']==digest(baseline['records']) and len(baseline['records'])==45
        assert source['portable_revision']['baseline_file_sha256']==provenance['baseline_file_sha256']
        assert source['study_id']==baseline['study_id'] and source['config_sha256']==baseline['config_sha256']
        row=next(r for r in source['records'] if r['alias']==event['alias'])
        old=next(r for r in baseline['records'] if r['alias']==event['alias'])
        assert row['annotator']==old['annotator']=='A02' and row['status']==old['status']=='SUBMITTED'
        assert prior==payload(old) and event['payload']==payload(row)
        assert provenance['source_record_version']==row['version']
        return True
    except (AssertionError,KeyError,ValueError,TypeError,OSError,StopIteration):return False
