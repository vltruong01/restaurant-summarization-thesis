"""Operational audit amendment for researcher-authorized, archived A02 imports."""
import json
from stage_p_human_evaluation_v1 import Study
from stage_p_revision_provenance_v1 import digest,valid_import_event

class RevisionAwareStudy(Study):
    def audit(self):
        problems=[];previous=None;versions={};states={};counts={'DRAFT':0,'SUBMITTED':0};cache={};imports=0
        with self.connect() as c:
            for row in c.execute('SELECT * FROM events ORDER BY seq'):
                e=json.loads(row['event_json']);a=e['alias']
                if digest(e)!=row['event_hash'] or e['previous_event_sha256']!=previous:problems.append('Event hash chain')
                if e['version']!=versions.get(a,0)+1 or e['rater']!=self.items[a]['annotator']:problems.append('Version/rater binding')
                if a in states:
                    if e['prior_state_sha256']!=digest(states[a]):problems.append('Prior state binding')
                    if states[a]['status']=='SUBMITTED':
                        if not valid_import_event(e,states[a],self.db.parent/'IMPORTS',cache):problems.append('Invalid submitted-state transition')
                        else:imports+=1
                elif e.get('action')=='import_a02_portable_revision':problems.append('Import without prior recorded state')
                previous=row['event_hash'];versions[a]=e['version'];states[a]=e['payload']
            rows=c.execute('SELECT * FROM ratings').fetchall()
            if set(states)!={r['alias'] for r in rows}:problems.append('State/journal item set')
            for row in rows:
                p=json.loads(row['payload']);counts[p['status']]+=1
                if p!=states.get(row['alias']) or row['version']!=versions.get(row['alias']):problems.append('State/journal mismatch')
            users=c.execute('SELECT id,role FROM users').fetchall()
        return dict(status='PASS' if not problems else 'FAIL',errors=problems,stored_rating_records=len(rows),counts=counts,
            expected_records=self.config['expected_records'],live_scoring_released=self.active(),preview_mode=self.preview,
            account_ids_and_roles=[dict(x) for x in users],event_chain_head_sha256=previous,
            participant_identity_and_blindness_independently_verified=False,verified_portable_import_transitions=imports,
            audit_amendment='Submitted-state changes require exact source-file/baseline SHA256 bindings and an explicit import event; ordinary UI submissions remain locked.')
