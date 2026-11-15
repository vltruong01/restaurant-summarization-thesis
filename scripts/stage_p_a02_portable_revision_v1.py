"""Isolated A02 revision sidecar; originals remain immutable and unmodified."""
import copy
import json
from pathlib import Path
from stage_p_human_evaluation_v1 import Study, Server
from human_app_core_v1 import digest, file_hash
from human_http_common_v1 import need

class RevisionStudy(Study):
    def __init__(self,workspace,state_dir):
        super().__init__(workspace,preview=False,state_dir=state_dir)
        self.baseline_path=Path(workspace)/'A02_BASELINE_FINAL_45.json'
        self.binding=json.loads((Path(workspace)/'REVISION_BINDING.json').read_text(encoding='utf8'))
        need(file_hash(self.baseline_path)==self.binding['baseline_file_sha256'],'Baseline changed')
        self.baseline=json.loads(self.baseline_path.read_text(encoding='utf8'))
        need(self.baseline['study_id']==self.config['study_id'] and self.baseline['rater']=='A02','Wrong baseline')
        need(self.baseline['records_sha256']==digest(self.baseline['records']),'Baseline record hash')
        self.original={r['alias']:r for r in self.baseline['records']}
        need(len(self.original)==45 and set(self.original)=={r['alias'] for r in self.key if r['annotator']=='A02'},'A02 coverage')

    def login(self,identity,password):
        need(identity=='A02','This portable revision is restricted to A02')
        return super().login(identity,password)

    def add_user(self,identity,role,password):
        need(identity=='A02' and role=='rater','Only A02 can be provisioned here')
        return super().add_user(identity,role,password)

    def rating(self,actor,alias,conn=None):
        self.owned(actor,alias)
        if conn is None:
            with self.connect() as connection:return self.rating(actor,alias,connection)
        row=conn.execute('SELECT version,payload FROM ratings WHERE alias=? AND rater=?',(alias,actor['id'])).fetchone()
        if row:return json.loads(row['payload']),row['version']
        r=self.original[alias]
        # Editable display only: never INSERT inferred drafts or submissions.
        return dict(scores={k:r[k] for k in ['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution']},
            **{k:copy.deepcopy(r[k]) for k in ['evidence_notes','serious_grounding_issue','issue_span','evidence_source_ids','issue_certainty']},status='BASELINE',submitted_at_utc=''),0

    def rater_export(self,actor,final=False):
        need(actor=={'id':'A02','role':'rater'},'A02 only')
        recorded=super().rater_export(actor,False)
        if final:need(all(r['status']=='SUBMITTED' for r in recorded['records']),'Submit saved revision drafts before final export')
        audit=self.audit();need(audit['status']=='PASS','Revision journal invalid')
        updates={r['alias']:r for r in recorded['records']}
        merged=[copy.deepcopy(updates.get(r['alias'],r)) for r in self.baseline['records']]
        fields=['review_faithfulness','aspect_coverage','coherence','non_redundancy','usefulness','visual_grounding_and_contribution','evidence_notes','serious_grounding_issue','issue_span','evidence_source_ids','issue_certainty']
        changes=[dict(alias=r['alias'],business_id=r['business_id'],fields={k:dict(before=self.original[r['alias']][k],after=r[k]) for k in fields if self.original[r['alias']][k]!=r[k]}) for r in merged if any(self.original[r['alias']][k]!=r[k] for k in fields)]
        with self.connect() as db:events=[json.loads(r[0]) for r in db.execute('SELECT event_json FROM events ORDER BY seq')]
        return {**recorded,'format':'stage_p_a02_portable_revision_v1','final':final,'records':merged,'records_sha256':digest(merged),'record_count':45,
            'submitted_count':sum(r['status']=='SUBMITTED' for r in merged),
            'portable_revision':dict(baseline_file_sha256=self.binding['baseline_file_sha256'],baseline_records_sha256=self.baseline['records_sha256'],
                baseline_source=self.binding['baseline_source'],retained_original_aliases=[r['alias'] for r in merged if r['alias'] not in updates],
                newly_saved_or_submitted_aliases=list(updates),newly_submitted_count=sum(r['status']=='SUBMITTED' for r in updates.values()),
                inherited_original_count=45-len(updates),changes=changes,revision_journal=events,journal_audit=audit,
                merge_applied_to_coordinator_database=False,
                note='Untouched records are retained original submissions, not new confirmations. This file is a proposed A02 revision, not an automatic replacement of the coordinator final dataset.')}
