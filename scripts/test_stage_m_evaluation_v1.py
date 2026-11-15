"""Synthetic technical tests only; genuine human drafts are never touched."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import io
from unittest.mock import patch
from stage_m_evaluation_v1 import score_components,summarize,METRICS,strip_header,human_references,write_tables
from stage_m_teacher_references_v1 import validate_task,completed_response,response_output,execute,load_teacher_references
from s1000_final_eval_core_v1 import digest,text_hash,write,read,file_hash

class Tokenizer:
    def encode(self,text,**kwargs):return [0]+list(range(len(text.split())))+[2]
class FakeScorer:
    _tokenizer=Tokenizer()
    def __init__(self):self.pairs=[]
    def score(self,candidates,references,**kwargs):
        self.pairs.extend(zip(candidates,references))
        vals=[1. if a==b else .2 for a,b in zip(candidates,references)]
        return vals,vals,vals

def frame(n=4):
    ids=[f'SYNTHETIC_{i}' for i in range(n)];raw=[];calls={};refs={}
    for b in ids:refs[b]={'review_consensus':'Soup good.','visual_evidence':'Photo '+b+'.','reference_sha256':digest(b)}
    for c in ['RO','U','F','S']:
        for i,b in enumerate(ids):
            for t in (['review'] if c=='RO' else ['review','visual']):
                text='Soup good.' if t=='review' else 'Photo '+(ids[(i+1)%n] if c=='S' else b)+'.'
                marker='[REVIEW_CONSENSUS]\n' if t=='review' else '[VISUAL_EVIDENCE]\n'
                r={'condition':c,'business_id':b,'task':t,'text':marker+text,'text_sha256':text_hash(marker+text),'source_sha256':'source','checkpoint_sha256':'adapter','token_ids':[2,0,10,2]}
                raw.append(r);calls[c,b,t]={'source':'SYNTHETIC SOURCE'}
    return ids,raw,calls,refs

class EvaluationTests(unittest.TestCase):
    def test_recipient_reference_not_donor_and_review_reuse(self):
        ids,raw,calls,refs=frame();scorer=FakeScorer();rows,receipt=score_components(raw,calls,refs,scorer,'teacher');by={(r['condition'],r['business_id'],r['task']):r for r in rows}
        self.assertEqual(len(rows),28);self.assertFalse(any(r['condition']=='RO' and r['task']=='visual' for r in rows))
        for b in ids:
            self.assertEqual(by['F',b,'review']['bertscore_f1'],by['S',b,'review']['bertscore_f1'])
            self.assertEqual(by['F',b,'visual']['bertscore_f1'],1.)
            self.assertEqual(by['S',b,'visual']['bertscore_f1'],.2)
            self.assertEqual(by['S',b,'visual']['reference_business_id'],b)
        self.assertEqual(receipt['unique_candidate_reference_pairs'],9)
    def test_all_empty_policies_retain_rows(self):
        ids,raw,calls,refs=frame();raw=raw[:1];raw[0]['text']='[REVIEW_CONSENSUS]\n';refs[ids[0]]['review_consensus']=''
        rows,_=score_components(raw,calls,refs,FakeScorer(),'human')
        self.assertTrue(rows[0]['empty_candidate'] and rows[0]['empty_reference'])
        self.assertTrue(all(rows[0][m]==0 for m in METRICS))
        raw[0]['text']='[REVIEW_CONSENSUS]\nSoup good.';rows,_=score_components(raw,calls,refs,FakeScorer(),'human');self.assertTrue(all(rows[0][m]==0 for m in METRICS))
    def test_headers_not_repaired(self):
        self.assertEqual(strip_header('bad [REVIEW_CONSENSUS]\nSoup','review'),'bad [REVIEW_CONSENSUS]\nSoup')
        ids,raw,calls,refs=frame();raw[0]['text']='[VISUAL_EVIDENCE]\nForeign';raw[0]['token_ids']=[2]*768
        r=score_components(raw,calls,refs,FakeScorer(),'human')[0][0]
        self.assertTrue(r['malformed_header'] and r['foreign_component_marker'] and r['at_generation_cap'])
    def test_prefix_truncation_is_reported_not_text_rewrite(self):
        ids,raw,calls,refs=frame();raw[0]['text']='[REVIEW_CONSENSUS]\n'+'soup '*600
        scorer=FakeScorer();r=score_components(raw,calls,refs,scorer,'human')[0][0]
        self.assertTrue(r['bert_candidate_prefix_truncated']);self.assertEqual(r['words'],600);self.assertEqual(r['bert_candidate_tokens_untruncated'],602)
    def test_paired_determinism_matrix_and_sensitivity(self):
        ids,raw,calls,refs=frame();rows,_=score_components(raw,calls,refs,FakeScorer(),'human')
        a=summarize(rows,ids,ids[:3],[ids[0]],bootstrap=200,signflips=200);self.assertEqual(a,summarize(rows,ids,ids[:3],[ids[0]],bootstrap=200,signflips=200))
        self.assertEqual(len(a[0]),7);self.assertEqual(len(a[1]),24);self.assertEqual(len(a[2]),24)
        self.assertEqual(sum('holm_p_four_prespecified_tests' in x for x in a[1]),4)
        self.assertTrue(all(x['n']==4 for x in a[1]));self.assertTrue(all(x['n']==3 for x in a[2]))
        with tempfile.TemporaryDirectory() as d:
            write_tables(Path(d),a[0],a[1],'human');self.assertTrue((Path(d)/'AUTOMATIC_METRIC_TABLES.tex').is_file())
    def test_incomplete_human_freeze_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/'REFERENCE_FREEZE.json',{'human_written_and_second_checked':True,'businesses':99})
            with self.assertRaisesRegex(ValueError,'Full genuine'):human_references(Path(d),Path(d))
    def test_full_stage_l_export_compatible_and_ai_provenance_rejected(self):
        from test_stage_l_reference_workflow_v1 import fixture,draft_for,AUTHOR,CHECKER,AUTHOR_ATTEST,CHECKER_ATTEST
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);store=fixture(d/'synthetic_workflow',100)
            for b in store.frame:
                draft=draft_for(store,b);store.apply(b,AUTHOR,'submit',{'expected_version':0,'draft':draft,'attestations':AUTHOR_ATTEST})
                store.apply(b,CHECKER,'approve',{'expected_version':1,'attestations':CHECKER_ATTEST})
            export=d/'export';store.export(export,final=True)
            prep=d/'prep';shutil.copytree(store.package,prep/'TEST100_REFERENCE_ANNOTATION_PACKAGE')
            refs,report=human_references(prep,export);self.assertEqual(len(refs),100);self.assertFalse(report['errors'])
            original=read(export/'APPROVED_REFERENCES.json');freeze=read(export/'REFERENCE_FREEZE.json')
            (export/'APPROVED_REFERENCES.json').write_text(json.dumps(original+[original[0]]),encoding='utf8');freeze['reference_file_sha256']=file_hash(export/'APPROVED_REFERENCES.json');(export/'REFERENCE_FREEZE.json').write_text(json.dumps(freeze),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'duplicate aggregate'):human_references(prep,export)
            (export/'APPROVED_REFERENCES.json').write_text(json.dumps(original),encoding='utf8');freeze['reference_file_sha256']=file_hash(export/'APPROVED_REFERENCES.json');(export/'REFERENCE_FREEZE.json').write_text(json.dumps(freeze),encoding='utf8')
            b=next(iter(store.frame));p=export/'ANNOTATIONS'/f'{b}.json';r=read(p);r['origin']='AI_TEACHER';p.write_text(json.dumps(r),encoding='utf8')
            agg=read(export/'APPROVED_REFERENCES.json')
            for x in agg:
                if x['business_id']==b:x['origin']='AI_TEACHER'
            (export/'APPROVED_REFERENCES.json').write_text(json.dumps(agg),encoding='utf8');freeze=read(export/'REFERENCE_FREEZE.json');freeze['reference_file_sha256']=file_hash(export/'APPROVED_REFERENCES.json');(export/'REFERENCE_FREEZE.json').write_text(json.dumps(freeze),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Nonhuman'):human_references(prep,export)

class TeacherTests(unittest.TestCase):
    source={'reviews':[{'review_id':'r1','text':'Soup is good but service slow.'}],'photos':[{'photo_id':'p1'}]}
    def review(self):return {'text':'Soup is good.','claims':[{'output_span':'Soup is good','evidence':[{'review_id':'r1','quote':'Soup is good'}]}]}
    def test_exact_evidence_and_attribution_ids(self):
        value=self.review();validate_task(value,'review',self.source)
        value['claims'][0]['evidence'][0]['review_id']='other'
        with self.assertRaises(ValueError):validate_task(value,'review',self.source)
        value=self.review();value['claims'][0]['evidence'][0]['quote']='Soup excellent'
        with self.assertRaises(ValueError):validate_task(value,'review',self.source)
    def test_visual_requires_original_photo_ids(self):
        v={'text':'A table.','claims':[{'output_span':'A table','evidence':[{'photo_id':'p1','direct_image_viewed':True}]}]};validate_task(v,'visual',self.source)
        v['claims'][0]['evidence'][0]['direct_image_viewed']=False
        with self.assertRaises(ValueError):validate_task(v,'visual',self.source)
    def test_partial_or_failed_provider_responses_not_accepted(self):
        for s in [b'',b'data: {"type":"response.incomplete"}\n',b'data: {"type":"response.failed"}\n']:
            with self.assertRaises(ValueError):completed_response(s)
        response={'status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(self.review())}]}]}
        wire=('data: '+json.dumps({'type':'response.completed','response':response})+'\n').encode()
        self.assertEqual(response_output(completed_response(wire)),self.review())
        with self.assertRaises(ValueError):completed_response(wire+wire)
    def test_no_paid_approval_no_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write(d/'plan.json',{});write(d/'release.json',{'paid_teacher_generation_approved':False})
            with patch('stage_m_teacher_references_v1.source_frame',return_value=({},'x')),patch('stage_m_teacher_references_v1.validate_plan'),patch('stage_m_teacher_references_v1.credentials') as cred,patch('urllib.request.urlopen') as network:
                with self.assertRaisesRegex(ValueError,'Explicit paid'):execute(d,d/'plan.json',d/'out',d/'release.json')
                network.assert_not_called();cred.assert_not_called();self.assertFalse((d/'out').exists())
    def test_full_synthetic_teacher_execution_and_import_no_real_network(self):
        from stage_m_teacher_references_v1 import ROOT,prepare
        import datetime
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);historical=read(ROOT/'configs/stage_g_teacher_production_d042.json')
            for rel in ['configs/stage_g_teacher_production_d042.json',*historical['prompts'].values()]:
                (d/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,d/rel)
            image=d/'demo.jpg';Image.new('RGB',(4,4),(20,30,40)).save(image);h=file_hash(image)
            sources={f'SYNTHETIC_TEACHER_ONLY_{i}':{'business_id':f'SYNTHETIC_TEACHER_ONLY_{i}','reviews':[{'review_id':f'r{i}','text':'Soup is good.'}],
                'photos':[{'photo_id':f'p{i}','path':'demo.jpg','sha256':h,'original_source_sha256':h,'bytes':image.stat().st_size}]} for i in range(100)}
            counter=[0]
            def fake_urlopen(request,**kwargs):
                counter[0]+=1;body=json.loads(request.data);payload=json.loads(body['input'][1]['content'][0]['text']);b=payload['business_id'];i=int(b.rsplit('_',1)[1]);task=body['text']['format']['name'].rsplit('_',1)[1]
                value={'text':'Soup is good.','claims':[{'output_span':'Soup is good','evidence':[{'review_id':f'r{i}','quote':'Soup is good'}]}]} if task=='review' else {'text':'A table.','claims':[{'output_span':'A table','evidence':[{'photo_id':f'p{i}','direct_image_viewed':True}]}]}
                response={'status':'completed','model':body['model'],'id':f'synthetic_response_{counter[0]}','usage':{'synthetic':True},'output':[{'content':[{'type':'output_text','text':json.dumps(value)}]}]}
                return io.BytesIO(('data: '+json.dumps({'type':'response.completed','response':response})+'\n').encode())
            with patch('stage_m_teacher_references_v1.ROOT',d),patch('stage_m_teacher_references_v1.source_frame',return_value=(sources,'synthetic_manifest')):
                prepare(d,d/'prepared');plan=d/'prepared/TEACHER_PLAN.json'
                release={'paid_teacher_generation_approved':True,'teacher_plan_sha256':file_hash(plan),'maximum_requests':200,'maximum_output_tokens_per_request':8192,
                    'expires_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat(),'synthetic_only':True}
                write(d/'release.json',release)
                with patch('stage_m_teacher_references_v1.credentials',return_value=('https://synthetic.invalid/v1','SYNTHETIC','NOT_A_SECRET')),patch('urllib.request.urlopen',side_effect=fake_urlopen):
                    execute(d,plan,d/'generated',d/'release.json')
                refs,report=load_teacher_references(d,d/'generated');self.assertEqual(len(refs),100);self.assertEqual(counter[0],200);self.assertEqual(report['origin'],'AI_TEACHER')
                done=read(d/'generated/GENERATION_COMPLETE.json');wire=d/'generated'/done['requests'][0]['wire_file'];wire.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'Raw Teacher hash'):load_teacher_references(d,d/'generated')

if __name__=='__main__':unittest.main(verbosity=2)
