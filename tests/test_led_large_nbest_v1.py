"""Synthetic CPU fixtures only; no model construction, inference, optimizer or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import stage_i_led_large_nbest_v1 as m


def fixture(visual=True):
    source='name=TEST_FIXTURE_ONLY\n[REV_1]\nFresh steak. Service quick.\n[REV_2]\nService slow.\n[VISUAL_CONTEXT]\n'
    if visual:source+='[IMG_CAP_1]\nA fish tank.\n'
    case=m.source_case(dict(business_id='SYNTHETIC_NOT_A_BUSINESS',split='train',condition='filtered',
        source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest()),[0,10,2])
    fact=lambda fid,text,aspect,polarity,marker,quote,domain='review':dict(fact_id=fid,domain=domain,text=text,
        salience='distinctive',aspect=aspect,polarity=polarity,evidence=[dict(marker=marker,quote=quote)])
    facts=[fact('f1','Fresh steak','steak','positive','REV_1','Fresh steak.'),
           fact('f2','Service quick','speed','positive','REV_1','Service quick.'),
           fact('f3','Service slow','speed','negative','REV_2','Service slow.')]
    if visual:facts.append(fact('v1','Fish tank','tank',None,'IMG_CAP_1','A fish tank.','visual'))
    inventory=dict(business_id=case['business_id'],source_sha256=case['source_sha256'],assessor_kind='researcher',facts=facts,contrasts=[['f2','f3']])
    return case,inventory


def candidate(cid='c0',full=True,visual=True,bad_visual=False,generic=False):
    review=['Fresh steak.','Service quick.']+(['Service slow.'] if full else [])
    if generic:review=['Food and service are mentioned.']
    image=['A flying fish.'] if bad_visual else (['A fish tank.'] if visual else [])
    text='[REVIEW_CONSENSUS]\n'+' '.join(review)+'\n\n[VISUAL_EVIDENCE]\n'+' '.join(image)
    row=dict(candidate_id=cid,text=text,token_ids=[2,10,2])
    claims=[]
    for sentence in review+image:
        start=text.index(sentence);domain='review' if sentence in review else 'visual'
        if generic:verdict='uncertain';refs=[];ids=[]
        elif sentence=='A flying fish.':verdict='unsupported';refs=[];ids=[]
        else:
            verdict='supported';lookup={'Fresh steak.':('REV_1','f1'),'Service quick.':('REV_1','f2'),
               'Service slow.':('REV_2','f3'),'A fish tank.':('IMG_CAP_1','v1')}
            marker,fid=lookup[sentence];refs=[dict(marker=marker,quote=sentence)];ids=[fid]
        claims.append(dict(claim_id='q'+str(len(claims)),start=start,end=start+len(sentence),domain=domain,
            verdict=verdict,evidence=refs,covered_fact_ids=ids,false_consensus=False,generic=generic,
            redundant_with=None,rationale='Synthetic fixture judgment, not real annotation.'))
    return row,dict(candidate_id=cid,assessor_kind='researcher',atomicity_reviewed=True,claims=claims)


class Contracts(unittest.TestCase):
    def test_generation_only_return_count_changes(self):
        c=dict(num_beams=4,num_return_sequences=1,do_sample=False,length_penalty=1,output_scores=False)
        a,b=m.generation_pair(c);self.assertEqual([k for k in a if a[k]!=b[k]],['num_return_sequences'])
        self.assertEqual(c,a)

    def test_wrong_beam_policy_rejected(self):
        with self.assertRaises(ValueError):m.generation_pair(dict(num_beams=8,num_return_sequences=1,do_sample=False))

    def test_projection_drops_teacher_references_photos(self):
        c,_=fixture();r=dict(c,target='SECRET_TEACHER',approved_references=['SECRET_REF'],photos=['SECRET_IMAGE'])
        projected=m.source_case(r,c['input_ids'])
        packet=m.selection_packet(projected,[candidate()[0]])
        self.assertNotIn('SECRET',json.dumps(packet))
        self.assertNotIn('input_ids',packet)

    def test_extra_case_field_rejected(self):
        c,_=fixture();c['target']='hidden'
        with self.assertRaises(ValueError):m.selection_packet(c,[candidate()[0]])

    def test_wrong_split_rejected(self):
        c,_=fixture();c['split']='test'
        with self.assertRaises(ValueError):m.validate_case(c)

    def test_source_tamper_rejected(self):
        c,_=fixture();c['reviews']['REV_1']='Modified'
        with self.assertRaises(ValueError):m.validate_case(c)

    def test_padding_difference_explicit_not_content_change(self):
        a=dict(text='unchanged',token_ids=[2,10,2],source_sha256='s',checkpoint_sha256='w')
        b=dict(a,token_ids=[2,10,2,1,1]);r=m.compare_top1(a,b,a)
        self.assertTrue(r['historical_content_ids_and_text_exact']);self.assertFalse(r['single_vs_nbest_raw_ids_exact'])

    def test_content_mismatch_stops(self):
        a=dict(text='x',token_ids=[2,10,2]);b=dict(a,token_ids=[2,11,2])
        with self.assertRaises(ValueError):m.compare_top1(a,b)

    def test_historical_mismatch_stops(self):
        a=dict(text='x',token_ids=[2,10,2],source_sha256='s',checkpoint_sha256='w')
        with self.assertRaises(ValueError):m.compare_top1(a,a,dict(a,text='different'))

    def test_semantic_not_inferred_from_lexical_distance(self):
        r=m.lexical_diversity([candidate()[0],candidate('c1')[0]])
        self.assertEqual(r['exact_unique_texts'],1);self.assertIn('UNASSESSED',r['semantic_diversity'])

    def test_supported_balance_beats_omission(self):
        c,inv=fixture();a,aa=candidate(full=False);b,bb=candidate('c1')
        result=m.rerank(c,[a,b],inv,[aa,bb]);self.assertEqual(result['selected_candidate_id'],'c1')
        self.assertEqual(result['selected_text'],b['text']);self.assertFalse(result['rewritten'])

    def test_unsupported_visual_penalty(self):
        c,inv=fixture();a,aa=candidate(full=False);b,bb=candidate('c1',bad_visual=True)
        self.assertEqual(m.rerank(c,[a,b],inv,[aa,bb])['selected_candidate_id'],'c0')

    def test_zero_caption_cannot_select_visual_assertion(self):
        c,inv=fixture(False);a,aa=candidate(bad_visual=True)
        r=m.rerank(c,[a],inv,[aa]);self.assertIsNone(r['selected_candidate_id'])
        self.assertIn('Zero-caption',r['scores'][0]['exclusion_reasons'][0])

    def test_generic_not_rewarded(self):
        c,inv=fixture();a,aa=candidate(generic=True,visual=False)
        self.assertIsNone(m.rerank(c,[a],inv,[aa])['selected_candidate_id'])

    def test_tie_keeps_original_order(self):
        c,inv=fixture();a,aa=candidate();b,bb=candidate('c1')
        self.assertEqual(m.rerank(c,[a,b],inv,[aa,bb])['selected_candidate_id'],'c0')

    def test_wrong_modality_quote_rejected(self):
        c,inv=fixture();a,aa=candidate();aa['claims'][-1]['evidence']=[dict(marker='REV_1',quote='Fresh steak.')]
        with self.assertRaises(ValueError):m.rerank(c,[a],inv,[aa])

    def test_uncovered_claim_rejected(self):
        c,inv=fixture();a,aa=candidate();aa['claims'].pop()
        with self.assertRaises(ValueError):m.rerank(c,[a],inv,[aa])

    def test_foreign_fact_rejected(self):
        c,inv=fixture();a,aa=candidate();aa['claims'][0]['covered_fact_ids']=['D053_FORBIDDEN_REFERENCE']
        with self.assertRaises(ValueError):m.rerank(c,[a],inv,[aa])

    def test_false_consensus_not_supported(self):
        c,inv=fixture();a,aa=candidate();aa['claims'][0]['false_consensus']=True
        with self.assertRaises(ValueError):m.rerank(c,[a],inv,[aa])

    def test_unsupported_quote_rejected(self):
        c,inv=fixture();a,aa=candidate();aa['claims'][0]['evidence'][0]['quote']='Not in source'
        with self.assertRaises(ValueError):m.rerank(c,[a],inv,[aa])

    def test_diversity_gate_requires_real_review(self):
        ids=[str(i) for i in range(12)]
        a=dict(split='train',assessor_kind='researcher',source_only_review_confirmed=True,cases=[dict(business_id=b,
            materially_distinct_profiles=2,oracle_material_gain_without_grounding_regression=i<4,rationale='Synthetic test') for i,b in enumerate(ids)])
        self.assertTrue(m.diversity_gate(a,ids)['status'].startswith('PASS'))
        a['cases'][0]['oracle_material_gain_without_grounding_regression']=False
        self.assertTrue(m.diversity_gate(a,ids)['status'].startswith('STOP'))

    def test_verifier_false_positives_and_negatives(self):
        rows=[dict(business_id='t',claim_key=str(i),gold='supported' if i<20 else 'unsupported',
                   predicted='supported' if i<20 else 'unsupported',rationale='Synthetic audit') for i in range(40)]
        a=dict(split='train',assessor_kind='researcher',independent_of_provisional_verifier=True,claims=rows)
        self.assertEqual(m.verification_audit(a,['t'])['status'],'PASS_TRAIN_ONLY')
        for i in range(5):rows[i]['predicted']='unsupported'
        for i in range(20,25):rows[i]['predicted']='supported'
        r=m.verification_audit(a,['t']);self.assertEqual((r['FP'],r['FN']),(5,5));self.assertTrue(r['status'].startswith('STOP'))

    def test_frozen_training_sample_and_validation_separation(self):
        p=ROOT/'outputs/stage_i/led_large_nbest_reranking_v1'
        a=m.read(p/'TRAIN_SOURCE_ONLY_CASES.json');b=m.read(p/'VALIDATION_SOURCE_ONLY_CASES.json')
        self.assertEqual(len(a),12);self.assertEqual(sum(not x['captions'] for x in a),3)
        self.assertFalse({x['business_id'] for x in a}&{x['business_id'] for x in b})
        for x in a+b:m.validate_case(x)

    def test_runtime_beam_finalization_only_cpu_no_model(self):
        import torch
        from transformers.generation.beam_search import BeamSearchScorer
        def finish(keep):
            scorer=BeamSearchScorer(batch_size=1,num_beams=4,device=torch.device('cpu'),num_beam_hyps_to_keep=keep)
            for i in range(4):scorer._beam_hyps[0].add(torch.tensor([2]+[10+i]*(i+1)),
                sum_logprobs=-float((i+1)**2),generated_len=i+1)
            scorer._done[0]=True
            return scorer.finalize(torch.tensor([[2,10],[2,11],[2,12],[2,13]]),torch.zeros(4),torch.zeros(4,dtype=torch.long),
                torch.zeros(4,dtype=torch.long),max_length=10,pad_token_id=1,eos_token_id=2)['sequences'].tolist()
        a,b=finish(1),finish(4)
        self.assertEqual(m.unpad(a[0]),m.unpad(b[0]));self.assertNotEqual(a[0],b[0]);self.assertEqual(len(b),4)

    def test_generation_config_pinned_roundtrip(self):
        from transformers import GenerationConfig
        expected=m.read(ROOT/'outputs/stage_i/led_large_nbest_reranking_v1/ORIGINAL_GENERATION_CONFIG.json')
        self.assertEqual(GenerationConfig(**expected).to_dict(),expected)

    def test_approval_false_blocks_before_ml_import(self):
        import run_stage_i_led_large_nbest_v1 as runner
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';p.write_text(json.dumps(dict(experiment=m.EXPERIMENT,GPU_inference_approved=False)),encoding='utf-8')
            with self.assertRaises(ValueError):runner.authorization(p,'training_pilot',{},True)

    def test_report_oracle_never_changes_selection(self):
        import tempfile
        import report_led_large_nbest_v1 as report
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);run=folder/'run';run.mkdir();cases=[];records=[];judgments=[];oracle=[];assessment=[]
            for i in range(12):
                c,inventory=fixture();c['business_id']='fixture'+str(i);inventory['business_id']=c['business_id']
                a,aa=candidate(full=False);b,bb=candidate('c1')
                cases.append(c);records.append(dict(business_id=c['business_id'],candidates=[a,b],
                    diversity=m.lexical_diversity([a,b]),candidates_seconds=2,single_seconds=1))
                judgments.append(dict(business_id=c['business_id'],inventory=inventory,annotations=[aa,bb]))
                oracle.append(dict(business_id=c['business_id'],assessor_kind='researcher',researcher_confirmed=True,
                    source_only_review_confirmed=True,acceptable_candidate_ids=['c0'],regression_vs_top1_ids=['c1'],rationale='Deliberately inverted synthetic oracle to test separation'))
                assessment.append(dict(business_id=c['business_id'],materially_distinct_profiles=2,
                    oracle_material_gain_without_grounding_regression=i<4,rationale='Synthetic fixture'))
            def save(path,data):path.write_text(json.dumps(data),encoding='utf-8')
            save(folder/'cases.json',cases);save(run/'CANDIDATES.json',records);save(folder/'judgments.json',judgments)
            save(folder/'oracle.json',oracle);save(folder/'assessment.json',dict(split='train',assessor_kind='researcher',source_only_review_confirmed=True,cases=assessment))
            save(run/'COMPLETION.json',dict(status='COMPLETE_STOP_FOR_TRAIN_SOURCE_REVIEW',phase='training_pilot',
                source_only_case_file_sha256=m.sha(folder/'cases.json'),candidates_sha256=m.sha(run/'CANDIDATES.json'),
                generation_calls=24,returned_sequences=60,seconds=36))
            args=['report','--run',str(run),'--cases',str(folder/'cases.json'),'--judgments',str(folder/'judgments.json'),
                  '--oracle',str(folder/'oracle.json'),'--training-assessment',str(folder/'assessment.json'),'--output',str(folder/'report')]
            with patch.object(sys,'argv',args):report.main()
            results=m.read(folder/'report/PAIRED_RESULTS.json')
            self.assertTrue(all(r['selection']['selected_candidate_id']=='c1' for r in results))
            self.assertEqual(m.read(folder/'report/REPORT.json')['selected_quality_regressions'],12)


if __name__=='__main__':unittest.main(verbosity=2)
