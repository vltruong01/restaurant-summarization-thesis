"""TEST-ONLY fixtures: pseudo-label isolation, fixed increments and immutable selection."""
import copy
import unittest
from stage_f_verification_workflow import (CHOICES, canonical_sha, verification_orders,
    active_ids, validate_export, balance_and_expansion)
from report_stage_f_k_distribution import summarize


def fixture():
    cases=[]
    for kind,number,keys in [('image_caption',80,['consistency','relevance']),
                              ('caption_pair',160,['redundancy','broad_topic','image_duplicate'])]:
        for i in range(number):
            cases.append({'case_id':kind+str(i),'business_id':'b'+str(i%4),'kind':kind,
                          'photo_ids':['p'+str(i)],'labels':dict.fromkeys(keys,'')})
    dataset={'dataset_kind':'official_calibration','dataset_sha256':'TEST_ONLY','cases':cases}
    dataset['dataset_sha256']=canonical_sha({k:v for k,v in dataset.items() if k!='dataset_sha256'})
    orders={kind:[c['case_id'] for c in cases if c['kind']==kind] for kind in ['image_caption','caption_pair']}
    order={'orders':orders,'manifest_sha256':'TEST_ONLY'}
    order['manifest_sha256']=canonical_sha({'orders':orders})
    selection={'image_cases':50,'pair_cases':100,'round':0}
    actual={c['case_id']:{'human_verified':False,'human_final_label':{k:CHOICES[k][0] for k in c['labels']}} for c in cases}
    return dataset,order,selection,actual


class VerifiedCalibrationTests(unittest.TestCase):
    cfg={'minimum_each_binary_class':10,'image_expansion_increment':20,'pair_expansion_increment':40}

    def test_unverified_pseudolabels_cannot_supply_counts(self):
        d,o,s,a=fixture();r=balance_and_expansion(d,o,s,a,self.cfg)
        self.assertEqual(r['status'],'WAITING_ACTIVE_HUMAN_VERIFICATION')
        self.assertEqual(r['counts']['consistency']['positive'],0)
        self.assertFalse(r['expanded'])

    def test_expansion_only_after_all_active_verified_and_for_balance(self):
        d,o,s,a=fixture()
        for cid in active_ids(o,s):a[cid]['human_verified']=True
        r=balance_and_expansion(d,o,s,a,self.cfg)
        self.assertEqual(r['selection']['image_cases'],70);self.assertEqual(r['selection']['pair_cases'],140)
        self.assertTrue(r['expanded'])

    def test_balanced_cases_do_not_expand_to_chase_threshold(self):
        d,o,s,a=fixture()
        for i,cid in enumerate(active_ids(o,s)):
            a[cid]['human_verified']=True
            a[cid]['human_final_label']={k:CHOICES[k][i%2] for k in a[cid]['human_final_label']}
        r=balance_and_expansion(d,o,s,a,self.cfg)
        self.assertFalse(r['expanded']);self.assertEqual(r['status'],'BALANCED_VERIFIED_LABELS_READY_FOR_THRESHOLD_SEARCH')

    def test_exhaustion_is_blocker_not_relaxed_class_gate(self):
        d,o,s,a=fixture();s.update(image_cases=80,pair_cases=160)
        for row in a.values():row['human_verified']=True
        r=balance_and_expansion(d,o,s,a,self.cfg)
        self.assertEqual(r['status'],'CLASS_BALANCE_BLOCKER_QUEUE_EXHAUSTED');self.assertFalse(r['expanded'])

    def test_uncertain_not_positive_or_negative(self):
        d,o,s,a=fixture()
        for cid in active_ids(o,s):
            a[cid]['human_verified']=True;a[cid]['human_final_label']=dict.fromkeys(a[cid]['human_final_label'],'uncertain')
        r=balance_and_expansion(d,o,s,a,self.cfg)
        self.assertEqual(r['counts']['consistency'],{'positive':0,'negative':0,'uncertain':50})

    def test_selection_ignores_label_contents(self):
        d,o,s,a=fixture();private={c['case_id']:{'clip_score':i/100,'relevance_margin':i/100,
             'semantic_cosine':i/100,'phash_hamming':i%64} for i,c in enumerate(d['cases'])}
        photos={p:{'label':'food'} for c in d['cases'] for p in c['photo_ids']}
        first=verification_orders(d,private,photos)
        for c in d['cases']:c['ai_suggested_label']='OPPOSITE_LABEL';c['human_final_label']='arbitrary'
        self.assertEqual(first,verification_orders(d,private,photos))

    def test_stale_export_and_out_of_prefix_verification_rejected(self):
        d,o,s,a=fixture();ai={'dataset_sha256':d['dataset_sha256'],'verification_order_sha256':o['manifest_sha256'],'cases':[{'case_id':c['case_id'],'ai_suggested_label':dict.fromkeys(c['labels'],'uncertain')} for c in d['cases']]}
        labels={'schema_version':2,'workflow':'AI_assisted_human_verification','dataset_kind':d['dataset_kind'],
                'dataset_sha256':d['dataset_sha256'],'verification_order_sha256':o['manifest_sha256'],
                'ai_artifact_sha256':canonical_sha(ai),'annotator':'TEST_ONLY','human_confirmed':True,
                'active_case_ids':active_ids(o,s),'cases':[{'case_id':c['case_id'],**a[c['case_id']],
                 'ai_suggested_label':ai['cases'][i]['ai_suggested_label'],'verified_at':'test','verification_action':'manual'} for i,c in enumerate(d['cases'])]}
        validate_export(d,o,s,ai,labels)
        labels['cases'][60]['human_verified']=True
        with self.assertRaises(AssertionError):validate_export(d,o,s,ai,labels)
        labels['cases'][60]['human_verified']=False
        larger={**s,'image_cases':70}
        with self.assertRaises(AssertionError):validate_export(d,o,larger,ai,labels)

    def test_K_distribution_keeps_zeros_and_never_changes_grid(self):
        rows=[{'business_id':str(i),'role':'K_study','clip_relevance_eligible_photo_ids':list(range(n)),
               'filtered_diversity_ranking':list(range(n)),'available_reference_groups':n} for i,n in enumerate([0,2,11])]
        r=summarize(rows);self.assertEqual(r['summary']['coarser_groups']['businesses_below_K']['3'],2)
        self.assertTrue(r['grid_adequacy_review_needed']);self.assertFalse(r['grid_inadequacy_proven']);self.assertEqual(r['K_candidates'],[3,5,8,10])


if __name__=='__main__':unittest.main()
