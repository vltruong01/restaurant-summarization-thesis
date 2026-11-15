"""D-032/source and K-study rule guards; synthetic fixtures never become scientific data."""
import copy
import unittest
from unittest.mock import patch
import yaml
from stage_f_d032_protocol import CONFIG,BASELINE,assert_exact_authorized_delta,SOURCE_TEMPLATE,AUDIT
from stage_f_verification_workflow import read
from run_stage_f_k_sensitivity_d032 import group_coverage,rule_candidate


class D032Tests(unittest.TestCase):
    def test_exact_budget_delta_without_filter_or_K_redefinition(self):
        old=yaml.safe_load(BASELINE.read_text(encoding='utf-8'));new=yaml.safe_load((CONFIG.parents[1]/'outputs/stage_f/methodology_amendment_d033/previous_state/configs/stage_f_validation_protocol.yaml').read_text(encoding='utf-8'))
        self.assertTrue(assert_exact_authorized_delta(old,new))
        for mutate in [lambda c:c['thresholds'].update(relevance=.09),lambda c:c['budget'].update(caption_tokens=800),
                       lambda c:c['K_candidates'].append(6),lambda c:c.update(K=5),lambda c:c['quality_order'].reverse()]:
            bad=copy.deepcopy(new);mutate(bad)
            with self.assertRaises(AssertionError):assert_exact_authorized_delta(old,bad)

    def test_production_template_core_equals_validated_audit(self):
        p=read(SOURCE_TEMPLATE);a=read(AUDIT/'audit_template.json')
        fields=set(a)-{'status'}
        self.assertEqual({k:p[k] for k in fields},{k:a[k] for k in fields})
        self.assertEqual(p['status'],'APPROVED_VALIDATED_FROZEN')

    def test_coverage_counts_distinct_frozen_groups_not_min_K_G(self):
        pool={'available_reference_group_ids':['x','y','z'],'available_reference_groups':3,
              'reference_group_by_photo_id':{'a':'x','b':'x','c':'y','d':'z'}}
        coverage,groups=group_coverage(pool,['a','b'])
        self.assertEqual(coverage,1/3);self.assertEqual(groups,['x'])
        self.assertEqual(group_coverage(pool,['a','c','d'])[0],1)

    def test_empty_business_coverage_NA_not_fabricated_zero(self):
        pool={'available_reference_group_ids':[],'available_reference_groups':0,'reference_group_by_photo_id':{}}
        self.assertEqual(group_coverage(pool,[]),(None,[]))
        with self.assertRaises(AssertionError):group_coverage(pool,['fake'])

    def test_smallest_feasible_95pct_candidate_without_rounding(self):
        rows=[{'K':k,'macro_coarser_group_coverage':v,'budget_feasible_both_conditions':True} for k,v in zip([3,5,8,10],[.90,.951,.99,1.])]
        self.assertEqual(rule_candidate(rows),(5,1.))
        rows[1]['macro_coarser_group_coverage']=.94999999
        self.assertEqual(rule_candidate(rows)[0],8)
        rows[1]['macro_coarser_group_coverage']=.96;rows[1]['budget_feasible_both_conditions']=False
        self.assertEqual(rule_candidate(rows)[0],8)

    def test_no_feasible_candidate_is_not_silently_frozen(self):
        rows=[{'K':k,'macro_coarser_group_coverage':v,'budget_feasible_both_conditions':False} for k,v in zip([3,5,8,10],[.90,.96,.99,1.])]
        self.assertEqual(rule_candidate(rows),(None,1.))

    def test_legacy_CLI_D032_freezes_budget_not_refit_caption_thresholds(self):
        import calibrate_stage_f_thresholds as legacy
        with patch('sys.argv',['calibrate','--labels','TEST_ONLY.json']), \
             patch('stage_f_d033_protocol.active_d033',return_value=False), \
             patch('stage_f_d032_protocol.active_d032',return_value=True), \
             patch('stage_f_d032_protocol.freeze_budget_manifest') as freeze, \
             patch.object(legacy,'threshold_search') as search,patch.object(legacy,'write') as old_mutation:
            legacy.main();freeze.assert_called_once();search.assert_not_called();old_mutation.assert_not_called()


if __name__=='__main__':unittest.main()
