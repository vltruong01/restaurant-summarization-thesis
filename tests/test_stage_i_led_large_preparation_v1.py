"""Offline proposal contracts; no model execution or GPU operations."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_stage_i_led_large_v1 import read,require,sha,expected_modules,validate_architecture,PREP,CONFIG


class LargePreparationTests(unittest.TestCase):
    def test_exact_revisions(self):
        r=read(PREP/'REMOTE_PROVENANCE.json')
        self.assertEqual(r['official']['revision'],'cd59d11c3528415b7dda4dfc95cc8f138aceda2e')
        self.assertEqual(r['amazon']['revision'],'4fdc41ed9bd4fbebe4fd6de131af1f27181f51ea')

    def test_downloaded_metadata_hashes(self):
        for role,r in read(PREP/'REMOTE_PROVENANCE.json').items():
            for name,v in r['files'].items():self.assertEqual(sha(PREP/'remote_metadata'/role/name),v['sha256'])

    def test_no_full_checkpoint_weights_downloaded(self):
        for role in ('official','amazon'):
            for name in ('pytorch_model.bin','model.safetensors','tf_model.h5'):
                self.assertFalse((PREP/'remote_metadata'/role/name).exists())

    def test_both_configs_same_architecture(self):
        a=read(PREP/'remote_metadata/official/config.json');b=read(PREP/'remote_metadata/amazon/config.json')
        self.assertEqual(validate_architecture(a),validate_architecture(b))

    def test_wrong_depth_rejected(self):
        c=read(PREP/'remote_metadata/official/config.json');c['encoder_layers']=6
        with self.assertRaises(ValueError):validate_architecture(c)

    def test_wrong_width_rejected(self):
        c=read(PREP/'remote_metadata/official/config.json');c['d_model']=768
        with self.assertRaises(ValueError):validate_architecture(c)

    def test_wrong_decoder_start_rejected(self):
        c=read(PREP/'remote_metadata/official/config.json');c['decoder_start_token_id']=0
        with self.assertRaises(ValueError):validate_architecture(c)

    def test_exact_lora_inventory(self):
        m=read(PREP/'META_LORA_INVENTORY.json')
        self.assertEqual({x['name'] for x in m['modules']},expected_modules())
        self.assertEqual(len(expected_modules()),228)
        self.assertEqual(sum(x['trainable_parameters'] for x in m['modules']),4915200)
        self.assertEqual(m['trainable_tensors'],456)

    def test_encoder_local_global_and_ffn_coverage(self):
        names=expected_modules()
        self.assertEqual(sum('.longformer_self_attn.' in n and n.endswith(('query','key','value')) for n in names),36)
        self.assertEqual(sum(n.endswith(('query_global','key_global','value_global')) for n in names),36)
        self.assertEqual(sum(n.endswith(('fc1','fc2')) for n in names),48)
        self.assertNotIn('lm_head',names)

    def test_meta_serialization_is_not_gpu_feasibility(self):
        m=read(PREP/'META_LORA_INVENTORY.json')
        self.assertTrue(m['adapter_serialization_exact'] and m['reloaded_meta_scope_exact'])
        self.assertFalse(m['full_checkpoint_weights_loaded'])
        self.assertEqual(m['actual_gradient_flow'],'NOT_TESTED')
        self.assertEqual(m['model_forward_backward_generation'],0)

    def test_hub_parameter_count_vs_buffer_distinguished(self):
        m=read(PREP/'META_LORA_INVENTORY.json')
        hub=read(PREP/'remote_metadata/amazon/HUB_METADATA.json')
        self.assertEqual(m['unique_backbone_parameters'],459801600)
        self.assertEqual(m['unique_backbone_parameters']+m['registered_buffer_elements'],hub['safetensors']['total'])

    def test_raw_amazon_truncation_failure_preserved(self):
        f=read(PREP/'TOKENIZER_FIRST_COMPATIBILITY_FAILURE.json')
        self.assertEqual(f['amazon']['backend_truncation']['max_length'],192)
        self.assertEqual(f['amazon']['details'][0]['candidate_len'],192)
        self.assertGreater(f['amazon']['details'][0]['gold_len'],192)
        self.assertIsNone(f['official']['details'][0]['first_mismatch'])

    def test_full_token_equivalence_all_train_validation(self):
        r=read(PREP/'TOKENIZER_DATA_COMPATIBILITY.json')
        self.assertEqual(r['business_targets'],6600)
        self.assertEqual(r['candidate_source_or_label_encodes'],26400)
        self.assertTrue(r['exact_against_frozen_packs'])
        self.assertFalse(r['new_tokenization_artifacts_or_truncation'])
        self.assertFalse(r['held_out_content_access'])

    def test_full_data_exposure_not_short128_pilot(self):
        c=read(CONFIG)
        self.assertEqual(c['training'],{'businesses':2000,'epochs':5,'steps':1250,'steps_per_epoch':250,'warmup_steps':63})
        self.assertEqual(c['evaluation_epochs'],[3,5]);self.assertEqual(c['F_evaluation_outputs'],48)

    def test_budgets_and_generation_unchanged(self):
        c=read(CONFIG);old=read(ROOT/'configs/stage_i_d059_expanded_lora_v2.json')
        for k in ('generation','optimization','lora','budgets','runtime_pins','seed'):
            self.assertEqual(c[k],old[k])
        self.assertEqual(c['global_attention'],'first_nonpadding_BOS_only; identical RO/U/F/Shuffled policy')

    def test_no_automatic_secondary_init_or_primary_replacement(self):
        c=read(CONFIG)
        self.assertEqual(c['model_id'],'allenai/led-large-16384')
        self.assertTrue(c['matched_family']['historical_base_family_primary_until_complete'])
        self.assertTrue(c['matched_family']['Shuffled_reuses_exact_F_adapter'])
        self.assertFalse(c['RO_U_authorized'])

    def test_fourteen_day_reserve_and_no_gpu_approval(self):
        from datetime import datetime
        c=read(CONFIG)
        freeze=datetime.fromisoformat(c['timeline']['proposed_absolute_model_family_freeze_utc'])
        defense=datetime.fromisoformat(c['timeline']['planning_earliest_defense_date']+'T00:00:00+00:00')
        self.assertGreaterEqual((defense-freeze).days,14)
        self.assertFalse(c['GPU_execution_approved'] or c['training_authorized'] or c['held_out_access'] or c['Teacher_calls'])
        self.assertIsNone(c['timeline']['GPU_approval_timestamp'])

    def test_unlikelihood_closed_without_quality_failure_claim(self):
        c=read(CONFIG)
        self.assertEqual(c['unlikelihood_status'],'CLOSED_NUMERICAL_FEASIBILITY_FAILURE_QUALITY_NOT_EVALUATED_NO_PATCH_NO_RETRY')
        self.assertEqual(c['D051_strict_replay'],'FAIL_UNCHANGED')


if __name__=='__main__':unittest.main(verbosity=2)
