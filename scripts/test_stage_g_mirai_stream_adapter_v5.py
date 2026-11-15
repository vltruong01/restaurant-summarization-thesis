"""SSE transport fixtures only; no real teacher content or API calls."""
import gzip
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import stage_g_mirai_stream_adapter_v5 as adapter
import run_stage_g_mirai_stream_recovery_after_retry_v2 as trial
import stage_g_production_d042 as frozen


class StreamTests(unittest.TestCase):
    def completed(self,text,**extra):
        p=adapter.StreamProjection()
        p.consume({'type':'response.output_text.delta','delta':text[:3]})
        p.consume({'type':'response.output_text.delta','delta':text[3:]})
        p.consume({'type':'response.output_text.done','text':text})
        raw={'id':'offline_fixture','status':'completed',**extra}
        return p,raw
    def test_exact_unicode_and_punctuation_no_invented_model(self):
        text='{"fixture":"café, exact\\ntext"}'
        p,raw=self.completed(text);response,lineage=p.project(raw)
        self.assertEqual(response['output'][0]['content'][0]['text'],text)
        self.assertNotIn('model',response);self.assertNotIn('output',raw)
        self.assertFalse(lineage['model_metadata_added'])
    def test_delta_done_mismatch_rejected(self):
        p=adapter.StreamProjection();p.consume({'type':'response.output_text.delta','delta':'first'})
        p.consume({'type':'response.output_text.done','text':'different'})
        with self.assertRaises(ValueError):p.project({'status':'completed'})
    def test_partial_delta_without_done_rejected(self):
        p=adapter.StreamProjection();p.consume({'type':'response.output_text.delta','delta':'partial'})
        with self.assertRaises(ValueError):p.project({'status':'completed'})
    def test_conflicting_complete_items_rejected(self):
        p=adapter.StreamProjection();p.consume({'type':'response.output_item.done','item':{'type':'message','content':[]}})
        with self.assertRaises(ValueError):p.consume({'type':'response.output_item.done','item':{'type':'message','content':[{}]}})
    def test_completed_body_conflicts_with_stream_rejected(self):
        p,raw=self.completed('stream')
        raw['output']=[{'type':'message','content':[{'type':'output_text','text':'other'}]}]
        with self.assertRaises(ValueError):p.project(raw)
    def test_full_standard_response_preserved(self):
        p,raw=self.completed('same',model='offline_fixture_model')
        raw['output']=[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'same'}]}]
        result,lineage=p.project(raw);self.assertIs(result,raw);self.assertFalse(lineage['reconstructed_output'])
    def test_complete_response_without_deltas_preserved(self):
        raw={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'fixture'}]}]}
        self.assertIs(adapter.StreamProjection().project(raw)[0],raw)
    def test_complete_item_text_must_match_deltas(self):
        p=adapter.StreamProjection();p.consume({'type':'response.output_text.delta','delta':'one'})
        p.consume({'type':'response.output_item.done','item':{'type':'message','content':[{'type':'output_text','text':'two'}]}})
        with self.assertRaises(ValueError):p.project({'status':'completed'})
    def test_wire_frame_bytes_preserved_and_replay_matches(self):
        text='{"fixture":"exact café"}'
        events=[{'type':'response.output_text.delta','delta':text},
                {'type':'response.output_text.done','text':text},
                {'type':'response.completed','response':{'id':'fixture','status':'completed'}}]
        wire=': heartbeat\r\n\r\n'+''.join('data: '+json.dumps(e,ensure_ascii=False)+'\r\n\r\n' for e in events)
        stream=io.BytesIO(wire.encode());stream.fp=SimpleNamespace(raw=SimpleNamespace(_sock=Mock()))
        archive=io.BytesIO();frames=list(adapter.data_frames(stream,'offline_secret_fixture',archive,time.monotonic()))
        self.assertEqual(archive.getvalue(),wire.encode())
        self.assertEqual([json.loads(f) for f in frames],events)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'fixture.sse.gz';path.write_bytes(gzip.compress(wire.encode()))
            projected,lineage=adapter.replay(path,events[-1]['response'])
            self.assertEqual(projected['output'][0]['content'][0]['text'],text)
    def test_missing_model_keeps_production_guard_unsatisfied(self):
        self.assertEqual(trial.model_state({}, {}, 'expected'),('MODEL_METADATA_MISSING',False))
    def test_header_observation_does_not_fabricate_final_model(self):
        response={};diagnostic={'provider_model_observations':[{'source':'HTTP.header.x-model','value':'expected'}]}
        self.assertEqual(trial.model_state(response,diagnostic,'expected'),('MODEL_OBSERVED_OUTSIDE_FINAL_RESPONSE_REQUIRES_REVIEW',False))
        self.assertNotIn('model',response)
    def test_actual_returned_model_match_and_conflict(self):
        self.assertEqual(trial.model_state({'model':'expected'},{},'expected'),('MODEL_FIELD_MATCHES_REQUEST',True))
        self.assertEqual(trial.model_state({'model':'other'},{},'expected'),('MODEL_MISMATCH',False))
        self.assertFalse(trial.model_state({'model':'expected'},{'provider_model_observations':[{'value':'other'}]},'expected')[1])
    def test_scientific_settings_and_guard_identical(self):
        new=frozen.read(trial.CONFIG);old=frozen.read(trial.prior.CONFIG)
        for key in frozen.FIXED_KEYS:self.assertEqual(new[key],old[key],key)
        self.assertEqual(new['identifier_guard'],old['identifier_guard'])
        self.assertEqual(new['execution']['overall_deadline_seconds'],180)


if __name__=='__main__':unittest.main()
