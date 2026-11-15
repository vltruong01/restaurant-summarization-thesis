"""Offline tests of recoverable native-Hub errors; no network/model/caption execution."""
from pathlib import Path
import hashlib
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import requests
import run_native_hf_smoke as worker


class NativeRetryTests(unittest.TestCase):
    def fixture(self,root):
        out=root/'out'
        (root/'configs').mkdir()
        (root/'configs/blip2_caption_feasibility_pilot.yaml').write_text('model_id: Salesforce/blip2-opt-2.7b-coco\nmodel_cache: cache\n')
        (out/'resume_20261006').mkdir(parents=True)
        (out/'pilot_identity.json').write_text('{"source_sha256":{}}')
        snapshot=root/'snapshot'
        snapshot.mkdir()
        files=[]
        for i in range(2):
            data=f'offline_fixture_not_model_{i}'.encode()
            name=f'model-{i+1:05d}-of-00002.safetensors'
            (snapshot/name).write_bytes(data)
            files.append({'filename':name,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        inventory={'resolved_commit':'a'*40,'total_bytes':sum(f['size_bytes'] for f in files),'weight_files':files}
        (out/'resume_20261006/checkpoint_download_inventory.json').write_text(json.dumps(inventory))
        prefix=root/'cache/models--Salesforce--blip2-opt-2.7b-coco/blobs'/(files[0]['sha256']+'.incomplete')
        prefix.parent.mkdir(parents=True)
        prefix.write_bytes(b'offline_prefix_retained')
        return out,snapshot,prefix

    def test_chunked_connection_failure_retries_and_checks_SHA_before_fresh_preflight(self):
        with tempfile.TemporaryDirectory(dir=worker.ROOT/'outputs') as directory:
            root=Path(directory)
            assert root.resolve().is_relative_to(worker.ROOT/'outputs')
            out,snapshot,prefix=self.fixture(root)
            original_wait=threading.Event.wait
            backoffs=[]
            def wait(event,timeout=None):
                if threading.current_thread() is threading.main_thread() and timeout==30:
                    backoffs.append(timeout);return False
                return original_wait(event,timeout)
            fetch=unittest.mock.Mock(side_effect=[requests.exceptions.ChunkedEncodingError('offline_fixture_disconnect'),str(snapshot)])
            def preflight(cfg,folder):
                state=json.loads((folder/'weight_download_progress.json').read_text())
                self.assertTrue(state['weights_SHA256_verified'])
                raise RuntimeError('offline_fixture_stop_before_any_GPU_model')
            with patch.object(worker,'ROOT',root),patch.object(worker,'OUT',out),patch.object(worker,'snapshot_download',fetch),patch.object(worker.backend,'check_sources'),patch.object(worker.backend,'preflight',preflight),patch.object(worker.backend,'generate') as generate,patch.object(threading.Event,'wait',wait),patch('sys.argv',['native','--remaining-hours','0.01']):
                self.assertEqual(worker.main(),1)
                generate.assert_not_called()
            self.assertEqual(fetch.call_count,2)
            self.assertEqual(backoffs,[30])
            self.assertEqual(prefix.read_bytes(),b'offline_prefix_retained')
            phase=json.loads((out/'local_smoke_controller_report.json').read_text())
            self.assertEqual(phase['last_network_error_type'],'ChunkedEncodingError')
            self.assertEqual(phase['network_attempt'],2)
            self.assertEqual(phase['captions_generated'],0)

    def test_permanent_error_stops_without_stale_running_progress(self):
        with tempfile.TemporaryDirectory(dir=worker.ROOT/'outputs') as directory:
            root=Path(directory)
            assert root.resolve().is_relative_to(worker.ROOT/'outputs')
            out,snapshot,prefix=self.fixture(root)
            fetch=unittest.mock.Mock(side_effect=ValueError('offline_fixture_permanent_error'))
            with patch.object(worker,'ROOT',root),patch.object(worker,'OUT',out),patch.object(worker,'snapshot_download',fetch),patch.object(worker.backend,'check_sources'),patch.object(worker.backend,'preflight') as preflight,patch('sys.argv',['native','--remaining-hours','0.01']):
                self.assertEqual(worker.main(),1)
                preflight.assert_not_called()
            self.assertEqual(fetch.call_count,1)
            progress=json.loads((out/'weight_download_progress.json').read_text())
            self.assertEqual(progress['status'],'native_download_stopped_error_resume_preserved')
            self.assertFalse(progress['worker_active'])
            self.assertEqual(prefix.read_bytes(),b'offline_prefix_retained')


if __name__=='__main__':unittest.main()
