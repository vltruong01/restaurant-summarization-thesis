"""Offline tests for a client wait adjustment with unchanged API messages."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stage_g_teacher_recovery_wait_v1 as adapter


class FakeStream:
    status = 200
    headers = {}

    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(('data: ' + json.dumps(e) + '\n\n').encode() for e in self.events)


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / 'outputs/stage_g/recovery_wait_test_temp'
        self.root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.directory = Path(self.temp.name).resolve()
        assert self.directory.is_relative_to(self.root.resolve())
        adapter.EXTENDED_JOBS = frozenset({'known'})
        adapter.set_context(self.directory, 'known')
        self.body = {'model': 'gpt-6.1-sol', 'stream': True, 'store': False, 'input': []}
        self.response = {'id': 'fixture-only', 'status': 'completed', 'model': 'gpt-6.1-sol'}

    def tearDown(self):
        adapter.EXTENDED_JOBS = frozenset()
        assert self.directory.is_relative_to(self.root.resolve())
        self.temp.cleanup()

    def test_normal_production_delegates_without_changes(self):
        adapter.set_context(self.directory, 'new')
        with patch.object(adapter.original, 'call', return_value=self.response) as delegate:
            self.assertIs(adapter.call('https://provider.example/v1', 'test-secret', self.body), self.response)
            delegate.assert_called_once_with('https://provider.example/v1', 'test-secret', self.body)

    def test_recovery_keeps_payload_headers_and_original_response(self):
        event = {'type': 'response.completed', 'response': self.response}
        with patch.object(adapter.urllib.request, 'urlopen', return_value=FakeStream([event])) as open_request:
            result = adapter.call('https://provider.example/v1', 'test-secret', self.body)
        self.assertEqual(result, self.response)
        request = open_request.call_args.args[0]
        self.assertEqual(request.data, adapter.canonical(self.body))
        self.assertEqual(request.get_header('User-agent'), 'codex-cli/0.149.0')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-secret')
        self.assertEqual(open_request.call_args.kwargs['timeout'], 300)
        diagnostic = json.loads((self.directory / 'known.json').read_text(encoding='utf-8'))
        self.assertNotIn('test-secret', json.dumps(diagnostic))
        self.assertEqual(diagnostic['wall_clock_check_seconds'], 360)

    def test_delayed_complete_response_is_preserved(self):
        event = {'type': 'response.completed', 'response': self.response}
        with patch.object(adapter.urllib.request, 'urlopen', return_value=FakeStream([event])), \
             patch.object(adapter.time, 'monotonic', side_effect=[0, 194, 195]):
            self.assertEqual(adapter.call('https://provider.example/v1', 'test-secret', self.body), self.response)

    def test_response_failure_does_not_return_target(self):
        event = {'type': 'response.failed', 'response': {'id': 'fixture-only', 'status': 'failed'}}
        with patch.object(adapter.urllib.request, 'urlopen', return_value=FakeStream([event])):
            with self.assertRaises(RuntimeError):
                adapter.call('https://provider.example/v1', 'test-secret', self.body)
        diagnostic = json.loads((self.directory / 'known.json').read_text(encoding='utf-8'))
        self.assertEqual(diagnostic['failure_kind'], 'response.failed')

    def test_later_response_still_obeys_new_wait_check(self):
        event = {'type': 'response.completed', 'response': self.response}
        with patch.object(adapter.urllib.request, 'urlopen', return_value=FakeStream([event])), \
             patch.object(adapter.time, 'monotonic', side_effect=[0, 361, 362]):
            with self.assertRaises(TimeoutError):
                adapter.call('https://provider.example/v1', 'test-secret', self.body)

    def test_existing_attempt_diagnostic_is_never_overwritten(self):
        path = self.directory / 'known.json'
        path.write_text('preserve fixture', encoding='utf-8')
        with self.assertRaises(AssertionError):
            adapter.call('https://provider.example/v1', 'test-secret', self.body)
        self.assertEqual(path.read_text(encoding='utf-8'), 'preserve fixture')


if __name__ == '__main__':
    unittest.main()
