"""Frozen Responses transport semantics with private, per-attempt diagnostics.

Same request payload, headers and SSE interpretation. Only the confirmed recovery
queue receives longer client-side waits; all normal production delegates unchanged.
"""
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from run_stage_g_teacher_pilot_v4_1_recovery import canonical, now

import stage_g_teacher_stream_diagnostics_v1 as original

CONTEXT = original.CONTEXT
EXTENDED_JOBS = frozenset()


def set_context(directory, job_id):
    original.set_context(directory, job_id)
    CONTEXT.job_id = job_id


def call(base, key, body):
    if getattr(CONTEXT, 'job_id', None) not in EXTENDED_JOBS:
        return original.call(base, key, body)
    path = CONTEXT.path
    assert not path.exists(), 'Diagnostic attempt already exists; no automatic resend'
    started = time.monotonic()
    report = {'started_utc': now(), 'requested_model': body['model'],
              'automatic_retries': 0, 'event_counts': {}, 'last_event_types': [],
              'execution_adapter': 'bounded_recovery_wait_v1', 'socket_timeout_seconds': 300,
              'wall_clock_check_seconds': 360, 'request_payload_headers_and_SSE_interpretation_unchanged': True}

    def safe(value):
        text = str(value).replace(key, '[REDACTED_TEACHER_CREDENTIAL]')
        text = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED_KEY_LIKE_TEXT]', text)
        text = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', text)
        return text[:6000]

    def metadata(response):
        if not isinstance(response, dict):
            return
        for name in ['id', 'status', 'model', 'usage']:
            if name in response:
                report['response_' + name] = response[name]
        for name in ['error', 'incomplete_details']:
            if response.get(name) is not None:
                report['provider_' + name] = safe(json.dumps(response[name], ensure_ascii=False))

    try:
        req = urllib.request.Request(base + '/responses', data=canonical(body), method='POST',
                headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                         'Accept': 'text/event-stream', 'User-Agent': 'codex-cli/0.149.0'})
        with urllib.request.urlopen(req, timeout=300) as stream:
            report['http_status'] = getattr(stream, 'status', None)
            headers = getattr(stream, 'headers', {})
            report['request_ids'] = {name: safe(headers.get(name)) for name in
                ['x-request-id', 'request-id', 'cf-ray'] if headers.get(name)}
            for line in stream:
                if time.monotonic() - started > 360:
                    report['failure_kind'] = 'wall_clock_deadline'
                    raise TimeoutError('Pilot request exceeded wall-clock deadline')
                line = line.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    report['done_marker_received'] = True
                    break
                event = json.loads(data)
                kind = str(event.get('type'))
                report['event_counts'][kind] = report['event_counts'].get(kind, 0) + 1
                report['last_event_types'] = (report['last_event_types'] + [kind])[-20:]
                if kind.startswith('response.') and isinstance(event.get('response'), dict):
                    metadata(event['response'])
                if kind == 'response.completed':
                    report['status'] = 'COMPLETE_RAW_RESPONSE_RECEIVED'
                    return event['response']  # Preserve original API output unchanged.
                if kind in {'error', 'response.failed', 'response.incomplete'}:
                    report['failure_kind'] = kind
                    if kind == 'error':
                        report['provider_error'] = safe(json.dumps(event, ensure_ascii=False))
                    raise RuntimeError('Response failed or incomplete; no automatic retry')
        report['failure_kind'] = 'stream_ended_without_completed_response'
        raise RuntimeError('Stream ended without a completed response')
    except Exception as error:
        report['status'] = 'TRANSPORT_FAILED_NO_AUTOMATIC_RETRY'
        report['error_type'] = type(error).__name__
        report['failure_kind'] = report.get('failure_kind', type(error).__name__)
        if isinstance(error, urllib.error.HTTPError):
            report['http_status'] = error.code
            report['provider_http_error_body_redacted'] = safe(error.read(65536).decode('utf-8', errors='replace'))
        report['error_message_redacted'] = safe(error)
        raise
    finally:
        report['finished_utc'] = now()
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        # Metadata is scrubbed too; no credentials or request headers are written.
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        encoded = encoded.replace(key, '[REDACTED_TEACHER_CREDENTIAL]')
        assert key not in encoded
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
