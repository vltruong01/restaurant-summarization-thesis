"""Mirai SSE compatibility adapter, with bounded 600-second request deadline.

Preserve every received SSE payload and a separate response projection. Never
invent model metadata or rewrite generated text. Missing metadata still requires
explicit provider/protocol review before production acceptance.
"""
import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from run_stage_g_teacher_pilot_v4_1_recovery import canonical, now

CONTEXT = threading.local()


def set_context(directory, job_id):
    CONTEXT.directory = Path(directory)
    CONTEXT.job_id = job_id


class StreamProjection:
    """Copy completed text events verbatim; reconstruct only the wire envelope."""

    def __init__(self):
        self.parts = defaultdict(list)
        self.done = {}
        self.item_done = {}

    def consume(self, event):
        kind = event.get('type')
        slot = (event.get('output_index', 0), event.get('content_index', 0))
        if kind == 'response.output_text.delta':
            if not isinstance(event.get('delta'), str):
                raise ValueError('Non-string output-text delta')
            self.parts[slot].append(event['delta'])
        elif kind == 'response.output_text.done':
            if not isinstance(event.get('text'), str):
                raise ValueError('Non-string completed output text')
            if slot in self.done and self.done[slot] != event['text']:
                raise ValueError('Conflicting completed output text')
            self.done[slot] = event['text']
        elif kind == 'response.output_item.done':
            if isinstance(event.get('item'), dict):
                self.item_done[event.get('output_index', 0)] = event['item']

    def project(self, completed):
        # A full standard response is preserved exactly, not normalized.
        if completed.get('output'):
            return completed, {'reconstructed_output': False, 'model_metadata_added': False}
        selected = {}
        for slot in set(self.parts) | set(self.done):
            joined = ''.join(self.parts.get(slot, []))
            if slot in self.done:
                if slot in self.parts and joined != self.done[slot]:
                    raise ValueError('Output delta/text.done disagreement')
                selected[slot] = self.done[slot]
            else:
                selected[slot] = joined
        # Prefer provider's complete message item and assert exact stream agreement.
        items = []
        indexes = set(i for i, _ in selected) | set(self.item_done)
        for index in sorted(indexes):
            item = self.item_done.get(index)
            if item and item.get('type') == 'message' and item.get('content'):
                for content_index, part in enumerate(item['content']):
                    slot = (index, content_index)
                    if part.get('type') == 'output_text' and slot in selected:
                        if part.get('text') != selected[slot]:
                            raise ValueError('Message item/stream output disagreement')
                items.append(item)
            elif any(i == index for i, _ in selected):
                items.append({'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': selected[slot]}
                    for slot in sorted(selected) if slot[0] == index]})
            elif item:
                items.append(item)
        if not any(part.get('type') == 'output_text' for item in items
                   for part in item.get('content', []) if isinstance(part, dict)):
            raise ValueError('Completed stream contains no recoverable output text')
        projected = dict(completed, output=items)
        assert projected.get('model') == completed.get('model')
        return projected, {'reconstructed_output': True, 'model_metadata_added': False,
                           'text_origin': 'exact_provider_SSE_text_or_complete_message_item',
                           'source_slots': [list(slot) for slot in sorted(selected)]}


def call(base, key, body):
    """Unexecuted compatibility transport; same API body/headers, 120-second socket idle and 600-second overall deadline, zero retries."""
    directory = CONTEXT.directory
    jid = CONTEXT.job_id
    directory.mkdir(parents=True, exist_ok=True)
    events_path = directory / (jid + '.sse.jsonl.gz')
    diagnostic_path = directory / (jid + '.json')
    assert not events_path.exists() and not diagnostic_path.exists()
    projection = StreamProjection()
    started = time.monotonic()
    diagnostic = {'started_utc': now(), 'requested_model': body['model'],
                  'automatic_retries': 0, 'transport_version': 'mirai_stream_adapter_v4',
                  'event_counts': {}, 'events_file': events_path.name,
                  'overall_deadline_seconds': 600, 'socket_idle_timeout_seconds': 120,
                  'first_data_seconds': None, 'first_output_text_seconds': None}

    def redact(value):
        text = str(value).replace(key, '[REDACTED_TEACHER_CREDENTIAL]')
        text = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '[REDACTED_KEY_LIKE_TEXT]', text)
        return re.sub(r'(?i)Bearer\s+\S+', 'Bearer [REDACTED]', text)

    try:
        req = urllib.request.Request(base + '/responses', data=canonical(body), method='POST',
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                     'Accept': 'text/event-stream', 'User-Agent': 'codex-cli/0.149.0'})
        # Never overwrite a previous attempt; gzip stores SSE data frames as received.
        with events_path.open('xb') as file, gzip.GzipFile(fileobj=file, mode='wb', mtime=0) as archive:
            with urllib.request.urlopen(req, timeout=120) as stream:
                diagnostic['http_status'] = getattr(stream, 'status', None)
                diagnostic['request_ids'] = {name: redact(stream.headers.get(name)) for name in ('x-request-id', 'request-id', 'cf-ray') if stream.headers.get(name)}
                last_flush = time.monotonic()
                # Bound both silent reads and overall time; heartbeat comments do not reset the overall deadline.
                while True:
                    remaining = 600 - (time.monotonic() - started)
                    if remaining <= 0:
                        diagnostic['failure_kind'] = 'wall_clock_deadline'
                        raise TimeoutError('Teacher request exceeded 600-second operational deadline')
                    # HTTPResponse can auto-close after its final line; classify EOF, not socket AttributeError.
                    if stream.fp is None: break
                    stream.fp.raw._sock.settimeout(min(120, remaining))
                    line = stream.readline()
                    if not line: break
                    diagnostic['last_received_seconds'] = round(time.monotonic() - started, 3)
                    if time.monotonic() - started > 600:
                        diagnostic['failure_kind'] = 'wall_clock_deadline'
                        raise TimeoutError('Teacher request exceeded 600-second operational deadline')
                    line = line.decode('utf-8').strip()
                    if not line.startswith('data:'): continue
                    data = line[5:].strip()
                    # Auth headers are not stored; unexpected credential echo is redacted.
                    archive.write((redact(data) + '\n').encode('utf-8'))
                    if time.monotonic() - last_flush >= 2:
                        archive.flush(); last_flush = time.monotonic()
                    if data == '[DONE]': break
                    event = json.loads(data)
                    kind = event.get('type')
                    elapsed = round(time.monotonic() - started, 3)
                    if diagnostic['first_data_seconds'] is None: diagnostic['first_data_seconds'] = elapsed
                    if kind == 'response.output_text.delta' and diagnostic['first_output_text_seconds'] is None:
                        diagnostic['first_output_text_seconds'] = elapsed
                    diagnostic['last_data_seconds'] = elapsed
                    diagnostic['event_counts'][str(kind)] = diagnostic['event_counts'].get(str(kind), 0) + 1
                    projection.consume(event)
                    if kind == 'response.completed':
                        raw = event['response']
                        # Wire response and projected response are separate artifacts.
                        original_path = directory / (jid + '.completed_wire.json')
                        raw_encoded = json.dumps(raw, ensure_ascii=False, indent=2)
                        assert key not in raw_encoded, 'Unexpected credential echo in provider response'
                        with original_path.open('x', encoding='utf-8') as handle:
                            handle.write(raw_encoded + '\n')
                        response, lineage = projection.project(raw)
                        diagnostic.update(status='COMPLETE_PROVIDER_STREAM_PROJECTED',
                                          response_id=raw.get('id'), response_model=raw.get('model'),
                                          output_projection=lineage)
                        return response
                    if kind in ('error', 'response.failed', 'response.incomplete'):
                        diagnostic['failure_kind'] = kind
                        raise RuntimeError('Provider stream failed; no automatic retry')
        diagnostic['failure_kind'] = 'stream_EOF_without_completed_response'
        raise RuntimeError('Stream ended without response.completed; no target accepted')
    except Exception as error:
        diagnostic.update(status='FAILED_NO_AUTOMATIC_RETRY', error_type=type(error).__name__, error_message_redacted=redact(error))
        if isinstance(error, urllib.error.HTTPError): diagnostic['http_status'] = error.code
        raise
    finally:
        diagnostic.update(finished_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
        encoded = redact(json.dumps(diagnostic, ensure_ascii=False, indent=2))
        with diagnostic_path.open('x', encoding='utf-8') as handle:
            handle.write(encoded + '\n')


def offline_self_check():
    """Transport fixtures only, unrelated to any business/teacher target."""
    value = '{"transport_fixture":"Unicode café\\nround-trip"}'
    p = StreamProjection()
    for part in (value[:12], value[12:]):
        p.consume({'type': 'response.output_text.delta', 'output_index': 0,
                   'content_index': 0, 'delta': part})
    p.consume({'type': 'response.output_text.done', 'output_index': 0, 'content_index': 0, 'text': value})
    raw = {'id': 'offline_fixture', 'status': 'completed', 'usage': {}}
    projected, provenance = p.project(raw)
    assert projected['output'][0]['content'][0]['text'] == value
    assert 'output' not in raw and 'model' not in projected and provenance['reconstructed_output']
    standard = dict(projected, model='offline_fixture_model')
    unchanged, provenance = p.project(standard)
    assert unchanged is standard and not provenance['reconstructed_output']
    p.consume({'type': 'response.output_text.done', 'output_index': 1, 'content_index': 0, 'text': 'done'})
    mismatch = StreamProjection()
    mismatch.consume({'type': 'response.output_text.delta', 'delta': 'a'})
    mismatch.consume({'type': 'response.output_text.done', 'text': 'b'})
    try:
        mismatch.project(raw)
    except ValueError:
        pass
    else:
        raise AssertionError('Disagreement should reject')
    return {'offline_fixture_checks': 'PASS', 'API_calls': 0, 'real_targets_generated_or_rewritten': 0,
            'adapter_connected_to_paid_runner': False, 'production_approved': False}


if __name__ == '__main__': print(json.dumps(offline_self_check()))
