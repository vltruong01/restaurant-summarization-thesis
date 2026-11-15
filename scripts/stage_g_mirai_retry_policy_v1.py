"""Bounded operational recovery; never hide upstream billing or unknown outcomes."""
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def retry_plan(diagnostic, recorded_attempts, retry_after=None, epoch=None):
    """Initial + two recorded retries; explicit upstream reset is not account billing."""
    if recorded_attempts >= 3:
        return {'action': 'defer', 'reason': 'attempt_limit', 'wait_seconds': 0}
    status = diagnostic.get('http_status')
    text = diagnostic.get('provider_http_error_body_redacted', '')
    try:
        error = json.loads(text).get('error', {})
    except (ValueError, AttributeError):
        error = {}
    message = error.get('message', '') if isinstance(error, dict) else ''
    reset = re.search(r'\breset after\s+(\d+)\s*(s|m)\b', message, re.I)
    upstream_reset = (status == 404 and isinstance(error, dict)
                      and error.get('type') == 'upstream_error'
                      and 'insufficient balance' in message.lower() and reset is not None)
    if status in (401, 402, 403):
        return {'action': 'stop', 'reason': 'provider_auth_or_account_billing', 'wait_seconds': 0}
    if upstream_reset:
        reason = 'explicit_upstream_billing_reset'
        delay = int(reset.group(1)) * (60 if reset.group(2).lower() == 'm' else 1)
    elif status == 429 or status in (500, 502, 503, 504):
        reason = 'retryable_provider_http_failure'
        delay = 0
    else:
        return {'action': 'defer', 'reason': 'not_explicitly_retryable_or_unknown_outcome', 'wait_seconds': 0}
    delay = max(delay, (30, 60)[recorded_attempts - 1])
    reference = datetime.fromisoformat(diagnostic['finished_utc']).timestamp()
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except (ValueError, TypeError):
            try:
                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - reference)
            except (ValueError, TypeError, OverflowError):
                pass
    now = datetime.now(timezone.utc).timestamp() if epoch is None else epoch
    # An excessive provider wait remains visible; do not ignore it or sleep forever.
    if delay > 600:
        return {'action': 'defer', 'reason': 'provider_wait_exceeds_bounded_trial', 'wait_seconds': delay}
    return {'action': 'retry', 'reason': reason, 'wait_seconds': max(0, reference + delay - now),
            'cooldown_seconds': delay}
