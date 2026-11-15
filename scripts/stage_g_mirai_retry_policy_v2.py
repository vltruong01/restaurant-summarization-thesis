"""Provider-aware retry classification; distinguish unavailable upstream from user auth."""
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def retry_plan(diagnostic, recorded_attempts, retry_after=None, epoch=None):
    if recorded_attempts < 1:
        raise ValueError('A recorded failed attempt is required')
    if recorded_attempts >= 3:
        return {'action': 'defer', 'reason': 'attempt_limit', 'wait_seconds': 0}
    status = diagnostic.get('http_status')
    try:
        error = json.loads(diagnostic.get('provider_http_error_body_redacted', '')).get('error', {})
    except (ValueError, AttributeError):
        error = {}
    message = error.get('message', '') if isinstance(error, dict) else ''
    message = message if isinstance(message, str) else ''
    upstream = isinstance(error, dict) and error.get('type') == 'upstream_error'
    if upstream and 'no active credentials for provider' in message.lower():
        return {'action': 'stop', 'reason': 'upstream_credentials_unavailable',
                'external_provider_repair_required': True, 'wait_seconds': 0}
    reset = re.search(r'\breset after\s+((?:\d+\s*[smh]\s*)+)', message, re.I)
    balance = re.search(r'\binsufficient\s+(?:account\s+)?balance\b|\binsufficient_balance\b', message, re.I)
    upstream_reset = upstream and reset is not None and balance is not None and status in (403, 404, 500, 502, 503, 504)
    if upstream_reset:
        reason = 'explicit_upstream_billing_reset'
        delay = sum(int(n) * {'s': 1, 'm': 60, 'h': 3600}[unit.lower()]
                    for n, unit in re.findall(r'(\d+)\s*([smh])', reset.group(1), re.I))
    elif status in (401, 402, 403):
        return {'action': 'stop', 'reason': 'provider_auth_or_account_billing_not_retryable', 'wait_seconds': 0}
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
    if delay > 600:
        return {'action': 'defer', 'reason': 'provider_wait_exceeds_bounded_trial', 'wait_seconds': delay}
    return {'action': 'retry', 'reason': reason, 'wait_seconds': max(0, reference + delay - now),
            'cooldown_seconds': delay}
