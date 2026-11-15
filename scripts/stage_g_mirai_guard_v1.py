"""Bind request-side business metadata to its immutable Stage G job.

This versioned adapter leaves the frozen schema, prompts, evidence and jobs
unchanged. A one-value string enum constrains only the business_id echoed by
the teacher. No returned target content or identifiers are rewritten.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path

from prepare_stage_g_teacher_pilot import ROOT, canonical, digest, file_sha, read
import run_stage_g_teacher_pilot_v4_1_recovery as legacy
from stage_g_canonical_review_ids import citation_diagnostics, resolve_review


GUARD_VERSION = 'stage_g_business_id_single_value_enum_v1'


def schema_for_job(job, base_schema):
    """Return an isolated request schema permitting exactly this business ID."""
    business_id = job.get('business_id')
    if not isinstance(business_id, str) or not business_id:
        raise ValueError('Guard requires a nonempty immutable business_id')
    schema = deepcopy(base_schema)
    if schema.get('type') != 'object' or 'business_id' not in schema.get('required', []):
        raise ValueError('Guard requires a required top-level business_id')
    field = schema.get('properties', {}).get('business_id')
    if not isinstance(field, dict) or field.get('type') != 'string':
        raise ValueError('Guard requires a string business_id property')
    if 'enum' in field and business_id not in field['enum']:
        raise ValueError('Guard cannot loosen a conflicting base business_id enum')
    if 'const' in field and field['const'] != business_id:
        raise ValueError('Guard cannot loosen a conflicting base business_id const')
    field['enum'] = [business_id]
    return schema


def payload_for_job(job, cfg, prompt, base_schema):
    """Keep the frozen request unchanged except for its business_id enum."""
    return legacy.payload(job, cfg, prompt, schema_for_job(job, base_schema))


def validate_output(job, output, base_schema):
    """Validate the guard and all frozen evidence/content checks without edits."""
    legacy.schema_check(output, schema_for_job(job, base_schema))
    return legacy.validate_output(job, output, base_schema)


def verify_checkpoint(job, cfg, directory, config_path):
    """Verify a new guarded checkpoint against the exact request and raw API output.

    Historical checkpoints require their original verifier and request schema.
    The frozen job hash and its original base-schema hash remain unchanged;
    the new checkpoint binding identifies the actual guarded request schema.
    """
    directory = Path(directory)
    config_path = Path(config_path)
    checkpoint_path = directory / 'checkpoints' / (job['job_id'] + '.json')
    cp = read(checkpoint_path)
    assert cp['status'] == 'STRUCTURAL_PASS_AWAITING_HUMAN_REVIEW', 'Unresolved/failed guarded checkpoint'
    base_schema = read(ROOT / cfg['schema'])[job['task']]
    guarded_schema = schema_for_job(job, base_schema)
    assert digest(canonical(base_schema)) == job['schema_sha256'], 'Frozen base schema changed'
    binding = cp['binding']
    assert binding['job_sha256'] == digest(canonical(job)), 'Frozen job changed'
    assert binding['config_sha256'] == file_sha(config_path), 'Guarded execution configuration changed'
    assert binding['prompt_sha256'] == job['prompt_sha256'], 'Frozen prompt binding changed'
    assert binding['schema_sha256'] == digest(canonical(guarded_schema)), 'Guarded request schema changed'
    assert (binding['base_url'], binding['provider'], binding['requested_model']) == (
        'https://api.miraiapi.com/v1', 'miraiapi', 'gpt-6.1-sol'), 'Teacher provider/model changed'
    prompt = (ROOT / cfg['prompts'][job['task']]).read_text(encoding='utf-8')
    assert digest(prompt.encode()) == job['prompt_sha256'], 'Frozen prompt changed'
    assert cp['validation'] == validate_output(job, cp['output'], base_schema), 'Guarded validation changed'

    if cp.get('api_called'):
        raw_path = directory / 'raw_responses' / checkpoint_path.name
        raw = read(raw_path)
        assert file_sha(raw_path) == cp['raw_response_sha256'], 'Raw teacher response changed'
        assert raw['status'] == 'completed' and raw['model'] == cfg['requested_model'], 'Incomplete/wrong-model response'
        assert json.loads(legacy.response_text(raw)) == cp['output'], 'Checkpoint differs from original teacher output'
        body = canonical(payload_for_job(job, cfg, prompt, base_schema))
        exact = gzip.decompress((directory / 'raw_requests' / (job['job_id'] + '.json.gz')).read_bytes())
        assert exact == body, 'Archived request differs from guarded request'
        intent = read(directory / 'request_intents' / checkpoint_path.name)
        assert intent['request_sha256'] == digest(body), 'Guarded request intent changed'
        assert intent['binding'] == binding, 'Request/checkpoint binding mismatch'
    else:
        assert cp.get('api_called') is False, 'Checkpoint must explicitly identify API usage'
        assert job.get('skip_api_empty_captions') and cp['output']['assessments'] == [], 'Invalid local empty condition'
        assert cp.get('creation') == 'deterministic_empty_visual_condition', 'Empty condition provenance changed'

    if job['task'] == 'review_consensus':
        resolved, evidence = resolve_review(job, cp['output'])
        assert resolved == cp['resolved_output'], 'Resolved review output changed'
        assert evidence == cp['evidence_resolution'], 'Review evidence resolution changed'
        assert [(c['text'], c['polarity']) for c in resolved['claims']] == [
            (c['text'], c['polarity']) for c in cp['output']['claims']], 'Teacher claim content changed'
        assert cp['citation_metadata_diagnostics'] == citation_diagnostics(job, cp['output']), 'Citation diagnostics changed'
    return cp
