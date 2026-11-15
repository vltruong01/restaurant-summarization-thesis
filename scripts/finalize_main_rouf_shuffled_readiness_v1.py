"""CPU identity checks, additive decision record, and compact offline audit export."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np

R = Path(__file__).resolve().parents[1]
O = R/'outputs/stage_i/main_rouf_shuffled_readiness_v1'
B = R/'outputs/stage_i/cloud_return_20261015_v1/stage_hi_v2/outputs/stage_i/runs_v3'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    checks=[]
    for c in ('reviews_only','unfiltered','filtered'):
        with np.load(R/f'outputs/stage_h/led_dataset_v1/train_{c}.npz',allow_pickle=False) as z:
            train=set(z['business_ids'].tolist())
        with np.load(R/f'outputs/stage_h/led_dataset_v1/validation_{c}.npz',allow_pickle=False) as z:
            validation=set(z['business_ids'].tolist())
        rows=[json.loads(s) for s in (B/c/'validation_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(train)==2000 and len(validation)==200 and not train & validation
        assert len(rows)==200 and {x['business_id'] for x in rows}==validation
        assert all(x['condition']==c and x['token_ids'][0]==2 and x['token_ids'][-1]==2 for x in rows)
        checks.append(dict(condition=c,train=2000,validation=200,no_overlap=True,
            prediction_ids_exact=True,decoder_start_and_terminal_EOS_present=True,
            forced_vs_natural_EOS_not_inferred=True))
    write(O/'CPU_IDENTITY_CHECKS.json',dict(status='PASS',conditions=checks,
        new_generations=0,model_loads=0,GPU_operations=0,held_out_reads=0))
    inputs=read(O/'INPUT_SHA256_MANIFEST.json')['files']
    for name, item in inputs.items():
        p=Path(name) if Path(name).is_absolute() else R/name
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
    write(O/'FINAL_INPUT_PRESERVATION.json',dict(status='PASS',files=len(inputs),
        all_hashed_inputs_unchanged=True,scientific_artifact_writes=0,
        checkpoint_restore_on_GPU_not_performed=True))
    update=(
        '## Final CAD NO-GO; main RO/U/F/Shuffled readiness audited (2026-10-24)\n\n'
        'Independent researcher finalized NO-GO for CAD V1 alpha0.3; no CAD tuning '
        'or Contrastive reopening. Historical pending/provisional reports, raw outputs '
        'and limitations remain unchanged; new closure receipt records final decision. '
        'Offline audit:275 original matched-base backup files, all50 components in '
        'five corrected-large-F checkpoints,21 D053 reference-freeze files and213 CAD '
        'review entries reverified;378 unique input hashes unchanged. Base RO/U/F all '
        'complete5epochs/1250updates, identical configs and600 validation predictions '
        'on matching200IDs. Large Original CE F has48 validation outputs; CE3 preserved '
        'as provisional, matched large RO/U missing. Shuffled assignment/inference, '
        'held-out source packs/inference, final test references and formal30-business '
        'human-study results not found in canonical audit namespaces. D053 approved '
        '261Core/315Supporting/88Excluded covers24 development cases, not test. '
        'Saved72-output blinded worksheet remains unconfirmed; narrative researcher '
        'reviews are not fabricated as granular labels. Shortest recommendation: '
        'finish historical matched base family with no retraining; no mixed-size main '
        'comparison. Larger-family continuation requires separate authorization. '
        'No GPU/network/model load/held-out payload read or lifecycle action; retain '
        'instance57481997. No execution approval inferred.\n'
        'Report: outputs/stage_i/main_rouf_shuffled_readiness_v1/READINESS_AND_COMPLETION_PLAN.md\n\n')
    for name in ('DECISIONS.md','THESIS_PROGRESS.md','thesis_sdd_docs/THESIS_PROGRESS.md'):
        p=R/name
        original=p.read_bytes()
        assert not original.startswith(update.encode('utf-8'))
        p.write_bytes(update.encode('utf-8')+original)
        assert p.read_bytes()[len(update.encode('utf-8')):]==original
    extras=[R/'scripts/audit_main_rouf_shuffled_readiness_v1.py',Path(__file__).resolve(),
        R/'configs/project.yaml',R/'configs/image_overlap_evaluation_policy.yaml',
        R/'configs/led_source_template_d032.json',R/'specs/009_lora_led_training_preparation.md',
        R/'specs/009m_d059_expanded_lora_viability.md',
        R/'outputs/stage_i/d053_final_references_v1/D053_FINAL_REFERENCE_READINESS.json',
        R/'outputs/stage_i/cloud_return_20261015_v1/audit_v1/AUDIT_REPORT.md',
        R/'outputs/stage_i/led_large_decoder_policy_v2_execution_v1/EXECUTION_REPORT.md']
    files=sorted(list(O.glob('*'))+extras)
    manifest={p.relative_to(R).as_posix():sha(p) for p in files if p.is_file()}
    archive=R/'outputs/cloud_handoff/MAIN_ROUF_SHUFFLED_READINESS_V1_REVIEW.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:
            z.write(R/name,name)
        content=(O/'READINESS_AND_COMPLETION_PLAN.md').read_bytes()
        z.writestr('READ_THIS_FIRST.md',content)
        manifest['READ_THIS_FIRST.md']=hashlib.sha256(content).hexdigest()
        z.writestr('SHA256_MANIFEST.json',json.dumps(dict(files=manifest),indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    digest=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),archive=str(archive),
        bytes=archive.stat().st_size,sha256=digest,manifest_entries=len(manifest),
        GPU_execution_authorized=False,decision='STOP_FOR_RESEARCHER_REVIEW',
        recommendation='Complete historical matched LED-base family without new training',
        instance_policy='KEEP; no lifecycle action')
    write(O/'DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
