"""Preserve raw outputs and descriptive diagnostics; no model or human labels."""
from pathlib import Path
from collections import Counter
from itertools import groupby
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'outputs/stage_i/led_large_vast_d059_v1_quality_review'
ARCHIVE = ROOT/'outputs/stage_i/led_large_vast_d059_v1_download_audit/downloaded_archives/LED_LARGE_F_QUALITY_REVIEW.zip'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    unpacked = DEST/'portable_review'
    with ARCHIVE.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    assert digest == '1d00c6b446a6c9fb47faf78d20e9faf79564cac1bb2bb3a030ea738b1afc3261'
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=digest,
                  status='PROVISIONAL_AI_ASSISTED_REVIEW_RESEARCHER_CONFIRMATION_PENDING', models={})
    with zipfile.ZipFile(ARCHIVE) as z:
        # Source ZIP was fully path/CRC/manifest checked by download audit.
        for info in z.infolist():
            if info.is_dir():
                continue
            output = unpacked/info.filename
            assert output.resolve().is_relative_to(unpacked.resolve())
            output.parent.mkdir(parents=True,exist_ok=True)
            raw = z.read(info)
            if output.exists():
                assert output.read_bytes() == raw
            else:
                output.write_bytes(raw)
        files = [('large_epoch3','predictions/epoch3.json'),('large_epoch5','predictions/epoch5.json'),
                 ('D059_epoch3','benchmarks/D059_Expanded_epoch3.json'),
                 ('D059_epoch5','benchmarks/D059_Expanded_epoch5.json'),
                 ('D058_B_epoch5','benchmarks/D058_B_epoch5.json')]
        for model, member in files:
            rows = json.loads(z.read(member))
            details = []
            for row in rows:
                text = row['text']
                # Descriptive punctuation-insensitive lexical runs, NOT human-confirmed errors.
                lexical = re.sub(r'\[[^\]\n]*\]', ' ', text)
                words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*",lexical.casefold())
                runs = [(word,len(list(g))) for word,g in groupby(words)]
                runs = [dict(word=w,count=n) for w,n in runs if n>=2]
                ids = row['token_ids']
                leading_bos = 0
                for value in ids[1:]:
                    if value != 0:
                        break
                    leading_bos += 1
                details.append(dict(business_id=row['business_id'],subset=row['subset'],
                    empty_text=not text.strip(), token_count=len(ids),
                    BOS_token_count=ids.count(0), leading_BOS_after_decoder_start=leading_bos,
                    token_frequency=Counter(ids).most_common(5),lexical_runs=runs,
                    exact_sentence_repeat=row['automatic']['repetition']['exact_repeated_sentence'],
                    legacy_adjacent_word_flag=row['automatic']['repetition']['consecutive_repeated_word'],
                    header_compliant=row['automatic']['structure']['section_headers_compliant'],
                    length_cap=row['automatic']['termination']['cap_reached']))
            groups = []
            for subset in ('random','targeted'):
                g = [r for r in details if r['subset']==subset]
                groups.append(dict(subset=subset,n=len(g),empty=sum(r['empty_text'] for r in g),
                    exact_repeat=sum(r['exact_sentence_repeat'] for r in g),
                    legacy_word_flag=sum(r['legacy_adjacent_word_flag'] for r in g),
                    lexical_run_at_least_3=sum(any(x['count']>=3 for x in r['lexical_runs']) for r in g),
                    headers=sum(r['header_compliant'] for r in g),caps=sum(r['length_cap'] for r in g)))
            result['models'][model] = dict(groups=groups,cases=details)
    (DEST/'DESCRIPTIVE_DIAGNOSTICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v['groups'] for k,v in result['models'].items()},indent=2))

if __name__ == '__main__':
    main()
