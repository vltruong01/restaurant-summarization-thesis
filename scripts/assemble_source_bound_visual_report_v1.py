"""Package the single authorized saved-output diagnostic and provisional source review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_i/source_bound_visual_diagnostic_v1'

# Case-specific, identity-aware model-assisted observations; not human adjudication.
ASSESSMENTS=[
    dict(name='Tempo Nashville',kind='zero_caption_abstention',
         unsupported='No selected visual evidence exists. Taco/bell/chicken relations have no caption support.',
         fusion='Not assessable across captions: there are none.',
         attribution='Removes all photo assertions; no photo or object identity is inferred.',
         coverage='No caption-supported visual information is lost.',
         tradeoff='Visual section becomes blank; not an independently readable visual description.',
         judgment='Clear source-compliance repair; review synthesis is unchanged.',
         quotes=[('The taco is next to a bell.',[]) ]),
    dict(name='Mesa Pizza',kind='zero_caption_abstention',
         unsupported='The claim that a photo shows pizza has no selected-caption support, even if pizza is discussed in reviews.',
         fusion='No captions; no cross-caption comparison applies.',
         attribution='Removes the review-to-photo attribution implicit in the baseline.',
         coverage='No caption-supported visual information is lost.',
         tradeoff='Loses a plausible but visually ungrounded sentence; blank visual body is intentional.',
         judgment='Clear source-compliance repair, not proof that the pictured object was wrong.',
         quotes=[('The photo shows a pizza.',[]) ]),
    dict(name='Terra Mar Brazilian Steakhouse',kind='zero_caption_abstention',
         unsupported='Steakhouse/table/salad-bar photo assertions have no selected-caption support.',
         fusion='No captions; no cross-caption comparison applies.',
         attribution='Does not treat business or review information as observed photographic evidence.',
         coverage='No caption-supported visual information is lost.',
         tradeoff='Visual section becomes blank. Existing review redundancy remains.',
         judgment='Clear source-compliance repair; no change to the LED model.',
         quotes=[('The steakhouse has a salad bar.',[]) ]),
    dict(name="Tujague's Restaurant",kind='caption_relative_grounding_repair',
         unsupported='Baseline changes making a drink into sitting, and invents drink-on-drink placement.',
         fusion='The self-referential drink relation is unsupported; its exact internal origin is unproven.',
         attribution='Each original caption remains separately labeled; action and object relations stay local.',
         coverage='Restores chocolate/purple flower, long table, and the woman making a drink; retains steak and room details.',
         tradeoff='Caption list is slightly longer (60 to 63 words including labels); no evidence that the caption identities are photo-true.',
         judgment='Clear improvement relative to captions; final prose usability remains provisional.',
         quotes=[('A woman is sitting at a bar.',['IMG_CAP_3']),
                 ('A drink is on top of the drink.',['IMG_CAP_1'])]),
    dict(name='DI Coffee Bar',kind='formatting_tradeoff',
         unsupported='No clear caption-unsupported visual assertion identified in this baseline.',
         fusion='No clear cross-caption entity/relation fusion identified.',
         attribution='Explicit CAP labels make provenance clearer; baseline already follows the caption content.',
         coverage='All five captions retained, including sign text, cake and coffee.',
         tradeoff='75 to 65 words and fewer split-off micro-sentences, but fragments/labels reduce narrative fluency. Raw caption repeats bottles.',
         judgment='Possible redundancy reduction, not a demonstrated factuality or overall-quality gain.',
         quotes=[('The room contains bottles of wine. The room contains other bottles.',['IMG_CAP_1'])]),
    dict(name='The Habit Burger Grill',kind='caption_relative_grounding_repair',
         unsupported='Caption lists tenders and sandwich separately; baseline puts tenders inside the sandwich and adds close-up specificity.',
         fusion='Baseline places the tray in front of the sign by joining CAP1 tray with CAP2 sign/building.',
         attribution='Separate caption lines remove these added within/cross-caption relations.',
         coverage='Restores fries and the building; retains menu, counter and blue tile wall.',
         tradeoff='Same 54-word length with labels. Separate restaurant captions can be repetitive; raw object identities remain unverified.',
         judgment='Clear caption-relative grounding repair without evidence of improved learned synthesis.',
         quotes=[('The sandwich contains chicken tenders.',['IMG_CAP_1']),
                 ('The tray is in front of a restaurant sign.',['IMG_CAP_1','IMG_CAP_2'])]),
    dict(name='Little Gourmand',kind='caption_relative_grounding_repair',
         unsupported='Pastries on bakery display are not supported by the separate pastries/cakes captions as that relation.',
         fusion='CAP4 pastries acquire the bakery-display relation from CAP5 cakes.',
         attribution='Restores pastries and cakes as separate caption observations.',
         coverage='Adds omitted masked woman/tray and many cakes; keeps pasta salad, milkshake and blueberry/cream pastries.',
         tradeoff='61 to 66 words with labels. Pastry references remain redundant; ambiguous BLIP-2 milkshake identity is copied, not verified or corrected.',
         judgment='Clear relation repair and caption coverage gain; longer and less synthesized.',
         quotes=[('The pastries are on display in a bakery.',['IMG_CAP_4','IMG_CAP_5'])]),
    dict(name='Kobe Japanese Steakhouse - Clearwater',kind='caption_relative_grounding_repair',
         unsupported='No caption mentions a chef. The baseline sushi-on-bar relation is not in the sushi caption.',
         fusion='CAP2 drink-on-bar and CAP3 sushi-on-plate are combined. Grill food and the separate shrimp plate are also linked ambiguously.',
         attribution='Each caption supplies only its own objects and relations; no common photo/table scene is inferred.',
         coverage='Restores tall drink/cherry and noodles/vegetables; retains sushi, black plate and shrimp plate.',
         tradeoff='47 to 54 words with labels. Raw CAP2 repeats on top; copying cannot fix that redundancy or verify object identities.',
         judgment='Clear removal of unsupported cross-caption relations; real-photo accuracy still unknown.',
         quotes=[('A long sushi roll sits on top of a bar.',['IMG_CAP_2','IMG_CAP_3']),
                 ('A chef is cooking steaks on a grill.',['IMG_CAP_1'])]),
    dict(name='Whistlers Inn',kind='formatting_tradeoff',
         unsupported='No clear caption-unsupported visual assertion identified.',
         fusion='No clear cross-caption fusion identified.',
         attribution='Captions remain distinct; original visual already preserves the local relations.',
         coverage='All three captions retained: wings/sauce/celery, alcohol bottles/counter, sandwich/fries.',
         tradeoff='50 to 40 words; fewer repeated photo/plate phrases, but list fragments are less natural than a concise edited summary.',
         judgment='Possible concision gain only; not a hallucination-repair success case.',
         quotes=[('The wings are served with sauce. The wings are served with celery.',['IMG_CAP_1'])]),
    dict(name="P J Whelihan's Pub & Restaurant - Oaks",kind='coverage_readability_tradeoff',
         unsupported='No clear serious unsupported visual relation identified in the original.',
         fusion='No clear cross-caption fusion identified.',
         attribution='Separate labels preserve provenance; no new photographic scene is asserted.',
         coverage='Restores sports-bar qualifier and tray under the three beer glasses; other visual content retained.',
         tradeoff='50 to 53 words with labels; added tray/qualifier may not outweigh reduced fluency and repeated table/bar phrasing.',
         judgment='Small caption-coverage gain with uncertain net usability benefit.',
         quotes=[('Three glasses of beer are sitting on a table.',['IMG_CAP_4'])]),
    dict(name='Opal Restaurant & Bar',kind='caption_relative_grounding_repair',
         unsupported='Dessert is absent from the captions; drink garnish, plate/fork and people at a long table are changed.',
         fusion='CAP2 drink/lime and CAP3 food/fork become dessert/lime/fork; people from CAP1 are placed at a bar.',
         attribution='Keeps drink, plate, table group, shelves and wine as separately attributed observations.',
         coverage='Restores blackberry and long table; retains plate/fork, alcohol shelves and wine/glass.',
         tradeoff='54 to 70 words with labels, the largest word increase. Restored coverage costs concision; all objects remain caption-derived.',
         judgment='Clear grounding repair but substantial verbosity/fluency tradeoff.',
         quotes=[('A dessert is served with a lime wedge.',['IMG_CAP_2']),
                 ('The dessert is served with a fork.',['IMG_CAP_3']),
                 ('A group of people is sitting at a bar.',['IMG_CAP_1'])]),
    dict(name='Minerva Indian Restaurant',kind='likely_display_regression',
         unsupported='No clear unsupported visual assertion identified.',
         fusion='No fusion identified: plate and background person are from the same caption.',
         attribution='Original already matches the two captions; labels add traceability but do not repair a content defect.',
         coverage='No material caption-supported gain or loss identified.',
         tradeoff='25 to 28 words with labels, lower-case fragments replace already readable sentences; likely worse displayed prose.',
         judgment='Explicit likely usability regression; not every source-bound alternative is better.',
         quotes=[('A plate of food is sitting on a table. A person is in the background.',['IMG_CAP_2'])]),
]


def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def write(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    rows=read('ALL_PAIRED_OUTPUTS.json')
    assert len(rows)==len(ASSESSMENTS)==12
    paired=[];case_docs=[];quote_count=0
    table=['# Case-level comparison — provisional model-assisted source review','',
           'TRAIN only; identity-aware, not blinded or researcher-confirmed. Word counts include Caption N labels. All review prefixes are unchanged. No photo truth adjudication was performed.','',
           '| Case | Caps | Visual words CE3 → alternative | Caption-relative findings | Coverage / regression | Provisional interpretation |',
           '|---|---:|---:|---|---|---|']
    for r,a in zip(rows,ASSESSMENTS):
        assert r['name']==a['name']
        quotes=[]
        bycap={x['caption_id']:x for x in r['caption_trace']}
        for quote,ids in a['quotes']:
            at=r['original_text'].index(quote)
            q=dict(original_quote=quote,original_char_start=at,original_char_end=at+len(quote),
                   original_UTF8_byte_start=len(r['original_text'][:at].encode()),
                   original_UTF8_byte_end=len(r['original_text'][:at+len(quote)].encode()),
                   related_frozen_captions=[bycap[k] for k in ids],
                   no_captions_available=not bool(bycap))
            assert r['original_text'][q['original_char_start']:q['original_char_end']]==quote
            quotes.append(q);quote_count+=1
        item=dict(business_id=r['business_id'],name=r['name'],assessment_origin='MODEL_ASSISTED_IDENTITY_AWARE_PROVISIONAL',
                  researcher_confirmed=False,original_photo_truth='UNVERIFIED',
                  **{k:v for k,v in a.items() if k not in ('name','quotes')},evidence_quotes=quotes)
        paired.append(item)
        old=r['checks']['original_visual_words'];new=r['checks']['alternative_visual_words']
        table.append(f"| [{r['name']}](cases/{r['business_id']}/COMPARISON.md) | {len(bycap)} | {old} → {new} | {a['unsupported']} {a['fusion']} | {a['coverage']} {a['tradeoff']} | {a['judgment']} |")
        doc=[f"# {r['name']}",'',f"Business: `{r['business_id']}` · TRAIN / Filtered · saved Original CE3 N1 only",'',
             'This is a provisional model-assisted comparison, not human evaluation or independent quality confirmation.',
             '',f"Source SHA-256: `{r['source_sha256']}`",'',
             '[Full frozen model-visible reviews/captions](SOURCE_MODEL_VISIBLE.txt) · [Exact original text](ORIGINAL_CE3.txt) · [Alternative displayed text](SOURCE_BOUND_ALTERNATIVE.txt) · [Caption offsets and hashes](PROVENANCE.json)','',
             '## Source-relative assessment','',
             f"- Unsupported visual claims: {a['unsupported']}",f"- Cross-caption fusion: {a['fusion']}",
             f"- Attribution: {a['attribution']}",f"- Visual coverage / information loss: {a['coverage']}",
             f"- Redundancy / readability / regression: {a['tradeoff']}",f"- Provisional interpretation: {a['judgment']}",'',
             '## Original visual body','', '```text',r['original_text'].split('[VISUAL_EVIDENCE]',1)[1].strip(),'```','',
             '## Alternative visual body','', '```text',r['alternative_text'].split('[VISUAL_EVIDENCE]',1)[1].strip() or '(empty visual body; this parenthetical explanation is not part of the alternative output)','```','',
             '## Exact source-to-output evidence','']
        for q in quotes:
            doc.extend([f"Baseline quotation at original characters [{q['original_char_start']}, {q['original_char_end']}): `{q['original_quote']}`",''])
            if not q['related_frozen_captions']:doc.extend(['No selected Filtered caption exists; no visual assertion can be supported from these inputs.',''])
            for c in q['related_frozen_captions']:
                doc.extend([f"- `{c['caption_id']}`, source characters [{c['source_char_start']}, {c['source_char_end']}), alternative characters [{c['alternative_char_start']}, {c['alternative_char_end']}): `{c['text']}`"])
            doc.append('')
        doc.extend(['All selected captions and both character/UTF-8 byte ranges appear in PROVENANCE.json. Caption IDs identify source blocks, not authenticated photo IDs. No original photos were opened. A caption copied verbatim may still be factually wrong.','',
                    'The complete REVIEW_CONSENSUS prefix is byte-identical in both text files. Missing Core, opinion balance and review repetition cannot improve through this replacement.'])
        path=OUT/'cases'/r['business_id']/'COMPARISON.md'
        path.write_text('\n'.join(doc)+'\n',encoding='utf-8')
        case_docs.append(path)
    write('PROVISIONAL_CASE_ASSESSMENTS.json',paired)
    (OUT/'CASE_LEVEL_COMPARISON.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    tests=['all captions once in order including duplicates','ambiguous headers rejected',
           'order/body modification rejected','non-TRAIN/non-Filtered rejected','review bytes and CRLF preserved',
           'Unicode character/UTF8 offsets recover exact text','wrong source hash rejected','zero-caption abstention']
    write('TEST_RESULTS.json',dict(status='PASS',passed=8,failed=0,tests=tests,
        command='.venv-caption-pilot/Scripts/python.exe -m unittest discover -s scripts -p test_source_bound_visual_diagnostic_v1.py -v',
        actual_reported_elapsed_seconds=0.007,scope='Standard-library synthetic string/provenance tests only; no model or GPU',
        input_integration_checks='12 saved N1 raw-token identities; 12 byte-identical review prefixes; 37 exact caption traces; 3 empty zero-caption visual bodies'))
    report="""# Source-Bound Visual Evidence — Bounded Offline Diagnostic V1

**Recommendation: NO-GO for automatic adoption as a replacement summary policy or as evidence of improved LED model quality. Retain the result as a useful offline diagnostic control and stop for researcher review.** It repairs clear caption-relative attribution errors, but does not establish a net improvement in complete summaries. No further experiment is initiated.

## Scope and frozen baseline

Exactly the existing 12 N-best Phase A TRAIN businesses were used: 9 captioned and 3 zero-caption, with 37 selected Filtered captions. There was no artifact shortage, resampling, new prediction, GPU call, optimizer update, TEST/validation access, reference creation or human-evaluation activity. The existing selection was deterministic from source metadata and deliberately oversampled zero-caption cases; it is not a population estimate or independent generalization benchmark.

Baseline: saved N1 outputs from Original CE DecoderPolicyV2 epoch 3 / step 750, `allenai/led-large-16384`, adapter SHA-256 `507dcdd92333dc0625559570f06140b84a14eb202191779ebd03393224090a60`. The previously accepted source-only N-best bundle passed **171/171 file hashes**. Each N1 is bound to its exact source, raw-token array and successful historical call receipt. No model weights were loaded; adapter identity is checked against archived integrity receipts, not re-measured by loading the checkpoint in this diagnostic.

## Fixed transformation and checks

The entire original prefix through the VISUAL_EVIDENCE header/newline is preserved, including **REVIEW_CONSENSUS bytes in all 12 cases**. Every caption is copied once, unchanged and in original order, to a separate line `Caption N: <exact selected caption>`. No object correction, semantic deduplication, shortening, merging or relation inference occurs. The label denotes an original IMG_CAP_N block, not an independently authenticated photograph. The three zero-caption visual bodies are empty; the header remains, with no generated assertion or invented explanation.

All 37 caption source/output character ranges, UTF-8 byte ranges and caption text hashes verify. Eight focused regression tests passed, including review-byte preservation, CRLF, Unicode offsets, duplicate-caption retention, malformed source rejection and zero-caption abstention. Used frozen artifacts hash-identically before and after. No Teacher target was opened or changed.

One initial offline assembly check stopped before creating any alternative: it incorrectly expected raw evidence status PASS. The historical schema uses EVIDENCE_FLUSHED plus a separate PASS receipt. The corrected diagnostic requires both receipts, empty errors, RNG PASS, exact tokens and source/checkpoint bindings; no historical file or model execution was changed. See PREPARATION_CORRECTION.json.

## Findings from every case

See [the 12-case table](CASE_LEVEL_COMPARISON.md) and linked case reports for originals, alternatives and exact evidence. Assessments below are **identity-aware, model-assisted and provisional**, not blinded judgments or researcher-confirmed quality labels.

- **Three zero-caption cases:** Tempo, Mesa and Terra Mar originally asserted photographic objects/relations with no selected caption evidence. All visual assertions are removed in the alternatives. This establishes caption-evidence compliance, not that the removed objects are absent from actual photos.
- **Five captioned cases show clear source-relative repairs:** Tujague's (woman action and drink-on-drink), Habit (tenders inside sandwich and tray/sign placement), Little Gourmand (pastries acquire cakes' display relation), Kobe (sushi acquires drink's bar location; unsupported chef), Opal (drink/food become a dessert and table group moves to bar). Separate caption blocks prevent these particular generated fusions from being displayed. Their true internal neural causes are not demonstrated.
- **Four captioned cases do not establish a clear hallucination-repair gain:** DI Coffee and Whistlers mainly reduce repetitive micro-sentences; P J adds tray/sports-bar detail at a fluency cost; Minerva already has a supported, readable description and is a likely display-quality regression after labels/fragments are added.
- **No selected caption content is dropped:** all 37 texts survive exactly. Some previously omitted caption details return, but this is mechanical retention, not demonstrated salience selection. All-caption copying can retain redundant, generic or irrelevant details.
- **Review quality does not change:** missing important Core details, incorrect mixed-opinion synthesis and review repetition remain exactly as in CE3. This intervention cannot repair them.

## Representative before / after

**Kobe:** CE3 says `A long sushi roll sits on top of a bar.` CAP2 describes a tall drink on a bar; CAP3 describes a sushi roll on a black plate. The alternative displays those two captions separately and adds no roll/bar relation. It also stops introducing a chef not named in CAP1. Full text/offsets: [Kobe](cases/gTKoJH4BWX67pVu-bRdsmw/COMPARISON.md).

**Habit:** CE3 says `The sandwich contains chicken tenders.` and `The tray is in front of a restaurant sign.` CAP1 lists tenders, fries and a sandwich on a tray; CAP2 places a sign in front of a building. Separate copied captions remove the added ingredient and tray/sign relations. Full evidence: [Habit](cases/5uhG34VRsD0A6lOY8F41Dg/COMPARISON.md).

**Tempo:** with zero captions, CE3 says `The taco is next to a bell.` The alternative leaves visual body empty. The review remains unchanged: [Tempo](cases/EzJI5x25U5GgbYBWhu5ZQQ/COMPARISON.md).

**Minerva — explicit worse/tradeoff example:** CE3 already describes a circular sign, plate and background person consistent with both captions. Replacing this with two labeled lower-case fragments adds no material coverage, increases 25 to 28 words, and weakens prose fluency. See [Minerva](cases/h0tUOsL3sRYVqSQlxWjW5g/COMPARISON.md).

## Length, redundancy and readability

| Group | Original visual words | Alternative visual words, including labels | Shorter / equal / longer cases |
|---|---:|---:|---:|
| All 12 | 520 | 493 | 5 / 1 / 6 |
| Captioned 9 | 476 | 493 | 2 / 1 / 6 |
| Zero-caption 3 | 44 | 0 | 3 / 0 / 0 |

These are Unicode word counts, not tokenizer lengths or quality scores. Raw caption text alone has 419 words; source labels add 74 words. The overall reduction is driven by zero-caption abstention and does **not** establish that captioned summaries become shorter. Captioned total length rises about 3.6%, and 6/9 become longer. DI and Whistlers are less repetitive, but caption-internal repetitions (e.g., bottles / on top), overlapping observations and lower-case fragments remain unchanged. Opal rises from 54 to 70 words. No supported content is lost relative to captions; actual photographic omissions cannot be evaluated from this source-only packet.

## What this diagnoses, and what it does not

**Confirmed implementation behavior:** the supplied CE3 text sometimes adds claims or relations absent from its supplied captions; source-bound display eliminates those particular additions by construction. It also enforces zero-caption abstention. This exposes a generation/output-policy weakness relative to the model-visible evidence.

**Caption-quality uncertainty:** copying provides no independent support for BLIP-2 object labels, actions, counts or relationships. In particular, food identities such as Little Gourmand's chocolate milkshake and Habit's tenders must remain unverified; this diagnostic establishes no specific caption error. Original photos were not inspected and no photo identity was reconstructed. Caption faithfulness and photographic truth are different measures. A false caption would remain false in this display.

**Unproven:** global factual accuracy, a population hallucination rate, human usability preference, learned grounding improvements, better caption filtering and improved review synthesis. The visible alternatives are **postprocessed display text, not LED predictions**. No loss, reference score or simulated human rating is used to imply otherwise.

## Decision and limits

There is useful evidence for a narrowly scoped display-level grounding control, particularly for the three zero-caption cases and five clear caption-relative error cases. There is insufficient evidence for unconditional adoption: it leaves the main review weaknesses intact, sometimes worsens readability, can lengthen visual sections, and faithfully propagates unverified caption content. The chosen sample is small, previously examined and source-stratified; qualitative counts are not independently adjudicated.

**NO-GO for promotion/default adoption on this evidence; diagnostic objective completed.** Preserve Original CE3 as baseline. Submit this package for researcher review, without another technique, inference run, model change or thesis-stage continuation.

## Portable evidence

- `ALL_PAIRED_OUTPUTS.json`: all 12 full original and alternative texts; original raw token IDs; explicit non-model alternative provenance.
- `cases/<business_id>/`: exact original/alternative text, frozen model-visible source, offsets/hashes, and case comparison.
- `CAPTION_PROVENANCE.json`, `PROVISIONAL_CASE_ASSESSMENTS.json`: exact source/output traces and quoted baseline evidence.
- `INPUT_INTEGRITY.json`, `AUTOMATIC_CHECKS.json`, `TEST_RESULTS.json`, `POLICY.json`: technical results and scope.
- `EXISTING_EVIDENCE/`: byte copies of relevant historical source, candidate and checkpoint metadata plus 12 N1 raw/PASS receipt pairs. The historical full manifest is included for provenance, not a claim that all historical files are embedded here.
- `SHA256_MANIFEST.json`: relative hashes for every included file except the manifest itself. No checkpoint weights or large execution archive are needed for this diagnostic review.
"""
    (OUT/'SOURCE_BOUND_VISUAL_DIAGNOSTIC_REPORT.md').write_text(report,encoding='utf-8')
    for name in ('test_source_bound_visual_diagnostic_v1.py','assemble_source_bound_visual_report_v1.py'):
        shutil.copyfile(ROOT/'scripts'/name,OUT/name)
    write('DIAGNOSTIC_COMPLETION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='OFFLINE_DIAGNOSTIC_COMPLETED_STOP_FOR_REVIEW',cases=12,selected_captions=37,
        original_quotation_offsets_verified=quote_count,new_model_calls=0,optimizer_updates=0,GPU_operations=0,
        validation_or_test_access=False,reference_creation_or_human_evaluation=False,
        recommendation='NO_GO_FOR_DEFAULT_ADOPTION_OR_MODEL_IMPROVEMENT_CLAIM',
        quality_assessments='MODEL_ASSISTED_PROVISIONAL_NOT_RESEARCHER_CONFIRMED',
        no_automatic_next_experiment=True))
    # Independent read-back of every copied source, output prefix, caption range and quotation.
    for r,a in zip(rows,paired):
        folder=OUT/'cases'/r['business_id']
        original=(folder/'ORIGINAL_CE3.txt').read_bytes();alternative=(folder/'SOURCE_BOUND_ALTERNATIVE.txt').read_bytes()
        source=(folder/'SOURCE_MODEL_VISIBLE.txt').read_bytes()
        assert original==r['original_text'].encode() and alternative==r['alternative_text'].encode()
        assert sha(source)==r['source_sha256']
        assert original.split(b'[VISUAL_EVIDENCE]')[0]==alternative.split(b'[VISUAL_EVIDENCE]')[0]
        for c in r['caption_trace']:
            assert source[c['source_UTF8_byte_start']:c['source_UTF8_byte_end']]==c['text'].encode()
            assert alternative[c['alternative_UTF8_byte_start']:c['alternative_UTF8_byte_end']]==c['text'].encode()
        for q in a['evidence_quotes']:
            assert original[q['original_UTF8_byte_start']:q['original_UTF8_byte_end']]==q['original_quote'].encode()
    manifest={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
    write('SHA256_MANIFEST.json',dict(algorithm='SHA-256',self_excluded=True,files=manifest))
    target=ROOT/'outputs/cloud_handoff/SOURCE_BOUND_VISUAL_DIAGNOSTIC_V1_REVIEW.zip'
    assert not target.exists(),'Do not overwrite a review archive'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        inside=json.loads(z.read('SHA256_MANIFEST.json'))['files']
        assert set(z.namelist())==set(inside)|{'SHA256_MANIFEST.json'}
        for rel,digest in inside.items():assert sha(z.read(rel))==digest,rel
    digest=sha(target.read_bytes())
    with Path(str(target)+'.sha256').open('x',encoding='ascii') as f:f.write(digest+'  '+target.name+'\n')
    print(json.dumps(dict(report=str(OUT/'SOURCE_BOUND_VISUAL_DIAGNOSTIC_REPORT.md'),
                         zip=str(target),zip_bytes=target.stat().st_size,sha256=digest,
                         portable_manifest_entries=len(manifest),quotation_offsets=quote_count),indent=2))


if __name__=='__main__':main()
