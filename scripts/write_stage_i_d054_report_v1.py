"""Summarize actual D054 evidence without inventing human quality scores."""
import json,statistics
from stage_i_d054 import OUT,read,write

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(c) for c in r)+' |' for r in rows])

def build():
    e=read(OUT/'AUTOMATIC_EVALUATION.json');completion=read(OUT/'GENERATION_COMPLETION.json')
    q=read(OUT/'CODEX_SOURCE_BASED_ASSESSMENT.json');per=e['per_output'];arms=['A_decoder_only','B_encoder_decoder'];lines=[]
    lines+=['# Stage I — D054 paired generation and quality evaluation','',
      '**Bounded execution completed. Researcher review pending.** Exactly48 predictions from the two existing D052 checkpoints,24 identical Filtered validation inputs each. No training,Teacher request,test access,decoding search,or accepted-artifact modification.','',
      '**Quality finding:** neither arm produces usable structured summaries on this set; small B information gains coexist with degeneration and grounding regressions. No overall architectural winner is established.','',
      'Read this report alongside the raw predictions and source-based assessment. Deterministic software checks establish technical integrity,not factual quality. Codex qualitative judgments are model-assisted and unconfirmed; no student-output human labels were fabricated.','',
      '## Execution integrity','',
      'The accepted strict D053 v2 validator and immutable reference manifest passed. All24 business IDs/order,selected review/photo mappings,source strings and tokenizer IDs match. Both models were fully checked before the first inference:295 pretrained backbone parameter fingerprints each; exact safetensors and adapter configs; intended24/48 LoRA modules and294912/589824 adapter parameters. All checkpoint/reference/config/input protected hashes and actual adapter/backbone tensors were unchanged after inference.','',
      'Backbone allenai/led-base-16384 revision38335783885b338d93791936c54bb4be46bebed9. Local RTX3050 6GB. torch2.1.0+cu118,transformers4.51.3,peft0.15.2. Original FP32 weights with FP16 CUDA autocast,batch1,TF32 disabled,global attention on token0 and original1024 padding multiple. Content source lengths1114–4193; maximum padded length5120 is padding,not source truncation.','',
      'Effective decoding was identical: num_beams4,do_sample=false,no_repeat_ngram_size0,max_length768,max_new_tokens=null,length_penalty1,repetition_penalty1,early_stopping=true,forced_eos_token_id2,use_cache=true. Complete effective configs are preserved per model and prediction. max_length includes decoder-start; the initial token2 is not treated as an ending EOS. No output text was cleaned,repaired or regenerated.','']
    lines+=[table(['Arm','Modules','Adapter parameters','Measured generation minutes','Peak allocated MiB','Peak reserved MiB'],[
      [a,24 if a.startswith('A_') else 48,294912 if a.startswith('A_') else 589824,round(r['seconds']/60,2),round(r['peak_allocated_bytes']/2**20,1),round(r['peak_reserved_bytes']/2**20,1)] for a in arms for r in completion['arms'] if r['arm']==a]),'',
      'Times above are measured arm generation loops (including ordinary per-case saving),not an estimate of total interactive review/preflight time. No runtime speedup setting or GPU fallback was introduced.','',
      '## Deterministic results — random20 only','']
    selected={a:[r['automatic'] for r in per if r['arm']==a and r['subset']=='random'] for a in arms}
    rows=[]
    def pair(label,f):rows.append([label]+[f(selected[a]) for a in arms])
    pair('Exact repeated-sentence outputs /20',lambda rr:sum(r['repetition']['exact_repeated_sentence'] for r in rr))
    pair('Adjacent repeated-word outputs /20',lambda rr:sum(r['repetition']['consecutive_repeated_word'] for r in rr))
    pair('Mean duplicate sentence fraction',lambda rr:round(statistics.mean(r['repetition']['duplicate_sentence_fraction'] for r in rr),4))
    pair('Mean adjacent repeated-word pair fraction',lambda rr:round(statistics.mean(r['repetition']['consecutive_repeated_word_pair_fraction'] for r in rr),4))
    pair('Compliant required section structure /20',lambda rr:sum(r['structure']['section_headers_compliant'] for r in rr))
    pair('Length min / mean / max (includes start)',lambda rr:' / '.join(str(round(v,2)) for v in [min(r['termination']['generated_tokens_including_decoder_start'] for r in rr),statistics.mean(r['termination']['generated_tokens_including_decoder_start'] for r in rr),max(r['termination']['generated_tokens_including_decoder_start'] for r in rr)]))
    pair('768-token cap hits /20',lambda rr:sum(r['termination']['cap_reached'] for r in rr))
    pair('EOS before cap /20',lambda rr:sum(r['termination']['eos_before_length_cap'] for r in rr))
    pair('EOS present after initial decoder-start /20',lambda rr:sum(r['termination']['eos_after_decoder_start'] for r in rr))
    lines+=[table(['Measure','A decoder-only','B encoder+decoder'],rows),'',
      'A low repeated-sentence count can result from one giant marker loop or no substantive text. Empty lexical n-gram denominators are N/A,not perfect diversity. See exact generated-BPE and lexical2/3/4-gram counts,opportunities and missingness in AUTOMATIC_EVALUATION.json. Early EOS can terminate nonsense or whitespace; it does not certify completeness. Forced EOS at token768 is reported separately.','',
      '### Paired random-business uncertainty','',
      'Unchanged D049 bootstrap: business pairs,NumPy default_rng seed20261016,5000 resamples,95% percentile interval. B−A; descriptive length/EOS changes are not quality wins. Metric-specific complete-pair counts matter when lexical opportunities are absent.','']
    names=['auto.repetition.duplicate_sentence_fraction','auto.repetition.consecutive_repeated_word_pair_fraction','auto.repetition.lexical_ngrams.4.duplicate_fraction','auto.repetition.generated_bpe_ngrams.4.duplicate_fraction','auto.structure.section_headers_compliant','auto.termination.cap_reached','auto.termination.generated_tokens_including_decoder_start']
    pp=e['paired']['random']['paired_metrics_B_minus_A'];rows=[]
    for name in names:
        if name not in pp:continue
        r=pp[name];ci=r['bootstrap']['interval'];rows.append([name,r['complete_pairs'],r['A_macro_mean'],r['B_macro_mean'],r['mean_paired_difference'],ci])
    lines+=[table(['Metric','n pairs','A mean','B mean','B−A','95% interval'],rows),'',
      'Full metric table and per-business changes are in PAIRED_COMPARISON.json and PAIRED_PER_BUSINESS.csv. There is no composite score,automatic architectural winner or targeted-case inferential CI.','',
      '## Targeted4 — separate diagnostic evidence','']
    ids=list(dict.fromkeys(r['business_id'] for r in per if r['subset']=='targeted'));rows=[]
    for b in ids:
        row=[b]
        for a in arms:
            r=next(r['automatic'] for r in per if r['business_id']==b and r['arm']==a)
            row += [r['termination']['generated_tokens_including_decoder_start'],r['structure']['section_headers_compliant'],r['structure']['empty_visual_structurally_compliant']]
        rows.append(row)
    lines+=[table(['Business','A tokens','A structure','A zero-caption','B tokens','B structure','B zero-caption'],rows),'',
      'j3kGicU3mPG_ciIKG1Vuyg is the sole zero-caption stress case. Missing/malformed headers fail the approved empty-visual rule even if no clear visual object assertion is present. Marker failures are not silently converted to an empty valid visual section. Targeted cases are never pooled into random20 prevalence.','',
      '## Informativeness and grounding assessment','',
      'SOURCE_BASED_QUALITY_ASSESSMENT.md and CODEX_SOURCE_BASED_ASSESSMENT.json inspect all48 outputs with exact representative output spans,REV/caption/photo bindings,source quotations and Core omission examples. They are explicitly **Codex-assisted judgments,not independently human-confirmed output annotations**. Sparse demonstrated reference matches are examples/lower bounds,not a complete coverage denominator or exhaustive precision score.','',
      'The source references remain261 Core315 Supporting;88 exclusions and two uncertain parent decisions are audit-only. Core importance does not require every atom verbatim,and Supporting omission alone is not an error. All full original reviews remain permitted evidence even when a detail was not retained as a scoring reference. Review opinions are not objective restaurant truths.','',
      'Formal human Core/Supporting coverage,review/visual support precision,uncertainty sensitivity and semantic-proposition redundancy remain N/A until actual complete output annotations are confirmed. This is not a claim of zero errors. Generic correct fragments are separated from useful restaurant-specific information. No new factual scorer,embedding model or similarity cutoff was fitted.','']
    lines+=['### Paired source-based observations (model-assisted)','']
    lines+=[table(['Business','Subset','Inspection conclusion'],[[r['business_id'],next(p['subset'] for p in per if p['business_id']==r['business_id']),r['assessment']] for r in q['paired_case_assessments']]),'',
      'See the per-business assessment for full source quotations and representative claim/reference links. Raw48 outputs remain available without sentence deduplication. SOURCE_GROUNDING_DOSSIER.json provides verbatim segment offsets and all source evidence; segment grouping is a reading aid,not atomic truth.','',
      '## Interpretation and recommendation','',
      (OUT/'D054_RESEARCH_INTERPRETATION.md').read_text(encoding='utf-8').strip(),'','## Review and reproducibility','',
      'Open D054_BLINDED_REVIEW.html inside the extracted review ZIP. X/Y uses the unchanged predeclared left/right mapping. The page contains complete reviews,LED-visible captions,original photographs,Core/Supporting references and raw outputs. Arm identity is kept in private/; reports and raw execution files unblind the comparison. Optional notes export/import stays pending and does not create formal human scores.','',
      'The ZIP includes the complete accepted D053 reference proof,the frozen generation/evaluation configurations,relevant source code,48 raw per-case artifacts,generated IDs,pre/post integrity records,automatic metrics,model-assisted inspection and blank researcher output-annotation shells. Run python code/verify_stage_i_d054_bundle.py . after extraction to verify hashes,source/reference bindings and reproduce metrics/paired bootstrap. No checkpoint weights/GPU/API/original project are needed for offline verification.','',
      'The external ZIP receipt and extracted-check report identify the exact delivered archive. Original D053 files retain their historical authorization wording; current D054 authorization is separately preserved. No full retraining,extra decoding,Teacher regeneration or held-out evaluation has been started. Stop for independent researcher review.','']
    markdown='\n'.join(lines)
    for old,new in [('Exactly48','Exactly 48'),('All24','All 24'),('all48','all 48'),('random20','random 20'),('Targeted4','Targeted 4'),('targeted4','targeted 4'),('295 pretrained','295 pretrained'),('intended24/48','intended 24/48'),('and294912/589824','and 294,912/589,824'),('1114–4193','1,114–4,193'),('maximum padded length5120','maximum padded length 5,120'),('beam4','beam 4'),('num_beams4','num_beams=4'),('no_repeat_ngram_size0','no_repeat_ngram_size=0'),('max_length768','max_length=768'),('length_penalty1','length_penalty=1'),('repetition_penalty1','repetition_penalty=1'),('forced_eos_token_id2','forced_eos_token_id=2'),('random20','random 20'),('business pairs,NumPy','business pairs, NumPy'),('seed20261016','seed 20261016'),('5000 resamples','5,000 resamples'),('all48','all 48'),('261 Core315 Supporting','261 Core + 315 Supporting'),('576 approved','576 approved'),('with88','with 88'),('and2 uncertain','and 2 uncertain'),('and24','and 24'),('the768','the 768'),('512 training','512 training'),('outputs,24','outputs, 24')]:
        markdown=markdown.replace(old,new)
    write(OUT/'STAGE_I_D054_PAIRED_GENERATION_REPORT.md',markdown)

if __name__=='__main__':build()
