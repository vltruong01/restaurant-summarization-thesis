"""Report, validate and package actual Stage O preparation; never launches a model."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,csv,hashlib,html,importlib.metadata,json,shutil,subprocess,sys,tempfile,zipfile
from PIL import Image,ImageOps
import psutil
from stage_o_tripadvisor_feasibility_v1 import ROOT,OUT,save,digest,utc
from stage_o_frozen_pipeline_v1 import verify_freeze,jl
from photo_integrity_audit import pixel_digest,phash

PKG=OUT/'REVIEW_PACKAGE'


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n',encoding='utf8')


def copy(path,relative):
    dest=PKG/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)


def source_diagnostics(frozen):
    yelp_path=ROOT/'outputs/stage_d/photo_integrity_duplicate_audit_v1/image_manifest.csv'
    with yelp_path.open(encoding='utf8',newline='') as f:yelp_pixels={r['pixel_sha256'] for r in csv.DictReader(f) if r['pixel_sha256']}
    rows=[];matches=[]
    for item in frozen['restaurants']:
        s=json.loads((OUT/item['source_file']).read_text(encoding='utf8'))
        for p in s['photos']:
            path=OUT/p['file'];assert digest(path)==p['sha256']
            with Image.open(path) as im:
                im.load();rgb=ImageOps.exif_transpose(im).convert('RGB');pix=pixel_digest(rgb);perceptual,_,_=phash(rgb)
                rows.append(dict(business_id=item['business_id'],photo_id=p['photo_id'],pixel_sha256=pix,phash64_hex=perceptual,width=rgb.width,height=rgb.height))
                if pix in yelp_pixels:matches.append(p['photo_id'])
    assert not matches,'STOP: identifiable Yelp pixel overlap; no automatic cohort replacement.'
    groups=defaultdict(set)
    for r in rows:groups[r['pixel_sha256']].add(r['business_id'])
    assert not any(len(v)>1 for v in groups.values())
    result=dict(status='PASS_EXACT_BYTE_AND_NATIVE_NORMALIZED_PIXEL_SOURCE_CHECKS',photos=rows,Yelp_normalized_pixel_matches=matches,
                native_pixel_and_phash_code_sha256=digest(ROOT/'scripts/photo_integrity_audit.py'),Yelp_image_manifest_sha256=digest(yelp_path),
                pixel_matching_scope='Exact EXIF-transposed RGB dimensions+pixels; not semantic/perceptual near-duplicate proof.',
                pHash='Recorded diagnostics only, never used to reject or select photos.',cross_selected_restaurant_exact_pixel_groups=0,
                sizes=dict(min_width=min(r['width'] for r in rows),max_width=max(r['width'] for r in rows),min_height=min(r['height'] for r in rows),max_height=max(r['height'] for r in rows)))
    save(OUT/'SOURCE_IMAGE_DIAGNOSTICS.json',result)
    return result


def resources():
    lock=json.loads((ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json').read_text(encoding='utf8'))
    models={}
    for c,r in lock['models'].items():
        p=ROOT/r['path']/'adapter_model.safetensors';exists=p.is_file();actual=digest(p) if exists else None
        models[c]=dict(file=p.relative_to(ROOT).as_posix(),exists=exists,expected_sha256=r['adapter_sha256'],actual_sha256=actual,verified=actual==r['adapter_sha256'],initialization_adapter_sha256=r['initialization_adapter_sha256'])
    backbone=ROOT/'outputs/cloud_handoff/LED_LARGE_DECODER_POLICY_V2_WORKSPACE/assets/models/led-large-16384/pytorch_model.bin'
    b=dict(exists=backbone.is_file(),expected_sha256=lock['backbone']['sha256'],actual_sha256=digest(backbone) if backbone.exists() else None)
    b['verified']=b['expected_sha256']==b['actual_sha256']
    cache=ROOT/'assets/model_cache/models--Salesforce--blip2-opt-2.7b-coco/snapshots/cda95b9319b722f79c9451a3d8ff92eea02048dc'
    index=json.loads((cache/'model.safetensors.index.json').read_text(encoding='utf8'))
    missing=[p for p in sorted(set(index['weight_map'].values())) if not (cache/p).is_file()]
    scorer=json.loads((ROOT/'outputs/stage_f/validation_calibration_v1/model_files_report.json').read_text(encoding='utf8'))
    clip=ROOT/scorer['CLIP_local_safetensors']/'model.safetensors';sbert=ROOT/scorer['SBERT_local_snapshot']/'model.safetensors'
    available=psutil.virtual_memory().available/2**30
    report=dict(observed_at_utc=utc(),GPU_nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'],text=True).strip(),
                RAM_total_GiB=psutil.virtual_memory().total/2**30,RAM_available_GiB=available,BLIP_frozen_minimum_available_RAM_GiB=8,
                frozen_RAM_profile_currently_satisfied=available>=8,free_disk_GiB=shutil.disk_usage(ROOT).free/2**30,
                BLIP_model='Salesforce/blip2-opt-2.7b-coco',BLIP_revision='cda95b9319b722f79c9451a3d8ff92eea02048dc',
                BLIP_index_total_weight_bytes=index['metadata']['total_size'],BLIP_missing_weight_shards=missing,
                CLIP_weights_available=clip.is_file(),SBERT_weights_available=sbert.is_file(),LED_backbone=b,LED_models=models,
                new_GPU_model_calls=0,new_model_downloads=0,new_paid_instances=0,paid_cost_actual_USD=0,
                decision='STOP_BEFORE_CAPTION_AND_LED_INFERENCE_MISSING_DEPENDENCIES',
                note='Existing Stage K receipts prove LED beam4 inference previously fit this RTX3050 (peak 2.92–3.19GB). This is not a diagnosis of a broken GPU.')
    assert all(r['verified'] for r in models.values()) and b['verified'],'Frozen LED artifacts missing/corrupt; report rather than substitute.'
    save(OUT/'RESOURCE_AND_MODEL_INTEGRITY.json',report)
    save(OUT/'FROZEN_MODEL_AND_DECODER_BINDINGS.json',dict(models=models,backbone=b,generation=lock['generation'],source_lock_sha256=digest(ROOT/'outputs/final_evaluation/s1000_preparation_v1/MODEL_AND_DECODER_LOCK.json')))
    return report


def human_app():
    base=OUT/'HUMAN_APP';code=base/'scripts';web=code/'stage_o_human_web_v1';web.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'scripts/stage_n_human_evaluation_v1.py').read_text(encoding='utf8')
    assert source.count("['RO', 'U', 'F', 'S']")==2
    adapted=source.replace("['RO', 'U', 'F', 'S']","['RO', 'U', 'F']").replace('Complete240','Complete90').replace('stage_n_session','stage_o_session').replace('stage_n_human_web_v1','stage_o_human_web_v1').replace('default=8766','default=8767').replace('TEST outputs','TripAdvisor outputs')
    write(code/'stage_o_human_evaluation_v1.py',adapted)
    for name in ['stage_l_reference_app_v1.py','s1000_final_eval_core_v1.py']:shutil.copyfile(ROOT/'scripts'/name,code/name)
    for name in ['index.html','app.js','app.css']:
        text=(ROOT/'scripts/stage_n_human_web_v1'/name).read_text(encoding='utf8').replace('Stage N','Stage O').replace('STAGE N · TEST30','STAGE O · TRIPADVISOR15').replace('stage_n_draft:','stage_o_draft:')
        write(web/name,text)
    save(base/'PREPARATION_STATUS.json',dict(state='APPLICATION_CODE_ADAPTED_REAL_OUTPUTS_PENDING',conditions=['RO','U','F'],restaurants=15,outputs_per_rater=45,raters=2,planned_records=90,
            real_outputs=0,real_rating_records=0,live_release_created=False,accounts_created=False,blind_mapping_created=False,
            tested='Python syntax and original invariant adaptations. Full Stage O HTTP/login/persistence test is pending real output workspace; Stage N implementation is reused.',
            no_simulated_human_ratings=True))
    write(base/'README.md','''# Prepared reference-free evaluation application

The accepted Stage N source/photo viewer, six-criterion rubric, password hashing, sessions, comments, draft persistence, submit-lock and export code are reused. The new copy has three conditions and a separate Stage O cookie/draft namespace and default port 8767. Original Stage N files and ratings are unchanged.

**Current state: code prepared; outputs absent; no live workspace, accounts, randomization key, release, ratings or approvals exist. Do not start collecting ratings.** Once genuine outputs exist, `scripts/prepare_stage_o_human_workspace_v1.py` in the original repository verifies all 75 native components and prepares 90 independently randomized aliases. It refuses missing outputs and refuses to overwrite an existing study. It does not create a live release. The researcher must inspect this package first.

After that review and a separately documented live release, the coordinator can provision users using the adapted application's existing interactive `add-user` command and start its server on `127.0.0.1:8767`. No account/password from Stage N is reused or shipped. The application can export partial or complete real records; a final export requires all 90 genuine submissions and a valid journal. Visual is structurally N/A for RO; an empty applicable U/F Visual is scored normally.

Full Stage O end-to-end HTTP/authentication/persistence testing remains pending real outputs. Source-viewing code is reused, not claimed newly validated with nonexistent summaries. The prepared original-repository command is:

```console
python scripts/prepare_stage_o_human_workspace_v1.py
```
''')


def reports(frozen,resource,diagnostic):
    protocol=OUT/'PROTOCOL';protocol.mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'outputs/stage_n/final_human_statistical_analysis_20261102_v1/PACKAGE/PROVENANCE/RUBRIC.json',protocol/'RUBRIC.json')
    write(protocol/'HUMAN_EVALUATION_PROTOCOL.md',f'''# Stage O exploratory reference-free human evaluation — protocol before outputs

## Scope and freeze

External cohort: 15 distinct restaurants from the publisher's Gijon-area v2 archive. Cohort SHA-256: `{digest(OUT/'COHORT15_FREEZE.json')}`. Review selection is frozen before any LED output. This small, city-specific sample does not replace Yelp TEST100/TEST30. Generalization cannot conclusively rule out Teacher style dependence or public-pretraining exposure.

## Design

RO/U/F final S1000 adapters only; 45 complete outputs, 45 items independently randomized for each of two evaluators (90 real rating records). No Shuffled, Qwen, training, GPT Teacher targets or reference summaries. Aliases and separate randomization seeds are coordinator-only. Model names, automatic scores and the other evaluator's ratings are hidden. RO's missing Visual component can reveal its structural condition; perfect model blindness cannot be guaranteed.

Use exactly the accepted six 1–5 criteria and anchors in RUBRIC.json. Review Faithfulness, Aspect Coverage and Usefulness judge Review; Coherence and Non-Redundancy judge all applicable text; Visual Grounding & Contribution judges Visual. Visual is N/A for RO, never zero. Applicable empty U/F Visual remains scoreable. No new aggregate composite score.

## Evidence and workload

Show the same selected original reviews and ALL frozen accessible original-source photographs for a restaurant across conditions. The cohort has {frozen['source_photo_count']:,} photographs, including a 2,040-photo venue. Source evidence is not silently reduced to five photos for annotation. Lazy image loading and expandable galleries reuse Stage N. Full-image browsing is potentially substantial; changes to evidence scope would need explicit approval. Captions are hidden from evaluators so photographs, not caption assertions, ground Visual judgments. No reference summary is needed.

Keep original outputs, generation failures, repetitions and generic templates. Do not regenerate selectively or remove difficult restaurants. For serious defects record the exact output span, valid recipient review/photo IDs, short explanation and confirmed/uncertain status. Ordinary comments may be blank. Raters work independently, with truthful prior-exposure/conflict declarations. Recorded declarations do not independently authenticate identity or source inspection.

## Future analysis (not run now)

Retain all 15 businesses and both original raters. Report per-condition means, SD/median/IQR, rating distributions and condition-specific quadratic-weighted kappa. For F–RO Review and F–U applicable criteria, average the two raters within business, then use the existing Stage N 20,000 paired-business bootstrap and 50,000 two-sided sign flips (+1), seed 20261028. Report pointwise 95% CIs and raw exploratory p-values with no confirmatory winner claim or post-hoc exclusions. Business n=15, not 90 independent observations. Report generations and source inspection limitations separately from mimicry conclusions.

## Stop boundary

Return the preparation package for researcher/ChatGPT inspection before any human ratings. No live release, participant accounts or human decisions are created now. Upcoming Qwen model/protocol requires separate approval; the same frozen reviews/photos and eventual captions are retained for that comparison.
''')
    ledger=json.loads((OUT/'PREPROCESSING/REVIEW_BUDGET_LEDGER.json').read_text())
    write(OUT/'DATASET_FEASIBILITY.md',f'''# TripAdvisor dataset feasibility and eligibility

## Actual result

**PASS: frozen 15-restaurant cohort; 320 eligible English reviews, 247 selected whole reviews; {frozen['source_photo_count']:,} accessible source photographs. No model outputs exist.**

Inspected the original Zenodo records [v2](https://zenodo.org/records/14622324) and [v1](https://zenodo.org/records/5644892). V2 documents itemId/reviewId/language/image URL associations and provides five city ZIPs; v1 was reviewed at metadata/archive-listing level only. V2 Gijon, the smallest city archive, was sufficient; no v1 or other-city data were needed. This is a Gijon-area convenience frame with deterministic within-frame selection, not a five-city representative sample.

The downloaded v2 city archive has 857 restaurants and 82,142 reviews. All 1,832 publisher-en reviews were run through the existing seeded langdetect 1.0.9 English policy; 1,828 passed. Other publisher languages were not reclassified, so this is not an exhaustive discovery of every possible English review. Forty-nine restaurants met the initial review/linked-photo criteria. Candidate order was frozen as SHA256(`STAGE_O_V1|20261104|14622324|Gijon|<itemId>`), ascending. Seventeen candidates were actually checked to retain the first fifteen eligible. Two had all photo links unavailable and were excluded, not replaced after seeing model outputs. Full receipts and rejected cases are retained.

## Source identity and independence

Every review joins its original itemId to an existing restaurant; its publisher review URL carries the same itemId/reviewId. Photo URLs come from that review's original images field. Hex photo IDs canonicalize only publisher-listed URL variants; ambiguous multiple-restaurant photo IDs are excluded from eligibility and logged. Selected sources have at least ten verified-English reviews and five distinct accessible decoded photographs each. Association is verified from publisher records, not independent proof of photographer identity or every image's semantic venue authenticity.

No normalized name+city match with frozen Yelp; no downloaded-image byte match against its 28,264 unique raw hashes; no exact normalized RGB match against the native Yelp image audit; no exact cross-selected-restaurant RGB identity. Geographic/source URL evidence is retained. These checks do not prove absence of renamed venues, chains, perceptual near-duplicates or public-pretraining exposure. No Yelp split, prediction, reference or rating was modified.

## Image access qualification

The archive contains tables and image URLs, not JPEG binaries. We downloaded exactly the publisher-linked photo-l/photo-f response bytes and decoded all selected files, retaining URL, SHA-256, dimensions and review associations. These are original-source photographs for model/human grounding, **not attested camera-original/full-resolution uploads**. Observed width range: {diagnostic['sizes']['min_width']}–{diagnostic['sizes']['max_width']} px; height range: {diagnostic['sizes']['min_height']}–{diagnostic['sizes']['max_height']} px. Some source renderings are low resolution; this is an external-data limitation. No CDN URL upgrade, replacement image or generated image is used. pHash is diagnostic, never a rejection filter.

## Frozen schema conversion and CPU preparation

Ratings are exactly 10/20/30/40/50 → 1/2/3/4/5. Publisher dates are day precision; ISO dates are padded with 00:00:00 for the established serializer, explicitly not observed time-of-day. Original IDs, dates, ratings, titles and unchanged text are preserved. The model-visible Review uses the original review text; separate titles remain metadata. Cuisine types map to categories; missing state is empty. No synthetic metadata or summaries.

Existing v4.2 review selection and pinned tokenizer were called directly. Result: 247 selected reviews, maximum {max(x['selected_reviews'] for x in ledger)} per restaurant, maximum {max(x['review_block_tokens'] for x in ledger)} review-block tokens, maximum {max(x['review_task_tokens'] for x in ledger)} Review-task tokens. Every source is below 4,000 Review / 5,000 task tokens. The 750-token caption increment and K=5 remain frozen for later U/F construction. All 5,143 accessible photos are planned for BLIP-2 before K selection; no pre-caption cap was introduced.

## Limits and next step

Caption generation/filtering and 45 LED output records are **PENDING** because pinned caption/scorer weights are absent and the local RAM profile is below the existing offload minimum. Human ratings: 0/90; Qwen: not started. See RESOURCE_AND_MODEL_INTEGRITY.json and EXECUTION_PLAN_AND_COST.md. No paid resources were provisioned.

The record page did not supply an explicit license string in the inspected metadata. This package preserves provenance and does not assert a new redistribution license over reviews/photos. Users/author profile tables are excluded from the review package.
''')
    write(OUT/'EXECUTION_PLAN_AND_COST.md',f'''# Concrete execution plan — STOP for dependency/compute decision

## Actual blocker

Local GPU: {resource['GPU_nvidia_smi']}. Available RAM at audit: {resource['RAM_available_GiB']:.2f} GiB; accepted BLIP offload profile requires at least 8 GiB available. Both pinned BLIP-2 weight shards are missing (index total {resource['BLIP_index_total_weight_bytes']:,} bytes). CLIP and SBERT weight files are also missing. No CUDA model was loaded and no substitute, quantization, precision or threshold change was attempted. This is not a diagnosis that the laptop GPU is defective.

All three final S1000 adapters and the 1,839,633,783-byte LED backbone were independently rehashed and match the final lock. Existing Stage K receipts show the same RTX3050 already ran frozen LED inference at roughly 2.9–3.2 GB peak allocation. The blocker is caption/scorer assets and present offload RAM, not proof that LED requires another GPU.

## Two feasible routes requiring researcher decision

1. **Local, no new paid GPU:** restore about 16.5 GB of exactly pinned BLIP/CLIP/SBERT weights and free at least 8 GiB RAM (preferably more) for the unchanged GPU+CPU offload profile. Recheck runtime and asset hashes before one finite caption job. This avoids rental charges but can be slow; performance on this new 5,143-photo cohort has not been measured. No download has been initiated automatically.
2. **Faster, separately approved GPU:** one existing/approved 24 GB GPU, ≥32 GB host RAM and about 80 GB disk. Restore pinned assets, caption all 5,143 photos with original beam3/64-token/one-Basic-retry policy, score using frozen CLIP/SBERT, apply unchanged exact-pixel/relevance 0.08/redundancy 0.59/grouping 0.46/farthest-first/K5, then verify 750/5,000-token budgets and freeze the 75-call plan. Restore original RO/U/F S1000 in turn, native FP32 parameters + FP16 autocast, beam4/length penalty1/max_length768; preserve raw generation token IDs, input hashes and every failure. Assemble 45 records verbatim (15 RO,15 U,15 F). No Teacher, training, references or Qwen.

## Cost estimate, not a reservation or spending authorization

[Runpod's public pricing page](https://www.runpod.io/pricing), checked 4 November 2026 (Asia/Saigon), lists RTX3090 24GB at **USD0.50/hour**. [Vast pricing documentation](https://docs.vast.ai/guides/instances/pricing) explains its prices are dynamic and total cost includes compute, storage and bandwidth. Actual availability/offer must be rechecked before purchase; no old instance or authorization is reused.

Planning assumption, not a benchmark: about 0.8–3 seconds per photograph on a 24GB GPU implies 1.14–4.29 hours for first attempts; native Basic retries, setup/transfer and scoring add time. Plan **2–6 GPU hours**, about **USD1–3 compute** at USD0.50/hour, plus storage/transfer/tax. A proposed total ceiling of **USD5 / six hours**, without automatic renewal, leaves a small buffer; exceeding it would stop with preserved partial evidence. These are estimates and proposed limits, not approved spending. Prior local LED timing scales to roughly 20 minutes for 75 ordinary calls, but long outputs can take longer.

## Prepared code and remaining verification

CPU preparation actually ran: `python scripts/stage_o_frozen_pipeline_v1.py reviews-cpu`.
The caption and score/construct entry points reuse the accepted functions and are syntax-checked, but have **not** run with missing weights. They require a new explicit execution record bound to the cohort; no fabricated record is included. The finite LED runner adaptation and full Stage O GPU validation remain pending the selected compute route; the original accepted Stage K runner is included for review. Do not run it unchanged against historical TEST plans or reuse its old approval file.

After genuine outputs exist, the prepared human-workspace builder validates all 75 raw components and makes independently blinded 45-item assignments for each evaluator. Return this package for independent inspection first. No live rating release is created now.
''')
    write(OUT/'README.md','''# Stage O — TripAdvisor external evaluation preparation

**Current status: 15-restaurant cohort and CPU Review preparation complete; caption/scorer dependencies blocked; 0/45 generated outputs and 0/90 human ratings.**

Start with DATASET_FEASIBILITY.md, then COHORT15_FREEZE.json and EXECUTION_PLAN_AND_COST.md. SOURCE_READER.html opens offline with exact selected reviews and every individual source photograph. Each image remains a separate file. Large photo pools use lazy-loading; no source evidence is silently shortened.

- RESTAURANTS/: all qualifying English source reviews, original IDs/ratings/dates, linked photo receipts and exclusions.
- SOURCE_PHOTOS/: exactly the 5,143 frozen accessible source photographs, unmodified bytes.
- PREPROCESSING/: exact 247 selected reviews, input token IDs, hashes, budget ledger and freeze receipt.
- PREPARED_CAPTION_JOB/: full 5,143-photo job plan and unchanged BLIP settings. No captions are present.
- PROTOCOL/: reference-free six-criterion rubric, 15×3×2 design, future analysis/export rules.
- HUMAN_APP/: adapted Stage N code; real-output workspace and live rating release remain absent.
- QWEN_SHARED_INPUT_MANIFEST.json: same frozen review/photo evidence; captions explicitly pending, Qwen configuration unapproved.
- REPRODUCIBILITY/: new Stage O scripts and the unchanged accepted helpers/configuration sources with hash bindings. Copy these code snapshots to their project-relative paths in the original repository, or inspect the authoritative scripts there. GPU execution needs the original pinned model assets and a later resource decision; this ZIP is not a model-weight bundle and does not auto-install/download anything.
- DATASET_PROVENANCE/: publisher acquisition/checksum and deterministic source/language/access audit; user-profile tables and the original archive are excluded.
- verify_package.py / PACKAGE_MANIFEST.csv / VALIDATION_REPORT.json: actual read-only integrity checks.

Extract the ZIP and open SOURCE_READER.html in a browser. For offline package verification with standard Python: `python verify_package.py`. Do not launch human collection before the researcher/ChatGPT review; no ratings or approvals have been simulated. Existing Yelp A–N evidence is unchanged. Successful external grounding would not conclusively disprove GPT Teacher mimicry.
''')


def source_reader(frozen):
    shared={r['business_id']:r for r in jl(OUT/'PREPROCESSING/shared_review_inputs.jsonl')};parts=[]
    for item in frozen['restaurants']:
        s=json.loads((OUT/item['source_file']).read_text(encoding='utf8'));row=shared[item['business_id']]
        text='<section><h2>'+html.escape(item['name'])+'</h2><p>'+item['business_id']+' · '+str(len(row['reviews']))+' selected reviews · '+str(len(s['photos']))+' source photographs</p>'
        for r in row['reviews']:text+='<details><summary>'+html.escape(r['review_id']+' · '+str(r['stars'])+' stars · '+r['original_date'])+'</summary><pre>'+html.escape(r['text'])+'</pre></details>'
        text+='<details><summary>All individual publisher-linked source photographs</summary><div class="gallery">'
        for p in s['photos']:text+='<figure><a href="'+p['file']+'" target="_blank"><img loading="lazy" src="'+p['file']+'" alt="'+p['photo_id']+'"></a><figcaption>'+p['photo_id']+'</figcaption></figure>'
        text+='</div></details></section>';parts.append(text)
    write(OUT/'SOURCE_READER.html','<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage O frozen source review</title><style>body{max-width:1200px;margin:40px auto;padding:0 20px;font:16px/1.5 system-ui;color:#17212b}section{border-top:2px solid #167a80;margin-top:35px}details{margin:10px 0;padding:8px;border:1px solid #dce3e8}summary{cursor:pointer}pre{white-space:pre-wrap}.gallery{display:flex;flex-wrap:wrap}figure{width:180px;margin:8px}img{max-width:180px;max-height:150px}figcaption{font:12px monospace}</style><h1>Stage O · frozen source evidence</h1><p>15 restaurants; selected original reviews and all frozen publisher-linked photographs. No model outputs, reference summaries or rating form. CDN source renderings are preserved without claiming camera-original resolution.</p>'+''.join(parts)+'</html>')


def package(frozen):
    PKG.mkdir(exist_ok=True)
    source_files=[OUT/'COHORT15_FREEZE.json',OUT/'SOURCE_IMAGE_DIAGNOSTICS.json',OUT/'RESOURCE_AND_MODEL_INTEGRITY.json',OUT/'FROZEN_MODEL_AND_DECODER_BINDINGS.json',OUT/'QWEN_SHARED_INPUT_MANIFEST.json',OUT/'DATASET_FEASIBILITY.md',OUT/'EXECUTION_PLAN_AND_COST.md',OUT/'README.md',OUT/'SOURCE_READER.html']
    for p in source_files:copy(p,p.relative_to(OUT))
    for folder in ['PREPROCESSING','PREPARED_CAPTION_JOB','PROTOCOL','HUMAN_APP']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file():copy(p,p.relative_to(OUT))
    for item in frozen['restaurants']:
        path=OUT/item['source_file'];copy(path,item['source_file']);source=json.loads(path.read_text(encoding='utf8'))
        for p in source['photos']:copy(OUT/p['file'],p['file'])
    for filename in ['SELECTION_PLAN_BEFORE_PHOTO_ELIGIBILITY.json','ENGLISH_LANGUAGE_AUDIT.json','PHOTO_ACCESS_AUDIT.json']:
        copy(OUT/filename,'DATASET_PROVENANCE/'+filename)
    for p in (OUT/'DATASET_DOWNLOADS').glob('*ACQUISITION.json'):copy(p,'DATASET_PROVENANCE/'+p.name)
    script_names=['stage_o_tripadvisor_feasibility_v1.py','prepare_stage_o_tripadvisor_cohort_v1.py','stage_o_frozen_pipeline_v1.py','prepare_stage_o_human_workspace_v1.py',
                  'budget_aware_review_selection_v4_2.py','budget_aware_review_selection_v4_1.py','budget_aware_review_selection_v4.py','budget_aware_review_selection.py','review_selection_pilot.py','token_length_cache.py','english_eligibility.py',
                  'source_budget_audit_template.py','generate_stage_f_validation_captions.py','blip2_caption_pilot.py','caption_basic_quality.py','stage_f_score_validation.py','apply_stage_f_frozen_filters.py','stage_f_math.py','stage_f_d030_protocol.py','stage_f_d031_protocol.py','stage_f_d033_protocol.py','stage_f_active_protocol.py','stage_f_verification_workflow.py','photo_integrity_audit.py','run_stage_k_s1000_phase1_v1.py','s1000_final_eval_core_v1.py']
    bindings=[]
    for name in script_names:
        p=ROOT/'scripts'/name;ast.parse(p.read_text(encoding='utf-8-sig'));copy(p,'REPRODUCIBILITY/scripts/'+name);bindings.append(dict(original='scripts/'+name,sha256=digest(p)))
    for name in ['english_eligibility.yaml','review_selection_source_length_expanded.yaml','led_source_template_d032.json','basic_caption_quality.yaml','stage_f_validation_caption_pool.yaml','stage_f_validation_protocol.yaml','stage_i_led_large_decoder_policy_v2.json']:
        p=ROOT/'configs'/name;copy(p,'REPRODUCIBILITY/configs/'+name);bindings.append(dict(original='configs/'+name,sha256=digest(p)))
    save(PKG/'REPRODUCIBILITY/CODE_AND_CONFIG_BINDINGS.json',bindings)
    for p in (PKG/'HUMAN_APP/scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    # Human-entry builder must refuse absent real outputs, without making a DB or fake content.
    refusal=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/prepare_stage_o_human_workspace_v1.py')],capture_output=True,text=True,encoding='utf8')
    assert refusal.returncode!=0 and 'PENDING: real captions' in refusal.stderr
    assert not (OUT/'HUMAN_APP/WORKSPACE').exists()
    save(PKG/'VALIDATION_REPORT.json',dict(status='PASS_PREPARATION_ONLY_GPU_DEPENDENCIES_BLOCKED',restaurants=15,eligible_English_source_reviews=320,selected_reviews=247,individual_source_photos=5143,
            cohort_and_review_freeze_hashes_verified=True,photos_independently_rehashed_and_decoded=True,association_join_and_URL_ID_checks=True,
            exact_Yelp_raw_or_native_normalized_pixel_overlap=0,cross_selected_restaurant_pixel_overlap=0,
            CPU_review_budget_checks='PASS',original_RO_U_F_adapter_and_backbone_rehash='PASS',
            caption_generation='PENDING_MISSING_WEIGHTS_AND_RAM_PROFILE',filter_scoring='PENDING',LED_inference='PENDING_0_OF_45',
            native_LED_components='PENDING_0_OF_75',human_ratings='NOT_STARTED_0_OF_90',Qwen='NOT_AUTHORIZED_NOT_STARTED',
            application_code='ADAPTED_SYNTAX_CHECKED; FULL_END_TO_END_TEST_PENDING_REAL_OUTPUTS',application_missing_output_refusal='PASS_NO_WORKSPACE_CREATED',
            new_model_calls=0,new_paid_resources=0,paid_cost_actual_USD=0,no_Yelp_result_changes=True,
            limitations=['CDN renderings are not attested camera-original uploads.','Single city frame and language-label prefilter.','ID/URL/hash association checks are not exhaustive semantic image/authorship authentication.','Full caption/inference/app runtime remains pending the resource decision.']))
    # Offline verifier is deliberately standard-library only.
    write(PKG/'verify_package.py','''from pathlib import Path
import csv,hashlib,json
P=Path(__file__).resolve().parent
with (P/'PACKAGE_MANIFEST.csv').open(encoding='utf8',newline='') as f:
    rows=list(csv.DictReader(f))
for r in rows:
    p=(P/r['file']).resolve()
    assert p.is_relative_to(P) and p.is_file()
    assert p.stat().st_size==int(r['bytes'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['file']
freeze=json.loads((P/'COHORT15_FREEZE.json').read_text(encoding='utf8'))
assert len(freeze['restaurants'])==len({r['business_id'] for r in freeze['restaurants']})==15
allphotos=set();reviews=0
for row in freeze['restaurants']:
    s=json.loads((P/row['source_file']).read_text(encoding='utf8'))
    assert len(s['English_reviews'])>=10 and len(s['photos'])>=5
    assert len({p['decoded_RGB_sha256'] for p in s['photos']})>=5
    assert len({r['review_id'] for r in s['English_reviews']})==len(s['English_reviews'])
    reviews+=len(s['English_reviews'])
    for photo in s['photos']:
        assert photo['photo_id'] not in allphotos
        allphotos.add(photo['photo_id'])
        assert hashlib.sha256((P/photo['file']).read_bytes()).hexdigest()==photo['sha256']
assert reviews==320 and len(allphotos)==5143
selected=[json.loads(x) for x in (P/'PREPROCESSING/shared_review_inputs.jsonl').read_text(encoding='utf8').splitlines()]
assert len(selected)==15 and sum(len(x['reviews']) for x in selected)==247
for x in selected:
    assert len(x['reviews'])<=30 and len(x['review_input_ids'])<=5000
    assert hashlib.sha256(x['review_block'].encode()).hexdigest()==x['review_block_sha256']
    for r in x['reviews']:assert hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256']
assert not list((P/'HUMAN_APP').rglob('*.sqlite3')) and not (P/'HUMAN_APP/WORKSPACE/LIVE_RELEASE.json').exists()
print('PASS:',len(rows),'package member hashes; 15 restaurants; 320 source /247 selected reviews; 5,143 individual photos. Outputs and ratings remain pending, not complete.')
''')
    files=[p for p in PKG.rglob('*') if p.is_file() and p.name!='PACKAGE_MANIFEST.csv']
    with (PKG/'PACKAGE_MANIFEST.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','bytes','sha256']);w.writeheader()
        for p in sorted(files):w.writerow(dict(file=p.relative_to(PKG).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    subprocess.run([sys.executable,'-X','utf8','-B',str(PKG/'verify_package.py')],check=True)
    archive=OUT/'STAGE_O_TRIPADVISOR_EVALUATION_REVIEW.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(PKG.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(PKG))
    with tempfile.TemporaryDirectory(prefix='stage_o_package_check_') as tmp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(tmp)
        subprocess.run([sys.executable,'-X','utf8','-B',str(Path(tmp)/'verify_package.py')],check=True)
    save(OUT/'FINAL_PACKAGE_VERIFICATION.json',dict(status='PASS_ACTUAL_ZIP_CRC_EXTRACTION_EVERY_MEMBER_HASH_AND_PARSE',zip=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),
                                                  actual_LED_output_records=0,actual_human_rating_records=0,status_boundary='STOP_BEFORE_DEPENDENCY_RESTORE_OR_PAID_COMPUTE'))
    write(OUT/(archive.name+'.sha256'),digest(archive)+'  '+archive.name)
    print((OUT/'FINAL_PACKAGE_VERIFICATION.json').read_text(),flush=True)


def main():
    frozen=verify_freeze()
    protected=[ROOT/'data/processed/splits/business_split.csv',ROOT/'outputs/stage_m/final_test100_evaluation_review_20261101_v1/PACKAGE/RESULTS_ORIGINAL/COMPLETION.json',
               ROOT/'outputs/stage_n/final_human_statistical_analysis_20261102_v1/PACKAGE/DATA/ORIGINAL_RATINGS_240.json',
               ROOT/'scripts/stage_n_human_evaluation_v1.py',ROOT/'configs/stage_f_validation_protocol.yaml']
    before={p.relative_to(ROOT).as_posix():digest(p) for p in protected}
    diagnostic=source_diagnostics(frozen);resource=resources();reports(frozen,resource,diagnostic);human_app();source_reader(frozen);package(frozen)
    assert before=={p.relative_to(ROOT).as_posix():digest(p) for p in protected}
    verify_freeze()
    save(OUT/'YELP_PRESERVATION_CHECK.json',dict(status='PASS',representative_protected_artifacts=before,all_Stage_O_freeze_bindings_unchanged=True))


if __name__=='__main__':main()
