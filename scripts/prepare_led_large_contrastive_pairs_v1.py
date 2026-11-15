"""Retrieve eight immutable TRAIN positives and propose <=24 source-linked edits.

No models, Teacher calls, researcher labels, scientific-file writes or GPU use.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import difflib
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_i/led_large_contrastive_pair_qualification_v1'
sys.path.insert(0,str(ROOT/'scripts'))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def save(p,v):
    p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    import numpy as np
    from transformers import AutoTokenizer
    from led_large_decoder_label_view_v1 import remove_initial_bos
    assert not OUT.exists(),'Versioned output exists; preserve it'
    OUT.mkdir(parents=True)
    pilot=read(ROOT/'outputs/stage_i/led_large_contrastive_feasibility_v1/portable_review/TRAIN_ONLY_PILOT_MANIFEST.json')
    fit=pilot['fit'];src={x['business_id']:x for x in read(ROOT/'outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json') if x['business_id'] in fit}
    assert len(src)==8
    manifest=read(ROOT/'outputs/stage_h/led_dataset_v1/manifest.json')
    info=manifest['datasets']['train_filtered'];rawpath=ROOT/info['stage_g_jsonl']
    assert sha(rawpath)==info['stage_g_jsonl_sha256'] and sha(ROOT/info['path'])==info['sha256']
    rows={};raw_hash={};raw_line_hash={}
    with rawpath.open('rb') as f:
        for line in f:
            row=json.loads(line)
            if row['business_id'] in fit:
                rows[row['business_id']]=row;raw_line_hash[row['business_id']]=hashlib.sha256(line).hexdigest()
    assert len(rows)==8
    views=read(ROOT/'outputs/stage_i/led_large_decoder_policy_v1/LABEL_VIEWS.json')['views']['train_filtered']
    assert sha(ROOT/views['path'])==views['sha256']
    with np.load(ROOT/info['path'],allow_pickle=False) as p: original={k:p[k].copy() for k in p.files}
    with np.load(ROOT/views['path'],allow_pickle=False) as p: corrected={k:p[k].copy() for k in p.files}
    ids=original['business_ids'].tolist();assert ids==corrected['business_ids'].tolist()
    tm=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    tokenizer_dir=ROOT/tm['relative_directory']
    for n,h in tm['sha256'].items():assert sha(tokenizer_dir/n)==h
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    packets={};summary=[]
    for bid in fit:
        row=rows[bid];source=src[bid];i=ids.index(bid)
        assert row['source']==source['source'] and digest(row['source'])==row['source_sha256']==source['source_sha256']
        a,b=original['label_offsets'][i:i+2];labels=original['labels'][a:b].astype(np.int64).tolist()
        a,b=corrected['label_offsets'][i:i+2];view=corrected['labels'][a:b].astype(np.int64).tolist()
        assert tok.encode(row['target'],add_special_tokens=True)==labels
        assert remove_initial_bos(labels)==view
        a,b=original['input_offsets'][i:i+2];input_ids=original['input_ids'][a:b].astype(np.int64).tolist()
        assert input_ids==source['input_ids']
        assert tok.encode(source['source'],add_special_tokens=True)==input_ids
        pos=row['target'];review,visual=pos.split('\n\n[VISUAL_EVIDENCE]\n')
        review=review.removeprefix('[REVIEW_CONSENSUS]\n')
        assert ' '.join(x['text'] for x in row['shared_review_evidence']['claims'])==review
        assert ' '.join(x['text'] for x in row['visual_claims_with_evidence'])==visual
        audit=[]
        for k,c in enumerate(row['shared_review_evidence']['claims'],1):
            evidence=[]
            for q in c['support']:
                text=source['reviews'][q['review_marker']];quote=q['quote']
                assert quote in text,(bid,q['review_marker'],quote)
                start=text.index(quote)
                evidence.append(dict(marker=q['review_marker'],quote=quote,start_in_marker=start,end_in_marker=start+len(quote)))
            audit.append(dict(claim_id=f'R{k:02d}',text=c['text'],domain='review',evidence=evidence,
                source_assessment='MODEL_REVIEWED_SOURCE_SUPPORTED_PENDING_RESEARCHER',
                researcher_decision='PENDING',researcher_confirmed=False))
        for k,c in enumerate(row['visual_claims_with_evidence'],1):
            evidence=[]
            for q in c['support']:
                cap=source['captions'][q['caption_id']];quote=q['caption_quote'];assert quote in cap
                start=cap.index(quote)
                evidence.append(dict(marker=q['caption_id'],quote=quote,caption=cap,start_in_marker=start,end_in_marker=start+len(quote)))
            audit.append(dict(claim_id=f'V{k:02d}',text=c['text'],domain='visual',evidence=evidence,
                source_assessment='MODEL_REVIEWED_SINGLE_CAPTION_SUPPORT_PENDING_RESEARCHER',
                researcher_decision='PENDING',researcher_confirmed=False))
        if not source['captions']:assert not visual.strip()
        packet=dict(business_id=bid,name=source['source'].splitlines()[1].removeprefix('name='),split='train',condition='filtered',
            source=source['source'],source_sha256=source['source_sha256'],reviews=source['reviews'],captions=source['captions'],input_ids=input_ids,
            positive_text=pos,positive_utf8_sha256=digest(pos),original_labels=labels,corrected_labels=view,
            original_label_ids_sha256=digest(json.dumps(labels,separators=(',',':'))),
            corrected_label_ids_sha256=digest(json.dumps(view,separators=(',',':'))),
            original_row_path=info['stage_g_jsonl'],original_row_bytes_sha256=raw_line_hash[bid],
            claim_audit=audit,complete_text_covered=True,zero_caption_visual_compliant=not visual.strip() if not source['captions'] else None,
            positive_admissibility='PROVISIONAL_SOURCE_SUPPORTED_NOT_RESEARCHER_ACCEPTED',
            researcher_decision='PENDING',researcher_confirmed=False,
            warnings=(["Original replacement character U+FFFD retained; do not repair target."] if '\ufffd' in pos else [])+
                (["Visual statements include redundant paraphrases; factual support is not a quality endorsement."] if bid=='wjZwXKDiFBQXlO8-TjQTXA' else []),
            forbidden_evidence_used=False,historical_photo_verdicts_used=False)
        packets[bid]=packet;save(Path('positives')/(bid+'.json'),packet)
        (OUT/'positives'/(bid+'_ORIGINAL_TARGET.txt')).write_bytes(pos.encode('utf-8'))
        summary.append({k:packet[k] for k in ['business_id','name','positive_utf8_sha256','corrected_label_ids_sha256','positive_admissibility','researcher_decision','warnings']})

    specs=[]
    def add(bid,category,old,new,evidence,meaning,recommend='QUALIFY_PENDING_RESEARCHER'):
        pos=packets[bid]['positive_text'];assert pos.count(old)==1,(bid,old)
        negative=pos.replace(old,new,1)
        assert negative!=pos
        neg=tok.encode(negative,add_special_tokens=True);assert neg[0]==0 and neg[-1]==2 and len(neg)<=768
        labels=remove_initial_bos(neg)
        links=[]
        for domain,marker,quote in evidence:
            text=packets[bid]['reviews' if domain=='review' else 'captions'][marker]
            assert quote in text,(bid,marker,quote)
            start=text.index(quote);links.append(dict(domain=domain,marker=marker,quote=quote,start_in_marker=start,end_in_marker=start+len(quote)))
        pair_id=f'P{len(specs)+1:02d}'
        prefix=0
        while prefix<min(len(pos),len(negative)) and pos[prefix]==negative[prefix]:prefix+=1
        suffix=0
        while suffix<min(len(pos),len(negative))-prefix and pos[-suffix-1]==negative[-suffix-1]:suffix+=1
        old_end=len(pos)-suffix if suffix else len(pos);new_end=len(negative)-suffix if suffix else len(negative)
        pair=dict(pair_id=pair_id,business_id=bid,category=category,positive_text=pos,positive_utf8_sha256=digest(pos),
            positive_labels=packets[bid]['corrected_labels'],negative_text=negative,negative_utf8_sha256=digest(negative),negative_labels=labels,
            negative_label_ids_sha256=digest(json.dumps(labels,separators=(',',':'))),
            source_sha256=packets[bid]['source_sha256'],model_assisted_recommendation=recommend,
            source_reason=meaning,evidence=links,
            minimal_edit=dict(start=prefix,positive_end=old_end,negative_end=new_end,removed=pos[prefix:old_end],inserted=negative[prefix:new_end]),
            diff=''.join(difflib.unified_diff(pos.splitlines(True),negative.splitlines(True),fromfile='immutable_positive',tofile='proposed_negative')),
            negative_property='incomplete' if category=='core_omission' else ('unsupported_visual_assertion' if category=='grounding_error' else 'misleading_false_consensus'),
            diagnostics=dict(positive_tokens=len(packets[bid]['corrected_labels']),negative_tokens=len(labels),
                token_length_delta=len(labels)-len(packets[bid]['corrected_labels']),review_nonempty=True,
                fabricated_unanimity_lexemes=[x for x in ('all','unanimously','consistently','without','uniformly') if x in new.lower().split()],
                shortcut_risk='Audit length and consensus-word shortcuts; no automatic negative label from shortness.'),
            researcher_decision='PENDING',researcher_confirmed=False,accepted_for_training=False)
        if recommend.startswith('REJECT'):pair['negative_property']='not_demonstrably_worse_or_shorter_but_useful'
        specs.append(pair);save(Path('pairs')/(pair_id+'.json'),pair)

    mesa='ZrxlfdFU_qpnIHI2DmoRxA';tempo='EzJI5x25U5GgbYBWhu5ZQQ';habit='5uhG34VRsD0A6lOY8F41Dg'
    little='fJb98kTGBuTzxP8x9p6VhQ';tuj='wjZwXKDiFBQXlO8-TjQTXA';di='TOgkopnDE5q9mT074TDeeA'
    pj='guxzC4GOY1MYY5KqfItwOQ';opal='CoZ2mpsMBP8HUG1ymKoZTg'
    add(mesa,'core_omission','Generous amounts of pizza toppings are a recurring highlight. ','',
        [('review','REV_10','You ask for toppings, and you get TOPPINGS!'),('review','REV_14',"All of the pizzas had more toppings than you'd get at most other pizza places.")],
        'Removes a repeated, distinctive pizza feature, not just decorative wording; independent salience adjudication required.')
    add(tempo,'core_omission','Coffee is repeatedly described as very good, with reviewers praising both regular coffee and espresso-based drinks. ','',
        [('review','REV_1','Great attentive staff with terrific ice coffee and good WiFi.'),('review','REV_7','The coffee here was actually really great!')],
        'Removes the coffee-specific Core of a coffee-and-taco business while leaving tacos/staff/atmosphere.')
    add(habit,'core_omission','Several reviewers praise the tempura green beans. ','',
        [('review','REV_19','Those tempura green beans!'),('review','REV_20','Also I got the tempura green beans which were good like always.'),('review','REV_25','And the tempura green beans are great.')],
        'Removes a recurring distinctive side dish; remaining generic burger/service facts cannot replace it.')
    old='Reviewers especially enjoy the croissants, describing them as fresh, light or flaky, with praise for almond and savory breakfast versions. '
    add(little,'core_omission',old,'',
        [('review','REV_7','genuine big, light, flaky, and tasty croissants.'),('review','REV_22','I had the almond and a breakfast croissant and they were super fresh, soft and crackly.')],
        'Removes a recurrent bakery staple and its specific variants; this is an adequacy preference, not a factual falsehood.')
    old="Service opinions are mixed: several reviews describe helpful, kind staff and a polite owner, while others report an obnoxious cashier or unfriendly phone interactions, including repeated hang-ups. One review describes a disputed cheeseless order and difficulty obtaining an eventual refund."
    add(mesa,'same_aspect_disagreement_loss',old,'Reviews agree that service is friendly and helpful, including polite phone interactions.',
        [('review','REV_23','The staff was so attentive, kind and professional.'),('review','REV_15','Asked if there were any deals and got hung up on.'),('review','REV_24','I think they could be a little friendlier on the phone when you call them.')],
        'Falsely turns conflicting service experiences into agreement; reverses the reported phone-service weakness.')
    old='Breakfast tacos, homemade tortillas, and salsa are frequently praised, though some reviewers found the tacos average or bland.'
    add(tempo,'same_aspect_disagreement_loss',old,'Reviewers find the breakfast tacos consistently flavorful rather than average or bland, and praise homemade tortillas and salsa.',
        [('review','REV_2','The tacos were really good.'),('review','REV_3',"It was very tasteless and salsa didn't help much."),('review','REV_4','The tacos were average.')],
        'Negates explicit same-aspect negative taco opinions rather than merely deleting a peripheral detail.')
    old='Reviews are mixed on burger quality: some describe the burgers as freshly made, good, juicy, and perfectly cooked, while others report dry, soggy, underseasoned, or poorly prepared burgers.'
    add(habit,'same_aspect_disagreement_loss',old,'Reviewers describe the burgers as consistently juicy and perfectly cooked, without reports of dry or soggy burgers.',
        [('review','REV_19','My burger was perfect!'),('review','REV_10','My burger was soggy, the bun looked toasted on one side?'),('review','REV_5','kinda dry burger')],
        'Creates false consensus on the main food aspect; negative reports remain visible in the source.')
    old='Selected reviews are divided on the food: some praise individual dishes such as brisket, Chicken Bonne Femme, entrees, desserts, and salads, while others describe meals as bland, uninspired, poorly prepared, or disappointing.'
    add(tuj,'same_aspect_disagreement_loss',old,'Selected reviews unanimously praise the food, including brisket, Chicken Bonne Femme, entrees, desserts, and salads.',
        [('review','REV_1','their brisket.'),('review','REV_19','the food was uninspired.'),('review','REV_27','The service was good, but the food was lacking.')],
        'Replaces explicit food disagreement with unanimity, contradicting retained model-visible food reports.')
    for bid,text in [(mesa,'A photo shows a pizza.'),(tempo,'The photo shows a frozen taco.')]:
        add(bid,'grounding_error','[VISUAL_EVIDENCE]\n','[VISUAL_EVIDENCE]\n'+text,[],
            'Filtered caption map is empty. A visual assertion is unsupported even if the food occurs in reviews; not a claim about real-world photographic truth.')
    add(habit,'grounding_error','The tray is on a table.','The tray is in front of the restaurant sign.',
        [('caption','IMG_CAP_1','a tray of chicken tenders, fries, and a sandwich on a table.'),('caption','IMG_CAP_2','a sign for a burger and sandwich restaurant in front of a building.')],
        'Invents a tray/sign relationship across two distinct captioned photos; source does not establish a joint scene.')
    add(di,'grounding_error','The coffee cup sits on a saucer on a table.','The coffee cup sits beside the cake on a white plate.',
        [('caption','IMG_CAP_2','a piece of cake sitting on top of a white plate.'),('caption','IMG_CAP_5','a cup of coffee sitting on a saucer on a table.')],
        'Fuses cake/plate and coffee/saucer from separate captions into an unsupported relationship.')
    old='Wings receive praise in several reviews, but others describe overcooked or gummy wings. Individual reviews also criticize excessively salty, oily wings and ghost pepper sauce with little heat. '
    add(pj,'core_omission',old,'',
        [('review','REV_1','wings are my fav'),('review','REV_3','The Mexican garlic wings had a nice flavor but were definitely overcooked.'),('review','REV_16','Wings were gummy, soup luke-warm, and fries cold.')],
        'Deletes the full recurring wings theme, including both praise and complaints; assess business-specific adequacy rather than shortness.')
    add(opal,'core_omission','The seafood paella receives strong praise from multiple selected reviewers, including for its flavor and plentiful seafood. ','',
        [('review','REV_6','seafood paella was good too.'),('review','REV_7','The seafood paella was ridiculously delicious.'),('review','REV_7','The shrimp, mussels, and scallops were fresh as well as plentiful.')],
        'Deletes the characteristic named dish and its repeated support, leaving mostly generic food/service/value descriptions.')
    old='Several reviews describe friendly, helpful and polite staff. Isolated service problems include being charged for an unordered baguette and being passed over for a later-arriving customer, prompting that reviewer to leave.'
    add(little,'same_aspect_disagreement_loss',old,'Reviews consistently describe friendly, helpful and polite staff, without service problems.',
        [('review','REV_14','friendly staff'),('review','REV_23','While we were waiting another customer entered and she decided to wait on them instead of us, so we were forced to leave.'),('review','REV_16','we were charged for a ham and Brie baguette that we did not order')],
        'Denies concrete service failures rather than merely prioritizing the majority opinion.')
    old='Experiences differ: one reports disruptive group noise, another finds the furniture impractical for work and the interior too warm, and another reports persistent flies and insects despite liking it as a workplace.'
    add(di,'same_aspect_disagreement_loss',old,'Reviewers describe the setting as consistently quiet and comfortable for work, without noise, heat or insect problems.',
        [('review','REV_18','But lately it is WAY too loud.'),('review','REV_19',"It's also way too warm temperature-wise."),('review','REV_23','BUT this place has a persistent FLY AND INSECT problem.')],
        'Erases three explicit work-setting drawbacks and replaces them with an unsupported denial; the original positive praise remains.')
    # Deliberately rejected shorter edits: a shorter, useful output is not a negative automatically.
    add(tuj,'core_omission','A steak is sitting on a plate with potatoes and gravy. ','',
        [('caption','IMG_CAP_4','a steak is sitting on a plate with potatoes and gravy.')],
        'Rejected: prior atomic steak/plate/potato/gravy clauses retain this information. Removing a redundant restatement can improve concision.', 'REJECT_NOT_MATERIALLY_WORSE')
    add(little,'core_omission','The image shows a close-up of a drink. ','',
        [('caption','IMG_CAP_3','a close up of a chocolate milkshake being poured into a cup.')],
        'Rejected: the retained pouring-into-cup sentence preserves the substantive visual event; dropping framing is not enough to call this worse.', 'REJECT_NOT_MATERIALLY_WORSE')
    assert len(specs)==18 and len(specs)<=24
    proposed=[x for x in specs if x['model_assisted_recommendation']=='QUALIFY_PENDING_RESEARCHER']
    assert len(proposed)==16
    from collections import Counter
    categories=Counter(x['category'] for x in proposed);counts=Counter(x['business_id'] for x in proposed)
    assert set(counts)==set(fit) and all(1<=n<=3 for n in counts.values())
    assert all(categories[k]>=4 for k in ('core_omission','same_aspect_disagreement_loss','grounding_error'))
    save('POSITIVE_AUDIT_SUMMARY.json',dict(cases=summary,source_supported_provisionally=8,researcher_accepted=0,
        full_positive_support_not_independently_certified=True,assessment_author_kind='model_assisted',
        archived_Teacher_evidence_used_for_retrieval_only=True,original_photos_accessed=False,validation_or_test_accessed=False))
    save('PAIR_QUALIFICATION_STATUS.json',dict(proposed_edits=18,provisionally_qualified_candidates=16,model_recommended_rejections=2,
        accepted_pairs=0,researcher_decisions='ALL_PENDING',category_counts=dict(categories),per_business_counts=dict(counts),
        continuation_allowed=False,reason='No independent researcher acceptance; no production numerical verification or GPU approval.',
        source_qualification_is_model_assisted=True,pairs_to_adjudicate=[x['pair_id'] for x in proposed],
        all_proposal_ids=[x['pair_id'] for x in specs],old_targets_modified=0,GPU_operations=0))
    save('FROZEN_BINDINGS.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        train_jsonl=dict(path=info['stage_g_jsonl'],sha256=info['stage_g_jsonl_sha256']),
        train_npz=dict(path=info['path'],sha256=info['sha256']),corrected_train_labels=views,
        tokenizer_manifest=tm,pilot_manifest_sha256=sha(ROOT/'outputs/stage_i/led_large_contrastive_feasibility_v1/portable_review/TRAIN_ONLY_PILOT_MANIFEST.json'),
        original_source_pack_sha256=sha(ROOT/'outputs/stage_i/led_large_nbest_reranking_v1/TRAIN_SOURCE_ONLY_CASES.json')))
    print(json.dumps(dict(positives=8,proposed_edits=18,provisionally_qualified=16,accepted=0,categories=dict(categories),GPU_operations=0)))

if __name__=='__main__':main()
