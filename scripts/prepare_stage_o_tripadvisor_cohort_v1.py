"""Deterministic external-cohort eligibility, source download and freeze. CPU/network only."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,datetime,hashlib,io,json,re,sys,unicodedata,urllib.request
import yaml
from PIL import Image,ImageOps
import english_eligibility
from stage_o_tripadvisor_feasibility_v1 import ROOT,OUT,RAW,load_tables,save,digest,utc

SEED=20261104


def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
def h(x):return hashlib.sha256(canonical(x)).hexdigest()
def norm(x):return ''.join(c for c in unicodedata.normalize('NFKD',x).casefold() if c.isalnum())


def photo_key(url):
    m=re.fullmatch(r'https://media-cdn\.tripadvisor\.com/media/(?:daodao/)?photo-[a-z]+/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{2})/[^?#]+',url)
    if not m:raise ValueError('Unsupported publisher image URL: '+url)
    return 'TA_PHOTO_'+''.join(m.groups())


def fetch_photo(task):
    pid,urls,links=task
    # Keep the publisher-listed variant; prefer photo-l when both supplied variants exist.
    urls=sorted(urls,key=lambda u:('/photo-l/' not in u,u))
    failures=[]
    directory=OUT/'SOURCE_PHOTOS';directory.mkdir(exist_ok=True)
    path=directory/(pid+'.jpg');receipt=directory/(pid+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text(encoding='utf8'))
        if r['status']=='ACCESSIBLE_DECODED':assert digest(path)==r['sha256']
        return r
    for url in urls:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Thesis-Stage-O-ReadOnly-Dataset-Audit/1.0'})
            with urllib.request.urlopen(req,timeout=18) as response:
                data=response.read(20*1024*1024+1);ctype=response.headers.get('Content-Type','');status=response.status
            assert status==200 and ctype.startswith('image/') and 0<len(data)<=20*1024*1024
            with Image.open(io.BytesIO(data)) as original:
                original.load();assert original.format in ['JPEG','PNG','WEBP']
                image=ImageOps.exif_transpose(original).convert('RGB')
                assert image.width>0 and image.height>0
                dimensions=[image.width,image.height]
                pixel_sha=hashlib.sha256(canonical(dimensions)+image.tobytes()).hexdigest()
                fmt=original.format
            path.write_bytes(data)
            result=dict(photo_id=pid,publisher_urls=urls,download_url=url,source_links=links,http_status=status,
                        content_type=ctype,image_format=fmt,width=dimensions[0],height=dimensions[1],bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(),decoded_RGB_sha256=pixel_sha,
                        accessed_at_utc=utc(),file=path.relative_to(OUT).as_posix(),status='ACCESSIBLE_DECODED',
                        provenance='Publisher-linked source photograph; unchanged CDN response bytes. Camera-original resolution/authenticity not attested.',
                        preceding_url_failures=failures)
            save(receipt,result);return result
        except Exception as exc:failures.append(dict(url=url,error=type(exc).__name__+': '+str(exc)))
    result=dict(photo_id=pid,publisher_urls=urls,source_links=links,status='UNAVAILABLE',failures=failures,accessed_at_utc=utc())
    save(receipt,result);return result


def main():
    frozen=OUT/'COHORT15_FREEZE.json'
    if frozen.exists():
        print('Cohort is already frozen; verifying existing freeze, not reselecting.')
        d=json.loads(frozen.read_text(encoding='utf8'))
        for r in d['artifact_bindings']:assert digest(OUT/r['file'])==r['sha256']
        return
    table=load_tables(next(RAW.glob('14622324_*.zip')));reviews=table['reviews'];items=table['items']
    partial=OUT/'PHOTO_ACCESS_AUDIT_PARTIAL.json'
    if partial.exists():
        for r in json.loads(partial.read_text(encoding='utf8'))['photos']:
            target=OUT/'SOURCE_PHOTOS'/(r['photo_id']+'.json')
            if not target.exists():save(target,r)
    assert reviews.reviewId.nunique()==len(reviews) and items.itemId.nunique()==len(items)
    assert set(reviews.rating)=={10,20,30,40,50}
    assert set(reviews.itemId)<=set(items.itemId)
    language=yaml.safe_load((ROOT/'configs/english_eligibility.yaml').read_text(encoding='utf8'))['language_detector']
    english_eligibility._initialize(language)
    cache=OUT/'ENGLISH_LANGUAGE_AUDIT.json'
    if cache.exists():
        detected=json.loads(cache.read_text(encoding='utf8'))['records']
    else:
        detected=[]
        for i,r in enumerate(reviews[reviews.language.eq('en')].itertuples()):
            lang,reason=english_eligibility.classify_text(r.text)
            detected.append(dict(review_id=str(r.reviewId),item_id=str(r.itemId),publisher_language='en',detected_language=lang,reason=reason,text_sha256=hashlib.sha256(r.text.encode()).hexdigest()))
            if i%300==0:print('English source gate',i,flush=True)
        save(cache,dict(policy=language,scope='All publisher-en reviews in the Gijon v2 archive; other publisher languages not reclassified.',records=detected))
    en_ids={int(r['review_id']) for r in detected if r['detected_language']=='en'}
    en=reviews[reviews.reviewId.isin(en_ids)]
    counts=en.groupby('itemId').size();qualified=set(int(x) for x in counts[counts>=10].index)
    itemrows={int(x.itemId):x for x in items.itertuples()}
    photo_associations=defaultdict(list);photo_urls=defaultdict(set)
    for r in reviews.itertuples():
        for url in r.images:
            pid=photo_key(url)
            link=dict(item_id=str(r.itemId),review_id=str(r.reviewId),review_url=r.url,original_photo_url=url)
            assert f'-d{r.itemId}-' in r.url and f'-r{r.reviewId}-' in r.url
            photo_associations[pid].append(link);photo_urls[pid].add(url)
    ambiguous={p for p,links in photo_associations.items() if len({x['item_id'] for x in links})>1}
    pools={iid:sorted(p for p,links in photo_associations.items() if p not in ambiguous and links[0]['item_id']==str(iid)) for iid in qualified}
    qualified={iid for iid in qualified if len(pools[iid])>=5}
    ordered=sorted(qualified,key=lambda iid:hashlib.sha256(f'STAGE_O_V1|{SEED}|14622324|Gijon|{iid}'.encode()).hexdigest())
    plan=dict(format='stage_o_selection_plan_v1',created_at_utc=utc(),dataset_record='14622324',version='2.0.0',city='Gijon',seed=SEED,
              rationale='Smallest v2 city archive, chosen before any model outputs. City-specific exploratory cohort, not representative of all cities.',
              eligibility=dict(minimum_English_reviews=10,English_policy=language,publisher_language_prefilter='en',minimum_accessible_unique_source_photos=5,
                               exclude_photos_with_multiple_item_associations=True,no_identifiable_Yelp_overlap=True),
              ordering='Ascending SHA256 UTF8 STAGE_O_V1|20261104|14622324|Gijon|<decimal itemId>',
              selection='First 15 eligible in fixed candidate order; retain all accessible unique source photo IDs for each, no pre-caption K cap.',
              candidate_item_ids=[str(x) for x in ordered],no_LED_outputs_inspected=True,structurally_qualified_restaurants=len(ordered),
              all_v2_Gijon_restaurants=len(items),all_reviews=len(reviews),publisher_en_reviews=int(reviews.language.eq('en').sum()),detector_en_reviews=len(en),
              ambiguous_photo_ids_excluded_from_eligibility=sorted(ambiguous),other_cities_or_v1_not_needed_if_15_pass=True)
    planpath=OUT/'SELECTION_PLAN_BEFORE_PHOTO_ELIGIBILITY.json'
    if planpath.exists():
        old=json.loads(planpath.read_text(encoding='utf8'));assert old['candidate_item_ids']==plan['candidate_item_ids'];plan=old
    else:save(planpath,plan)
    yelp=[json.loads(line) for line in (ROOT/'data/processed/yelp_2300_v1/businesses.jsonl').read_text(encoding='utf8').splitlines()]
    yelp_name_city={(norm(x['name']),norm(x['city'])) for x in yelp}
    with (ROOT/'data/processed/yelp_2300_v1/photo_manifest.csv').open(encoding='utf8',newline='') as f:yelp_photo_hashes={r['sha256'] for r in csv.DictReader(f)}
    selected=[];audit=[];all_access=[]
    for iid in ordered:
        if len(selected)==15:break
        row=itemrows[iid];bid='TA14622324_'+str(iid)
        assert f'-d{iid}-' in row.url
        name_city_overlap=(norm(row.name),norm(row.city)) in yelp_name_city
        if name_city_overlap:
            audit.append(dict(item_id=str(iid),status='EXCLUDED_NAME_CITY_OVERLAP'));continue
        tasks=[(pid,photo_urls[pid],photo_associations[pid]) for pid in pools[iid]]
        print('Source photographs',len(selected)+1,'candidate',iid,row.name,'pool',len(tasks),flush=True)
        fetched=[]
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures=[executor.submit(fetch_photo,t) for t in tasks]
            for j,future in enumerate(as_completed(futures),1):
                fetched.append(future.result())
                if j%100==0:print('  downloaded/checked',j,'/',len(tasks),flush=True)
        fetched.sort(key=lambda r:r['photo_id']);all_access.extend(fetched)
        valid=[r for r in fetched if r['status']=='ACCESSIBLE_DECODED']
        duplicate_yelp=[r['photo_id'] for r in valid if r['sha256'] in yelp_photo_hashes]
        status='ELIGIBLE' if len(valid)>=5 and not duplicate_yelp else 'INELIGIBLE'
        audit.append(dict(item_id=str(iid),name=row.name,English_reviews=int(counts.loc[iid]),unique_associated_photos=len(tasks),accessible=len(valid),unavailable=len(tasks)-len(valid),yelp_exact_photo_matches=duplicate_yelp,status=status))
        save(OUT/'PHOTO_ACCESS_AUDIT_PARTIAL.json',dict(restaurants=audit,photos=all_access))
        if status!='ELIGIBLE':continue
        english=[]
        for r in en[en.itemId.eq(iid)].itertuples():
            date=r.date.isoformat()
            english.append(dict(review_id='TA_REVIEW_'+str(r.reviewId),business_id=bid,original_review_id=str(r.reviewId),original_item_id=str(iid),
                                text=r.text,title=r.title,stars=int(r.rating)//10,original_rating=int(r.rating),date=date+' 00:00:00',original_date=date,
                                date_precision='day; 00:00:00 formatting placeholder, not an observed review time',language='en',url=r.url,
                                original_photo_urls=r.images,text_sha256=hashlib.sha256(r.text.encode()).hexdigest()))
        english.sort(key=lambda r:(r['date'],r['review_id']))
        source=dict(business_id=bid,original_item_id=str(iid),dataset_record='14622324',dataset_version='2.0.0',city=row.city,country='Spain',
                    business_metadata={'name':row.name,'categories':', '.join(row.type),'city':row.city,'state':''},
                    metadata_conversion='Cuisine types mapped to categories; state unavailable and kept empty. No fabricated address/state.',restaurant_url=row.url,
                    English_reviews=english,photos=valid,unavailable_photos=[r for r in fetched if r['status']!='ACCESSIBLE_DECODED'],
                    no_reference_summary=True,no_predictions=True,selection_order=len(selected)+1,
                    overlap_audit=dict(normalized_name_city_match=False,exact_downloaded_photo_byte_match=False,
                        location_check='Publisher Gijon-area archive, Asturias, Spain; original restaurant URL preserves its exact locality (including neighboring localities). Frozen Yelp has no Gijon city.',
                        limits='Not proof against chains, renamed venues, perceptual image matches or public-pretraining exposure.'))
        p=OUT/'RESTAURANTS'/f'{bid}.json';save(p,source)
        selected.append(dict(order=len(selected)+1,business_id=bid,item_id=str(iid),name=row.name,city=row.city,English_reviews=len(english),accessible_photos=len(valid),source_file=p.relative_to(OUT).as_posix(),source_sha256=digest(p)))
    save(OUT/'PHOTO_ACCESS_AUDIT.json',dict(restaurants=audit,photos=all_access))
    if len(selected)!=15:
        save(OUT/'ELIGIBILITY_STOP.json',dict(status='STOP_INSUFFICIENT_QUALIFYING_RESTAURANTS',selected=len(selected),actual_audit=audit));raise RuntimeError('Fewer than 15 eligible restaurants; no relaxed criteria.')
    selected_hashes=defaultdict(list)
    for x in selected:
        source=json.loads((OUT/x['source_file']).read_text(encoding='utf8'))
        for p in source['photos']:selected_hashes[p['sha256']].append((x['business_id'],p['photo_id']))
    cross=[v for v in selected_hashes.values() if len({x[0] for x in v})>1]
    if cross:
        save(OUT/'ELIGIBILITY_STOP.json',dict(status='STOP_CROSS_RESTAURANT_IDENTICAL_PHOTO_BYTES',matches=cross));raise RuntimeError('Cross-restaurant image identity needs review; not silently exclude/reselect.')
    bindings=[]
    for p in [planpath,cache,OUT/'PHOTO_ACCESS_AUDIT.json']+[OUT/x['source_file'] for x in selected]:bindings.append(dict(file=p.relative_to(OUT).as_posix(),sha256=digest(p)))
    freeze=dict(status='FROZEN_15_ELIGIBLE_EXTERNAL_RESTAURANTS_BEFORE_MODEL_OUTPUTS',frozen_at_utc=utc(),restaurants=selected,
                source_review_count=sum(x['English_reviews'] for x in selected),source_photo_count=sum(x['accessible_photos'] for x in selected),
                artifact_bindings=bindings,selection_plan_sha256=digest(planpath),cohort_hash=h(selected),no_model_outputs_exist=True,
                original_archive_sha256=digest(next(RAW.glob('14622324_*.zip'))),
                Yelp_businesses_checked=len(yelp),Yelp_photo_raw_hashes_checked=len(yelp_photo_hashes),cross_restaurant_identical_photo_bytes=0,
                photo_resolution_limit='Only publisher-listed photo-l/photo-f CDN renderings are available. Preserved response bytes are not claimed to be camera-original files.')
    save(frozen,freeze)
    print(json.dumps({k:v for k,v in freeze.items() if k not in ['restaurants','artifact_bindings']},indent=2),flush=True)


if __name__=='__main__':main()
