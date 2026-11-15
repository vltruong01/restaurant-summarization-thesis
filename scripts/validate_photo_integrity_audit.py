"""Fresh file/decode checks plus independent duplicate graphs and DCT reference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import hashlib,json,math,time,warnings
import numpy as np,pandas as pd,yaml
from PIL import Image,ImageOps,ImageFile
from photo_integrity_audit import ROOT,CONFIG,digest_file,phash
ImageFile.LOAD_TRUNCATED_IMAGES=False


def worker(args):
    row,cohort,raw_photos,reference=args;path=Path(cohort)/row['relative_path'];pid=row['photo_id']
    assert digest_file(path)==row['expected_sha256']==row['raw_sha256']
    assert digest_file(Path(raw_photos)/(pid+'.jpg'))==row['raw_sha256'],pid
    success=False;pixel='';perceptual='';normal_size=(0,0);original_size=(0,0);mode='';fmt='';orientation=1;error='';reference_checked=False
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            with Image.open(path) as image:
                original_size=image.size;mode=image.mode;fmt=image.format;image.verify()
            with Image.open(path) as image:
                image.load();orientation=int(image.getexif().get(274,1));rgb=ImageOps.exif_transpose(image).convert('RGB');normal_size=rgb.size
                h=hashlib.sha256();h.update(b'EXIF_RGB_dimensions_v1:');h.update(rgb.width.to_bytes(8,'big'));h.update(rgb.height.to_bytes(8,'big'));h.update(rgb.tobytes());pixel=h.hexdigest()
                perceptual,std,bits=phash(rgb)
                if reference:
                    grey=np.array(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
                    cosine=np.array([[math.cos(math.pi*k*(n+.5)/32) for n in range(32)] for k in range(8)])
                    coefficients=np.empty((8,8),dtype=np.float64)
                    for u in range(8):
                        for v in range(8):coefficients[u,v]=np.sum(grey*cosine[u,:,None]*cosine[v,None,:],dtype=np.float64)
                    coefficients=np.round(coefficients,6);cut=np.median(coefficients);number=0
                    for value in coefficients.flat:number=number*2+int(value>cut)
                    assert f'{number:016x}'==perceptual,pid;reference_checked=True
                success=True
        except (OSError,ValueError,SyntaxError,Image.DecompressionBombError) as exc:error=type(exc).__name__
    assert success==row['readable'],pid
    assert original_size==(row['width'],row['height']) and mode==row['original_mode'] and fmt==row['image_format'],pid
    if success:
        assert normal_size==(row['normalized_width'],row['normalized_height']) and orientation==row['exif_orientation'],pid
        assert pixel==row['pixel_sha256'] and perceptual==row['phash64_hex'],pid
        assert math.isclose(std,row['grayscale32_std'],abs_tol=1e-12) and bits==row['phash_set_bits'],pid
    else:assert error==row['error_type'],pid
    return reference_checked


def main():
    started=time.perf_counter();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    completion=json.loads((out/'completion_report.json').read_text());assert completion['status']=='PASS'
    for name,digest in completion['artifact_sha256'].items():assert digest_file(out/name)==digest,name
    identity=json.loads((out/'scientific_identity.json').read_text());assert digest_file(CONFIG)==identity['config_sha256']
    for name,digest in identity['scripts_sha256'].items():assert digest_file(ROOT/'scripts'/name)==digest,name
    for name,digest in identity['protected_sha256'].items():assert digest_file(ROOT/name)==digest,name
    manifest=pd.read_csv(out/'image_manifest.csv',keep_default_na=False);source=pd.read_csv(ROOT/cfg['photo_manifest'],keep_default_na=False)
    assert len(manifest)==len(source)==30027 and manifest.photo_id.is_unique
    for column in ['photo_id','business_id','split','label','relative_path','bytes']:assert source[column].tolist()==manifest[column].tolist(),column
    assert source.sha256.tolist()==manifest.expected_sha256.tolist()
    ref=set(json.loads((out/'precommitted_reference_photo_ids.json').read_text()));assert len(ref)==256 and not set(manifest[manifest.split.eq('test')].photo_id)&ref
    project=yaml.safe_load((ROOT/'configs/project.yaml').read_text());raw=ROOT/project['paths']['data_root']/project['paths']['photo_directory'];count=0;refs=0
    with ProcessPoolExecutor(max_workers=8) as pool:
        args=((row,str(ROOT/cfg['cohort_directory']),str(raw),row['photo_id'] in ref) for row in manifest.to_dict('records'))
        for checked in pool.map(worker,args,chunksize=32):
            count+=1;refs+=int(checked)
            if count%5000==0:print(f'Fresh photo/raw-source validation {count}/30027.',flush=True)
    assert refs==256
    valid=manifest[manifest.readable];lookup=valid.set_index('photo_index').to_dict('index');numbers={i:int(row['phash64_hex'],16) for i,row in lookup.items()}
    pairs=pd.read_csv(out/'near_pair_candidates.csv');actual_pairs={(int(r.photo_index_a),int(r.photo_index_b)):int(r.hamming_distance) for r in pairs.itertuples()};expected_pairs={};pair_checks=0
    groups=pd.read_csv(out/'within_business_duplicate_groups.csv',keep_default_na=False);members=pd.read_csv(out/'within_business_group_memberships.csv',keep_default_na=False);business=pd.read_csv(out/'business_photo_counts.csv',keep_default_na=False).set_index('business_id');verified_groups=0
    expected_group_ids=set()
    def check_group(bid,kind,threshold,nodes,edge_count):
        nonlocal verified_groups
        photos=sorted(lookup[i]['photo_id'] for i in nodes);gid=kind+'_'+hashlib.sha256((str(threshold)+'|'+bid+'|'+'|'.join(photos)).encode()).hexdigest()[:20];expected_group_ids.add(gid)
        rows=groups[groups.group_id.eq(gid)];assert len(rows)==1;row=rows.iloc[0]
        assert set(members.loc[members.group_id.eq(gid),'photo_index'])==set(nodes)
        diameter=max((numbers[a]^numbers[b]).bit_count() for j,a in enumerate(nodes) for b in nodes[j+1:])
        rep=min(nodes,key=lambda i:lookup[i]['photo_id'])
        assert row.photos==len(nodes) and row.representative_index==rep and row.representative_photo_id==lookup[rep]['photo_id']
        assert row.edges==edge_count and row.phash_diameter==diameter and row.chained_beyond_threshold==(kind=='near' and diameter>threshold)
        verified_groups+=1
    for bid,original in manifest.groupby('business_id'):
        part=original[original.readable].sort_values('photo_id');indices=part.photo_index.astype(int).tolist();expected={'source_photos':len(original),'readable_photos':len(part),'unreadable_photos':int((~original.readable).sum()),'unfiltered_readable_pool_count':len(part),'raw_exact_unique':part.raw_sha256.nunique(),'pixel_exact_unique':part.pixel_sha256.nunique(),'below5_readable':len(part)<5,'below5_raw_unique':part.raw_sha256.nunique()<5,'below5_pixel_unique':part.pixel_sha256.nunique()<5}
        for kind,column in [('raw','raw_sha256'),('pixel','pixel_sha256')]:
            for _,g in part.groupby(column):
                if len(g)>1:check_group(bid,kind,-1,g.photo_index.astype(int).tolist(),len(g)*(len(g)-1)//2)
        local=[]
        for j,a in enumerate(indices):
            for b in indices[j+1:]:
                distance=(numbers[a]^numbers[b]).bit_count();pair_checks+=1
                if distance<=12:expected_pairs[a,b]=distance;local.append((a,b,distance))
        for threshold in [0,4,8,12]:
            adjacency={i:set() for i in indices};edges=[(a,b) for a,b,d in local if d<=threshold]
            for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
            remaining=set(indices);components=[]
            while remaining:
                stack=[min(remaining)];nodes=set()
                while stack:
                    node=stack.pop()
                    if node in nodes:continue
                    nodes.add(node);stack.extend(adjacency[node]-nodes)
                remaining-=nodes;components.append(sorted(nodes))
            expected.update({f'near_t{threshold}_components':len(components),f'near_t{threshold}_duplicate_groups':sum(len(c)>1 for c in components),f'near_t{threshold}_edges':len(edges),f'below5_near_t{threshold}':len(components)<5})
            for nodes in components:
                if len(nodes)>1:
                    scope=set(nodes);check_group(bid,'near',threshold,nodes,sum(a in scope and b in scope for a,b in edges))
        for column,value in expected.items():assert business.loc[bid,column]==value,(bid,column)
    assert actual_pairs==expected_pairs and expected_group_ids==set(groups.group_id) and verified_groups==len(groups)
    globals=pd.read_csv(out/'global_duplicate_collision_groups.csv',keep_default_na=False);gmembers=pd.read_csv(out/'global_group_memberships.csv',keep_default_na=False);global_checks=0;expected_globals=set()
    for kind,column in [('raw','raw_sha256'),('pixel','pixel_sha256'),('phash_collision','phash64_hex')]:
        for value,g in valid.groupby(column):
            if len(g)<2:continue
            gid='global_'+kind+'_'+hashlib.sha256(str(value).encode()).hexdigest()[:20];expected_globals.add(gid);row=globals[globals.group_id.eq(gid)].iloc[0]
            assert row.photos==len(g) and row.businesses==g.business_id.nunique() and row.cross_business==(g.business_id.nunique()>1) and row.cross_split==(g.split.nunique()>1)
            assert row.splits=='|'.join(sorted(g.split.unique())) and set(gmembers.loc[gmembers.group_id.eq(gid),'photo_index'])==set(g.photo_index)
            assert row.confirmed_byte_identity==(kind=='raw') and row.confirmed_normalized_pixel_identity==(kind=='pixel') and row.perceptual_collision_only==(kind=='phash_collision');global_checks+=1
    assert expected_globals==set(globals.group_id)
    pool=pd.read_csv(out/'unfiltered_readable_photo_pool.csv',keep_default_na=False);assert pool.photo_id.tolist()==valid.photo_id.tolist()
    for name,digest in identity['protected_sha256'].items():assert digest_file(ROOT/name)==digest,name
    report={'validation_passed':True,'cohort_and_original_raw_photo_SHA_checks':30027,'full_fresh_decode_pixel_phash_replays':30027,'independent_direct_DCT_reference_photos':refs,'within_business_readable_pairs_verified':pair_checks,'candidate_pairs_verified':len(pairs),'within_groups_verified':verified_groups,'global_exact_collision_groups_verified':global_checks,'business_counts_verified':2300,'unfiltered_pool_IDs_preserved':True,'source_and_prior_scientific_evidence_unchanged':True,'test_thresholds_tuned':False,'photos_deleted':False,'businesses_resampled':False,'caption_generation_started':False,'final_duplicate_policy':None,'validator_sha256':digest_file(__file__),'elapsed_seconds':round(time.perf_counter()-started,3)}
    (out/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
