"""Read-only photo decoding, declared pHash and duplicate graph diagnostics."""
from pathlib import Path
import hashlib,json,math,warnings
import numpy as np,pandas as pd
from PIL import Image,ImageOps,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=False
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/photo_integrity_duplicate_audit.yaml'
BASIS=np.cos(np.pi*np.arange(8)[:,None]*(np.arange(32)[None,:]+.5)/32)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def pixel_digest(rgb):
    h=hashlib.sha256(b'EXIF_RGB_dimensions_v1:')
    h.update(rgb.width.to_bytes(8,'big'));h.update(rgb.height.to_bytes(8,'big'));h.update(rgb.tobytes())
    return h.hexdigest()


def phash(rgb):
    grey=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    coefficients=np.round(BASIS@grey@BASIS.T,6)
    bits=(coefficients>np.median(coefficients)).flatten()
    value=0
    for bit in bits:value=(value<<1)|int(bit)
    return f'{value:016x}',float(grey.std()),int(bits.sum())


def inspect_photo(args):
    row,cohort=args;row=dict(row);path=(Path(cohort)/row['relative_path']).resolve()
    assert path.is_relative_to(Path(cohort).resolve()) and path.is_file(),str(path)
    assert path.stat().st_size==int(row['bytes']),row['photo_id']
    actual=digest_file(path);assert actual==row['sha256'],row['photo_id']
    result={**row,'expected_sha256':row['sha256'],'raw_sha256':actual,'readable':False,'status':'unreadable','error_type':'','error':'','warnings':'','image_format':'','original_mode':'','width':0,'height':0,'exif_orientation':1,'normalized_width':0,'normalized_height':0,'pixel_sha256':'','phash64_hex':'','grayscale32_std':0.0,'phash_set_bits':0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            with Image.open(path) as image:
                result.update(image_format=image.format or '',original_mode=image.mode,width=image.width,height=image.height)
                image.verify()
            with Image.open(path) as image:
                image.load();orientation=image.getexif().get(274,1)
                result['exif_orientation']=int(orientation) if orientation is not None else 1
                rgb=ImageOps.exif_transpose(image).convert('RGB');rgb.load()
                perceptual,std,bits=phash(rgb)
                result.update(readable=True,status='readable',normalized_width=rgb.width,normalized_height=rgb.height,pixel_sha256=pixel_digest(rgb),phash64_hex=perceptual,grayscale32_std=std,phash_set_bits=bits)
        except (OSError,ValueError,SyntaxError,Image.DecompressionBombError) as error:
            result.update(error_type=type(error).__name__,error=str(error)[:400])
        result['warnings']=' | '.join(f'{type(w.message).__name__}: {w.message}' for w in caught)
    result.pop('sha256')
    return result


def hamming(a,b):return (int(a,16)^int(b,16)).bit_count()


def duplicate_diagnostics(manifest,thresholds):
    valid=manifest[manifest.readable].copy();byindex=valid.set_index('photo_index').to_dict('index')
    pairs=[];groups=[];memberships=[];global_groups=[];global_members=[];business_rows=[]
    group_ids=set()
    def save_group(bid,kind,threshold,indices,edges):
        photos=sorted(byindex[i]['photo_id'] for i in indices)
        gid=kind+'_'+hashlib.sha256((str(threshold)+'|'+bid+'|'+'|'.join(photos)).encode()).hexdigest()[:20]
        assert gid not in group_ids;group_ids.add(gid)
        diameter=max((hamming(byindex[a]['phash64_hex'],byindex[b]['phash64_hex']) for j,a in enumerate(indices) for b in indices[j+1:]),default=0)
        representative=min(indices,key=lambda i:byindex[i]['photo_id'])
        groups.append({'group_id':gid,'business_id':bid,'split':byindex[indices[0]]['split'],'metric':kind,'hamming_threshold':threshold,'photos':len(indices),'representative_index':representative,'representative_photo_id':byindex[representative]['photo_id'],'edges':edges,'phash_diameter':diameter,'chained_beyond_threshold':bool(kind=='near' and diameter>threshold)})
        memberships.extend({'group_id':gid,'photo_index':i} for i in indices)
    for bid,source in manifest.groupby('business_id',sort=True):
        part=source[source.readable].sort_values('photo_id');indices=part.photo_index.astype(int).tolist()
        row={'business_id':bid,'split':source.iloc[0]['split'],'source_photos':len(source),'readable_photos':len(part),'unreadable_photos':int((~source.readable).sum()),'unfiltered_readable_pool_count':len(part),'raw_exact_unique':int(part.raw_sha256.nunique()),'pixel_exact_unique':int(part.pixel_sha256.nunique()),'below5_readable':len(part)<5,'below5_raw_unique':part.raw_sha256.nunique()<5,'below5_pixel_unique':part.pixel_sha256.nunique()<5}
        for kind,column in [('raw','raw_sha256'),('pixel','pixel_sha256')]:
            for _,g in part.groupby(column,sort=True):
                if len(g)>1:save_group(bid,kind,-1,g.photo_index.astype(int).tolist(),len(g)*(len(g)-1)//2)
        local=[]
        for j,a in enumerate(indices):
            for b in indices[j+1:]:
                distance=hamming(byindex[a]['phash64_hex'],byindex[b]['phash64_hex'])
                if distance<=max(thresholds):
                    edge={'photo_index_a':a,'photo_index_b':b,'hamming_distance':distance};pairs.append(edge);local.append(edge)
        for threshold in thresholds:
            parent={i:i for i in indices}
            def find(i):
                while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
                return i
            edges=[edge for edge in local if edge['hamming_distance']<=threshold]
            for edge in edges:
                a=find(edge['photo_index_a']);b=find(edge['photo_index_b']);parent[b]=a
            components={}
            for i in indices:components.setdefault(find(i),[]).append(i)
            row[f'near_t{threshold}_components']=len(components)
            row[f'near_t{threshold}_duplicate_groups']=sum(len(value)>1 for value in components.values())
            row[f'near_t{threshold}_edges']=len(edges)
            row[f'below5_near_t{threshold}']=len(components)<5
            for component in components.values():
                if len(component)>1:
                    nodes=set(component);count=sum(edge['photo_index_a'] in nodes and edge['photo_index_b'] in nodes for edge in edges)
                    save_group(bid,'near',threshold,component,count)
        business_rows.append(row)
    for kind,column in [('raw','raw_sha256'),('pixel','pixel_sha256'),('phash_collision','phash64_hex')]:
        for value,g in valid.groupby(column,sort=True):
            if len(g)<=1:continue
            gid='global_'+kind+'_'+hashlib.sha256(str(value).encode()).hexdigest()[:20]
            bids=sorted(g.business_id.unique());splits=sorted(g.split.unique())
            global_groups.append({'group_id':gid,'metric':kind,'fingerprint':value,'photos':len(g),'businesses':len(bids),'splits':'|'.join(splits),'cross_business':len(bids)>1,'cross_split':len(splits)>1,'confirmed_byte_identity':kind=='raw','confirmed_normalized_pixel_identity':kind=='pixel','perceptual_collision_only':kind=='phash_collision'})
            global_members.extend({'group_id':gid,'photo_index':int(i)} for i in g.photo_index)
    return {'pairs':pd.DataFrame(pairs,columns=['photo_index_a','photo_index_b','hamming_distance']),'groups':pd.DataFrame(groups,columns=['group_id','business_id','split','metric','hamming_threshold','photos','representative_index','representative_photo_id','edges','phash_diameter','chained_beyond_threshold']),'memberships':pd.DataFrame(memberships,columns=['group_id','photo_index']),'global_groups':pd.DataFrame(global_groups,columns=['group_id','metric','fingerprint','photos','businesses','splits','cross_business','cross_split','confirmed_byte_identity','confirmed_normalized_pixel_identity','perceptual_collision_only']),'global_memberships':pd.DataFrame(global_members,columns=['group_id','photo_index']),'businesses':pd.DataFrame(business_rows)}
