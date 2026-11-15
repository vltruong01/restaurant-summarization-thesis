"""Read-only acquisition/inspection of the publisher's TripAdvisor restaurant data.

Writes only a new Stage O workspace. Does not caption, infer, train, or collect ratings.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import urllib.parse
import zipfile
import datetime
import io
import pickle

ALLOWED_PICKLE_GLOBALS={
    'pandas.core.frame':{'DataFrame'},
    'pandas.core.internals.managers':{'BlockManager'},
    'pandas._libs.internals':{'_unpickle_block'},
    'numpy.core.multiarray':{'_reconstruct'},
    'numpy.core.numeric':{'_frombuffer'},
    'numpy':{'ndarray','dtype'},
    'builtins':{'slice'},
    'datetime':{'date','datetime'},
    'pandas.core.indexes.base':{'_new_Index','Index'},
    'pandas.core.indexes.range':{'RangeIndex'},
}


class DatasetUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if name not in ALLOWED_PICKLE_GLOBALS.get(module,set()):
            raise ValueError('Unsupported dataset pickle global: '+module+'.'+name)
        return super().find_class(module,name)


def load_tables(path):
    tables={}
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if member.endswith(('items.pkl','restaurants.pkl','reviews.pkl')):
                tables[Path(member).stem]=DatasetUnpickler(io.BytesIO(z.read(member))).load()
    return tables

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stage_o/tripadvisor_external_v1'
RAW=OUT/'DATASET_DOWNLOADS'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path,kind='sha256'):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,kind).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def acquire(record,filename,expected_md5):
    RAW.mkdir(parents=True,exist_ok=True)
    destination=RAW/f'{record}_{filename}'
    url=f'https://zenodo.org/records/{record}/files/{urllib.parse.quote(filename)}?download=1'
    if not destination.exists():
        temporary=destination.with_suffix('.partial')
        request=urllib.request.Request(url,headers={'User-Agent':'Thesis-Stage-O-ReadOnly-Dataset-Audit/1.0'})
        with urllib.request.urlopen(request,timeout=45) as response,temporary.open('wb') as f:
            status=response.status
            while True:
                block=response.read(1024*1024)
                if not block:break
                f.write(block)
        assert digest(temporary,'md5')==expected_md5,'Publisher checksum mismatch'
        temporary.rename(destination)
    assert digest(destination,'md5')==expected_md5
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        members=[{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size} for x in z.infolist()]
    receipt=dict(record=record,url=url,downloaded_or_verified_at_utc=utc(),bytes=destination.stat().st_size,
                 publisher_md5=expected_md5,actual_md5=digest(destination,'md5'),sha256=digest(destination),zip_crc='PASS',members=members)
    save(RAW/f'{record}_ACQUISITION.json',receipt)
    return destination,receipt


if __name__=='__main__':
    path,receipt=acquire('14622324','Gijón.zip','1f43c25c039625073eb4139c4e455def')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    tables=load_tables(path)
    for name,df in tables.items():
        print(name,df.shape,df.dtypes.astype(str).to_dict())
        print(df.drop(columns=['userId','name'],errors='ignore').head(1).to_dict('records'))
        if 'language' in df:
            print('languages',df.language.value_counts(dropna=False).head(15).to_dict())
            print('ratings',df.rating.value_counts(dropna=False).to_dict())
            en=df[df.language.eq('en')];counts=en.groupby('itemId').size()
            print('EnglishEligible',int(counts.ge(10).sum()),'Englishreviewtotal',len(en))
