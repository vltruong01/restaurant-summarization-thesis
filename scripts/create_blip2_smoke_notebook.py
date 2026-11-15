"""Build a direct, user-run BLIP-2 caption notebook; no downloads/model runs here."""
from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def add(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    cell={'cell_type':kind,'id':f'blip2-smoke-{len(cells):03d}','metadata':{},'source':source.splitlines(True)}
    if kind=='code':
        compile(source,cell['id'],'exec')
        cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''
# Thử BLIP-2 tạo caption trực tiếp — 1 ảnh trước

Notebook chạy trực tiếp trong kernel của bạn, không tạo background jobs/cài packages/thuê GPU. Chọn kernel **`.venv-caption-pilot/Scripts/python.exe`** hoặc Python đã có Torch CUDA. Mặc định dùng đúng `Salesforce/blip2-opt-2.7b-coco`, beam3/FP16, một ảnh train của pilot hiện tại. Thay `LOCAL_MODEL_PATH` hoặc `CACHE_DIR_OVERRIDE` nếu lần chạy thành công trước của bạn dùng model/cache ở chỗ khác.

**Tải tự động mặc định OFF.** Run All sẽ tìm weights có sẵn; nếu chưa có đủ thì báo blocked, không bắt đầu tải15GB. Chỉ đổi `ALLOW_MODEL_DOWNLOAD=True` nếu chủ động muốn tải từ Hugging Face. Có thể dùng ảnh riêng qua `IMAGE_PATH`.

Script lịch sử trong `_archive_old/blip2_trials` dùng **`Salesforce/blip2-opt-2.7b` (không COCO)**, beam1/max20/FP32 và offload. Nếu muốn thử model đó, đổi MODEL_ID/REVISION/NUM_BEAMS/MAX_NEW_TOKENS ở ô cấu hình. Kết quả model khác chỉ là diagnostic, không thay checkpoint thesis. Notebook mặc định FP16 và tính CPU budget từ RAM trống, không chép mức24GiBCPU cố định của script cũ.

Mọi kết quả ở `outputs/blip2_notebook_smoke/`, tách khỏi formalE0/caption pool/teacher/training. Không tự lọc captions, thêm dấu câu, đổi reviews/splits hay chốt allocation. **Cell execution PASS không đồng nghĩa pretrained model/caption PASS**: đọc `run_report.status` cuối notebook.
''')
add('code','''
from pathlib import Path
import os,sys,json,hashlib,time,gc,re,datetime
ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'configs/blip2_caption_feasibility_pilot.yaml').is_file())
# Bind pinned preprocessing dependencies read-only inside this kernel, before imports.
substrate=ROOT/'.venv-preprocess/Lib/site-packages'
if substrate.is_dir() and str(substrate) not in sys.path:sys.path.insert(0,str(substrate))
os.environ['TOKENIZERS_PARALLELISM']='false'
from IPython.display import display
import pandas as pd

MODEL_ID='Salesforce/blip2-opt-2.7b-coco'
REVISION='cda95b9319b722f79c9451a3d8ff92eea02048dc' # use 'main' for explicit old non-COCO diagnostic
LOCAL_MODEL_PATH='' # optional complete local snapshot directory, not an offload-only directory
CACHE_DIR_OVERRIDE='' # optional Hugging Face cache ROOT used by your previous kernel
IMAGE_PATH='' # optional image file; empty => unchanged train/validation pilot images
NUM_IMAGES=1 # 1 first; batch remains1 even if you later choose2-5
RUN_INFERENCE=True
ALLOW_MODEL_DOWNLOAD=False
WEIGHT_DTYPE='float16' # explicit float32 diagnostic also supported; no automatic quantization
NUM_BEAMS=3
MAX_NEW_TOKENS=64
PROMPT=None
ALLOW_DISK_OFFLOAD=False # explicit exploratory opt-in, not the formal GPU/CPU-only E0 profile
GPU_RESERVE_GiB=1.25
CPU_RESERVE_GiB=3.0
assert 1<=NUM_IMAGES<=5 and WEIGHT_DTYPE in ['float16','float32']
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
RUN_DIR=ROOT/'outputs/blip2_notebook_smoke'/STAMP
run_report={'status':'configured_not_executed','run_id':STAMP,'model_id':MODEL_ID,'requested_revision':REVISION,
            'local_model_path':LOCAL_MODEL_PATH,'weight_dtype':WEIGHT_DTYPE,'num_beams':NUM_BEAMS,
            'max_new_tokens':MAX_NEW_TOKENS,'prompt':PROMPT,'allow_model_download':ALLOW_MODEL_DOWNLOAD,
            'standalone_diagnostic_not_formal_E0':True,'model_loaded':False,'captions_generated':0,
            'teacher_or_training_started':False,'source_dataset_changed':False}
display(pd.DataFrame([run_report]))
if MODEL_ID!='Salesforce/blip2-opt-2.7b-coco':print('EXPLORATORY MODEL: this does not replace the thesis checkpoint.')
''')
add('markdown','''
## 1. Môi trường và bộ nhớ thực tế

Không cài hoặc sửa environment tự động. Nếu CUDA/dependency không đúng, chọn kernel caption đã chuẩn bị rồi Restart Kernel/Run All. Mặc định import phiên bản pinned nếu workspace có preprocessing environment; không sửa packages global.
''')
add('code','''
import torch,psutil,transformers,huggingface_hub
from PIL import Image,ImageOps
from transformers import Blip2Processor,Blip2ForConditionalGeneration
environment={'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,
             'huggingface_hub':huggingface_hub.__version__,'CUDA':torch.cuda.is_available(),
             'available_RAM_GiB':psutil.virtual_memory().available/2**30}
if environment['CUDA']:
    free,total=torch.cuda.mem_get_info()
    environment.update(GPU=torch.cuda.get_device_name(0),GPU_total_GiB=total/2**30,GPU_free_GiB=free/2**30)
display(pd.DataFrame([environment]))
run_report['environment']=environment
environment_compatible=transformers.__version__=='4.51.3' and huggingface_hub.__version__=='0.36.2'
if not environment_compatible:print('Select the prepared caption kernel for Transformers4.51.3/Hub0.36.2; current imports are incompatible with this tested notebook.')
if not environment['CUDA']:print('CUDA unavailable: no automatic CPU inference fallback.')
''')
add('markdown','''
## 2. Xem ảnh sẽ thử

Mặc định là ảnh của Press'd The Sandwich trong fixedpilot, không phải held-outtest. Đường dẫn được mở read-only; EXIF/RGB chỉ xử lý trong memory.
''')
add('code','''
PLAN=ROOT/'outputs/stage_e/blip2_caption_feasibility_pilot_v1/pilot_photo_plan.json'
if IMAGE_PATH:
    images=[{'image_path':str(Path(IMAGE_PATH).expanduser().resolve()),'photo_id':'custom_image','split':'user_supplied'}]
else:
    images=json.loads(PLAN.read_text(encoding='utf-8'))[:NUM_IMAGES]
for record in images:
    path=Path(record['image_path'])
    if not path.is_absolute():path=ROOT/path
    assert path.is_file(),path
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if 'raw_sha256' in record:assert digest==record['raw_sha256']
    record['absolute_path']=str(path)
    record['input_sha256']=digest
    print(record.get('name',record['photo_id']),record.get('split',''),path)
    with Image.open(path) as original:display(ImageOps.exif_transpose(original).convert('RGB').copy())
''')
add('markdown','''
## 3. Tìm model cache có sẵn

Tìm cả cache mặc định của kernel, HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE và cache của project. Partialrange `.part` files không được coi là weights hoàn chỉnh. `C:/thesis_proposal/blip2_offload` trong script cũ là folder offload, không tự coi là fullcheckpoint.
''')
add('code','''
def complete_snapshot(path):
    path=Path(path)
    if not (path/'config.json').is_file():return False,[]
    index=path/'model.safetensors.index.json'
    if index.is_file():files=sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    else:files=['model.safetensors']
    missing=[name for name in files if not (path/name).is_file() or (path/name).stat().st_size<16]
    return not missing,missing
cache_roots=[]
def add_cache(value):
    if value:
        path=Path(value).expanduser().resolve()
        if path not in cache_roots:cache_roots.append(path)
add_cache(CACHE_DIR_OVERRIDE)
for key in ['HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE']:add_cache(os.environ.get(key))
add_cache(Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub')
add_cache(ROOT/'assets/model_cache')
cache_rows=[]
ready=[]
for cache in cache_roots:
    for model_id in ['Salesforce/blip2-opt-2.7b-coco','Salesforce/blip2-opt-2.7b']:
        snapshots=cache/('models--'+model_id.replace('/','--'))/'snapshots'
        for path in sorted(snapshots.glob('*')) if snapshots.exists() else []:
            ok,missing=complete_snapshot(path)
            cache_rows.append({'model_id':model_id,'revision':path.name,'snapshot':str(path),'complete_weights':ok,'missing_weights':', '.join(missing)})
            if ok and model_id==MODEL_ID and (REVISION=='main' or path.name==REVISION):ready.append((cache,path))
if cache_rows:display(pd.DataFrame(cache_rows))
else:print('No BLIP-2 snapshots found in inspected caches:',[str(p) for p in cache_roots])
if LOCAL_MODEL_PATH:
    MODEL_SOURCE=str(Path(LOCAL_MODEL_PATH).expanduser().resolve())
    CACHE_DIR=str(Path(CACHE_DIR_OVERRIDE).expanduser().resolve()) if CACHE_DIR_OVERRIDE else str(ROOT/'assets/model_cache')
elif ready:
    CACHE_DIR,chosen=ready[0]
    MODEL_SOURCE=str(chosen)
    CACHE_DIR=str(CACHE_DIR)
else:
    MODEL_SOURCE=MODEL_ID
    CACHE_DIR=str(Path(CACHE_DIR_OVERRIDE).expanduser().resolve()) if CACHE_DIR_OVERRIDE else str(ROOT/'assets/model_cache')
print('Model source:',MODEL_SOURCE,'; cache:',CACHE_DIR,'; download allowed:',ALLOW_MODEL_DOWNLOAD)
run_report.update(model_source=MODEL_SOURCE,cache_dir=CACHE_DIR,cache_audit=cache_rows)
''')
add('markdown','''
## 4. Load BLIP-2 — GPU + CPU offload

Nếu weights chưa đủ và downloadOFF, cell báo blocked nhanh; không âm thầm tải tiếp. Không thay model/quantization khi lỗi. RAM/GPU budgets được tính lại từ bộ nhớ trống. Chỉ khi chủ động bật diskoffload mới tạo folder offload riêng cho diagnostic này.
''')
add('code','''
model=None
processor=None
results=[]
def concise_error(error):
    return re.sub(r'https?://[^\\s]+','[URL omitted]',str(error))[:1400]
if not RUN_INFERENCE:
    run_report['status']='inspection_only_inference_disabled'
elif not environment_compatible:
    run_report['status']='blocked_dependency_versions'
elif not environment['CUDA']:
    run_report['status']='blocked_CUDA_unavailable'
elif psutil.virtual_memory().available/2**30<8:
    run_report['status']='blocked_available_RAM_below8GiB'
else:
    begin=time.perf_counter()
    try:
        free,total=torch.cuda.mem_get_info()
        budgets={0:int(min(free,total)-GPU_RESERVE_GiB*2**30),
                 'cpu':int(max(0,psutil.virtual_memory().available-CPU_RESERVE_GiB*2**30))}
        if budgets[0]<2*2**30:raise RuntimeError('GPU budget below2GiB after reserve; no model change/fallback.')
        kwargs={'cache_dir':CACHE_DIR,'local_files_only':not ALLOW_MODEL_DOWNLOAD,'trust_remote_code':False}
        if MODEL_SOURCE==MODEL_ID:kwargs['revision']=REVISION
        processor=Blip2Processor.from_pretrained(MODEL_SOURCE,**kwargs)
        options={**kwargs,'torch_dtype':getattr(torch,WEIGHT_DTYPE),'device_map':'auto',
                 'max_memory':budgets,'low_cpu_mem_usage':True,'offload_state_dict':True,'use_safetensors':True}
        if ALLOW_DISK_OFFLOAD:options['offload_folder']=str(RUN_DIR/'offload')
        model=Blip2ForConditionalGeneration.from_pretrained(MODEL_SOURCE,**options)
        model.eval()
        device_map={key:str(value) for key,value in model.hf_device_map.items()}
        if not ALLOW_DISK_OFFLOAD and 'disk' in device_map.values():raise RuntimeError('Unexpected disk mapping; not permitted by current diagnostic settings.')
        run_report.update(status='model_loaded_pending_caption',model_loaded=True,
                          load_seconds=round(time.perf_counter()-begin,3),device_map=device_map,
                          model_config_commit=getattr(model.config,'_commit_hash',None),
                          model_name_or_path=model.config._name_or_path,
                          GPU_allocated_after_load_GiB=torch.cuda.memory_allocated()/2**30,
                          budgets_bytes={str(k):v for k,v in budgets.items()})
        print('Model loaded:',device_map)
    except Exception as error:
        run_report.update(status='blocked_model_load',error_type=type(error).__name__,error=concise_error(error))
        print('Model NOT loaded:',type(error).__name__,concise_error(error))
        print('If your successful model was cached elsewhere, set LOCAL_MODEL_PATH/CACHE_DIR_OVERRIDE. For old non-COCO weights select that MODEL_ID and REVISION explicitly.')
        model=None
        gc.collect();torch.cuda.empty_cache()
''')
add('markdown','''
## 5. Sinh caption và đo VRAM/thời gian

Batch1; giữ nguyên output text. Không tự thêm dấu chấm/tự lọc. Một pass unprompted mặc định, không giả lập output. Generation errors được ghi và model giải phóng trong finally.
''')
add('code','''
if model is None:
    print('Caption generation skipped. Actual status:',run_report['status'])
else:
    try:
        for record in images:
            torch.cuda.reset_peak_memory_stats()
            tick=time.perf_counter()
            with Image.open(record['absolute_path']) as original:
                rgb=ImageOps.exif_transpose(original).convert('RGB')
                inputs=processor(images=rgb,return_tensors='pt',**({'text':PROMPT} if PROMPT else {}))
            dtype=getattr(torch,WEIGHT_DTYPE)
            inputs={key:value.to('cuda:0',dtype=dtype) if value.is_floating_point() else value.to('cuda:0') for key,value in inputs.items()}
            prefix=inputs['input_ids'].shape[1] if 'input_ids' in inputs else 1+(model.config.num_query_tokens if getattr(model.config,'image_token_index',None) is not None else 0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=WEIGHT_DTYPE=='float16'):
                output=model.generate(**inputs,num_beams=NUM_BEAMS,do_sample=False,max_new_tokens=MAX_NEW_TOKENS)
            torch.cuda.synchronize()
            suffix=output[0,prefix:] if model.config.use_decoder_only_language_model else output[0]
            caption=processor.tokenizer.decode(suffix,skip_special_tokens=True).strip()
            row={**record,'raw_caption':caption,'seconds':round(time.perf_counter()-tick,3),
                 'generated_suffix_tokens':int(suffix.numel()),
                 'peak_GPU_allocated_GiB':round(torch.cuda.max_memory_allocated()/2**30,3),
                 'peak_GPU_reserved_GiB':round(torch.cuda.max_memory_reserved()/2**30,3),
                 'words_diagnostic':len(caption.split()),'terminal_punctuation_diagnostic':caption.endswith(('.', '!', '?')),
                 'caption_quality':'manual_review_pending','formal_thesis_caption':False}
            results.append(row)
            print('RAW CAPTION:',caption)
        run_report.update(status='actual_captions_generated_pending_manual_review',captions_generated=len(results))
    except Exception as error:
        run_report.update(status='blocked_generation_partial_outputs_preserved',captions_generated=len(results),
                          error_type=type(error).__name__,error=concise_error(error))
        print('Generation failure:',type(error).__name__,concise_error(error))
    finally:
        del model
        model=None
        gc.collect();torch.cuda.empty_cache()
        run_report['model_released']=True
''')
add('markdown','''
## 6. Kết quả thực tế

Đọc status, rawcaption, memory/time để biết có thực sự chạy được không. Cache/model-load failure vẫn tạo report rõ ràng, nhưng không tạo caption giả. Không nhập các kết quả standalone vào datasettrain hoặc formalE0 tự động.
''')
add('code','''
RUN_DIR.mkdir(parents=True,exist_ok=True)
(RUN_DIR/'run_report.json').write_text(json.dumps(run_report,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
if results:
    (RUN_DIR/'raw_captions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
    display(pd.DataFrame(results)[['photo_id','raw_caption','seconds','peak_GPU_allocated_GiB','peak_GPU_reserved_GiB','caption_quality']])
else:print('Actual generated captions:0. This is NOT a successful pretrained caption test.')
display(pd.DataFrame([run_report]))
print('Reports:',RUN_DIR)
print('GPU allocated after cleanupGiB:',torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0)
''')
add('markdown','''
## Sau khi chạy

- Nếu có rawcaption: đối chiếu với ảnh, đọc VRAM/time; đây mới là standalone smoke, chưa xác nhận chất lượng/capacity của pilot24ảnh.
- Nếu báo missingweights: tìm đúng cache/model path từ lần chạy thành công trước, hoặc chủ động bật download; không cần thuêGPU chỉ để mở notebook này.
- Nếu OOM/load/dependency failure: gửi run_report để xem lỗi cụ thể; không tự đổi checkpoint/quantization/source length.
- Notebook00 vẫn là index tiến độ/dataset/filtering chính. Notebook08 là harness thử model riêng theo yêu cầu mới, không thay hoặc chạy lại C1/C2/D1.
''')

notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3 (caption environment)','language':'python','name':'python3'},
           'language_info':{'name':'python','version':'3.11.7','file_extension':'.py','mimetype':'text/x-python','pygments_lexer':'ipython3'}},
          'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/08_blip2_caption_smoke_test.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(f'Created {path}; {sum(c["cell_type"]=="code" for c in cells)} code cells. No model executed by builder.')
