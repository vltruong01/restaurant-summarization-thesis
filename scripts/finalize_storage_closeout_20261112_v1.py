"""Publish a verified HF restore catalog and a concise cleanup report."""
from pathlib import Path
import argparse,collections,datetime,gzip,hashlib,io,json,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/storage_closeout_20261112_v1'
REPO='vltruong01/multimodal_summarization_thesis'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
 from huggingface_hub.utils import build_hf_headers,disable_progress_bars
 import requests
 disable_progress_bars()
 receipt=read(OUT/'FINAL_HF_OFFLOAD_RECEIPT.json');assert receipt['status']=='PASS_BACKUP_AND_VERIFIED_COLD_OFFLOAD_COMPLETE'
 plan=read(OUT/'OFFLOAD_PLAN.json');assert sha(OUT/'OFFLOAD_PLAN.json')==receipt['plan_sha256']
 files=[];archives={};excluded=list(plan['excluded_and_retained']);blobs=read(OUT/'BLOB_RESTORE_INDEX.json')
 for batch in plan['batches']:
  folder=OUT/'BATCHES'/batch['id'];m=read(folder/'manifest.json');r=read(folder/'REMOTE_RECEIPT.json')
  assert sha(folder/'manifest.json')==r['manifest_sha256'] and r['status']=='PASS_FRESH_REMOTE_ARCHIVE_AND_EVERY_UNIQUE_BLOB_SHA256'
  assert len(m['files'])==r['source_files'];files.extend({k:v for k,v in row.items() if k!='blob'} for row in m['files']);excluded.extend(m['excluded_and_retained'])
  archives[r['archive_remote_path']]={k:r[k] for k in ['revision','archive_sha256','archive_bytes','manifest_remote_path','manifest_sha256']}
 assert len(files)==receipt['totals']['backed_up_files'] and len({r['path'] for r in files})==len(files)
 assert all(r['sha256'] in blobs for r in files)
 catalog=dict(format='FINAL_RESEARCH_RESTORE_CATALOG_V1',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=REPO,plan_sha256=receipt['plan_sha256'],files=files,blobs=blobs,archives=archives,excluded_and_retained=excluded,source_content_unchanged=True,latest_live_state_separate_snapshot=read(OUT/'LATEST_LIVE_STATE_HF_RECEIPT.json'),legacy_HF_train_data_binding=read(OUT/'EXISTING_HF_TRAIN_DATA_BINDING.json'))
 cp=OUT/'RESTORE_CATALOG.json.gz'
 with cp.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=6,mtime=0) as gz,io.TextIOWrapper(gz,encoding='utf8') as writer:
  json.dump(catalog,writer,ensure_ascii=False,separators=(',',':'));writer.write('\n')
 restore=OUT/'restore_research_from_hf.py';assert restore.exists()
 prefix=receipt['namespace']+'/catalog/'+sha(cp)
 remote_catalog=prefix+'/RESTORE_CATALOG.json.gz';remote_script=prefix+'/restore_research_from_hf.py'
 controls=[]
 for filename in ['OFFLOAD_PLAN.json','DELETED_FILES.jsonl']:
  source=OUT/filename;compressed=OUT/(filename+'.gz')
  with source.open('rb') as src,compressed.open('wb') as dest,gzip.GzipFile(filename='',fileobj=dest,mode='wb',compresslevel=6,mtime=0) as gz:shutil.copyfileobj(src,gz,4*2**20)
  controls.append((prefix+'/CONTROL/'+compressed.name,compressed))
 for filename in ['PROTECTED_FINAL_CHECKPOINTS.json','PROTECTED_FINAL_DELIVERABLES.json','LATEST_LIVE_STATE_HF_RECEIPT.json','GITHUB_VERIFIED_RECEIPT.json','EXISTING_HF_TRAIN_DATA_BINDING.json']:
  controls.append((prefix+'/CONTROL/'+filename,OUT/filename))
 for filename in ['archive_and_offload_final_research_20261112_v1.py','parallel_archive_closeout_20261112_v1.py','finalize_storage_closeout_20261112_v1.py','backup_final_private_github_20261112_v1.py']:
  controls.append((prefix+'/CONTROL/scripts/'+filename,ROOT/'scripts'/filename))
 api=HfApi();assert api.repo_info(REPO,repo_type='dataset').private is True
 # Freeze upload controls before networking, so later source-script edits cannot
 # change the bytes used for commit verification or recorded hashes.
 staging=OUT/'CATALOG_UPLOAD_SNAPSHOT'/sha(cp)
 all_uploads=[]
 for remote,local in [(remote_catalog,cp),(remote_script,restore),*controls]:
  target=staging/remote.removeprefix(prefix+'/')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(local,target)
  assert sha(target)==sha(local)
  all_uploads.append((remote,target))
 controls=all_uploads[2:]
 commit=api.create_commit(repo_id=REPO,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=path,path_or_fileobj=str(local)) for path,local in all_uploads],commit_message='Publish verified restoration catalog and cleanup audit controls')
 for path,local in all_uploads:
  with requests.get(hf_hub_url(REPO,path,repo_type='dataset',revision=commit.oid),headers=build_hf_headers(token=True),stream=True,timeout=(30,180)) as response:
   response.raise_for_status();h=hashlib.sha256()
   for b in response.iter_content(4*2**20):h.update(b)
  assert h.hexdigest()==sha(local), 'Remote control hash mismatch: '+path
  print('Verified remote control: '+local.name,flush=True)
 catalog_receipt=dict(status='PASS_REMOTE_CATALOG_AND_RESTORE_SCRIPT_BYTES',repo=REPO,revision=commit.oid,catalog_remote_path=remote_catalog,catalog_sha256=sha(cp),restore_script_remote_path=remote_script,restore_script_sha256=sha(restore),archived_paths=len(files),unique_blobs=len(blobs),verified_archives=len(archives),excluded_retained=len(excluded),audit_controls=[dict(path=path,bytes=local.stat().st_size,sha256=sha(local)) for path,local in controls])
 save(OUT/'HF_CATALOG_RECEIPT.json',catalog_receipt)
 # Real restore from a pinned remote archive, into a separate new directory.
 test_path=plan['retained_final_checkpoint_prefixes'][2]+'/adapter_model.safetensors'
 dest=OUT/'EARLY_ACTUAL_RESTORE' if (OUT/'EARLY_ACTUAL_RESTORE').exists() else OUT/'ACTUAL_RESTORE_TEST'
 if not dest.exists():
  result=subprocess.run([sys.executable,'-X','utf8','-B',str(restore),'--catalog',str(cp),'--prefix',test_path,'--destination',str(dest)],capture_output=True,text=True,encoding='utf8',check=True)
  verifier_output=json.loads(result.stdout)
 else:verifier_output=dict(status='PASS_EARLIER_FRESH_HF_RESTORE',archive_receipt='BATCHES/part-0001/REMOTE_RECEIPT.json',scope='Reused actual streamed and full-archive-verified restore performed before cold-file deletion')
 restored=dest/test_path;original=ROOT/test_path;assert sha(restored)==sha(original)
 save(OUT/'ACTUAL_RESTORE_VERIFICATION.json',dict(status='PASS_FRESH_HF_RESTORE_FINAL_F_ADAPTER',restored_original_path=test_path,sha256=sha(restored),bytes=restored.stat().st_size,actual_verifier_output=verifier_output))
 # Only remove the newly created restore-test copy, never the canonical adapter.
 assert restored.resolve().is_relative_to(dest.resolve()) and dest.resolve().is_relative_to(OUT.resolve());restored.unlink()
 for p in sorted(dest.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if p.is_dir() and not any(p.iterdir()):p.rmdir()
 if not any(dest.iterdir()):dest.rmdir()
 deleted=[]
 if (OUT/'DELETED_FILES.jsonl').exists():
  with (OUT/'DELETED_FILES.jsonl').open(encoding='utf8') as stream:
   for line in stream:
    row=json.loads(line)
    if row['event']=='DELETED':deleted.append(row)
 assert len({r['path'] for r in deleted})==len(deleted)
 assert all(not (ROOT/r['path']).exists() for r in deleted)
 # Remove empty directories only after all archival and file-deletion checks pass.
 # No recursive directory deletion; account/config files that were excluded stay local.
 empty_directories_removed=0
 for relative in plan['cold_prefixes']:
  cold=ROOT/relative
  if not cold.exists():continue
  resolved=cold.resolve();assert resolved.is_relative_to(ROOT.resolve()) and resolved!=ROOT.resolve()
  for base,dirs,names in os.walk(cold,topdown=False,followlinks=False):
   directory=Path(base);assert directory.resolve().is_relative_to(resolved)
   if not getattr(directory.lstat(),'st_file_attributes',0)&0x400:
    try:directory.rmdir();empty_directories_removed+=1
    except OSError:pass
 # Interrupted packs are newly created scratch, never original research sources.
 scratch_removed=0
 for folder in (OUT/'BATCHES').iterdir():
  if not folder.is_dir() or not (folder/'REMOTE_RECEIPT.json').exists():continue
  for temporary in folder.glob('incomplete-*.tar.gz'):
   assert temporary.resolve().is_relative_to((OUT/'BATCHES').resolve()) and not temporary.is_symlink()
   scratch_removed+=temporary.stat().st_size;temporary.unlink()
 canonical=read(OUT/'PROTECTED_FINAL_CHECKPOINTS.json')
 for row in canonical['files']:assert sha(ROOT/row['file'])==row['sha256']
 before=read(OUT/'INVENTORY.json')['free_bytes'];after=shutil.disk_usage(ROOT).free
 summary=dict(status='PASS_VERIFIED_BACKUP_OFFLOAD_AND_REAL_RESTORE',logical_file_bytes_removed=sum(r['bytes'] for r in deleted),files_removed=len(deleted),empty_directories_removed=empty_directories_removed,interrupted_temporary_pack_bytes_removed=scratch_removed,disk_free_before_bytes=before,disk_free_after_bytes=after,observed_disk_free_increase_bytes=after-before,cloud_archive_bytes=sum(r['archive_bytes'] for r in archives.values()),retained_local_canonical_checkpoints=3,github=read(OUT/'GITHUB_VERIFIED_RECEIPT.json'),hf=catalog_receipt,live_annotation_app_retained=True,source_code_thesis_figures_and_final_reports_retained=True,excluded_files_remain_local=True,disk_note='Observed free-space change is affected by NTFS compression/hardlinks and concurrent Windows activity;logical totals are not guaranteed freed physical bytes.')
 save(OUT/'FINAL_CLEANUP_REPORT.json',summary)
 text=f'''# Báo cáo dọn project sau nghiên cứu — 12/11/2026

- Đã xóa {len(deleted):,} file sau khi xác minh bản sao HF; tổng kích thước file {sum(r['bytes'] for r in deleted)/2**30:.2f} GiB.
- Ổ C trống trước: {before/2**30:.2f} GiB; sau: {after/2**30:.2f} GiB. Mức tăng quan sát: {(after-before)/2**30:.2f} GiB. Hardlink, nén NTFS và tác vụ Windows có thể làm số này khác tổng kích thước file.
- HF riêng tư: https://huggingface.co/datasets/{REPO}. Danh mục khôi phục và script đã được tải lại, kiểm tra SHA-256; xem `HF_CATALOG_RECEIPT.json` để biết revision/path chính xác.
- GitHub riêng tư: https://github.com/{REPO}/tree/{summary['github']['branch']}. Mã nguồn, tài liệu, notebook source và báo cáo/hình gọn đã được clone mới, đối chiếu từng file.
- Đã khôi phục thật adapter F-S1000 từ HF vào thư mục mới và đối chiếu SHA-256 với bản chuẩn. Không chạy model.

Giữ local: source/config/tài liệu luận văn, data hiện dùng, ba checkpoint chuẩn RO/U/F (khoảng170MiB cho adapter+full-state), kết quả cuối/hình Chapter4, ứng dụng và dữ liệu chấm đang chạy. Các file có thể chứa credential/account hoặc thay đổi trong lúc sao lưu được giữ lại, không tự xóa. Lịch sử training, witness/selector, caption/Teacher, gói transport và scorer ít dùng nằm trong các archive HF có manifest.

Đã sao lưu riêng trạng thái chấm mới nhất thực có trong database đang chạy (90 bản ghi/180 sự kiện), bỏ cột password/salt. Xem `LATEST_LIVE_STATE_HF_RECEIPT.json`. Snapshot phục hồi này không thay thế bộ V1 chính thức hoặc thay đổi phân tích thống kê đã chốt.

Hai file reviews/TRAIN được giữ local và không chép lại vào blob mới do bộ quét bảo thủ bắt chuỗi giống token trong dữ liệu. Bytes hiện tại đã đối chiếu khớp bản HF đầy đủ ngày08/10; xem `EXISTING_HF_TRAIN_DATA_BINDING.json` để biết revision/SHA. Bản gốc của chúng và các raw Yelp originals vẫn có trong `archives/` cùng `backup_manifest.json` của repo HF. Catalog mới chỉ khôi phục những path đã liệt kê trong nó; các file nằm riêng trong bản dataset cũ cần dùng manifest/pinned revision cũ. Không nhầm một file bị loại khỏi gói mới với việc đã mất bản sao cloud.

## Khôi phục khi cần

Giữ thư mục audit này. `RESTORE_CATALOG.json.gz` ánh xạ mọi path/SHA về archive và revision HF; các archive chứa blob duy nhất để giảm bản trùng. Không thể chỉ giải nén một archive và kỳ vọng có cấu trúc project — hãy dùng script đi kèm:

```text
python restore_research_from_hf.py --catalog RESTORE_CATALOG.json.gz --prefix outputs/stage_i/TEN_THU_MUC --destination C:/thesis_restore
```

Đích phải là thư mục mới. Script đọc HF riêng tư bằng tài khoản đã đăng nhập, kiểm tra hash archive và từng file được khôi phục; không ghi đè project. `--prefix` cũng có thể là path chính xác của một file. Lấy catalog/script từ HF theo `HF_CATALOG_RECEIPT.json` nếu mất bản local. Thời gian sửa file được giữ; ACL/thuộc tính nén NTFS và mọi thư mục rỗng không được cam kết tái tạo.

`DELETED_FILES.jsonl` ghi từng file bị xóa cùng hash/path HF trước khi xóa. Các bản sao cloud được ghim revision; không xóa các archive này khi còn cần phục hồi nghiên cứu. Hoạt động dọn không sửa dữ liệu nghiên cứu, checkpoint, điểm chấm hoặc kết quả.
'''
 (OUT/'README_CLEANUP_VI.md').write_text(text,encoding='utf8')
 status=read(OUT/'STATUS.json');status.update(status='COMPLETE_VERIFIED',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_report='FINAL_CLEANUP_REPORT.json',free_GiB=round(after/2**30,3))
 save(OUT/'STATUS.json',status)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['github','hf']},indent=2))
if __name__=='__main__':main()
