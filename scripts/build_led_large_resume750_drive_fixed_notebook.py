"""Notebook-only path/visible-output fix; preserve previous notebooks and scientific helpers."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/'outputs/cloud_handoff'

def build():
    original=json.loads((HANDOFF/'LED_LARGE_F_RESUME_CHECKPOINT750_V2.ipynb').read_text(encoding='utf-8'))
    n=copy.deepcopy(original)
    shared=(HANDOFF/'LED_LARGE_RESUME750_DRIVE_FIX.py').read_text(encoding='utf-8').split("\nif __name__ == '__main__':",1)[0]
    for index,mode in ((20,'prepare'),(22,'train'),(24,'evaluate')):
        n['cells'][index]['source']=(shared+f"\n_run_colab_resume_mode('{mode}')\n").splitlines(True)
    for index in (26,28):
        old=''.join(n['cells'][index]['source'])
        assert "DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')" in old
        new=old.replace("DRIVE=Path('/content/drive/MyDrive/LED-Large_multimodal')",'DRIVE=_resolve_colab_drive()')
        n['cells'][index]['source']=(shared+'\n'+new).splitlines(True)
    note='''
### Bản sửa đường dẫn Drive và hiển thị lỗi

Nếu Colab báo không thấy runs, thư mục dữ liệu và shortcut runs có thể nằm ở
hai vị trí khác nhau. Bản này nhận diện các đường dẫn project đã biết, shortcut
đúng ID và runs trực tiếp trong MyDrive, xác minh source binding/gate hash,
rồi dùng đường dẫn đó thống nhất cho prepare/train/evaluate/report/backup.
Không tạo runs rỗng, không chọn ngẫu nhiên nếu có nhiều bản khác nhau.

Nếu không tìm thấy: mở https://drive.google.com/drive/folders/1swRC1tz3u83pgbjybt6wn0MfZsOaH5Z1
bằng đúng tài khoản mount trên Colab, thêm shortcut tên runs trực tiếp vào MyDrive.
Tài khoản cần quyền trên folder gốc; thêm shortcut không tự cấp thêm quyền.
Sau đó chạy lại cell xác minh đường dẫn; không chạy song song notebook cũ.

Mọi lỗi từ tiến trình con được in đầy đủ và lưu log tại /content. Bản sửa chỉ
thay xử lý đường dẫn/output, không thay helper phục hồi, runner, khoa học,
approval, deadline hay các integrity gates. Cell train vẫn chỉ chạy một lần sau preparePASS.
'''
    n['cells'][0]['source']=(''.join(n['cells'][0]['source'])+note).splitlines(True)
    path=HANDOFF/'LED_LARGE_F_RESUME750_DRIVE_FIXED.ipynb'
    n['metadata'].setdefault('colab',{})['name']=path.name
    for c in n['cells']:
        if c['cell_type']=='code':
            compile(''.join(c['source']),str(path),'exec')
            c['execution_count']=None;c['outputs']=[]
    path.write_text(json.dumps(n,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)

if __name__=='__main__':build()
