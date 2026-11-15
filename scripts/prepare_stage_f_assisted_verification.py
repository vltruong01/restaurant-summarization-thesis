"""Freeze D-028 verification order before AI labels; build blinded contact sheets."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import math
import textwrap
import yaml
from stage_f_verification_workflow import read, write, sha, canonical_sha, verification_orders, active_ids

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/stage_f/validation_calibration_v1'
OUT = ROOT / 'outputs/stage_f/ai_assisted_calibration_v1'


def prepare():
    cfg = yaml.safe_load((ROOT / 'configs/stage_f_human_verification.yaml').read_text(encoding='utf-8'))
    dataset = read(BASE / 'human_check/annotation_dataset.json')
    private = read(BASE / 'human_check/private_calibration_scores.json')
    plan = {r['photo_id']: r for r in read(BASE / 'validation_photo_plan.json')}
    roles = {r['business_id']: r['role'] for r in read(BASE / 'business_roles.json')}
    assert dataset['dataset_kind'] == 'official_calibration'
    assert Counter(c['kind'] for c in dataset['cases']) == {'image_caption': 200, 'caption_pair': 400}
    assert all(roles[c['business_id']] == 'calibration' and
               all(plan[p]['split'] == 'validation' for p in c['photo_ids']) for c in dataset['cases'])
    order = {'schema_version': 2, 'decision': 'D-028', 'dataset_sha256': dataset['dataset_sha256'],
             'workflow_sha256': sha(ROOT / 'configs/stage_f_human_verification.yaml'),
             'private_stratification_scores_sha256': sha(BASE / 'human_check/private_calibration_scores.json'),
             'role_manifest_sha256': sha(BASE / 'business_roles.json'),
             'selection_reads_AI_or_human_labels': False,
             'orders': verification_orders(dataset, private, plan)}
    order['manifest_sha256'] = canonical_sha(order)
    destination = OUT / 'verification_order.json'
    if destination.exists(): assert read(destination) == order, 'Frozen verification order changed'
    else: write(destination, order)
    selection = {'image_cases': cfg['initial_image_cases'], 'pair_cases': cfg['initial_pair_cases'],
                 'round': 0, 'verification_order_sha256': order['manifest_sha256']}
    if not (OUT / 'active_selection.json').exists(): write(OUT / 'active_selection.json', selection)
    selected = active_ids(order, selection); lookup = {c['case_id']: c for c in dataset['cases']}
    initial = {'status': 'INITIAL_VERIFICATION_SET_FROZEN_BEFORE_AI_LABELS', 'image_cases': 50, 'pair_cases': 100,
               'image_businesses': len({lookup[c]['business_id'] for c in selected[:50]}),
               'pair_businesses': len({lookup[c]['business_id'] for c in selected[50:]}),
               'active_case_ids': selected, 'verification_order_sha256': order['manifest_sha256']}
    assert initial['image_businesses'] == initial['pair_businesses'] == 40
    target = OUT / 'initial_selection.json'
    if target.exists(): assert read(target) == initial
    else: write(target, initial)
    print('FROZEN_INITIAL_50_IMAGES_100_PAIRS_ALL_40_BUSINESSES_EACH')
    return dataset


def galleries(dataset):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    groups = defaultdict(list)
    for i, case in enumerate(dataset['cases']): groups[case['business_id']].append((i, case))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    batches = []
    for number, (bid, cases) in enumerate(sorted(groups.items()), 1):
        photos = {}
        for i, c in cases:
            for j, pid in enumerate(c['photo_ids']):
                photos[pid] = {'photo_id': pid, 'image_path': c['image_paths'][j],
                               'caption': c['normalized_captions'][j]}
        ordered = sorted(photos.values(), key=lambda p: p['photo_id'])
        ids = {p['photo_id']: i for i, p in enumerate(ordered)}
        panels = []
        for offset in range(0, len(ordered), 18):
            page = ordered[offset:offset+18]
            canvas = Image.new('RGB', (1260, 430 * math.ceil(len(page)/3)), 'white')
            draw = ImageDraw.Draw(canvas)
            for index, p in enumerate(page, offset):
                cell = index-offset; x=(cell%3)*420; y=(cell//3)*430
                draw.text((x+6,y+5),f"P{index}: {p['photo_id']}",font=small,fill='black')
                source = ROOT / p['image_path']
                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB'); image.thumbnail((400,290))
                    canvas.paste(image,(x+8,y+35))
                lines=textwrap.wrap(p['caption'],width=43)
                assert len(lines)<=6, 'Caption needs a taller gallery; do not crop text'
                draw.multiline_text((x+8,y+335),'\n'.join(lines),font=small,fill='black',spacing=2)
                draw.rectangle((x,y,x+419,y+429),outline='#888888')
            name=f'business_{number:02d}_{offset//18+1}.png'
            path=OUT/'ai_review_galleries'/name;path.parent.mkdir(exist_ok=True);canvas.save(path)
            panels.append(path.relative_to(ROOT).as_posix())
        entry={'batch':number,'business_id':bid,'name':cases[0][1]['name'],'panels':panels,
               'photos':[{**p,'P':i} for i,p in enumerate(ordered)],
               'cases':[{'case_index':i,'case_id':c['case_id'],'kind':c['kind'],
                         'photos':[ids[p] for p in c['photo_ids']], 'label_keys':list(c['labels'])} for i,c in cases]}
        batches.append(entry)
    write(OUT/'ai_review_batches.json',batches)
    print('BLINDED_REVIEW_GALLERIES',len(batches),'business batches; no model scores shown')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--galleries',action='store_true');args=parser.parse_args()
    data=prepare()
    if args.galleries:galleries(data)
