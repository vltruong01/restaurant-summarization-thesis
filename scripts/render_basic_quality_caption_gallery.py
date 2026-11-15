"""Actual E1 original photos, raw text and separately normalized captions."""
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/stage_e/blip2_caption_basic_quality_pilot_v2'


def figures():
    frame = pd.read_csv(OUT/'caption_statistics.csv', keep_default_na=False)
    for offset in range(0, len(frame), 6):
        fig, axes = plt.subplots(3, 2, figsize=(13, 15))
        for ax in axes.flatten(): ax.axis('off')
        for ax, (_, r) in zip(axes.flatten(), frame.iloc[offset:offset+6].iterrows()):
            with Image.open(ROOT/r.image_path) as image:
                ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
            raw = textwrap.fill(r.raw_caption, width=76)
            normalized = textwrap.fill(r.normalized_caption, width=76)
            ax.set_title(f'{r["name"]} | {r.photo_id} | {r.split}\nRaw: {raw}\nNormalized: {normalized}\n'
                         f'basic_ok={r.basic_eligible}; attempts={r.attempts}; EOS={r.eos_seen}; cap={r.token_cap_reached}', fontsize=8)
        fig.suptitle('Actual E1 minimal Basic Quality pilot — raw preserved, no substantive filtering', fontsize=11, y=1.025)
        yield fig, f'blip2_basic_quality_gallery_{offset//6+1}.png'
