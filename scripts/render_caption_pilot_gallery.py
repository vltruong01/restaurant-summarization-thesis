"""Show real E0 outputs with their original photos, no image/source changes."""
from pathlib import Path
import textwrap
import pandas as pd,yaml
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]


def figures():
    cfg=yaml.safe_load((ROOT/'configs/blip2_caption_feasibility_pilot.yaml').read_text(encoding='utf-8'));out=ROOT/cfg['output_directory']
    frame=pd.read_csv(out/'raw_caption_statistics.csv',keep_default_na=False)
    for offset in range(0,len(frame),6):
        part=frame.iloc[offset:offset+6];fig,axes=plt.subplots(3,2,figsize=(13,12));flat=axes.flatten()
        for ax in flat:ax.axis('off')
        for ax,(_,row) in zip(flat,part.iterrows()):
            with Image.open(ROOT/row.image_path) as image:ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
            title=f'{row["name"]} | {row["photo_id"]} | {row["split"]}\nRaw: '+textwrap.fill(row.raw_caption,width=76)+f'\nwords={row.word_count}; LED={row.LED_body_tokens}; surface_ok={row.surface_eligible}'
            ax.set_title(title,fontsize=8)
        fig.suptitle('Actual BLIP-2 pilot captions — manual picture-caption review, not final datasets',fontsize=11)
        yield fig,f'blip2_caption_pilot_gallery_{offset//6+1}.png'
