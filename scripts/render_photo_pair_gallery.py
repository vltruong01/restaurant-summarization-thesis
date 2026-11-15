"""Small original-image pair galleries; no source image writes."""
from pathlib import Path
import pandas as pd,yaml
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]


def figures():
    cfg=yaml.safe_load((ROOT/'configs/photo_integrity_duplicate_audit.yaml').read_text(encoding='utf-8'));out=ROOT/cfg['output_directory'];examples=pd.read_csv(out/'visual_pair_examples.csv',keep_default_na=False)
    for offset in range(0,len(examples),4):
        part=examples.iloc[offset:offset+4];fig,axes=plt.subplots(len(part),2,figsize=(13,3.1*len(part)),squeeze=False)
        for i,(_,row) in enumerate(part.iterrows()):
            for j,side in enumerate(['a','b']):
                with Image.open(ROOT/cfg['cohort_directory']/row['path_'+side]) as image:rgb=ImageOps.exif_transpose(image).convert('RGB');axes[i,j].imshow(rgb)
                axes[i,j].set_title(f"{row['example_id']} | {row['name_'+side]}\n{row['photo_id_'+side]} | {row['split_'+side]}\nd={row['hamming_distance']} | raw_equal={row['raw_equal']} | pixel_equal={row['pixel_equal']}",fontsize=8)
                axes[i,j].axis('off')
        fig.suptitle('Train/validation original-photo examples — diagnostic pairs, no duplicate policy adopted',fontsize=11)
        yield fig,f'photo_audit_pair_gallery_{offset//4+1}.png'


def main():
    plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
    out=ROOT/'outputs/unified_dataset_inspection/figures';out.mkdir(parents=True,exist_ok=True)
    for fig,name in figures():fig.tight_layout();fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig);print(name)


if __name__=='__main__':main()
