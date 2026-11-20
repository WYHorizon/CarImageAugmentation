"""Composite a Qwen windshield repair while keeping protected source pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mask-dir',type=Path,required=True)
    p.add_argument('--accept-near-opaque-rgba',action='store_true')
    a=p.parse_args();out=a.output_dir;maskdir=a.mask_dir
    generation=json.loads((out/'manifest.json').read_text())
    masks=json.loads((maskdir/'manifest.json').read_text())
    source_image=Image.open(out/'source.png').convert('RGB');source=np.asarray(source_image)
    if sha(out/'source.png')!=generation['source_sha256']:raise ValueError('Source hash changed')
    reference=masks['inputs']['source_rgb']
    if not np.array_equal(source,np.asarray(Image.open(reference['path']).convert('RGB'))):
        raise ValueError('Mask reference differs from the Qwen source')
    if sha(out/'raw.png')!=generation['raw_sha256']:raise ValueError('Raw model output changed')
    raw_image=Image.open(out/'raw.png')
    if raw_image.size!=source_image.size:raise ValueError('Raw dimensions changed')
    alpha_review=None
    if raw_image.mode=='RGBA':
        model_alpha=np.asarray(raw_image.getchannel('A'))
        alpha_review={'minimum':int(model_alpha.min()),'maximum':int(model_alpha.max()),
            'nonopaque_fraction':float((model_alpha<255).mean()),'handling':'Stored RGB directly, no white matte; original RGBA retained.'}
        if model_alpha.min()<255 and (not a.accept_near_opaque_rgba or model_alpha.min()<249):
            raise ValueError('Model alpha needs explicit visual review')
    raw=np.asarray(raw_image.convert('RGB'))
    alpha8=np.asarray(Image.open(maskdir/'soft_alpha.png').convert('L'))
    if alpha8.shape!=source.shape[:2]:raise ValueError('Mask dimensions differ')
    alpha=alpha8.astype(np.float32)[...,None]/255
    clean=np.rint(source.astype(np.float32)*(1-alpha)+raw.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
    locked=alpha8==0;clean[locked]=source[locked]
    if not np.array_equal(clean[locked],source[locked]):raise AssertionError('Locked source pixels changed')
    clean_image=Image.fromarray(clean);clean_image.save(out/'clean.png')
    shutil.copy2(maskdir/'soft_alpha.png',out/'edit_alpha.png')
    Image.fromarray(locked.astype(np.uint8)*255).save(out/'protected_mask.png')
    Image.fromarray(np.abs(clean.astype(np.int16)-source.astype(np.int16)).max(2).astype(np.uint8)).save(out/'difference.png')
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',28)
    glass_box=(460,238,1075,393)
    width=(glass_box[2]-glass_box[0])*2;height=(glass_box[3]-glass_box[1])*2
    comparison=Image.new('RGB',(width,2*(height+48)),(245,245,245));draw=ImageDraw.Draw(comparison)
    for i,(im,label) in enumerate([(source_image,'原图 · 前挡风玻璃'),(clean_image,'修复后 · 保留原图细节')]):
        y=i*(height+48);draw.text((12,y+8),label,font=font,fill=(25,25,25))
        comparison.paste(im.crop(glass_box).resize((width,height),Image.Resampling.LANCZOS),(0,y+48))
    comparison.save(out/'comparison_glass.png')
    full=Image.new('RGB',(1536,560),(245,245,245));draw=ImageDraw.Draw(full)
    for i,(im,label) in enumerate([(source_image,'原图'),(clean_image,'仅修复前挡玻璃')]):
        draw.text((i*768+12,8),label,font=font,fill=(25,25,25));full.paste(im.resize((768,512),Image.Resampling.LANCZOS),(i*768,48))
    full.save(out/'comparison_full.png')
    record=dict(status='composited_pending_visual_review',method='Qwen_background_replacement_with_original_detail_lock',
        source_sha256=sha(out/'source.png'),raw_sha256=sha(out/'raw.png'),clean_sha256=sha(out/'clean.png'),
        mask_manifest=str((maskdir/'manifest.json').resolve()),mask_sha256=sha(maskdir/'soft_alpha.png'),
        mix=1.,alpha_formula='soft_alpha / 255; copy original RGB exactly where alpha is zero',
        raw_alpha_review=alpha_review,geometric_transform='none',color_correction='none',
        source_size_wh=list(source_image.size),protected_pixels_exact=int(locked.sum()),
        edit_support_pixels=int((~locked).sum()),changed_pixels_outside_support=0,
        changed_pixels=int(np.any(clean!=source,axis=2).sum()),comparison_crop_xyxy=list(glass_box),
        comparison_display_zoom=2,three_d_modified=False,
        limitation='A 2D appearance repair. Generated dark background is not recovered hidden cabin geometry.')
    (out/'compositing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['protected_pixels_exact','edit_support_pixels','changed_pixels_outside_support','changed_pixels']},indent=2))


if __name__=='__main__':main()
