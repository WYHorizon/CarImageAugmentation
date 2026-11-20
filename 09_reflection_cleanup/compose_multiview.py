"""Apply reviewed masks to Qwen candidates, retaining exact original details."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mask-dir',type=Path,required=True)
    p.add_argument('--accept-near-opaque-rgba',action='store_true')
    a=p.parse_args();out=a.output_dir
    generated=json.loads((out/'manifest.json').read_text())
    if generated['status']!='raw_candidates_complete_pending_protection_and_review':
        raise ValueError('Wait for all Qwen candidates')
    variant=generated.get('variant','mild')
    if variant not in ('mild','strong'):raise ValueError('Unknown cleanup variant')
    mix=.85 if variant=='strong' else .45
    if generated.get('intended_compositing_mix')!=mix:raise ValueError('Variant and declared mixing weight disagree')
    label='重度去反光' if variant=='strong' else '轻度去反光'
    record=dict(method=f'protected_original_plus_{round(mix*100)}_percent_Qwen',mix=mix,variant=variant,views=[],
        mask_directory=str(a.mask_dir.resolve()),geometric_transform='none',
        limitation='Independent single-image cleanup, not physical albedo or multiview-consistency optimization')
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',24)
    pairs=[];raw_pairs=[]
    for v in generated['views']:
        name=v['id'];folder=out/name;maskdir=a.mask_dir/name
        src=np.asarray(Image.open(folder/'source.png').convert('RGB'))
        if sha(folder/'source.png')!=v['source_sha256']:raise ValueError('Source hash changed')
        if sha(folder/'raw.png')!=v['raw_sha256']:raise ValueError('Raw hash changed')
        mask_record=json.loads((maskdir/'manifest.json').read_text())
        ref=mask_record['inputs']['source_rgb']
        if not np.array_equal(src,np.asarray(Image.open(ref['path']).convert('RGB'))):
            raise ValueError(f'Mask belongs to another image: {name}')
        alpha8=np.asarray(Image.open(maskdir/'soft_alpha.png').convert('L'))
        if alpha8.shape!=src.shape[:2]:raise ValueError('Mask size mismatch')
        raw_image=Image.open(folder/'raw.png')
        if raw_image.size!=(src.shape[1],src.shape[0]):raise ValueError('Raw size mismatch')
        alpha_review=None
        if raw_image.mode=='RGBA':
            model_alpha=np.asarray(raw_image.getchannel('A'))
            alpha_review=dict(minimum=int(model_alpha.min()),maximum=int(model_alpha.max()),
                nonopaque_fraction=float((model_alpha<255).mean()))
            if model_alpha.min()<255 and (not a.accept_near_opaque_rgba or model_alpha.min()<249):
                raise ValueError(f'{name}: inspect model alpha before accepting')
        raw=np.asarray(raw_image.convert('RGB'))
        alpha=alpha8.astype(np.float32)[...,None]/255*mix
        clean=np.rint(src.astype(np.float32)*(1-alpha)+raw.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        locked=alpha8==0;clean[locked]=src[locked]
        if not np.array_equal(clean[locked],src[locked]):raise AssertionError('Protected pixels changed')
        if name=='side_004' and not np.array_equal(clean,np.asarray(Image.open(folder/'accepted_clean.png').convert('RGB'))):
            raise AssertionError('Previously reviewed side image changed')
        Image.fromarray(clean).save(folder/'clean.png')
        shutil.copy2(maskdir/'soft_alpha.png',folder/'edit_alpha.png')
        Image.fromarray(locked.astype(np.uint8)*255).save(folder/'protected_mask.png')
        # Crop all panels identically to the original geometry-mask vehicle bbox.
        geom_path=Path(generated['prepared_root'])/name/'mask_geometry.png'
        ys,xs=np.where(np.asarray(Image.open(geom_path))>127)
        box=(max(0,int(xs.min())-35),max(0,int(ys.min())-35),
             min(src.shape[1],int(xs.max())+36),min(src.shape[0],int(ys.max())+36))
        width=720;height=round((box[3]-box[1])*width/(box[2]-box[0]))
        def pair(second,second_label):
            canvas=Image.new('RGB',(width*2,height+44),(245,245,245));draw=ImageDraw.Draw(canvas)
            for i,(pixels,label) in enumerate([(src,'原图'),(second,second_label)]):
                canvas.paste(Image.fromarray(pixels).crop(box).resize((width,height),Image.Resampling.LANCZOS),(i*width,44))
                draw.text((i*width+12,8),v['label']+' · '+label,font=font,fill=(25,25,25))
            return canvas
        panel=pair(clean,label);panel.save(folder/'comparison.png');pairs.append(panel)
        raw_panel=pair(raw,'Qwen 原始输出');raw_panel.save(folder/'raw_comparison.png');raw_pairs.append(raw_panel)
        record['views'].append(dict(id=name,source_sha256=sha(folder/'source.png'),raw_sha256=sha(folder/'raw.png'),
            clean_sha256=sha(folder/'clean.png'),mask_sha256=sha(maskdir/'soft_alpha.png'),
            model_alpha_review=alpha_review,locked_pixels_exact=int(locked.sum()),
            changed_pixels_outside_support=0,edit_support_pixels=int((~locked).sum()),
            mean_absolute_rgb_change_in_support=float(np.abs(clean.astype(float)-src)[~locked].mean()),
            reused_accepted_side_exact=(True if name=='side_004' else None),comparison_crop_xyxy=box))
    for filename,panels in [('comparison_all.png',pairs),('raw_comparison_all.png',raw_pairs)]:
        sheet=Image.new('RGB',(1440,sum(x.height for x in panels)),(245,245,245));y=0
        for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
        sheet.save(out/filename)
    record['status']='complete_pending_visual_review'
    (out/'compositing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),views=len(record['views']),side_reuse_exact=True),indent=2))


if __name__=='__main__':main()
