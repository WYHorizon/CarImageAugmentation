"""Protect source pixels around Qwen paint edits and assemble honest comparisons.

This applies no geometric alignment, warping, sharpening or global color change.
Both raw model outputs remain intact. All comparison crops use the same boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(images, labels, crop, width=720, height=None):
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 28)
    crop_width, crop_height = crop[2]-crop[0], crop[3]-crop[1]
    height = height or round(crop_height * width / crop_width)
    canvas = Image.new('RGB', (width * 3, height + 60), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        draw.text((i*width+16, 10), label, font=font, fill=(25,25,25))
        canvas.paste(image.crop(crop).resize((width, height), Image.Resampling.LANCZOS), (i*width, 60))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--mild-mix', type=float, default=1.)
    parser.add_argument('--strong-mix', type=float, default=1.)
    parser.add_argument('--accept-near-opaque-rgba', action='store_true',
                        help='After review, use RGB from model RGBA only when every alpha is >=249/255; keep raw RGBA intact')
    args = parser.parse_args()
    out, masks = args.output_dir, args.mask_dir
    source_path = out / 'source.png'
    source_image = Image.open(source_path).convert('RGB')
    source = np.asarray(source_image)
    if source_image.size != (1536, 1024):
        raise ValueError('Reviewed masks require a 1536x1024 source')
    mask_manifest = json.loads((masks/'manifest.json').read_text())
    mask_source = np.asarray(Image.open(mask_manifest['inputs']['source_rgb']['path']).convert('RGB'))
    if not np.array_equal(source, mask_source):
        raise ValueError('Mask source and current working photo differ; do not silently misalign them')
    alpha_image = Image.open(masks/'soft_alpha.png').convert('L')
    alpha8 = np.asarray(alpha_image)
    if alpha8.shape != source.shape[:2]:
        raise ValueError('Mask dimensions differ from source')
    alpha_image.save(out/'edit_alpha.png')
    locked = alpha8 == 0
    Image.fromarray((locked*255).astype(np.uint8)).save(out/'protected_mask.png')
    record = {'method':'original_pixel_lock_with_inward_feathered_Qwen_paint_edits',
              'mask_manifest':str((masks/'manifest.json').resolve()),
              'source_sha256':sha256(source_path), 'mask_sha256':sha256(masks/'soft_alpha.png'),
              'color_space':'8-bit sRGB interpolation, rounded to nearest; exact source copy wherever alpha=0',
              'geometric_transform':'none', 'upscaling':'none',
              'variants':{}, 'crop_boxes_xyxy':{'vehicle':[60,285,1480,755], 'doors':[450,440,1090,650]}}
    clean_images, raw_images = [source_image], [source_image]
    for name, amount in [('mild',args.mild_mix),('strong',args.strong_mix)]:
        if not 0 <= amount <= 1:
            raise ValueError('Mix must be in [0,1]')
        raw_path = out/name/'raw.png'
        raw_image = Image.open(raw_path)
        if raw_image.size != source_image.size:
            raise ValueError('Raw prediction must retain exact source coordinates; no post-warp allowed')
        alpha_review = None
        if raw_image.mode == 'RGBA':
            model_alpha = np.asarray(raw_image.getchannel('A'))
            alpha_review = {'minimum':int(model_alpha.min()), 'maximum':int(model_alpha.max()),
                            'minimum_inside_edit_support':int(model_alpha[~locked].min()),
                            'nonopaque_fraction':float((model_alpha<255).mean())}
            if model_alpha.min() != 255:
                if not args.accept_near_opaque_rgba or model_alpha.min() < 249:
                    raise ValueError('Nonopaque model prediction needs explicit review')
                alpha_review['handling'] = 'Reviewed nearly opaque RGBA; use unassociated RGB directly, retain original RGBA. No white matte compositing.'
        raw_image = raw_image.convert('RGB')
        raw = np.asarray(raw_image)
        alpha = alpha8.astype(np.float32)[...,None]/255*amount
        clean = np.rint(source.astype(np.float32)*(1-alpha)+raw.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        clean[locked] = source[locked]
        if not np.array_equal(clean[locked],source[locked]):
            raise AssertionError('Protected pixels changed')
        result = Image.fromarray(clean)
        result.save(out/name/'clean.png')
        difference = np.abs(clean.astype(np.int16)-source.astype(np.int16)).max(axis=2)
        # Absolute channel difference (no contrast normalization), not a quality score.
        Image.fromarray(difference.astype(np.uint8)).save(out/name/'difference.png')
        record['variants'][name] = {'mix':amount, 'raw_sha256':sha256(raw_path),
            'model_alpha_review':alpha_review,
            'clean_sha256':sha256(out/name/'clean.png'), 'protected_pixels_exact':True,
            'protected_pixel_count':int(locked.sum()), 'modified_pixel_count':int((difference>0).sum()),
            'changed_pixels_outside_support':int(((difference>0)&locked).sum()),
            'mean_absolute_rgb_change_in_support':float(np.abs(clean.astype(np.float32)-source)[~locked].mean())}
        clean_images.append(result);raw_images.append(raw_image)
    labels = ['原图', '轻度去反光 · 细节保护', '较强去反光 · 细节保护']
    vehicle = row(clean_images, labels, record['crop_boxes_xyxy']['vehicle'])
    doors = row(clean_images, ['原图 · 车门局部','轻度 · 车门局部','较强 · 车门局部'], record['crop_boxes_xyxy']['doors'])
    combined = Image.new('RGB',(vehicle.width,vehicle.height+doors.height+12),(245,245,245))
    combined.paste(vehicle,(0,0));combined.paste(doors,(0,vehicle.height+12))
    combined.save(out/'comparison.png')
    vehicle.save(out/'vehicle_comparison.png');doors.save(out/'door_comparison.png')
    raw_panel = row(raw_images,['原图','Qwen 原始候选 A','Qwen 原始候选 B'],record['crop_boxes_xyxy']['vehicle'])
    raw_panel.save(out/'raw_comparison.png')
    # An optional stacked layout keeps the complete vehicle large in narrow chats.
    stack_width = 1420
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', 32)
    stack = Image.new('RGB',(stack_width,3*526),(245,245,245));draw=ImageDraw.Draw(stack)
    for i,(image,label) in enumerate(zip(clean_images,labels)):
        draw.text((16,i*526+8),label,font=font,fill=(25,25,25))
        stack.paste(image.crop(record['crop_boxes_xyxy']['vehicle']),(0,i*526+56))
    stack.save(out/'comparison_stacked.png')
    (out/'compositing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
