"""Refine the windshield background boundary using the original photo colors.

Unlike v1, full background pixels stay alpha=1 up to the actual object edge.
Only the narrow mixed-pixel boundary has fractional alpha. Original interior
cores and stickers are protected; coarse annotation boxes are not hard masks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from prepare_windshield_mask import annotations, draw_regions, polygon, sha256, SIZE, SOURCE_SHA256, CROP


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-rgb', type=Path, default=root/'08_3d_probe/photo_texture/outputs/bmw5_inputs_v2/front_002/source_rgb.png')
    p.add_argument('--output-dir', type=Path, default=root/'09_reflection_cleanup/outputs/bmw5_front_glass_mask_v2')
    a = p.parse_args()
    if (a.output_dir/'manifest.json').exists():
        raise FileExistsError('Use a new output directory; existing reviewed masks are immutable')
    if sha256(a.source_rgb) != SOURCE_SHA256:
        raise ValueError('Coordinates require the exact reviewed source')
    photo = Image.open(a.source_rgb).convert('RGB')
    if photo.size != SIZE:
        raise ValueError('Expected 1536x1024')
    rgb = np.asarray(photo)
    intensity = rgb.astype(np.float32).mean(axis=2)
    edits, features = annotations()
    proposal = draw_regions(edits)
    coarse_features = draw_regions(features)
    seed = proposal & ~coarse_features
    search = distance_transform_edt(~seed) <= 10
    mirror_gap = draw_regions([polygon('confirmed_background_slit_above_right_mirror_edge',
        [(793,285),(800,283),(811,282),(811,288),(800,288)])])
    search |= mirror_gap
    # Dark interior cores are locked, but bright shutter pixels accidentally
    # enclosed by the old coarse polygons remain eligible in the boundary band.
    interior_core = (distance_transform_edt(coarse_features) >= 3) & (intensity < 130)
    interior_core[mirror_gap & (intensity > 100)] = False
    stickers = draw_regions([f for f in features if 'sticker' in f['name']])
    cover = polygon('white_interior_cover_refined_actual_outline', [
        (703,342),(713,342),(724,343),(738,345),(750,344),(762,344),
        (774,345),(785,347),(796,349),(807,353),(811,358),(808,363),
        (804,366),(805,375),(716,375),(715,363),(710,356),(707,351)])
    cover_mask = draw_regions([cover])
    # Gate only the narrow candidate edge band by measured source intensity.
    # Existing confirmed background, including dark shutter slots, is full edit.
    alpha_f = np.clip((intensity - 80) / 25, 0, 1) * search
    alpha_f[seed] = 1
    alpha_f[interior_core | stickers | cover_mask] = 0
    alpha = np.rint(alpha_f*255).astype(np.uint8)
    support = alpha > 0
    protected = ~support
    assert not np.any(support & (interior_core | stickers | cover_mask))
    out = a.output_dir
    out.mkdir(exist_ok=True, parents=True)
    arrays = {'soft_alpha.png':alpha,'hard_support.png':support.astype(np.uint8)*255,
              'protected_mask.png':protected.astype(np.uint8)*255,
              'search_region.png':search.astype(np.uint8)*255,
              'interior_core_mask.png':interior_core.astype(np.uint8)*255,
              'sticker_and_cover_protection.png':(stickers|cover_mask).astype(np.uint8)*255}
    for name, data in arrays.items():
        Image.fromarray(data).save(out/name)
    blend = alpha[...,None].astype(np.float32)/255*.56
    overlay = Image.fromarray(np.rint(rgb*(1-blend)+np.array([30,230,145])*blend).astype(np.uint8))
    overlay.save(out/'overlay.png')
    overlay.crop(CROP).resize((1770,435),Image.Resampling.NEAREST).save(out/'overlay_windshield_crop.png')
    photo.crop(CROP).resize((1770,435),Image.Resampling.NEAREST).save(out/'source_windshield_crop.png')
    detail = rgb.astype(np.float32).copy()
    hard = interior_core | stickers | cover_mask
    detail[hard] = detail[hard]*.55+np.array([230,70,170])*.45
    detail[support] = detail[support]*.5+np.array([30,230,145])*.5
    Image.fromarray(np.rint(detail).astype(np.uint8)).crop(CROP).resize(
        (1770,435),Image.Resampling.NEAREST).save(out/'detail_protection_overlay.png')
    samples = {'yellow_sticker':(590,290),'oval_sticker':(570,267),'mirror':(770,293),
               'left_headrest':(630,320),'right_headrest':(920,320),'steering_wheel':(950,363),
               'white_cover':(755,357),'dashboard':(770,374),
               'background_left_hole':(692,289),'background_center_hole':(781,316),
               'background_right_hole':(832,316)}
    checks = {k:{'xy':list(v),'alpha':int(alpha[v[1],v[0]])} for k,v in samples.items()}
    assert all(v['alpha']==0 for k,v in checks.items() if not k.startswith('background'))
    assert all(v['alpha']==255 for k,v in checks.items() if k.startswith('background'))
    ys,xs=np.where(support)
    record = {'status':'complete_pending_composite_visual_review','view_id':'front_002','size':list(SIZE),
              'method':'source_color_guided_background_edges_no_inward_feather',
              'inputs':{'source_rgb':{'path':str(a.source_rgb.resolve()),'sha256':SOURCE_SHA256}},
              'parameters':{'search_band_px':10,'interior_core_inset_px':3,'mixed_boundary_mean_rgb_range':[80,105],
                            'inward_background_feather_px':0},
              'annotation_role':'Coarse regions seed background and identify dark interior cores; they do not hard-exclude bright boundary background.',
              'refined_cover_annotation':cover,'sample_points':checks,
              'support_pixels':int(support.sum()),'full_alpha_pixels':int((alpha==255).sum()),
              'fractional_alpha_pixels':int(((alpha>0)&(alpha<255)).sum()),
              'support_bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
              'contract':{'soft_alpha':'uint8, full replacement on background up to real edge; fractional only on mixed edge pixels',
                          'hard_support':'255 iff soft_alpha>0','protected_mask':'255 iff soft_alpha==0',
                          'composite':'round(source*(1-alpha/255)+rawRGB*alpha/255); exact source where alpha==0'},
              'limitations':['Photo-specific intensity thresholds; not reusable on other views without review.',
                             'Mixed anti-aliased interior/background edge pixels may change; solid interior cores are exact.',
                             'Uncertain narrow bright slit beneath sticker is still preserved.',
                             'No model output, original photo, old mask or 3D file is changed.'],
              'outputs':{n:{'path':str((out/n).resolve()),'sha256':sha256(out/n)} for n in [*arrays,'overlay.png',
                             'overlay_windshield_crop.png','source_windshield_crop.png','detail_protection_overlay.png']}}
    (out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['support_pixels','full_alpha_pixels','fractional_alpha_pixels','support_bbox_xyxy']}))


if __name__=='__main__':
    main()
