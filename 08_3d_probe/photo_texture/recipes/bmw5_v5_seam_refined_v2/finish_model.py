"""Assemble the frozen v5 geometry with reviewed paint and optional lamp corrections."""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
PHOTO = HERE.parents[1]
PROJECT = PHOTO.parents[1]
BASE = PHOTO / 'outputs/bmw5_mv_textured_v5'
REGIONS = PHOTO / 'outputs/material_regions_v1'
STRONG = PROJECT / '09_reflection_cleanup/outputs/bmw5_v5_qwen_strong_texture_v1'
sys.path[:0] = [str(PHOTO), str(PROJECT/'09_reflection_cleanup')]
from raster import render
from replace_frozen_v5_texture import patch_glb, parse_glb, render_texture, sha256


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def smoothstep(x, low, high):
    t = np.clip((x-low)/(high-low), 0., 1.)
    return t*t*(3.-2.*t)


def bonnet_blend(paint, vertices, faces, uv, labels):
    # The locked strong-Qwen texture has the same atlas, cameras, and sawtooth trim
    # as v5. Restrict its de-reflection to the bonnet, with soft surface-space limits.
    assert sha256(STRONG/'textured_mesh.npz') == sha256(BASE/'textured_mesh.npz')
    strong = read_rgb(STRONG/'texture.png')
    height, width = labels.shape
    screen = np.column_stack((uv[:,0]*(width-1), (1.-uv[:,1])*(height-1), np.ones(len(uv))))
    _, face_ids, bary = render(screen, faces, dict(projection='orthographic',focal_px=1,cx=0,cy=0), labels.shape)
    mask = (face_ids>=0) & (labels==1)
    tri = vertices[faces[face_ids[mask]]]
    xyz = np.einsum('ij,ijk->ik', bary[mask], tri)
    nx = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    up = nx[:,1]/np.maximum(np.linalg.norm(nx,axis=1),1e-9)
    x,y,z=xyz.T
    weight = (.88*smoothstep(z,.32,.51)*(1.-smoothstep(z,.92,.985)) *
              (1.-smoothstep(np.abs(x),.262,.325)) * smoothstep(up,.38,.79) *
              smoothstep(y,-.01,.032))
    subset = paint[mask].astype(np.float32)
    paint[mask] = np.rint(subset+(strong[mask]-subset)*weight[:,None]).astype(np.uint8)
    return int(np.count_nonzero(weight>.02))


def source_seam_score(texture, source_ids, safe):
    luma=texture.astype('float32')@np.array([.2126,.7152,.0722],np.float32)
    diffs=[]
    for axis in (0,1):
        active=safe & (source_ids<8)
        both = (np.take(active,range(1,active.shape[axis]),axis=axis) &
                np.take(active,range(active.shape[axis]-1),axis=axis))
        cross=both & (np.diff(source_ids,axis=axis)!=0)
        diffs.append(np.abs(np.diff(luma,axis=axis))[cross])
    values=np.concatenate(diffs)
    return {'count':int(values.size),'median_8bit_luma_jump':round(float(np.median(values)),2)}



def rear_patch_report(renders):
    """Fixed visible patches straddling the rear paint boundary in the saved turntable."""
    previous=PHOTO/'outputs/bmw5_v5_paint_seams_trial_v1'
    rois={150:((711,454,728,478),(749,454,766,478)),
          210:((430,455,448,477),(471,455,489,477))}
    report={}
    for angle,boxes in rois.items():
        report[angle]={'xyxy_boxes':boxes}
        for name,folder in [('original',BASE/'renders'),('previous',previous),('refined',renders)]:
            with Image.open(folder/f'az_{angle:03d}.png') as im:
                means=[float(np.mean(np.array(im.crop(box))@np.array([.2126,.7152,.0722]))) for box in boxes]
            report[angle][name]={'means':[round(v,2) for v in means],'absolute_difference':round(abs(means[1]-means[0]),2)}
    return report

def compare(board_path, renders):
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',25)
    small=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',20)
    canvas=Image.new('RGB',(1188,940),'#f7f8fa')
    draw=ImageDraw.Draw(canvas)
    rows=[('左后方 150°',150,(269,285,1035,637), (562,258)),
          ('后翼子板与尾灯',150,(645,380,843,530),(342,259)),
          ('前机盖 30°',30,(212,382,575,484),(557,157))]
    for i,heading in enumerate(['原版 v5','优化版']):
        draw.text((18+590*i,8),heading,font=font,fill='#192229')
    y=54
    for title,angle,box,sz in rows:
        draw.text((18,y),title,font=small,fill='#424c55')
        y+=34
        for j,directory in enumerate([BASE/'renders',renders]):
            with Image.open(directory/f'az_{angle:03d}.png') as image:
                crop=image.crop(box).resize(sz,Image.Resampling.LANCZOS)
                canvas.paste(crop,(j*590+(590-sz[0])//2,y))
        y+=sz[1]+17
    canvas.crop((0,0,canvas.width,min(canvas.height,y))).save(board_path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--paint',type=Path,default=HERE/'paint_texture.png')
    ap.add_argument('--output',type=Path,default=HERE)
    ap.add_argument('--lamp',type=Path)
    args=ap.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    original=read_rgb(BASE/'texture.png')
    texture=read_rgb(args.paint)
    base_hash=sha256(BASE/'textured_mesh.glb')
    assert base_hash==json.loads((BASE/'manifest.json').read_text())['mesh']['glb_sha256']
    mesh=np.load(BASE/'textured_mesh.npz')
    v,f,uv=[mesh[k] for k in ('vertices','faces','uv')]
    src=np.asarray(Image.open(BASE/'texture_source_ids.png'))
    valid=src!=255
    labels=np.asarray(Image.open(REGIONS/'inferred_material_ids.png'))
    safe=np.asarray(Image.open(REGIONS/'paint_safe_mask.png'))>0
    assert np.array_equal(texture[valid&(labels!=1)],original[valid&(labels!=1)])
    bonnet_count=bonnet_blend(texture,v,f,uv,labels)
    lamp=np.zeros(valid.shape,np.uint8)
    if args.lamp:
        patch=np.load(args.lamp)
        lamp=patch['alpha']
        col=patch['rgb']
        assert col.shape==original.shape and lamp.shape==valid.shape and lamp.dtype==np.uint8
        mask=valid&(lamp>0)
        alpha=lamp[mask].astype(np.float32)[:,None]/255.
        texture[mask]=np.rint(texture[mask]*(1-alpha)+col[mask]*alpha).astype(np.uint8)
    locked=valid & (labels!=1) & (lamp==0)
    assert np.array_equal(texture[locked],original[locked])
    dist,near=distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(dist<=12)
    texture[pad]=texture[near[0][pad],near[1][pad]]
    Image.fromarray(texture).save(out/'texture.png')
    tex_bytes=(out/'texture.png').read_bytes()
    glb,checks=patch_glb((BASE/'textured_mesh.glb').read_bytes(),tex_bytes)
    assert checks['embedded_png_exact'] and checks['geometry_normals_uv_scene_material_metadata_exact']
    (out/'textured_mesh.glb').write_bytes(glb)
    tree,binary=parse_glb((out/'textured_mesh.glb').read_bytes())
    image=tree['bufferViews'][tree['images'][0]['bufferView']]
    start=image['byteOffset']
    embedded=np.array(Image.open(io.BytesIO(binary[start:start+image['byteLength']])))
    np.testing.assert_array_equal(embedded,texture)
    camset=json.loads((BASE/'render_cameras.json').read_text())
    renders=out/'renders'
    renders.mkdir(exist_ok=True)
    same_bg=True
    for view in camset:
        angle=view['azimuth']
        cam=view['camera']
        h,w=view['height'],view['width']
        im=render_texture(render,v,f,uv,embedded,cam,w,h)
        im.save(renders/f'az_{angle:03d}.png')
        ref=read_rgb(BASE/f'renders/az_{angle:03d}.png')
        bg=(235,237,240)
        outside=(render(v,f,cam,(h,w),return_barycentric=False)[1]<0)
        same_bg &= bool(np.all(np.array(im)[outside]==bg) and np.all(ref[outside]==bg))
    assert same_bg
    compare(out/'comparison.png',renders)
    info=dict(base_glb_sha256=base_hash,glb_sha256=sha256(out/'textured_mesh.glb'),
              mesh_UV_normals_material_exact=True,original_background_exact_all_12=True,
              locked_nonpaint_exact_outside_lamp_texels=int(locked.sum()),changed_valid_texels=int(np.any(original[valid]!=texture[valid],axis=1).sum()),
              safe_paint_cross_source_v5=source_seam_score(original,src,safe), safe_paint_cross_source_refined=source_seam_score(texture,src,safe),
              fixed_rear_patches_8bit_luma=rear_patch_report(renders),
              bonnet_eligible_texels=bonnet_count,bonnet_donor_texture_sha256=sha256(STRONG/'texture.png'),
              lamp_eligible_texels=int(((lamp>0)&valid).sum()),paint_texture_sha256=sha256(args.paint),
              sample_angles=[view['azimuth'] for view in camset])
    if args.lamp:
        info['lamp_patch_sha256']=sha256(args.lamp)
    (out/'verification.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    main()
