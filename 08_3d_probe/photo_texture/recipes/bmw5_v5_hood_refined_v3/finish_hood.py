"""Package the hood edit and compare the embedded texture with v2 at the fitted cameras."""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

OUT=Path(__file__).resolve().parent
PHOTO=OUT.parents[1]
BASE=OUT.parent/'bmw5_mv_textured_v5'
PREVIOUS=OUT.parent/'bmw5_v5_seam_refined_v2'
sys.path[:0]=[str(PHOTO),str(PHOTO.parents[1]/'09_reflection_cleanup')]
from bake_texture import sample
from raster import render
from replace_frozen_v5_texture import patch_glb, parse_glb, sha256, accessor_view_hashes
from camera_seam_measure import evaluate


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def build_panel(left, right, filename):
    height,width=left.shape[:2]
    result=Image.new('RGB',(width*2+6,height+46),'#f3f5f7')
    draw=ImageDraw.Draw(result)
    font=ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',24)
    for j,(frame,label) in enumerate([(left,'上一版 v2'),(right,'机盖修正版 v3')]):
        x=j*(width+6)
        draw.text((14+x,7),label,fill='#263039',font=font)
        result.paste(Image.fromarray(frame),(x,46))
    result.save(filename)


def project_texture(texture, texcoords, hit, background):
    frame=np.empty((*hit.shape,3),np.uint8)
    frame[:]=background
    frame[hit]=sample(texture,texcoords[:,0]*(texture.shape[1]-1),(1-texcoords[:,1])*(texture.shape[0]-1))
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--fender',type=Path,default=OUT/'fender_patch.npz')
    parser.add_argument('--hood',type=Path,default=OUT/'hood_gloss_texture.png')
    args=parser.parse_args()
    dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    previous=rgb(PREVIOUS/'texture.png')
    updated=rgb(args.hood)
    valid=np.asarray(Image.open(BASE/'texture_source_ids.png'))!=255
    footprint=np.asarray(Image.open(OUT/'hood_footprint.png')).copy()
    if args.fender:
        with np.load(args.fender) as data:
            alpha=data['alpha']
            colors=data['rgb']
        assert alpha.dtype==np.uint8 and alpha.shape==valid.shape and colors.shape==previous.shape
        active=(alpha>0)&valid
        w=alpha[active,None].astype('float32')/255
        updated[active]=np.rint(updated[active]*(1-w)+colors[active]*w).astype('uint8')
        np.maximum(footprint,alpha,out=footprint)
    # UV chart padding can sample the foreground under bilinear and mip filtering.
    dist,nearest=distance_transform_edt(~valid,return_indices=True)
    pad=(dist<=12)&~valid
    updated[pad]=updated[nearest[0][pad],nearest[1][pad]]
    delta=np.abs(updated.astype(np.int16)-previous.astype(np.int16))
    changed=(delta.max(2)>0)&valid
    retained=valid&(footprint==0)
    assert np.max(delta[retained])<=1
    mesh=np.load(BASE/'textured_mesh.npz')
    verts,faces,uv=[mesh[k] for k in ('vertices','faces','uv')]
    h,w=valid.shape
    atlas=np.column_stack([uv[:,0]*(w-1),(1-uv[:,1])*(h-1),np.ones(len(uv))])
    _,fid,bary=render(atlas,faces,{'projection':'orthographic','focal_px':1,'cx':0,'cy':0},valid.shape)
    assert np.array_equal(valid,fid>=0)
    positions=np.einsum('ij,ijk->ik',bary[changed],verts[faces[fid[changed]]])
    assert positions[:,2].min()>.25
    embedded=io.BytesIO()
    Image.fromarray(updated).save(embedded,format='PNG')
    # Using the original GLB avoids adding the previous model's obsolete PNG again.
    glb,checks=patch_glb((BASE/'textured_mesh.glb').read_bytes(),embedded.getvalue())
    new_tree,new_bin=parse_glb(glb)
    old_tree,old_bin=parse_glb((PREVIOUS/'textured_mesh.glb').read_bytes())
    assert accessor_view_hashes(new_tree,new_bin)==accessor_view_hashes(old_tree,old_bin)
    for key in ['meshes','accessors','textures','materials','samplers','nodes','scenes']:
        assert new_tree.get(key)==old_tree.get(key)
    image_spec=new_tree['bufferViews'][new_tree['images'][0]['bufferView']]
    start=image_spec['byteOffset']
    embedded_tex=np.array(Image.open(io.BytesIO(new_bin[start:start+image_spec['byteLength']])))
    assert np.array_equal(updated,embedded_tex)
    (dest/'textured_mesh.glb').write_bytes(glb)
    (dest/'texture.png').write_bytes(embedded.getvalue())
    views=json.loads((OUT/'camera_views.json').read_text())
    results={}
    for view in views:
        key=view['key']
        size=(view['height'],view['width'])
        _,fid,bary=render(verts,faces,view['camera'],size)
        hit=fid>=0
        coords=np.einsum('ij,ijk->ik',bary[hit],uv[faces[fid[hit]]])
        reference=rgb(OUT/f'camera_{key}_v2.png')
        background=reference[~hit][0]
        v2=project_texture(previous,coords,hit,background)
        v3=project_texture(embedded_tex,coords,hit,background)
        repro_diff=np.abs(reference.astype(float)-v2.astype(float)).mean()
        assert repro_diff<1.
        assert np.array_equal(v2[~hit],v3[~hit])
        image_path=dest/f'{key}_v3.png'
        Image.fromarray(v3).save(image_path)
        results[key]={'baseline_replay_mean_absolute_rgb_error':round(float(repro_diff),3),
                      'changed_pixels':int(np.any(v2!=v3,axis=2).sum())}
        if key in ['screenshot','bonnet']:
            build_panel(reference,v3,dest/f'{key}_comparison.png')
        if key=='screenshot':
            features={}
            for label,(x0,y0,x1,y1) in {'inner_headlight':(258,438,349,499),'grille':(445,494,624,561)}.items():
                before=v2[y0:y1,x0:x1]
                after=v3[y0:y1,x0:x1]
                features[label]={'box':[x0,y0,x1,y1],'changed_pixels':int(np.any(before!=after,axis=2).sum())}
                assert np.array_equal(before,after)
            results[key]['protected_crops']=features
            results[key]['features_before']=evaluate(OUT/f'camera_{key}_v2.png')
            results[key]['features_after']=evaluate(image_path)
    report={'base_sha256':sha256(PREVIOUS/'textured_mesh.glb'),'glb_sha256':sha256(dest/'textured_mesh.glb'),
            'hood_texture_name':args.hood.name,'hood_texture_sha256':sha256(args.hood),
            'fender_patch_sha256':sha256(args.fender) if args.fender else None,
            'same_mesh_uv_normals_material_as_v2':True,'changed_valid_texels':int(changed.sum()),
            'changed_world_min_xyz':positions.min(0).tolist(),'changed_world_max_xyz':positions.max(0).tolist(),
            'comparisons':results,'embedding_verified':checks['embedded_png_exact']}
    (dest/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['comparisons']['screenshot'],indent=2))

if __name__=='__main__':
    main()
