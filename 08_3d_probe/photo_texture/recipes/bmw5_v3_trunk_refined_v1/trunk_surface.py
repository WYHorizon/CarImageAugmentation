"""Transfer a revised rear-deck chart back onto the photographed BMW mesh."""
import argparse
import copy
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
SOURCE = OUTPUTS / "bmw5_v3_photo_detail_v1"
sys.path[:0] = [str(OUTPUTS/"bmw5_v3_roof_refined_v1"),str(OUTPUTS.parent)]
from build_model import digest, parse_glb, read_accessor, rgb, view_bytes, write_glb
from bake_texture import project, sample
from raster import render

HEIGHT, WIDTH = 768, 1024
ANGLE = np.deg2rad(51)
CAMERA = dict(projection="orthographic", R=[[-1,0,0],[0,-np.cos(ANGLE),-np.sin(ANGLE)],
                      [0,-np.sin(ANGLE),np.cos(ANGLE)]], t=[0,0,3],
              focal_px=1430, cx=512, cy=-580)


def load():
    raw=(SOURCE/"textured_mesh.glb").read_bytes()
    assert digest(raw)==json.loads((SOURCE/"verification.json").read_text())["glb_sha256"]
    tree,binary=parse_glb(raw)
    prim=tree["meshes"][0]["primitives"][0]
    vertices=read_accessor(tree,binary,prim["attributes"]["POSITION"])
    mesh=np.load(OUTPUTS/"bmw5_mv_textured_v5/textured_mesh.npz")
    faces=mesh["faces"]
    uv=mesh["uv"]
    texture=rgb(view_bytes(tree,binary,tree["images"][0]["bufferView"]))
    return raw,tree,binary,vertices,faces,uv,texture


def chart_render(vertices,faces,uv,texture):
    d,face,bary=render(vertices,faces,CAMERA,(HEIGHT,WIDTH))
    hit=face>=0
    st=np.einsum("ij,ijk->ik",bary[hit],uv[faces[face[hit]]])
    image=np.full((HEIGHT,WIDTH,3),28,dtype=np.uint8)
    h,w=texture.shape[:2]
    image[hit]=sample(texture,st[:,0]*(w-1),(1-st[:,1])*(h-1))
    return image,d,face


def chart_support():
    # The rear glazing and the two long side shut lines are independently
    # visible. The lower curve is the top of the retained narrow trunk lip.
    left=[(182,182),(184,211),(184,234),(185,253),(188,288),
          (189,313),(192,340),(195,365),(197,370)]
    lower=[(216,371),(239,375),(260,378),(283,381),(312,385),(337,387),
           (371,391),(406,394),(440,397),(475,398),(504,402),(522,400),
           (555,398),(595,396),(635,394),(665,392),(701,388),(734,383),
           (765,379),(790,375),(805,372)]
    right=[(808,350),(810,323),(815,293),(818,265),(821,239),
           (823,220),(827,194),(827,182)]
    upper=[(815,195),(787,200),(736,205),(660,209),(599,212),(531,214),
           (494,215),(454,214),(400,214),(352,211),(299,208),(248,205),
           (215,201),(193,196)]
    points=np.asarray(left+lower+right+upper,np.int32)
    mask=np.zeros((HEIGHT,WIDTH),np.uint8)
    cv2.fillPoly(mask,[points],1)
    # The same fitted chart is used from the source to the target. Keep the
    # thin structural outline from the photograph and feather inward on paint.
    distance=distance_transform_edt(mask)
    t=np.clip((distance-.65)/5.25,0,1)
    return (t*t*(3-2*t)).astype(np.float32)


def shoulder_support():
    # Upper rear quarter-panel paint, between the trunk shut lines and the
    # inner boundaries of the rear lamp housings. The lower tips follow their
    # curved black seals. The outer edges feather into surviving vehicle paint.
    profile = {
       "left": [
           (159,163,169),(175,157,174),(190,152,177),(210,150,179),(235,149,181),
           (255,146,182),(275,143,184),(295,142,185),(310,144,186),(322,149,187),
           (333,153,188),(345,158,190),(355,162,191),(365,170,193),(374,181,194),
           (380,189,195),(383,194,195)],
       "right": [
           (156,845,850),(173,839,861),(188,834,865),(209,829,869),(234,825,873),
           (255,822,878),(274,821,883),(297,817,887),(311,815,890),(323,814,886),
           (336,813,879),(346,812,873),(356,811,868),(365,810,861),(372,809,855),
           (379,807,847),(385,806,835),(389,805,821),(391,805,808)]
    }
    yy,xx=np.indices((HEIGHT,WIDTH),dtype=np.float32)
    def ease(v):
        t=np.clip(v,0,1)
        return t*t*(3-2*t)
    alpha=np.zeros_like(xx)
    for side,rows in profile.items():
        rows=np.array(rows)
        y=rows[:,0]
        left=np.interp(yy,y,rows[:,1]); right=np.interp(yy,y,rows[:,2])
        to_left=xx-left; to_right=right-xx
        if side=="left":
            outer,inner=to_left,to_right
        else:
            outer,inner=to_right,to_left
        border=np.interp(yy,[y[0],270,315,347,380],[6,10,5,3.1,2.7])
        weight=ease((yy-y[0])/14)*ease((y[-1]-yy)/5)
        weight*=ease((outer-.5)/border)*ease((inner-.3)/2.5)
        alpha=np.maximum(alpha,weight)
    return alpha.astype(np.float32)


def prepare():
    _,_,_,vertices,faces,uv,texture=load()
    img,_,_=chart_render(vertices,faces,uv,texture)
    Image.fromarray(img).save(HERE/"trunk_before.png")
    weight=chart_support()
    Image.fromarray(np.uint8(np.rint(weight*255))).save(HERE/"trunk_chart_mask.png")
    Image.fromarray(np.uint8(np.rint(shoulder_support()*255))).save(HERE/"trunk_shoulder_mask.png")
    (HERE/"chart_camera.json").write_text(json.dumps(dict(width=WIDTH,height=HEIGHT,camera=CAMERA),indent=2)+"\n")


def apply(edited,output):
    original,tree,binary,vertices,faces,uv,before=load()
    h,w=before.shape[:2]
    candidate=Image.open(edited).convert("RGB")
    if candidate.size!=(WIDTH,HEIGHT):
        raise ValueError(f"Expected projected donor chart {WIDTH}x{HEIGHT}, got {candidate.size}")
    candidate=np.asarray(candidate)
    # Recover the 3D point for each observed body texel, then sample its
    # unique donor color in the registered edit. Use the full original
    # atlas face list to keep other islands occupied during gutter padding.
    screen=np.c_[uv[:,0]*(w-1),(1-uv[:,1])*(h-1),np.ones(len(uv))]
    atlas_camera=dict(projection="orthographic",focal_px=1,cx=0,cy=0)
    _,face,bary=render(screen,faces,atlas_camera,(h,w))
    valid=face>=0
    ids=np.flatnonzero(valid)
    xyz=np.einsum("ij,ijk->ik",bary[valid],vertices[faces[face[valid]]])
    del bary
    back=(xyz[:,2]<-.58)&(xyz[:,1]>-.105)&(abs(xyz[:,0])<.27)
    ids,xyz=ids[back],xyz[back]
    x,y,d,_=project(xyz,CAMERA)
    alpha=sample(np.maximum(chart_support(),shoulder_support()),x,y)
    inside=alpha>0
    ids,x,y,d,xyz,alpha=[r[inside] for r in (ids,x,y,d,xyz,alpha)]
    world_depth,_,_=render(vertices,faces,CAMERA,(HEIGHT,WIDTH),False)
    front=sample(np.nan_to_num(world_depth,posinf=10),x,y)
    windows=np.asarray(Image.open(OUTPUTS/"bmw5_v3_opaque_glass_v1/window_ids.png"))!=0
    # Behind the lid there are vertical badge/plate faces sharing the same
    # camera rays. Check against a chart z-buffer with a sub-texel tolerance.
    visible=(abs(d-front)<.0035)&(~windows.flat[ids])
    ids,x,y,xyz,alpha=[r[visible] for r in (ids,x,y,xyz,alpha)]
    mask=np.zeros((h,w),np.uint8)
    mask.flat[ids]=np.maximum(1,np.rint(alpha*255)).astype(np.uint8)
    after=before.copy()
    old=before.reshape(-1,3)[ids].astype(np.float32)
    new=sample(candidate,x,y).astype(np.float32)
    after.reshape(-1,3)[ids]=np.rint(old+(new-old)*alpha[:,None]).clip(0,255).astype(np.uint8)
    dist,nearest=distance_transform_edt(~valid,return_indices=True)
    gutter=(~valid)&(dist<=12)&(~windows)&(mask[tuple(nearest)]>0)
    mask[gutter]=mask[nearest[0][gutter],nearest[1][gutter]]
    after[gutter]=after[nearest[0][gutter],nearest[1][gutter]]
    changed=np.any(after!=before,axis=2)
    assert changed[valid].any() and not changed[mask==0].any()
    assert np.array_equal(before[windows],after[windows])
    stream=io.BytesIO()
    Image.fromarray(after).save(stream,format="PNG")
    blocks=[view_bytes(tree,binary,i) for i in range(len(tree["bufferViews"]))]
    image_block=tree["images"][0]["bufferView"]
    blocks[image_block]=stream.getvalue()
    updated=write_glb(copy.deepcopy(tree),blocks)
    checked,b2=parse_glb(updated)
    for key in ("meshes","nodes","scenes","accessors","materials","textures","images"):
        assert checked[key]==tree[key]
    for index,block in enumerate(blocks):
        assert view_bytes(checked,b2,index)==block
        if index!=image_block:
            assert block==view_bytes(tree,binary,index)
    assert np.array_equal(rgb(view_bytes(checked,b2,image_block)),after)
    output.mkdir(parents=True,exist_ok=True)
    tmp=output/"trunk_model.tmp"
    tmp.write_bytes(updated)
    tmp.replace(output/"textured_mesh.glb")
    Image.fromarray(mask).save(output/"trunk_texture_mask.png")
    preview,_,_=chart_render(vertices,faces,uv,after)
    Image.fromarray(preview).save(output/"trunk_after.png")
    report={
        "base_glb_sha256":digest(original), "glb_sha256":digest(updated),
        "input_chart_sha256":digest(edited.read_bytes()),
        "body_texture_only_changed":True,"geometry_and_all_other_buffers_exact":True,
        "unmodified_glass_texels_exact":True,
        "outside_lid_shoulders_and_padding_exact":True,
        "body_shutlines_protected_separate_chart_regions":True,
        "changed_valid_texels":int((changed&valid).sum()),"changed_padding":int((changed&~valid).sum()),
        "selected_surface_xyz_bounds":[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
    }
    (output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare")
    change=sub.add_parser("apply")
    change.add_argument("edited",type=Path)
    change.add_argument("--output",type=Path,default=HERE)
    args=parser.parse_args()
    if args.command=="prepare":
        prepare()
    else:
        apply(args.edited,args.output)
