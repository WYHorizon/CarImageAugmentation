"""Render orthographic surface charts for the BMW5 front and near-side glass."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
PHOTO = HERE.parents[1]
PROJECT = PHOTO.parents[1]
OUT = PHOTO / 'outputs'
BASE = OUT / 'bmw5_mv_textured_v5'
V3 = OUT / 'bmw5_v5_hood_refined_v3'
DONORS = PROJECT / '09_reflection_cleanup/outputs'
sys.path[:0] = [str(PHOTO), str(PROJECT / '09_reflection_cleanup')]
from raster import render
from bake_texture import sample, project
from replace_frozen_v5_texture import render_texture, patch_glb, sha256

def rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def chart(az, el, scale, cx, cy):
    az, el = np.radians([az, el])
    direction = np.array([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)])
    right = np.array([np.cos(az), 0, -np.sin(az)])
    down = np.cross(-direction, right)
    return dict(projection='orthographic', R=np.stack((right, down, -direction)).tolist(),
                t=[0,0,3], focal_px=scale, cx=cx, cy=cy)

CHARTS = dict(front=chart(0,48,1100,430,195), side=chart(90,2,800,430,340))
SHAPES = dict(front=(540,860), side=(420,1000))

def diagnostics(v,f,uv,tex):
    for name,cam in CHARTS.items():
        h,w=SHAPES[name]
        im = np.array(render_texture(render,v,f,uv,tex,cam,w,h))
        Image.fromarray(im).save('/tmp/glassmap_'+name+'.png')
        for axis, length in enumerate([w,h]):
            for at in range(0,length,50):
                org=(at,16) if axis==0 else (3,at)
                if axis==0: cv2.line(im,(at,0),(at,h-1),(70,100,115),1)
                else: cv2.line(im,(0,at),(w-1,at),(70,100,115),1)
                cv2.putText(im,str(at),org,cv2.FONT_HERSHEY_SIMPLEX,.38,(200,240,45),1)
        Image.fromarray(im).save('/tmp/glassgrid_'+name+'.png')

# Outlines traced from v3 itself. The fitted source cameras are close to the
# car silhouette, but do not agree on the windshield border between views.
FRONT = [(178,172),(184,150),(196,133),(213,122),(244,124),(291,131),
         (352,137),(431,139),(499,137),(556,132),(608,125),(647,119),
         (664,129),(677,145),(685,166),(696,215),(705,283),(716,355),
         (723,410),(718,426),(705,434),(681,437),(645,430),(609,424),
         (575,424),(550,431),(512,441),(479,446),(425,448),(375,447),
         (316,441),(257,434),(205,428),(171,427),(153,433),(143,429),
         (138,418),(140,390),(149,333),(157,279),(167,222)]
SIDE = [
    [(254,211),(272,199),(308,177),(344,161),(380,152),(417,147),
     (455,144),(499,141),(504,141),(466,248),(278,258),(276,246),
     (274,230),(268,219),(256,215)],
    [(545,141),(588,139),(641,141),(689,143),(731,147),(764,152),
     (798,159),(833,168),(861,179),(865,191),(871,211),(869,223),
     (865,231),(833,235),(792,239),(750,241),(701,244),(652,245),
     (586,247),(532,248)]
]
FRONT_DONOR = DONORS/'bmw5_front_glass_qwen_v2/clean.png'
FRONT_CLOSEUP = DONORS/'bmw5_glass_pilot_qwen_v1/front_002_closeup/raw.png'
SIDE_DONOR = DONORS/'bmw5_glass_pilot_qwen_v1/side_004/raw.png'
SIDE_CLOSEUP = DONORS/'bmw5_glass_pilot_qwen_v1/side_004_closeup/raw.png'

def outline(shape,paths,feather):
    mask=np.zeros(shape,np.uint8)
    for path in paths: cv2.fillPoly(mask,[np.asarray(path,np.int32)],1)
    dist=distance_transform_edt(mask)
    t=np.clip(dist/feather,0,1)
    return (t*t*(3-2*t)).astype(np.float32)

def front_field(path):
    image=rgb(path)
    closeup=rgb(FRONT_CLOSEUP).astype('float32')
    # Reuse the prior hand-audited protected foreground. Only the view through
    # the cabin receives the newly generated out-of-focus rear headrests.
    matte=np.array(Image.open(FRONT_DONOR.parent/'edit_alpha.png').convert('L'),dtype='float32')/255.
    # The cropped inference still has a few soft, light lines well behind the
    # rear seats. Smooth that light band without mixing in the dark headrests.
    light=np.clip((closeup.mean(axis=2)-80)/30.,0,1)*matte
    denominator=cv2.GaussianBlur(light,(0,0),5,sigmaY=11)
    blurred=cv2.GaussianBlur(closeup*light[:,:,None],(0,0),5,sigmaY=11)
    blurred/=np.maximum(denominator[:,:,None],.001)
    closeup+=light[:,:,None]*(blurred-closeup)
    image=np.rint(image+(closeup-image)*matte[:,:,None]).astype('uint8')
    yy,xx=np.indices(SHAPES['front'],dtype=np.float32)
    # Reparameterize a single coherent donor sheet over the traced windshield.
    # Its top/bottom contours curve differently in the head-on photo and chart.
    s=(xx-430)/290
    for _ in range(4):
        a=np.clip(abs(s),0,1)
        top=np.interp(a,[0,.4,.76,1],[139,135,123,147])
        bottom=np.interp(a,[0,.35,.64,.84,1],[448,441,424,432,427])
        t=np.clip((yy-top)/(bottom-top),0,1)
        half=np.interp(t,[0,.18,.6,1],[247,256,275,293])
        s=(xx-430)/half
    a=np.clip(abs(s),0,1)
    # Keep sampled colors just inside the rubber frame on the donor.
    left_width=np.interp(t,[0,.25,.73,1],[221,240,264,274])
    right_width=np.interp(t,[0,.25,.73,1],[218,235,260,267])
    u=770+s*np.where(s<0,left_width,right_width)
    donor_top=np.interp(a,[0,.5,.84,1],[249,250,253,261])
    donor_bottom=np.interp(a,[0,.65,.91,1],[373,373,373,368])
    v=donor_top+t*(donor_bottom-donor_top)
    return cv2.remap(image,u.astype('f'),v.astype('f'),cv2.INTER_LINEAR)

def side_field():
    raw=rgb(SIDE_DONOR).astype('float32')
    # Limit normalized convolution to the dark model-painted pane, excluding
    # bright frame pixels even close to the edges. This also removes the last
    # few low-contrast horizontal marks left by the model.
    weight=np.clip((86-raw.mean(axis=2))/26,0,1)
    den=cv2.GaussianBlur(weight,(0,0),3, sigmaY=9)
    sums=cv2.GaussianBlur(raw*weight[:,:,None],(0,0),3,sigmaY=9)
    col=np.clip(sums/np.maximum(den[:,:,None],.001),15,75)
    detailed=rgb(SIDE_CLOSEUP).astype('float32')
    # Pane margins on the higher-resolution closeup. Extend the nearest glass
    # colors for a few edge pixels instead of importing its redrawn chrome.
    polygons=[
        [(292,266),(348,230),(423,197),(482,173),(575,153),(640,145),
         (703,136),(747,132),(680,325),(367,345),(332,345),(335,319),
         (330,293),(323,278),(308,272)],
        [(828,132),(904,128),(1015,128),(1117,134),(1196,146),(1284,164),
         (1369,190),(1384,199),(1400,247),(1402,262),(1386,286),
         (1310,296),(1197,306),(1090,313),(969,319),(814,324)]]
    pane=np.zeros(detailed.shape[:2],np.uint8)
    for points in polygons:
        p=np.rint(np.array(points)*.5+[460,260]).astype('int32')
        cv2.fillPoly(pane,[p],1)
    _,idx=distance_transform_edt(pane==0,return_indices=True)
    detailed=detailed[idx[0],idx[1]]
    # The first edit contributes a small neutral glass tint; visible seating
    # and smooth background come from the fresh closeup candidate.
    return np.rint(.85*detailed+.15*col).astype('uint8')

def apply_windows(v,f,uv,base):
    h,w=base.shape[:2]
    uvscreen=np.column_stack((uv[:,0]*(w-1),(1-uv[:,1])*(h-1),np.ones(len(uv))))
    _,ids,bary=render(uvscreen,f,dict(projection='orthographic',focal_px=1,cx=0,cy=0),(h,w))
    valid=ids>=0
    hood=np.array(Image.open(V3/'hood_footprint.png').convert('L'))>0
    earlier=rgb(OUT/'bmw5_v5_seam_refined_v2/texture.png')
    hood_lock=hood|np.any(base!=earlier,axis=2)
    active=np.flatnonzero(valid)
    xyz=np.einsum('ij,ijk->ik',bary[valid],v[f[ids[valid]]])
    del bary
    result=base.copy()
    tex=result.reshape(-1,3)
    allowed=np.zeros((h,w),np.uint8)
    camera_file=OUT/'bmw5_mv_proportion_fit_v2/cameras.json'
    side_cam=next(rec['camera'] for rec in json.loads(camera_file.read_text())['views'] if rec['id']=='side_004')
    report={}
    donors={'front':front_field(FRONT_DONOR),'side':side_field()}
    for index,(name,cam) in enumerate(CHARTS.items(),1):
        coords=SHAPES[name]
        paths=[FRONT] if name=='front' else SIDE
        mask=outline(coords,paths,4 if name=='front' else 2)
        x,y,depth,_=project(xyz,cam)
        covered=(x>=0)&(x<coords[1]-1)&(y>=0)&(y<coords[0]-1)
        shortlist=np.flatnonzero(covered)
        opacity=sample(mask,x[covered],y[covered])
        shortlist=shortlist[opacity>0]
        opacity=opacity[opacity>0]
        world_depth,_,_=render(v,f,cam,coords,False)
        front_depth=sample(np.nan_to_num(world_depth,posinf=5.),x[shortlist],y[shortlist])
        visible=np.abs(depth[shortlist]-front_depth)<.009
        visible &= ~hood_lock.flat[active[shortlist]]
        shortlist,opacity=shortlist[visible],opacity[visible]
        if name=='front': u,vv=x[shortlist],y[shortlist]
        else: u,vv,_,_=project(xyz[shortlist],side_cam)
        color=sample(donors[name],u,vv)
        flat=active[shortlist]
        assert not np.any(allowed.flat[flat])
        allowed.flat[flat]=index
        orig=tex[flat].astype('float32')
        tex[flat]=np.rint(orig+(color.astype('float32')-orig)*opacity[:,None]).astype('uint8')
        report[name]={'valid_window_texels':len(flat),'changed_texels':int(np.any(tex[flat]!=orig,axis=1).sum())}
    dist,near=distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(dist<=12)&(allowed[near[0],near[1]]!=0)&(~hood_lock)
    allowed[pad]=3
    result[pad]=result[near[0][pad],near[1][pad]]
    assert np.array_equal(base[allowed==0],result[allowed==0])
    assert np.array_equal(base[hood_lock],result[hood_lock])
    return result,allowed,report

def main():
    m=np.load(BASE/'textured_mesh.npz')
    v,f,uv=(m[k] for k in ('vertices','faces','uv'))
    base=rgb(V3/'texture.png')
    if '--charts' in sys.argv:
        diagnostics(v,f,uv,base)
        return
    texture,window_ids,report=apply_windows(v,f,uv,base)
    Image.fromarray(texture).save(HERE/'texture.png')
    Image.fromarray(window_ids).save(HERE/'window_ids.png')
    png=(HERE/'texture.png').read_bytes()
    glb,verification=patch_glb((V3/'textured_mesh.glb').read_bytes(),png)
    assert verification['embedded_png_exact'] and verification['geometry_normals_uv_scene_material_metadata_exact']
    (HERE/'candidate.tmp').write_bytes(glb)
    os.replace(HERE/'candidate.tmp',HERE/'textured_mesh.glb')
    record=dict(base_sha256=sha256(V3/'textured_mesh.glb'),candidate_sha256=sha256(HERE/'textured_mesh.glb'),
                windows=report,front_donor=str(FRONT_DONOR),front_donor_sha256=sha256(FRONT_DONOR),
                front_interior_donor=str(FRONT_CLOSEUP),front_interior_donor_sha256=sha256(FRONT_CLOSEUP),
                side_donor=str(SIDE_DONOR),side_donor_sha256=sha256(SIDE_DONOR),
                side_interior_donor=str(SIDE_CLOSEUP),side_interior_donor_sha256=sha256(SIDE_CLOSEUP),
                exact_outside_window_and_padding=bool(np.array_equal(texture[window_ids==0],base[window_ids==0])),
                hood_footprint_and_all_prior_v3_refinements_pixel_exact=True,
                window_ids={'0':'locked','1':'windshield','2':'side_004 window panes','3':'adjacent UV padding'},
                geometry_uv_normals_material_exact=verification['geometry_normals_uv_scene_material_metadata_exact'])
    (HERE/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    diagnostics(v,f,uv,texture)
    print(json.dumps(report))

if __name__=='__main__':
    main()
