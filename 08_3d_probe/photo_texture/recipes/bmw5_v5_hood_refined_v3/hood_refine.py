"""Map one locally cleaned front photograph continuously over the BMW5 bonnet.

The footprint is traced in canonical XZ space because the old source-projected
material labels incorrectly classify several paint patches as glass and trim.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

OUT=Path(__file__).resolve().parent
PHOTO=OUT.parents[1]
PROJ=PHOTO.parents[1]
BASE=PHOTO/'outputs/bmw5_mv_textured_v5'
V2=PHOTO/'outputs/bmw5_v5_seam_refined_v2'
DONOR=PROJ/'09_reflection_cleanup/outputs/bmw5_multiview_qwen_strong_v1/front_002/clean.png'
sys.path.insert(0,str(PHOTO))
from bake_texture import sample, sha256
from raster import render

def rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def smooth(t,lo,hi):
    a=np.clip((t-lo)/(hi-lo),0,1)
    return a*a*(3-2*a)

def donor_field():
    image=rgb(DONOR)
    # Reparameterize the visible paint strip between the windshield and grille.
    # Each side widens towards the front. Avoid the badge, glass and chrome.
    t,s=np.mgrid[0:1:384j,-1:1:512j]
    ax=abs(s)
    center=771.
    width=np.interp(t,[0,.56,1],[304,328,335])
    x=center+width*s
    upper=np.interp(ax,[0,.72,.92,1],[380.0,380.5,383,388.0])
    lower=np.interp(ax,[0,.16,.38,.65,.83,1],[465,465.5,465,477,470,457])
    y=upper+t*(lower-upper)
    values=sample(image,x.ravel(),y.ravel()).reshape(*t.shape,3).astype('float32')
    # The photo includes a badge and blind reflections close to the front edge.
    # Mask the badge before the chart-space convolution so its white logo stays
    # solely on its original texels.
    mask=~(((x-771)/34)**2+((y-471)/17)**2<1)
    den=cv2.GaussianBlur(mask.astype('float32'),(0,0),sigmaX=8,sigmaY=22)
    sums=cv2.GaussianBlur(values*mask[:,:,None],(0,0),sigmaX=8,sigmaY=22)
    field=sums/np.maximum(den[:,:,None],1e-5)
    # A wide filter in the foreshortened direction suppresses the studio slats.
    return field.astype('float32')

def main():
    cv2.setNumThreads(2)
    tex=rgb(V2/'texture.png')
    m=np.load(BASE/'textured_mesh.npz')
    vert,faces,uv=[m[k] for k in ['vertices','faces','uv']]
    height,width=tex.shape[:2]
    screen=np.column_stack((uv[:,0]*(width-1),(1-uv[:,1])*(height-1),np.ones(len(uv))))
    _,ids,bary=render(screen,faces,dict(projection='orthographic',focal_px=1,cx=0,cy=0),(height,width))
    valid=ids>=0
    valid_ids=np.flatnonzero(valid)
    xyz=np.einsum('ij,ijk->ik',bary[valid],vert[faces[ids[valid]]])
    del bary
    x,y,z=xyz.T
    # Outline in a scale-preserving canonical overhead drawing (800 px per world unit).
    # The rear contour follows the bottom of the windshield. The front contour
    # follows the headlamps and grille return, including the rounded corners.
    top=np.array([(129,490),(132,440),(130,330),(131,251),(140,221),(151,208),
                  (167,198),(179,184),(190,179),(209,182),(248,189),(295,195),
                  (360,199),(425,195),(472,189),(511,182),(530,179),(541,184),
                  (553,198),(569,208),(580,221),(589,251),(590,330),(588,440),
                  (591,490),(590,530),(584,555),(569,570),(549,582),(526,593),
                  (509,608),(478,623),(446,637),(402,647),(360,653),(318,647),
                  (274,637),(242,623),(211,608),(194,593),(171,582),(151,570),
                  (136,555),(130,530)],np.int32)
    outline=np.zeros((688,720),np.uint8)
    cv2.fillPoly(outline,[top],1)
    inside=ndimage.distance_transform_edt(outline)
    outside=ndimage.distance_transform_edt(1-outline)
    signed=(inside-outside).astype('float32')
    u_top=x*800+360
    v_top=z*800-120
    sd=sample(signed,u_top,v_top)
    weight=smooth(sd,-1,8)
    weight[(u_top<100)|(u_top>620)|(v_top<160)|(v_top>666)]=0
    # In this projection, vertical faces/underside can lie below the hood.
    # Interpolate only the upper visible surface at each (x,z) to avoid bleeding
    # through the bonnet down onto the headlamps, grille or floor.
    cam=dict(R=[[1,0,0],[0,0,1],[0,-1,0]],t=[0,0,2],projection='orthographic',focal_px=800,cx=360,cy=-120)
    depth,_,_=render(vert,faces,cam,(688,720),False)
    zsurf=sample(np.nan_to_num(depth,posinf=3),u_top,v_top)
    weight*=1-smooth(np.abs((2-y)-zsurf),.0015,.012)
    # The BMW emblem is physically attached at the nose; its border and nearby
    # grille paint are kept. Geometry coordinates are stable across source IDs.
    emblem=((x/.027)**2+((z-.957)/.024)**2)**.5
    weight*=smooth(emblem,.9,1.36)
    # Feather broadly onto the original front lip, keeping bumper and headlights.
    ax=np.clip(np.abs(x)/.287,0,1)
    front=np.interp(ax,[0,.2,.42,.66,.84,1],[.957,.951,.935,.903,.866,.816])
    weight*=1-smooth(z-front,-.023,.002)
    # Use one continuous photo chart on all top-facing bonnet points, including
    # previously unobserved paint and pieces erroneously labelled as protected.
    rear=np.interp(ax,[0,.43,.75,.91,1],[.400,.386,.374,.394,.46])
    t=np.clip((z-rear)/(front-rear),0,1)
    width_world=np.interp(t,[0,.15,.55,.86,1],[.25,.277,.289,.282,.279])
    s=np.clip(x/width_world,-1,1)
    field=donor_field()
    candidate=sample(field,(s+1)*.5*(field.shape[1]-1),t*(field.shape[0]-1))
    # Recover broad pressed-metal creases from canonical surface relief. Work in
    # a metric overhead grid so UV seams and face-normal noise cannot imprint.
    solid=np.isfinite(depth)
    surface=np.where(solid,2-depth,0).astype('float32')
    support=cv2.GaussianBlur(solid.astype('float32'),(0,0),8)
    surface=cv2.GaussianBlur(surface,(0,0),8)/np.maximum(support,.001)
    lateral=np.gradient(surface,axis=1)*800
    longitudinal=np.gradient(surface,axis=0)*800
    lx=sample(lateral,u_top,v_top)
    lz=sample(longitudinal,u_top,v_top)
    # A broad studio reflector highlights the gently rolled shoulder. Limit the
    # influence where slope becomes steep at the outer sheet-metal boundary.
    grad=19*np.clip(-lx*np.sign(x),-.42,.58)+14*np.clip(lz+.17,-.3,.3)
    grad*=smooth(t,.035,.15)*(1-smooth(t,.85,.99))
    candidate+=grad[:,None]
    # Retain some detail from the real donor after suppressing the periodic
    # slats vertically. The smoothed texture alone would flatten the pressings.
    candidate=(candidate-[60,64,73])*1.13+[60,64,73]
    # The original photographed, thin side shut lines remain outside the patch.
    # Mark the well-localized portion of each side; do not propagate this line
    # into the low-confidence strip by the headlamps.
    side_x=np.interp(v_top,[185,204,214,226,245,300,357,395,430],
               [181,166,150,141,136,135,136,138,140])
    dside=abs(np.minimum(u_top,720-u_top)-side_x)
    keep_line=(1-smooth(dside,1,5))*smooth(v_top,195,224)*(1-smooth(v_top,404,437))
    weight*=1-keep_line
    # Project a slim cowl transition from the actual front photo to avoid a
    # painted-over windshield base. The source pixels span the border between
    # the lower glass mounting strip and the beginning of the painted hood.
    rear_center=np.interp(abs(u_top-360),[0,65,112,151,170,181],[199,195,189,182,179,184])
    back_dist=v_top-rear_center
    cowl_weight=(1-smooth(back_dist,7,22))*(1-smooth(abs(s),.83,.995))
    cowl_y=373.4+2.8*abs(s)**2+np.clip(back_dist,0,18)*.48
    cowl=sample(rgb(DONOR),771+s*301,cowl_y)
    candidate=candidate*(1-cowl_weight[:,None])+cowl*cowl_weight[:,None]
    blend=weight>0
    old=tex.reshape(-1,3)[valid_ids[blend]].astype('float32')
    new=tex.copy()
    new.reshape(-1,3)[valid_ids[blend]]=np.rint(old+(candidate[blend]-old)*weight[blend,None]).clip(0,255).astype('uint8')
    # Only refresh padding of changed charts.
    dist,near=ndimage.distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(dist<=12)
    new[pad]=new[near[0][pad],near[1][pad]]
    Image.fromarray(new).save(OUT/'hood_texture.png')
    # An explicit affected-texel mask allows downstream verification of scope.
    alpha=np.zeros((height,width),np.uint8);alpha.flat[valid_ids]=np.rint(weight*255).astype('uint8')
    Image.fromarray(alpha).save(OUT/'hood_footprint.png')
    # Compare the whole bonnet in an orthographic top view covering the two XY charts.
    _,idx,bar=render(vert,faces,cam,(688,720))
    hit=idx>=0
    texcoords=np.einsum('ij,ijk->ik',bar[hit],uv[faces[idx[hit]]])
    pics=[]
    for im in [tex,new]:
        px=np.full((688,720,3),24,np.uint8)
        px[hit]=sample(im,texcoords[:,0]*(width-1),(1-texcoords[:,1])*(height-1))
        pics.append(px)
    Image.fromarray(np.hstack(pics)).save(OUT/'hood_top_check.jpg',quality=94)
    print('Changed',int(np.any(tex[valid]!=new[valid],axis=1).sum()),flush=True)
    (OUT/'hood_method.json').write_text(json.dumps(dict(donor=str(DONOR),donor_sha256=sha256(DONOR),base_sha256=sha256(V2/'texture.png'),source='single front_002 locally cleaned photo; strip reparameterization, masked anisotropic Gaussian',top_only=True,changed_valid=int(np.any(tex[valid]!=new[valid],axis=1).sum()))),encoding='utf-8')

if __name__=='__main__':
    main()
