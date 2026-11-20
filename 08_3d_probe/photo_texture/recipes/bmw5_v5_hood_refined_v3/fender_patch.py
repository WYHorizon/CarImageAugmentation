"""Remove two tiny misplaced light-colored pieces above the actual lens in the matched screenshot."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
import cv2,trimesh
OUT=Path(__file__).resolve().parent
P=OUT.parents[1]
sys.path.insert(0,str(P))
from bake_texture import project,sample,render_textured
from raster import render

v5=P/'outputs/bmw5_mv_textured_v5'
npz=np.load(v5/'textured_mesh.npz')
v,f,uv=[npz[k] for k in ('vertices','faces','uv')]
mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
cam=json.loads((OUT/'camera_views.json').read_text())[0]['camera']
prior=OUT/'hood_gloss_texture.png'
if not prior.exists():prior=OUT/'hood_texture.png'
texture=np.asarray(Image.open(prior).convert('RGB'))
prior_im,hit,depth=render_textured(mesh,uv,texture,cam,628,564,background=(26,26,26))
img=np.array(prior_im)
h,w=img.shape[:2]
# This narrow footprint terminates above the physical lens: no sample at or below row 375 is covered.
poly=[(219,267),(225,269),(228,279),(233,289),(236,300),(235,311),(234,327),
      (234,342),(233,353),(231,362),(228,365),(220,364),(213,357),
      (207,348),(201,338),(199,323),(199,310),(202,298),(207,284),(212,275)]
mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.int32(poly)],1)
near=distance_transform_edt(mask==0)
inside=distance_transform_edt(mask>0)
broad=np.clip((inside-near+4)/6,0,1).astype(np.float32)
# Harmonic interpolation matches every point on the surrounding painted boundary;
# ordinary color smoothing would drag the false highlight back into the correction.
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
reconstruct = broad>0
ry,rx=np.where(reconstruct)
index=np.full(reconstruct.shape,-1,np.int32)
index[ry,rx]=np.arange(len(rx))
rows=[np.arange(len(rx))]
columns=[np.arange(len(rx))]
values=[np.full(len(rx),4.)]
rhs=np.zeros((len(rx),3))
for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
    neighbor=index[ry+dy,rx+dx]
    inside=neighbor>=0
    rows.append(np.flatnonzero(inside))
    columns.append(neighbor[inside])
    values.append(np.full(inside.sum(),-1.))
    rhs[~inside]+=img[ry[~inside]+dy,rx[~inside]+dx]
matrix=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(len(rx),len(rx))).tocsc()
stripe=img.copy()
stripe[reconstruct]=np.clip(np.rint(spsolve(matrix,rhs)),0,255).astype(np.uint8)
# Directly reproject the correct side photo to the isolated speck.
spot=np.zeros((h,w),np.uint8)
cv2.fillPoly(spot,[np.array([(191,362),(198,358),(203,356),(205,363),(205,370),(202,377),
                            (200,382),(198,381),(194,375),(191,369)])],1)
spot=np.clip((distance_transform_edt(spot!=0)-distance_transform_edt(spot==0)+2.)/4,0,1).astype(np.float32)
n=texture.shape[0]
uvv=np.column_stack([uv[:,0]*(n-1),(1-uv[:,1])*(n-1),np.ones(len(uv))])
_,fid,b=render(uvv,f,dict(projection='orthographic',focal_px=1,cx=0,cy=0),(n,n))
valid=fid>=0
idx=np.flatnonzero(valid)
pts=np.einsum('ij,ijk->ik',b[valid],v[f[fid[valid]]])
roi=(pts[:,0]<-.25)&(pts[:,0]>-.34)&(pts[:,2]>.70)&(pts[:,2]<.87)&(pts[:,1]>.0-.03)&(pts[:,1]<.083)
idx,pts=idx[roi],pts[roi]
u,y,z,_=project(pts,cam)
on_surface=sample(np.where(np.isfinite(depth),depth,-10).astype(np.float32),u,y,cv2.INTER_NEAREST)
visible=(on_surface>0)&(abs(on_surface-z)<.003)
weight_a=sample(broad,u,y)*visible
weight_b=sample(spot,u,y)*visible
assert not np.any((weight_a>0)&(weight_b>0))
rgb=np.zeros_like(texture);alpha=np.zeros((n,n),np.uint8)
side=next(e['camera'] for e in json.loads((P/'outputs/bmw5_mv_proportion_fit_v2/cameras.json').read_text())['views'] if e['id']=='side_008')
q,sy,_,_=project(pts,side)
paint=np.array(Image.open(P/'outputs/bmw5_inputs_v2/side_008/source_rgb.png'))
paint_labels=np.array(Image.open(P/'outputs/material_regions_v1/side_008/material_ids.png'))
use=(weight_a>0)
rgb.reshape(-1,3)[idx[use]]=sample(stripe,u[use],y[use])
use=(weight_b>0)&(sample(paint_labels,q,sy,cv2.INTER_NEAREST)==1)
weight_b[~use]=0
rgb.reshape(-1,3)[idx[use]]=sample(paint,q[use],sy[use])
alpha.flat[idx]=np.rint(255*np.maximum(weight_a,weight_b)).astype(np.uint8)
dist, donor=distance_transform_edt(~valid,return_indices=True)
pad=(~valid)&(dist<=12)
rgb[pad]=rgb[donor[0][pad],donor[1][pad]]
alpha[pad]=alpha[donor[0][pad],donor[1][pad]]
np.savez_compressed(OUT/'fender_patch.npz',rgb=rgb,alpha=alpha)
mix=alpha[:,:,None].astype(np.float32)/255
updated=np.rint(texture*(1.-mix)+rgb*mix).astype(np.uint8)
after,_,_=render_textured(mesh,uv,updated,cam,628,564,background=(26,26,26))
new=np.array(after)
diff=np.any(img!=new,axis=2)
assert not diff[385:].any()
crop=(151,245,275,428);panel=Image.new('RGB',(744,549))
for i,im in enumerate([prior_im,after]):
    panel.paste(im.crop(crop).resize((372,549),Image.Resampling.LANCZOS),(372*i,0))
panel.save(OUT/'fender_compare.png')
print('valid edited',int(np.count_nonzero(alpha[valid])),'row_range',np.where(diff)[0].min(),np.where(diff)[0].max(),'lamp_from_row_385_identical',not diff[385:].any())
