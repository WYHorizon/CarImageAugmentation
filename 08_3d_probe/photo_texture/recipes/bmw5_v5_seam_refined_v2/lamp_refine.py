"""Realign each complete rear lamp to one oblique photograph in fitted camera space."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parent
photo = out.parents[1]
sys.path.insert(0,str(photo))
from bake_texture import project, sample, render_textured
from raster import render
import trimesh

base = photo/'outputs/bmw5_mv_textured_v5'
prep = photo/'outputs/bmw5_inputs_v2'
fit = photo/'outputs/bmw5_mv_proportion_fit_v2'
original = np.array(Image.open(base/'texture.png').convert('RGB'))
mesh = np.load(base/'textured_mesh.npz')
v, f, uv = [mesh[k] for k in ['vertices','faces','uv']]
geom = trimesh.Trimesh(vertices=v,faces=f,process=False)
cams = {view['id']:view['camera'] for view in json.loads((fit/'cameras.json').read_text())['views']}
n = original.shape[0]
uvverts = np.column_stack((uv[:,0]*(n-1), (1-uv[:,1])*(n-1), np.ones(len(uv))))
_, faces, bary = render(uvverts, f, dict(projection='orthographic', focal_px=1,cx=0,cy=0), original.shape[:2])
valid = faces>=0
ids = np.flatnonzero(valid)
points = np.einsum('ni,nij->nj', bary[valid], v[f[faces[valid]]])
# Tail paint and fittings elsewhere remain outside this explicit camera-local crop.
rear = (points[:,2]<-0.50) & (np.abs(points[:,0])>.10)
ids, points = ids[rear], points[rear]
configs = {
    # Correspondences span outer lamp tip, raised outer segment, hatch seam and inner tip.
    'rear_005': {
        'x_from':[780,793,852,933,1045,1060], 'x_to':[839,850,912,980,1073,1086],
        'y_from':[449,449,440,443,449,449], 'y_to':[447,447,441,452,455,455],
        'old_outline':[(845,455),(895,408),(924,408),(939,426),(976,433),(1049,438),(1078,449),(1079,475),(1059,482),(984,483),(979,474),(911,474),(904,454)],
        'new_outline':[(790,459),(800,450),(835,418),(851,412),(876,413),(883,418),(977,422),(1014,428),(1048,442),(1050,453),(1040,469),(1029,476),(935,479),(933,464),(797,462)]},
    'rear_007': {
        'x_from':[482,497,608,672,745,761], 'x_to':[439,452,544,611,698,713],
        'y_from':[450,450,445,440,449,449], 'y_to':[455,455,455,448,454,454],
        'old_outline':[(450,450),(463,442),(537,435),(588,434),(611,420),(617,414),(640,414),(657,424),(699,461),(697,468),(622,467),(619,475),(543,475),(537,483),(468,480)],
        'new_outline':[(491,451),(504,435),(528,426),(609,422),(657,420),(666,415),(686,413),(702,418),(722,430),(749,460),(749,464),(609,464),(605,478),(509,479)]},
}

def polygon_field(shape, poly):
    mask = np.zeros(shape,np.uint8)
    cv2.fillPoly(mask,[np.array(poly,np.int32)],1)
    # A narrow collar covers isolated displaced pixels and permits a smooth transition.
    outer=distance_transform_edt(~mask.astype(bool)).astype(np.float32)
    return np.clip((4-outer)/4,0,1)

rgb = np.zeros_like(original)
alpha = np.zeros(original.shape[:2],np.uint8)
summaries={}
for name,c in configs.items():
    cam=cams[name]
    src=np.array(Image.open(prep/name/'source_rgb.png').convert('RGB'))
    h,w=src.shape[:2]
    u,y,z,_ = project(points,cam)
    ux=np.interp(u,c['x_to'],c['x_from'])
    center_ref=np.interp(ux,c['x_from'],c['y_from'])
    center_output=np.interp(u,c['x_to'],c['y_to'])
    vy=y-center_output+center_ref
    # Interpolated x coordinates clamp outside the control interval, mask projection there.
    new_field=polygon_field((h,w), c['new_outline'])
    old_field=polygon_field((h,w), c['old_outline'])
    # Original displaced red pixels can lie just outside the aligned outline.
    # Retire those pixels without repainting the surrounding trunk or lettering.
    repro=np.asarray(Image.open(base/'reprojections'/f'{name}.png').convert('RGB'),dtype=np.int16)
    red=(repro[:,:,0]-np.maximum(repro[:,:,1],repro[:,:,2])>12)&(old_field>.5)
    distance_from_old_red=distance_transform_edt(~red)
    old_field*=np.clip((6-distance_from_old_red)/4,0,1).astype('float32')
    weight=np.maximum(sample(old_field,u,y),sample(new_field,ux,vy))
    # The real rear badge rides just above each lamp, outside the lens. Its RGB is locked.
    x1,y1,x2,y2 = ((984,418,1025,437) if name=='rear_005' else (481,417,540,436))
    dx=np.maximum(np.maximum(x1-u,0),u-x2)
    dy=np.maximum(np.maximum(y1-y,0),y-y2)
    weight*=np.clip(np.hypot(dx,dy)/3,0,1)
    weight[(u<c['x_to'][0])|(u>c['x_to'][-1])]=0
    depth,_,_=render(v,f,cam,(h,w),return_barycentric=False)
    zz=sample(depth,u,y,cv2.INTER_NEAREST)
    weight[(~np.isfinite(zz))|(np.abs(zz-z)>.0018)] = 0
    use=weight>0
    if np.any(alpha.flat[ids[use]]):
        raise RuntimeError('Rear light patches overlap')
    rgb.reshape(-1,3)[ids[use]]=sample(src,ux[use],vy[use])
    alpha.flat[ids[use]]=np.rint(255*weight[use]).astype(np.uint8)
    summaries[name]=int(use.sum())
# Copy the local patch into the existing mipmap margin only where a valid donor is edited.
d,nearest=distance_transform_edt(~valid,return_indices=True)
pad=(~valid)&(d<=12)
rgb[pad]=rgb[nearest[0][pad],nearest[1][pad]]
alpha[pad]=alpha[nearest[0][pad],nearest[1][pad]]
tmp=out/'lamp_patch.tmp.npz'
np.savez_compressed(tmp,rgb=rgb,alpha=alpha)
tmp.replace(out/'lamp_patch.npz')
changed=np.rint((original.astype(float)*(255-alpha[:,:,None])+rgb.astype(float)*alpha[:,:,None])/255).astype('uint8')
rows=[]
views=json.loads((base/'render_cameras.json').read_text())
crops={120:(770,390,1190,510),150:(665,386,860,480),180:(350,373,850,502),210:(315,380,525,481),240:(70,394,477,496)}
for angle,box in crops.items():
    view=next(view for view in views if view['azimuth']==angle)
    updated,_,_=render_textured(geom,uv,changed,view['camera'],view['width'],view['height'])
    old=Image.open(base/f'renders/az_{angle:03}.png').convert('RGB')
    delta=np.any(np.array(old)!=np.array(updated),axis=2)
    ys,xs=np.where(delta)
    if xs.size==0:continue
    # Crop from both photos' footprints for correct side/back inspection.
    box=(max(0,xs.min()-20), max(0,ys.min()-20), min(view['width'],xs.max()+21), min(view['height'],ys.max()+21))
    if angle==150:box=(665,385,860,480)
    if angle==210:box=(340,385,550,480)
    a=old.crop(box)
    b=updated.crop(box)
    scale=min(2.0,550/a.width)
    size=(int(a.width*scale),int(a.height*scale))
    a=a.resize(size,Image.Resampling.LANCZOS)
    b=b.resize(size,Image.Resampling.LANCZOS)
    row=Image.new('RGB',(1120,a.height+20),'#eeeeee')
    ImageDraw.Draw(row).text((4,3),f'{angle} degrees  |  v5  /  single-photo lamps',fill='black')
    row.paste(a,(0,20));row.paste(b,(560,20))
    rows.append(row)
panel=Image.new('RGB',(1120,sum(r.height for r in rows)),'white')
y=0
for r in rows:panel.paste(r,(0,y));y+=r.height
panel.save(out/'lamp_checks.png')
print('visible UV counts',summaries,flush=True)
