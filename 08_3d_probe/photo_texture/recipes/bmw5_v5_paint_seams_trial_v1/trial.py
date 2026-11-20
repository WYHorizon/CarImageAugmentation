"""Local, paint-only texture leveling for the frozen BMW5 v5 asset."""
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy import ndimage

out = Path(__file__).resolve().parent
photo = out.parents[1]
project = photo.parents[1]
base = photo / 'outputs/bmw5_mv_textured_v5'
sys.path[:0] = [str(photo), str(project / '09_reflection_cleanup')]
from raster import render
from replace_frozen_v5_texture import patch_glb, parse_glb, render_texture, sha256

def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))

mesh = np.load(base / 'textured_mesh.npz')
v, f, uv = (mesh[k] for k in ('vertices', 'faces', 'uv'))
original = rgb(base / 'texture.png')
src = np.asarray(Image.open(base / 'texture_source_ids.png'))
regions = photo / 'outputs/material_regions_v1'
safe = np.asarray(Image.open(regions / 'paint_safe_mask.png')) > 0
labels = np.asarray(Image.open(regions / 'inferred_material_ids.png'))
replay = np.load(project / '09_reflection_cleanup/outputs/v5_frozen_texture_replay_v1/replay_map.npz')
donor = replay['donor_observed_index']
valid = src != 255
n = src.shape[0]
screen = np.column_stack((uv[:,0]*(n-1), (1-uv[:,1])*(n-1), np.ones(len(uv))))
_, faces, bary = render(screen, f, dict(projection='orthographic', focal_px=1, cx=0, cy=0), src.shape)
assert np.array_equal(valid, faces>=0)
# Estimate low-frequency source colors only on conservatively labeled observed paint.
sel = safe & (src<8)
fid = faces[sel]
tri = v[f[fid]]
p = np.einsum('ij,ijk->ik', bary[sel], tri)
normal = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
cols = original[sel].astype(np.float64)
ids = src[sel].astype(np.int32)
# Group by position and face orientation to avoid mixing opposite sides of thin panels.
length = float(np.ptp(v[:,2]))
voxel = length/320
direction = np.abs(normal).argmax(1)
orientation = direction*2+(normal[np.arange(len(p)),direction]>0)
cells, group = np.unique(np.column_stack((np.floor(p/voxel),orientation)).astype(int),axis=0,return_inverse=True)
count = np.bincount(group)
def group_mean(x):
    return np.column_stack([np.bincount(group,weights=a)/count for a in x.T])
centers = group_mean(p)
normals = group_mean(normal)
normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-10)
combined = group*8+ids
counts = np.bincount(combined,minlength=len(cells)*8).reshape(-1,8)
sums = np.stack([np.bincount(combined,weights=c,minlength=len(cells)*8).reshape(-1,8) for c in cols.T],axis=-1)
# Equalize spatial influence across UV islands of different texel densities.
sums /= np.maximum(count[:,None,None],1)
weights = counts/count[:,None]
tree = cKDTree(centers)
fields = np.zeros((len(cells),8,3),np.float32)
sigma = length*.014
for start in range(0,len(cells),512):
    stop=min(start+512,len(cells))
    ds, index=tree.query(centers[start:stop], k=min(160,len(cells)),workers=2)
    cosine=np.einsum('ijk,ik->ij',normals[index],normals[start:stop])
    kernel=np.exp(-.5*(ds/sigma)**2)*np.clip((cosine-.35)/.65,0,1)**2
    sums_local=np.einsum('ij,ijkm->ikm',kernel,sums[index])
    weights_local=np.einsum('ij,ijk->ik',kernel,weights[index])
    common=sums_local.sum(1)/np.maximum(weights_local.sum(1,keepdims=True),1e-8)
    local=sums_local/np.maximum(weights_local[:,:,None],1e-8)
    fields[start:stop]=np.clip(common[:,None,:]-local,-68,68)*.92
# Interpolate the offsets across cells so the correction itself has no voxel-grid edges.
offsets = np.zeros_like(cols)
for start in range(0,len(p),20000):
    stop=min(start+20000,len(p))
    ds, ix=tree.query(p[start:stop],k=12,workers=2)
    alignment=np.einsum('ijk,ik->ij',normals[ix],normal[start:stop])
    nearby=np.exp(-.5*(ds/(voxel*.9))**2)*np.clip((alignment-.4)/.6,0,1)**2
    owner=ids[start:stop,None]
    nearby*=counts[ix,owner]>0
    den=nearby.sum(1)
    off=np.einsum('ij,ijk->ik',nearby,fields[ix,owner])/np.maximum(den[:,None],1e-8)
    off[den<1e-8]=fields[group[start:stop][den<1e-8],ids[start:stop][den<1e-8]]
    offsets[start:stop]=off
# Feather toward protected regions using the original photo-space masks (continuous across UV cuts).
obs_flat = replay['observed_atlas_flat_indices']
obs_source = replay['observed_source_ids']
obs_uv = replay['source_pixels_uv']
photo_weight = np.zeros(len(obs_flat),np.float32)
for i,ref in enumerate(json.loads((base/'manifest.json').read_text())['sources']):
    active = obs_source==i
    mask = np.asarray(Image.open(regions/ref['id']/'paint_safe_mask.png'))>0
    dist = ndimage.distance_transform_edt(mask)
    xy = obs_uv[active]
    photo_weight[active]=np.clip(ndimage.map_coordinates(dist,[xy[:,1],xy[:,0]],order=1)/5.,0,1)
delta = np.zeros((len(obs_flat),3),np.float32)
delta[donor[sel]]=offsets*photo_weight[donor[sel],None]
new=original.copy()
paint=valid & (labels==1) & (donor>=0)
new[paint]=np.clip(np.rint(original[paint]+delta[donor[paint]]),0,255).astype(np.uint8)
# Mirror corrected edge colors into chart padding exactly as in the original bake.
dist,nearest=ndimage.distance_transform_edt(~valid,return_indices=True)
pad=(~valid)&(dist<=12)
new[pad]=new[nearest[0][pad],nearest[1][pad]]
assert np.array_equal(new[valid&(labels!=1)],original[valid&(labels!=1)])
buffer=io.BytesIO()
Image.fromarray(new).save(buffer,format='PNG')
old_glb = base/'textured_mesh.glb'
old_hash=sha256(old_glb)
data,checks=patch_glb(old_glb.read_bytes(),buffer.getvalue())
assert checks['embedded_png_exact']
(out/'textured_mesh.glb').write_bytes(data)
glb_tree,glb_bin=parse_glb((out/'textured_mesh.glb').read_bytes())
image_view=glb_tree['bufferViews'][glb_tree['images'][0]['bufferView']]
offset=image_view['byteOffset']
new=np.asarray(Image.open(io.BytesIO(glb_bin[offset:offset+image_view['byteLength']])))
views=json.loads((base/'render_cameras.json').read_text())
snapshots={}
for angle in (30,90,150,210,330):
    view=next(view for view in views if view['azimuth']==angle)
    cam=view['camera']
    w,h=view['width'],view['height']
    old=render_texture(render,v,f,uv,original,cam,w,h)
    assert np.array_equal(np.asarray(old),rgb(base/f'renders/az_{angle:03}.png'))
    updated=render_texture(render,v,f,uv,new,cam,w,h)
    snapshots[angle]=(old,updated)
    updated.save(out/f'az_{angle:03}.png')
# Match the supplied model and validate directly adjacent cross-source paint texels.
before=[]
after=[]
for axis in (0,1):
    a = (slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
    b = (slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
    k = sel[a]&sel[b]&(src[a]!=src[b])
    for arr,vals in [(original,before),(new,after)]:
        vals.extend(np.abs((arr[a][k].astype(float)-arr[b][k])@np.array([.2126,.7152,.0722])))
assert old_hash==sha256(old_glb)
info=dict(original_sha256=old_hash,glb_mesh_uv_normals_material_unchanged=checks['geometry_normals_uv_scene_material_metadata_exact'],
    protected_nonpaint_exact=True,changed_valid_paint_texels=int(np.any(new[paint]!=original[paint],axis=1).sum()),
    adjacent_observed_safe_paint_cross_source_pairs=len(before),median_luma_jump_8bit_before=float(np.median(before)),
    median_luma_jump_8bit_after=float(np.median(after)),view_150_y464_x735_736_RGB=[[np.asarray(im)[464,735:737].tolist() for im in snapshots[150]]])
(out/'metrics.json').write_text(json.dumps(info,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
canvas=Image.new('RGB',(1120,780),'#f8f9fa')
draw=ImageDraw.Draw(canvas)
for j,title in enumerate(['Original v5','Paint seam trial']):
    draw.text((20+560*j,9),title,fill='#20252a',font=font)
for j in (0,1):
    for y,angle,box,sz in [(42,150,(266,290,1030,636),(550,249)),(302,150,(675,394,842,537),(300,257)),(575,30,(189,300,924,642),(435,202))]:
        im=snapshots[angle][j].crop(box).resize(sz,Image.Resampling.LANCZOS)
        canvas.paste(im,(j*560+(560-im.width)//2,y))
canvas.save(out/'comparison.png')
print(json.dumps(info,indent=2))
