"""Bake low-frequency, overlap-calibrated paint corrections into the frozen v5 texture."""
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent
PHOTO = OUT.parents[1]
PROJECT = PHOTO.parents[1]
BASE = PHOTO / 'outputs/bmw5_mv_textured_v5'
PREP = PHOTO / 'outputs/bmw5_inputs_v2'
REGIONS = PHOTO / 'outputs/material_regions_v1'
FIT = PHOTO / 'outputs/bmw5_mv_proportion_fit_v2'
TRIAL = PHOTO / 'outputs/bmw5_v5_paint_seams_trial_v1'
sys.path[:0] = [str(PHOTO), str(PROJECT / '09_reflection_cleanup')]
from bake_texture import project, sample
from raster import render
from replace_frozen_v5_texture import parse_glb

def read_image(path):
    return np.asarray(Image.open(path))

def glb_texture(path):
    tree, data = parse_glb(path.read_bytes())
    view = tree['bufferViews'][tree['images'][0]['bufferView']]
    offset = view['byteOffset']
    return np.asarray(Image.open(io.BytesIO(data[offset:offset+view['byteLength']])).convert('RGB'))

def smoothstep(v, lo, hi):
    t = np.clip((v-lo)/(hi-lo), 0., 1.)
    return t*t*(3.-2.*t)

def blur(array, sigma):
    return cv2.GaussianBlur(array, (0,0), sigmaX=sigma, sigmaY=sigma)

def photo_lows(photo, paint):
    # Normalized convolution excludes lamps and windows from paint base colors.
    denom = blur(paint, 10)
    return blur(photo*paint[:,:,None], 10)/np.maximum(denom[:,:,None], 1e-6)

def splat_smooth(values, weights, xy, shape):
    # The overlap observations are point samples in the original photo.
    # Smooth and interpolate the adjustment over that one source's sensor plane.
    # Depth and mask tests were applied before creating these samples.
    h,w = shape
    pixels = np.rint(xy).astype(int)
    index = np.clip(pixels[:,1],0,h-1)*w+np.clip(pixels[:,0],0,w-1)
    sums = lambda v: np.bincount(index,weights=v,minlength=w*h).astype(np.float32).reshape(h,w)
    den = blur(sums(weights),6.0)
    return np.stack([blur(sums(weights*channel),6.0)/np.maximum(den,.0001) for channel in values.T],axis=-1), den

def run():
    cv2.setNumThreads(2)
    mesh = np.load(BASE/'textured_mesh.npz')
    verts,faces,uv = (mesh[k] for k in ('vertices','faces','uv'))
    sources = [s['id'] for s in json.loads((BASE/'manifest.json').read_text())['sources']]
    cameras = {v['id']:v['camera'] for v in json.loads((FIT/'cameras.json').read_text())['views']}
    replay = np.load(PROJECT/'09_reflection_cleanup/outputs/v5_frozen_texture_replay_v1/replay_map.npz')
    original = read_image(BASE/'texture.png')
    trial = glb_texture(TRIAL/'textured_mesh.glb')
    src = read_image(BASE/'texture_source_ids.png')
    labels = read_image(REGIONS/'inferred_material_ids.png')
    donor = replay['donor_observed_index']
    sel = (src<8)&(labels==1)
    flatobs = replay['observed_atlas_flat_indices']
    selected_donor = donor[sel]
    ids = src[sel]
    orig_color = original[sel].astype(np.float32)
    uvobs = replay['source_pixels_uv'][selected_donor]
    n = src.shape[0]
    screen = np.column_stack((uv[:,0]*(n-1),(1-uv[:,1])*(n-1),np.ones(len(uv))))
    _, fid, bar = render(screen,faces,dict(projection='orthographic',focal_px=1,cx=0,cy=0),src.shape)
    assert np.array_equal(fid>=0,src!=255)
    tri = verts[faces]
    p = np.einsum('ij,ijk->ik',bar[sel],tri[fid[sel]])
    del bar,fid
    meshes = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    meshes /= np.maximum(np.linalg.norm(meshes,axis=1,keepdims=True),1e-12)
    xyz = 2*(p-verts.min(0))/np.ptp(verts,axis=0)-1
    rear_side = smoothstep(-xyz[:,2],.45,.74)*smoothstep(abs(xyz[:,0]),.48,.78)
    front_side = smoothstep(xyz[:,2],.52,.80)*smoothstep(abs(xyz[:,0]),.48,.78)
    upper_hood = smoothstep(xyz[:,2],.15,.58)*smoothstep(xyz[:,1],-.25,.06)
    upper_hood *= 1-smoothstep(xyz[:,1],.25,.48)
    # A left/right rear three-quarter picture actually sees both sides of the
    # quarter-panel joint. Use it as our appearance reference for that corner.
    lows = np.zeros((len(p),len(sources),3),np.float32)
    weights = np.zeros((len(p),len(sources)),np.float32)
    safe_weight = np.zeros(len(p),np.float32)
    visibility_counts = {}
    for i,name in enumerate(sources):
        cam = cameras[name]
        rgb = read_image(PREP/name/'source_rgb.png').astype(np.float32)
        shape = rgb.shape[:2]
        paint = (read_image(REGIONS/name/'material_ids.png')==1).astype(np.float32)
        edge_dist = cv2.distanceTransform(paint.astype(np.uint8),cv2.DIST_L2,5)
        lower = photo_lows(rgb,paint)
        _,visible,_= render(verts,faces,cam,shape,False)
        u,v,z,center=project(p,cam)
        lows[:,i]=sample(lower,u,v)
        is_owner=ids==i
        assert np.max(np.abs(uvobs[is_owner]-np.column_stack([u[is_owner],v[is_owner]])))<.002
        interior = sample(edge_dist,u,v)
        safe_weight[is_owner] = smoothstep(interior[is_owner],.35,3.5)
        hit = sample(visible.astype(np.float32),u,v,cv2.INTER_NEAREST).astype(np.int32)
        safe_hit=np.maximum(hit,0)
        r=np.asarray(cam['R'])
        pc=p@r.T+cam['t']
        rays=pc/pc[:,2,None]
        plane_n=meshes@r.T
        plane_p=tri[:,0]@r.T+cam['t']
        numerator=np.einsum('ij,ij->i',plane_n,plane_p)[safe_hit]
        denominator=np.einsum('ij,ij->i',plane_n[safe_hit],rays)
        plane_z=np.divide(numerator,denominator,out=np.full(len(p),np.inf),where=abs(denominator)>1e-7)
        facing=np.clip(-(plane_n[safe_hit]*rays).sum(1)/np.linalg.norm(rays,axis=1),0,1)
        visible=(hit>=0)&(abs(plane_z-z)<.0006+.0001*z)
        trusted=smoothstep(interior,4.,13.)*visible*smoothstep(facing,.04,.38)
        # An owner at the edge of the object might project just outside the mask.
        # Its RGB remains in the frozen texture but cannot be a calibration reference.
        prior=np.ones(len(p),np.float32)
        if name in ('rear_005','rear_007','front_001','front_003'):
            sign=1 if name in ('rear_005','front_001') else -1
            corner=rear_side if name.startswith('rear') else front_side*.6
            prior += 30*corner*smoothstep(sign*xyz[:,0],.08,.48)
        elif name in ('rear_006','front_002'):
            prior *= 1-.82*(rear_side if name=='rear_006' else front_side*.65)
        weights[:,i]=trusted*prior*(.23+.77*facing)
        visibility_counts[name]=int((trusted>.05).sum())
    norm=weights.sum(1)
    target=(weights[:,:,None]*lows).sum(1)/np.maximum(norm[:,None],1e-6)
    owner_low=lows[np.arange(len(p)),ids]
    alt_weight=norm-weights[np.arange(len(p)),ids]
    raw_delta=np.clip(target-owner_low,-125,95)
    # Compute a slowly varying correction for each source with confidence from
    # at least one other visible paint observation at that same 3D position.
    corrected=np.zeros_like(orig_color)
    overlap_quality=np.zeros(len(p),np.float32)
    for i,name in enumerate(sources):
        active=ids==i
        w=smoothstep(alt_weight[active],.0,.45)*safe_weight[active]
        field,density=splat_smooth(raw_delta[active],w,uvobs[active],(1024,1536))
        u,v=uvobs[active].T
        corrected[active]=sample(field,u,v)
        overlap_quality[active]=smoothstep(sample(density,u,v),.004,.15)
    region_mix = np.maximum.reduce([.28*np.ones_like(rear_side), .98*rear_side, .64*front_side, .47*upper_hood])
    transition=region_mix*overlap_quality*safe_weight
    trial_delta=trial[sel].astype(np.float32)-orig_color
    delta=(1-transition[:,None])*trial_delta+transition[:,None]*corrected
    # Keep the observed side-panel RGB as an anchor. A straight rear photograph
    # introduces strong silver reflections on the corners of this dark car.
    # Recover a reference color row by row from the selected neighboring side
    # paint on the same side, then use its oblique photo to retain curvature.
    rear_zone = smoothstep(-xyz[:,2],.52,.77)*smoothstep(abs(xyz[:,0]),.30,.52)
    rear_zone *= 1-smoothstep(xyz[:,1],.40,.70)
    anchor_transition=smoothstep(-xyz[:,2],.45,.72)*smoothstep(abs(xyz[:,0]),.40,.63)
    anchor_transition*=1-smoothstep(xyz[:,1],.48,.72)
    for sign,side_name,oblique_name in [(1,'side_004','rear_005'),(-1,'side_008','rear_007')]:
        this_side=xyz[:,0]*sign>0
        side_id=sources.index(side_name)
        oblique_id=sources.index(oblique_name)
        fixed=this_side & np.isin(ids,[side_id,oblique_id])
        delta[fixed]*=(1-anchor_transition[fixed,None])
        band=(ids==side_id)&this_side&(xyz[:,2]<-.70)&(safe_weight>.6)
        levels=np.linspace(-.9,.7,42)
        colors=[]
        for y in levels:
            segment=band&(abs(xyz[:,1]-y)<.07)
            colors.append(np.median(orig_color[segment],axis=0) if segment.sum()>24 else np.full(3,np.nan))
        colors=np.array(colors)
        exist=np.isfinite(colors[:,0])
        if exist.sum()<2:
            raise RuntimeError('Insufficient neighboring paint samples on '+side_name)
        owner=this_side & (ids==sources.index('rear_006')) & (rear_zone>.0)
        anchor=np.stack([np.interp(xyz[owner,1],levels[exist],c[exist]) for c in colors.T],axis=-1)
        # Account for actual regional shading, bounded by what the real adjacent
        # side paint supports; prevent projection error from producing a dark blotch.
        variation=np.clip(lows[owner,oblique_id]-anchor,-28,62)
        reliable=smoothstep(weights[owner,oblique_id],.0,.6)
        preferred=anchor+.36*variation*reliable[:,None]
        # Roll off the bright reflection as a band. Keep naturally darker paint
        # and small features. This is an appearance choice for the frozen texture.
        at_most=np.minimum(preferred-owner_low[owner],0.)
        blend=rear_zone[owner]*safe_weight[owner]
        delta[owner] = delta[owner]*(1-blend[:,None])+at_most*blend[:,None]
    # Keep original high-frequency structure by adding the bias to original RGB.
    correction=np.zeros((len(flatobs),3),np.float32)
    correction[selected_donor]=delta
    valid=src!=255
    paint_texels=valid&(labels==1)&(donor>=0)
    new=original.copy()
    new[paint_texels]=np.clip(np.rint(original[paint_texels]+correction[donor[paint_texels]]),0,255).astype(np.uint8)
    dist,nearest=ndimage.distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(dist<=12)
    new[pad]=new[nearest[0][pad],nearest[1][pad]]
    assert np.array_equal(original[valid&(labels!=1)],new[valid&(labels!=1)])
    Image.fromarray(new).save(OUT/'paint_texture.png')
    log=dict(method='Visibility and paint-mask gated low-frequency RGB bias per source, measured at shared 3D locations. Normalize bright rear-view corner paint to adjacent same-height side paint, retaining a bounded share of the oblique photo variation. The side reference stays fixed in the corner. Elsewhere mix with the prior seam correction.',
             overlap_scale='10-pixel source-image low-pass, 6-pixel correction-field interpolation; sRGB code units',
             visibility_counts=visibility_counts,observed_paint=len(ids),covered_by_other_photo=int((alt_weight>.1).sum()),
             corrected_rear_corner_texels=int(((rear_side>.6)&(transition>.55)).sum()),protected_nonpaint_identical=True)
    (OUT/'paint_calibration.json').write_text(json.dumps(log,indent=2))
    print(json.dumps(log),flush=True)

if __name__=='__main__':
    run()
