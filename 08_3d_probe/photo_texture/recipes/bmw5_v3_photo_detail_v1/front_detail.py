"""Restore BMW grille and headlamps from the frontal and two oblique source photographs."""
from pathlib import Path
import json
import struct
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
PHOTO = OUTPUTS.parent
sys.path.insert(0, str(PHOTO))
from raster import render
from bake_texture import project, sample
sys.path.insert(0, str(OUTPUTS.parents[2] / '09_reflection_cleanup'))
from replace_frozen_v5_texture import render_texture

BASE = OUTPUTS / 'bmw5_v3_roof_refined_v1'
PREP = OUTPUTS / 'bmw5_inputs_v2'
CAMS = {v['id']:v['camera'] for v in json.loads((OUTPUTS/'bmw5_mv_proportion_fit_v2/cameras.json').read_text())['views']}

# Appearance outline just outside the two joined kidney surrounds, tracing the
# photo's top edge to protect the original paint, hood badge, and lower plate.
GRILLE = [(541,494),(549,490),(557,483),(571,478),(596,472),(659,472),(706,472),
          (732,475),(747,481),(758,484),(786,484),(800,478),(817,474),(843,472),
          (910,472),(943,474),(964,478),(979,487),(989,497),(991,532),(983,549),
          (950,577),(928,593),(902,604),(816,605),(784,603),(774,595),(764,595),
          (752,604),(627,606),(606,597),(582,580),(554,554),(544,537)]
# Each lens gets its own near-side source, including the optical corner visible
# in the oblique photo but severely foreshortened in the head-on view. Tie
# vertical coordinates independently at the upper/lower housing contours.
LAMPS = {
  'front_001': {
    'outline': [(582,487),(618,482),(669,474),(711,464),(749,455),(776,449),
                (780,459),(782,470),(777,478),(747,506),(738,510),(697,515),
                (666,518),(648,521),(606,521),(599,518)],
    'old': [(529,501),(552,494),(622,483),(676,466),(701,447),(726,443),(730,459),
            (719,473),(696,498),(704,510),(705,529),(638,532),(577,537),(539,525)]},
  'front_003': {
    'outline': [(752,446),(799,458),(842,468),(893,480),(948,489),(942,501),
                (925,522),(908,523),(867,522),(856,519),(803,515),(785,511),
                (771,502),(748,480),(745,470),(746,457)],
    'old': [(810,491),(831,480),(835,453),(887,464),(944,479),(997,488),
            (1012,493),(999,518),(965,536),(922,531),(878,526),(817,523)]},
}

def rgb(path):
    return np.array(Image.open(path).convert('RGB'))

def coverage(points, inner=2., outer=5.):
    binary=np.zeros((1024,1536),np.uint8)
    cv2.fillPoly(binary,[np.rint(points).astype('int32')],1)
    d=distance_transform_edt(binary)-distance_transform_edt(1-binary)
    t=np.clip((d+outer)/(outer+inner),0,1)
    return (t*t*(3-2*t)).astype('float32')

def donor_coordinates(key, x, y):
    if key == 'front_001':
        return 582+(x-536)*1.066, 487+(y-501)*.98+.079*(x-536)
    if key == 'front_003':
        return 745+(x-825)*1.17, 446+(y-462)*1.035+.057*(x-825)
    return x,y

def lens_chart(key, photo, original_render):
    # Map the whole near-side donor with one affine map, so its upper edge
    # stays continuous and the optical internals retain their relative shapes.
    y,x = np.indices(photo.shape[:2],dtype='float32')
    px,py=donor_coordinates(key,x,y)
    optical=coverage(LAMPS[key]['outline'],inner=0.,outer=1.6)
    weight=cv2.remap(optical,px,py,cv2.INTER_LINEAR)
    src=cv2.remap(photo,px,py,cv2.INTER_LINEAR)
    previous=coverage(LAMPS[key]['old'],1.,5.)
    active=np.maximum(weight,previous)
    # Use original model paint for the collar; inpaint only where the old
    # misaligned light fragments need retiring. Lens pixels themselves are
    # sampled directly from the real photo, never painted over.
    remove=np.uint8(np.maximum(weight,previous)>.01)*255
    bg=cv2.inpaint(original_render,remove,5,cv2.INPAINT_NS)
    merged=np.rint(src*weight[...,None]+bg*(1.-weight[...,None])).astype('uint8')
    return merged,active

def smooth_grille(vertices,faces,uv):
    # An angle change used to kink the slats because the proxy has deep
    # corrugations and a forward lip near their lower end. Fit a gently bowed
    # grille surface through the observed upper/mid slats. Sliding vertices on
    # frontal camera rays preserves every front-view image coordinate.
    cam=CAMS['front_002'];x,y,depth,_=project(vertices,cam)
    area=coverage(GRILLE,inner=21,outer=-2)
    alpha=sample(area,x,y)
    internal=(vertices[:,2]>.86)&(vertices[:,2]<1.01)
    alpha*=internal
    # Broadly curved profile across the kidneys; do not fill old hidden backs.
    xx=vertices[:,0]; yy=vertices[:,1]
    target=.965-.76*xx*xx + .012*(yy+.025)
    alpha *= np.clip((.08-np.abs(vertices[:,2]-target))/.025,0,1)
    shift=(target-vertices[:,2])*alpha
    camera_center=-np.array(cam['R']).T@np.array(cam['t'])
    rays=vertices-camera_center
    requested=rays*(shift/rays[:,2])[:,None]
    # Keep narrow existing rim bevels when a full correction would locally
    # turn over a triangle. UV-split vertices that coincide move together.
    _, group = np.unique(vertices,axis=0,return_inverse=True)
    group = group.reshape(-1)
    scale = np.ones(group.max()+1)
    near = faces[np.any(shift[faces]!=0,axis=1)]
    def cross(v):
        a,b,c = v[near].transpose(1,0,2)
        return np.cross(b-a,c-a)
    before=cross(vertices)
    for _ in range(35):
        new=(vertices+requested*scale[group,None]).astype('float32')
        after=cross(new)
        norms=np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)
        bad=(np.sum(before*after,axis=1)<.10*norms)
        if not np.any(bad):
            return new
        scale[np.unique(group[near[bad]])]*=.7
    raise RuntimeError("Cannot regularize the grille rim")

def normal_update(old_normals,vertices,new,faces):
    unique,index,inverse=np.unique(new,axis=0,return_index=True,return_inverse=True)
    inverse=inverse.reshape(-1)
    f=inverse[faces]
    a,b,c=unique[f].transpose(1,0,2)
    areas=np.cross(b-a,c-a)
    weighted=np.zeros_like(unique)
    for k in range(3):
        np.add.at(weighted,f[:,k],areas)
    n=weighted/np.maximum(np.linalg.norm(weighted,axis=1,keepdims=True),1e-15)
    selected=np.any(new!=vertices,axis=1)&(np.linalg.norm(n[inverse],axis=1)>.99)
    result=old_normals.copy()
    result[selected]=n[inverse[selected]]
    return result

def atlas_geometry(vertices,faces,uv,shape):
    h,w=shape
    screen=np.column_stack([uv[:,0]*(w-1),(1-uv[:,1])*(h-1),np.ones(len(uv))])
    _,idx,b=render(screen,faces,dict(projection='orthographic',focal_px=1,cx=0,cy=0),shape)
    valid=idx>=0
    ind=np.flatnonzero(valid)
    world=np.einsum('ij,ijk->ik',b[valid],vertices[faces[idx[valid]]])
    keep=(world[:,2]>.57)&(world[:,1]>-.16)&(world[:,1]<.07)
    return valid,ind[keep],world[keep]

def apply_patches(source, vertices,faces,uv):
    out=source.copy()
    fields=np.zeros(source.shape[:2],np.uint8)
    valid,ind,world=atlas_geometry(vertices,faces,uv,fields.shape)
    for key in ['front_002','front_001','front_003']:
        cam=CAMS[key]; photo=rgb(PREP/key/'source_rgb.png')
        x,y,depth,_=project(world,cam)
        zmap,_,_=render(vertices,faces,cam,(1024,1536),False)
        dz=depth-sample(np.nan_to_num(zmap,posinf=5.),x,y)
        if key=='front_002':
            a=sample(coverage(GRILLE,1,2),x,y)
            a*=np.clip((.065-dz)/.012,0,1)
            px,py=x,y
        else:
            before=np.asarray(render_texture(render,vertices,faces,uv,source,cam,1536,1024))
            photo,support=lens_chart(key,photo,before)
            a=sample(support,x,y)
            a*=np.clip((.036-dz)/.008,0,1)
            a*=((world[:,0]>0) if key=='front_001' else (world[:,0]<0))
            px,py=x,y
        use=a>0
        patch=sample(photo,px[use],py[use]).astype('float32')
        flat=ind[use]; wt=a[use,None]
        target=out.reshape(-1,3)
        target[flat]=np.clip(np.rint(target[flat]*(1-wt)+patch*wt),0,255).astype('uint8')
        fields.flat[flat]=np.maximum(1,np.rint(a[use]*255)).astype('uint8')
    d,nearest=distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(d<=12)&(fields[tuple(nearest)]>0)
    out[pad]=out[nearest[0][pad],nearest[1][pad]]
    fields[pad]=fields[nearest[0][pad],nearest[1][pad]]
    return out,fields

def inspect(vertices,f,uv,tex):
    for n in ['front_001','front_002','front_003','side_004','side_008']:
        image=render_texture(render,vertices,f,uv,tex,CAMS[n],1536,1024)
        image.save('/tmp/fascia_new_'+n+'.png')

def main():
    m=np.load(OUTPUTS/'bmw5_mv_textured_v5/textured_mesh.npz')
    f,uv=m['faces'],m['uv']
    shape=np.load(BASE/'roof_shape.npz');v=shape['vertices']
    src=rgb(BASE/'roof_texture.png');vnew=smooth_grille(v,f,uv)
    tex,alpha=apply_patches(src,vnew,f,uv)
    Image.fromarray(tex).save(HERE/'fascia_rgb.png')
    Image.fromarray(alpha).save(HERE/'fascia_mask.png')
    inspect(vnew,f,uv,tex)
    np.savez_compressed(HERE/'fascia_vertices.npz',vertices=vnew,normals=normal_update(shape['normals'],v,vnew,f))
    print('changed texels',np.any(tex!=src,axis=2).sum(),'changed vertices',np.any(v!=vnew,axis=1).sum())

if __name__=='__main__': main()
