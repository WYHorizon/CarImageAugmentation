"""Recover short photographed roof edges and fin detail on the existing repaired roof."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
LAST = OUT / 'bmw5_v3_roof_refined_v1'
sys.path.insert(0, str(LAST))
import build_appearance as roof
import roof_shape as shape

pilot = roof.pilot
CAMERAS = {v['id']: v['camera'] for v in json.loads((OUT/'bmw5_mv_proportion_fit_v2/cameras.json').read_text())['views']}
SOURCE = OUT / 'bmw5_inputs_v2'
FULL = OUT.parents[2] / 'test_images/宝马5系'


def read_photo(view):
    folder,num = view.split('_')
    filename = FULL/folder/f'5_2026_255841_151515_{num}.jpg'
    return np.asarray(Image.open(filename).convert('RGB'))


def sample(photo,u,v):
    k = photo.shape[1]/1536
    return pilot.sample(photo,(u+.5)*k-.5,(v+.5)*k-.5).astype('float32')


def step(v):
    return shape.ramp(v)


def antenna(x,z):
    # Side_004 has an extended, low, nearly straight top; rear_006 has
    # sloping shoulders, a short flat upper face and a flared lower rim.
    t = np.clip((z+.479)/.084,0,1)
    h = np.interp(t,[0,.08,.19,.30,.58,.76,.86,.98,1],
                  [0,.004,.0086,.0116,.0123,.0108,.0077,.0005,0])
    width = np.interp(t,[0,.1,.2,.31,.58,.73,.90,1],
                         [.008,.016,.019,.020,.019,.017,.011,.002])
    shoulder = np.clip((1-np.abs(x)/width)/.55,0,1)
    cross = shoulder*(1.12-.12*shoulder)
    return h*cross,width,h,t


def fin_mesh(verts,faces,orig_n):
    x,y,z = verts.T
    new_h,_,_,_=antenna(x,z)
    old_h=shape.fin_height(x,z)
    weight=step((z+.487)/.012)*step((-.383-z)/.02)*(y>.242)
    new=verts.copy()
    new[:,1]+=(new_h-old_h)*weight
    changed=np.any(new!=verts,axis=1)
    _,glass=shape.read_mesh_data()
    assert not changed[glass].any()
    unique,ix,inv=np.unique(verts,axis=0,return_index=True,return_inverse=True)
    grouped=new[ix]
    assert np.array_equal(new,grouped[inv])
    uf=inv[faces]
    corners=grouped[uf]
    tri_n=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
    touch=changed[faces].any(axis=1)
    assert np.all(tri_n[touch,1]>0)
    acc=np.zeros_like(unique)
    for i in range(3):
        np.add.at(acc,uf[:,i],tri_n)
    acc/=np.maximum(1e-12,np.linalg.norm(acc,axis=1,keepdims=True))
    normal=orig_n.copy()
    normal[changed]=acc[inv[changed]]
    return new,normal,changed


def ribbon(x_coord, dist, zone, view, photo, old):
    # x is the screen coordinate in the actual donor; the fitted source
    # mesh exposes fewer pixels of the rear band than the photograph.
    # Register the visible strip by its outline and glass-border landmarks.
    u=x_coord.copy()
    if view=='rear_006':
        xx=[524,550,580,610,650,685,725,765,800,845,880,920,960,997,1012]
        yy=[253,247,243,240,236.5,233.8,232.6,232.3,232.6,234.1,235.6,239.3,242.1,248,252]
        upper=[236,230,225,222,219,217,216.4,216,216.5,218,220,223,226,232,237]
        width=.102
        fade=.029
    else:
        xx=[525,545,575,615,660,710,750,788,825,870,908,943,977,1002]
        yy=[260,255,250,246,242.3,240,239.1,239.6,240.5,243,246.5,251,255,259]
        upper=[248,240,236,232,229,226,225,225,226,228,232,236,241,248]
        width=.040
        fade=.013
    near=np.interp(u,xx,yy)
    far=np.interp(u,xx,upper)
    frac=np.clip(dist/width,0,1)
    v=near+(far-near)*frac
    # The antenna has its own 3D-correct mapping. Remove it from the
    # strip donor where it would otherwise appear twice at grazing views.
    area=sample(photo,u,v)
    if view=='rear_006':
        obstruct=(u>749)&(u<792)&(v<226)
        if obstruct.any():
            vv=v[obstruct]
            xa=np.zeros_like(vv)+746
            xb=np.zeros_like(vv)+794
            w=((u[obstruct]-746)/48)[:,None]
            area[obstruct]=(1-w)*sample(photo,xa,vv)+w*sample(photo,xb,vv)
    # Attenuate broad lighting differences but retain the measured dark
    # panel cuts and narrow trim highlights with the donor colour ordering.
    # Transfer measured detail contrast only at features resolved in the
    # front/rear views. Suppress the broad photographed shadow/illumination
    # ramp; stretching that ramp into overhead atlas space would leave a
    # painted block on the roof away from the actual edge.
    luminance=photo@np.array([.2126,.7152,.0722],dtype='float32')
    k=photo.shape[1]/1536
    smooth=cv2.GaussianBlur(luminance,(0,0),sigmaX=4.2*k,sigmaY=2.2*k)
    reference=pilot.sample(smooth,(u+.5)*k-.5,(v+.5)*k-.5)
    L=area@np.array([.2126,.7152,.0722])
    if view=='rear_006':
        # Traced from the 4464x2976 rear image. No top/front continuation of
        # the panel is extrapolated, only two photographed 15 px side lines.
        curve_x=[552,598,641,673,691,720,745,770,803,833,851,869,912,958,992]
        curve_y=[244,238,235.5,231.0,230.1,228.7,228.1,227.9,228.3,229.4,230.5,232.6,235.4,239.1,244]
        lip=np.exp(-.5*((v-np.interp(u,curve_x,curve_y))/2.3)**2)
        ly=np.interp(v,[217,219,222,225,228,232,234],[696,688.7,682.7,678.6,675.9,673,672.3])
        ry=np.interp(v,[217,220,223,226,229,232,235],[843,852.3,858,862.8,866,868.8,870.2])
        side_lines=np.maximum(np.exp(-.5*((u-ly)/2.6)**2),np.exp(-.5*((u-ry)/2.6)**2))
        side_lines*=step((v-215)/3)*step((235.5-v)/3)
        support=np.maximum(lip,side_lines)
    else:
        curve_x=[535,564,588,615,645,680,720,751,780,805,833,865,891,920,948,977,995]
        curve_y=[248,244,242,239,237,234.6,233.5,232.8,233.1,233.4,234.4,235.7,237.4,239.4,242,245.8,248.8]
        support=np.exp(-.5*((v-np.interp(u,curve_x,curve_y))/2.1)**2)
    detail=np.clip(.82*(L-reference),-34,24)
    target=old + detail[:,None] + .12*(area-L[:,None])

    a=step((width-dist)/fade)*step((dist+.001)/.004)*zone
    a*=support
    a*=step((u-xx[0])/(xx[1]-xx[0]))*step((xx[-1]-u)/(xx[-1]-xx[-2]))
    return a,target


def fin_photo(points, rear, side, roof_rgb):
    x,y,z=points.T
    h,w,peak,t=antenna(x,z)
    f=np.clip(h/np.maximum(peak,1e-6),0,1)
    # Side donor original pixel positions for upper surface and paint
    # boundary around the low, trapezoidal antenna.
    side_u=np.interp(t,[0,.1,.2,.30,.58,.76,.86,1],[1089,1086,1082,1079,1063,1052,1047,1039])
    top=np.interp(side_u,[1039,1043,1048,1054,1064,1073,1079,1084,1089],
                          [313.6,312,309.8,309.4,309.5,310,311.2,315,320])
    bottom=np.interp(side_u,[1039,1057,1089],[313.8,316.5,320.8])
    sy=bottom-.5-(bottom-top-1.3)*np.clip(f,.05,.93)
    side_rgb=sample(side,side_u,sy)
    # Sample within the rear photographed housing, including true flat
    # face / side bevel. Both vertical source edges slope inward at top.
    u=772.3-((x/np.maximum(.003,w)).clip(-1,1))*(15.7-6.7*f)
    v=225-16.5*f
    rear_rgb=sample(rear,u,v)
    back=step((.49-t)/.27)
    col=side_rgb*(1-back[:,None])+rear_rgb*back[:,None]
    col=np.clip(col,23,131)
    # Slightly lift the upper bevel in the angled/top view; the source
    # head-on fin samples include both near-black back and lighter side.
    lit=step((.075-t)/.09) + .7*step((t-.12)/.18)
    ridge=np.exp(-((x+.004)/.0065)**2)*step(h/.0075)*lit
    col+=ridge[:,None]*11

    # The short band around the base is sampled from its photographed
    # rear lip. Its contracted footprint is confined to the housing.
    lip_h,_,_,_=antenna(x*.85,-.438+(z+.438)*.91)
    lip=step(lip_h/.0012)
    u_lip=772.2-17.8*np.clip(x/np.maximum(w,.006),-1,1)
    lip_color=sample(rear,u_lip,224.5+.9*np.cos(2*np.pi*t))
    a=step(h/.0020)
    layer=roof_rgb+lip[:,None]*(lip_color*.87-roof_rgb)
    return layer+a[:,None]*(col-layer),np.maximum(lip,a)


def main():
    mesh=np.load(pilot.BASE/'textured_mesh.npz')
    oldroof=np.load(LAST/'roof_shape.npz')
    geom=oldroof['vertices']
    f,uv=mesh['faces'], mesh['uv']
    new,norm,moved=fin_mesh(geom,f,oldroof['normals'])
    np.savez_compressed(HERE/'roof_vertices.npz',vertices=new,normals=norm)
    base=pilot.rgb(LAST/'roof_texture.png')
    oldmask=np.asarray(Image.open(LAST/'roof_mask.png'))
    windows=np.asarray(Image.open(OUT/'bmw5_v3_opaque_glass_v1/window_ids.png'))
    W,H=base.shape[1],base.shape[0]
    screen=np.column_stack([uv[:,0]*(W-1),(1-uv[:,1])*(H-1),np.ones(len(uv))])
    _,fid,bary=pilot.render(screen,f,dict(projection='orthographic',focal_px=1,cx=0,cy=0),(H,W))
    valid=fid>=0
    sel=np.flatnonzero(valid&(oldmask>0)&(windows==0))
    points=np.einsum('ij,ijk->ik',bary.reshape(-1,3)[sel],new[f[fid.flat[sel]]])
    xc,yc,_,_=pilot.project(points,roof.CAM)
    xl,xr,yr,yf,sided,_=roof.chart_boundaries(xc,yc)
    zone=step((sided-3)/14)
    distR=(yc-yr)/1150;distF=(yf-yc)/1150
    target=base.copy()
    buf=target.reshape(-1,3)
    alpha=np.zeros((H,W),np.uint8)
    rear,side,front=[read_photo(v) for v in ('rear_006','side_004','front_002')]
    for dist,view,src in [(distR,'rear_006',rear),(distF,'front_002',front)]:
        old=buf[sel].astype(float)
        u,_,_,_=pilot.project(points,CAMERAS[view])
        a,color=ribbon(u,dist,zone,view,src,old)
        buf[sel]=np.uint8(np.clip(np.rint(old+a[:,None]*(color-old)),0,255))
        alpha.flat[sel]=np.maximum(alpha.flat[sel],np.uint8(np.rint(255*a)))
    regular=buf[sel].astype(float)
    # Clear original simulated fin in its narrow outline. Its replacement
    # profile has a slightly different base and photographed paint.
    old_h=shape.fin_height(points[:,0],points[:,2])
    old_a=step(old_h/.0013)
    s=(xc-xl)/(xr-xl); t=(yc-yr)/(yf-yr)
    underlying=regular+old_a[:,None]*(roof.roof_color(s,t)-regular)
    housing,active=fin_photo(points,rear,side,underlying)
    af=np.maximum(active,old_a)
    buf[sel]=np.uint8(np.clip(np.rint(regular+af[:,None]*(housing-regular)),0,255))
    alpha.flat[sel]=np.maximum(alpha.flat[sel],np.uint8(np.rint(255*af)))
    dist,near=distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(windows==0)&(dist<=12)&(alpha[tuple(near)]>0)&(oldmask>0)
    target[pad]=target[near[0][pad],near[1][pad]]
    alpha[pad]=alpha[near[0][pad],near[1][pad]]
    change=(target!=base).any(axis=2)
    assert not np.any(change&(windows!=0))
    alpha[change&(alpha==0)]=1
    Image.fromarray(target).save(HERE/'roof_rgb.png')
    Image.fromarray(alpha).save(HERE/'roof_mask.png')
    samples={}
    for view in ('rear_006','rear_005','side_004','front_002'):
        im=pilot.render_texture(pilot.render,new,f,uv,target,CAMERAS[view],1536,1024)
        im.save('/tmp/roofdetail_current_'+view+'.png')
        if view in ('side_004','rear_006'):
            patches={'rear_006':(748,189,792,225),'side_004':(1038,295,1096,328)}
            x0,y0,x1,y1=patches[view]
            _,f0,_=pilot.render(geom,f,CAMERAS[view],(1024,1536),False)
            _,f1,_=pilot.render(new,f,CAMERAS[view],(1024,1536),False)
            actual=np.asarray(Image.open(SOURCE/view/'mask_geometry.png'))>0
            def mae(model):return float(np.abs(np.argmax(model[y0:y1,x0:x1],axis=0)-np.argmax(actual[y0:y1,x0:x1],axis=0)).mean())
            samples[view]=[mae(f0>=0),mae(f1>=0)]
    pilot.render_texture(pilot.render,new,f,uv,target,roof.CAM,roof.CHART_SHAPE[1],roof.CHART_SHAPE[0]).save('/tmp/roofdetail_current_top.png')
    r=dict(actual_photo_sources=['rear_006','front_002','side_004'],shape_reference=['side_004','side_008','rear_006','rear_005','rear_007'],changed_vertices=int(moved.sum()),protected_glass_changes=0,changed_texture_texels=int(change.sum()),fin_silhouette_mean_error_pixels_before_after=samples)
    (HERE/'roof_detail_verification.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r)
if __name__=='__main__':main()
