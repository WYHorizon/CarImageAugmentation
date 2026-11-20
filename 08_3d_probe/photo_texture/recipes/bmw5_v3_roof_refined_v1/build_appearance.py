"""Bake one graphite roof onto the existing v3 UVs, preserving all glass texels."""
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
sys.path.insert(0, str(OUTPUTS / "bmw5_v3_glass_pilot_v1"))
import build_glass as pilot
from roof_shape import fin_height

CAM = pilot.chart(0, 90, 1150, 350, 650)
CHART_SHAPE = (930, 700)


def smoothstep(distance, width):
    t = np.clip(distance / width, 0, 1)
    return t * t * (3 - 2*t)


def chart_boundaries(x, y):
    # Traced on a true-scale overhead orthographic chart of the accepted car.
    # The chrome rails bound either side. End curves run just into the dedicated
    # windscreen mask so it can provide the last subpixel edge of the pane.
    ys = [57, 105, 175, 250, 330, 430, 510, 600, 700, 775, 825, 860, 890]
    left = np.interp(y, ys, [73,78,83,88,92,93,93,91,84,75,66,60,54])
    right = np.interp(y, ys, [630,627,622,616,611,609,610,613,621,632,643,651,657])
    xs = [60,80,100,120,144,180,225,270,330,370,430,480,520,560,580,600,620,640,661]
    rear = np.interp(x, xs, [102,96,92,103,99,92,82,75,70,71,75,80,88,97,100,90,96,102,106])
    front = np.interp(x, xs, [841,844,827,814,811,820,828,835,842,841,835,830,821,813,811,826,845,846,842])
    side_in = np.minimum(x - left, right - x)
    end_in = np.minimum(y - rear, front - y)
    return left, right, rear, front, side_in, end_in


def roof_color(s, t):
    # The side_004 upper rails bracket RGB (51,56,69)..(74,81,94).
    # A continuous, broader daylight sheen across the crown accounts for
    # its shallower inclination. No source-photo edges or panoramic cut lines.
    cross = s * 2 - 1
    edge = np.abs(cross)
    main_reflection = np.exp(-((cross+.64-.30*t+.08*np.cos(5*t))/.30)**2)
    second_reflection = np.exp(-((cross-.71+.11*np.cos(3*t))/.18)**2)
    rear_light = np.exp(-((t-.04)/.29)**2) * (.7 + .3*np.cos(2.8*cross))
    sky = 9*(1 - cross**2) + 7*np.cos((t-.28)*np.pi)
    level = 48 + sky + 28*main_reflection + 16*second_reflection + 11*rear_light
    level -= 7*edge**8 + 6*np.exp(-((1-t)/.085)**2)
    return np.column_stack([level, level + 5, level + 15 - 3*rear_light])


def fin_color(points, color):
    # Geometry and paint share the same model-space orientation. In side
    # profile the fin runs forward from the middle of the rear roof strip.
    x = points[:, 0]
    z = points[:, 2]
    h = fin_height(x,z)
    strength = smoothstep(h,.002)
    eps=.0015
    dx=(fin_height(x+eps,z)-fin_height(x-eps,z))/(2*eps)
    dz=(fin_height(x,z+eps)-fin_height(x,z-eps))/(2*eps)
    normals=np.column_stack([-dx,np.ones_like(dx),-dz])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    overcast=np.clip(normals@np.array([-.64,.68,.35]),0,1)
    narrow_highlight=np.maximum(0,normals@np.array([-.54,.765,.35]))**14
    # Raised flanks can reflect much more sky than the panel under the fin.
    # Include the roof's broad radiance on its exposed upper ridge, keeping
    # the vertical/back surfaces close to the dark moulded antenna housing.
    highlight = np.exp(-((x+.0035)/.0048)**2)
    ridge = 16*highlight*smoothstep(h,.010)
    light=31+53*overcast+31*narrow_highlight+ridge
    fin = np.column_stack([light, light+5, light+10])
    cast = smoothstep(fin_height(x-.0025,z+.0015),.004)*(1-strength)
    color = color - 8*cast[:,None]
    result = color + (fin-color)*strength[:,None]
    assert np.all(np.isfinite(result))
    return result


def bake(vertices, corrected_vertices, faces, uv, base, protected):
    height, width = base.shape[:2]
    screen = np.column_stack([uv[:,0]*(width-1), (1-uv[:,1])*(height-1),
                              np.ones(len(uv))])
    _, fid, bary = pilot.render(screen, faces, dict(projection="orthographic",focal_px=1,cx=0,cy=0),
                                (height,width))
    valid=fid>=0
    index = np.flatnonzero(valid & ~protected)
    points = np.einsum("ij,ijk->ik", bary.reshape(-1,3)[index], vertices[faces[fid.flat[index]]])
    x,y,depth,_ = pilot.project(points,CAM)
    left,right,rear,front,side_in,end_in = chart_boundaries(x,y)
    footprint = (side_in>0) & (end_in>0)
    top,_,_ = pilot.render(vertices,faces,CAM,CHART_SHAPE,False)
    top_depth = pilot.sample(np.nan_to_num(top,posinf=4), x, y)
    # Include steep fins and local fold faces within the roof envelope. Their
    # former hidden RGB would otherwise become visible when ghost bumps flatten.
    keep = footprint & (depth-top_depth<.050) & (np.abs(depth-top_depth)<.060)
    idx = index[keep]
    s = np.clip(((x-left)/(right-left))[keep],0,1)
    t = np.clip(((y-rear)/(front-rear))[keep],0,1)
    corrected_points=np.einsum("ij,ijk->ik",bary.reshape(-1,3)[idx],
                               corrected_vertices[faces[fid.flat[idx]]])
    color = fin_color(corrected_points,roof_color(s,t))
    edge_of_pillar = smoothstep(np.abs(x[keep]-350)-213,48)
    a=smoothstep(side_in[keep],5.5)*smoothstep(end_in[keep],5+29*edge_of_pillar)
    result=base.copy()
    old=base.reshape(-1,3)[idx].astype(float)
    result.reshape(-1,3)[idx]=np.uint8(np.clip(np.rint(old+a[:,None]*(color-old)),0,255))
    alpha=np.zeros((height,width),np.uint8)
    alpha.flat[idx]=np.maximum(1,np.rint(255*a)).astype('uint8')
    dist,near=distance_transform_edt(~valid,return_indices=True)
    pad=(~valid)&(~protected)&(dist<=12)&(alpha[tuple(near)]>0)
    result[pad]=result[near[0][pad],near[1][pad]]
    alpha[pad]=alpha[near[0][pad],near[1][pad]]
    assert np.array_equal(result[protected], base[protected])
    assert np.array_equal(result[alpha==0],base[alpha==0])
    return result,alpha,{'surface_texels':int(np.count_nonzero(alpha[valid])),
                         'gutter_texels':int(np.count_nonzero(pad)),
                         'changed_protected_texels':0}


def main():
    base=pilot.rgb(OUTPUTS/'bmw5_v3_daylight_glass_v1/texture.png')
    window_ids=np.array(Image.open(OUTPUTS/'bmw5_v3_opaque_glass_v1/window_ids.png'))
    hood=np.array(Image.open(OUTPUTS/'bmw5_v5_hood_refined_v3/hood_footprint.png').convert('L'))>0
    m=np.load(pilot.BASE/'textured_mesh.npz')
    v,f,uv=[m[k] for k in ('vertices','faces','uv')]
    nv=np.load(HERE/'roof_shape.npz')['vertices']
    tex,mask,report=bake(v,nv,f,uv,base,(window_ids>0)|hood)
    Image.fromarray(tex).save(HERE/'roof_texture.png')
    Image.fromarray(mask).save(HERE/'roof_mask.png')
    (HERE/'appearance_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,cam,shape in [('top',CAM,CHART_SHAPE),
                           ('rear',pilot.chart(155,47,860,440,210),(690,880))]:
        im=pilot.render_texture(pilot.render,nv,f,uv,tex,cam,shape[1],shape[0])
        im.save('/tmp/bmwroof_newpaint_'+name+'.png')
    print(report)


if __name__=='__main__':
    main()
