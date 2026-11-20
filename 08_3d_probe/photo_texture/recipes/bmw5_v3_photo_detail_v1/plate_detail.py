"""Restore the printed display plate from its frontal photograph on a single thin carrier."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
PROJECT = OUTPUTS.parents[2]
BASE = OUTPUTS / "bmw5_v3_roof_refined_v1"
sys.path.insert(0, str(OUTPUTS.parent))
from bake_texture import project, sample
from raster import render


def ramp(x):
    t = np.clip(x, 0, 1)
    return t*t*(3-2*t)


def main():
    original = Image.open(PROJECT / "test_images/宝马5系/front/5_2026_255841_151515_002.jpg").convert("RGB")
    photo = np.asarray(Image.open(OUTPUTS / "bmw5_inputs_v2/front_002/source_rgb.png").convert("RGB"))
    cameras = json.loads((OUTPUTS / "bmw5_mv_proportion_fit_v2/cameras.json").read_text())["views"]
    cam = next(r["camera"] for r in cameras if r["id"] == "front_002")
    # Very small real dark margin includes the plate's genuine rounded corners.
    rect = (673, 610, 868, 676)
    sx, sy = original.width/photo.shape[1], original.height/photo.shape[0]
    box = [round(k*scale) for k,scale in zip(rect,[sx,sy,sx,sy])]
    donor = original.crop(box)
    donor.save(HERE / "plate_photo.png")
    mesh = np.load(OUTPUTS / "bmw5_mv_textured_v5/textured_mesh.npz")
    faces, uv = mesh["faces"], mesh["uv"]
    shape = np.load(BASE / "roof_shape.npz")
    v, n = shape["vertices"], shape["normals"]
    tex = np.array(Image.open(BASE / "roof_texture.png").convert("RGB"))
    H,W = tex.shape[:2]
    screen = np.c_[uv[:,0]*(W-1), (1-uv[:,1])*(H-1), np.ones(len(uv))]
    _, fid, bary = render(screen, faces, dict(projection="orthographic",focal_px=1,cx=0,cy=0),(H,W))
    valid = fid >= 0
    idx = np.flatnonzero(valid)
    xyz = np.einsum("ij,ijk->ik",bary[valid],v[faces[fid[valid]]])
    del bary
    subset = (xyz[:,2]>.925) & (abs(xyz[:,0])<.13) & (xyz[:,1]<-.075) & (xyz[:,1]>-.17)
    idx, xyz = idx[subset], xyz[subset]
    x,y,_,_ = project(xyz,cam)
    boundary = np.minimum.reduce([x-652.,882.-x,y-608.,685.-y])
    inside = boundary>0
    idx,x,y,boundary = idx[inside],x[inside],y[inside],boundary[inside]
    a = ramp(boundary/6)
    # The carrier hides the middle of the original intake in the photograph.
    # Make its exposed sides/underside from real neighboring black plastic.
    # This explicitly removes yellow pixels/print on the old side-facing shells.
    left = sample(photo,np.full_like(x,664),y)
    right = sample(photo,np.full_like(x,878),y)
    weight = np.clip((x-664)/214,0,1)[:,None]
    backing = left*(1-weight)+right*weight
    clear = ramp(np.minimum.reduce([x-669.,873.-x,y-605.,681.-y])/4)[:,None]
    color = sample(photo,x,y)*(1-clear)+backing*clear
    flat = tex.reshape(-1,3)
    flat[idx] = np.uint8(np.rint(flat[idx]+a[:,None]*(color-flat[idx])).clip(0,255))
    mask = np.zeros((H,W),np.uint8)
    mask.flat[idx] = np.maximum(1,np.rint(255*a)).astype('uint8')
    dist, nn = distance_transform_edt(~valid,return_indices=True)
    pad = (~valid)&(dist<=12)&(mask[tuple(nn)]>0)
    tex[pad] = tex[nn[0][pad],nn[1][pad]]
    mask[pad] = mask[nn[0][pad],nn[1][pad]]
    Image.fromarray(tex).save(HERE/"plate_rgb.png")
    Image.fromarray(mask).save(HERE/"plate_mask.png")

    # Compress only the excess depth of the black mounting block. Keep its
    # anchoring geometry and use one surface for the lettering at every view.
    X,Y,Z = v.T
    # Ease the transition into the surrounding bumper over both plate edges.
    # A shorter upper transition folds the two narrow upper corner triangles.
    w = ramp((.136-abs(X))/.045)*ramp((Y+.173)/.035)*ramp((-.031-Y)/.035)
    camera_center=-np.asarray(cam["R"]).T@cam["t"]
    rays=v-camera_center
    shift=-.53*w*np.maximum(Z-.949,0)
    new_v = (v+rays*(shift/rays[:,2])[:,None]).astype(v.dtype)
    change = np.any(new_v!=v,axis=1)
    unique, first, inverse = np.unique(v,axis=0,return_index=True,return_inverse=True)
    inverse=inverse.reshape(-1)
    assert np.array_equal(new_v,new_v[first][inverse])
    triangles=new_v[faces]
    areas=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    sums=np.zeros_like(unique)
    for j in range(3):
        np.add.at(sums,inverse[faces[:,j]],areas)
    smooth=sums[inverse]
    new_n=n.copy()
    ok=change&(np.linalg.norm(smooth,axis=1)>1e-9)
    new_n[ok]=smooth[ok]/np.linalg.norm(smooth[ok],axis=1)[:,None]
    # Backproject the donor corners onto the seated plate plane. Pixel half
    # offsets account for the high-resolution crop and fitted photo convention.
    l,t,r,b=[edge/scale-.5 for edge,scale in zip(box,[sx,sy,sx,sy])]
    xy=np.array([[l,t],[r,t],[r,b],[l,b]])
    R=np.array(cam["R"]); center=-R.T@cam["t"]
    rays=np.c_[(xy-[cam['cx'],cam['cy']])/cam['focal_px'],np.ones(4)]@R
    depths=np.asarray([.9754,.9754,.9758,.9758])
    quad=center+rays*((depths-center[2])/rays[:,2])[:,None]
    # Six faces, including a very thin dark edge and back. The edge samples
    # the physical black outer margin of the source crop.
    thin=.00085
    pts=np.vstack((quad,quad-[0,0,thin]))
    sides=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]]
    corner_uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    p_positions=[];p_uv=[];p_normals=[];p_faces=[]
    for j,side in enumerate(sides):
        q=pts[side]
        normal=np.cross(q[1]-q[0],q[2]-q[0]);normal/=np.linalg.norm(normal)
        p_positions.extend(q);p_normals.extend([normal]*4)
        p_uv.extend(corner_uv[side] if j==0 else [[.002,.003]]*4)
        p_faces.extend([[4*j,4*j+1,4*j+2],[4*j,4*j+2,4*j+3]])
    np.savez_compressed(HERE/"plate_part.npz",vertices=np.asarray(p_positions,np.float32),
                        uv=np.asarray(p_uv,np.float32),normals=np.asarray(p_normals,np.float32),
                        faces=np.asarray(p_faces,np.uint32))
    np.savez_compressed(HERE/"plate_vertices.npz",vertices=new_v,normals=new_n)
    print("carrier moved vertices",int(change.sum()),"atlas texels incl. gutter",int((mask>0).sum()),
          "source crop",box,donor.size,"photo digest",hashlib.sha256(donor.tobytes()).hexdigest())


if __name__=="__main__":
    main()
