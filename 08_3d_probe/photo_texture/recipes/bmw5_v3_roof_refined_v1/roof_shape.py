"""Local rear roof-band reconstruction using the accepted BMW5 mesh and its top projection."""
from pathlib import Path
import json
import struct
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
PHOTO = HERE.parents[1]
sys.path.insert(0, str(PHOTO))
from raster import render


def ramp(value):
    x = np.clip(value, 0, 1)
    return x*x*(3-2*x)


def fin_height(x, z):
    """Reconstruct the centered single antenna from its side and rear silhouettes."""
    x, z = np.broadcast_arrays(x, z)
    t = np.clip((z + .477) / .077, 0, 1)  # rear tip to front tip
    profile = np.interp(t, [0,.08,.20,.33,.56,.78,.92,1], [0,.008,.018,.020,.016,.009,.0034,0])
    width = np.interp(t, [0,.12,.28,.53,.78,1], [.008,.016,.018,.017,.012,.002])
    cross = np.sqrt(np.maximum(0, 1 - np.square(np.abs(x) / width)))
    return profile*cross


def projection(vertices, faces):
    scale, cx, cy = 1600., 430., 838.
    cam = dict(projection='orthographic', R=[[1,0,0],[0,0,1],[0,-1,0]],
               t=[0,0,3], focal_px=scale, cx=cx, cy=cy)
    d, _, _ = render(vertices, faces, cam, (336, 864), False)
    return 3-d, scale, cx, cy


def target_heights(vertices, faces, samples):
    h, scale, cx, cy = projection(vertices, faces)
    x, y, z = samples.T
    xs = (np.arange(h.shape[1])-cx)/scale
    zs = (np.arange(h.shape[0])-cy)/scale
    # The intact strips between the duplicated central fittings and the rain
    # channels constrain the cross-car arch separately for each axial station.
    good = (abs(xs) >= .112) & (abs(xs) <= .188)
    along = xs/.18
    basis = np.stack([np.ones_like(along), along, along**2], axis=1)
    coeff = np.linalg.lstsq(basis[good], h[:,good].T, rcond=None)[0].T
    coeff = gaussian_filter1d(coeff, sigma=3, axis=0)
    surface = (coeff @ basis.T).astype(np.float32)
    coordinates = [z*scale+cy, x*scale+cx]
    arch = map_coordinates(surface, coordinates, order=1, mode='nearest')
    mid = ramp((.132-abs(x))/.023)*ramp((z+.487)/.012)*ramp((-.383-z)/.02)
    # Two small ghost fittings nearer the roof rails need the local directed
    # curvature of the rail. Interpolate each longitudinal profile from
    # uncontaminated rows at either end of the elevated patch.
    row_z = np.array([-.485,-.468,-.455,-.442,-.438,-.347,-.338,-.327,-.32])
    rows = np.rint(row_z*scale+cy).astype(int)
    usable = np.abs(xs)<.24
    side_field = np.zeros_like(h)
    side_field[:,usable] = PchipInterpolator(zs[rows],h[rows][:,usable],axis=0)(zs)
    rim = ramp((abs(x)-.192)/.013)*ramp((.237-abs(x))/.008)
    rim *= ramp((z+.442)/.008)*ramp((-.347-z)/.01)
    rim_height = map_coordinates(side_field, coordinates, order=1, mode='nearest')
    central_target = arch + fin_height(x,z)
    final_y = y + mid*(central_target-y) + rim*(rim_height-y)
    weight = np.maximum(mid,rim)
    # No penetration to floor, through the glass, or underside of the body.
    top = map_coordinates(np.nan_to_num(h,neginf=-10),coordinates,order=1,mode='nearest')
    weight[(y<.238) | (y<top-.035)] = 0
    return final_y, weight


def unfold_roof_xy(positions, triangles, allowed):
    """Use a positive mean-value harmonic map on the original folded patch.

    Old overhanging fin faces cannot be flattened by editing height alone.
    Hold every boundary coordinate fixed, and redistribute interior face
    samples smoothly across the roof's horizontal projection before fitting y.
    """
    pairs=[]; weights=[]
    for corner in range(3):
        target=triangles[:,corner]
        a,b=triangles[:,(corner+1)%3],triangles[:,(corner+2)%3]
        u,v=positions[a]-positions[target],positions[b]-positions[target]
        lu,lv=np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1)
        cosine=(u*v).sum(axis=1)/np.maximum(lu*lv,1e-24)
        half_tan=np.sqrt((1-np.clip(cosine,-.99999,.99999))/(1+np.clip(cosine,-.99999,.99999)))
        for end,length in [(a,lu),(b,lv)]:
            valid=(length>1e-10)&allowed[target]
            pairs.append((target[valid],end[valid]))
            weights.append(half_tan[valid]/length[valid])
    a,b=[np.concatenate(x) for x in zip(*pairs)]
    mat=coo_matrix((np.concatenate(weights),(a,b)),shape=(len(positions),len(positions))).tocsr()
    laplacian=diags(np.asarray(mat.sum(axis=1)).ravel())-mat
    interior=np.where(allowed)[0]
    exterior=np.where(~allowed)[0]
    A=laplacian[interior][:,interior]
    rhs=mat[interior][:,exterior]@positions[exterior][:,[0,2]]
    updated=positions.copy()
    updated[np.ix_(interior,[0,2])]=spsolve(A,rhs)
    return updated


def read_mesh_data():
    path=HERE.parent/'bmw5_v3_daylight_glass_v1/textured_mesh.glb'
    data=path.read_bytes()
    n=struct.unpack_from('<I',data,12)[0]
    tree=json.loads(data[20:20+n])
    binary=memoryview(data)[28+n:]
    def accessor(i,typ,dim):
        a=tree['accessors'][i]
        view=tree['bufferViews'][a['bufferView']]
        return np.frombuffer(binary,dtype=typ,count=a['count']*dim,
            offset=view.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,dim).copy()
    body,glass=tree['meshes'][0]['primitives']
    return accessor(body['attributes']['NORMAL'],'<f4',3),accessor(glass['indices'],'<u4',1).ravel()


def apply(vertices,faces,uv,body_normals=None):
    original_normals, glass_indices = read_mesh_data()
    body_normals = original_normals if body_normals is None else body_normals
    unique,inverse=np.unique(vertices,axis=0,return_inverse=True)
    unique=unique.astype(np.float64)
    # Numeric rasterizer accepts the extant original indexing.
    _,support=target_heights(vertices,faces,unique)
    allowed=support>0
    allowed[inverse[glass_indices]]=False
    new_unique=unfold_roof_xy(unique,inverse[faces],allowed)
    target, _=target_heights(vertices,faces,new_unique)
    new_unique[allowed,1]=target[allowed]
    result=new_unique[inverse].astype(vertices.dtype)
    changed=np.any(result!=vertices,axis=1)
    assert not np.any(changed[glass_indices])
    active_faces=faces[np.any(changed[faces],axis=1)]
    old_corners=vertices[active_faces].astype(np.float64)
    new_corners=result[active_faces].astype(np.float64)
    old_cross=np.cross(old_corners[:,1]-old_corners[:,0],old_corners[:,2]-old_corners[:,0])
    new_cross=np.cross(new_corners[:,1]-new_corners[:,0],new_corners[:,2]-new_corners[:,0])
    assert np.all(new_cross[:,1]>0), "Roof has down-facing or collapsed triangles"
    assert np.all(np.einsum('ij,ij->i',old_cross,new_cross)>0), "Local triangle inverted"
    # Recompute the body shading normals only in the deformed patch (including
    # UV-split copies). Neighboring unchanged positions supply the edge tangents.
    uf=inverse[faces]
    e1=new_unique[uf[:,1]]-new_unique[uf[:,0]]
    e2=new_unique[uf[:,2]]-new_unique[uf[:,0]]
    face_normals=np.cross(e1,e2)
    accum=np.zeros_like(new_unique)
    for i in range(3):
        np.add.at(accum,uf[:,i],face_normals)
    n=accum/np.maximum(np.linalg.norm(accum,axis=1,keepdims=True),1e-12)
    new_n=body_normals.copy()
    good=changed&(np.linalg.norm(n[inverse],axis=1)>.99)
    new_n[good]=n[inverse[good]].astype(new_n.dtype)
    return dict(vertices=result, normals=new_n, changed_vertices=changed,
        fin_vertices=changed&(fin_height(result[:,0],result[:,2])>.0005))


if __name__ == '__main__':
    raw=np.load(HERE.parent/'bmw5_mv_textured_v5/textured_mesh.npz')
    v,f,uv=(raw[k] for k in ('vertices','faces','uv'))
    data=apply(v,f,uv)
    diff=data['vertices'][:,1]-v[:,1]
    sel=data['changed_vertices']
    print('changed',int(sel.sum()),'range',v[sel].min(0),v[sel].max(0),
          'height shifts',np.percentile(diff[sel],[0,10,50,90,100]),'fin vertices',int(data['fin_vertices'].sum()))
    np.savez_compressed(HERE/'roof_shape.npz',**data)
