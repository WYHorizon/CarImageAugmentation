"""Opaque daylight automotive glazing on the v3 body, for standard glTF viewers."""
import copy
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDIO = ROOT / "bmw5_v3_studio_glass_v1"
spec_studio=importlib.util.spec_from_file_location("studio_base",STUDIO/"build_glass.py")
studio=importlib.util.module_from_spec(spec_studio)
spec_studio.loader.exec_module(studio)
from replace_frozen_v5_texture import parse_glb

pilot = studio.pilot


def sticker_alpha(u, v):
    a = np.array(Image.open(pilot.DONORS / "bmw5_front_glass_mask_v2/sticker_and_cover_protection.png").convert("L"))
    a[320:] = 0
    a[:,619:] = 0
    return pilot.sample(cv2.GaussianBlur(a.astype(np.float32)/255, (3,3), .45), u, v)


def local_coords(label, points):
    rear = studio.rear_side
    if label < 3:
        name = ("front","side")[label-1]
        cam = pilot.CHARTS[name]
        shape = pilot.SHAPES[name]
        paths = [pilot.FRONT] if label == 1 else pilot.SIDE
    else:
        cam,shape,paths = ((rear.SIDE_CAMERA, rear.SIDE_SHAPE, rear.SIDE_PANES) if label == 3
                           else (rear.REAR_CAMERA, rear.REAR_SHAPE, [rear.REAR_PANE]))
    x,y,_,_ = pilot.project(points,cam)
    edge = pilot.outline(shape,paths,[3,2,2.25,2.25][label-1])
    alpha = pilot.sample(edge,x,y)
    # Distance to the real chrome/rubber contour, in units of pane height.
    distance=distance_transform_edt(edge>0).astype(np.float32)
    margin=pilot.sample(distance,x,y)/[305,110,110,258][label-1]
    if label == 1:
        u,v,s,t = studio.front_side.sticker_coordinates(x,y)
        alpha *= 1-sticker_alpha(u,v)
        s = (s+1)/2
    elif label == 2:
        s = (x-254)/617
        roof = np.interp(x,[254,308,380,455,545,641,731,798,861],[211,177,152,144,141,141,147,159,179])
        sill = np.interp(x,[254,278,466,545,701,798,871],[259,258,248,248,244,239,230])
        t = (y-roof)/(sill-roof)
        divider = np.interp(y,[150,190,242],[761.8,756.6,749.8])
        alpha *= 1-np.clip(5.4-abs(x-divider),0,1)
    elif label == 3:
        s = (755-x)/625
        roof=np.interp(x,[131,190,246,301,383,455,533,633,697,756],[185,165,152,147,143,144,146,156,180,215])
        sill=np.interp(x,[131,190,259,383,472,540,679,756],[233,240,243,247,249,252,258,260])
        t=(y-roof)/(sill-roof)
    else:
        s,t = (x-244)/374,(y-230)/260
    return np.clip(s,0,1), np.clip(t,0,1), alpha, margin


def daylight(s,t,label,margin):
    # A brighter, gently asymmetric silver-blue base. Sharper grazing sheen
    # is supplied by the dielectric material and its view-dependent response.
    if label==1:
        lo,hi=[100,119,130],[168,184,192]
        horizon=.50+.16*s
    elif label==4:
        lo,hi=[94,112,122],[161,177,187]
        horizon=.48+.15*s
    else:
        lo,hi=[85,103,114],[157,173,183]
        horizon=.47+.22*np.cos(np.pi*(s-.12))
    sky=.5+.5*np.tanh((horizon-t)/.22)
    out=np.asarray(lo)+sky[:,None]*(np.asarray(hi)-lo)
    sweep=.5+.5*np.cos(np.pi*(s-.31*t-.18))
    # A diffuse daylight ribbon turns across the crowned pane. Its breadth,
    # brightness and curvature fade progressively instead of forming a line.
    center=.17+.40*s+.17*(s-.48)**2
    band=np.exp(-((t-center)/(.15+.10*s))**2)
    rim=(.55+.45*t)*np.exp(-(margin/.025)**2)
    overall=6*sweep + (10+8*(1-s))*band - 4.4*(2*s-1)**6 - 23*rim
    out+=overall[:,None]
    return out


def fit_pane_normals(points, targets, orientation):
    # Fit one smooth, weakly curved sheet per window orientation. A robust
    # quadratic removes capture dents and UV seams from the specular response.
    center=np.mean(points,axis=0)
    axis=np.linalg.eigh(np.cov((points-center).T))[1][:,0]
    axis*=np.sign(axis@orientation)
    horizontal=np.array([orientation[2],0,-orientation[0]],float)
    horizontal-=axis*(horizontal@axis)
    horizontal/=np.linalg.norm(horizontal)
    up=np.cross(axis,horizontal)
    basis=np.stack([horizontal,up,axis],axis=1)
    x,y,z=((points-center)@basis).T
    matrix=np.column_stack([np.ones_like(x),x,y,x*x,x*y,y*y])
    keep=np.ones(len(x),bool)
    for _ in range(3):
        coeff=np.linalg.lstsq(matrix[keep],z[keep],rcond=None)[0]
        error=np.abs(matrix@coeff-z)
        keep=error<max(.0025,3.5*np.median(error))
    x,y,_=((targets-center)@basis).T
    _,a,b,c,d,e=coeff
    dx=np.clip(a+2*c*x+d*y,-.5,.5)
    dy=np.clip(b+d*x+2*e*y,-.5,.5)
    normals=np.column_stack([-dx,-dy,np.ones_like(dx)])@basis.T
    return normals/np.linalg.norm(normals,axis=1)[:,None]


def png(image):
    buf=io.BytesIO()
    Image.fromarray(image).save(buf,format="PNG")
    return buf.getvalue()


def pack(source, rgb_png, gloss_png, faces, triangle_labels, normals):
    tree,binary=parse_glb(source)
    # Compact the inherited GLB, whose historical revisions left old PNGs in
    # the binary; only live buffer views are needed.
    blob=bytearray()
    def add_bytes(data):
        blob.extend(bytes(-len(blob)%4))
        start=len(blob)
        blob.extend(data)
        return {"buffer":0,"byteOffset":start,"byteLength":len(data)}
    old_views=tree["bufferViews"]
    active_texture=tree["images"][0]["bufferView"]
    for index,view in enumerate(old_views):
        offset=view.get("byteOffset",0)
        content=rgb_png if index==active_texture else binary[offset:offset+view["byteLength"]]
        old_views[index]=dict(view,**add_bytes(content))
    def view(data,target=None):
        record=add_bytes(data)
        if target: record["target"]=target
        tree["bufferViews"].append(record)
        return len(tree["bufferViews"])-1
    def accessor(values,typ,component_type,target):
        vals=values.reshape(-1,3 if typ=="VEC3" else 1)
        record=dict(bufferView=view(values.tobytes(),target),componentType=component_type,
                    count=len(vals),type=typ,min=vals.min(0).tolist(),max=vals.max(0).tolist())
        tree["accessors"].append(record)
        return len(tree["accessors"])-1
    prim=tree["meshes"][0]["primitives"][0]
    glass=copy.deepcopy(prim)
    # Preserve triangle winding and group each triangle exactly once.
    prim["indices"]=accessor(faces[triangle_labels==0].astype("<u4"),"SCALAR",5125,34963)
    glass["indices"]=accessor(faces[triangle_labels>0].astype("<u4"),"SCALAR",5125,34963)
    glass["attributes"]["NORMAL"]=accessor(normals.astype("<f4"),"VEC3",5126,34962)
    glass["material"]=len(tree["materials"])
    tree["meshes"][0]["primitives"].append(glass)
    tree["images"].append({"mimeType":"image/png","bufferView":view(gloss_png),"name":"glass_specular_weight"})
    tree["textures"].append({"source":1})
    tree["materials"].append({
        "name":"Daylight automotive glass · opaque dielectric",
        "doubleSided":True, "alphaMode":"OPAQUE",
        "pbrMetallicRoughness":{"baseColorFactor":[0,0,0,1],"metallicFactor":0,"roughnessFactor":.16},
        "emissiveTexture":{"index":0}, "emissiveFactor":[1,1,1],
        "extensions":{"KHR_materials_specular":{"specularFactor":.62,"specularTexture":{"index":1}}}})
    tree["extensionsUsed"]=list(dict.fromkeys(tree.get("extensionsUsed",[])+["KHR_materials_specular"]))
    blob.extend(bytes(-len(blob)%4))
    tree["buffers"][0]["byteLength"]=len(blob)
    meta=json.dumps(tree,separators=(",",":")).encode()
    meta+=b" "*(-len(meta)%4)
    size=12+8+len(meta)+8+len(blob)
    return struct.pack("<4sII",b"glTF",2,size)+struct.pack("<II",len(meta),0x4e4f534a)+meta+struct.pack("<II",len(blob),0x004e4942)+blob


def main():
    ids=np.array(Image.open(studio.PREVIOUS/"window_ids.png"))
    texture=studio.read_rgb(STUDIO/"texture.png")
    updated=texture.copy()
    spec=np.zeros_like(ids)
    source=(STUDIO/"textured_mesh.glb").read_bytes()
    gltf,binary=parse_glb(source)
    def read_accessor(idx,dtype,components):
        acc=gltf["accessors"][idx]
        view=gltf["bufferViews"][acc["bufferView"]]
        start=view.get("byteOffset",0)+acc.get("byteOffset",0)
        return np.frombuffer(binary,dtype,count=acc["count"]*components,offset=start).reshape(-1,components)
    with np.load(pilot.BASE/"textured_mesh.npz") as m:
        vertices,faces,uv=(m[k] for k in ("vertices","faces","uv"))
    prim=gltf["meshes"][0]["primitives"][0]
    assert np.array_equal(faces,read_accessor(prim["indices"],"<u4",1).reshape(-1,3))
    assert np.array_equal(vertices,read_accessor(prim["attributes"]["POSITION"],"<f4",3))
    normals=read_accessor(prim["attributes"]["NORMAL"],"<f4",3).copy()
    h,w=ids.shape
    screen=np.column_stack((uv[:,0]*(w-1),(1-uv[:,1])*(h-1),np.ones(len(uv))))
    _,fid,bary=pilot.render(screen,faces,dict(projection="orthographic",focal_px=1,cx=0,cy=0),(h,w))
    tri=np.zeros(len(faces),np.uint8)
    measures={}
    for label,orient in enumerate(([0,.7,1],[1,.3,0],[-1,.3,0],[0,.8,-1]),1):
        area=ids==label
        f=fid[area]
        assert np.all(f>=0)
        pts=np.einsum("ij,ijk->ik",bary[area],vertices[faces[f]])
        s,t,alpha,margin=local_coords(label,pts)
        delta=daylight(s,t,label,margin)-studio.light_field(s,t,label)
        updated[area]=np.clip(np.rint(texture[area]+alpha[:,None]*delta),0,255).astype(np.uint8)
        spec[area]=np.rint(255*alpha).astype(np.uint8)
        used=np.unique(f)
        assert np.all(tri[used]==0)
        tri[used]=label
        indices=np.unique(faces[used])
        normals[indices]=fit_pane_normals(pts[alpha>.97][::6],vertices[indices],np.asarray(orient,float))
        measures[str(label)]=[np.median(v[area],axis=0).tolist() for v in [texture,updated]]
    pad=ids==5
    _,nearest=distance_transform_edt(fid<0,return_indices=True)
    for img in [updated,spec]: img[pad]=img[nearest[0][pad],nearest[1][pad]]
    assert np.array_equal(texture[ids==0],updated[ids==0])
    assert not np.any(spec[ids==0])
    rgb_png=png(updated)
    gloss_png=png(np.stack([spec,spec,spec,spec],axis=2))
    (HERE/"texture.png").write_bytes(rgb_png)
    final=HERE/"textured_mesh.glb"
    artifact=pack(source,rgb_png,gloss_png,faces,tri,normals)
    g,b=parse_glb(artifact)
    restored=[]
    for p in g["meshes"][0]["primitives"]:
        v=g["bufferViews"][g["accessors"][p["indices"]]["bufferView"]]
        restored.append(np.frombuffer(b[v["byteOffset"]:v["byteOffset"]+v["byteLength"]],"<u4").reshape(-1,3))
    assert np.array_equal(restored[0],faces[tri==0]) and np.array_equal(restored[1],faces[tri>0])
    assert np.array_equal(studio.embedded_rgb(artifact),updated)
    (HERE/"model.tmp").write_bytes(artifact)
    (HERE/"model.tmp").replace(final)
    report=dict(glb_sha256=studio.sha256(final),previous_glb_sha256=studio.sha256(STUDIO/"textured_mesh.glb"),
                shader="Opaque dielectric; opaque daylight-color baseline plus standard KHR_materials_specular; smooth pane normals",
                triangle_counts=dict(body=len(restored[0]),glass_and_boundary=len(restored[1])),
                no_light_texture_median_rgb_before_after=measures,triangles_exactly_once=True,
                shared_position_uv_accessors=True,changed_colors_outside_window_masks_and_gutters=0,
                specular_weight_outside_glass_and_gutters=0,transmission=0)
    (HERE/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
