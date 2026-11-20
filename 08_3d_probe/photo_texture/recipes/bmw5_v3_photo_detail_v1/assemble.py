"""Combine the photo-based local corrections with the accepted roof and glass."""
import argparse
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bmw5_v3_roof_refined_v1"
sys.path.insert(0, str(SOURCE))
from build_model import digest, parse_glb, read_accessor, rgb, view_bytes, write_glb


def insert_plate(tree, blocks):
    carrier = np.load(HERE / "plate_part.npz")
    def add_view(data, target=None):
        index = len(blocks)
        spec = {"buffer": 0}
        if target:
            spec["target"] = target
        tree["bufferViews"].append(spec)
        blocks.append(data)
        return index
    def accessor(array, fmt, typ, target):
        count = int(array.size) if typ == "SCALAR" else len(array)
        array = array.astype("<f4" if fmt == 5126 else "<u4")
        spec = dict(bufferView=add_view(array.tobytes(), target), componentType=fmt,
                    type=typ,count=count)
        if typ=="VEC3":
            spec.update(min=array.min(0).tolist(),max=array.max(0).tolist())
        tree["accessors"].append(spec)
        return len(tree["accessors"])-1
    image_idx=len(tree["images"])
    tree["images"].append({"mimeType":"image/png","bufferView":add_view((HERE/"plate_photo.png").read_bytes()),
                           "name":"Display plate from frontal source photograph"})
    texture_idx=len(tree["textures"])
    tree["textures"].append({"source":image_idx})
    mat_idx=len(tree["materials"])
    tree["materials"].append({"name":"Original BMW photo display plate",
        "pbrMetallicRoughness":{"baseColorTexture":{"index":texture_idx},"metallicFactor":0,"roughnessFactor":1},
        "extensions":{"KHR_materials_unlit":{}},"doubleSided":False})
    tree["meshes"][0]["primitives"].append({
        "material":mat_idx,"mode":4,
        "attributes":{"POSITION":accessor(carrier["vertices"],5126,"VEC3",34962),
                      "NORMAL":accessor(carrier["normals"],5126,"VEC3",34962),
                      "TEXCOORD_0":accessor(carrier["uv"],5126,"VEC2",34962)},
        "indices":accessor(carrier["faces"],5125,"SCALAR",34963)})


def verify_deformation(before, after, faces):
    moving = np.any(before != after, axis=1)
    touched = faces[moving[faces].any(axis=1)]
    def area_vectors(positions):
        a, b, c = positions[touched].transpose(1, 0, 2)
        return np.cross(b - a, c - a)
    a, b = area_vectors(before), area_vectors(after)
    flipped = np.sum(a * b, axis=1) < 0
    degenerate = (np.linalg.norm(a, axis=1) >= 1e-10) & (np.linalg.norm(b, axis=1) < 1e-10)
    assert not np.any(flipped) and not np.any(degenerate), "Local repair damaged the mesh"
    return {"affected_body_triangles": len(touched), "flipped": int(flipped.sum()),
            "new_degenerate": int(degenerate.sum())}


def build(dest, regions):
    raw=(SOURCE/"textured_mesh.glb").read_bytes()
    assert digest(raw)==json.loads((SOURCE/"verification.json").read_text())["glb_sha256"]
    old,binary=parse_glb(raw)
    tree=copy.deepcopy(old)
    blocks=[view_bytes(tree,binary,i) for i in range(len(tree["bufferViews"]))]
    body,glass=tree["meshes"][0]["primitives"]
    pidx,nidx=body["attributes"]["POSITION"],body["attributes"]["NORMAL"]
    v=read_accessor(old,binary,pidx); n=read_accessor(old,binary,nidx)
    vout,nout=v.copy(),n.copy()
    updated_v=np.zeros(len(v),bool)
    pane_vertices=np.unique(read_accessor(tree,binary,glass["indices"]))
    imidx=tree["images"][0]["bufferView"]
    tex=rgb(blocks[imidx]); result=tex.copy()
    pane=np.asarray(Image.open(HERE.parent/"bmw5_v3_opaque_glass_v1/window_ids.png"))!=0
    prior_hood=np.asarray(Image.open(HERE.parent/"bmw5_v5_hood_refined_v3/hood_footprint.png"))!=0
    support=np.zeros(tex.shape[:2],bool)
    stats={}
    for region in regions:
        values=np.array(Image.open(HERE/f"{region}_rgb.png").convert("RGB"))
        mask=np.asarray(Image.open(HERE/f"{region}_mask.png").convert("L"))!=0
        changed=np.any(values!=tex,axis=2)
        assert not np.any(changed & ~mask), f"{region} changes outside support"
        assert not np.any(changed & pane), f"{region} changes approved glass"
        overlap=changed & support
        if np.any(overlap):
            assert not np.any(np.any(result!=values,axis=2)&overlap), f"{region}: conflicting overlapping colour edits"
        result[changed]=values[changed]
        support|=changed
        stats[region]={"changed_atlas_texels_including_padding":int(changed.sum()),
                       "overlap_prior_hood_footprint":int((changed & prior_hood).sum())}
        fn=HERE/f"{region}_vertices.npz"
        if fn.exists():
            with np.load(fn) as delta:
                dst=np.asarray(delta["vertices"],dtype="<f4")
                nor=np.asarray(delta["normals"],dtype="<f4")
            assert dst.shape==v.shape and nor.shape==n.shape
            changed_v=np.any(dst!=v,axis=1)
            assert not changed_v[pane_vertices].any(), f"{region} moves glass"
            common=updated_v&changed_v
            if np.any(common):
                # The grille and mounting block meet along the same physical
                # strip. Both fits slide vertices on frontal camera rays.
                # Keep the deeper fitted surface there instead of adding the
                # two corrections and carving an artificial groove.
                assert region=="plate" and np.all(v[common,2]>.90)
                # Part of the grille correction pushes the chrome forward.
                # As the plate recess fades, carry that offset continuously
                # into the border instead of dropping it at the first vertex.
                forward=common & (vout[:,2]>v[:,2])
                fade=np.clip((v[forward,2]-dst[forward,2])/.008,0,1)
                retain=1-fade*fade*(3-2*fade)
                dst[forward]+=(vout[forward]-v[forward])*retain[:,None]
                changed_v=changed_v & (~common | (dst[:,2]<vout[:,2]))
                stats[region]["blended_front_mount_vertices"]=int(common.sum())
            nchange=np.any(nor!=n,axis=1)
            vout[changed_v]=dst[changed_v]; nout[nchange]=nor[nchange]
            updated_v|=changed_v
            stats[region]["moved_vertices"]=int(changed_v.sum())
    assert np.isfinite(vout).all() and np.isfinite(nout).all()
    _,first,inv=np.unique(v,axis=0,return_index=True,return_inverse=True)
    inv=inv.reshape(-1)
    assert np.array_equal(vout,vout[first][inv])
    body_faces=read_accessor(old,binary,body["indices"]).reshape(-1,3)
    geometry=verify_deformation(v,vout,body_faces)
    # Both fits meet at the plate mount. Recompute front normals from the
    # assembled surface so normals on the shared rim follow its final shape.
    if "fascia" in regions and "plate" in regions:
        face_ids=inv[body_faces]
        a,b,c=vout[body_faces].transpose(1,0,2)
        vectors=np.cross(b-a,c-a)
        sums=np.zeros_like(vout[first])
        for corners in face_ids.T:
            np.add.at(sums,corners,vectors)
        normals=sums[inv]
        lengths=np.linalg.norm(normals,axis=1)
        update=(v[:,2]>.90) & (lengths>1e-9) & (updated_v | np.any(n!=nout,axis=1))
        nout[update]=normals[update]/lengths[update,None]
    for ai,arr in ((pidx,vout),(nidx,nout)):
        a=tree["accessors"][ai];blocks[a["bufferView"]]=arr.tobytes()
        a.update(min=arr.min(0).tolist(),max=arr.max(0).tolist())
    buf=io.BytesIO();Image.fromarray(result).save(buf,format="PNG")
    blocks[imidx]=buf.getvalue()
    if "plate" in regions:insert_plate(tree,blocks)
    packed=write_glb(tree,blocks)
    checked,b=parse_glb(packed)
    for i,data in enumerate(blocks):assert view_bytes(checked,b,i)==data
    assert checked["meshes"][0]["primitives"][:2]==old["meshes"][0]["primitives"]
    assert checked["materials"][:2]==old["materials"]
    assert np.array_equal(rgb(view_bytes(checked,b,imidx)),result)
    assert np.array_equal(tex[pane],result[pane])
    assert np.array_equal(read_accessor(checked,b,pidx)[pane_vertices],v[pane_vertices])
    for aidx in [glass["indices"], *(a for key,a in glass["attributes"].items() if key!="POSITION")]:
        block_id=old["accessors"][aidx]["bufferView"]
        assert view_bytes(old,binary,block_id)==view_bytes(checked,b,block_id)
    if "fascia" in regions:
        from front_hd import apply_hd
        packed=apply_hd(packed)
    dest.mkdir(parents=True,exist_ok=True)
    tmp=dest/"model.tmp";tmp.write_bytes(packed);tmp.replace(dest/"textured_mesh.glb")
    (dest/"verification.json").write_text(json.dumps({
        "base_glb_sha256":digest(raw),"glb_sha256":digest(packed),"by_region":stats,
        "body_geometry":geometry,
        "previous_glass_positions_normals_material_and_texture_exact": True,
        "base_triangle_connectivity_retained":True,"protected_texture_outside_edit_regions_exact":True,
        "grille_and_headlamp_interiors_use_compact_original_photo_texture":"fascia" in regions,
        "display_plate_source":"5_2026_255841_151515_002.jpg, original-resolution frontal crop",
    },indent=2)+"\n")
    print((dest/"verification.json").read_text())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--regions",nargs="+",choices=["roof","fascia","plate"],default=["roof","fascia","plate"])
    args=parser.parse_args()
    build(args.output,args.regions)
