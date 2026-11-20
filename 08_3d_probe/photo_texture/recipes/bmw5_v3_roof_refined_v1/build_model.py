"""Install the local roof shape and colour repair into the approved daylight GLB."""
import copy
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
SOURCE = OUTPUTS / "bmw5_v3_daylight_glass_v1"
sys.path.insert(0, str(OUTPUTS.parents[2] / "09_reflection_cleanup"))
from replace_frozen_v5_texture import parse_glb


def digest(data):
    return hashlib.sha256(data).hexdigest()


def view_bytes(tree, binary, index):
    view = tree["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return binary[start:start + view["byteLength"]]


def read_accessor(tree, binary, index):
    a = tree["accessors"][index]
    width = {"SCALAR": 1, "VEC3": 3}[a["type"]]
    dtype = {5126: "<f4", 5125: "<u4"}[a["componentType"]]
    assert "byteStride" not in tree["bufferViews"][a["bufferView"]]
    return np.frombuffer(view_bytes(tree, binary, a["bufferView"]), dtype=dtype,
                         count=a["count"] * width, offset=a.get("byteOffset", 0)).reshape(-1, width)


def write_glb(tree, blocks):
    binary = bytearray()
    for spec, data in zip(tree["bufferViews"], blocks, strict=True):
        binary.extend(bytes(-len(binary) % 4))
        spec.update(byteOffset=len(binary), byteLength=len(data))
        binary.extend(data)
    binary.extend(bytes(-len(binary) % 4))
    tree["buffers"][0]["byteLength"] = len(binary)
    meta = json.dumps(tree, separators=(",", ":"), ensure_ascii=True).encode()
    meta += b" " * (-len(meta) % 4)
    return b"".join([struct.pack("<4sII", b"glTF", 2, len(meta) + len(binary) + 28),
                     struct.pack("<II", len(meta), 0x4E4F534A), meta,
                     struct.pack("<II", len(binary), 0x004E4942), binary])


def rgb(png):
    with Image.open(io.BytesIO(png)) as img:
        assert img.mode == "RGB"
        return np.array(img)


def main():
    base = (SOURCE / "textured_mesh.glb").read_bytes()
    assert digest(base) == json.loads((SOURCE / "verification.json").read_text())["glb_sha256"]
    original, binary = parse_glb(base)
    tree = copy.deepcopy(original)
    body, glass = tree["meshes"][0]["primitives"]
    assert body["attributes"]["POSITION"] == glass["attributes"]["POSITION"]
    p_idx = body["attributes"]["POSITION"]
    n_idx = body["attributes"]["NORMAL"]
    assert n_idx != glass["attributes"]["NORMAL"]
    old_pos = read_accessor(tree, binary, p_idx)
    old_norm = read_accessor(tree, binary, n_idx)
    with np.load(HERE / "roof_shape.npz") as shape:
        pos = np.asarray(shape["vertices"], dtype="<f4")
        norm = np.asarray(shape["normals"], dtype="<f4")
    assert pos.shape == norm.shape == old_pos.shape == old_norm.shape
    assert np.isfinite(pos).all() and np.isfinite(norm).all()
    moved = np.any(old_pos != pos, axis=1)
    assert 0 < moved.sum() < len(pos) / 4
    # Duplicates on different UV islands must continue meeting at the same point.
    _, first, inverse = np.unique(old_pos, axis=0, return_index=True, return_inverse=True)
    assert np.array_equal(pos, pos[first][inverse.reshape(-1)])
    glass_verts = np.unique(read_accessor(tree, binary, glass["indices"]))
    assert not np.any(moved[glass_verts]), "A daylight glass triangle moved"
    assert np.min(old_pos[moved, 1]) > .12, "Unexpected displacement beneath the roof"
    blocks = [view_bytes(original, binary, i) for i in range(len(tree["bufferViews"]))]
    for idx, array in [(p_idx, pos), (n_idx, norm)]:
        a = tree["accessors"][idx]
        assert a.get("byteOffset", 0) == 0 and array.nbytes == len(blocks[a["bufferView"]])
        blocks[a["bufferView"]] = array.tobytes()
        a.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())

    image_idx = tree["images"][0]["bufferView"]
    prev_rgb = rgb(blocks[image_idx])
    new_png = (HERE / "roof_texture.png").read_bytes()
    new_rgb = rgb(new_png)
    mask = np.asarray(Image.open(HERE / "roof_mask.png").convert("L")) > 0
    assert mask.shape == new_rgb.shape[:2] == prev_rgb.shape[:2]
    changed = np.any(prev_rgb != new_rgb, axis=2)
    assert changed.any() and not np.any(changed & ~mask)
    protected = np.asarray(Image.open(OUTPUTS / "bmw5_v3_opaque_glass_v1/window_ids.png")) != 0
    protected |= np.asarray(Image.open(OUTPUTS / "bmw5_v5_hood_refined_v3/hood_footprint.png")) != 0
    assert not np.any(changed & protected), "Glass or hood changed"
    blocks[image_idx] = new_png
    packed = write_glb(tree, blocks)
    checked, new_binary = parse_glb(packed)
    for key in ("meshes", "textures", "images", "materials", "nodes", "scenes"):
        assert original[key] == checked[key]
    edited_views = {image_idx, tree["accessors"][p_idx]["bufferView"], tree["accessors"][n_idx]["bufferView"]}
    for i in range(len(blocks)):
        actual = view_bytes(checked, new_binary, i)
        assert actual == blocks[i]
        if i not in edited_views:
            assert actual == view_bytes(original, binary, i)

    f = read_accessor(checked, new_binary, body["indices"]).reshape(-1, 3)
    revised = f[moved[f].any(axis=1)]
    def crosses(points):
        a, b, c = points[revised].transpose(1, 0, 2)
        return np.cross(b - a, c - a)
    old_cross, new_cross = crosses(old_pos), crosses(pos)
    area = np.linalg.norm(new_cross, axis=1)
    report = {
        "base_glb_sha256": digest(base), "glb_sha256": digest(packed),
        "roof_texture_sha256": digest(new_png), "positions_sha256": digest(pos.tobytes()),
        "coincident_uv_seam_vertices_still_coincident": True,
        "changed_vertices": int(moved.sum()), "total_vertices": len(pos),
        "original_changed_bounds": [old_pos[moved].min(axis=0).tolist(), old_pos[moved].max(axis=0).tolist()],
        "maximum_vertex_displacement": float(np.linalg.norm(pos[moved]-old_pos[moved], axis=1).max()),
        "affected_triangles": len(revised),
        "changed_faces_turning_over_90_deg": int((np.sum(old_cross * new_cross, axis=1) < 0).sum()),
        "new_near_zero_area_faces": int(((area < 1e-10) & (np.linalg.norm(old_cross,axis=1) >= 1e-10)).sum()),
        "changed_texels_including_padding": int(changed.sum()),
        "glass_positions_normals_material_specular_map_exact": True,
        "glass_and_hood_texture_pixels_exact": True,
        "indices_uv_and_nonroof_texture_exact": True,
    }
    tmp = HERE / "textured_mesh.glb.tmp"
    tmp.write_bytes(packed)
    tmp.replace(HERE / "textured_mesh.glb")
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
