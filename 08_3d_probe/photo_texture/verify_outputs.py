"""Independently check a baked asset's files, cameras and source provenance.

Does not import the baker or rasterizer, re-fit cameras, or run a model. Passing
these checks establishes artifact consistency, not geometric/identity accuracy.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import trimesh


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hash(path, expected):
    require(sha256(path) == expected, f"SHA256 mismatch: {path}")


def read_png(path, size=None, mode=None):
    with Image.open(path) as opened:
        require(opened.format == "PNG", f"Not a PNG: {path}")
        opened.verify()
    with Image.open(path) as opened:
        opened.load()
        if size is not None:
            require(opened.size == tuple(size), f"Wrong image size: {path}")
        if mode is not None:
            require(opened.mode == mode, f"Wrong image mode: {path}")
        return np.asarray(opened).copy()


def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    require(magic == 0x46546C67 and version == 2 and length == len(data), "Invalid GLB header")
    offset, tree, binary = 12, None, None
    while offset < len(data):
        chunk_size, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_size]
        require(len(chunk) == chunk_size, "Truncated GLB chunk")
        if chunk_type == 0x4E4F534A:
            tree = json.loads(chunk)
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += chunk_size
    require(offset == len(data) and tree is not None and binary is not None, "Missing GLB chunks")
    require(len(tree["buffers"]) == 1 and "uri" not in tree["buffers"][0], "Expected one embedded GLB buffer")
    return tree, binary


def accessor(tree, binary, index):
    descriptor = tree["accessors"][index]
    require("sparse" not in descriptor and not descriptor.get("normalized", False), "Unexpected sparse/normalized accessor")
    dimensions = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    columns = dimensions[descriptor["type"]]
    dtype = np.dtype(types[descriptor["componentType"]])
    view = tree["bufferViews"][descriptor["bufferView"]]
    require(view.get("buffer", 0) == 0, "Accessor references an external buffer")
    row_size = columns * dtype.itemsize
    stride = view.get("byteStride", row_size)
    offset = view.get("byteOffset", 0) + descriptor.get("byteOffset", 0)
    required = descriptor.get("byteOffset", 0) + (descriptor["count"] - 1) * stride + row_size
    require(required <= view["byteLength"], "Accessor exceeds buffer view")
    return np.ndarray((descriptor["count"], columns), dtype=dtype, buffer=binary,
                      offset=offset, strides=(stride, dtype.itemsize)).copy()


def triangle_keys(vertices, faces):
    """Compare triangle multisets independent of UV seam duplication/order."""
    triangles = np.asarray(vertices, np.float32)[faces]
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    ordered = np.take_along_axis(triangles, order[:, :, None], axis=1)
    rows = np.ascontiguousarray(ordered.reshape(-1, 9), dtype=np.float32)
    return np.sort(rows.view(np.dtype((np.void, rows.dtype.itemsize * 9))).reshape(-1))


def check_camera(camera, width, height):
    require(width > 0 and height > 0, "Invalid camera image size")
    require(camera.get("projection", "perspective") == "perspective", "Expected perspective camera")
    R, t = np.asarray(camera["R"]), np.asarray(camera["t"])
    require(R.shape == (3, 3) and t.shape == (3,) and np.isfinite(R).all() and np.isfinite(t).all(), "Invalid camera R/t")
    require(np.allclose(R @ R.T, np.eye(3), atol=1e-6, rtol=0), "Camera rotation is not orthogonal")
    require(abs(np.linalg.det(R) - 1) < 1e-6, "Camera rotation is reflected or singular")
    f = np.asarray(camera["focal_px"])
    require(f.shape in ((), (2,)) and np.isfinite(f).all() and (f > 0).all(), "Invalid focal length")
    require(np.isfinite([camera["cx"], camera["cy"]]).all(), "Invalid principal point")
    require(camera.get("near", 1e-4) > 0, "Invalid near plane")
    require(np.isfinite(-R.T @ t).all(), "Nonfinite camera center")


def verify(output, fit, prepared):
    manifest = json.loads((output / "manifest.json").read_text())
    cameras = json.loads((fit / "cameras.json").read_text())
    sources = json.loads((prepared / "manifest.json").read_text())
    require(str(manifest["status"]).startswith("complete"), "Baking is incomplete")
    require(cameras["status"] == "complete", "Camera fitting is incomplete")
    check_hash(fit / "canonical_mesh.glb", manifest["mesh_sha256"])
    check_hash(fit / "canonical_mesh.glb", cameras["canonical_mesh_sha256"])
    check_hash(fit / "cameras.json", manifest["cameras_sha256"])
    check_hash(prepared / "manifest.json", manifest["prepared_manifest_sha256"])
    check_hash(Path(cameras["mesh_path"]), cameras["mesh_sha256"])
    check_hash(output / "bake_script_snapshot.py", manifest["script_sha256"])
    check_hash(output / "textured_mesh.glb", manifest["mesh"]["glb_sha256"])

    refs = {item["id"]: item for item in sources["views"]}
    require(len(refs) == len(sources["views"]), "Duplicate prepared view IDs")
    prepared_files = 0
    for ref in refs.values():
        check_hash(ref["source_path"], ref["source_sha256"])
        for item in ref["files"].values():
            path = prepared / item["path"]
            check_hash(path, item["sha256"])
            if path.suffix == ".png":
                read_png(path, ref["working_size"])
                prepared_files += 1
    view_map = {item["id"]: item for item in cameras["views"]}
    require(len(view_map) == len(cameras["views"]), "Duplicate camera view IDs")
    for item in view_map.values():
        check_camera(item["camera"], item["width"], item["height"])
        require([item["width"], item["height"]] == refs[item["id"]]["working_size"], "Camera/prepared resolution mismatch")
    for item in manifest["sources"]:
        ref = refs[item["id"]]
        require(item["id"] in view_map, "Baked source has no fitted camera")
        require(item["source_sha256"] == ref["source_sha256"], "Baker/preparation source hash mismatch")
        check_hash(prepared / ref["files"]["source_rgb.png"]["path"], item["rgb_sha256"])
        check_hash(prepared / ref["files"]["mask_geometry.png"]["path"], item["geometry_mask_sha256"])

    with np.load(output / "textured_mesh.npz", allow_pickle=False) as arrays:
        vertices, faces, uv = (arrays[key] for key in ("vertices", "faces", "uv"))
    require(vertices.shape == (manifest["mesh"]["vertices"], 3), "Wrong NPZ vertex count")
    require(faces.shape == (manifest["mesh"]["faces"], 3), "Wrong NPZ face count")
    require(uv.shape == (len(vertices), 2), "Wrong NPZ UV shape")
    require(np.isfinite(vertices).all() and np.isfinite(uv).all(), "Nonfinite geometry/UV")
    require(faces.dtype.kind in "iu" and faces.min() >= 0 and faces.max() < len(vertices), "Invalid mesh indices")
    require(uv.min() >= -1e-7 and uv.max() <= 1 + 1e-7, "UV coordinates outside atlas")
    canonical = trimesh.load(fit / "canonical_mesh.glb", force="mesh", process=False)
    require(np.array_equal(triangle_keys(vertices, faces), triangle_keys(canonical.vertices, canonical.faces)), "Baked mesh changes canonical geometry")

    tree, binary = read_glb(output / "textured_mesh.glb")
    require(len(tree["meshes"]) == 1 and len(tree["meshes"][0]["primitives"]) == 1, "Expected one mesh/primitive")
    primitive = tree["meshes"][0]["primitives"][0]
    require(primitive.get("mode", 4) == 4, "GLB is not triangle geometry")
    glb_vertices = accessor(tree, binary, primitive["attributes"]["POSITION"])
    glb_faces = accessor(tree, binary, primitive["indices"]).reshape(-1, 3)
    glb_uv = accessor(tree, binary, primitive["attributes"]["TEXCOORD_0"])
    require(np.array_equal(glb_vertices, vertices) and np.array_equal(glb_faces, faces), "GLB/NPZ geometry mismatch")
    expected_uv = uv.copy()
    expected_uv[:, 1] = 1 - expected_uv[:, 1]
    require(np.allclose(glb_uv, expected_uv, rtol=0, atol=1e-7), "GLB UVs do not use glTF image origin")
    normals = accessor(tree, binary, primitive["attributes"]["NORMAL"])
    require(normals.shape == vertices.shape and np.isfinite(normals).all(), "Invalid GLB normals")
    for node in tree.get("nodes", []):
        if "matrix" in node:
            require(np.allclose(np.array(node["matrix"]).reshape(4, 4), np.eye(4)), "Unexpected GLB node matrix")
        for key, default in (("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]), ("scale", [1, 1, 1])):
            require(np.allclose(node.get(key, default), default), f"Unexpected GLB node {key}")
    material = tree["materials"][primitive["material"]]
    require("KHR_materials_unlit" in material.get("extensions", {}) and "KHR_materials_unlit" in tree.get("extensionsUsed", []), "Missing unlit material extension")
    require(material.get("doubleSided") is True, "Expected double-sided material")
    pbr = material["pbrMetallicRoughness"]
    require(np.allclose(pbr.get("baseColorFactor", [1, 1, 1, 1]), [1, 1, 1, 1]), "Unexpected material color multiplier")
    texture_id = pbr["baseColorTexture"]["index"]
    image = tree["images"][tree["textures"][texture_id]["source"]]
    image_view = tree["bufferViews"][image["bufferView"]]
    start = image_view.get("byteOffset", 0)
    embedded = np.asarray(Image.open(io.BytesIO(binary[start:start + image_view["byteLength"]])).convert("RGB"))
    size = manifest["texture_size"]
    texture = read_png(output / "texture.png", (size, size), "RGB")
    require(np.array_equal(embedded, texture), "Embedded GLB texture differs from texture.png")
    observed = read_png(output / "observed_texel_mask.png", (size, size), "L")
    source_ids = read_png(output / "texture_source_ids.png", (size, size), "L")
    count = len(manifest["sources"])
    require(set(np.unique(observed)).issubset({0, 255}), "Observed mask is not binary")
    require(set(np.unique(source_ids)).issubset(set(range(count)) | {254, 255}), "Invalid texture source IDs")
    require(np.array_equal(observed == 255, source_ids < count), "Observed mask and source IDs disagree")
    valid = source_ids != 255
    require(valid.any(), "No valid atlas texels")
    coverage = float((observed[valid] == 255).mean())
    require(abs(coverage - manifest["observed_atlas_texel_fraction"]) < 1e-12, "Observed texel fraction mismatch")

    turntable = json.loads((output / "render_cameras.json").read_text())
    expected_angles = list(range(0, 360, 30))
    require(sorted(item["azimuth"] for item in turntable) == expected_angles, "Expected twelve distinct 30-degree views")
    require(sorted(p.name for p in (output / "renders").glob("*.png")) == [f"az_{a:03d}.png" for a in expected_angles], "Missing/extra turntable PNGs")
    distances, focals, principal_points = [], [], []
    output_pngs = [output / name for name in ("texture.png", "observed_texel_mask.png", "texture_source_ids.png")]
    for item in turntable:
        camera = item["camera"]
        check_camera(camera, item["width"], item["height"])
        path = output / "renders" / f"az_{item['azimuth']:03d}.png"
        read_png(path, (item["width"], item["height"]), "RGB")
        output_pngs.append(path)
        distances.append(np.linalg.norm(-np.asarray(camera["R"]).T @ np.asarray(camera["t"])))
        focals.append(np.asarray(camera["focal_px"]).reshape(-1))
        principal_points.append([camera["cx"], camera["cy"]])
    require(np.allclose(distances, distances[0], rtol=0, atol=1e-6), "Turntable camera distance changes")
    require(all(np.allclose(focal, focals[0]) for focal in focals), "Turntable focal length changes")
    require(np.allclose(principal_points, principal_points[0]), "Turntable principal point changes")
    for item in manifest["sources"]:
        view = view_map[item["id"]]
        path = output / "reprojections" / (item["id"] + ".png")
        read_png(path, (view["width"], view["height"]), "RGB")
        output_pngs.append(path)
    artifact_paths = output_pngs + [output / name for name in (
        "textured_mesh.glb", "textured_mesh.npz", "manifest.json", "render_cameras.json", "bake_script_snapshot.py")]
    return {
        "status": "passed", "artifact_png_files_checked": len(output_pngs),
        "turntable_views_checked": len(turntable), "reprojection_views_checked": count,
        "prepared_png_files_checked": prepared_files, "original_source_files_checked": len(refs),
        "source_hashes_unchanged": True, "manifest_hashes_match": True,
        "canonical_npz_glb_geometry_match": True, "glb_uv_origin_and_texture_match": True,
        "unlit_extension_present": True, "camera_rotations_valid": True,
        "fixed_turntable_camera_distance_and_intrinsics": True,
        "blend_mode": manifest.get("blend_mode"),
        "source_policy": manifest.get("source_policy", "angle"),
        "observed_atlas_texel_fraction": coverage,
        "turntable_elevation_degrees": sorted(set(item["elevation"] for item in turntable)),
        "input_fit_mean_silhouette_iou": cameras.get("mean_input_fit_iou"),
        "metric_scope": "Input masks were used to fit cameras/proportions; their IoU is not held-out reconstruction, identity, or training accuracy.",
        "verification_scope": "File integrity, geometry/UV/texture consistency and source provenance; no appearance-quality, physical-geometry, or identity acceptance is implied.",
        "artifact_sha256": {str(path.relative_to(output)): sha256(path) for path in artifact_paths},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    args = parser.parse_args()
    output, fit, prepared = (path.resolve() for path in (args.output_dir, args.fit_dir, args.prepared_root))
    require(output.is_dir(), f"No output directory: {output}")
    base = {"checked_utc": datetime.now(timezone.utc).isoformat(), "verifier_sha256": sha256(__file__),
            "output_dir": str(output), "fit_dir": str(fit), "prepared_root": str(prepared)}
    try:
        result = {**base, **verify(output, fit, prepared)}
    except Exception as error:
        result = {**base, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    path = output / "verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_sha256"}, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
