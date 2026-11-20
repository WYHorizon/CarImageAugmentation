"""Bake observed vehicle photographs into a single, view-independent UV texture.

Requires the canonical mesh and fitted cameras produced by fit_cameras.py.
This is an appearance texture with captured illumination/reflections, not PBR
albedo and not a recovery of unseen surfaces. All source images remain local.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from raster import render


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample(image, u, v, interpolation=cv2.INTER_LINEAR):
    """OpenCV remap is limited to <32767 rows; sample vectors in blocks."""
    if not len(u):
        return np.empty((0,) + image.shape[2:], dtype=image.dtype)
    result = []
    for i in range(0, len(u), 20000):
        result.append(cv2.remap(image, u[i:i + 20000, None].astype(np.float32),
                                v[i:i + 20000, None].astype(np.float32), interpolation,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, 0])
    return np.concatenate(result)


def project(points, camera):
    R = np.asarray(camera.get("R", np.eye(3)))
    t = np.asarray(camera.get("t", [0, 0, 0]))
    pc = points @ R.T + t
    focal = np.asarray(camera["focal_px"])
    fx, fy = (float(focal), float(focal)) if focal.ndim == 0 else focal
    if camera.get("projection", "perspective") == "perspective":
        u = pc[:, 0] / pc[:, 2] * fx + camera["cx"]
        v = pc[:, 1] / pc[:, 2] * fy + camera["cy"]
    else:
        u = pc[:, 0] * fx + camera["cx"]
        v = pc[:, 1] * fy + camera["cy"]
    return u, v, pc[:, 2], -R.T @ t


def camera_for_angle(vertices, angle, elevation=2, width=1200, height=900):
    az, el = np.radians([angle, elevation])
    direction = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.array([np.cos(az), 0, -np.sin(az)])
    forward = -direction
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    radius = float(np.linalg.norm(vertices, axis=1).max())
    distance = radius * 5
    # Shared distance/focal for the whole turntable, no per-view bbox scaling.
    focal = min(width / 2.3, height / 1.6) * distance / radius
    return dict(R=R.tolist(), t=[0, 0, distance], focal_px=focal,
                cx=(width - 1) / 2, cy=(height - 1) / 2,
                projection="perspective", near=.001)


def render_textured(mesh, uv, texture, camera, width, height, background=(235, 237, 240)):
    depth, face_id, bary = render(mesh.vertices, mesh.faces, camera, (height, width))
    hit = face_id >= 0
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background
    coords = (uv[mesh.faces[face_id[hit]]] * bary[hit, :, None]).sum(axis=1)
    image[hit] = sample(texture, coords[:, 0] * (texture.shape[1] - 1),
                        (1 - coords[:, 1]) * (texture.shape[0] - 1))
    return Image.fromarray(image), hit, depth


def unlit_gltf(tree):
    for mat in tree.get("materials", []):
        mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
    used = tree.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in used:
        used.append("KHR_materials_unlit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--angle-power", type=float, default=8)
    parser.add_argument("--blend-mode", choices=["weighted", "best"], default="best")
    parser.add_argument("--source-policy", choices=["angle", "vehicle_regions"], default="angle")
    parser.add_argument("--views", nargs="*", help="Optional subset of observed images for baking")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "bake_script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    mesh_path = args.fit_dir / "canonical_mesh.glb"
    cameras_path = args.fit_dir / "cameras.json"
    source_manifest_path = args.prepared_root / "manifest.json"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.int32)
    normals = np.asarray(mesh.vertex_normals, np.float32)
    cameras = json.loads(cameras_path.read_text())
    references = json.loads(source_manifest_path.read_text())
    refs = {v["id"]: v for v in references["views"]}
    views = [v for v in cameras["views"] if not args.views or v["id"] in args.views]
    if not views:
        raise SystemExit("No source views selected")
    record = dict(method="multi_photo_visibility_weighted_uv_bake", status="running",
                  mesh_sha256=sha256(mesh_path), cameras_sha256=sha256(cameras_path),
                  prepared_manifest_sha256=sha256(source_manifest_path),
                  script_sha256=sha256(__file__), angle_power=args.angle_power, blend_mode=args.blend_mode,
                  source_policy=args.source_policy,
                  texture_size=args.texture_size, sources=[],
                  limitation="Input-fitted photo appearance; baked illumination/reflections; unseen surfaces are approximate; no independent identity validation.")
    (output / "manifest.json").write_text(json.dumps(record, indent=2))

    print("[1/5] Generate a shared UV atlas", flush=True)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces.astype(np.uint32))
    chart = xatlas.ChartOptions()
    chart.max_iterations = 2
    pack = xatlas.PackOptions()
    pack.resolution = args.texture_size
    pack.padding = 4
    atlas.generate(chart_options=chart, pack_options=pack)
    mapping, indices, uv = atlas[0]
    indices = np.asarray(indices, np.int32)
    uv = np.asarray(uv, np.float32)
    av = vertices[mapping]
    an = normals[mapping]
    size = args.texture_size
    uv_screen = np.column_stack([uv[:, 0] * (size - 1), (1 - uv[:, 1]) * (size - 1), np.ones(len(uv))])
    _, uv_faces, uv_bary = render(uv_screen, indices,
                                dict(projection="orthographic", focal_px=1, cx=0, cy=0), (size, size))
    valid = uv_faces >= 0
    atlas_faces = indices[uv_faces[valid]]
    bary = uv_bary[valid]
    points = np.einsum("ni,nij->nj", bary, av[atlas_faces])
    point_normals = np.einsum("ni,nij->nj", bary, an[atlas_faces])
    point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1, keepdims=True), 1e-8)
    sums = np.zeros((len(points), 3), np.float32)
    weights = np.zeros(len(points), np.float32)
    best_weight = np.zeros(len(points), np.float32)
    best_source = np.full(len(points), -1, np.int16)
    best_colors = np.zeros((len(points), 3), np.float32)
    # A coherent photo for the front/rear panels and each wheel/door side avoids
    # fragmenting semantic details when the inferred local normals are noisy.
    preferred = np.full(len(points), "", dtype="U12")
    lo, hi = vertices.min(0), vertices.max(0)
    span = hi - lo
    preferred[points[:, 2] > lo[2] + .75 * span[2]] = "front_002"
    preferred[points[:, 2] < lo[2] + .25 * span[2]] = "rear_006"
    side_band = (np.abs(points[:, 0]) > .25 * span[0]) & (np.abs(points[:, 2]) < .3 * span[2])
    wheel_band = (points[:, 1] < lo[1] + .4 * span[1]) & (np.abs(points[:, 2]) < .4 * span[2])
    preferred[(side_band | wheel_band) & (points[:, 0] > 0)] = "side_004"
    preferred[(side_band | wheel_band) & (points[:, 0] <= 0)] = "side_008"
    del uv_bary, atlas_faces, bary

    print(f"[2/5] Project {len(views)} observed photos to {len(points):,} atlas texels", flush=True)
    for source_index, view in enumerate(views):
        ref = refs[view["id"]]
        rgb_path = args.prepared_root / ref["files"]["source_rgb.png"]["path"]
        mask_path = args.prepared_root / ref["files"]["mask_geometry.png"]["path"]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path)) > 127
        height, width = mask.shape
        camera = view["camera"]
        depth, visible_faces, _ = render(vertices, faces, camera, (height, width), return_barycentric=False)
        interior = distance_transform_edt(mask).astype(np.float32)
        u, v, z, center = project(points, camera)
        # Evaluate the nearest visible triangle's plane at the actual subpixel
        # location. A global depth tolerance leaks colors through thin layers.
        nearest_face = sample(visible_faces.astype(np.float32), u, v, cv2.INTER_NEAREST).astype(np.int32)
        safe_face = np.maximum(nearest_face, 0)
        R = np.asarray(camera["R"])
        t = np.asarray(camera["t"])
        plane_normals = np.asarray(mesh.face_normals) @ R.T
        plane_points = vertices[faces[:, 0]] @ R.T + t
        plane_d = np.einsum("ij,ij->i", plane_normals, plane_points)
        pc = points @ R.T + t
        camera_rays = pc / pc[:, 2:3]
        denominators = np.einsum("ij,ij->i", plane_normals[safe_face], camera_rays)
        sampled_depth = np.divide(plane_d[safe_face], denominators,
                                  out=np.full(len(points), np.inf), where=np.abs(denominators) > 1e-9)
        inside = sample(interior, u, v)
        ray = center[None, :] - points
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-8)
        facing = np.clip(np.einsum("ij,ij->i", point_normals, ray), 0, 1)
        depth_tolerance = .0003 * np.ptp(vertices, axis=0).max() + .0001 * z
        visibility = (nearest_face >= 0) & (sampled_depth > 0) & (np.abs(sampled_depth - z) < depth_tolerance)
        visibility &= (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1) & (z > 0)
        visibility &= facing > .05
        w = np.power(facing, args.angle_power) * np.clip(inside / 4, 0, 1) * visibility
        if args.source_policy == "vehicle_regions":
            w *= np.where(preferred == view["id"], 1000, 1)
        colors = sample(rgb, u, v).astype(np.float32)
        sums += colors * w[:, None]
        weights += w
        better = w > best_weight
        best_source[better] = source_index
        best_weight[better] = w[better]
        best_colors[better] = colors[better]
        record["sources"].append(dict(id=view["id"], source_sha256=ref["source_sha256"],
                                      rgb_sha256=sha256(rgb_path), geometry_mask_sha256=sha256(mask_path),
                                      visible_atlas_texels=int((w > 1e-14).sum())))
        print(f"  {view['id']}: {(w > 1e-14).sum():,} observed texels", flush=True)

    print("[3/5] Fill unobserved texels with nearby observed surface colors; record their mask", flush=True)
    observed = weights > 1e-14
    if not observed.any():
        raise RuntimeError("No texels pass visibility checks; inspect cameras and mesh normals")
    colors = np.zeros((len(points), 3), np.float32)
    colors[observed] = (best_colors[observed] if args.blend_mode == "best"
                        else sums[observed] / weights[observed, None])
    missing = ~observed
    if missing.any():
        tree = cKDTree(points[observed])
        _, nearest = tree.query(points[missing], k=1, workers=4)
        colors[missing] = colors[observed][nearest]
    texture = np.zeros((size, size, 3), np.uint8)
    texture[valid] = np.clip(colors, 0, 255).astype(np.uint8)
    observed_map = np.zeros((size, size), np.uint8)
    observed_map[valid] = observed.astype(np.uint8) * 255
    source_map = np.full((size, size), 255, np.uint8)
    source_map[valid] = np.where(observed, best_source, 254).astype(np.uint8)
    # Only pad *outside* charts. It prevents bilinear/mipmap sampling of empty atlas pixels.
    padding_distance, nearest = distance_transform_edt(~valid, return_indices=True)
    padding = (~valid) & (padding_distance <= 12)
    texture[padding] = texture[nearest[0][padding], nearest[1][padding]]
    Image.fromarray(texture).save(output / "texture.png")
    Image.fromarray(observed_map).save(output / "observed_texel_mask.png")
    Image.fromarray(source_map).save(output / "texture_source_ids.png")
    record["observed_atlas_texel_fraction"] = float(observed.mean())
    record["unobserved_fill"] = "Nearest observed point in 3D, not generative completion; separate observed_texel_mask.png provided"

    print("[4/5] Export a textured GLB with one fixed atlas", flush=True)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture),
                                                  metallicFactor=0, roughnessFactor=1, doubleSided=True)
    textured = trimesh.Trimesh(vertices=av, faces=indices, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    (output / "textured_mesh.glb").write_bytes(trimesh.exchange.gltf.export_glb(
        trimesh.Scene(textured), include_normals=True, tree_postprocessor=unlit_gltf))
    np.savez_compressed(output / "textured_mesh.npz", vertices=av, faces=indices, uv=uv)
    record["mesh"] = dict(vertices=len(av), faces=len(indices), glb_sha256=sha256(output / "textured_mesh.glb"))

    print("[5/5] Render a fixed-camera-distance turntable and input reprojections", flush=True)
    renders = output / "renders"
    renders.mkdir()
    cameras_out = []
    for angle in range(0, 360, 30):
        camera = camera_for_angle(av, angle)
        image, _, _ = render_textured(textured, uv, texture, camera, 1200, 900)
        image.save(renders / f"az_{angle:03d}.png")
        cameras_out.append(dict(azimuth=angle, elevation=2, width=1200, height=900, camera=camera))
    (output / "render_cameras.json").write_text(json.dumps(cameras_out, indent=2))
    reprojections = output / "reprojections"
    reprojections.mkdir()
    for view in views:
        image, _, _ = render_textured(textured, uv, texture, view["camera"], view["width"], view["height"])
        image.save(reprojections / (view["id"] + ".png"))
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["status"] = "complete_pending_visual_review"
    (output / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps(dict(output=str(output), seconds=record["seconds"], observed_fraction=observed.mean())), flush=True)


if __name__ == "__main__":
    main()
