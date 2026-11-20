"""Fit perspective cameras to prepared vehicle silhouettes using a fixed mesh.

This estimates object-relative cameras, not the original scene-camera poses.
Silhouette IoU is INPUT FIT, not held-out reconstruction or identity accuracy.
Example:
  python fit_cameras.py --mesh mesh.glb --prepared-root prepared --output-dir fit
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize

from raster import render


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decimate(mesh, face_count):
    if len(mesh.faces) <= face_count:
        return mesh.copy()
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=np.float64),
                             np.asarray(mesh.faces, dtype=np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(face_count))
    out = ms.current_mesh()
    return trimesh.Trimesh(out.vertex_matrix().copy(), out.face_matrix().copy(), process=False)


def canonicalize(mesh, yaw_offset=0., flip_front=False, axis_scale=(1., 1., 1.)):
    """PCA only in the horizontal XZ plane; preserve known Y-up convention."""
    axis_scale = np.asarray(axis_scale, dtype=np.float64)
    if axis_scale.shape != (3,) or not np.isfinite(axis_scale).all() or (axis_scale <= 0).any():
        raise ValueError("axis_scale must contain three finite positive values")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    horizontal = vertices[:, [0, 2]]
    _, eigenvectors = np.linalg.eigh(np.cov(horizontal.T))
    long_axis = eigenvectors[:, -1]
    if long_axis[1] < 0:
        long_axis *= -1
    z_axis = np.array([long_axis[0], 0., long_axis[1]])
    y_axis = np.array([0., 1., 0.])
    x_axis = np.cross(y_axis, z_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    angle = np.deg2rad(yaw_offset + (180 if flip_front else 0))
    c, s = np.cos(angle), np.sin(angle)
    yaw_rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rotation = yaw_rotation @ basis.T
    rotated = vertices @ rotation.T
    center = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
    scale = 2. / np.ptp(rotated[:, 2])
    canonical = (rotated - center) * scale * axis_scale
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale * axis_scale[:, None]
    transform[:3, 3] = -center * scale * axis_scale
    result = trimesh.Trimesh(canonical, np.asarray(mesh.faces).copy(), process=False)
    return result, {
        "world_convention": "+Y up; +Z vehicle front; nominal azimuth 90 views +X side",
        "input_long_axis": z_axis.tolist(),
        "front_sign_rule": "PCA axis with positive original Z, then explicit flip/offset",
        "yaw_offset_deg": float(yaw_offset), "flip_front": bool(flip_front),
        "axis_scale_xyz": axis_scale.tolist(),
        "axis_scale_scope": "Global proxy proportion adjustment supplied by caller; not measured physical dimensions",
        "input_to_canonical_4x4": transform.tolist(),
        "canonical_extents": np.ptp(canonical, axis=0).tolist(),
        "front_semantics_verified": False,
    }


def rotation_for_view(azimuth, elevation, roll=0.):
    az, el, roll = np.deg2rad([azimuth, elevation, roll])
    outward = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    forward = -outward
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ rotation


def camera_from_params(params):
    azimuth, elevation, roll, log_scale, cx, cy, log_distance = params
    distance = float(np.exp(log_distance))
    return {"projection": "perspective",
            "R": rotation_for_view(azimuth, elevation, roll),
            "t": np.array([0., 0., distance]),
            "focal_px": float(np.exp(log_scale) * distance),
            "cx": float(cx), "cy": float(cy)}


def mask_iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def fit_view(vertices, faces, target, nominal, max_evals=700, min_elevation=0.):
    height, width = target.shape
    ys, xs = np.nonzero(target)
    if not len(xs):
        raise ValueError("reference geometry mask is empty")
    target_min = np.array([xs.min(), ys.min()], dtype=float)
    target_max = np.array([xs.max(), ys.max()], dtype=float)
    target_span = np.maximum(target_max - target_min, 1.)
    target_center = (target_min + target_max) / 2
    target_boundary = target & ~ndimage.binary_erosion(target)
    boundary_distance = ndimage.distance_transform_edt(~target_boundary)
    length_scale = math.sqrt(np.count_nonzero(target))
    calls = 0
    best = {"loss": float("inf"), "params": None, "iou": 0.}

    def objective(params):
        nonlocal calls
        calls += 1
        _, ids, _ = render(vertices, faces, camera_from_params(params), size=(height, width),
                           return_barycentric=False)
        predicted = ids >= 0
        if not predicted.any():
            return 2.
        iou = mask_iou(predicted, target)
        boundary = predicted & ~ndimage.binary_erosion(predicted)
        contour = float(boundary_distance[boundary].mean() / length_scale)
        loss = 1. - iou + .3 * contour + .0001 * ((params[0] - nominal) / 25.) ** 2
        if loss < best["loss"]:
            best.update(loss=float(loss), params=np.asarray(params).copy(), iou=iou)
        return float(loss)

    # Small deterministic initialization grid. The bbox gives initial scale/shift.
    seeds = []
    for delta in (-16., 0., 16.):
        for elevation in sorted({max(min_elevation, el) for el in (3., 10., 19.)}):
            for distance in (3.5, 7., 14.):
                rotation = rotation_for_view(nominal + delta, elevation)
                cam = vertices @ rotation.T + [0., 0., distance]
                uv = cam[:, :2] / cam[:, 2, None]
                lo, hi = uv.min(axis=0), uv.max(axis=0)
                span = np.maximum(hi - lo, 1e-6)
                focal = math.sqrt(np.prod(target_span / span))
                principal = target_center - focal * (lo + hi) / 2
                p = np.array([nominal + delta, elevation, 0., math.log(focal / distance),
                              principal[0], principal[1], math.log(distance)])
                seeds.append((objective(p), p))
    seeds.sort(key=lambda item: item[0])
    initial_iou = best["iou"]
    # Normalized coordinates give Powell meaningful stopping/step scales.
    anchor = seeds[0][1].copy()
    units = np.array([25., 25., 3., .25, width * .08, height * .08, 1.])
    lower = np.array([nominal - 25., min_elevation, -3., anchor[3] - .45,
                      anchor[4] - width * .14, anchor[5] - height * .14, math.log(2.5)])
    upper = np.array([nominal + 25., 25., 3., anchor[3] + .45,
                      anchor[4] + width * .14, anchor[5] + height * .14, math.log(24.)])
    bounds = list(zip((lower - anchor) / units, (upper - anchor) / units))
    optimized = minimize(lambda p: objective(anchor + p * units), np.zeros(7),
                         method="Powell", bounds=bounds,
                         options={"maxfev": max_evals, "maxiter": 14,
                                  "xtol": .008, "ftol": .0003})
    return best["params"], {
        "input_fit_iou_low_resolution": best["iou"],
        "initial_grid_input_fit_iou": initial_iou,
        "objective": best["loss"], "render_evaluations": calls,
        "optimizer_success": bool(optimized.success), "optimizer_message": str(optimized.message),
        "parameter_bounds": {"azimuth_deg": [nominal - 25, nominal + 25],
                             "elevation_deg": [float(min_elevation), 25], "roll_deg": [-3, 3],
                             "distance": [2.5, 24]},
    }


def serial_camera(camera):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in camera.items()}


def save_overlay(path, source, reference, predicted, caption):
    image = np.asarray(source.convert("RGB"), dtype=np.float32)
    mask_color = np.zeros_like(image)
    # Agreement green, reference-only red, render-only blue.
    mask_color[reference & predicted] = [30, 210, 80]
    mask_color[reference & ~predicted] = [245, 50, 45]
    mask_color[predicted & ~reference] = [45, 110, 245]
    union = reference | predicted
    image[union] = .63 * image[union] + .37 * mask_color[union]
    image = Image.fromarray(image.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(250, 250, 250))
    draw.text((8, 9), caption, fill=(20, 20, 20))
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yaw-offset", type=float, default=0.)
    parser.add_argument("--flip-front", action="store_true")
    parser.add_argument("--axis-scale", type=float, nargs=3, default=(1., 1., 1.),
                        metavar=("X", "Y", "Z"),
                        help="Global XYZ scale after PCA; input-driven proxy adjustment, not dimension calibration")
    parser.add_argument("--fit-long-side", type=int, default=320)
    parser.add_argument("--fit-faces", type=int, default=10000)
    parser.add_argument("--output-faces", type=int, default=80000)
    parser.add_argument("--max-evals", type=int, default=700)
    parser.add_argument("--min-elevation", type=float, default=0.,
                        help="Object-relative camera elevation lower bound; a prior, not calibration")
    args = parser.parse_args()
    if not 0 <= args.min_elevation < 25:
        parser.error("--min-elevation must be in [0, 25)")
    started = time.perf_counter()
    prepared_root, output = Path(args.prepared_root).resolve(), Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((prepared_root / "manifest.json").read_text())
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    canonical, transform = canonicalize(mesh, args.yaw_offset, args.flip_front, args.axis_scale)
    export_mesh = decimate(canonical, args.output_faces)
    export_mesh.export(output / "canonical_mesh.glb")
    fit_mesh = decimate(export_mesh, args.fit_faces)
    print(f"canonical: {len(export_mesh.faces)} export faces, {len(fit_mesh.faces)} fit faces", flush=True)
    record = {
        "method": "fixed_mesh_silhouette_perspective_camera_fit",
        "status": "running", "mesh_path": str(Path(args.mesh).resolve()),
        "mesh_sha256": sha256(args.mesh), "canonical_mesh": "canonical_mesh.glb",
        "canonical_mesh_sha256": sha256(output / "canonical_mesh.glb"),
        "prepared_root": str(prepared_root), "canonicalization": transform,
        "camera_convention": "world-to-camera; +X right,+Y down,+Z forward; integer pixel centers",
        "metric_scope": "Silhouette IoU measures fit to INPUT masks, not held-out identity/geometry accuracy",
        "optimization": {"fit_long_side": args.fit_long_side, "fit_faces": len(fit_mesh.faces),
                         "output_faces": len(export_mesh.faces), "max_evals": args.max_evals,
                         "min_elevation_deg": args.min_elevation,
                         "elevation_prior_scope": "User-specified view-height prior, not camera calibration",
                         "background": "excluded via supplied geometry masks",
                         "shape_optimized": False,
                         "explicit_global_axis_scale": list(args.axis_scale)},
        "views": [],
    }
    for view in prepared["views"]:
        view_started = time.perf_counter()
        mask_path = prepared_root / view["files"]["mask_geometry.png"]["path"]
        source_path = prepared_root / view["files"]["source_rgb.png"]["path"]
        full_mask = Image.open(mask_path).convert("L")
        width, height = full_mask.size
        ratio = args.fit_long_side / max(width, height)
        small_w, small_h = round(width * ratio), round(height * ratio)
        target = np.asarray(full_mask.resize((small_w, small_h), Image.Resampling.NEAREST)) >= 128
        params, metrics = fit_view(np.asarray(fit_mesh.vertices), np.asarray(fit_mesh.faces),
                                   target, float(view["nominal_azimuth_deg"]), args.max_evals,
                                   args.min_elevation)
        camera = camera_from_params(params)
        sx, sy = width / small_w, height / small_h
        camera["focal_px"] = [camera["focal_px"] * sx, camera["focal_px"] * sy]
        camera["cx"] = (camera["cx"] + .5) * sx - .5
        camera["cy"] = (camera["cy"] + .5) * sy - .5
        _, face_id, _ = render(export_mesh.vertices, export_mesh.faces, camera,
                                size=(height, width), return_barycentric=False)
        predicted = face_id >= 0
        reference = np.asarray(full_mask) >= 128
        full_iou = mask_iou(predicted, reference)
        overlay_name = f"{view['id']}_input_fit.png"
        save_overlay(output / overlay_name, Image.open(source_path), reference, predicted,
                     f"{view['id']} | INPUT FIT IoU {full_iou:.3f} | green overlap / red reference / blue mesh")
        entry = {
            "id": view["id"], "camera": serial_camera(camera), "width": width, "height": height,
            "nominal_azimuth_deg": view["nominal_azimuth_deg"],
            "azimuth_deg": float(params[0] % 360), "elevation_deg": float(params[1]),
            "roll_deg": float(params[2]), "distance": float(np.exp(params[6])),
            "input_fit_iou": full_iou, "fit_iou": full_iou, "metrics": metrics,
            "source_rgb": str(source_path), "mask_geometry": str(mask_path),
            "overlay": overlay_name, "seconds": round(time.perf_counter() - view_started, 3),
        }
        record["views"].append(entry)
        (output / "cameras.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"{view['id']}: INPUT IoU {full_iou:.4f}; az/el {entry['azimuth_deg']:.1f}/{entry['elevation_deg']:.1f}; {entry['seconds']:.1f}s", flush=True)
    record["status"] = "complete"
    record["mean_input_fit_iou"] = float(np.mean([v["input_fit_iou"] for v in record["views"]]))
    record["total_seconds"] = round(time.perf_counter() - started, 3)
    (output / "cameras.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    # Small contact sheet for checking correspondence and front/back semantics.
    tiles = []
    for view in record["views"]:
        image = Image.open(output / view["overlay"])
        image.thumbnail((576, 384))
        tiles.append(image.copy())
    contact = Image.new("RGB", (1152, 384 * math.ceil(len(tiles) / 2)), "white")
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % 2) * 576, (index // 2) * 384))
    contact.save(output / "input_fit_contact_sheet.jpg", quality=92)
    print(f"complete: mean INPUT IoU {record['mean_input_fit_iou']:.4f}; {record['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
