"""Repair and denoise normals without moving vertices or changing topology/UVs.

``welded_vertex_normals(vertices, faces)`` returns an (N, 3) float32 array.
Set ``return_info=True`` to receive ``(normals, statistics)`` instead.
Only exact duplicate positions share normals; no proximity-based welding occurs.
``smooth_vertex_normals`` additionally averages nearby compatible normals at a
declared object-relative spatial scale. ``sample_normal_field`` transfers that
full-mesh field to a coarse preview instead of deriving faceted preview normals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def welded_vertex_normals(vertices, faces, weighting="angle", return_info=False):
    """Return shared angle- or area-weighted normals at exact duplicate XYZs.

    Computation uses temporary float64 geometry. Input arrays are never mutated.
    Degenerate triangles do not contribute. A zero weighted sum uses the largest
    incident valid face (lowest face index breaks ties); isolated vertices use +Y.
    This is normal smoothing only, not geometry smoothing or an edge-crease model.
    """
    if weighting not in {"angle", "area"}:
        raise ValueError("weighting must be 'angle' or 'area'")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite, nonempty (N, 3) array")
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must be an integer (F, 3) array")
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError("face indices are outside vertices")

    unique, inverse = np.unique(v, axis=0, return_inverse=True)
    welded_faces = inverse[f]
    tri = unique[welded_faces]
    crosses = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(crosses, axis=1)
    scale = max(float(np.linalg.norm(np.ptp(unique, axis=0))), 1e-12)
    area_epsilon = np.finfo(np.float64).eps * scale * scale
    valid = twice_area > area_epsilon
    face_normals = np.zeros_like(crosses)
    face_normals[valid] = crosses[valid] / twice_area[valid, None]
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    for corner in range(3):
        if weighting == "area":
            contribution = crosses.copy()
            contribution[~valid] = 0
        else:
            edge1 = tri[:, (corner + 1) % 3] - tri[:, corner]
            edge2 = tri[:, (corner + 2) % 3] - tri[:, corner]
            sine_term = np.linalg.norm(np.cross(edge1, edge2), axis=1)
            cosine_term = np.einsum("ij,ij->i", edge1, edge2)
            angle = np.arctan2(sine_term, cosine_term)
            contribution = face_normals * angle[:, None]
        np.add.at(sums, welded_faces[:, corner], contribution)

    length = np.linalg.norm(sums, axis=1)
    zero = length < (1e-12 if weighting == "angle" else area_epsilon)
    welded_normals = np.zeros_like(sums)
    welded_normals[~zero] = sums[~zero] / length[~zero, None]
    isolated = np.zeros(len(unique), dtype=bool)
    if zero.any():
        largest_area = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(largest_area, welded_faces.reshape(-1), np.repeat(twice_area, 3))
        chosen_face = np.full(len(unique), len(f), dtype=np.int64)
        face_index = np.arange(len(f), dtype=np.int64)
        for corner in range(3):
            vertex_index = welded_faces[:, corner]
            choose = valid & (twice_area == largest_area[vertex_index])
            np.minimum.at(chosen_face, vertex_index[choose], face_index[choose])
        available = zero & (chosen_face < len(f))
        welded_normals[available] = face_normals[chosen_face[available]]
        isolated = zero & ~available
        welded_normals[isolated] = [0.0, 1.0, 0.0]

    normals = welded_normals[inverse].astype(np.float32)
    info = {
        "weighting": weighting,
        "input_vertices": int(len(v)), "unique_positions": int(len(unique)),
        "shared_duplicate_records": int(len(v) - len(unique)),
        "faces": int(len(f)), "degenerate_faces_ignored": int((~valid).sum()),
        "zero_sum_positions_using_fallback": int(zero.sum()),
        "positions_without_valid_incident_face": int(isolated.sum()),
        "finite": bool(np.isfinite(normals).all()),
        "max_unit_length_error": float(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1) - 1).max()),
        "geometry_changed": False,
        "limitation": "Exact-position seam smoothing; intentionally does not preserve separately authored hard-normal creases.",
    }
    return (normals, info) if return_info else normals


def normal_seam_statistics(vertices, normals):
    """Describe agreement between normals attached to identical positions."""
    v, n = np.asarray(vertices), np.asarray(normals, dtype=np.float64)
    unique, first, inverse, counts = np.unique(v, axis=0, return_index=True,
                                             return_inverse=True, return_counts=True)
    if n.shape != v.shape or not np.isfinite(n).all():
        raise ValueError("Normals must be finite and have the vertex-array shape")
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, n)
    means = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-30)
    unit_n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", unit_n, means[inverse]), -1, 1)))
    duplicate = counts[inverse] > 1
    values = angle[duplicate]
    return {
        "unique_positions": len(unique), "total_vertex_records": len(v),
        "duplicate_position_records": int(duplicate.sum()),
        "same_position_normals_bitwise_equal": bool(np.array_equal(n, n[first][inverse])),
        "seam_normal_deviation_degrees_p50_p95_max": np.percentile(values, [50, 95, 100]).tolist() if len(values) else [0., 0., 0.],
        "duplicate_records_deviating_gt10deg": int((values > 10).sum()),
    }


def sample_normal_field(vertices, reference_vertices, reference_normals, k=8, return_info=False):
    """Interpolate a full-mesh normal field onto an unchanged preview mesh.

    Exact matching positions copy the full field. Other positions use inverse
    squared distance interpolation from at most eight nearest unique positions,
    rejecting neighbors pointing more than 70 degrees from the closest normal.
    This is an appearance-only approximation, not a new geometric surface.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float64)
    rv = np.asarray(reference_vertices, dtype=np.float64)
    rn = np.asarray(reference_normals, dtype=np.float64)
    if (v.ndim != 2 or v.shape[1] != 3 or rv.ndim != 2 or rv.shape[1] != 3
            or rn.shape != rv.shape or not len(v) or not len(rv) or int(k) < 1):
        raise ValueError("Positions and reference normals must be (N, 3) arrays")
    if not (np.isfinite(v).all() and np.isfinite(rv).all() and np.isfinite(rn).all()):
        raise ValueError("Normal-field inputs must be finite")
    unique, first = np.unique(rv, axis=0, return_index=True)
    field = rn[first]
    field_length = np.linalg.norm(field, axis=1, keepdims=True)
    if np.any(field_length <= 1e-30):
        raise ValueError("Reference normals must have nonzero length")
    field /= field_length
    query, inverse = np.unique(v, axis=0, return_inverse=True)
    distance, index = cKDTree(unique).query(query, k=min(int(k), len(unique)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    closest = field[index[:, 0]]
    candidate = field[index]
    compatible = np.einsum("ijk,ik->ij", candidate, closest) >= np.cos(np.radians(70.))
    scale = max(float(np.ptp(unique, axis=0).max()), 1e-12)
    weight = compatible / np.maximum(distance, scale * 1e-9) ** 2
    sums = np.einsum("ij,ijk->ik", weight, candidate)
    sampled = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-30)
    exact = distance[:, 0] == 0
    sampled[exact] = field[index[exact, 0]]
    normals = sampled[inverse].astype(np.float32)
    info = {
        "method": "nearest_8_inverse_squared_distance_from_full_normal_field",
        "reference_unique_positions": int(len(unique)), "target_unique_positions": int(len(query)),
        "exact_matching_positions": int(exact.sum()),
        "nearest_distance_fraction_of_longest_extent_p50_p95_max":
            (np.percentile(distance[:, 0], [50, 95, 100]) / scale).tolist(),
        "geometry_changed": False,
    }
    return (normals, info) if return_info else normals


def smooth_vertex_normals(vertices, faces, radius_fraction=0.02, iterations=2,
                          max_angle_degrees=70., return_info=False):
    """Denoise welded normals at a scale relative to the longest mesh extent.

    The compact Gaussian kernel has support ``radius_fraction * longest_extent``
    and sigma half that radius. Weights include incident surface area, reducing
    sensitivity to triangulation density. A direction gate rejects opposite
    surfaces and strong creases. Two rounds remove local triangulation noise;
    no vertex, face or UV is moved, merged, removed or rewritten.

    For a preview mesh, call this on the FULL mesh, then call
    ``sample_normal_field(preview_vertices, full_vertices, full_normals)``.
    Re-estimating normals on the coarse preview can reintroduce facet artifacts.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    if not (0 < radius_fraction <= .1 and 1 <= iterations <= 8
            and 0 < max_angle_degrees < 90):
        raise ValueError("Use radius_fraction in (0,.1], 1..8 iterations, and angle in (0,90)")
    v, f = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    initial, initial_info = welded_vertex_normals(v, f, return_info=True)
    unique, first, inverse = np.unique(v, axis=0, return_index=True, return_inverse=True)
    welded_faces = inverse[f]
    tri = unique[welded_faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    support_area = np.zeros(len(unique), dtype=np.float64)
    np.add.at(support_area, welded_faces.reshape(-1), np.repeat(area / 3., 3))
    positive = support_area > 0
    support_area[~positive] = np.median(support_area[positive]) if positive.any() else 1.
    longest_extent = float(np.ptp(unique, axis=0).max())
    if longest_extent <= 0:
        raise ValueError("Cannot smooth a zero-extent mesh")
    radius = radius_fraction * longest_extent
    tree = cKDTree(unique)
    distances = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    row, col = distances.row, distances.col
    base_weights = np.exp(-.5 * (distances.data / (radius * .5)) ** 2) * support_area[col]
    original = initial[first].astype(np.float64)
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    field = original.copy()
    cos_limit = np.cos(np.radians(max_angle_degrees))
    retained = 0
    for _ in range(int(iterations)):
        # Anchor the receiver direction to the original normal, so repeated
        # rounds cannot gradually rotate across a crease or flip a thin part.
        compatible = np.einsum("ij,ij->i", original[row], field[col]) >= cos_limit
        retained = int(compatible.sum())
        matrix = coo_matrix((base_weights[compatible], (row[compatible], col[compatible])),
                            shape=(len(unique), len(unique))).tocsr()
        sums = matrix @ field
        length = np.linalg.norm(sums, axis=1, keepdims=True)
        field = np.divide(sums, length, out=original.copy(), where=length > 1e-30)
    normals = field[inverse].astype(np.float32)
    change = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", field, original), -1, 1)))
    info = {
        "method": "welded_spatial_gaussian_area_weighted_normal_filter",
        "radius_fraction_of_longest_extent": float(radius_fraction),
        "radius_world_units": radius, "gaussian_sigma_world_units": radius * .5,
        "iterations": int(iterations), "max_angle_degrees": float(max_angle_degrees),
        "neighbor_pairs": int(len(row)), "compatible_pairs_last_round": retained,
        "normal_change_degrees_p50_p95_max": np.percentile(change, [50, 95, 100]).tolist(),
        "finite": bool(np.isfinite(normals).all()),
        "max_unit_length_error": float(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1)-1).max()),
        "geometry_changed": False, "initial_welded_normal_statistics": initial_info,
        "limitation": "Appearance-only normal denoising; does not repair silhouettes, geometric dents or semantic boundaries. Nearby similarly oriented surfaces may share smoothing weights.",
    }
    return (normals, info) if return_info else normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--weighting", choices=["angle", "area"], default="angle")
    parser.add_argument("--smooth-radius-fraction", type=float)
    parser.add_argument("--smooth-iterations", type=int, default=2)
    parser.add_argument("--reference-normal-field", type=Path,
                        help="NPZ with full vertices/normals to sample onto the input mesh")
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.mesh_npz.read_bytes()).hexdigest()
    with np.load(args.mesh_npz) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    before = {key: arrays[key].copy() for key in ["vertices", "faces", "uv"] if key in arrays}
    if args.reference_normal_field:
        with np.load(args.reference_normal_field) as reference:
            normals, info = sample_normal_field(arrays["vertices"], reference["vertices"],
                                               reference["normals"], return_info=True)
        info["reference_path"] = str(args.reference_normal_field.resolve())
        info["reference_sha256"] = hashlib.sha256(args.reference_normal_field.read_bytes()).hexdigest()
        info["finite"] = bool(np.isfinite(normals).all())
        info["max_unit_length_error"] = float(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1)-1).max())
    elif args.smooth_radius_fraction:
        normals, info = smooth_vertex_normals(arrays["vertices"], arrays["faces"],
                            radius_fraction=args.smooth_radius_fraction,
                            iterations=args.smooth_iterations, return_info=True)
    else:
        normals, info = welded_vertex_normals(arrays["vertices"], arrays["faces"],
                                              weighting=args.weighting, return_info=True)
    for key, value in before.items():
        if not np.array_equal(value, arrays[key]):
            raise AssertionError(f"Input {key} changed")
    info["source_path"] = str(args.mesh_npz.resolve())
    info["source_sha256"] = source_hash
    info["source_file_unchanged"] = source_hash == hashlib.sha256(args.mesh_npz.read_bytes()).hexdigest()
    info["vertices_faces_uv_unchanged"] = True
    info["seams"] = normal_seam_statistics(arrays["vertices"], normals)
    if not (info["seams"]["same_position_normals_bitwise_equal"] and info["finite"]
            and info["max_unit_length_error"] < 1e-6 and info["source_file_unchanged"]):
        raise AssertionError("Normal validation failed")
    if args.output_npz:
        if args.output_npz.exists():
            raise SystemExit(f"Refusing to overwrite {args.output_npz}")
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **{**arrays, "normals": normals})
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
