"""Independent checks for a two-dimensional Qwen reflection-cleanup experiment.

No model is loaded. Existing 3D assets and generated images are read only. The
JSON report separates exact compositing guarantees from approximate raw-image
alignment diagnostics; neither establishes naturalness or successful delighting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path, mode="RGB", expected_shape=None):
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode))
    require(expected_shape is None or array.shape == expected_shape,
            f"Unexpected image shape at {path}: {array.shape}")
    return array


def feature_alignment(source, target, region):
    """Measure feature displacement without warping or changing either image."""
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.02)
    mask = region.astype(np.uint8) * 255
    gray0, gray1 = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (source, target)]
    key0, desc0 = detector.detectAndCompute(gray0, mask)
    key1, desc1 = detector.detectAndCompute(gray1, mask)
    result = {"source_features": len(key0), "target_features": len(key1),
              "method": "SIFT ratio .72 + similarity transform RANSAC; diagnostic only"}
    if desc0 is None or desc1 is None or len(desc1) < 2:
        return {**result, "status": "insufficient_features"}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc0, desc1, k=2)
    matches = [a for a, b in pairs if a.distance < .72 * b.distance]
    result["ratio_test_matches"] = len(matches)
    if len(matches) < 6:
        return {**result, "status": "insufficient_matches"}
    p0 = np.asarray([key0[m.queryIdx].pt for m in matches], dtype=np.float32)
    p1 = np.asarray([key1[m.trainIdx].pt for m in matches], dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
        ransacReprojThreshold=2.5, maxIters=3000, confidence=.995, refineIters=10)
    if matrix is None or inliers is None:
        return {**result, "status": "transform_not_estimated"}
    keep = inliers.ravel().astype(bool)
    require(keep.any(), "Feature estimator unexpectedly returned no inliers")
    predicted = p0 @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(predicted - p1, axis=1)[keep]
    displacement = np.linalg.norm(p1 - p0, axis=1)[keep]
    a, b = matrix[0, 0], matrix[1, 0]
    h, w = source.shape[:2]
    center = np.array([w / 2., h / 2.])
    center_motion = matrix[:, :2] @ center + matrix[:, 2] - center
    result.update(status="estimated", inliers=int(keep.sum()),
        inlier_fraction=float(keep.mean()), affine_source_to_raw=matrix.tolist(),
        isotropic_scale=float(np.hypot(a, b)), rotation_degrees=float(np.degrees(np.arctan2(b, a))),
        image_center_motion_xy_pixels=center_motion.tolist(),
        matched_point_motion_pixels_p50_p95_max=np.percentile(displacement, [50, 95, 100]).tolist(),
        residual_pixels_p50_p95_max=np.percentile(residual, [50, 95, 100]).tolist(),
        limitation="Matched details may stay stable while other regions distort; inspect the raw image visually.")
    return result


def edge_alignment(source, target, region):
    edges = []
    for image in (source, target):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges.append(cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .6), 30, 90) > 0)
    source_edges = edges[0] & region
    if not source_edges.any() or not edges[1].any():
        return {"status": "insufficient_edges"}
    distance = distance_transform_edt(~edges[1])[source_edges]
    return {"status": "measured", "source_edge_pixels": int(source_edges.sum()),
        "source_to_nearest_target_edge_pixels_p50_p95_max": np.percentile(distance, [50, 95, 100]).tolist(),
        "fraction_source_edges_within_2px": float((distance <= 2).mean()),
        "limitation": "Illumination changes also alter detected edges; this is not a ground-truth shape metric."}


def change_statistics(source, target, region):
    if not region.any():
        return {"pixels": 0}
    difference = np.abs(target.astype(np.int16) - source.astype(np.int16))
    delta = difference[region].mean(axis=1)
    return {"pixels": int(region.sum()), "mean_rgb_absolute_change": float(delta.mean()),
        "rgb_absolute_change_p50_p95_max": np.percentile(delta, [50, 95, 100]).tolist(),
        "fraction_pixels_any_channel_changed": float((difference[region].max(axis=1) > 0).mean()),
        "source_mean_rgb": source[region].mean(axis=0).tolist(),
        "target_mean_rgb": target[region].mean(axis=0).tolist()}


def verify(args):
    baseline = json.loads(args.baseline.read_text())
    for path, record in baseline["files"].items():
        require(sha256(path) == record["sha256"], f"A protected 3D file changed: {path}")
    out = args.output_dir
    source = read_image(out / "source.png")
    require(source.shape == (1024, 1536, 3), "Expected original 1536 x 1024 working image")
    protected = read_image(args.protected_mask, "L", source.shape[:2]) > 127
    support = read_image(args.edit_mask, "L", source.shape[:2]) > 127
    require(support.any() and np.array_equal(protected, ~support),
            "Protected mask must be the exact complement of editable support")
    alpha = read_image(args.soft_alpha, "L", source.shape[:2]) if args.soft_alpha else None
    if alpha is not None:
        require(np.array_equal(alpha > 0, support), "Alpha support differs from the hard edit mask")
    mask_manifest_path = args.protected_mask.parent / "manifest.json"
    mask_manifest = json.loads(mask_manifest_path.read_text())
    for record in mask_manifest["inputs"].values():
        require(sha256(record["path"]) == record["sha256"], "Mask input file changed")
    for name, record in mask_manifest["outputs"].items():
        require(sha256(mask_manifest_path.parent / name) == record["sha256"], "Mask output file changed")
    reference = read_image(mask_manifest["inputs"]["source_rgb"]["path"], expected_shape=source.shape)
    require(np.array_equal(source, reference), "Qwen working source and mask reference RGB differ")
    geometry = read_image(mask_manifest["inputs"]["geometry_mask"]["path"], "L", source.shape[:2]) > 127
    detail = read_image(mask_manifest_path.parent / "detail_protection_mask.png", "L", source.shape[:2]) > 127
    require(not np.any(support & (~geometry | detail)), "Editable support includes protected details or background")
    run_manifest = json.loads((out / "manifest.json").read_text())
    composition_manifest = json.loads((out / "compositing.json").read_text())
    require(sha256(out / "source.png") == composition_manifest["source_sha256"], "Compositing source hash differs")
    if args.soft_alpha:
        require(sha256(args.soft_alpha) == composition_manifest["mask_sha256"], "Compositing alpha hash differs")
    require(sha256(run_manifest["source"]) == run_manifest["source_sha256"], "Original source photo changed")
    require(np.array_equal(read_image(out / "source_canvas.png"), source), "Unexpected crop/pad in this same-size experiment")
    variants = {}
    for name in ("mild", "strong"):
        raw_path, clean_path = out / name / "raw.png", out / name / "clean.png"
        with Image.open(raw_path) as raw_file:
            raw_mode = raw_file.mode
            raw_alpha = np.asarray(raw_file.getchannel("A")) if "A" in raw_file.getbands() else None
        raw = read_image(raw_path, expected_shape=source.shape)
        clean = read_image(clean_path, expected_shape=source.shape)
        require(sha256(raw_path) == run_manifest["variants"][name]["raw_sha256"], f"{name} raw output hash changed")
        composition_record = composition_manifest["variants"][name]
        require(sha256(raw_path) == composition_record["raw_sha256"], f"{name}: compositing raw hash differs")
        require(sha256(clean_path) == composition_record["clean_sha256"], f"{name}: compositing clean hash differs")
        mix = float(composition_record["mix"])
        require(np.isfinite(mix) and 0 <= mix <= 1, f"{name}: invalid blend weight")
        require(np.array_equal(clean[protected], source[protected]), f"{name}: protected pixels changed")
        require(np.array_equal(clean[detail], source[detail]), f"{name}: explicit detail pixels changed")
        composition = None
        if alpha is not None:
            a = alpha.astype(np.float64)[..., None] / 255 * mix
            expected = np.rint(raw * a + source * (1 - a)).astype(np.uint8)
            rounding_error = np.abs(expected.astype(np.int16) - clean.astype(np.int16))
            require(rounding_error.max() <= 1, f"{name}: clean output is not the declared alpha composite")
            if mix == 1.:
                require(np.array_equal(clean[alpha == 255], raw[alpha == 255]), f"{name}: fully edited core differs from raw")
            composition = {"matches_declared_alpha_composite_within_1_lsb": True,
                "max_rgb_rounding_difference": int(rounding_error.max()),
                "declared_compositing_mix": mix,
                "effective_alpha_formula": "soft_alpha / 255 * compositing_mix",
                "fully_editable_core_matches_raw_exactly": True if mix == 1. else None,
                "strength_interpretation": "Explicit postprocessing blend weight; not a native Qwen regional strength parameter."}
        variants[name] = {
            "raw_sha256": sha256(raw_path), "clean_sha256": sha256(clean_path),
            "raw_image_mode": raw_mode,
            "raw_alpha_range": [int(raw_alpha.min()), int(raw_alpha.max())] if raw_alpha is not None else None,
            "raw_alpha_range_on_edit_support": [int(raw_alpha[support].min()), int(raw_alpha[support].max())] if raw_alpha is not None else None,
            "raw_rgb_interpretation": "Use stored RGB directly; discard near-opaque model alpha rather than compositing it against white or treating it as an edit mask.",
            "protected_pixels_exact": int(protected.sum()), "explicit_detail_pixels_exact": int(detail.sum()),
            "source_size_preserved": True, "alpha_composition": composition,
            "raw_alignment_vehicle": feature_alignment(source, raw, geometry),
            "raw_alignment_protected_details": feature_alignment(source, raw, detail),
            "raw_edge_alignment_protected_details": edge_alignment(source, raw, detail),
            "raw_change_in_protected_details": change_statistics(source, raw, detail),
            "clean_change_in_editable_paint": change_statistics(source, clean, support),
        }
    mild, strong = [read_image(out / name / "clean.png") for name in ("mild", "strong")]
    archived_report = out / "initial_mask_review" / "verification.json"
    raw_unchanged = None
    if archived_report.exists():
        archived = json.loads(archived_report.read_text())
        for name in ("mild", "strong"):
            require(variants[name]["raw_sha256"] == archived["variants"][name]["raw_sha256"],
                    f"{name}: raw prediction changed during mask refinement")
        raw_unchanged = True
    return {
        "status": "integrity_and_protection_passed_pending_visual_review",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(args.baseline.resolve()), "unchanged_3d_files": len(baseline["files"]),
        "source_png_sha256": sha256(out / "source.png"), "source_and_mask_reference_rgb_identical": True,
        "source_size_wh": [1536, 1024], "editable_pixels": int(support.sum()),
        "raw_outputs_unchanged_from_initial_composite": raw_unchanged,
        "locked_pixels": int(protected.sum()), "explicit_detail_pixels": int(detail.sum()),
        "variants": variants,
        "mild_vs_strong_in_editable_paint": change_statistics(mild, strong, support),
        "scope": "2D files only. Exact protected-pixel retention is caused by explicit compositing, not model-native region locking.",
        "visual_review_required": ["Raw vehicle position and proportions", "Paint crease and door-surface continuity",
            "Reflection suppression versus retained natural sheen", "Over-smoothed or repainted appearance",
            "Feather transitions next to locked seams, handles, wheel arches and silhouette"],
        "not_established": "These measurements do not establish true albedo, physical reflection separation or improved visual quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("/tmp/qwen-reflection-baseline.json"))
    parser.add_argument("--protected-mask", type=Path, required=True)
    parser.add_argument("--edit-mask", type=Path, required=True)
    parser.add_argument("--soft-alpha", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args)
    report = args.report or args.output_dir / "independent_verification.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
