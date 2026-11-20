"""Independently check a 2D windshield repair and frozen previous 3D assets.

No model is loaded and no existing source or 3D asset is written. This checks
pixel protection and provenance; image quality still needs visual inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb(path, allow_rgba=False):
    with Image.open(path) as image:
        require(image.mode == "RGB" or (allow_rgba and image.mode == "RGBA"),
                f"Unexpected image mode: {path}")
        return np.asarray(image.convert("RGB")).copy()


def verify(baseline_path, candidate):
    baseline = json.loads(baseline_path.read_text())
    for path, entry in baseline["files"].items():
        require(sha(path) == entry["sha256"], f"Frozen previous asset changed: {path}")
    source_path = Path(baseline["source_image"])
    require(sha(source_path) == baseline["source_image_sha256"], "Frozen front reference changed")
    generation = json.loads((candidate / "manifest.json").read_text())
    composite = json.loads((candidate / "compositing.json").read_text())
    mask_manifest_path = Path(composite["mask_manifest"])
    masks = json.loads(mask_manifest_path.read_text())
    mask_dir = mask_manifest_path.parent
    source, raw, clean = [rgb(candidate / name, allow_rgba=name == "raw.png")
                          for name in ("source.png", "raw.png", "clean.png")]
    require(source.shape == raw.shape == clean.shape == (1024, 1536, 3), "Unexpected image dimensions")
    require(np.array_equal(source, rgb(source_path)), "Working source differs from frozen original")
    require(sha(candidate / "source.png") == baseline["source_image_sha256"] ==
            generation["source_sha256"] == composite["source_sha256"], "Source hash mismatch")
    require(sha(candidate / "raw.png") == generation["raw_sha256"] == composite["raw_sha256"],
            "Raw output differs from generation provenance")
    require(sha(candidate / "clean.png") == composite["clean_sha256"], "Clean output hash mismatch")
    reused_prediction = None
    if generation.get("raw_reuse"):
        original_run = Path(generation["raw_reuse"]["directory"])
        original_generation = json.loads((original_run / "manifest.json").read_text())
        for name in ("source.png", "raw.png"):
            require(sha(candidate / name) == sha(original_run / name), f"Reused file differs: {name}")
        for key in ("model_revision", "prompt", "negative_prompt", "true_cfg_scale", "seed", "steps", "size_wh"):
            require(generation[key] == original_generation[key], f"Reused generation setting differs: {key}")
        reused_prediction = {"directory": str(original_run), "source_and_raw_byte_exact": True,
                             "generation_settings_unchanged": True,
                             "meaning": "Same model prediction; only the protection/compositing boundary was revised."}
    require(masks["inputs"]["source_rgb"]["sha256"] == baseline["source_image_sha256"],
            "Mask was prepared against a different photograph")
    feature_names = (["feature_protection_mask.png"] if "feature_protection_mask.png" in masks["outputs"]
                     else ["interior_core_mask.png", "sticker_and_cover_protection.png"])
    for name in ["soft_alpha.png", "hard_support.png", "protected_mask.png", *feature_names]:
        require(sha(mask_dir / name) == masks["outputs"][name]["sha256"], f"Reviewed mask file changed: {name}")
    require(sha(mask_dir / "soft_alpha.png") == composite["mask_sha256"] == sha(candidate / "edit_alpha.png"),
            "Composited alpha differs from reviewed mask")
    alpha = np.asarray(Image.open(mask_dir / "soft_alpha.png").convert("L"))
    support = np.asarray(Image.open(mask_dir / "hard_support.png").convert("L")) > 127
    protected = np.asarray(Image.open(mask_dir / "protected_mask.png").convert("L")) > 127
    features = np.logical_or.reduce([np.asarray(Image.open(mask_dir / name).convert("L")) > 127
                                    for name in feature_names])
    require(alpha.shape == source.shape[:2], "Mask dimensions differ")
    require(np.array_equal(support, alpha > 0) and np.array_equal(protected, alpha == 0), "Mask contract mismatch")
    require(not (features & support).any(), "Editable region overlaps named preserved interior/trim details")
    require(np.array_equal(np.asarray(Image.open(candidate / "protected_mask.png")) > 127, protected),
            "Saved protected region differs from reviewed mask")
    require(np.array_equal(clean[protected], source[protected]), "Source pixels changed outside reviewed edit support")
    require(np.array_equal(clean[features], source[features]), "Named preserved detail pixels changed")
    # Independently use the previous reference's semantic glass annotation,
    # rather than trusting the new repair mask to label itself as glass.
    project = Path(__file__).resolve().parent.parent
    labels_path = project / "08_3d_probe/photo_texture/outputs/material_regions_v1/front_002/material_ids.png"
    labels = np.asarray(Image.open(labels_path).convert("L"))
    # The old coarse annotation includes protected fringes around stickers and
    # mirrors, so disagreement there is recorded, not treated as ground truth.
    # Independently bound editing to its original full-windshield polygon.
    glass_envelope = Image.new("L", (1536, 1024))
    ImageDraw.Draw(glass_envelope).polygon([(486, 375), (552, 251), (988, 251), (1044, 376)], fill=255)
    require(not (support & (np.asarray(glass_envelope) == 0)).any(),
            "Edit support extends outside the independently annotated windshield")
    prior_label_difference = int((support & (labels != 2)).sum())
    mix = float(composite["mix"])
    require(0 < mix <= 1, "Invalid compositing weight")
    a = alpha.astype(np.float64)[..., None] / 255 * mix
    expected = np.rint(source * (1 - a) + raw * a).clip(0, 255).astype(np.uint8)
    error = np.abs(expected.astype(np.int16) - clean.astype(np.int16))
    require(error.max() <= 1, "Final image does not match independent protected-compositing calculation")
    changed = np.any(clean != source, axis=-1)
    require(changed.any(), "No image pixels changed")
    require(not (changed & ~support).any(), "Image change outside reviewed background region")
    for name, sample in masks["sample_points"].items():
        x, y = sample["xy"]
        require(int(alpha[y, x]) == sample["alpha"], f"Named sample alpha differs: {name}")
        if sample["alpha"] == 0:
            require(np.array_equal(clean[y, x], source[y, x]), f"Named source feature changed: {name}")
    require(generation["input_crop"] is None and generation["warp"] is None and
            composite["geometric_transform"] == "none", "Recorded crop/warp is outside this experiment's scope")
    require(not [p for p in candidate.rglob("*") if p.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".npz"}],
            "Candidate unexpectedly contains new 3D assets")
    y, x = np.where(changed)
    return {"status": "pixel_and_asset_integrity_passed_pending_visual_review",
            "verified_utc": datetime.now(timezone.utc).isoformat(), "baseline": str(baseline_path.resolve()),
            "candidate": str(candidate.resolve()), "unchanged_frozen_files": len(baseline["files"]),
            "source_file_exact": True, "only_reviewed_glass_background_changed": True,
            "source_pixels_exact_outside_support": int(protected.sum()),
            "named_feature_pixels_exact": int(features.sum()), "edit_support_pixels": int(support.sum()),
            "feature_protection_masks": feature_names,
            "changed_pixels": int(changed.sum()), "changed_pixels_outside_support": 0,
            "changed_bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
            "maximum_compositing_error_lsb": int(error.max()), "compositing_mix": mix,
            "reused_prediction_check": reused_prediction,
            "mask_manifest": str(mask_manifest_path), "mask_manifest_sha256": sha(mask_manifest_path),
            "prior_glass_annotation_sha256": sha(labels_path), "no_new_3d_files_in_candidate": True,
            "support_within_prior_full_windshield_polygon": True,
            "support_pixels_disagreeing_with_coarse_prior_glass_label": prior_label_difference,
            "coarse_label_disagreement_requires_visual_review": prior_label_difference > 0,
            "raw_uses_rgb_without_alpha_as_edit_mask": True,
            "scope": "Measured 2D pixel protection and preservation of all frozen existing assets. This is not proof of recovered cabin, transparency, or acceptable visual quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.baseline, args.candidate_dir)
    destination = args.report or args.candidate_dir / "independent_verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
