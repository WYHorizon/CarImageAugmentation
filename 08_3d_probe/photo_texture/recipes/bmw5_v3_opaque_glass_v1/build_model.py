"""Compose the opaque window textures on the original BMW 5-series v3 asset."""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
PHOTO = OUTPUTS.parent
sys.path.insert(0, str(PHOTO.parents[1] / "09_reflection_cleanup"))
from replace_frozen_v5_texture import parse_glb, patch_glb, sha256

import front_side
import rear_side


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def embedded_rgb(data):
    tree, binary = parse_glb(data)
    view = tree["bufferViews"][tree["images"][0]["bufferView"]]
    offset = view.get("byteOffset", 0)
    with Image.open(io.BytesIO(binary[offset:offset + view["byteLength"]])) as image:
        return np.array(image.convert("RGB"))


def main():
    baseline = OUTPUTS / "bmw5_v5_hood_refined_v3"
    original = (baseline / "textured_mesh.glb").read_bytes()
    base_hash = sha256(baseline / "textured_mesh.glb")
    assert base_hash == json.loads((baseline / "verification.json").read_text())["glb_sha256"]
    base = embedded_rgb(original)
    assert np.array_equal(base, read_rgb(baseline / "texture.png"))
    prior = read_rgb(OUTPUTS / "bmw5_v5_seam_refined_v2/texture.png")
    protected = np.array(Image.open(baseline / "hood_footprint.png").convert("L")) > 0
    protected |= np.any(prior != base, axis=2)
    base.setflags(write=False)
    texture = base.copy()
    regions = np.zeros(base.shape[:2], dtype=np.uint8)
    valid = np.asarray(Image.open(OUTPUTS / "material_regions_v1/atlas_valid_mask.png")) > 0
    with np.load(OUTPUTS / "bmw5_mv_textured_v5/textured_mesh.npz") as mesh:
        vertices, faces, uv = (mesh[k] for k in ("vertices", "faces", "uv"))
        for module, expected in [(front_side, (1, 2)), (rear_side, (3, 4))]:
            pixels, ids = module.apply(base, vertices, faces, uv)
            assert pixels.shape == texture.shape and ids.shape == regions.shape
            assert set(np.unique(ids)) == {0, *expected}
            assert np.array_equal(pixels[ids == 0], base[ids == 0])
            target = (ids > 0) & valid & ~protected
            assert not np.any(regions[target]), "Window footprints overlap"
            texture[target] = pixels[target]
            regions[target] = ids[target]

    # Extend each affected texture island into its existing gutter for filtering.
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    padding = ~valid & ~protected & (distance <= 12) & (regions[tuple(nearest)] != 0)
    texture[padding] = texture[nearest[0][padding], nearest[1][padding]]
    regions[padding] = 5
    changed = np.any(base != texture, axis=2)
    assert not np.any(changed & protected)
    assert not np.any(changed & (regions == 0))
    names = ("front_windshield", "left_windows", "right_windows", "rear_windshield")
    counts = {name: int(np.count_nonzero(changed & (regions == i))) for i, name in enumerate(names, 1)}
    assert all(count > 1000 for count in counts.values())
    Image.fromarray(regions).save(HERE / "window_ids.png")
    Image.fromarray(texture).save(HERE / "texture.png")
    packed, check = patch_glb(original, (HERE / "texture.png").read_bytes())
    assert check["geometry_normals_uv_scene_material_metadata_exact"] and check["embedded_png_exact"]
    assert np.array_equal(embedded_rgb(packed), texture)
    staging = HERE / "textured_mesh.glb.tmp"
    staging.write_bytes(packed)
    os.replace(staging, HERE / "textured_mesh.glb")

    report = {
        "base_glb_sha256": base_hash,
        "glb_sha256": sha256(HERE / "textured_mesh.glb"),
        "texture_sha256": sha256(HERE / "texture.png"),
        "changed_surface_texels": counts,
        "changed_padding_texels": int(np.count_nonzero(changed & padding)),
        "changes_outside_windows_or_adjacent_gutter": 0,
        "changes_on_hood_or_prior_v3_refinements": 0,
        "geometry_normals_uv_materials_unchanged": True,
        "embedded_texture_decodes_to_verified_pixels": True,
        "method": "Opaque smoke-grey appearance baked into the original v3 window areas. Shading is fixed in texture coordinates.",
    }
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
