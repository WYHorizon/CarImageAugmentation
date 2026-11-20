"""Pack the tested material asset into a self-contained chat preview fragment."""

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from material_normals import sample_normal_field


def encode_bytes(data):
    return base64.b64encode(data).decode("ascii")


def data_image(image, fmt, **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=fmt, **kwargs)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + encode_bytes(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--photo-texture", type=Path, required=True)
    parser.add_argument("--preview-mesh", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("material_preview_template.html"))
    parser.add_argument("--three-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=64)
    args = parser.parse_args()
    with np.load(args.preview_mesh) as arrays:
        vertices, faces, uv = [arrays[k] for k in ("vertices", "faces", "uv")]
    if len(vertices) > 65535:
        raise ValueError("Compact preview expects uint16 vertex indices")
    minimum, span = vertices.min(axis=0), np.ptp(vertices, axis=0)
    quantized = np.rint((vertices - minimum) / np.maximum(span, 1e-10) * 65535).astype("<u2")
    quantized_uv = np.rint(np.clip(uv, 0, 1) * 65535).astype("<u2")
    geometry = b"".join([quantized.tobytes(), quantized_uv.tobytes(), faces.astype("<u2").tobytes()])
    # Transfer the full-resolution normal field instead of recomputing it on
    # the simplified preview, where large triangles exaggerate specular ripples.
    with np.load(args.material_dir / "mesh.npz") as full:
        normals, normal_info = sample_normal_field(vertices, full["vertices"], full["normals"], return_info=True)
    # 255 levels/component reduce preview payload entropy. Keep the int16 wire
    # format; the shader normalizes after interpolation. Full GLB uses float32.
    quantized_normals = np.rint(np.rint(normals * 127) / 127 * 32767).astype("<i2")
    decoded_normals = quantized_normals.astype(np.float64) / 32767
    decoded_normals /= np.linalg.norm(decoded_normals, axis=1, keepdims=True)
    reference_normals = normals.astype(np.float64)
    reference_normals /= np.linalg.norm(reference_normals, axis=1, keepdims=True)
    normal_angle_error = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", decoded_normals, reference_normals), -1, 1)))
    # JPEG compresses only the legacy photograph. Categorical IDs remain lossless
    # nearest-neighbor PNG, and all new material colors are exact explicit RGBs.
    photo = Image.open(args.photo_texture).convert("RGB").resize(
        (args.texture_size, args.texture_size), Image.Resampling.LANCZOS)
    ids = Image.open(args.material_dir / "material_ids.png").convert("L").resize(
        photo.size, Image.Resampling.NEAREST)
    manifest = json.loads((args.material_dir / "manifest.json").read_text())
    labels = {key: {"pbr": True, "color": material["rgb"],
                    **{k:material[k] for k in ("roughness", "metalness", "clearcoat")}}
              for key, material in manifest["material_parameters"].items()}
    data = {
        "vertexCount": len(vertices), "faceCount": len(faces),
        "minimum": minimum.tolist(), "span": span.tolist(),
        "geometry": encode_bytes(gzip.compress(geometry, compresslevel=9, mtime=0)),
        "normals": encode_bytes(gzip.compress(quantized_normals.tobytes(), compresslevel=9, mtime=0)),
        "oldTexture": data_image(photo, "JPEG", quality=args.jpeg_quality, optimize=True),
        "materialIds": data_image(ids, "PNG", optimize=True),
        "materialConfig": {"textureSize": args.texture_size, "clearcoatRoughness": manifest["clearcoatRoughness"], "labels": labels},
    }
    template = args.template.read_text()
    for token in ("__MATERIAL_DATA_JSON__", "__MATERIAL_THREE_BUNDLE__"):
        if template.count(token) != 1:
            raise ValueError(f"Template must contain exactly one {token}")
    fragment = template.replace("__MATERIAL_DATA_JSON__", json.dumps(data, separators=(",", ":")))
    fragment = fragment.replace("__MATERIAL_THREE_BUNDLE__", args.three_bundle.read_text())
    size = len(fragment.encode("utf8"))
    if size >= 1_000_000:
        raise ValueError(f"Fragment too large ({size} bytes): reduce JPEG size/quality or mesh complexity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    record = {
        "fragment": str(args.output.resolve()), "fragment_bytes": size,
        "sha256": hashlib.sha256(fragment.encode()).hexdigest(),
        "vertices": len(vertices), "faces": len(faces), "texture_size": args.texture_size,
        "jpeg_quality": args.jpeg_quality, "preview_normal_repair": normal_info,
        "geometry_bytes_compressed": len(base64.b64decode(data["geometry"])),
        "normal_bytes_compressed": len(base64.b64decode(data["normals"])),
        "max_normal_quantization_error_degrees": float(normal_angle_error.max()),
        "approximation": "Existing 15k preview geometry and reduced photo texture; full GLB retains original 80k geometry and 2048 textures.",
    }
    (args.material_dir / "preview_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
