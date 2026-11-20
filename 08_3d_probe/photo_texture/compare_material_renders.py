"""Assemble equal-camera material comparisons from actual WebGL render files."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.render_dir
    report = json.loads((folder / "browser_verification.json").read_text())
    states = {item["file"]: item["state"] for item in report["renders"]}
    angles = [30, 90, 180, 270]
    sheet = Image.new("RGB", (1200, 4 * 475), "white")
    draw = ImageDraw.Draw(sheet)
    for row, angle in enumerate(angles):
        names = [f"photo_az_{angle:03d}.png", f"pbr_az_{angle:03d}.png"]
        for key in ("yaw", "elevation", "zoom", "environment", "environmentYaw"):
            if states[names[0]][key] != states[names[1]][key]:
                raise ValueError(f"Comparison states differ: {key}")
        for column, name in enumerate(names):
            image = Image.open(folder / name).convert("RGB")
            if image.size != (1200, 900):
                raise ValueError(f"Unexpected image dimensions: {name}")
            sheet.paste(image.resize((600, 450), Image.Resampling.LANCZOS), (column * 600, row * 475 + 25))
            label = "v5 fixed photo appearance" if column == 0 else "v6 approximate PBR + protected photo details"
            draw.text((column * 600 + 10, row * 475 + 6), f"{label} | az {angle}, elevation 8", fill="black")
    sheet.save(folder / "comparison.png")
    metrics = {}
    for mode in ("photo", "pbr"):
        first = np.asarray(Image.open(folder / f"{mode}_light_0.png").convert("RGB"), np.int16)
        second = np.asarray(Image.open(folder / f"{mode}_light_90.png").convert("RGB"), np.int16)
        difference = np.abs(first - second)
        metrics[mode] = {"exactly_equal": bool(np.array_equal(first, second)),
                         "changed_pixels": int((difference.max(axis=2) > 0).sum()),
                         "pixels_changed_gt10": int((difference.max(axis=2) > 10).sum()),
                         "whole_image_rgb_mae_0_255": float(difference.mean())}
    (folder / "light_change_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
