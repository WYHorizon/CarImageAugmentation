"""Verify lossless comparisons, exact crops, and links for quality reports."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageOps

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from quality.report import build_report, report_groups


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs if name in {"href", "src"})


class ReportTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source image.png"
        Image.new("RGB", (80, 60), (40, 50, 60)).save(source)
        presets, outputs, colors = {}, {}, {}
        for index, (name, group) in enumerate([
            ("one", "single"), ("two", "triple"),
            ("three", "triple"), ("four", "triple"),
        ]):
            presets[name] = {"label": f"Variant {index + 1}", "kind": "defocus",
                             "group": group, "group_label": group + " <test>"}
            colors[name] = (80 + index * 25, 80 + index * 20, 90 + index * 15)
            path = root / f"generated_{name}.png"
            Image.new("RGB", (80, 60), colors[name]).save(path)
            outputs[name] = path
        rows = [{"source": source, "vehicle_id": "test_car", "label": "Test <car>",
                 "scene": {"regions": {"headlamp": [0.11, 0.21, 0.36, 0.46]}},
                 "outputs": outputs}]
        return rows, presets, colors

    def test_groups_png_columns_links_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, presets, colors = self.fixture(root)
            source_paths = [rows[0]["source"], *rows[0]["outputs"].values()]
            before_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
            output = root / "report output"
            output.mkdir()
            (output / "manifest.json").write_text("{}\n")
            report = build_report(output, rows, presets)
            self.assertEqual(report["native_crop_count"], 5)
            self.assertEqual(report["preview_png_count"], 6)
            self.assertFalse(list(output.rglob("*.jpg")))
            for group, members in [("single", ["one"]), ("triple", ["two", "three", "four"])]:
                with Image.open(output / "previews" / f"{group}_overview.png") as sheet:
                    self.assertEqual(sheet.format, "PNG")
                    self.assertEqual(sheet.size, (520 * (len(members) + 1), 452))
                    for column, name in enumerate(members, start=1):
                        self.assertEqual(sheet.getpixel((column * 520 + 260, 257)), colors[name])
            html_path = output / report["report"]
            page = html_path.read_text(encoding="utf-8")
            self.assertIn("1 张原图 × 4 种预设", page)
            self.assertIn("triple &lt;test&gt;", page)
            self.assertIn("无损 PNG", page)
            self.assertIn("不混入额外降采样", page)
            self.assertIn("缩小会削弱", page)
            links = LinkCollector()
            links.feed(page)
            self.assertGreater(len(links.targets), 10)
            for target in links.targets:
                if target.startswith("#"):
                    continue
                self.assertTrue((html_path.parent / unquote(target)).exists(), target)
            self.assertEqual(before_hashes, {
                path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
            })

    def test_native_crops_exact_after_exif_and_224_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, presets, _ = self.fixture(root)
            rng = np.random.default_rng(12)
            source_pixels = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
            exif = Image.Exif()
            exif[274] = 6
            Image.fromarray(source_pixels).save(rows[0]["source"], exif=exif)
            for path in rows[0]["outputs"].values():
                Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(path)
            output = root / "report output"
            build_report(output, rows, presets)
            metadata = json.loads((output / "crop_regions.json").read_text())
            record = metadata["regions"][0]
            self.assertEqual(record["native_image_size"], [80, 60])
            self.assertEqual(record["native_box_xyxy"], [8, 12, 29, 28])
            self.assertEqual(record["native_crop_size"], [21, 16])
            self.assertEqual(record["short_edge_224_size"], [299, 224])
            self.assertEqual(record["short_edge_224_box_xyxy"], [32, 47, 108, 104])
            for name, path in {"original": rows[0]["source"], **rows[0]["outputs"]}.items():
                with Image.open(path) as image:
                    expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[12:28, 8:29]
                with Image.open(output / "crops/test_car/headlamp" / f"{name}.png") as crop:
                    np.testing.assert_array_equal(np.asarray(crop), expected)

    def test_group_metadata_rejected_before_artifact_creation(self):
        invalid_metadata = [
            {"group": "../../outside"}, {"group": "a/b"}, {"group": ""},
            {"group": 123}, {"group": "valid", "group_label": ""},
            {"group_label": "Missing group"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, presets, _ = self.fixture(root)
            output = root / "should_not_exist"
            for metadata in invalid_metadata:
                with self.subTest(metadata=metadata):
                    bad_presets = dict(presets)
                    bad_presets["one"] = {"label": "Bad", "kind": "defocus", **metadata}
                    with self.assertRaises(ValueError):
                        build_report(output, rows, bad_presets)
                    self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                report_groups({
                    "one": {"group": "same", "group_label": "First"},
                    "two": {"group": "same", "group_label": "Second"},
                })

    def test_group_order_follows_presets(self):
        groups = report_groups({
            "first": {"group": "noise", "group_label": "Noise"},
            "second": {"group": "blur", "group_label": "Blur"},
            "third": {"group": "blur", "group_label": "Blur"},
        })
        self.assertEqual(list(groups), ["noise", "blur"])
        self.assertEqual(groups["blur"], ("Blur", ["second", "third"]))


if __name__ == "__main__":
    unittest.main()
