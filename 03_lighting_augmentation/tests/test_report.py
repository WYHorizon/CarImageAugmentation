"""Verify arbitrary report groups and linked artifacts without losing columns."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np
from PIL import Image

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from lighting.report import build_report, report_groups


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
            presets[name] = {"label": f"Variant {index + 1}", "kind": "optical",
                             "group": group, "group_label": group + " <test>"}
            colors[name] = (80 + index * 25, 80 + index * 20, 90 + index * 15)
            path = root / f"generated_{name}.png"
            Image.new("RGB", (80, 60), colors[name]).save(path)
            outputs[name] = path
        rows = [{"source": source, "vehicle_id": "test_car", "label": "Test <car>",
                 "scene": {"regions": {"headlamp": [0.1, 0.2, 0.35, 0.45]}},
                 "outputs": outputs}]
        return rows, presets, colors

    def test_single_and_three_member_groups_keep_columns_and_links(self):
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
            self.assertEqual(report["preview_jpeg_count"], 6)
            for group, members in [("single", ["one"]), ("triple", ["two", "three", "four"])]:
                with Image.open(output / "previews" / f"{group}_overview.jpg") as sheet:
                    self.assertEqual(sheet.size, (520 * (len(members) + 1), 452))
                    for column, name in enumerate(members, start=1):
                        # The center of each comparison cell must contain its
                        # corresponding image, not an omitted/reordered variant.
                        actual = sheet.getpixel((column * 520 + 260, 257))
                        np.testing.assert_allclose(actual, colors[name], atol=3)
            html_path = output / report["report"]
            page = html_path.read_text(encoding="utf-8")
            self.assertIn("1 张原图 × 4 种预设", page)
            self.assertIn("triple &lt;test&gt;", page)
            self.assertIn("二维图像空间", page)
            self.assertNotIn("四车六预设", page)
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
                    bad_presets["one"] = {"label": "Bad", "kind": "optical", **metadata}
                    with self.assertRaises(ValueError):
                        build_report(output, rows, bad_presets)
                    self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                report_groups({
                    "one": {"group": "same", "group_label": "First"},
                    "two": {"group": "same", "group_label": "Second"},
                })

    def test_basic_groups_remain_compatible_and_explicit_metadata_wins(self):
        presets = json.loads((METHOD / "configs/presets.json").read_text(encoding="utf-8"))
        groups = report_groups(presets)
        self.assertEqual(list(groups), ["exposure", "contrast", "white_balance"])
        self.assertEqual(groups["exposure"][1], ["exposure_down", "exposure_up"])
        presets["exposure_down"]["group"] = "custom"
        groups = report_groups(presets)
        self.assertEqual(groups["custom"][1], ["exposure_down"])
        self.assertNotIn("exposure", groups)


if __name__ == "__main__":
    unittest.main()
