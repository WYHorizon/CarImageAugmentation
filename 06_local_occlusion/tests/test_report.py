"""Report checks use nontrivial coordinates and exact pixels, not screenshots."""

import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occlusion.report import build_report, region_box


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("src", "href"))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "results"
        self.output.mkdir()
        yy, xx = np.indices((257, 389))
        base = np.stack((xx % 251, yy % 251, (xx + 3 * yy) % 251), axis=-1).astype(np.uint8)
        changed = base.copy()
        changed[63:149, 79:200] = (0, 210, 40)
        alpha = np.zeros(base.shape[:2], dtype=np.uint8)
        alpha[63:149, 79:200] = 255
        self.original, self.edited = Image.fromarray(base), Image.fromarray(changed)
        self.source = self.root / '车辆 & "original".png'
        self.augmented, self.alpha = self.root / "edited.png", self.root / "alpha.png"
        self.original.save(self.source)
        self.edited.save(self.augmented)
        Image.fromarray(alpha).save(self.alpha)
        self.scene = {"label": '<Audi & "sample">', "subject_polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                      "regions": {"headlamp": [0.17, 0.19, 0.56, 0.62], "rear_wheel": [0.71, 0.56, 0.94, 0.91]}}
        self.statistics = {"vehicle": None, "opaque_alpha_threshold": 128, "regions": {}}
        for name, box in self.scene["regions"].items():
            crop = region_box(box, self.original.size)
            x0, y0, x1, y1 = crop
            region = alpha[y0:y1, x0:x1]
            self.statistics["regions"][name] = {"opaque_fraction": float((region >= 128).mean()),
                "weighted_fraction": float(region.mean() / 255), "crop_box_xyxy": list(crop)}

    def generate(self):
        return build_report(self.output, self.source, self.augmented, self.alpha, self.scene, self.statistics)

    def test_native_and_224_crops_match_actual_images(self):
        result = self.generate()
        self.assertEqual(result["native_crop_count"], 4)
        self.assertEqual(result["diagnostic_size"], [339, 224])
        metadata = json.loads((self.output / "crop_regions.json").read_text())
        for name, box in self.scene["regions"].items():
            for variant, image in (("original", self.original), ("occluded", self.edited)):
                expected_native = image.crop(region_box(box, image.size))
                with Image.open(self.output / f"crops/{variant}/{name}.png") as actual:
                    np.testing.assert_array_equal(actual, expected_native)
                resized = image.resize((339, 224), Image.Resampling.LANCZOS)
                # Independently derive outward coordinates at the rounded canvas size.
                expected_box = (math.floor(box[0] * 339), math.floor(box[1] * 224),
                                math.ceil(box[2] * 339), math.ceil(box[3] * 224))
                self.assertEqual(metadata["regions"][name]["diagnostic_box_xyxy"], list(expected_box))
                with Image.open(self.output / f"crops_224/{variant}/{name}.png") as actual:
                    np.testing.assert_array_equal(actual, resized.crop(expected_box))

    def test_html_escapes_titles_and_links_all_exist(self):
        before = self.source.read_bytes(), self.augmented.read_bytes(), self.alpha.read_bytes()
        self.generate()
        document = (self.output / "report/index.html").read_text()
        self.assertIn(html.escape(self.scene["label"]), document)
        self.assertNotIn('<Audi & "sample">', document)
        parser = LinkParser()
        parser.feed(document)
        self.assertGreaterEqual(len(parser.links), 20)
        for link in parser.links:
            self.assertTrue((self.output / "report" / unquote(link)).is_file(), link)
        self.assertEqual(before, (self.source.read_bytes(), self.augmented.read_bytes(), self.alpha.read_bytes()))

    def test_wrong_coordinates_and_unsafe_names_fail(self):
        self.statistics["regions"]["headlamp"]["crop_box_xyxy"][0] += 1
        with self.assertRaisesRegex(ValueError, "coordinates disagree"):
            self.generate()
        self.scene["regions"] = {"../outside": [0, 0, 1, 1]}
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.generate()

    def test_mismatched_canvas_fails(self):
        Image.new("L", (10, 10)).save(self.alpha)
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            self.generate()


if __name__ == "__main__":
    unittest.main()
