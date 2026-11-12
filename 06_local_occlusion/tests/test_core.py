import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occlusion.core import compose_occlusion, compute_visibility


class CompositingTests(unittest.TestCase):
    def test_opaque_rectangle_and_exposed_source_pixels(self):
        rng = np.random.default_rng(4)
        source = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
        asset = Image.new("RGBA", (10, 10), (30, 160, 70, 255))
        result = compose_occlusion(Image.fromarray(source), asset,
                                   width_fraction=0.2, position_xy=(0.25, 0.25))
        output, alpha = np.asarray(result.image), np.asarray(result.alpha)
        np.testing.assert_array_equal(output[alpha == 0], source[alpha == 0])
        self.assertTrue(np.all(output[alpha == 255] == [30, 160, 70]))
        self.assertEqual(result.alpha.getbbox(), (15, 10, 27, 22))

    def test_off_canvas_clipping(self):
        result = compose_occlusion(Image.new("RGB", (20, 20)),
                                   Image.new("RGBA", (10, 10), (200, 80, 10, 255)),
                                   width_fraction=0.5, position_xy=(-0.25, 0.25))
        self.assertEqual(result.alpha.getbbox(), (0, 5, 5, 15))
        self.assertEqual(np.count_nonzero(np.asarray(result.alpha)), 50)

    def test_hidden_white_rgb_does_not_produce_halos(self):
        rgba = np.full((12, 12, 4), 255, dtype=np.uint8)
        rgba[..., 3] = 0
        rgba[3:9, 3:9] = (0, 255, 0, 255)
        result = compose_occlusion(Image.new("RGB", (80, 80)), Image.fromarray(rgba),
                                   width_fraction=0.65, position_xy=(0.05, 0.05),
                                   angle_degrees=23, blur_radius=2)
        output = np.asarray(result.image)
        self.assertEqual(int(output[..., 0].max()), 0)
        self.assertEqual(int(output[..., 2].max()), 0)
        self.assertGreater(int(output[..., 1].max()), 200)
        np.testing.assert_array_equal(output[..., 1], np.asarray(result.alpha))

    def test_half_alpha_blends_instead_of_erasing(self):
        result = compose_occlusion(Image.new("RGB", (10, 10), (20, 40, 60)),
                                   Image.new("RGBA", (10, 10), (220, 140, 100, 128)),
                                   width_fraction=1, position_xy=(0, 0))
        self.assertTrue(np.all(np.abs(np.asarray(result.image)[0, 0].astype(int)
                                     - [120, 90, 80]) <= 1))

    def test_flip_rotation_and_blur_preserve_inputs_and_are_deterministic(self):
        source = Image.new("RGB", (70, 60), (60, 80, 100))
        rgba = np.zeros((12, 20, 4), dtype=np.uint8)
        rgba[2:10, 2:18] = [150, 200, 40, 255]
        asset = Image.fromarray(rgba)
        old_source, old_asset = source.tobytes(), asset.tobytes()
        kwargs = dict(width_fraction=0.8, position_xy=(-0.1, 0.2), angle_degrees=-17,
                      flip_horizontal=True, blur_radius=1.2, brightness=0.8, saturation=0.9)
        a, b = compose_occlusion(source, asset, **kwargs), compose_occlusion(source, asset, **kwargs)
        self.assertEqual(a.image.tobytes(), b.image.tobytes())
        self.assertEqual(a.alpha.tobytes(), b.alpha.tobytes())
        self.assertEqual(source.tobytes(), old_source)
        self.assertEqual(asset.tobytes(), old_asset)
        output, alpha, original = np.asarray(a.image), np.asarray(a.alpha), np.asarray(source)
        np.testing.assert_array_equal(output[alpha == 0], original[alpha == 0])

    def test_positive_angle_is_counterclockwise(self):
        rgba = np.zeros((2, 6, 4), dtype=np.uint8)
        rgba[:, :3] = [255, 0, 0, 255]
        rgba[:, 3:] = [0, 0, 255, 255]
        result = compose_occlusion(Image.new("RGB", (10, 10)), Image.fromarray(rgba),
                                   width_fraction=0.6, position_xy=(0, 0), angle_degrees=90)
        self.assertEqual(result.image.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result.image.getpixel((0, 5)), (255, 0, 0))

    def test_invalid_or_invisible_assets_fail(self):
        base = Image.new("RGB", (10, 10))
        asset = Image.new("RGBA", (10, 10), (0, 100, 0, 255))
        for wrong in [Image.new("RGB", (10, 10)), Image.new("RGBA", (10, 10))]:
            with self.assertRaises(ValueError):
                compose_occlusion(base, wrong, width_fraction=1, position_xy=(0, 0))
        for kwargs in [dict(width_fraction=float("nan")), dict(width_fraction=-1),
                       dict(position_xy=(5, 5)), dict(blur_radius=-1), dict(brightness=True)]:
            parameters = dict(width_fraction=1, position_xy=(0, 0)) | kwargs
            with self.assertRaises(ValueError):
                compose_occlusion(base, asset, **parameters)


class VisibilityTests(unittest.TestCase):
    def test_pixel_center_polygon_and_part_rectangles(self):
        alpha = np.zeros((10, 10), dtype=np.uint8)
        alpha[:, :5] = 255
        scene = {"subject_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                 "regions": {"middle": [0.2, 0.2, 0.8, 0.8], "clear": [0.5, 0, 1, 1]}}
        result = compute_visibility(Image.fromarray(alpha), scene)
        self.assertEqual(result.statistics["vehicle"]["pixel_count"], 100)
        self.assertEqual(result.statistics["vehicle"]["opaque_fraction"], 0.5)
        self.assertEqual(result.statistics["vehicle"]["weighted_fraction"], 0.5)
        self.assertEqual(result.statistics["regions"]["middle"]["crop_box_xyxy"], [2, 2, 8, 8])
        self.assertEqual(result.statistics["regions"]["middle"]["pixel_count"], 36)
        self.assertEqual(result.statistics["regions"]["middle"]["opaque_fraction"], 0.5)
        self.assertEqual(result.statistics["regions"]["clear"]["visible_fraction"], 1)
        self.assertEqual(np.count_nonzero(np.asarray(result.vehicle_visible_mask)), 50)

    def test_soft_edges_have_distinct_binary_and_weighted_metrics(self):
        alpha = Image.new("L", (8, 8), 128)
        scene = {"regions": {"all": [0, 0, 1, 1]}}
        result = compute_visibility(alpha, scene)
        self.assertIsNone(result.vehicle_mask)
        self.assertIsNone(result.statistics["vehicle"])
        self.assertEqual(result.statistics["regions"]["all"]["opaque_fraction"], 1)
        self.assertAlmostEqual(result.statistics["regions"]["all"]["weighted_fraction"], 128 / 255)
        other = compute_visibility(alpha, scene, opaque_alpha_threshold=129)
        self.assertEqual(other.statistics["regions"]["all"]["opaque_fraction"], 0)

    def test_subject_rectangle_uses_half_open_pixel_centers(self):
        result = compute_visibility(Image.new("L", (10, 10), 0),
                                    {"subject_polygon": [[0.2, 0.2], [0.8, 0.2],
                                                         [0.8, 0.8], [0.2, 0.8]]})
        self.assertEqual(result.statistics["vehicle"]["pixel_count"], 36)
        self.assertEqual(result.vehicle_mask.getbbox(), (2, 2, 8, 8))

    def test_invalid_annotations_fail(self):
        alpha = Image.new("L", (10, 10))
        for scene in [{"regions": {"bad": [0.2, 0.2, 0.1, 0.3]}},
                      {"subject_polygon": [[0, 0], [1, float("nan")], [0, 1]]},
                      {"subject_polygon": [[0, 0], [0, 0], [0, 0]]}]:
            with self.assertRaises(ValueError):
                compute_visibility(alpha, scene)


if __name__ == "__main__":
    unittest.main()
