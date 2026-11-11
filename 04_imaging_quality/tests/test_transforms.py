"""Behavioral checks for independent imaging-quality transforms."""

from __future__ import annotations

import copy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, JpegImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality import apply_quality, validate_preset
from quality.transforms import linear_to_srgb, srgb_to_linear


PRESETS = json.loads((Path(__file__).resolve().parents[1] / "configs/presets.json")
                     .read_text(encoding="utf-8"))


class QualityTransformsTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(51).integers(0, 256, (96, 128, 3),
                                                    dtype=np.uint8)

    def test_shipped_presets_are_valid_and_serializable(self):
        self.assertEqual(len(PRESETS), 8)
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                validate_preset(preset)
                result = apply_quality(self.rgb, preset, seed=91)
                self.assertEqual(result.image.shape, self.rgb.shape)
                self.assertEqual(result.image.dtype, np.uint8)
                self.assertEqual(result.metadata["input_size"], [128, 96])
                self.assertEqual(result.metadata["output_size"], [128, 96])
                json.dumps(result.metadata, allow_nan=False)

    def test_no_transform_mutates_or_aliases_input(self):
        before = self.rgb.copy()
        presets_before = copy.deepcopy(PRESETS)
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                result = apply_quality(self.rgb, preset, 123)
                np.testing.assert_array_equal(self.rgb, before)
                self.assertFalse(np.shares_memory(self.rgb, result.image))
                if result.native_image is not None:
                    self.assertFalse(np.shares_memory(self.rgb, result.native_image))
                    self.assertFalse(np.shares_memory(result.image, result.native_image))
        self.assertEqual(PRESETS, presets_before)

    def test_every_operation_is_repeatable(self):
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                first, second = (apply_quality(self.rgb, preset, 779) for _ in range(2))
                np.testing.assert_array_equal(first.image, second.image)
                self.assertEqual(first.metadata, second.metadata)
                self.assertEqual(first.encoded_jpeg, second.encoded_jpeg)

    def test_blurs_preserve_flat_fields_including_singleton_axes(self):
        for name in ("defocus_mild", "defocus_strong", "motion_blur"):
            preset = dict(PRESETS[name], reference_short_edge=32)
            for shape in ((96, 128, 3), (1, 49, 3), (49, 1, 3)):
                with self.subTest(preset=name, shape=shape):
                    rgb = np.empty(shape, dtype=np.uint8)
                    rgb[:] = [27, 128, 244]
                    np.testing.assert_array_equal(apply_quality(rgb, preset).image, rgb)

    def test_defocus_scales_relative_to_short_edge(self):
        preset = dict(PRESETS["defocus_mild"], reference_short_edge=32)
        first = apply_quality(self.rgb, preset)
        second = apply_quality(self.rgb[::2, ::2], preset)
        self.assertEqual(first.metadata["effective_sigma_target_px"], 6)
        self.assertEqual(second.metadata["effective_sigma_target_px"], 3)
        self.assertTrue(all(width % 2 == 1 for width in first.metadata["box_widths_px"]))
        self.assertGreater(first.metadata["effective_sigma_kernel_px"],
                           second.metadata["effective_sigma_kernel_px"])

    def test_zero_sigma_is_identity(self):
        preset = dict(PRESETS["defocus_mild"], sigma=0)
        result = apply_quality(self.rgb, preset)
        np.testing.assert_array_equal(result.image, self.rgb)
        self.assertEqual(result.metadata["effective_sigma_kernel_px"], 0)

    def test_strong_defocus_removes_more_high_frequency_contrast(self):
        yy, xx = np.indices((96, 128))
        # Broad checker squares retain measurable signal after the mild blur.
        checker = (((xx // 6 + yy // 6) % 2) * 255).astype(np.uint8)
        rgb = np.repeat(checker[..., None], 3, axis=2)
        mild = apply_quality(rgb, dict(PRESETS["defocus_mild"], reference_short_edge=96))
        strong = apply_quality(rgb, dict(PRESETS["defocus_strong"], reference_short_edge=96))
        self.assertLess(np.std(strong.image[20:-20, 20:-20]),
                        np.std(mild.image[20:-20, 20:-20]))

    def test_blur_mixes_light_in_linear_space(self):
        rgb = np.zeros((65, 65, 3), dtype=np.uint8)
        rgb[:, 33:] = 255
        result = apply_quality(rgb, dict(PRESETS["defocus_mild"], reference_short_edge=65))
        # A transition near 50% light encodes substantially brighter than sRGB128.
        self.assertGreater(int(result.image[32, 32, 0]), 160)

    def test_motion_preserves_impulse_center_and_direction(self):
        rgb = np.zeros((81, 81, 3), dtype=np.uint8)
        rgb[40, 40] = 255
        preset = dict(PRESETS["motion_blur"], reference_short_edge=81, length=25)
        result = apply_quality(rgb, preset)
        plane = srgb_to_linear(result.image[..., 0].astype(np.float32) / 255)
        yy, xx = np.indices(plane.shape)
        self.assertAlmostEqual(float(np.sum(xx * plane) / plane.sum()), 40, places=5)
        self.assertAlmostEqual(float(np.sum(yy * plane) / plane.sum()), 40, places=5)
        np.testing.assert_array_equal(result.image, result.image[::-1, ::-1])
        variance_x = np.sum((xx - 40) ** 2 * plane)
        variance_y = np.sum((yy - 40) ** 2 * plane)
        self.assertGreater(variance_x, variance_y * 5)
        self.assertGreater(np.sum((xx - 40) * (yy - 40) * plane), 0)
        self.assertTrue(all(size % 2 == 1 for size in result.metadata["kernel_size"]))

    def test_motion_does_not_wrap_across_image_boundary(self):
        rgb = np.zeros((49, 81, 3), dtype=np.uint8)
        rgb[24, 0] = 255
        preset = dict(PRESETS["motion_blur"], reference_short_edge=49, length=15, angle=0)
        result = apply_quality(rgb, preset)
        self.assertGreater(result.image[:, :10].sum(), 0)
        self.assertEqual(result.image[:, 20:].sum(), 0)

    def test_noise_changes_with_seed_and_records_preclip_diagnostics(self):
        preset = PRESETS["sensor_noise"]
        first = apply_quality(self.rgb, preset, 101)
        second = apply_quality(self.rgb, preset, 102)
        self.assertFalse(np.array_equal(first.image, second.image))
        self.assertTrue(first.metadata["seed_used"])
        self.assertLess(first.metadata["unclipped_linear_min"], 0)
        self.assertGreater(first.metadata["unclipped_linear_max"], 1)
        for key in ("clipped_below_zero_fraction_channels", "clipped_above_one_fraction_channels"):
            self.assertGreater(first.metadata[key], 0)
            self.assertLess(first.metadata[key], 0.2)

    def test_noise_shot_variance_grows_with_signal(self):
        values = []
        preset = dict(PRESETS["sensor_noise"], read_sigma=0)
        for gray in (56, 170):
            rgb = np.full((160, 160, 3), gray, dtype=np.uint8)
            result = apply_quality(rgb, preset, 94)
            linear = srgb_to_linear(result.image.astype(np.float32) / 255)
            values.append(float(np.var(linear)))
        self.assertGreater(values[1], values[0] * 6)

    def test_default_noise_preserves_legacy_pixels_and_native_grid_equivalence(self):
        preset = PRESETS["sensor_noise"]
        legacy = apply_quality(self.rgb, preset, 101)
        # Captured from the released independent-noise algorithm before adding
        # the optional grain grid; verifies that existing runs remain reproducible.
        self.assertEqual(hashlib.sha256(legacy.image.tobytes()).hexdigest(),
                         "0beab58e9754fb4460c8d332a5358f86757d3877e393a46b803efa835c1cf773")
        for edge in (96, 192):
            result = apply_quality(self.rgb, dict(preset, grain_short_edge=edge), 101)
            np.testing.assert_array_equal(result.image, legacy.image)
            self.assertFalse(result.metadata["grain_resampled"])
            self.assertEqual(result.metadata["grain_grid_size"], [128, 96])

    def test_coarse_noise_is_reproducible_and_keeps_inputs_unchanged(self):
        preset = dict(PRESETS["sensor_noise"], peak=24, read_sigma=0.025,
                      grain_short_edge=24)
        rgb_before, preset_before = self.rgb.copy(), copy.deepcopy(preset)
        first, second = (apply_quality(self.rgb, preset, 101) for _ in range(2))
        np.testing.assert_array_equal(first.image, second.image)
        self.assertEqual(first.metadata, second.metadata)
        self.assertFalse(np.array_equal(first.image, apply_quality(self.rgb, preset, 102).image))
        np.testing.assert_array_equal(self.rgb, rgb_before)
        self.assertEqual(preset, preset_before)
        self.assertFalse(np.shares_memory(self.rgb, first.image))
        self.assertEqual(first.metadata["grain_grid_size"], [32, 24])
        self.assertTrue(first.metadata["grain_resampled"])
        self.assertEqual(first.metadata["grain_residual_restore_filter"], "BILINEAR")
        json.dumps(first.metadata, allow_nan=False)

    def test_coarse_noise_survives_reduction_and_has_spatial_correlation(self):
        rgb = np.full((896, 1196, 3), 128, dtype=np.uint8)
        preset = dict(PRESETS["sensor_noise"], peak=24, read_sigma=0.025)
        independent = apply_quality(rgb, preset, 76).image
        coarse = apply_quality(rgb, dict(preset, grain_short_edge=112), 76).image

        def reduced_error(image):
            reduced = np.asarray(Image.fromarray(image).resize((299, 224), Image.Resampling.LANCZOS))
            return np.mean(np.abs(reduced.astype(np.float32) - 128))

        self.assertGreater(reduced_error(coarse), reduced_error(independent) * 2)

        def neighboring_correlation(image):
            field = image[..., 0].astype(np.float32) - 128
            return np.corrcoef(field[:, 1:].ravel(), field[:, :-1].ravel())[0, 1]

        self.assertGreater(neighboring_correlation(coarse), 0.85)
        self.assertLess(abs(neighboring_correlation(independent)), 0.02)

    def test_coarse_noise_does_not_resize_or_blur_original_signal(self):
        yy, xx = np.indices((96, 128))
        checker = np.where((xx + yy) % 2, 50, 200).astype(np.uint8)
        rgb = np.repeat(checker[..., None], 3, axis=2)
        # Negligible noise exposes unintended signal resampling: coarse signal
        # restoration would destroy this one-pixel checkerboard.
        preset = dict(PRESETS["sensor_noise"], peak=1e9, read_sigma=0,
                      grain_short_edge=12)
        np.testing.assert_array_equal(apply_quality(rgb, preset, 97).image, rgb)

    def test_low_resolution_retains_native_pixels_and_input_dimensions(self):
        preset = dict(PRESETS["lowres_720"], short_edge=24)
        result = apply_quality(self.rgb, preset)
        self.assertEqual(result.native_image.shape, (24, 32, 3))
        self.assertEqual(result.image.shape, self.rgb.shape)
        self.assertEqual(result.metadata["effective_short_edge"], 24)
        self.assertEqual(result.metadata["native_size"], [32, 24])
        self.assertEqual(result.metadata["native_pixel_fraction"], 1 / 16)
        self.assertFalse(np.array_equal(result.image, self.rgb))
        expected_native = Image.fromarray(self.rgb).resize((32, 24), Image.Resampling.LANCZOS)
        np.testing.assert_array_equal(result.native_image, np.array(expected_native))

    def test_low_resolution_never_upsamples_small_input_at_downsample_stage(self):
        result = apply_quality(self.rgb, PRESETS["lowres_720"])
        self.assertEqual(result.metadata["effective_short_edge"], 96)
        self.assertFalse(result.metadata["resized"])
        np.testing.assert_array_equal(result.image, self.rgb)
        np.testing.assert_array_equal(result.native_image, self.rgb)

    def test_low_resolution_handles_portrait_and_odd_dimensions(self):
        rgb = np.zeros((129, 95, 3), dtype=np.uint8)
        result = apply_quality(rgb, dict(PRESETS["lowres_720"], short_edge=30))
        self.assertEqual(result.native_image.shape[:2], (41, 30))
        self.assertEqual(result.image.shape, rgb.shape)
        self.assertEqual(result.metadata["effective_short_edge"], 30)

    def test_jpeg_result_is_exact_decode_of_returned_original_size_stream(self):
        for name in ("jpeg_medium", "jpeg_strong"):
            with self.subTest(preset=name):
                result = apply_quality(self.rgb, PRESETS[name])
                self.assertTrue(result.encoded_jpeg.startswith(b"\xff\xd8"))
                with Image.open(BytesIO(result.encoded_jpeg)) as decoded:
                    self.assertEqual(decoded.size, (128, 96))
                    self.assertEqual(JpegImagePlugin.get_sampling(decoded), 2)
                    np.testing.assert_array_equal(result.image, np.array(decoded.convert("RGB")))
                self.assertIsNone(result.native_image)
                self.assertFalse(result.metadata["resized"])
                self.assertEqual(result.metadata["encoded_bytes"], len(result.encoded_jpeg))

    def test_srgb_roundtrip_preserves_all_uint8_codes(self):
        codes = np.arange(256, dtype=np.uint8)
        encoded = np.rint(linear_to_srgb(srgb_to_linear(codes.astype(np.float32) / 255)) * 255)
        np.testing.assert_array_equal(encoded.astype(np.uint8), codes)

    def test_invalid_images_and_seeds_fail_explicitly(self):
        images = [np.zeros((3, 3)), np.zeros((3, 3, 4), np.uint8),
                  np.zeros((0, 3, 3), np.uint8), np.zeros((3, 3, 3), np.float32),
                  [[[0, 0, 0]]]]
        for rgb in images:
            with self.subTest(shape=getattr(rgb, "shape", None)):
                with self.assertRaises(ValueError):
                    apply_quality(rgb, PRESETS["sensor_noise"])
        for seed in (-1, 2 ** 128, 1.2, True, None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    apply_quality(self.rgb, PRESETS["sensor_noise"], seed)

    def test_invalid_presets_fail_explicitly(self):
        invalid = [None, [], {}, dict(PRESETS["defocus_mild"], label=""),
                   dict(PRESETS["defocus_mild"], sigma=-1),
                   dict(PRESETS["defocus_mild"], sigma=math.nan),
                   dict(PRESETS["defocus_mild"], sigma=10 ** 500),
                   dict(PRESETS["defocus_mild"], sigma=True),
                   dict(PRESETS["defocus_mild"], reference_short_edge=0),
                   dict(PRESETS["defocus_mild"], reference_short_edge=1.2),
                   dict(PRESETS["motion_blur"], length=0),
                   dict(PRESETS["motion_blur"], angle=181),
                   dict(PRESETS["sensor_noise"], peak=0),
                   dict(PRESETS["sensor_noise"], read_sigma=-0.1),
                   dict(PRESETS["sensor_noise"], peak=math.inf),
                   dict(PRESETS["sensor_noise"], grain_short_edge=0),
                   dict(PRESETS["sensor_noise"], grain_short_edge=65537),
                   dict(PRESETS["sensor_noise"], grain_short_edge=True),
                   dict(PRESETS["sensor_noise"], grain_short_edge=24.5),
                   dict(PRESETS["sensor_noise"], grain_short_edge=None),
                   dict(PRESETS["lowres_720"], short_edge=True),
                   dict(PRESETS["jpeg_medium"], quality=0),
                   dict(PRESETS["jpeg_medium"], quality=96),
                   dict(PRESETS["jpeg_medium"], subsampling=0),
                   dict(PRESETS["jpeg_medium"], group="../unsafe"),
                   dict(PRESETS["jpeg_medium"], kind=[]),
                   dict(PRESETS["jpeg_medium"], kind="unknown")]
        for preset in invalid:
            with self.subTest(preset=preset):
                with self.assertRaises(ValueError):
                    validate_preset(preset)


if __name__ == "__main__":
    unittest.main()
