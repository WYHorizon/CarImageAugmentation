"""Numerical properties that matter for reproducible fine-grained augmentation."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lighting.transforms import LUMA_WEIGHTS, apply_lighting, linear_to_srgb, srgb_to_linear, validate_preset


def encode(linear):
    return np.rint(linear_to_srgb(linear) * 255).astype(np.uint8)


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = json.loads((ROOT / 'configs/presets.json').read_text(encoding='utf-8'))

    def test_all_srgb_bytes_roundtrip(self):
        values = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (2, 1, 3))
        decoded = srgb_to_linear(values.astype(np.float32) / 255)
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_array_equal(encode(decoded), values)

    def test_neutral_transforms_preserve_pixels_and_source(self):
        pixels = np.random.default_rng(2).integers(0, 256, (23, 19, 3), dtype=np.uint8)
        linear = srgb_to_linear(pixels.astype(np.float32) / 255)
        saved = linear.copy()
        identities = [
            {'label': 'Identity', 'kind': 'exposure', 'ev': 0},
            {'label': 'Identity', 'kind': 'contrast', 'factor': 1},
            {'label': 'Identity', 'kind': 'white_balance', 'gains': [1, 1, 1]},
        ]
        for preset in identities:
            with self.subTest(kind=preset['kind']):
                result = apply_lighting(linear, preset)
                np.testing.assert_array_equal(encode(result), pixels)
                np.testing.assert_array_equal(linear, saved)
                self.assertFalse(np.shares_memory(result, linear))

    def test_exposure_factor_and_unclipped_highlights(self):
        source = np.array([[[0.0, 0.18, 1.0]]], dtype=np.float32)
        result = apply_lighting(source, self.presets['exposure_up'])
        np.testing.assert_allclose(result, source * 2 ** 0.3, rtol=1e-6)
        self.assertGreater(float(result[0, 0, 2]), 1)
        self.assertEqual(int(encode(result)[0, 0, 2]), 255)
        recovered = apply_lighting(result, self.presets['exposure_down'])
        np.testing.assert_allclose(recovered, source, rtol=1e-6)

    def test_contrast_is_monotonic_has_fixed_pivot_and_keeps_black(self):
        values = np.sort(np.r_[np.linspace(0, 1, 256), 0.18]).astype(np.float32)
        source = np.repeat(values[None, :, None], 3, axis=-1)
        for name in ['contrast_low', 'contrast_high']:
            with self.subTest(preset=name):
                result = apply_lighting(source, self.presets[name])
                self.assertTrue(np.all(np.diff(result[0, :, 0]) > 0))
                np.testing.assert_array_equal(result[0, 0], np.zeros(3))
                np.testing.assert_allclose(result[0, values == np.float32(0.18)], 0.18, atol=1e-7)
                np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-7)
        self.assertLess(apply_lighting(source, self.presets['contrast_low'])[0, -1, 0], 1)
        self.assertGreater(apply_lighting(source, self.presets['contrast_high'])[0, -1, 0], 1)

    def test_contrast_preserves_channel_ratios_before_clipping(self):
        source = np.array([[[0.02, 0.04, 0.08], [0.4, 0.8, 1.2]]], dtype=np.float32)
        for name in ['contrast_low', 'contrast_high']:
            result = apply_lighting(source, self.presets[name])
            ratio = result / source
            np.testing.assert_allclose(ratio[..., 0], ratio[..., 1], rtol=1e-6)
            np.testing.assert_allclose(ratio[..., 1], ratio[..., 2], rtol=1e-6)

    def test_white_balance_preserves_neutral_luminance_and_has_correct_direction(self):
        source = np.full((2, 3, 3), 0.4, dtype=np.float32)
        for name in ['warm', 'cool']:
            result = apply_lighting(source, self.presets[name])
            np.testing.assert_allclose(result @ LUMA_WEIGHTS, source @ LUMA_WEIGHTS, rtol=1e-6)
            if name == 'warm':
                self.assertTrue(np.all(result[..., 0] > result[..., 1]))
                self.assertTrue(np.all(result[..., 1] > result[..., 2]))
            else:
                self.assertTrue(np.all(result[..., 0] < result[..., 1]))
                self.assertTrue(np.all(result[..., 1] < result[..., 2]))

    def test_all_presets_are_deterministic_and_do_not_mutate_inputs(self):
        self.assertEqual(list(self.presets), [
            'exposure_down', 'exposure_up', 'contrast_low', 'contrast_high', 'warm', 'cool'])
        source = np.random.default_rng(12).random((18, 24, 3), dtype=np.float32)
        saved = source.copy()
        for name, preset in self.presets.items():
            with self.subTest(preset=name):
                snapshot = copy.deepcopy(preset)
                validate_preset(preset)
                first = apply_lighting(source, preset)
                second = apply_lighting(source, preset)
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(source, saved)
                self.assertEqual(preset, snapshot)
                self.assertEqual(first.shape, source.shape)
                self.assertEqual(first.dtype, np.float32)

    def test_encoding_clips_only_at_output_and_does_not_mutate(self):
        source = np.array([[[-0.2, 0.18, 2.0]]], dtype=np.float32)
        saved = source.copy()
        result = linear_to_srgb(source)
        np.testing.assert_array_equal(source, saved)
        self.assertEqual(float(result[0, 0, 0]), 0)
        self.assertAlmostEqual(float(result[0, 0, 2]), 1, places=6)

    def test_invalid_parameters_fail(self):
        cases = [
            {}, {'label': 'x', 'kind': 'unknown'},
            {'label': '', 'kind': 'exposure', 'ev': 0},
            {'label': 'x', 'kind': 'exposure', 'ev': float('nan')},
            {'label': 'x', 'kind': 'exposure', 'ev': float('inf')},
            {'label': 'x', 'kind': 'exposure', 'ev': True},
            {'label': 'x', 'kind': 'exposure', 'ev': 1000},
            {'label': 'x', 'kind': 'contrast', 'factor': 0},
            {'label': 'x', 'kind': 'contrast', 'factor': float('inf')},
            {'label': 'x', 'kind': 'contrast', 'factor': 1, 'pivot': 0},
            {'label': 'x', 'kind': 'contrast', 'factor': 1, 'pivot': float('nan')},
            {'label': 'x', 'kind': 'white_balance', 'gains': [1, 1]},
            {'label': 'x', 'kind': 'white_balance', 'gains': [1, 0, 1]},
            {'label': 'x', 'kind': 'white_balance', 'gains': [1, -1, 1]},
            {'label': 'x', 'kind': 'white_balance', 'gains': [1, float('nan'), 1]},
        ]
        for preset in cases:
            with self.subTest(preset=preset):
                with self.assertRaises(ValueError):
                    validate_preset(preset)

    def test_invalid_images_fail(self):
        source = np.ones((2, 3, 3), dtype=np.float32)
        for invalid in [source * np.nan, source * np.inf, -source,
                        np.zeros((0, 3, 3)), np.zeros((2, 3)),
                        np.zeros((2, 3, 4)), source.astype(complex)]:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    apply_lighting(invalid, self.presets['exposure_up'])
        with self.assertRaises(ValueError):
            srgb_to_linear(source * 255)


if __name__ == '__main__':
    unittest.main()
