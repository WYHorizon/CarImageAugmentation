"""Properties of strong optical augmentation, including optical-field scaling."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from lighting.effects import apply_effect, validate_effect


def preset(effect='glare', **kwargs):
    return {'label': 'Test optics', 'kind': 'optical', 'effect': effect, **kwargs}


class OpticalEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = json.loads((METHOD / 'configs/strong_presets.json').read_text())

    def test_zero_strength_is_exact_identity_for_every_optical_preset(self):
        source = np.random.default_rng(4).random((19, 31, 3), dtype=np.float32)
        for name, config in self.presets.items():
            if config['kind'] != 'optical':
                continue
            with self.subTest(preset=name):
                result = apply_effect(source, {**config, 'strength': 0})
                np.testing.assert_array_equal(result, source)
                self.assertFalse(np.shares_memory(result, source))

    def test_strong_presets_are_deterministic_finite_and_preserve_inputs(self):
        source = np.random.default_rng(9).random((36, 48, 3), dtype=np.float32)
        original = source.copy()
        for name, config in self.presets.items():
            if config['kind'] != 'optical':
                continue
            with self.subTest(preset=name):
                unchanged_config = copy.deepcopy(config)
                first, second = apply_effect(source, config), apply_effect(source, config)
                self.assertEqual(first.shape, source.shape)
                self.assertEqual(first.dtype, np.float32)
                self.assertTrue(np.isfinite(first).all())
                self.assertTrue((first >= 0).all())
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(source, original)
                self.assertEqual(config, unchanged_config)
                self.assertFalse(np.shares_memory(first, source))

    def test_bloom_spreads_only_from_bright_source_and_keeps_unclipped_energy(self):
        config = preset('bloom', bloom_threshold=0.5, bloom_radius=0.04,
                        bloom_strength=2, color=[1, 1, 1])
        dark = np.full((81, 81, 3), 0.1, dtype=np.float32)
        np.testing.assert_array_equal(apply_effect(dark, config), dark)
        bright = dark.copy()
        bright[37:44, 37:44] = 3
        result = apply_effect(bright, config)
        self.assertGreater(float(result[40, 40, 0]), 3)
        self.assertGreater(float(result[40, 47, 0]), 0.1)
        self.assertGreater(float(result[40, 47, 0]), float(result[0, 0, 0]))
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(bright[37:44, 37:44], 3)

    def test_source_mirroring_and_locality(self):
        blank = np.zeros((70, 100, 3), dtype=np.float32)
        left = apply_effect(blank, preset(source=[0.2, 0.3], radius=0.2,
                                         veil=0.3, core=2, color=[1, 1, 1]))
        right = apply_effect(blank, preset(source=[0.8, 0.3], radius=0.2,
                                          veil=0.3, core=2, color=[1, 1, 1]))
        np.testing.assert_allclose(left[:, ::-1], right, atol=4e-6)
        self.assertGreater(float(left[20, 20, 0]), float(left[60, 90, 0]))

    def test_normalized_radius_is_consistent_across_resolution(self):
        config = preset(source=[0.25, 0.3], radius=0.35, veil=0.6, color=[1, 1, 1])
        coarse = apply_effect(np.zeros((60, 80, 3), dtype=np.float32), config)[..., 0]
        fine = apply_effect(np.zeros((240, 320, 3), dtype=np.float32), config)[..., 0]
        fine_reduced = np.asarray(Image.fromarray(fine).resize((80, 60), Image.Resampling.BOX))
        np.testing.assert_allclose(coarse, fine_reduced, atol=0.0002)
        # Also exercise the floating-point 960px working-field cap.
        large = apply_effect(np.zeros((750, 1000, 3), dtype=np.float32), config)[..., 0]
        large_reduced = np.asarray(Image.fromarray(large).resize((80, 60), Image.Resampling.BOX))
        np.testing.assert_allclose(coarse, large_reduced, atol=0.0007)

    def test_strength_scales_ev_and_zero_bloom_keeps_expected_exposure(self):
        source = np.full((17, 21, 3), 0.4, dtype=np.float32)
        config = preset('bloom', ev=-2, strength=0.5, bloom_strength=0)
        np.testing.assert_allclose(apply_effect(source, config), source * 0.5)
        doubled = apply_effect(source, {**config, 'strength': 1})
        np.testing.assert_allclose(doubled, source * 0.25)

    def test_ghosts_add_energy_along_source_optical_center_axis(self):
        blank = np.zeros((100, 140, 3), dtype=np.float32)
        config = preset('flare', source=[0.8, 0.15], ghosts=1, color=[1, 1, 1])
        result = apply_effect(blank, config)
        # The second ghost is close to the optical center; a remote lower corner
        # must receive far less optical energy than that region.
        self.assertGreater(float(result[53, 65].mean()), float(result[95, 135].mean()) + 0.02)
        self.assertGreater(float(result.max()), 0)

    def test_small_images_remain_supported(self):
        for shape in [(1, 1, 3), (1, 9, 3), (8, 1, 3)]:
            source = np.ones(shape, dtype=np.float32)
            result = apply_effect(source, self.presets['highlight_bloom'])
            self.assertEqual(result.shape, shape)
            self.assertTrue(np.isfinite(result).all())

    def test_invalid_parameters_and_images_fail(self):
        invalid = [
            {'effect': 'unknown'}, {'strength': -1}, {'strength': float('nan')},
            {'ev': 10}, {'radius': 0}, {'bloom_radius': -0.1},
            {'source': [0.5]}, {'source': [float('inf'), 0.5]},
            {'source': [2, 0.5]}, {'color': [0, 0, 0]}, {'color': [1, -1, 1]},
            {'veil': True}, {'ghosts': 3}, {'bloom_threshold': -1},
            {'bloom_strength': float('inf')}, {'mispelled_strength': 1},
        ]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    validate_effect({**preset(), **changes})
        for image in [np.zeros((0, 2, 3)), np.zeros((2, 3)), np.zeros((2, 3, 4)),
                      np.full((2, 3, 3), np.nan), -np.ones((2, 3, 3)),
                      np.ones((2, 3, 3), dtype=complex)]:
            with self.subTest(shape=image.shape):
                with self.assertRaises(ValueError):
                    apply_effect(image, preset())


if __name__ == '__main__':
    unittest.main()
