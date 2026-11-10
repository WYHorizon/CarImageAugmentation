"""Properties of off-frame camera glare and source-coupled aperture ghosts."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from lighting.glare import apply_glare, validate_glare


def preset(effect='sun', **changes):
    return {'kind': 'camera_glare', 'effect': effect, 'label': effect,
            'source': [-0.08, 0.18], 'veil': 0.34, 'near_spill': 0.45,
            'ghost_strength': 0.22 if effect == 'lens' else 0, **changes}


class CameraGlareTests(unittest.TestCase):
    def test_zero_strength_is_exact_identity_without_shared_memory(self):
        image = np.random.default_rng(3).random((37, 51, 3), dtype=np.float32)
        saved = image.copy()
        for effect in ['sun', 'lens']:
            result = apply_glare(image, preset(effect, strength=0, ev=1.5))
            np.testing.assert_array_equal(result, image)
            self.assertFalse(np.shares_memory(image, result))
        np.testing.assert_array_equal(image, saved)

    def test_effects_are_deterministic_finite_and_do_not_mutate_parameters(self):
        image = np.random.default_rng(4).random((70, 91, 3), dtype=np.float32)
        saved = image.copy()
        for effect in ['sun', 'lens']:
            configuration = preset(effect)
            original_configuration = copy.deepcopy(configuration)
            first = apply_glare(image, configuration)
            second = apply_glare(image, configuration)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(image, saved)
            self.assertEqual(configuration, original_configuration)
            self.assertEqual(first.dtype, np.float32)
            self.assertEqual(first.shape, image.shape)
            self.assertTrue(np.isfinite(first).all())

    def test_source_scattering_is_strongest_at_edge_and_decays_into_frame(self):
        image = np.zeros((180, 240, 3), dtype=np.float32)
        result = apply_glare(image, preset())
        self.assertGreater(float(result[:, :30].mean()), float(result[:, -30:].mean()) * 5)
        y, x = np.unravel_index(np.argmax(result[..., 0]), result.shape[:2])
        self.assertEqual(x, 0)  # An off-frame source, not a painted interior orb.
        self.assertLess(y, image.shape[0] // 2)
        self.assertGreater(float(result.max()), 0.2)

    def test_scattering_mirrors_with_source_and_exposure_is_separate(self):
        image = np.zeros((121, 171, 3), dtype=np.float32)
        left = apply_glare(image, preset(source=[-0.08, 0.18]))
        right = apply_glare(image, preset(source=[1.08, 0.18]))
        np.testing.assert_allclose(left, right[:, ::-1], rtol=3e-5, atol=1e-7)
        neutral = np.full_like(image, 0.2)
        exposed = apply_glare(neutral, preset(strength=0.5, ev=1.0, veil=0, near_spill=0))
        np.testing.assert_allclose(exposed, neutral * 2 ** 0.5, rtol=1e-6)

    def test_strength_scales_additive_energy_without_clipping(self):
        image = np.full((63, 87, 3), 0.8, dtype=np.float32)
        for effect in ['sun', 'lens']:
            low = apply_glare(image, preset(effect, strength=0.5))
            high = apply_glare(image, preset(effect, strength=1))
            np.testing.assert_allclose(high - image, 2 * (low - image), rtol=1e-4, atol=1e-7)
            self.assertGreater(float(high.max()), 1)
            self.assertTrue(np.all(high >= low))

    def test_ghost_energy_follows_source_center_axis_and_remains_translucent(self):
        image = np.zeros((200, 300, 3), dtype=np.float32)
        result = apply_glare(image, preset('lens', source=[-0.1, 0.5], veil=0, near_spill=0))
        self.assertGreater(float(result[90:110].mean()), float(result[:20].mean()) * 10)
        self.assertGreater(float(result.max()), 0.03)
        self.assertLess(float(result.max()), 0.1)
        np.testing.assert_array_equal(apply_glare(image, preset('lens', veil=0,
                                                              near_spill=0, ghost_strength=0)), image)

    def test_normalized_spatial_fields_survive_resolution_change_and_field_cap(self):
        for effect in ['sun', 'lens']:
            small = apply_glare(np.zeros((180, 240, 3), dtype=np.float32), preset(effect))
            large = apply_glare(np.zeros((900, 1200, 3), dtype=np.float32), preset(effect))
            reduced = np.stack([np.asarray(Image.fromarray(large[..., c]).resize(
                (240, 180), Image.Resampling.BOX)) for c in range(3)], axis=-1)
            np.testing.assert_allclose(small, reduced, atol=0.0007, rtol=0.015)
        for shape in [(1, 1, 3), (1, 7, 3), (8, 1, 3)]:
            result = apply_glare(np.zeros(shape, dtype=np.float32), preset('lens'))
            self.assertEqual(result.shape, shape)
            self.assertTrue(np.isfinite(result).all())

    def test_invalid_sources_parameters_and_images_fail(self):
        invalid = [preset(source=[0.5, 0.5]), preset(source=[0, 0]),
                   preset(source=[1, 1]), preset(source=[-1, 0.5]),
                   preset(source=[np.nan, 0.2]), preset(source=[True, 0.2]),
                   preset(source=[-0.1]), preset(strength=-1), preset(strength=3),
                   preset(veil=float('inf')), preset(scatter_radius=0),
                   preset(scatter_aspect=False), preset(ghost_strength=0.1),
                   preset(color=[0, 0, 0]), preset(color=[1, 2]),
                   preset(effect='backlight'), preset(unknown=3),
                   {'kind': 'camera_glare', 'effect': 'sun', 'label': ''}]
        for configuration in invalid:
            with self.subTest(configuration=configuration):
                with self.assertRaises(ValueError):
                    validate_glare(configuration)
        for image in [np.zeros((0, 2, 3)), np.zeros((2, 3)),
                      np.full((2, 3, 3), np.nan), np.full((2, 3, 3), -0.1),
                      np.zeros((2, 3, 3), dtype=complex)]:
            with self.assertRaises(ValueError):
                apply_glare(image, preset())


if __name__ == '__main__':
    unittest.main()
