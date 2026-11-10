"""Physical limits and vehicle-isolation checks for wet/overcast processing."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene_weather import apply_scene_weather, relight_overcast, scene_masks, water_fresnel, wet_road


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = json.loads((ROOT / 'configs/scenes.json').read_text())
        cls.config = json.loads((ROOT / 'configs/presets.json').read_text())['overcast_wet']['surface_weather']

    def test_water_fresnel_has_correct_normal_and_grazing_limits(self):
        reflectance = water_fresnel(np.array([1., 0.5, 0.1, 0.], np.float32))
        self.assertAlmostEqual(float(reflectance[0]), ((1.333 - 1) / (1.333 + 1)) ** 2, places=6)
        self.assertAlmostEqual(float(reflectance[-1]), 1, places=6)
        self.assertTrue(np.all(np.diff(reflectance) > 0))

    def test_wet_layer_never_changes_guarded_vehicle_and_cannot_blur_its_pattern(self):
        # Alternating bright/dark pixels expose accidental texture smoothing,
        # resampling, or reflection leaking into the vehicle.
        yy, xx = np.indices((180, 240))
        pattern = (0.08 + 0.70 * ((xx + yy) % 2)).astype(np.float32)
        source = np.repeat(pattern[..., None], 3, axis=2)
        for scene in self.scenes.values():
            with self.subTest(scene=scene['label']):
                masks = scene_masks((240, 180), scene)
                protected = masks['protected'] > 0.99
                np.testing.assert_array_equal(masks['road'][protected], np.zeros(np.count_nonzero(protected)))
                original = source.copy()
                output, stats = apply_scene_weather(source, scene, self.config, 42)
                ratio = output[protected] / source[protected]
                self.assertGreaterEqual(float(ratio.min()), 0.74 - 1e-6)
                self.assertLessEqual(float(ratio.max()), 1.08 + 1e-6)
                self.assertEqual(stats['scene_lighting']['vehicle_stage_equation_max_error'], 0)
                self.assertEqual(stats['wet_surface']['wet_layer_max_change_on_vehicle_guard'], 0)
                np.testing.assert_array_equal(source, original)

    def test_zero_wetness_is_identity_after_relighting(self):
        scene = self.scenes['奥迪A6L.jpg']
        source = np.random.default_rng(5).uniform(0.1, 0.8, (180, 240, 3)).astype(np.float32)
        masks = scene_masks((240, 180), scene)
        relit, _, sky, _ = relight_overcast(source, scene, self.config['overcast'], masks)
        config = copy.deepcopy(self.config['wet_road'])
        config['wetness'] = 0
        result, wetness, _, _, _ = wet_road(relit, scene, config, masks, sky, 3)
        np.testing.assert_array_equal(result, relit)
        self.assertFalse(np.any(wetness))

    def test_wet_pattern_seed_changes_only_road_and_is_reproducible(self):
        scene = self.scenes['小米YU7.jpg']
        source = np.full((180, 240, 3), 0.4, np.float32)
        a, _ = apply_scene_weather(source, scene, self.config, 4)
        b, _ = apply_scene_weather(source, scene, self.config, 4)
        c, _ = apply_scene_weather(source, scene, self.config, 5)
        np.testing.assert_array_equal(a, b)
        masks = scene_masks((240, 180), scene)
        np.testing.assert_array_equal(a[masks['road'] == 0], c[masks['road'] == 0])
        self.assertGreater(float(np.max(np.abs(a - c))), 0)

    def test_nonfinite_or_invalid_surface_parameters_are_rejected(self):
        scene = self.scenes['本田CR-V.jpg']
        source = np.full((48, 64, 3), 0.3, np.float32)
        for section, key, value in [('overcast', 'exposure', float('nan')),
                                    ('wet_road', 'wetness', 1.5),
                                    ('wet_road', 'reflection_blur_ratio', -0.1)]:
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(ValueError):
                apply_scene_weather(source, scene, config, 42)


if __name__ == '__main__':
    unittest.main()
