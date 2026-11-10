"""Tests for invariants relevant to fine-grained augmentation. No model downloads."""

import copy
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import stable_seed
from src.physics_weather import build_depth, fog, rain_transmittance, ray_distance, render, srgb_to_linear, to_image


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT / 'configs/scenes.json').read_text())['奥迪A6L.jpg']
        cls.presets = json.loads((ROOT / 'configs/presets.json').read_text())

    def test_zero_fog_roundtrips_every_srgb_byte(self):
        # Includes gamma toe, rounding boundaries, and bright/dark fine detail.
        original = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (3, 1, 3))
        linear = srgb_to_linear(original.astype(np.float32) / 255)
        output, t = fog(linear, np.ones((3, 256), np.float32), 0, np.array([0.8] * 3))
        np.testing.assert_array_equal(np.asarray(to_image(output)), original)
        np.testing.assert_array_equal(t, np.ones_like(t))

    def test_fog_scales_local_contrast_without_moving_pixels(self):
        source = np.array([[[0.1] * 3, [0.7] * 3]], dtype=np.float32)
        output, t = fog(source, np.array([[8., 8.]]), 0.02, np.array([0.8] * 3))
        np.testing.assert_allclose(output[0, 1] - output[0, 0], (source[0, 1] - source[0, 0]) * t[0, 0], rtol=1e-6)
        self.assertTrue(output[0, 1, 0] > output[0, 0, 0])

    def test_far_scene_receives_more_fog(self):
        source = np.zeros((1, 2, 3), np.float32)
        output, t = fog(source, np.array([[6., 80.]], np.float32), 0.02, np.ones(3))
        self.assertGreater(t[0, 0], t[0, 1])
        self.assertGreater(output[0, 1, 0], output[0, 0, 0])

    def test_scene_geometry_is_positive_and_subject_is_near(self):
        scenes = json.loads((ROOT / 'configs/scenes.json').read_text())
        for scene in scenes.values():
            with self.subTest(scene=scene['label']):
                depth, mask = build_depth((320, 240), scene)
                self.assertTrue(np.isfinite(depth).all())
                self.assertGreater(depth.min(), 0)
                self.assertLess(depth[mask == 1].max(), 8)
                self.assertGreater(np.count_nonzero(mask > 0.99), 5000)
                self.assertTrue(np.all(ray_distance(depth, scene) >= depth))

    def test_rain_reproducible_and_occluded_by_nearer_surface(self):
        source = np.full((240, 320, 3), 0.2, np.float32)
        preset = self.presets['rain_light']
        # Everything closer than the nearest sampled rain particle must hide rain.
        near_depth = np.full((240, 320), 0.5, np.float32)
        _, _, rain_t, stats = render(source, near_depth, self.scene, preset, 42)
        self.assertGreater(stats['particles_sampled'], 0)
        self.assertEqual(stats['particles_visible'], 0)
        np.testing.assert_array_equal(rain_t, np.ones_like(rain_t))
        far_depth = np.full((240, 320), 30, np.float32)
        a = render(source, far_depth, self.scene, preset, 42)
        b = render(source, far_depth, self.scene, preset, 42)
        c = render(source, far_depth, self.scene, preset, 43)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[2], c[2]))
        self.assertLess(a[2].min(), 1)
        self.assertTrue(np.all((a[0] >= 0) & (a[0] <= 1)))

    def test_fog_is_deterministic_across_seeds_and_inputs_not_mutated(self):
        source = np.random.default_rng(2).random((24, 32, 3), dtype=np.float32)
        depth, _ = build_depth((32, 24), self.scene)
        original, saved_depth, saved_scene = source.copy(), depth.copy(), copy.deepcopy(self.scene)
        a = render(source, depth, self.scene, self.presets['fog_light'], 1)[0]
        b = render(source, depth, self.scene, self.presets['fog_light'], 2)[0]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(depth, saved_depth)
        self.assertEqual(self.scene, saved_scene)

    def test_stronger_rain_has_larger_effective_rain_signal(self):
        # Isolate the particle signal from fog and compare exposure-integrated
        # radiance on an identical, unoccluded view, rather than visual labels.
        depth = np.full((180, 240), 30, np.float32)
        signals = []
        for name in ['rain_light', 'rain_medium', 'rain_heavy']:
            preset = self.presets[name]
            transmittance, _ = rain_transmittance(depth, self.scene, preset, np.random.default_rng(123))
            signals.append(float(np.mean(1 - transmittance)) * np.mean(preset['radiance_linear']))
        self.assertGreater(signals[1], signals[0] * 2)
        self.assertGreater(signals[2], signals[1] * 2)

    def test_hdr_rain_is_finite_and_display_conversion_is_bounded(self):
        source = np.ones((48, 64, 3), np.float32)
        depth = np.full((48, 64), 8, np.float32)
        preset = copy.deepcopy(self.presets['rain_heavy'])
        preset['beta_per_m'] = 0
        # A short range keeps this test focused on close HDR particles.
        preset['max_rain_depth_m'] = 5
        a = render(source, depth, self.scene, preset, 4)[0]
        b = render(source, depth, self.scene, preset, 4)[0]
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(a.max()), 1)
        np.testing.assert_array_equal(a, b)
        displayed = np.asarray(to_image(a))
        self.assertEqual(displayed.dtype, np.uint8)
        self.assertEqual(displayed.shape, source.shape)
        for bad_key, bad_value in [('radiance_linear', [-1, 1, 1]), ('max_rain_depth_m', 0.5)]:
            bad = copy.deepcopy(preset)
            bad[bad_key] = bad_value
            with self.assertRaises(ValueError):
                render(source, depth, self.scene, bad, 4)

    def test_seed_is_stable_and_separates_sources_and_variants(self):
        self.assertEqual(stable_seed(42, 'a.jpg', 'rain', 0), stable_seed(42, 'a.jpg', 'rain', 0))
        self.assertNotEqual(stable_seed(42, 'a.jpg', 'rain', 0), stable_seed(42, 'b.jpg', 'rain', 0))
        self.assertNotEqual(stable_seed(42, 'a.jpg', 'rain', 0), stable_seed(42, 'a.jpg', 'rain', 1))

    def test_reject_invalid_fog_and_camera_parameters(self):
        with self.assertRaises(ValueError):
            fog(np.ones((2, 2, 3)), np.ones((2, 2)), -0.1, np.ones(3))
        with self.assertRaises(ValueError):
            fog(np.ones((2, 2, 3)), np.full((2, 2), np.nan), 0.1, np.ones(3))
        bad_scene = copy.deepcopy(self.scene)
        bad_scene['camera']['height_m'] = -1
        with self.assertRaises(ValueError):
            build_depth((32, 24), bad_scene)


class CommandTests(unittest.TestCase):
    def test_cli_preserves_source_and_records_output_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs, outputs = directory / 'inputs', directory / 'outputs'
            inputs.mkdir()
            source = inputs / '奥迪A6L.jpg'
            Image.fromarray(np.random.default_rng(4).integers(0, 256, (96, 128, 3), dtype=np.uint8)).save(source)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            command = [sys.executable, str(ROOT / 'run.py'), '--input-dir', str(inputs),
                       '--output-dir', str(outputs), '--weather', 'fog_light', '--count', '3']
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            manifest = json.loads((outputs / 'manifest.json').read_text())
            self.assertEqual(len(manifest['records']), 1)  # no duplicate deterministic fog
            record = manifest['records'][0]
            self.assertEqual(record['source_sha256'], original_hash)
            with Image.open(outputs / record['output']) as image:
                self.assertEqual(image.size, (128, 96))
                self.assertEqual(image.format, 'PNG')
            prior = (outputs / 'manifest.json').read_bytes()
            second = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((outputs / 'manifest.json').read_bytes(), prior)

    def test_unknown_scene_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs, outputs = directory / 'inputs', directory / 'outputs'
            inputs.mkdir()
            Image.new('RGB', (32, 24)).save(inputs / 'new_car.png')
            result = subprocess.run(
                [sys.executable, str(ROOT / 'run.py'), '--input-dir', str(inputs), '--output-dir', str(outputs)],
                cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('new_car.png', result.stderr)
            self.assertFalse(outputs.exists())


if __name__ == '__main__':
    unittest.main()
