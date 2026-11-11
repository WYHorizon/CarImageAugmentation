"""Integration checks for source protection, reproducibility and artifact provenance."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from run import derive_seed, diagnostic_size, metrics


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root / 'inputs'
        inputs.mkdir()
        pixels = np.random.default_rng(17).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        source = inputs / 'sample.png'
        Image.fromarray(pixels).save(source)
        scene_path = root / 'scenes.json'
        scene_path.write_text(json.dumps({'sample.png': {
            'label': 'Test vehicle',
            'subject_polygon': [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            'regions': {'headlamp': [0.1, 0.2, 0.3, 0.4], 'grille': [0.3, 0.4, 0.6, 0.7],
                        'front_wheel': [0.1, 0.6, 0.25, 0.9], 'rear_wheel': [0.7, 0.6, 0.85, 0.9]},
        }}), encoding='utf-8')
        return inputs, source, scene_path

    def invoke(self, inputs, scenes, output, *extra):
        return subprocess.run([sys.executable, str(METHOD / 'run.py'), '--input-dir', str(inputs),
                               '--scenes', str(scenes), '--output-dir', str(output), *extra],
                              capture_output=True, text=True)

    def test_full_run_and_noise_subset_have_identical_pixels_and_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, source, scenes = self.fixture(root)
            original = source.read_bytes()
            output = root / 'run'
            checked = self.invoke(inputs, scenes, output, '--check-config')
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)['expected_images'], 8)
            self.assertFalse(output.exists())
            completed = self.invoke(inputs, scenes, output)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete_pending_visual_review')
            self.assertEqual(manifest['verification'], {'source_unchanged': True, 'png_files_checked': 54,
                             'generated_images': 8, 'encoded_jpeg_checked': 2, 'native_low_resolution_checked': 2})
            self.assertEqual(manifest['report']['native_crop_count'], 36)
            self.assertEqual(manifest['report']['preview_png_count'], 8)
            self.assertEqual(source.read_bytes(), original)
            crops = json.loads((output / 'crop_regions.json').read_text())
            boxes = {item['region']: item['native_box_xyxy'] for item in crops['regions']}
            for item in crops['regions']:
                self.assertEqual(item['short_edge_224_size'], list(diagnostic_size((64, 48))))
            for record in manifest['records']:
                with Image.open(output / record['output']) as image:
                    self.assertEqual(image.size, (64, 48))
                    self.assertEqual(record['diagnostic_224']['size'], list(diagnostic_size(image.size)))
                    for region, box in boxes.items():
                        path = output / 'crops' / record['vehicle_id'] / region / (record['preset_id'] + '.png')
                        with Image.open(path) as crop:
                            np.testing.assert_array_equal(np.asarray(crop), np.asarray(image.crop(box)))
                    if 'encoded_jpeg' in record:
                        with Image.open(output / record['encoded_jpeg']['path']) as encoded:
                            np.testing.assert_array_equal(np.asarray(image), np.asarray(encoded.convert('RGB')))
            subset = root / 'subset'
            result = self.invoke(inputs, scenes, subset, '--effects', 'sensor_noise')
            self.assertEqual(result.returncode, 0, result.stderr)
            subset_record = json.loads((subset / 'manifest.json').read_text())['records'][0]
            noise = next(record for record in manifest['records'] if record['preset_id'] == 'sensor_noise')
            self.assertEqual(subset_record['seed'], noise['seed'])
            self.assertEqual((subset / subset_record['output']).read_bytes(), (output / noise['output']).read_bytes())
            saved_manifest = (output / 'manifest.json').read_bytes()
            self.assertNotEqual(self.invoke(inputs, scenes, output).returncode, 0)
            self.assertEqual((output / 'manifest.json').read_bytes(), saved_manifest)

    def test_invalid_config_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, _, scenes = self.fixture(root)
            for extra in [('--seed', '-1'), ('--effects', 'nonexistent'), ('--effects', 'sensor_noise', 'sensor_noise')]:
                output = root / 'invalid'
                self.assertNotEqual(self.invoke(inputs, scenes, output, *extra).returncode, 0)
                self.assertFalse(output.exists())
            self.assertNotEqual(self.invoke(inputs, scenes, inputs / 'generated').returncode, 0)
            self.assertFalse((inputs / 'generated').exists())
            data = json.loads(scenes.read_text())
            data['sample.png']['regions']['headlamp'] = [0.5, 0.2, 0.1, 0.3]
            scenes.write_text(json.dumps(data))
            self.assertNotEqual(self.invoke(inputs, scenes, root / 'bad-region').returncode, 0)
            self.assertFalse((root / 'bad-region').exists())

    def test_strong_profile_retains_defocus_and_records_changed_parameters(self):
        basic = json.loads((METHOD / 'configs/presets.json').read_text())
        strong = json.loads((METHOD / 'configs/strong_presets.json').read_text())
        for name in ('defocus_mild', 'defocus_strong'):
            self.assertEqual(strong[name], basic[name])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, _, scenes = self.fixture(root)
            output = root / 'strong'
            # A small custom grain grid exercises the resampled residual branch.
            strong['coarse_noise']['grain_short_edge'] = 24
            config = root / 'strong.json'
            config.write_text(json.dumps(strong))
            result = self.invoke(inputs, scenes, output, '--profile', 'strong', '--presets', str(config))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['profile'], 'strong')
            self.assertTrue(manifest['custom_presets'])
            self.assertEqual(manifest['verification']['png_files_checked'], 54)
            self.assertTrue(manifest['verification']['source_unchanged'])
            self.assertEqual([record['preset_id'] for record in manifest['records']], list(strong))
            for record in manifest['records']:
                self.assertEqual(record['size'], [64, 48])
                self.assertEqual(record['preset'], strong[record['preset_id']])
                if 'encoded_jpeg' in record:
                    self.assertFalse(record['effective_parameters']['resized'])
            checked = self.invoke(inputs, scenes, root / 'check', '--profile', 'strong', '--check-config')
            self.assertEqual(checked.returncode, 0, checked.stderr)
            summary = json.loads(checked.stdout)
            self.assertEqual(summary['profile'], 'strong')
            self.assertEqual(summary['presets'], list(strong))
            self.assertFalse((root / 'check').exists())

    def test_metrics_handle_identical_and_flat_images_without_invalid_json(self):
        original = np.full((4, 4, 3), 100, dtype=np.uint8)
        same = metrics(original, original)
        self.assertIsNone(same['psnr_db'])
        self.assertTrue(same['identical_pixels'])
        self.assertEqual(same['mae_code_values'], 0)
        self.assertIsNone(same['display_luma_gradient_ratio'])
        changed = metrics(original, original + 10)
        self.assertEqual(changed['mae_code_values'], 10)
        self.assertEqual(changed['mse_code_values'], 100)
        self.assertAlmostEqual(changed['psnr_db'], 28.1308, places=4)
        json.dumps([same, changed], allow_nan=False)

    def test_stable_seed_records_source_and_preset_identity(self):
        digest = hashlib.sha256(b'source').hexdigest()
        actual = derive_seed(42, digest, 'sensor_noise')
        self.assertEqual(actual, derive_seed(42, digest, 'sensor_noise'))
        self.assertNotEqual(actual, derive_seed(43, digest, 'sensor_noise'))
        self.assertNotEqual(actual, derive_seed(42, digest, 'other'))
        self.assertNotEqual(actual, derive_seed(42, hashlib.sha256(b'other').hexdigest(), 'sensor_noise'))


if __name__ == '__main__':
    unittest.main()
