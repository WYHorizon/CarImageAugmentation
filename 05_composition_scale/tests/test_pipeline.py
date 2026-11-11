"""End-to-end source protection and geometry-aware comparison contracts."""

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
from run import annotation_record, region_polygon


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root / 'inputs'
        inputs.mkdir()
        pixels = np.random.default_rng(15).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        source = inputs / 'custom_photo.png'
        Image.fromarray(pixels).save(source)
        scene = {
            'label': 'Test vehicle',
            'subject_polygon': [[0.2, 0.55], [0.35, 0.4], [0.65, 0.4],
                                [0.8, 0.55], [0.8, 0.75], [0.2, 0.75]],
            'regions': {'headlamp': [0.3, 0.42, 0.47, 0.54], 'grille': [0.22, 0.54, 0.4, 0.7],
                        'front_wheel': [0.6, 0.6, 0.79, 0.79], 'edge_patch': [0.01, 0.5, 0.1, 0.65]},
        }
        scenes = root / 'scenes.json'
        scenes.write_text(json.dumps({source.name: scene}))
        return inputs, source, scenes

    def invoke(self, inputs, scenes, output, *extra):
        return subprocess.run([sys.executable, str(METHOD / 'run.py'), '--input-dir', str(inputs),
                               '--scenes', str(scenes), '--output-dir', str(output), *extra],
                              capture_output=True, text=True)

    def test_full_run_maps_crops_and_preserves_sources_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, source, scenes = self.fixture(root)
            original = source.read_bytes()
            output = root / 'run'
            checked = self.invoke(inputs, scenes, output, '--check-config')
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)['expected_images'], 8)
            self.assertFalse(output.exists())
            run = self.invoke(inputs, scenes, output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete_pending_visual_review')
            self.assertEqual(len(manifest['records']), 8)
            self.assertTrue(manifest['verification']['source_unchanged'])
            self.assertEqual(manifest['verification']['valid_masks_checked'], 8)
            self.assertEqual(manifest['report']['preview_png_count'], 10)
            self.assertEqual(source.read_bytes(), original)
            expected_crops = 4
            invisible = 0
            for record in manifest['records']:
                self.assertEqual(record['retention_check'], 'passed')
                matrix = np.asarray(record['transform']['forward_matrix'])
                inverse = np.asarray(record['transform']['inverse_matrix'])
                np.testing.assert_allclose(matrix @ inverse, np.eye(3), atol=1e-12)
                with Image.open(output / record['valid_mask']) as mask:
                    self.assertEqual(mask.mode, 'L')
                    self.assertTrue(np.all(np.asarray(mask) == 255))
                with Image.open(output / record['output']) as full:
                    self.assertEqual(full.size, (128, 96))
                    for name, region in record['regions'].items():
                        crop_path = output / 'crops' / record['vehicle_id'] / name / (record['preset_id'] + '.png')
                        if not region['visible']:
                            self.assertFalse(crop_path.exists())
                            invisible += 1
                            continue
                        expected_crops += 1
                        with Image.open(crop_path) as crop:
                            np.testing.assert_array_equal(np.asarray(crop), np.asarray(full.crop(region['crop_box_xyxy'])))
            self.assertGreater(invisible, 0)
            self.assertEqual(manifest['report']['native_crop_count'], expected_crops)
            self.assertEqual(manifest['verification']['png_files_checked'], 16 + 10 + expected_crops)
            left = next(r for r in manifest['records'] if r['preset_id'] == 'position_left')
            original_box = region_polygon([0.3, 0.42, 0.47, 0.54], (128, 96))
            self.assertLess(left['regions']['headlamp']['transformed_polygon_xy'][0][0], original_box[0, 0])
            saved = (output / 'manifest.json').read_bytes()
            self.assertNotEqual(self.invoke(inputs, scenes, output).returncode, 0)
            self.assertEqual((output / 'manifest.json').read_bytes(), saved)

    def test_complete_subject_policy_is_checked_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, source, scenes = self.fixture(root)
            config = json.loads(scenes.read_text())
            config[source.name]['subject_polygon'] = [[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]]
            scenes.write_text(json.dumps(config))
            output = root / 'invalid'
            result = self.invoke(inputs, scenes, output, '--effects', 'zoom_close')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('complete-subject preset', result.stderr)
            self.assertFalse(output.exists())
            allowed = self.invoke(inputs, scenes, root / 'allowed', '--effects', 'zoom_tight', '--check-config')
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            self.assertFalse((root / 'allowed').exists())

    def test_custom_images_without_subject_and_repeated_labels_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, source, scenes = self.fixture(root)
            second = inputs / 'portrait.png'
            Image.new('RGB', (71, 99), '#75a3c2').save(second)
            scenes.write_text(json.dumps({name: {'label': 'Audi A6L', 'regions': {'full_image': [0, 0, 1, 1]}}
                                         for name in [source.name, second.name]}))
            output = root / 'custom'
            result = self.invoke(inputs, scenes, output, '--effects', 'zoom_close', 'roll_right')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = json.loads((output / 'manifest.json').read_text())['records']
            self.assertEqual(len(records), 4)
            self.assertEqual(len({r['vehicle_id'] for r in records}), 2)
            self.assertEqual({tuple(r['size']) for r in records}, {(128, 96), (71, 99)})
            self.assertTrue(all(r['subject'] is None for r in records))
            self.assertTrue(all(r['retention_check'] == 'not_available_without_subject_polygon' for r in records))

    def test_invalid_effects_regions_and_protected_paths_fail_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, source, scenes = self.fixture(root)
            for args in [('--effects', 'unknown'), ('--effects', 'zoom_close', 'zoom_close')]:
                self.assertNotEqual(self.invoke(inputs, scenes, root / 'bad', *args).returncode, 0)
                self.assertFalse((root / 'bad').exists())
            self.assertNotEqual(self.invoke(inputs, scenes, inputs / 'bad').returncode, 0)
            self.assertFalse((inputs / 'bad').exists())
            config = json.loads(scenes.read_text())
            config[source.name]['regions']['headlamp'] = [0.5, 0.2, 0.1, 0.3]
            scenes.write_text(json.dumps(config))
            self.assertNotEqual(self.invoke(inputs, scenes, root / 'bad').returncode, 0)
            self.assertFalse((root / 'bad').exists())

    def test_geometry_metrics_use_unclipped_center_and_true_retention(self):
        scene = {'subject_polygon': [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                 'regions': {'part': [0.7, 0.3, 0.8, 0.5]}}
        transform = {'forward_matrix': [[1, 0, 40], [0, 1, 0], [0, 0, 1]],
                     'scale': 1, 'angle_degrees': 0, 'retention_policy': 'partial_allowed'}
        record = annotation_record(scene, (100, 100), transform)
        self.assertAlmostEqual(record['subject']['retained_fraction'], 2 / 3)
        self.assertEqual(record['geometry_metrics']['transformed_subject_center_normalized'], [0.9, 0.5])
        self.assertEqual(record['geometry_metrics']['subject_center_shift_normalized'], [0.4, 0])
        self.assertAlmostEqual(record['geometry_metrics']['visible_subject_bbox_width_fraction'], 0.4)
        self.assertFalse(record['regions']['part']['visible'])
        self.assertIsNone(record['regions']['part']['crop_box_xyxy'])


if __name__ == '__main__':
    unittest.main()
