"""Check input protection, comparison outputs, and diagnostic semantics."""

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
from run import metrics, resolve_preset, validate_preset_template


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        inputs = root / 'inputs'
        inputs.mkdir()
        y, x = np.mgrid[:48, :64]
        pixels = np.stack([x * 4, y * 5, (x + y) * 2], axis=-1).astype(np.uint8)
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

    def test_full_run_preserves_source_and_builds_native_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, source, scenes = self.fixture(root)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output = root / 'run'
            checked = self.invoke(inputs, scenes, output, '--check-config')
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertFalse(output.exists())
            result = self.invoke(inputs, scenes, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete_pending_visual_review')
            self.assertEqual(len(manifest['records']), 6)
            self.assertEqual(manifest['verification']['png_files_checked'], 34)
            self.assertTrue(manifest['verification']['source_unchanged'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            crops = json.loads((output / 'crop_regions.json').read_text(encoding='utf-8'))
            boxes = {item['region']: item['native_box_xyxy'] for item in crops['regions']}
            for record in manifest['records']:
                with Image.open(output / record['output']) as image:
                    self.assertEqual(image.size, (64, 48))
                self.assertEqual(set(record['regions']), {'headlamp', 'grille', 'front_wheel', 'rear_wheel'})
                for region, metric in record['regions'].items():
                    self.assertEqual(metric['box_xyxy'], boxes[region])
                with Image.open(output / record['output']) as full_image:
                    for region, box in boxes.items():
                        crop_path = output / 'crops' / record['vehicle_id'] / region / (record['preset_id'] + '.png')
                        with Image.open(crop_path) as crop:
                            np.testing.assert_array_equal(np.asarray(crop), np.asarray(full_image.crop(box)))
            self.assertEqual(len(list((output / 'crops').rglob('*.png'))), 28)
            self.assertTrue((output / 'report/index.html').is_file())
            self.assertTrue((output / 'previews/exposure_overview.jpg').is_file())
            previous_manifest = (output / 'manifest.json').read_bytes()
            repeated = self.invoke(inputs, scenes, output)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / 'manifest.json').read_bytes(), previous_manifest)

    def test_invalid_region_and_protected_output_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, source, scenes = self.fixture(root)
            self.assertNotEqual(self.invoke(inputs, scenes, inputs / 'generated').returncode, 0)
            self.assertFalse((inputs / 'generated').exists())
            data = json.loads(scenes.read_text())
            data['sample.png']['regions']['headlamp'] = [0.4, 0.2, 0.3, 0.4]
            scenes.write_text(json.dumps(data))
            self.assertNotEqual(self.invoke(inputs, scenes, root / 'invalid').returncode, 0)
            self.assertFalse((root / 'invalid').exists())

    def test_strong_profile_runs_all_optical_presets_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, source, scenes = self.fixture(root)
            output = root / 'strong'
            result = self.invoke(inputs, scenes, output, '--profile', 'strong')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['profile'], 'strong')
            self.assertEqual(len(manifest['records']), 6)
            self.assertTrue(manifest['verification']['source_unchanged'])
            self.assertIn('lighting/effects.py', manifest['implementation_sha256'])
            for group in ['overexposure_bloom', 'glare_flare', 'backlight']:
                self.assertTrue((output / 'previews' / (group + '_overview.jpg')).is_file())
            for record in manifest['records']:
                self.assertEqual(record['size'], [64, 48])
                self.assertIn('changed_pixel_fraction_gt20', record['global'])
                self.assertIn('new_near_white_fraction', record['regions']['headlamp'])
            explicit = self.invoke(inputs, scenes, root / 'override', '--profile', 'strong',
                                   '--presets', str(METHOD / 'configs/presets.json'), '--check-config')
            self.assertEqual(explicit.returncode, 0, explicit.stderr)
            self.assertIn('exposure_down', json.loads(explicit.stdout)['presets'])

    def test_diagnostics_separate_existing_and_new_clipping_and_darkness(self):
        before = np.array([[[200, 200, 200], [255, 255, 255], [0, 0, 0], [20, 20, 20]]], dtype=np.uint8)
        after = np.array([[[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
        raw = np.array([[[1.1]*3, [1.1]*3, [0]*3, [0]*3]], dtype=np.float32)
        result = metrics(before, after, raw)
        self.assertEqual(result['preclip_above_one_pixel_fraction'], 0.5)
        self.assertEqual(result['new_clipped_pixel_fraction'], 0.25)
        self.assertEqual(result['new_saturated_pixel_fraction'], 0.25)
        self.assertEqual(result['source_near_black_fraction'], 0.25)
        self.assertEqual(result['new_near_black_fraction'], 0.25)
        self.assertEqual(result['new_near_white_fraction'], 0.25)
        self.assertEqual(result['changed_pixel_fraction_gt20'], 0.25)

    def test_refined_profile_preserves_retained_presets_and_records_effective_sources(self):
        refined = json.loads((METHOD / 'configs/refined_presets.json').read_text())
        historical = json.loads((METHOD / 'configs/strong_presets.json').read_text())
        for name in ['strong_overexposure', 'highlight_bloom']:
            self.assertEqual(refined[name], historical[name])
        self.assertEqual(list(refined), ['strong_overexposure', 'highlight_bloom', 'sun_glare', 'lens_flare'])
        for name in ['sun_glare', 'lens_flare']:
            refined[name]['source_overrides']['sample.png'] = {'source': [-0.2, 0.04], 'strength': 0.8}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, source, scenes = self.fixture(root)
            presets = root / 'presets.json'
            presets.write_text(json.dumps(refined))
            output = root / 'refined'
            result = self.invoke(inputs, scenes, output, '--profile', 'refined', '--presets', str(presets))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(manifest['records']), 4)
            self.assertEqual(manifest['verification']['png_files_checked'], 24)
            self.assertTrue(manifest['verification']['source_unchanged'])
            self.assertIn('lighting/glare.py', manifest['implementation_sha256'])
            self.assertFalse((output / 'previews/backlight_overview.jpg').exists())
            for record in manifest['records']:
                if record['preset_id'] in ['sun_glare', 'lens_flare']:
                    self.assertTrue(record['source_override_applied'])
                    self.assertEqual(record['preset']['source'], [-0.2, 0.04])
                    self.assertEqual(record['preset']['strength'], 0.8)
                    self.assertNotIn('source_overrides', record['preset'])
            refined['sun_glare']['source_overrides']['not_in_this_run.jpg'] = {'source': [0.5, 0.5]}
            presets.write_text(json.dumps(refined))
            bad = root / 'bad'
            self.assertNotEqual(self.invoke(inputs, scenes, bad, '--presets', str(presets)).returncode, 0)
            self.assertFalse(bad.exists())

    def test_source_overrides_are_validated_and_resolved_without_mutation(self):
        preset = json.loads((METHOD / 'configs/refined_presets.json').read_text())['sun_glare']
        before = json.dumps(preset)
        validate_preset_template(preset)
        known = next(iter(preset['source_overrides']))
        resolved = resolve_preset(preset, known)
        self.assertEqual(resolved['source'], preset['source_overrides'][known]['source'])
        resolved['source'][0] = 99
        self.assertEqual(json.dumps(preset), before)
        self.assertEqual(resolve_preset(preset, 'unlisted.jpg')['source'], preset['source'])
        for invalid in [None, [], {'../image.png': {'strength': 1}}, {'image.png': {'ev': 2}}, {'image.png': {}}, {'image.png': {'strength': -1}}]:
            with self.subTest(overrides=invalid), self.assertRaises(ValueError):
                validate_preset_template({**preset, 'source_overrides': invalid})


if __name__ == '__main__':
    unittest.main()
