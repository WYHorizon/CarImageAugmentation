"""Integration checks for reusable inputs, preflight and source protection."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
SPEC = importlib.util.spec_from_file_location('occlusion_run', METHOD / 'run.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / '新图片.png'
        pixels = np.arange(80 * 60 * 3, dtype=np.uint8).reshape(60, 80, 3)
        Image.fromarray(pixels).save(self.source)
        self.asset = self.root / 'material.png'
        rgba = np.zeros((12, 20, 4), dtype=np.uint8)
        rgba[2:10, 3:17] = [32, 140, 20, 255]
        Image.fromarray(rgba).save(self.asset)
        self.preset = self.root / 'preset.json'
        self.preset.write_text(json.dumps({'name': 'test', 'transform': {
            'width_fraction': 0.4, 'position_xy': [0.25, 0.35],
            'blur_short_edge_fraction': 0.003,
        }}))
        self.scenes = self.root / 'scenes.json'
        self.scenes.write_text(json.dumps({self.source.name: {
            'label': 'Custom test image',
            'subject_polygon': [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            'regions': {'detail': [0.2, 0.2, 0.6, 0.7]},
        }}))
        self.args = argparse.Namespace(input=self.source, asset=self.asset, preset=self.preset,
                                       scenes=self.scenes, output_dir=self.root / 'result', check_config=False)

    def run_quietly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.args)

    def test_custom_image_end_to_end_preserves_uncovered_pixels_and_inputs(self):
        source_hash, asset_hash = runner.sha256(self.source), runner.sha256(self.asset)
        result = self.run_quietly()
        self.assertEqual(source_hash, runner.sha256(self.source))
        self.assertEqual(asset_hash, runner.sha256(self.asset))
        output = self.args.output_dir
        before = np.asarray(Image.open(output / 'inputs/original.png'))
        after = np.asarray(Image.open(output / 'images/occluded.png'))
        alpha = np.asarray(Image.open(output / 'masks/alpha.png'))
        self.assertTrue(np.array_equal(before[alpha == 0], after[alpha == 0]))
        self.assertTrue(np.any(before != after))
        self.assertTrue(result['verification']['outside_alpha_support_pixel_identical'])
        self.assertIsNotNone(result['statistics']['vehicle'])
        self.assertTrue((output / 'report/index.html').is_file())
        self.assertEqual(result['verification']['png_files_checked'], len(list(output.rglob('*.png'))))

    def test_check_config_writes_nothing(self):
        self.args.check_config = True
        self.assertEqual(self.run_quietly()['status'], 'configuration_valid')
        self.assertFalse(self.args.output_dir.exists())

    def test_existing_results_are_not_overwritten(self):
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_quietly()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_rgb_material_rejected_before_writes(self):
        Image.new('RGB', (12, 12), 'white').save(self.asset)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            self.run_quietly()
        self.assertFalse(self.args.output_dir.exists())

    def test_missing_scene_does_not_invent_vehicle_annotations(self):
        self.scenes.write_text('{}')
        self.args.check_config = True
        result = self.run_quietly()
        self.assertIsNone(result['statistics']['vehicle'])
        self.assertIsNone(result['review']['vehicle_coverage_in_target_range'])
        self.assertIsNone(result['review']['minimum_region_count_met'])

    def test_empty_scene_regions_get_reportable_full_image_fallback(self):
        self.scenes.write_text(json.dumps({self.source.name: {'label': 'Custom', 'regions': {}}}))
        result = self.run_quietly()
        self.assertEqual(list(result['statistics']['regions']), ['full_image'])
        self.assertIsNone(result['review']['minimum_region_count_met'])
        self.assertTrue((self.args.output_dir / 'report/index.html').is_file())


if __name__ == '__main__':
    unittest.main()
