"""Check transformed crop correspondence, clipped regions and report links."""

import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageOps

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from composition.report import build_report, report_groups, _reduced_box, _subject_row
from composition.transforms import describe_polygon, plan_transform, transform_points


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs if name in {'href', 'src'})


class ReportTests(unittest.TestCase):
    def fixture(self, root, exif=False):
        source = root / 'source image.png'
        yy, xx = np.indices((60, 80))
        rgb = np.stack([xx * 3, yy * 4, (xx + yy) % 2 * 200], axis=-1).astype(np.uint8)
        if exif:
            orientation = Image.Exif()
            orientation[274] = 6
            Image.fromarray(np.rot90(rgb)).save(source, exif=orientation)
        else:
            Image.fromarray(rgb).save(source)
        scene = {'regions': {'headlamp': [0.11, 0.21, 0.36, 0.46],
                             'front_wheel': [0.6, 0.6, 0.9, 0.95]},
                 'subject_polygon': [[0.1, 0.2], [0.9, 0.2], [0.9, 0.95], [0.1, 0.95]]}
        roll = plan_transform((80, 60), {'kind': 'roll', 'angle': 12,
                                       'label': 'Roll', 'retention_policy': 'partial_allowed'})
        matrices = {
            'shift': np.array([[1, 0, 12], [0, 1, -4], [0, 0, 1.0]]),
            'scale': np.array([[1.5, 0, -20], [0, 1.5, -15], [0, 0, 1.0]]),
            'roll': np.array(roll['forward_matrix']),
            'outside': np.array([[1, 0, 200], [0, 1, 0], [0, 0, 1.0]]),
        }
        presets, outputs, geometry = {}, {}, {}
        subject = [[x * 80, y * 60] for x, y in scene['subject_polygon']]
        corners = [[0, 0], [80, 0], [80, 60], [0, 60]]
        for name, matrix in matrices.items():
            group = {'shift': 'horizontal', 'scale': 'zoom', 'roll': 'roll', 'outside': 'horizontal'}[name]
            presets[name] = {'label': name.title(), 'group': group,
                             'group_label': group + ' <test>', 'retention_policy': 'partial_allowed'}
            inverse = np.linalg.inv(matrix)
            path = root / f'output_{name}.png'
            Image.fromarray(rgb).transform((80, 60), Image.Transform.AFFINE,
                tuple(inverse[:2].ravel()), resample=Image.Resampling.BICUBIC).save(path)
            outputs[name] = path
            regions = {}
            for region, (x0, y0, x1, y1) in scene['regions'].items():
                quad = [[x0 * 80, y0 * 60], [x1 * 80, y0 * 60],
                        [x1 * 80, y1 * 60], [x0 * 80, y1 * 60]]
                regions[region] = describe_polygon(quad, matrix, (80, 60))
            geometry[name] = {
                'transform': {'forward_matrix': matrix.tolist(), 'inverse_matrix': inverse.tolist(),
                    'source_view_polygon_xy': transform_points(corners, inverse).tolist(),
                    'scale': float(math.sqrt(abs(np.linalg.det(matrix[:2, :2])))),
                    'angle_degrees': 12 if name == 'roll' else 0,
                    'retention_policy': 'partial_allowed'},
                'subject': describe_polygon(subject, matrix, (80, 60)), 'regions': regions,
            }
        row = {'source': source, 'label': 'Test <vehicle>', 'vehicle_id': 'test_car',
               'size': (80, 60), 'scene': scene, 'outputs': outputs, 'geometry': geometry}
        return row, presets

    def test_native_crops_follow_translation_scale_and_rotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, presets = self.fixture(root, exif=True)
            output = root / 'report output'
            report = build_report(output, [row], presets)
            crop_records = json.loads((output / 'crop_regions.json').read_text())['regions']
            lookup = {(r['preset_id'], r['region']): r for r in crop_records}
            self.assertEqual(lookup['original', 'headlamp']['native_box_xyxy'], [8, 12, 29, 28])
            self.assertEqual(lookup['shift', 'headlamp']['native_box_xyxy'], [20, 8, 41, 24])
            self.assertEqual(lookup['scale', 'headlamp']['native_box_xyxy'], [0, 3, 24, 27])
            self.assertLess(lookup['scale', 'headlamp']['retained_fraction'], 1)
            self.assertNotEqual(lookup['roll', 'headlamp']['native_box_xyxy'], lookup['original', 'headlamp']['native_box_xyxy'])
            native_count = 0
            for item in crop_records:
                if not item['visible']:
                    self.assertIsNone(item['crop_path'])
                    self.assertIsNone(item['native_box_xyxy'])
                    self.assertEqual(item['retained_fraction'], 0)
                    continue
                path = row['source'] if item['preset_id'] == 'original' else row['outputs'][item['preset_id']]
                with Image.open(path) as image:
                    expected = ImageOps.exif_transpose(image).convert('RGB').crop(item['native_box_xyxy'])
                with Image.open(output / item['crop_path']) as crop:
                    self.assertEqual(crop.format, 'PNG')
                    np.testing.assert_array_equal(np.asarray(crop), np.asarray(expected))
                native_count += 1
            self.assertEqual(report['native_crop_count'], native_count)
            self.assertEqual(native_count, 8)
            self.assertEqual(report['invisible_part_count'], 2)
            self.assertFalse((output / 'crops/test_car/headlamp/outside.png').exists())

    def test_224_boxes_scale_visible_geometry_after_whole_frame_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, presets = self.fixture(root)
            output = root / 'report output'
            build_report(output, [row], presets)
            crop_records = json.loads((output / 'crop_regions.json').read_text())['regions']
            lookup = {(r['preset_id'], r['region']): r for r in crop_records}
            original = lookup['original', 'headlamp']
            self.assertEqual(original['short_edge_224_size'], [299, 224])
            self.assertEqual(original['short_edge_224_box_xyxy'], [32, 47, 108, 104])
            shifted = lookup['shift', 'headlamp']
            self.assertEqual(shifted['short_edge_224_box_xyxy'], [77, 32, 153, 89])
            # Scaling an already-rounded native box would give x0=74 instead of77.
            self.assertNotEqual(shifted['short_edge_224_box_xyxy'][0], math.floor(shifted['native_box_xyxy'][0] * 299 / 80))
            for item in crop_records:
                if not item['visible']:
                    self.assertIsNone(item['short_edge_224_box_xyxy'])
                    continue
                xy = np.array(item['visible_polygon_xy']) * [299 / 80, 224 / 60]
                expected = [math.floor(xy[:, 0].min()), math.floor(xy[:, 1].min()),
                            math.ceil(xy[:, 0].max()), math.ceil(xy[:, 1].max())]
                self.assertEqual(item['short_edge_224_box_xyxy'], expected)

    def test_png_overlays_groups_links_and_inputs_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, presets = self.fixture(root)
            source_paths = [row['source'], *row['outputs'].values()]
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
            output = root / 'report output'
            output.mkdir()
            (output / 'manifest.json').write_text('{}\n')
            report = build_report(output, [row], presets)
            self.assertEqual(report['preview_png_count'], 9)
            self.assertFalse(list(output.rglob('*.jpg')))
            for suffix in ['comparison', 'framing', 'annotations', 'details', 'details_224']:
                with Image.open(output / f'previews/test_car_{suffix}.png') as image:
                    self.assertEqual(image.format, 'PNG')
            with Image.open(output / 'previews/horizontal_overview.png') as group:
                self.assertEqual(group.size, (1560, 452))
            page = output / report['report']
            text = page.read_text()
            for phrase in ['OUT OF FRAME', '裁切', '同一变换矩阵', '原生像素', '不使用 MAE', 'horizontal &lt;test&gt;']:
                self.assertIn(phrase, text)
            self.assertIn('Test &lt;vehicle&gt;', text)
            links = Links(); links.feed(text)
            for target in links.targets:
                if not target.startswith('#'):
                    self.assertTrue((page.parent / unquote(target)).exists(), target)
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
            for geometry in row['geometry'].values():
                self.assertEqual(set(geometry['regions']), {'headlamp', 'front_wheel'})

    def test_invalid_group_and_mismatched_geometry_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, presets = self.fixture(root)
            output = root / 'invalid'
            for metadata in [{'group': '../escape'}, {'group': ''}, {'group_label': 'Missing group'},
                             {'group': 'ok', 'group_label': ''}]:
                invalid = {**presets, 'shift': {'label': 'Invalid', **metadata}}
                with self.assertRaises(ValueError):
                    build_report(output, [row], invalid)
                self.assertFalse(output.exists())
            row['geometry'].pop('shift')
            with self.assertRaises(ValueError):
                build_report(output, [row], presets)
            self.assertFalse(output.exists())

    def test_group_order_and_polygon_reduced_bounds(self):
        groups = report_groups({'a': {'group': 'roll'}, 'b': {'group': 'zoom'}, 'c': {'group': 'roll'}})
        self.assertEqual(list(groups), ['roll', 'zoom'])
        self.assertEqual(groups['roll'][1], ['a', 'c'])
        self.assertIsNone(_reduced_box([], (80, 60), (299, 224)))
        self.assertEqual(_reduced_box([[0, 0], [80, 0], [80, 60], [0, 60]], (80, 60), (299, 224)), [0, 0, 299, 224])
        with self.assertRaises(ValueError):
            report_groups({'a': {'group': 'same', 'group_label': 'One'}, 'b': {'group': 'same', 'group_label': 'Two'}})

    def test_subject_center_tracks_original_center_for_asymmetric_rotated_polygon(self):
        # The axis-aligned envelope of this rotated triangle changes center.
        # Vehicle position must track the same original reference point instead.
        triangle = [[0, 0], [40, 0], [0, 20]]
        r = math.sqrt(0.5)
        matrix = [[r, -r, 20], [r, r, 5], [0, 0, 1]]
        original = {'subject': describe_polygon(triangle, np.eye(3), (80, 60))}
        record = {'subject': describe_polygon(triangle, matrix, (80, 60)),
                  'transform': {'forward_matrix': matrix, 'scale': 1.0,
                                'angle_degrees': 45, 'retention_policy': 'complete'}}
        row = _subject_row('roll', record, original, (80, 60), {'roll': {'label': 'Roll'}})
        # Original bbox center (20, 10) maps to (27.071..., 26.213...).
        self.assertIn('<td>33.8%, 43.7%</td>', row)
        self.assertIn('<td>+8.8%, +27.0%</td>', row)
        self.assertNotIn('<td>33.8%, 31.9%</td>', row)


if __name__ == '__main__':
    unittest.main()
