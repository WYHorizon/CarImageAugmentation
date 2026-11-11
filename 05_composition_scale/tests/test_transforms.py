"""Behavioral geometry checks for frame transforms and corresponding annotations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD))
from composition import (apply_composition, clip_polygon, describe_polygon,
                         plan_transform, polygon_area, transform_points,
                         validate_preset)

PRESETS = json.loads((METHOD / "configs/presets.json").read_text())


class CompositionTransformTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(19).integers(0, 256, (71, 97, 3), dtype=np.uint8)
        self.subject = [[0.15, 0.35], [0.85, 0.35], [0.85, 0.8], [0.15, 0.8]]

    def test_presets_valid_outputs_repeatable_and_inputs_unchanged(self):
        before, presets_before = self.rgb.copy(), copy.deepcopy(PRESETS)
        self.assertEqual(len(PRESETS), 8)
        for name, preset in PRESETS.items():
            with self.subTest(name=name):
                validate_preset(preset)
                first = apply_composition(self.rgb, preset, self.subject)
                second = apply_composition(self.rgb, preset, self.subject)
                self.assertEqual(first.image.shape, self.rgb.shape)
                self.assertEqual(first.image.dtype, np.uint8)
                self.assertFalse(np.shares_memory(first.image, self.rgb))
                np.testing.assert_array_equal(first.image, second.image)
                np.testing.assert_array_equal(self.rgb, before)
                np.testing.assert_array_equal(first.valid_mask, 255)
                self.assertEqual(first.metadata, second.metadata)
                json.dumps(first.metadata, allow_nan=False)
        self.assertEqual(PRESETS, presets_before)

    def test_identity_preserves_every_pixel_without_half_pixel_shift(self):
        for preset in [dict(PRESETS['zoom_close'], scale=1),
                       dict(PRESETS['position_left'], scale=1),
                       dict(PRESETS['roll_left'], angle=0)]:
            with self.subTest(kind=preset['kind']):
                result = apply_composition(self.rgb, preset, self.subject)
                np.testing.assert_array_equal(result.image, self.rgb)
                np.testing.assert_allclose(result.metadata['forward_matrix'], np.eye(3), atol=1e-12)

    def test_pillow_render_uses_same_edge_coordinates_as_annotation_matrix(self):
        yy, xx = np.indices((65, 65))
        # A ramp gives an analytic interpolation result away from image edges.
        rgb = np.stack([xx * 3, yy * 3, np.full(xx.shape, 120)], axis=-1).astype(np.uint8)
        result = apply_composition(rgb, dict(PRESETS['zoom_close'], scale=2))
        points = np.asarray([[12.5, 17.5], [30.5, 35.5], [45.5, 48.5]])
        source = transform_points(points, result.metadata['inverse_matrix'])
        expected = np.column_stack([(source[:, 0] - .5) * 3,
                                    (source[:, 1] - .5) * 3, np.full(3, 120)])
        samples = result.image[points[:, 1].astype(int), points[:, 0].astype(int)]
        np.testing.assert_allclose(samples, expected, atol=1, rtol=0)

    def test_zoom_keeps_subject_anchor_and_has_uniform_scale(self):
        subject = [[.1, .2], [.5, .2], [.5, .6], [.1, .6]]
        meta = plan_transform((400, 300), PRESETS['zoom_close'], subject)
        anchor = np.asarray([[120, 120]])
        np.testing.assert_allclose(transform_points(anchor, meta['forward_matrix']), anchor)
        self.assertEqual(meta['anchor_kind'], 'subject_bbox_center')
        self.assertAlmostEqual(meta['scale'], 1.3)
        points = np.asarray([[10, 20], [20, 20], [10, 30]])
        transformed = transform_points(points, meta['forward_matrix'])
        self.assertAlmostEqual(np.linalg.norm(transformed[1] - transformed[0]), 13)
        self.assertAlmostEqual(np.linalg.norm(transformed[2] - transformed[0]), 13)
        fallback = plan_transform((400, 300), PRESETS['zoom_close'])
        self.assertEqual(fallback['anchor_xy'], [200, 150])

    def test_position_names_describe_vehicle_motion_not_viewport_motion(self):
        point = [[200, 180]]
        centers = {}
        for direction in ['left', 'right', 'up', 'down']:
            meta = plan_transform((400, 300), PRESETS['position_' + direction])
            centers[direction] = transform_points(point, meta['forward_matrix'])[0]
        self.assertLess(centers['left'][0], 200)
        self.assertGreater(centers['right'][0], 200)
        self.assertLess(centers['up'][1], 180)
        self.assertGreater(centers['down'][1], 180)
        self.assertAlmostEqual(centers['left'][0] / 400, .4)
        self.assertAlmostEqual(centers['right'][0] / 400, .6)
        self.assertAlmostEqual(centers['up'][1] / 300, .52)
        self.assertAlmostEqual(centers['down'][1] / 300, .72)

    def test_matrices_are_inverse_and_viewport_corners_map_to_output_corners(self):
        size = (403, 269)
        corners = np.asarray([[0, 0], [403, 0], [403, 269], [0, 269]])
        for preset in PRESETS.values():
            meta = plan_transform(size, preset, self.subject)
            forward, inverse = np.asarray(meta['forward_matrix']), np.asarray(meta['inverse_matrix'])
            np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(transform_points(meta['source_view_polygon_xy'], forward), corners, atol=1e-10)
            view = np.asarray(meta['source_view_polygon_xy'])
            self.assertTrue(np.all(view >= -1e-10))
            self.assertTrue(np.all(view <= np.asarray(size) + 1e-10))

    def test_roll_sign_is_clockwise_and_image_center_stays_fixed(self):
        for name, sign in [('roll_left', -1), ('roll_right', 1)]:
            meta = plan_transform((403, 269), PRESETS[name])
            center = np.asarray([[403 / 2, 269 / 2]])
            np.testing.assert_allclose(transform_points(center, meta['forward_matrix']), center)
            right = transform_points(center + [20, 0], meta['forward_matrix'])[0]
            self.assertGreater(sign * (right[1] - center[0, 1]), 0)
            linear = np.asarray(meta['forward_matrix'])[:2, :2]
            singular_values = np.linalg.svd(linear, compute_uv=False)
            np.testing.assert_allclose(singular_values, [meta['scale']] * 2)

    def test_rendered_roll_landmark_agrees_with_transformed_position(self):
        rgb = np.zeros((101, 101, 3), dtype=np.uint8)
        rgb[49:52, 69:72] = 255
        for name in ['roll_left', 'roll_right']:
            result = apply_composition(rgb, PRESETS[name])
            predicted = transform_points([[70.5, 50.5]], result.metadata['forward_matrix'])[0]
            intensity = result.image[..., 0].astype(float)
            yy, xx = np.indices(intensity.shape)
            actual = np.array([np.sum((xx + .5) * intensity), np.sum((yy + .5) * intensity)]) / intensity.sum()
            np.testing.assert_allclose(actual, predicted, atol=.15)

    def test_cover_scale_handles_portrait_landscape_and_singleton_axes_without_padding(self):
        for shape in [(48, 64, 3), (65, 37, 3), (3, 41, 3), (1, 19, 3), (19, 1, 3)]:
            rgb = np.full(shape, [27, 128, 244], dtype=np.uint8)
            for name in ['zoom_close', 'position_up', 'roll_left', 'roll_right']:
                with self.subTest(shape=shape, name=name):
                    result = apply_composition(rgb, PRESETS[name])
                    np.testing.assert_array_equal(result.image, rgb)
                    np.testing.assert_array_equal(result.valid_mask, 255)
                    self.assertEqual(result.metadata['valid_pixel_fraction'], 1)

    def test_rotation_margin_and_effective_scale_are_reported(self):
        meta = plan_transform((3840, 2880), PRESETS['roll_left'])
        self.assertEqual(meta['rotation_source_margin_px'], 2)
        self.assertAlmostEqual(meta['rotation_minimum_cover_scale'], 1.255363188491124, places=12)
        self.assertGreater(meta['scale'], meta['rotation_minimum_cover_scale'])
        self.assertLess(meta['scale'], 1.26)
        view = np.asarray(meta['source_view_polygon_xy'])
        self.assertTrue(np.all(view >= 2 - 1e-9))
        self.assertTrue(np.all(view <= np.asarray([3840, 2880]) - 2 + 1e-9))

    def test_polygon_clip_and_retention_measure_full_half_and_absent_regions(self):
        polygon = [[10, 20], [30, 20], [30, 40], [10, 40]]
        identity = np.eye(3)
        full = describe_polygon(polygon, identity, (50, 50))
        self.assertEqual(full['retained_fraction'], 1)
        self.assertEqual(full['crop_box_xyxy'], [10, 20, 30, 40])
        shifted = identity.copy(); shifted[0, 2] = -20
        half = describe_polygon(polygon, shifted, (50, 50))
        self.assertEqual(half['original_area'], 400)
        self.assertEqual(half['transformed_area'], 400)
        self.assertEqual(half['visible_area'], 200)
        self.assertEqual(half['retained_fraction'], .5)
        self.assertEqual(half['crop_box_xyxy'], [0, 20, 10, 40])
        shifted[0, 2] = 100
        absent = describe_polygon(polygon, shifted, (50, 50))
        self.assertFalse(absent['visible'])
        self.assertEqual(absent['retained_fraction'], 0)
        self.assertIsNone(absent['crop_box_xyxy'])
        self.assertIsNone(absent['visible_bbox_xyxy'])
        self.assertEqual(absent['visible_polygon_xy'], [])

    def test_polygon_area_is_winding_independent_and_retention_uses_scaled_area(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertEqual(polygon_area(polygon), polygon_area(list(reversed(polygon))))
        matrix = np.diag([2., 2., 1.])
        result = describe_polygon(polygon, matrix, (10, 20))
        self.assertEqual(result['original_area'], 100)
        self.assertEqual(result['transformed_area'], 400)
        self.assertEqual(result['visible_area'], 200)
        self.assertEqual(result['retained_fraction'], .5)

    def test_rotated_roi_uses_polygon_retention_and_bounding_crop(self):
        # A diamond spanning [-5,15] in both axes retains half its area in [0,10]^2.
        polygon = [[5, -5], [15, 5], [5, 15], [-5, 5]]
        result = describe_polygon(polygon, np.eye(3), (10, 10))
        self.assertEqual(result['transformed_area'], 200)
        self.assertEqual(result['visible_area'], 100)
        self.assertEqual(result['retained_fraction'], .5)
        self.assertEqual(result['crop_box_xyxy'], [0, 0, 10, 10])
        self.assertEqual(polygon_area(clip_polygon(list(reversed(polygon)), 10, 10)), 100)

    def test_crop_box_uses_floor_ceil_and_does_not_include_invisible_zero_area(self):
        polygon = [[1.2, 2.3], [8.7, 2.3], [8.7, 9.1], [1.2, 9.1]]
        result = describe_polygon(polygon, np.eye(3), (10, 10))
        self.assertEqual(result['crop_box_xyxy'], [1, 2, 9, 10])
        touching = [[10, 2], [12, 2], [12, 8], [10, 8]]
        result = describe_polygon(touching, np.eye(3), (10, 10))
        self.assertFalse(result['visible'])
        self.assertIsNone(result['crop_box_xyxy'])

    def test_actual_sample_geometry_preserves_complete_presets_and_flags_tight_crops(self):
        scenes = json.loads((METHOD.parent / '01_physics_based/configs/scenes.json').read_text())
        for filename, scene in scenes.items():
            points = np.asarray(scene['subject_polygon']) * [3840, 2880]
            for name, preset in PRESETS.items():
                with self.subTest(filename=filename, name=name):
                    meta = plan_transform((3840, 2880), preset, scene['subject_polygon'])
                    annotation = describe_polygon(points, meta['forward_matrix'], (3840, 2880))
                    if preset['retention_policy'] == 'complete':
                        self.assertAlmostEqual(annotation['retained_fraction'], 1, places=10)
                    elif name == 'zoom_tight':
                        self.assertGreater(annotation['retained_fraction'], .9)
                        self.assertLess(annotation['retained_fraction'], .99)
                    self.assertGreater(annotation['visible_area'], 0)

    def test_invalid_presets_geometry_and_images_fail_explicitly(self):
        bad_presets = [None, {}, dict(PRESETS['zoom_close'], label=''),
                       dict(PRESETS['zoom_close'], scale=.5), dict(PRESETS['zoom_close'], scale=True),
                       dict(PRESETS['zoom_close'], scale=float('nan')),
                       dict(PRESETS['roll_left'], angle=46), dict(PRESETS['roll_left'], angle=None),
                       dict(PRESETS['position_left'], direction='diagonal'),
                       dict(PRESETS['zoom_close'], kind=[]),
                       dict(PRESETS['zoom_close'], retention_policy='magic'),
                       dict(PRESETS['zoom_close'], group='../bad')]
        for preset in bad_presets:
            with self.subTest(preset=preset), self.assertRaises(ValueError):
                validate_preset(preset)
        for size in [(0, 20), (20, -1), (True, 20), (20., 20), (65537, 20), [20]]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                plan_transform(size, PRESETS['zoom_close'])
        for subject in [[], [[0, 0], [1, 1]], [[0, 0], [1, 1], [2, 2]],
                        [[0, 0], [1, 0], [float('nan'), 1]], [[0, 0], [1, 1], [.5, .5]]]:
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                plan_transform((40, 30), PRESETS['zoom_close'], subject)
        for rgb in [np.zeros((2, 2)), np.zeros((0, 2, 3), dtype=np.uint8),
                    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2, 3)), []]:
            with self.subTest(shape=getattr(rgb, 'shape', None)), self.assertRaises(ValueError):
                apply_composition(rgb, PRESETS['zoom_close'])
        for matrix in [np.zeros((3, 3)), np.ones((2, 2)), [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]]]:
            with self.assertRaises(ValueError):
                transform_points([[1, 2]], matrix)


if __name__ == '__main__':
    unittest.main()
