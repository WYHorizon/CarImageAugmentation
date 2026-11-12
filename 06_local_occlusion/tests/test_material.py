import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occlusion.material import prepare_material


def checker(width=160, height=120, tile=10):
    yy, xx = np.indices((height, width))
    gray = np.where(((xx // tile) + (yy // tile)) % 2, 255, 238).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


class MaterialTests(unittest.TestCase):
    def test_existing_alpha_is_preserved_exactly_including_hidden_rgb(self):
        pixels = np.array([[[14, 90, 30, 0], [10, 80, 20, 37]],
                           [[40, 50, 60, 128], [55, 25, 65, 255]]], dtype=np.uint8)
        for mode in ("alpha", "light-checker"):
            result, metadata = prepare_material(Image.fromarray(pixels, "RGBA"), mode)
            np.testing.assert_array_equal(np.asarray(result), pixels)
            self.assertTrue(metadata["alpha_preserved"])

    def test_rgb_requires_explicit_background_mode(self):
        with self.assertRaisesRegex(ValueError, "no transparency"):
            prepare_material(Image.fromarray(checker()))

    def test_fully_transparent_material_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completely transparent"):
            prepare_material(Image.new("RGBA", (20, 20), (40, 100, 20, 0)))

    def test_checker_and_holes_removed_but_leaf_and_thin_twig_retained(self):
        pixels = checker()
        pixels[25:65, 25:80] = (45, 125, 25)
        pixels[38:48, 45:55] = checker()[38:48, 45:55]
        pixels[75, 20:110] = (35, 28, 20)  # One-pixel twig, no eroded interior.
        rgba, metadata = prepare_material(Image.fromarray(pixels), "light-checker")
        result = np.asarray(rgba)
        self.assertTrue(np.all(result[:20, :, 3] == 0))
        self.assertTrue(np.all(result[39:47, 46:54, 3] == 0))
        np.testing.assert_array_equal(result[30, 30], [45, 125, 25, 255])
        np.testing.assert_array_equal(result[75, 60], [35, 28, 20, 255])
        self.assertAlmostEqual(metadata["checker_agreement"], 1)

    def test_antialiased_edge_is_unmixed_instead_of_carrying_white_halo(self):
        background = checker()
        alpha = np.zeros(background.shape[:2], dtype=np.float32)
        alpha[29:71, 39:101] = 0.25
        alpha[30:70, 40:100] = 1
        color = np.array([40, 110, 30], dtype=np.float32)
        observed = np.rint(background * (1 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
        result, _ = prepare_material(Image.fromarray(observed), "light-checker")
        pixels = np.asarray(result)
        self.assertLess(abs(int(pixels[29, 60, 3]) - 64), 5)
        self.assertLess(np.max(np.abs(pixels[29, 60, :3].astype(int) - color)), 8)
        self.assertEqual(pixels[28, 60, 3], 0)

    def test_plain_white_or_nonperiodic_background_is_rejected(self):
        for background in (np.full((120, 160, 3), 255, dtype=np.uint8),
                           np.tile(np.linspace(220, 255, 160).astype(np.uint8)[None, :, None], (120, 1, 3))):
            background[30:70, 40:90] = (50, 130, 30)
            with self.assertRaisesRegex(ValueError, "regular pale neutral checkerboard"):
                prepare_material(Image.fromarray(background), "light-checker")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must"):
            prepare_material(Image.new("RGBA", (10, 10)), "automatic")

    def test_cli_rejects_output_that_would_collide_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            Image.new('RGBA', (20, 20), (40, 100, 20, 128)).save(source)
            before = source.read_bytes()
            output = root / 'material.json'
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / 'prepare_asset.py'),
                 '--input', str(source), '--output', str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('.png extension', completed.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
