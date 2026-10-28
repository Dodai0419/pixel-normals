import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pixel_normals as pn  # noqa: E402

GOLDEN = ROOT / "tests" / "golden"
SAMPLES = ROOT / "docs" / "samples"


def disk(size):
    c = size // 2
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = (145, 99, 63)
    yy, xx = np.mgrid[0:size, 0:size]
    img[..., 3] = np.where(np.hypot(xx - c, yy - c) <= c - 3, 255, 0)
    return img


def gen(img, **kw):
    return pn.generate_normal(img, pn.settings(**kw)).astype(int)


class MatchesUpstream(unittest.TestCase):
    """Golden files were written by the upstream Rust library (see README, Benchmarks)."""

    def check(self, src, golden, **kw):
        normal, color, meta = pn.process(SAMPLES / src, pn.settings(**kw))
        want = np.asarray(Image.open(GOLDEN / golden).convert("RGBA")).astype(int)
        self.assertEqual(normal.shape, want.shape)
        diff = np.abs(normal.astype(int) - want).max(-1)
        self.assertLessEqual(diff.max(), 1)
        self.assertGreaterEqual((diff == 0).mean(), 0.99)
        self.assertTrue((normal[..., 3] == want[..., 3]).all())
        return color, meta

    def test_single_sprites(self):
        for name in ("orb", "gem", "mushroom", "sword"):
            with self.subTest(name):
                self.check(f"{name}.png", f"{name}_n.png")

    def test_gif_atlas_color_and_metadata(self):
        color, meta = self.check("slime.gif", "slime_n.png")
        self.assertTrue(np.array_equal(color, np.asarray(Image.open(GOLDEN / "slime_n.color.png"))))
        self.assertEqual(json.loads(json.dumps(meta)), json.loads((GOLDEN / "slime_n.frames.json").read_text()))

    def test_sheet_with_frame_size(self):
        color, meta = self.check("slime_sheet.png", "slime_sheet_n.png", frame_size=(24, 24))
        self.assertEqual(json.loads(json.dumps(meta)), json.loads((GOLDEN / "slime_sheet_n.frames.json").read_text()))


class UpstreamBehaviour(unittest.TestCase):
    """Ports of upstream tests/generation.rs."""

    def test_normals_point_outward_and_are_unit_vectors(self):
        r = gen(disk(31))
        self.assertLess(r[15, 7, 0], 100)
        self.assertGreater(r[15, 23, 0], 155)
        self.assertGreater(r[7, 15, 1], 155)
        self.assertLess(r[23, 15, 1], 100)
        self.assertEqual(r[15, 15].tolist(), [128, 128, 255, 255])
        length = np.sqrt(((r[..., :3] / 127.5 - 1) ** 2).sum(-1))
        self.assertLess(np.abs(length - 1).max(), 0.01)

    def test_zero_strength_is_neutral(self):
        img = disk(31)
        for kw in ({"normal_strength": 0.0}, {"shape_strength": 0.0}):
            r = gen(img, **kw)
            self.assertTrue((r[..., :3] == (128, 128, 255)).all())
            self.assertTrue((r[..., 3] == img[..., 3]).all())

    def test_broad_painted_shading_keeps_relief(self):
        img = np.zeros((32, 64, 4), np.uint8)
        img[..., :3] = (40 + np.arange(64) * 3)[None, :, None]
        img[..., 3] = 255
        r = gen(img, shape_strength=0.0)
        self.assertTrue((r[16, 12:52, 0] < 123).all())
        self.assertTrue((np.abs(r[16, 12:52, 1] - 128) <= 1).all())

    def test_alpha_kept_and_hidden_rgb_ignored(self):
        img = np.zeros((16, 16, 4), np.uint8)
        img[..., :3] = (90, 160, 210)
        img[..., 3] = np.arange(256).reshape(16, 16)
        r = gen(img)
        self.assertTrue((r[..., 3] == img[..., 3]).all())
        self.assertTrue((r[img[..., 3] <= 7][:, :3] == (128, 128, 255)).all())
        a, b = disk(31), disk(31)
        b[b[..., 3] == 0] = (255, 20, 200, 0)
        self.assertTrue(np.array_equal(gen(a), gen(b)))

    def test_nearby_islands_do_not_change_a_sprite(self):
        sprite = np.zeros((9, 9, 4), np.uint8)
        sprite[..., 0] = 30 + np.arange(9) * 20
        sprite[..., 1:] = (80, 120, 255)
        sheet = np.zeros((34, 42, 4), np.uint8)
        sheet[5:14, 5:14] = sprite
        sheet[5:28, 15:36] = 255
        self.assertTrue(np.array_equal(gen(sprite), gen(sheet)[5:14, 5:14]))

    def test_sheet_cells_are_isolated(self):
        sheet = np.zeros((16, 32, 4), np.uint8)
        sheet[:, :16] = (220, 70, 40, 255)
        sheet[:, 16:] = (30, 80, 200, 255)
        r = gen(sheet, frame_size=(16, 16))
        for x in (0, 16):
            self.assertTrue(np.array_equal(gen(sheet[:, x:x + 16]), r[:, x:x + 16]))

    def test_tiny_and_diagonal_shapes(self):
        for w, h in ((1, 1), (1, 19), (23, 1), (20, 20)):
            img = np.zeros((h, w, 4), np.uint8)
            img[..., :3] = 80
            yy, xx = np.mgrid[0:h, 0:w]
            img[..., 3] = np.where((w == 1) | (h == 1) | (xx == yy), 255, 0)
            self.assertTrue(np.array_equal(gen(img), gen(img)))

    def test_conventions_flip_only_requested_channels(self):
        gl, dx, flip = gen(disk(31)), gen(disk(31), directx=True), gen(disk(31), invert_x=True, invert_y=True)
        self.assertTrue((gl[..., [0, 2, 3]] == dx[..., [0, 2, 3]]).all())
        self.assertLessEqual(np.abs(gl[..., 1] + dx[..., 1] - 255).max(), 1)
        self.assertLessEqual(np.abs(gl[..., 0] + flip[..., 0] - 255).max(), 1)
        self.assertTrue((dx[..., 1] == flip[..., 1]).all())

    def test_invalid_settings_and_dimensions(self):
        for v in (math.nan, math.inf, -1.0, 9.0):
            self.assertRaises(ValueError, pn.settings, normal_strength=v)
        self.assertRaises(ValueError, pn.settings, shape_radius=0.0)
        self.assertRaises(ValueError, pn.settings, frame_size=(0, 1))
        for size in ((16, 16), (32, 32)):
            self.assertRaises(ValueError, pn.generate_normal, disk(31), pn.settings(frame_size=size))
        self.assertRaises(ValueError, pn.generate_normal, np.zeros((0, 0, 4), np.uint8), pn.settings())


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "pixel_normals.py"), *map(str, args)], capture_output=True, text=True)

    def test_outputs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            r = self.run_cli(SAMPLES / "slime.gif", "-o", d / "walk_n.png")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(all((d / n).exists() for n in ("walk_n.png", "walk_n.color.png", "walk_n.frames.json")))
            self.assertNotEqual(self.run_cli(SAMPLES / "slime.gif", "-o", d / "walk_n.png").returncode, 0)
            self.assertEqual(self.run_cli(SAMPLES / "slime.gif", "-o", d / "walk_n.png", "--force").returncode, 0)
            r = self.run_cli(SAMPLES / "orb.png", SAMPLES / "gem.png", "--output-dir", d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((d / "orb_n.png").exists() and (d / "gem_n.png").exists())
            self.assertNotEqual(self.run_cli(SAMPLES / "orb.png", "-o", SAMPLES / "orb.png", "--force").returncode, 0)


if __name__ == "__main__":
    unittest.main()
