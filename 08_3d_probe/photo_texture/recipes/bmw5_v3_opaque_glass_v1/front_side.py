"""Opaque glass color fields for the fitted v3 windshield and left side."""
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
PILOT = OUTPUTS / "bmw5_v3_glass_pilot_v1"
sys.path.insert(0, str(PILOT))
import build_glass as pilot


def side_smoke(s, t):
    """Closed charcoal glass. s increases toward the trunk, t toward the sill."""
    s, t = np.broadcast_arrays(np.clip(s, 0, 1), np.clip(t, 0, 1))
    broad_light = np.exp(-((s - .33 - .34 * t) / .35) ** 2)
    light = 17 * broad_light + 18 * (1 - t) ** 1.6 + 4 * (1 - s)
    return np.array([24., 29., 33.]) + light[..., None] * [1., 1.13, 1.20]


def windshield(s, t):
    # One broad light falloff across the full pane. A smooth edge darkening
    # gives the glass depth without introducing an identifiable reflected lamp.
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    lobe = np.exp(-((s - .64 + .25 * t) / .56) ** 2)
    sweep = np.exp(-((s - .29 - .43 * t - .05 * t**2) / .23) ** 2)
    light = 26 * (1 - t) + 13 * lobe + 17 * sweep - 8 * (2 * s - 1) ** 4
    light -= 8 * np.exp(-(t / .055) ** 2)
    return np.array([33., 40., 46.]) + light[..., None] * [1., 1.12, 1.18]


def sticker_coordinates(x, y):
    # Same chart-to-source mapping as the reviewed pilot. Only the original
    # head-on photo stickers are sampled. Their surroundings receive smoke.
    s = (x - 430) / 290
    for _ in range(4):
        a = np.clip(abs(s), 0, 1)
        top = np.interp(a, [0, .4, .76, 1], [139, 135, 123, 147])
        bottom = np.interp(a, [0, .35, .64, .84, 1], [448, 441, 424, 432, 427])
        t = np.clip((y - top) / (bottom - top), 0, 1)
        half = np.interp(t, [0, .18, .6, 1], [247, 256, 275, 293])
        s = (x - 430) / half
    a = np.clip(abs(s), 0, 1)
    left = np.interp(t, [0, .25, .73, 1], [221, 240, 264, 274])
    right = np.interp(t, [0, .25, .73, 1], [218, 235, 260, 267])
    photo_top = np.interp(a, [0, .5, .84, 1], [249, 250, 253, 261])
    photo_bottom = np.interp(a, [0, .65, .91, 1], [373, 373, 373, 368])
    return 770 + s * np.where(s < 0, left, right), photo_top + t * (photo_bottom - photo_top), s, t


def front_colors(x, y):
    u, v, s, t = sticker_coordinates(x, y)
    color = windshield((s + 1) / 2, t)
    photo = pilot.rgb(OUTPUTS / "bmw5_inputs_v2/front_002/source_rgb.png")
    masks = pilot.DONORS / "bmw5_front_glass_mask_v2"
    alpha = np.asarray(Image.open(masks / "sticker_and_cover_protection.png").convert("L")).copy()
    alpha[320:, :] = 0
    alpha[:, 619:] = 0
    fringe = cv2.GaussianBlur(alpha.astype(np.float32) / 255, (3, 3), .45)
    weight = pilot.sample(fringe, u, v)[:, None]
    rgb = pilot.sample(photo, u, v)
    return color + weight * (rgb - color)


def left_side_colors(x, y):
    roof = np.interp(x, [254, 308, 380, 455, 545, 641, 731, 798, 861],
                     [211, 177, 152, 144, 141, 141, 147, 159, 179])
    sill = np.interp(x, [254, 278, 466, 545, 701, 798, 871],
                     [259, 258, 248, 248, 244, 239, 230])
    return side_smoke((x - 254) / 617, (y - roof) / (sill - roof))


def apply(base, vertices, faces, uv):
    """Return edited RGB atlas and labels (1 windshield, 2 left windows)."""
    height, width = base.shape[:2]
    ids = np.asarray(Image.open(PILOT / "window_ids.png")).copy()
    ids[ids > 2] = 0
    valid = ids > 0
    screen = np.column_stack((uv[:, 0] * (width - 1), (1 - uv[:, 1]) * (height - 1), np.ones(len(uv))))
    _, face_ids, bary = pilot.render(screen, faces, {"projection":"orthographic", "focal_px":1, "cx":0, "cy":0}, (height, width))
    assert np.all(face_ids[valid] >= 0)
    xyz = np.einsum("ij,ijk->ik", bary[valid], vertices[faces[face_ids[valid]]])
    flat = np.flatnonzero(valid)
    result = base.copy().reshape(-1, 3)
    for i, name in enumerate(("front", "side"), 1):
        pane = ids.flat[flat] == i
        texels = flat[pane]
        x, y, _, _ = pilot.project(xyz[pane], pilot.CHARTS[name])
        paths = [pilot.FRONT] if name == "front" else pilot.SIDE
        edge = pilot.outline(pilot.SHAPES[name], paths, 3 if name == "front" else 2)
        alpha = pilot.sample(edge, x, y)
        if name == "front":
            colors = front_colors(x, y)
        else:
            colors = left_side_colors(x, y)
            # Near quarter-light divider on the actual outer pane. The other
            # bar farther forward in the reference is inside the cabin.
            divider_center = np.interp(y, [150, 190, 242], [761.8, 756.6, 749.8])
            trim = np.clip(5.4 - abs(x - divider_center), 0, 1)
            alpha *= 1 - trim
            ids.flat[texels[alpha == 0]] = 0
        orig = result[texels].astype(np.float32)
        result[texels] = np.rint(orig + alpha[:, None] * (colors - orig)).clip(0,255).astype(np.uint8)
    result = result.reshape(base.shape)
    hood = np.asarray(Image.open(pilot.V3 / "hood_footprint.png")) > 0
    assert np.array_equal(base[hood], result[hood])
    return result, ids


if __name__ == "__main__":
    mesh = np.load(pilot.BASE / "textured_mesh.npz")
    vertices, faces, uv = (mesh[k] for k in ("vertices", "faces", "uv"))
    base = pilot.rgb(pilot.V3 / "texture.png")
    edited, labels = apply(base, vertices, faces, uv)
    for name, camera in pilot.CHARTS.items():
        h, w = pilot.SHAPES[name]
        frame = pilot.render_texture(pilot.render, vertices, faces, uv, edited, camera, w, h)
        Image.fromarray(np.asarray(frame)).save("/tmp/bmw5_opaque_" + name + ".png")
