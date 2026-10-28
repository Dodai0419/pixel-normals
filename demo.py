"""Draw a few pixel-art sprites, make their normal maps, and light them with a moving lamp.

Writes docs/samples/*.png|gif (inputs), docs/out/* (normal maps) and docs/gallery.png + docs/lit.gif.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

import pixel_normals as pn

ROOT = Path(__file__).parent / "docs"
SCALE = 6


def canvas(n):
    return np.zeros((n, n, 4), np.uint8)


def shade(base, t):
    """Pixel-art ramp: 4 flat bands from dark to light."""
    k = (0.45, 0.7, 0.9, 1.15)[min(3, int(t * 4))]
    return [min(255, int(c * k)) for c in base] + [255]


def orb(n=24, base=(70, 130, 220)):
    img, c = canvas(n), (n - 1) / 2
    for y in range(n):
        for x in range(n):
            d = math.hypot(x - c, y - c) / (n / 2 - 1)
            if d <= 1:
                img[y, x] = shade(base, (1 - d) * 0.6 + max(0, (c - x) + (c - y)) / n * 0.5)
    return img


def gem(n=24, base=(200, 60, 120)):
    img, c = canvas(n), n // 2
    for y in range(2, n - 2):
        half = (y - 2) if y < 8 else int((n - 2 - y) * 6 / (n - 10))
        for x in range(c - half, c + half + 1):
            facet = 0.85 if x < c and y < 8 else 0.55 if x >= c and y >= 8 else 0.3 if x >= c else 0.7
            img[y, x] = shade(base, facet)
    return img


def mushroom(n=24):
    img, c = canvas(n), (n - 1) / 2
    for y in range(n):
        for x in range(n):
            if y < 13 and math.hypot((x - c) / 10.5, (y - 12) / 10) <= 1:
                spot = math.hypot(x - 7, y - 6) < 2 or math.hypot(x - 15, y - 8) < 2 or math.hypot(x - 11, y - 3) < 1.5
                img[y, x] = shade((240, 240, 230) if spot else (210, 50, 40), 0.9 - (y / 13) * 0.6 + (c - x) / n * 0.4)
            elif 13 <= y < 22 and abs(x - c) <= 3.5:
                img[y, x] = shade((235, 215, 180), 0.75 - (x - c) / 8)
    return img


def sword(n=24):
    img = canvas(n)
    for i in range(3, 17):
        img[n - 1 - i - 2, i + 2] = shade((200, 210, 225), 0.95)
        img[n - 1 - i - 1, i + 2] = shade((200, 210, 225), 0.6)
        img[n - 1 - i - 2, i + 3] = shade((200, 210, 225), 0.4)
    for i in range(-3, 4):
        img[n - 6 + i, 5 + i] = shade((150, 100, 40), 0.7)
    for i in range(3):
        img[n - 3 + i - 1, 2 - i + 1] = shade((110, 70, 30), 0.5)
    return img


def slime_frames(n=24, count=6):
    frames = []
    for f in range(count):
        squash = 1 + 0.18 * math.sin(f / count * 2 * math.pi)
        img, c = canvas(n), (n - 1) / 2
        for y in range(n):
            for x in range(n):
                rx, ry = 9 * squash, 8 / squash
                d = math.hypot((x - c) / rx, (y - (n - 2 - ry)) / ry)
                if d <= 1 and y <= n - 2:
                    img[y, x] = shade((80, 200, 90), (1 - d) * 0.7 + (c - x) / n * 0.4 + 0.15)
        img[int(n - 2 - 8 / squash), [9, 14]] = (20, 30, 20, 255)
        frames.append(img)
    return frames


def light(color, normal, lx, ly, lz=0.7):
    """Diffuse lighting from a point lamp at (lx, ly) in pixel space; OpenGL normal convention."""
    n = normal[..., :3].astype(np.float32) / 127.5 - 1
    h, w = n.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    L = np.stack([lx - xx, -(ly - yy), np.full((h, w), lz * max(h, w))], -1).astype(np.float32)
    L /= np.linalg.norm(L, axis=-1, keepdims=True)
    diffuse = np.clip((n * L).sum(-1), 0, 1)
    out = color.copy()
    out[..., :3] = np.clip(color[..., :3] * (0.25 + 0.95 * diffuse)[..., None], 0, 255)
    return out


def big(a):
    return Image.fromarray(a, "RGBA").resize((a.shape[1] * SCALE, a.shape[0] * SCALE), Image.NEAREST)


def main():
    (ROOT / "samples").mkdir(parents=True, exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    sprites = {"orb": orb(), "gem": gem(), "mushroom": mushroom(), "sword": sword()}
    for name, img in sprites.items():
        Image.fromarray(img, "RGBA").save(ROOT / "samples" / f"{name}.png")
    frames = slime_frames()
    Image.fromarray(np.concatenate(frames, 1), "RGBA").save(ROOT / "samples" / "slime_sheet.png")
    Image.fromarray(frames[0], "RGBA").save(ROOT / "samples" / "slime.gif", save_all=True, duration=120, loop=0,
                                            append_images=[Image.fromarray(f, "RGBA") for f in frames[1:]], disposal=2)
    s = pn.settings()
    rows, maps = [], {}
    for name in sprites:
        normal, color, _ = pn.process(ROOT / "samples" / f"{name}.png", s)
        Image.fromarray(normal, "RGBA").save(ROOT / "out" / f"{name}_n.png")
        maps[name] = (color, normal)
        rows.append([color, normal, light(color, normal, 2, 2), light(color, normal, 22, 20)])
    sheet_normal, sheet_color, meta = pn.process(ROOT / "samples" / "slime_sheet.png", pn.settings(frame_size=(24, 24)))
    Image.fromarray(sheet_normal, "RGBA").save(ROOT / "out" / "slime_sheet_n.png")
    print(f"slime sheet: {meta['animation']['frame_count']} frames")

    # gallery: sprite | normal map | lit from top-left | lit from bottom-right
    cell, gap = 24 * SCALE, 12
    bg = (24, 26, 32, 255)
    gallery = Image.new("RGBA", (4 * cell + 5 * gap, len(rows) * cell + (len(rows) + 1) * gap), bg)
    for r, row in enumerate(rows):
        for c, a in enumerate(row):
            tile = big(a)
            gallery.alpha_composite(tile, (gap + c * (cell + gap), gap + r * (cell + gap)))
    gallery.save(ROOT / "gallery.png")

    # lamp orbiting all sprites and the slime animation
    names = list(maps)
    gif = []
    for t in range(24):
        ang = t / 24 * 2 * math.pi
        lx, ly = 12 + 14 * math.cos(ang), 12 + 14 * math.sin(ang)
        frame = Image.new("RGBA", (5 * cell + 6 * gap, cell + 2 * gap), bg)
        for i, name in enumerate(names):
            frame.alpha_composite(big(light(*maps[name], lx, ly)), (gap + i * (cell + gap), gap))
        k = t % 6
        sl = light(sheet_color[:, k * 24:(k + 1) * 24], sheet_normal[:, k * 24:(k + 1) * 24], lx, ly)
        frame.alpha_composite(big(sl), (gap + 4 * (cell + gap), gap))
        gif.append(frame.convert("RGB"))
    gif[0].save(ROOT / "lit.gif", save_all=True, append_images=gif[1:], duration=80, loop=0)
    print(f"wrote {len(rows)} normal maps, docs/gallery.png and docs/lit.gif ({len(gif)} frames)")


if __name__ == "__main__":
    main()
