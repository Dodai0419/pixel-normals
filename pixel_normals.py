"""Normal maps for pixel-art sprites, sprite sheets, GIFs and APNGs.

A NumPy port of Hugo-Dz/spritefusion-sprite-normals (MIT): same pipeline, same
defaults, same output files, in one file you can run with `pip install numpy pillow`.
"""
import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

F = np.float32
MAX_PIXELS = 4_194_304
MAX_FRAMES = 256
MAX_INPUT_BYTES = 64 * 1024 * 1024
DEFAULTS = dict(alpha_threshold=0.03, shape_radius=4.0, shape_strength=1.15, shape_softness=1.15,
                detail_strength=6.0, detail_scale=0.6, normal_strength=1.5)
RANGES = dict(alpha_threshold=(0.0, 0.99), shape_radius=(0.5, 512.0), shape_strength=(0.0, 8.0),
              shape_softness=(0.0, 8.0), detail_strength=(0.0, 8.0), detail_scale=(0.6, 8.0),
              normal_strength=(0.0, 8.0))


def settings(**kw):
    s = dict(DEFAULTS, directx=False, invert_x=False, invert_y=False, frame_size=None)
    s.update({k: v for k, v in kw.items() if v is not None})
    for name, (lo, hi) in RANGES.items():
        v = float(s[name])
        if not math.isfinite(v) or v < lo or v > hi:
            raise ValueError(f"{name.replace('_', '-')} must be finite and between {lo} and {hi}")
        s[name] = F(v)
    if s["frame_size"] and min(s["frame_size"]) <= 0:
        raise ValueError("frame-size dimensions must be positive")
    return s


def _linear(v):
    v = v / F(255)
    return np.where(v <= F(0.04045), v / F(12.92), ((v + F(0.055)) / F(1.055)) ** F(2.4)).astype(F)


LINEAR = _linear(np.arange(256, dtype=F))


def lightness(rgb):
    """Oklab L of 8-bit sRGB, shape (..., 3) -> (...)."""
    r, g, b = (LINEAR[rgb[..., i]] for i in range(3))
    l = np.cbrt(F(0.41222146) * r + F(0.53633255) * g + F(0.051445995) * b)
    m = np.cbrt(F(0.2119035) * r + F(0.6806995) * g + F(0.10739696) * b)
    s = np.cbrt(F(0.08830246) * r + F(0.28171885) * g + F(0.6299787) * b)
    return F(0.21045426) * l + F(0.7936178) * m - F(0.004072047) * s


def gaussian(a, sigma):
    """Separable Gaussian, zero outside the field; taps summed in kernel order like upstream."""
    if sigma < 0.05:
        return a.copy()
    r = int(math.ceil(float(sigma * F(3))))
    k = np.exp((F(-0.5) * (np.arange(-r, r + 1, dtype=F) / sigma) ** 2).astype(np.float64)).astype(F)
    k = k / _f32_sum(k)
    for axis in (1, 0):
        out = np.zeros_like(a)
        n = a.shape[axis]
        for off, w in zip(range(-r, r + 1), k):
            s = min(abs(off), n)
            src = [slice(None)] * 2
            dst = [slice(None)] * 2
            src[axis], dst[axis] = (slice(0, n - s), slice(s, n)) if off < 0 else (slice(s, n), slice(0, n - s))
            out[tuple(dst)] += a[tuple(src)] * w
        a = out
    return a


def _f32_sum(k):
    t = F(0)
    for v in k:  # sequential f32 sum, as Rust's Iterator::sum
        t = F(t + v)
    return t


def masked_gaussian(a, weight, sigma):
    return gaussian(a * weight, sigma) / np.maximum(gaussian(weight, sigma), F(1e-8))


def edt(mask):
    """Exact Euclidean distance to the nearest False pixel (padding guarantees one exists)."""
    h, w = mask.shape
    xs = np.arange(w, dtype=np.int32)
    left = xs - np.maximum.accumulate(np.where(mask, -w, xs), axis=1)  # distance to background along the row
    right = np.minimum.accumulate(np.where(mask, 2 * w, xs)[:, ::-1], axis=1)[:, ::-1] - xs
    g2 = np.minimum(left, right) ** 2
    out = g2.copy()
    cols = np.arange(w)
    k = 1
    while k < h and len(cols):  # rows farther than the current best distance cannot win
        o, g = out[:, cols], g2[:, cols]
        np.minimum(o[k:], g[:-k] + k * k, out=o[k:])
        np.minimum(o[:-k], g[k:] + k * k, out=o[:-k])
        out[:, cols] = o
        k += 1
        cols = cols[o.max(0) > k * k]
    return np.sqrt(out.astype(F))


def relief(island, s):
    """Normal map for one opaque island (RGBA uint8 array)."""
    pad = int(math.ceil(3 * float(max(s["shape_softness"], s["detail_scale"] * F(2))))) + 3
    h, w = island.shape[:2]
    alpha = island[..., 3].astype(F) / F(255)
    inside = alpha > s["alpha_threshold"]
    P = lambda a: np.pad(a, pad)
    mask = P(inside)
    opacity = P(np.where(inside, alpha, F(0)).astype(F))
    light = np.where(inside, lightness(island[..., :3]), F(0)).astype(F)
    lo, hi = light[inside].min(), light[inside].max()
    light = P(np.where(inside, light - (lo + hi) * F(0.5), F(0)).astype(F))
    reach = s["shape_radius"]
    inset = np.maximum(edt(mask) - F(0.5), F(0))
    shape = (s["shape_strength"] * reach * np.log1p(inset / reach)).astype(F)
    combined = gaussian(shape, s["shape_softness"])
    fine = masked_gaussian(light, opacity, s["detail_scale"])
    broad = masked_gaussian(light, opacity, s["detail_scale"] * F(2))
    combined = combined + s["detail_strength"] * (F(0.7) * fine + F(0.3) * broad)
    c = combined
    dx = (c[:-2, 2:] + F(2) * c[1:-1, 2:] + c[2:, 2:] - c[:-2, :-2] - F(2) * c[1:-1, :-2] - c[2:, :-2]) * F(0.125)
    dy = (c[2:, :-2] + F(2) * c[2:, 1:-1] + c[2:, 2:] - c[:-2, :-2] - F(2) * c[:-2, 1:-1] - c[:-2, 2:]) * F(0.125)
    core = (slice(pad - 1, pad - 1 + h), slice(pad - 1, pad - 1 + w))
    x = -dx[core] * s["normal_strength"]
    y = dy[core] * s["normal_strength"]
    if s["directx"]:
        y = -y
    if s["invert_x"]:
        x = -x
    if s["invert_y"]:
        y = -y
    x64, y64 = x.astype(np.float64), y.astype(np.float64)  # f64 products are exact, so this matches Rust's f32 mul_add
    length = np.sqrt((x64 * x64 + (y64 * y64 + 1).astype(F)).astype(F))
    rgb = np.stack([x / length, y / length, F(1) / length], -1)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.floor((rgb * F(0.5) + F(0.5)) * F(255) + F(0.5)).clip(0, 255)
    out[..., 3] = island[..., 3]
    out[~inside, :3] = (128, 128, 255)
    return out


def components(mask):
    """Eight-connected islands, so diagonal pixel-art strokes stay together."""
    # Union-find over horizontal runs: Python touches runs, NumPy touches pixels.
    edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    sy, sx = np.nonzero(edges == 1)
    ex = np.nonzero(edges == -1)[1]  # run [sx, ex) on row sy, both in row-major order
    parent = list(range(len(sy)))

    def find(i):
        while parent[i] != i:
            parent[i] = i = parent[parent[i]]
        return i

    prev, cur, row = [], [], -2
    for i in range(len(sy)):
        if sy[i] != row:
            prev, cur, row = (cur if sy[i] == row + 1 else []), [], sy[i]
        cur.append(i)
        for j in prev:  # 8-connected: runs touch if they overlap after widening by one pixel
            if sx[j] <= ex[i] and ex[j] >= sx[i]:
                parent[find(i)] = find(j)
    labels = np.zeros(mask.shape, np.int64)
    roots = np.array([find(i) for i in range(len(sy))], np.int64)
    for y, a, b, r in zip(sy, sx, ex, roots):
        labels[y, a:b] = r + 1
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    keys, starts = np.unique(flat[order], return_index=True)
    for k, a, b in zip(keys, starts, list(starts[1:]) + [len(order)]):
        if k:
            idx = order[a:b]
            yield np.stack([idx // mask.shape[1], idx % mask.shape[1]], 1)


def generate_normal(rgba, s):
    """RGBA uint8 (h, w, 4) -> normal map with the same size and alpha."""
    h, w = rgba.shape[:2]
    pad = int(math.ceil(3 * float(max(s["shape_softness"], s["detail_scale"] * F(2))))) + 3
    if not h or not w or (w + 2 * pad) * (h + 2 * pad) > MAX_PIXELS:
        raise ValueError(f"image is empty or exceeds the {MAX_PIXELS}-pixel processing limit")
    fw, fh = s["frame_size"] or (w, h)
    if w % fw or h % fh:
        raise ValueError("image dimensions must be exact multiples of frame-size")
    result = np.empty_like(rgba)
    result[..., :3] = (128, 128, 255)
    result[..., 3] = rgba[..., 3]
    for y0 in range(0, h, fh):
        for x0 in range(0, w, fw):
            frame = rgba[y0:y0 + fh, x0:x0 + fw]
            for part in components(frame[..., 3].astype(F) / F(255) > s["alpha_threshold"]):
                (ya, xa), (yb, xb) = part.min(0), part.max(0) + 1
                island = np.zeros((yb - ya, xb - xa, 4), np.uint8)
                py, px = part[:, 0], part[:, 1]
                island[py - ya, px - xa] = frame[py, px]
                result[y0 + py, x0 + px] = relief(island, s)[py - ya, px - xa]
    return result


def decode(path):
    """PNG -> (rgba, None); GIF/APNG -> (atlas, animation metadata)."""
    if Path(path).stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"{path} exceeds the 64 MiB input limit")
    with Image.open(path) as im:
        return _decode(im)


def _decode(im):
    if im.format not in ("PNG", "GIF"):
        raise ValueError("use PNG, a PNG sprite sheet, GIF, or APNG")
    w, h = im.size
    if not w or not h or w * h > MAX_PIXELS:
        raise ValueError("image exceeds the input, frame-count, or decoded-pixel limit")
    if im.format == "PNG" and not getattr(im, "is_animated", False):
        return np.asarray(im.convert("RGBA")), None
    frames, delays = [], []
    for fr in ImageSequence.Iterator(im):
        if len(frames) >= MAX_FRAMES or (len(frames) + 1) * w * h > MAX_PIXELS:
            raise ValueError("image exceeds the input, frame-count, or decoded-pixel limit")
        frames.append(np.asarray(fr.convert("RGBA")))
        ms = Fraction(fr.info.get("duration", 0)).limit_denominator(1000)
        delays.append([ms.numerator, ms.denominator])
    cols = math.ceil(math.sqrt(len(frames)))
    rows = -(-len(frames) // cols)
    atlas = np.zeros((rows * h, cols * w, 4), np.uint8)
    for i, fr in enumerate(frames):
        atlas[i // cols * h:(i // cols + 1) * h, i % cols * w:(i % cols + 1) * w] = fr
    return atlas, dict(frame_width=w, frame_height=h, columns=cols, frame_count=len(frames), delays_ms=delays)


def process(path, s):
    """Returns (normal, color, metadata) exactly as the upstream CLI writes them."""
    rgba, anim = decode(path)
    if anim:
        if s["frame_size"] and tuple(s["frame_size"]) != (anim["frame_width"], anim["frame_height"]):
            raise ValueError("animation frame dimensions come from the file")
        s = dict(s, frame_size=(anim["frame_width"], anim["frame_height"]))
    normal = generate_normal(rgba, s)
    if not anim and s["frame_size"]:
        fw, fh = s["frame_size"]
        cols = rgba.shape[1] // fw
        n = cols * (rgba.shape[0] // fh)
        anim = dict(frame_width=fw, frame_height=fh, columns=cols, frame_count=n, delays_ms=[[100, 1]] * n)
    meta = dict(width=normal.shape[1], height=normal.shape[0], animation=anim)
    return normal, rgba, meta


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate normal maps from PNG sprites, sprite sheets, and GIF/APNG animations.")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, help="output normal PNG (single input only)")
    p.add_argument("--output-dir", type=Path, help="write <stem>_n.png here (default: beside each input)")
    p.add_argument("--force", action="store_true", help="replace existing outputs")
    p.add_argument("--frame-size", help="isolate PNG sheet cells, e.g. 32x32")
    for name in DEFAULTS:
        p.add_argument("--" + name.replace("_", "-"), type=float, help=f"default {DEFAULTS[name]}")
    p.add_argument("--normal-format", choices=["opengl", "directx"], default="opengl")
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    a = p.parse_args(argv)
    if a.output and len(a.inputs) != 1:
        p.error("--output requires exactly one input; use --output-dir for a batch")
    if a.output and a.output_dir:
        p.error("--output conflicts with --output-dir")
    fs = None
    if a.frame_size:
        try:
            fs = tuple(int(v) for v in a.frame_size.split("x"))
            assert len(fs) == 2 and min(fs) > 0
        except (ValueError, AssertionError):
            p.error("--frame-size expects positive WIDTHxHEIGHT, for example 32x32")
    try:
        s = settings(frame_size=fs, directx=a.normal_format == "directx", invert_x=a.invert_x,
                     invert_y=a.invert_y, **{k: getattr(a, k) for k in DEFAULTS})
    except ValueError as e:
        p.error(str(e))
    sources = {i.resolve() for i in a.inputs}
    jobs, taken = [], set()
    for inp in a.inputs:
        if not inp.is_file():
            p.error(f"cannot find input {inp}")
        out = a.output or (a.output_dir or inp.parent) / f"{inp.stem}_n.png"
        if out.suffix.lower() != ".png":
            p.error("normal output must have a .png extension")
        with Image.open(inp) as im:
            animated = fs or im.format == "GIF" or getattr(im, "is_animated", False)
        paths = [out] + ([out.with_suffix(".color.png"), out.with_suffix(".frames.json")] if animated else [])
        for path in paths:
            r = path.resolve()
            if r in sources:
                p.error(f"refusing to overwrite input {path}")
            if r in taken:
                p.error(f"duplicate output name: {path}")
            taken.add(r)
            if path.exists() and not a.force:
                p.error(f"output already exists: {path}; use --force to replace outputs")
        jobs.append((inp, out))
    for inp, out in jobs:
        try:
            normal, color, meta = process(inp, s)
        except (ValueError, OSError) as e:
            sys.exit(f"error: processing {inp}: {e}")
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normal, "RGBA").save(out)
        if meta["animation"]:
            Image.fromarray(color, "RGBA").save(out.with_suffix(".color.png"))
            out.with_suffix(".frames.json").write_text(json.dumps(meta, indent=2))
        print(f"{inp} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
