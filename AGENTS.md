<!-- Codex: same content as skills/pixel-normals/SKILL.md -->

# pixel-normals

Turns a sprite into a normal map (`<name>_n.png`) with the same size and alpha. Requires `pip install numpy pillow`.

## Commands (run from the repo root)
- One sprite: `python pixel_normals.py sprite.png -o sprite_n.png`
- A folder: `python pixel_normals.py sprites/*.png --output-dir normals`
- A sheet with uniform cells: `python pixel_normals.py sheet.png --frame-size 32x32 -o sheet_n.png`
- An animation: `python pixel_normals.py walk.gif -o walk_n.png` (also writes `walk_n.color.png` and `walk_n.frames.json`)
- Stronger relief: add `--normal-strength 2.5`; Unity/Unreal-style green channel: add `--normal-format directx`.
- See the effect: `python demo.py` writes `docs/gallery.png` and `docs/lit.gif`.

## Notes for the agent
- Existing outputs are never overwritten without `--force`, and an input is never overwritten.
- Normal PNGs are data textures: tell the user to disable sRGB for them in their engine.
- For a sheet, ask the user for the cell size if it is not obvious; without `--frame-size`, touching sprites are treated as one shape.
