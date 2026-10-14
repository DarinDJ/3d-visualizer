# 3D Model Viewer, written from scratch

A small 3-D visualiser in Python for the *"Make your own visualizer"* homework. It has the three
required parts, each in its own module, plus a short report on recent 3-D technology:

| Requirement | Where | What it does |
|-------------|-------|--------------|
| **Camera module** | `viewer/camera.py` | look-at / perspective / orthographic matrices written by hand, an orbit camera (target, distance, yaw, pitch), and the link to the pinhole model (`K`, `[R\|t]`) |
| **Object loader** | `viewer/loader.py`, `viewer/mesh.py` | OBJ, STL (binary + ASCII) and PLY (ASCII + binary) readers and writers, polygon triangulation, vertex welding, normal computation |
| **Mouse manipulation** | `viewer/controls.py` | left-drag rotate, right/middle-drag pan, wheel zoom, with pixel-exact panning |
| Rendering | `viewer/renderer.py`, `viewer/shaders.py` | OpenGL 3.3 core, GLSL Blinn-Phong shading, flat/smooth, wireframe, grid, axes gizmo |
| Window / events | `viewer/app.py`, `main.py` | GLFW window, keyboard shortcuts, drag-and-drop loading, screenshots |
| Converter | `convert.py` | OBJ / STL / PLY in any direction, built on the same loader |
| Report | `report/report.md`, `report/report.pdf` | one page on 3-D representations, converters and camera models |

No viewer or geometry library is used (no Open3D, trimesh, pyrender, three.js): the matrices, parsers,
interaction and shaders are all in this repository. The only dependencies are `numpy`, `PyOpenGL`
(bindings only) and `glfw` (window + input).

![viewer screenshot](docs/screenshot.png)

## Quick start

```bash
pip install -r requirements.txt
python main.py                      # opens models/teapot.obj
python main.py models/bunny.obj     # any .obj / .stl / .ply
python main.py model.stl --ortho    # start in orthographic projection
```

Requires Python 3.9+ and a graphics driver with OpenGL 3.3 (any desktop or laptop from the last decade,
including macOS, where the forward-compatible core profile is requested automatically).
You can also drag a mesh file onto the window to load it.

## Controls

| Input | Action |
|-------|--------|
| left drag | rotate (orbit around the model) |
| right or middle drag, or Shift + left drag | pan |
| mouse wheel | zoom (dolly towards the target) |
| `R` / `F` | reset the view / re-fit the model |
| `W` | cycle solid, solid + wireframe, wireframe |
| `S` | smooth / flat shading |
| `P` | perspective / orthographic |
| `G` / `A` | toggle the ground grid / axes and gizmo |
| `C` | save a screenshot to `screenshots/` |
| `H` | print this help in the terminal |
| `Esc` or `Q` | quit |

## How it works

The complete walk-through, with the derivations of every matrix and parser, is in
[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md). In one paragraph:

1. **Loader.** A file is parsed into a `Mesh`: an `(N,3)` array of vertex positions, an `(M,3)` array
   of triangle indices and optional per-vertex normals. OBJ polygons are fan-triangulated and the
   separate position/normal index spaces are merged into one GPU-friendly index. STL triangles are
   welded into shared vertices. PLY is parsed from its header with numpy structured dtypes. Missing
   normals are computed as area-weighted averages of the adjacent face normals.
2. **Camera.** The orbit camera stores a target, a distance, a yaw and a pitch, turns them into an eye
   position, and builds the view matrix with `look_at` and the projection with `perspective` (or
   `orthographic`). `fit()` chooses the distance so that the model's bounding sphere fills the frustum:
   `distance = margin * radius / sin(fov / 2)`.
3. **Mouse.** GLFW events go to `MouseControls`, which converts pixel deltas into camera operations:
   one window height of dragging is one full turn; panning converts pixels into world units at the
   target's depth (`2 * distance * tan(fov/2) / height`), so the model follows the cursor exactly;
   the wheel multiplies the distance by `0.9^steps`.
4. **Rendering.** Positions and normals are interleaved in one vertex buffer; the vertex shader applies
   `P * V * M` and the fragment shader computes Blinn-Phong lighting from a light fixed to the camera.
   Flat shading uses screen-space derivatives (`dFdx`, `dFdy`) of the eye-space position.

## Converter

```bash
python convert.py models/teapot.obj teapot.stl          # binary STL
python convert.py teapot.stl teapot.ply --binary        # binary PLY with computed normals
python convert.py teapot.ply teapot.obj --no-normals
python convert.py some_mesh.ply --info                  # statistics only
```

## Tests

```bash
python -m unittest discover tests
```

23 tests cover the matrix maths (look-at, perspective, intrinsics equivalence, pixel projection), the
orbit camera (fit, pan accuracy, clamps, reset), the mouse controls, normal computation and a round
trip of a cube through every supported file format. They run without an OpenGL context.
`python main.py model.obj --screenshot out.png --hidden` renders one frame off-screen, which is how
the screenshots in `docs/` were produced.

## Project layout

```
main.py                 entry point (argument parsing, default model)
convert.py              format converter CLI
viewer/
  camera.py             camera module: matrices + OrbitCamera
  controls.py           mouse manipulation
  loader.py             object loader: OBJ / STL / PLY readers and writers
  mesh.py               Mesh container, bounds, normals
  renderer.py           OpenGL buffers, shaders, grid, axes, gizmo, screenshots
  shaders.py            GLSL sources
  app.py                GLFW window, callbacks, main loop
models/                 sample models (teapot, bunny, cube, sphere, torus) + generator
tests/test_core.py      unit tests
docs/HOW_IT_WORKS.md    detailed explanation of the implementation
report/                 one-page report (Markdown + PDF)
```

## Limitations and possible extensions

- Materials, textures and vertex colours are ignored; the mesh is drawn in one colour.
- Polygon faces are fan-triangulated, which assumes convex polygons (true for almost all files).
- The parsers are pure Python/numpy: the 70k-triangle bunny loads in about half a second,
  so million-triangle files take several seconds.
- Natural next steps: a trackball (arcball) rotation mode, picking with the inverse projection,
  glTF loading, and a point-splat renderer for Gaussian-splat `.ply` files.
