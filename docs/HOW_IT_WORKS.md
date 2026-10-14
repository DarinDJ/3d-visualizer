# How the viewer works

This document explains the implementation in the order the data flows through it, so that every
line of the code can be understood and defended. The three graded components (camera module, object
loader, mouse manipulation) each have their own section; rendering and testing follow.

```
 file (.obj/.stl/.ply)            MouseControls              OrbitCamera
        |                               |                        |
   loader.load()                 pixel deltas  --->   target / distance / yaw / pitch
        |                                                        |
      Mesh  (vertices, faces, normals)                 view V, projection P
        |                                                        |
   GpuMesh (VBO + EBO + VAO)  ---->  vertex shader:  clip = P * V * M * position
                                            |
                                      rasteriser (GPU)
                                            |
                                     fragment shader: Blinn-Phong lighting  --->  pixels
```

---

## 1. Camera module (`viewer/camera.py`)

### 1.1 Coordinate spaces

A vertex travels through four coordinate spaces:

| Space | Origin / axes | Produced by |
|-------|---------------|-------------|
| **world** | the file's coordinates, +Y up, right-handed | the loader |
| **eye / camera** | camera at the origin, looking down **-Z**, +X right, +Y up | view matrix `V` |
| **clip** | homogeneous `(x, y, z, w)`; visible if `-w <= x,y,z <= w` | projection matrix `P` |
| **NDC / pixels** | cube `[-1, 1]^3` after dividing by `w`; then the viewport transform | the GPU |

All matrices are 4x4 numpy arrays in normal mathematical layout so that `clip = P @ V @ [x, y, z, 1]`.
OpenGL stores matrices column-major, which is why `ShaderProgram.set_mat4` uploads with
`transpose=GL_TRUE`. Keeping the maths in the natural layout avoids a whole class of
transposition bugs.

### 1.2 `look_at`: the view matrix (extrinsics)

Given the eye position `e`, the target `t` and an up hint `up`:

```
f = normalize(t - e)            forward
s = normalize(f x up)           right  ("side")
u = s x f                       true up (orthogonal to both)

V = | s.x  s.y  s.z  -s.e |
    | u.x  u.y  u.z  -u.e |
    |-f.x -f.y -f.z   f.e |
    |  0    0    0     1  |
```

Why this works: the camera's orientation is the rotation whose *columns* are `s`, `u`, `-f`. The view
matrix must undo that rotation, and the inverse of a rotation is its transpose, so `s`, `u`, `-f`
become the *rows*. The last column translates the eye to the origin (`-R e`). The tests verify both
facts directly: `V @ eye = origin` and `V @ target` lands on the negative Z axis.

### 1.3 `perspective`: the projection matrix (intrinsics)

With vertical field of view `fov`, aspect ratio `a = width/height`, near plane `n` and far plane `f`:

```
c = 1 / tan(fov / 2)

P = | c/a   0      0          0      |
    |  0    c      0          0      |
    |  0    0  (f+n)/(n-f)  2fn/(n-f)|
    |  0    0     -1          0      |
```

* The last row copies `-z_eye` into `w_clip`. Dividing by `w` afterwards is the perspective divide
  (far objects become small): `x_ndc = c/a * x / -z`.
* Rows 1 and 2 scale so that the frustum edges map to `x_ndc, y_ndc = +-1`.
* Row 3 maps `z = -n` to `z_ndc = -1` and `z = -f` to `+1` (tested), which is what the depth test uses.

`orthographic` is the same idea without the divide (`w = 1`), so parallel lines stay parallel.
Switching with `P` keeps the framing because the orthographic half-height is set to
`distance * tan(fov/2)`, the height the perspective view sees at the target's depth.

### 1.4 The pinhole link: `perspective_from_intrinsics`

Computer-vision pipelines describe a camera by the matrix `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
and the pose `[R | t]`. The viewer's `look_at` *is* `[R | t]` (`OrbitCamera.extrinsics()` returns it) and
the OpenGL projection is a re-parameterisation of `K` with near/far planes. Starting from the pinhole
equations `u = fx * x/(-z) + cx` and `v = cy - fy * y/(-z)` (image `v` grows downwards) and the pixel
to NDC mapping `x_ndc = 2u/width - 1`, `y_ndc = 1 - 2v/height`, solving for the clip coordinates gives

```
P = | 2fx/w   0     (w-2cx)/w     0      |
    |  0    2fy/h   (2cy-h)/h     0      |
    |  0      0    (f+n)/(n-f)  2fn/(n-f)|
    |  0      0       -1          0      |
```

A test checks that with `fx = fy = (h/2)/tan(fov/2)` and the principal point at the image centre this is
exactly `perspective()`, and that `project()` agrees with the pinhole formula to six decimals. This is
the bridge between the viewer and the "camera models" part of the report.

### 1.5 `OrbitCamera`

An orbit (turntable) camera stores four things: the `target` it looks at, the `distance` to it, and two
angles. The eye position is spherical coordinates around the target:

```
position = target + distance * ( cos(pitch) sin(yaw),  sin(pitch),  cos(pitch) cos(yaw) )
```

* `yaw` rotates around the world Y axis, `pitch` is the elevation above the XZ plane. The pitch is
  clamped to +-89 degrees so the forward vector never becomes parallel to the up vector (which would
  make `look_at`'s cross product zero).
* `view_matrix()` is just `look_at(position, target)`.
* `fit(lo, hi)` frames a bounding box: the target is the box centre, the scene radius is half the box
  diagonal, and the distance is `margin * r / sin(fov/2)`, the distance at which a sphere of radius
  `r` exactly touches the frustum edges (in a portrait window the horizontal fov is the limiting one).
  The result is stored as the "home" view used by `reset()`.
* `near_far()` picks the clip planes from the current distance and scene radius, so depth-buffer
  precision is spent on the model instead of on empty space.

---

## 2. Object loader (`viewer/loader.py`, `viewer/mesh.py`)

### 2.1 The `Mesh` container

Everything is reduced to the same *indexed triangle list*:

```
vertices : (N, 3) float32    positions
faces    : (M, 3) int32      three indices per triangle, counter-clockwise seen from outside
normals  : (N, 3) float32    optional, one per vertex
```

Indexing matters because the GPU reuses each vertex for every triangle that touches it
(`glDrawElements`), and because smooth shading needs to know which triangles share a vertex.

### 2.2 Normals

The face normal is the cross product of two edges, `n = (v1 - v0) x (v2 - v0)`. Its length is twice
the triangle area. `compute_vertex_normals` *adds* these un-normalised face normals onto each of the
three vertices (so large faces count more, which is the standard area weighting) and normalises at
the end. `np.bincount(indices, weights=...)` performs the scatter-add for all faces at once, without
a Python loop. Generated models carry analytic normals (`vn` records in the OBJ) and the generator
checks that the triangle winding agrees with them.

### 2.3 OBJ

A text format: `v x y z` for positions, `vn` for normals, `vt` for texture coordinates and `f` for
faces. Three things need care:

1. **Indices are 1-based**, and **negative indices** are relative to the most recently defined
   vertex (`-1` = last one). The loader resolves both.
2. **Faces may be polygons.** A polygon `p0 p1 p2 p3 ...` is split into a fan
   `(p0,p1,p2), (p0,p2,p3), ...`, which is correct for convex polygons.
3. **Positions and normals have separate index spaces** (`f 1//3 2//3 7//5`), but a GPU vertex is one
   index for all attributes. The loader therefore creates one output vertex for every distinct
   `(position index, normal index)` pair with `np.unique(pairs, axis=0, return_inverse=True)`;
   the `inverse` array is the new face list. A cube written with per-face normals correctly
   becomes 24 vertices, while a smooth sphere stays at its original count.

If any face corner lacks a normal, the file's normals are dropped and smooth normals are computed.

### 2.4 STL

Binary STL is an 80-byte header, a `uint32` triangle count and then 50-byte records
`(normal[3], v0[3], v1[3], v2[3], uint16 attribute)`. One numpy structured `dtype` describes the record
and `np.frombuffer` reads the whole file in one call. The file is binary if its size equals
`84 + 50 * count`; otherwise it is parsed as ASCII with a regular expression on the `vertex` lines
(an ASCII file also starts with `solid`, but so do some binary files, hence the size test).

STL has no shared vertices, so `_weld` merges identical corner positions with `np.unique(axis=0)`;
the inverse mapping again gives the index triples. Welding is what makes smooth shading possible.

### 2.5 PLY

PLY is header-driven: `element vertex N`, then one `property type name` line per attribute, then
`element face M` with a `property list uchar int vertex_indices`. The parser reads the header into
`(name, count, properties)` triples and then, for binary files, builds a numpy `dtype` per element
(including the endianness) so fixed-size elements are read in one `np.frombuffer`. Face lists are
variable length in principle; the fast path tries "every face is a triangle" (the common case) and
falls back to a record-by-record reader that fan-triangulates polygons. ASCII files are parsed the
same way from text tokens. Files without faces are loaded as point clouds and drawn as points.

### 2.6 Writers and the converter

`save_obj`, `save_stl` and `save_ply` write the same structures back out, so `convert.py` can go from
any supported format to any other. The round-trip test writes a cube in all five variants (OBJ,
binary and ASCII STL, ASCII and binary PLY), reads it back and checks vertex and face counts, bounds
and total surface area.

---

## 3. Mouse manipulation (`viewer/controls.py`)

GLFW only reports raw events: button pressed/released, cursor at `(x, y)`, wheel moved. `MouseControls`
keeps the state machine (which action is active, where the cursor was last) and translates pixel
deltas into camera operations. Because it has no dependency on GLFW it is tested by feeding it
synthetic events.

| Event | Action | Conversion |
|-------|--------|------------|
| left button + move | `camera.orbit(-dx * s, dy * s)` | `s = 2π / window_height`: dragging across the whole window height is one full turn |
| right/middle button (or Shift + left) + move | `camera.pan(dx, dy, height)` | world units per pixel at the target's depth = `2 * distance * tan(fov/2) / height` |
| wheel `k` notches | `camera.dolly(0.9 ** k)` | multiplicative, so zooming feels uniform at every scale and never crosses the target |

Sign conventions: dragging right makes the model turn to the right (the camera orbits to the left),
and dragging down raises the camera so you look down onto the model. These match Blender,
three.js and most CAD viewers.

Panning is pixel-exact: the test projects the target to the screen, pans by `(100, -40)` pixels and
checks that the same world point now appears exactly 100 px to the right and 40 px higher. The
model therefore sticks to the cursor at any zoom level.

HiDPI note: mouse coordinates arrive in *screen* units while the framebuffer may be larger (Retina
displays). `app.py` passes the window size to the controls and the framebuffer size to the renderer.

---

## 4. Rendering (`viewer/renderer.py`, `viewer/shaders.py`)

* **Buffers.** `GpuMesh` interleaves position and normal (`x y z nx ny nz`) in one vertex buffer object,
  stores the faces in an element buffer, and records the attribute layout in a vertex array object.
  Drawing is a single `glDrawElements(GL_TRIANGLES, ...)`.
* **Vertex shader.** `gl_Position = P * V * M * position`. The normal is transformed by the *inverse
  transpose* of the upper 3x3 of `V * M` (the "normal matrix"), which keeps normals perpendicular to
  surfaces even under non-uniform scaling.
* **Fragment shader.** Blinn-Phong: ambient + diffuse (`max(n.l, 0)`) + specular (`(n.h)^48` with the
  half vector `h = normalize(l + v)`). The key light direction is fixed in eye space, so it behaves
  like a headlight: the model is always lit from the viewer's side, whichever way it is turned.
  Back faces are lit two-sided so open meshes do not turn black.
* **Flat shading** needs no extra data: `cross(dFdx(pos), dFdy(pos))` of the eye-space position gives
  the facet plane from the rasteriser's own derivatives.
* **Wireframe.** Mode 1 draws the filled mesh with `glPolygonOffset` pushing it slightly back, then
  draws the same mesh with `glPolygonMode(GL_LINE)` in a dark colour so the edges win the depth test.
* **Helpers.** A ground grid (1-2-5 spacing chosen from the model radius, just below the model's lowest
  point) and axis lines are drawn with a second, unlit shader. A corner gizmo re-draws the axes in a
  small viewport using the view's rotation only, so it always shows the current orientation.
* **Screenshots.** `glReadPixels` reads the back buffer before the swap; rows are flipped because
  OpenGL's origin is bottom-left.

---

## 5. Tests (`tests/test_core.py`)

| Group | What is proven |
|-------|----------------|
| `MatrixTests` | `look_at` sends the eye to the origin and the target onto -Z; the rotation is orthonormal; `perspective` maps the near/far planes to NDC -1/+1; the intrinsics-based projection equals the fov-based one; `project` equals the pinhole formula |
| `OrbitCameraTests` | all eight corners of a fitted box are inside the frustum; panning is pixel-exact; orbit keeps the distance and clamps the pitch; dolly is clamped; reset restores the home view; orthographic has no perspective divide |
| `ControlsTests` | left-drag rotates without moving the target; a window-height drag is one full turn; right-drag and Shift-drag pan; wheel zooms in/out symmetrically; moving without a button does nothing |
| `MeshTests` | vertex normals are unit length and point outward; face normals point outward; bounds/centre/radius; bad indices are rejected |
| `LoaderTests` | OBJ quads, negative indices and `v//vn` syntax; normals computed when absent; round trip through OBJ / STL (bin+ascii) / PLY (ascii+bin) preserves counts, bounds and surface area |

The rendering path is checked by `python main.py model --screenshot out.png --hidden`, which creates
a hidden window, draws one frame and saves it.

---

## 6. Ideas for extending the viewer

* **Arcball rotation**: map the cursor to a virtual sphere and rotate with quaternions, which removes
  the pole clamp of the turntable camera.
* **Picking**: invert `P * V`, cast the pixel's ray into the scene and intersect it with the triangles.
* **glTF / USD loading** and PBR materials (see the converters section of the report).
* **Gaussian-splat `.ply` files**: the PLY parser already reads arbitrary properties; a splat renderer
  would sort the points by depth and draw screen-aligned quads with the Gaussian footprint.
