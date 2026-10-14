"""OpenGL rendering: shader programs, GPU buffers and the per-frame draw call."""
from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL import GL as gl

from . import shaders
from .camera import orthographic
from .mesh import Mesh


def _text(msg) -> str:
    return msg.decode(errors="replace") if isinstance(msg, bytes) else str(msg)


# ----------------------------------------------------------------------------
# Shader program wrapper
# ----------------------------------------------------------------------------
class ShaderProgram:
    def __init__(self, vertex_src: str, fragment_src: str):
        self.id = gl.glCreateProgram()
        for src, kind in ((vertex_src, gl.GL_VERTEX_SHADER), (fragment_src, gl.GL_FRAGMENT_SHADER)):
            sh = gl.glCreateShader(kind)
            gl.glShaderSource(sh, src)
            gl.glCompileShader(sh)
            if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
                raise RuntimeError("shader compile error:\n" + _text(gl.glGetShaderInfoLog(sh)))
            gl.glAttachShader(self.id, sh)
            gl.glDeleteShader(sh)                     # freed once the program is deleted
        gl.glLinkProgram(self.id)
        if not gl.glGetProgramiv(self.id, gl.GL_LINK_STATUS):
            raise RuntimeError("shader link error:\n" + _text(gl.glGetProgramInfoLog(self.id)))
        self._locations: dict[str, int] = {}

    def use(self) -> None:
        gl.glUseProgram(self.id)

    def _loc(self, name: str) -> int:
        if name not in self._locations:
            self._locations[name] = gl.glGetUniformLocation(self.id, name)
        return self._locations[name]

    # numpy matrices are row-major -> transpose=GL_TRUE converts to OpenGL's column-major layout
    def set_mat4(self, name: str, m) -> None:
        gl.glUniformMatrix4fv(self._loc(name), 1, gl.GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))

    def set_mat3(self, name: str, m) -> None:
        gl.glUniformMatrix3fv(self._loc(name), 1, gl.GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))

    def set_vec3(self, name: str, v) -> None:
        gl.glUniform3f(self._loc(name), float(v[0]), float(v[1]), float(v[2]))

    def set_bool(self, name: str, b: bool) -> None:
        gl.glUniform1i(self._loc(name), 1 if b else 0)


# ----------------------------------------------------------------------------
# GPU buffers
# ----------------------------------------------------------------------------
class GpuMesh:
    """One VAO holding an interleaved [x y z nx ny nz] vertex buffer and a triangle index buffer."""

    def __init__(self, mesh: Mesh):
        normals = mesh.ensure_normals()
        data = np.hstack([mesh.vertices, normals]).astype(np.float32)
        indices = mesh.faces.astype(np.uint32).ravel()
        self.vertex_count = int(len(data))
        self.index_count = int(indices.size)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        stride = 6 * 4                                  # six float32 per vertex
        gl.glEnableVertexAttribArray(0)                 # location 0 = position
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)                 # location 1 = normal
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)

    def draw(self) -> None:
        gl.glBindVertexArray(self.vao)
        if self.index_count:
            gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        else:                                           # a point cloud (no faces)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.vertex_count)
        gl.glBindVertexArray(0)

    def delete(self) -> None:
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(2, [self.vbo, self.ebo])


class LineBatch:
    """Static coloured line segments: rows of [x y z r g b], two rows per segment."""

    def __init__(self, data: np.ndarray):
        data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, 6)
        self.count = int(len(data))
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)

    def draw(self) -> None:
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.count)
        gl.glBindVertexArray(0)

    def delete(self) -> None:
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteBuffers(1, [self.vbo])


# ----------------------------------------------------------------------------
# Helper geometry (ground grid and axis gizmo)
# ----------------------------------------------------------------------------
def _nice_step(x: float) -> float:
    """Round x down to 1, 2 or 5 times a power of ten (gives readable grid spacing)."""
    e = 10.0 ** math.floor(math.log10(x))
    m = x / e
    return e * (5.0 if m >= 5 else 2.0 if m >= 2 else 1.0)


def build_grid(center, floor_y, radius, color=(0.30, 0.31, 0.34), center_color=(0.46, 0.47, 0.51)):
    step = _nice_step(max(radius, 1e-9) / 4.0)
    n = int(math.ceil(2.0 * radius / step))
    half = n * step
    cx, cz = float(center[0]), float(center[2])
    rows = []
    for i in range(-n, n + 1):
        d = i * step
        c = center_color if i == 0 else color
        rows.append([cx + d, floor_y, cz - half, *c]); rows.append([cx + d, floor_y, cz + half, *c])
        rows.append([cx - half, floor_y, cz + d, *c]); rows.append([cx + half, floor_y, cz + d, *c])
    return np.array(rows, np.float32)


def build_axes(origin, length):
    o = np.asarray(origin, np.float64)
    rows = []
    for axis, col in (((1, 0, 0), (0.90, 0.25, 0.25)),      # X red
                      ((0, 1, 0), (0.35, 0.85, 0.35)),      # Y green
                      ((0, 0, 1), (0.30, 0.50, 1.00))):     # Z blue
        rows.append([*o, *col]); rows.append([*(o + np.asarray(axis) * length), *col])
    return np.array(rows, np.float32)


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
class Renderer:
    WIRE_MODES = ("solid", "solid+wire", "wire")

    def __init__(self):
        self.mesh_prog = ShaderProgram(shaders.MESH_VS, shaders.MESH_FS)
        self.line_prog = ShaderProgram(shaders.LINE_VS, shaders.LINE_FS)
        self.gpu_mesh: GpuMesh | None = None
        self.grid: LineBatch | None = None
        self.axes: LineBatch | None = None
        self.gizmo = LineBatch(build_axes((0.0, 0.0, 0.0), 1.0))   # corner orientation gizmo
        self.wire_mode = 0
        self.flat_shading = False
        self.show_grid = True
        self.show_axes = True
        self.mesh_color = (0.78, 0.80, 0.85)
        self.wire_color = (0.10, 0.10, 0.12)
        self.background = (0.11, 0.12, 0.14)
        self.model_matrix = np.identity(4)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

    # -- scene content --------------------------------------------------------------
    def set_mesh(self, mesh: Mesh) -> None:
        self.clear_mesh()
        self.gpu_mesh = GpuMesh(mesh)
        lo, hi = mesh.bounds()
        center, r = (lo + hi) / 2.0, mesh.radius()
        floor_y = lo[1] - 0.002 * r                 # a hair below the model so the two never z-fight
        self.grid = LineBatch(build_grid(center, floor_y, r))
        self.axes = LineBatch(build_axes((center[0], floor_y, center[2]), 0.5 * r))

    def clear_mesh(self) -> None:
        for obj in (self.gpu_mesh, self.grid, self.axes):
            if obj is not None:
                obj.delete()
        self.gpu_mesh = self.grid = self.axes = None

    def cycle_wire_mode(self) -> None:
        self.wire_mode = (self.wire_mode + 1) % len(self.WIRE_MODES)

    # -- drawing -----------------------------------------------------------------------
    def draw(self, camera, fb_width: int, fb_height: int) -> None:
        gl.glViewport(0, 0, fb_width, fb_height)
        gl.glClearColor(*self.background, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        view, proj = camera.view_matrix(), camera.projection_matrix()

        if self.gpu_mesh is not None:
            self._draw_mesh(view, proj)

        self.line_prog.use()
        self.line_prog.set_mat4("u_view", view)
        self.line_prog.set_mat4("u_proj", proj)
        if self.show_grid and self.grid is not None:
            self.grid.draw()
        if self.show_axes and self.axes is not None:
            self.axes.draw()
        if self.show_axes:
            self._draw_gizmo(camera, fb_width, fb_height)

    def _draw_gizmo(self, camera, fb_width: int, fb_height: int) -> None:
        """Small XYZ gizmo in the bottom-left corner: the view's rotation without its translation."""
        size = max(64, min(fb_width, fb_height) // 7)
        gl.glViewport(10, 10, size, size)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)                  # always draw on top of the scene
        rotation_only = camera.view_matrix()
        rotation_only[:3, 3] = 0.0
        self.line_prog.set_mat4("u_view", rotation_only)
        self.line_prog.set_mat4("u_proj", orthographic(1.25, 1.0, -2.0, 2.0))
        self.gizmo.draw()
        gl.glViewport(0, 0, fb_width, fb_height)

    def _draw_mesh(self, view, proj) -> None:
        p = self.mesh_prog
        p.use()
        model_view = view @ self.model_matrix
        p.set_mat4("u_model", self.model_matrix)
        p.set_mat4("u_view", view)
        p.set_mat4("u_proj", proj)
        p.set_mat3("u_normal_matrix", np.linalg.inv(model_view[:3, :3]).T)
        p.set_bool("u_flat", self.flat_shading)

        if self.gpu_mesh.index_count == 0:              # point cloud
            p.set_bool("u_unlit", True)
            p.set_vec3("u_color", self.mesh_color)
            gl.glPointSize(3.0)
            self.gpu_mesh.draw()
            return

        if self.wire_mode in (0, 1):                     # filled surface
            p.set_bool("u_unlit", False)
            p.set_vec3("u_color", self.mesh_color)
            if self.wire_mode == 1:                      # push the fill back so the lines win the depth test
                gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glPolygonOffset(1.0, 1.0)
            self.gpu_mesh.draw()
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        if self.wire_mode in (1, 2):                     # edges
            p.set_bool("u_unlit", True)
            p.set_vec3("u_color", self.wire_color if self.wire_mode == 1 else self.mesh_color)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            self.gpu_mesh.draw()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def read_pixels(self, fb_width: int, fb_height: int) -> np.ndarray:
        """Return the current back buffer as an (H, W, 3) uint8 RGB image, top row first."""
        buf = gl.glReadPixels(0, 0, fb_width, fb_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        return np.frombuffer(buf, dtype=np.uint8).reshape(fb_height, fb_width, 3)[::-1].copy()
