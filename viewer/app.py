"""Window, event loop and keyboard/mouse plumbing (GLFW)."""
from __future__ import annotations

import os
import sys
import time

import glfw

from . import loader
from .camera import OrbitCamera
from .controls import MouseControls
from .renderer import Renderer

HELP = """
Mouse
  left drag                rotate (orbit around the model)
  right / middle drag      pan
  shift + left drag        pan
  wheel                    zoom (dolly)
  drag & drop a file       load it (.obj .stl .ply)
Keys
  R  reset view            F  re-fit the model in the view
  W  solid / solid+wire / wireframe
  S  smooth / flat shading
  P  perspective / orthographic
  G  grid on/off           A  axes on/off
  C  save a screenshot     H  this help
  Esc or Q  quit
"""


def _save_image(path: str, rgb) -> str:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return path
    except ImportError:                                   # Pillow is optional: fall back to PPM
        path = os.path.splitext(path)[0] + ".ppm"
        h, w = rgb.shape[:2]
        with open(path, "wb") as fh:
            fh.write(b"P6 %d %d 255\n" % (w, h))
            fh.write(rgb.tobytes())
        return path


class ViewerApp:
    def __init__(self, width=1024, height=768, title="3D Viewer", visible=True):
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialised")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)      # needed on macOS
        glfw.window_hint(glfw.SAMPLES, 4)                             # 4x MSAA
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create an OpenGL 3.3 window (update your graphics driver?)")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)                                         # v-sync

        self.title = title
        self.renderer = Renderer()
        self.camera = OrbitCamera()
        self.controls = MouseControls(self.camera)
        self.mesh = None
        self.mesh_path: str | None = None
        self.fb_size = (width, height)
        self._fps = 0.0
        self._update_sizes()

        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_drop_callback(self.window, self._on_drop)

    # -- loading -------------------------------------------------------------------
    def load(self, path: str):
        mesh = loader.load(path)
        mesh.ensure_normals()
        self.renderer.set_mesh(mesh)
        self.camera.fit(*mesh.bounds())
        self.mesh, self.mesh_path = mesh, path
        print(f"loaded {path}: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} triangles")
        self._update_title()
        return mesh

    # -- window geometry -----------------------------------------------------------
    def _update_sizes(self) -> None:
        fw, fh = glfw.get_framebuffer_size(self.window)     # pixels (may differ on HiDPI screens)
        ww, wh = glfw.get_window_size(self.window)          # screen units (what the mouse reports)
        self.fb_size = (max(fw, 1), max(fh, 1))
        self.camera.aspect = self.fb_size[0] / self.fb_size[1]
        self.controls.set_viewport(ww, wh)

    # -- GLFW callbacks ------------------------------------------------------------
    def _on_resize(self, window, width, height) -> None:
        self._update_sizes()

    def _on_mouse_button(self, window, button, action, mods) -> None:
        x, y = glfw.get_cursor_pos(window)
        if action == glfw.PRESS:
            self.controls.press(button, x, y, mods)
        else:
            self.controls.release(button)

    def _on_cursor(self, window, x, y) -> None:
        self.controls.move(x, y)

    def _on_scroll(self, window, dx, dy) -> None:
        self.controls.scroll(dy)

    def _on_drop(self, window, paths) -> None:
        try:
            self.load(paths[0])
        except Exception as exc:                                  # keep the viewer alive on bad files
            print(f"could not load {paths[0]}: {exc}", file=sys.stderr)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return
        r = self.renderer
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.camera.reset()
        elif key == glfw.KEY_F and self.mesh is not None:
            self.camera.fit(*self.mesh.bounds())
        elif key == glfw.KEY_W:
            r.cycle_wire_mode()
        elif key == glfw.KEY_S:
            r.flat_shading = not r.flat_shading
        elif key == glfw.KEY_P:
            self.camera.orthographic = not self.camera.orthographic
        elif key == glfw.KEY_G:
            r.show_grid = not r.show_grid
        elif key == glfw.KEY_A:
            r.show_axes = not r.show_axes
        elif key == glfw.KEY_C:
            self.screenshot()
        elif key in (glfw.KEY_H, glfw.KEY_F1):
            print(HELP)
        self._update_title()

    # -- frame loop -----------------------------------------------------------------
    def render_frame(self) -> None:
        self.renderer.draw(self.camera, *self.fb_size)

    def run(self) -> None:
        print(HELP)
        last, frames = time.perf_counter(), 0
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.render_frame()
            glfw.swap_buffers(self.window)
            frames += 1
            now = time.perf_counter()
            if now - last >= 0.5:
                self._fps = frames / (now - last)
                frames, last = 0, now
                self._update_title()
        self.close()

    def screenshot(self, path: str | None = None) -> str:
        if path is None:
            os.makedirs("screenshots", exist_ok=True)
            path = time.strftime("screenshots/view_%Y%m%d_%H%M%S.png")
        self.render_frame()                                   # draw, then read before the buffers swap
        image = self.renderer.read_pixels(*self.fb_size)
        path = _save_image(path, image)
        print(f"saved {path}")
        return path

    def _update_title(self) -> None:
        parts = [self.title]
        if self.mesh is not None:
            parts.append(f"{os.path.basename(self.mesh_path)}: {self.mesh.n_vertices:,} vertices, "
                         f"{self.mesh.n_faces:,} triangles")
        parts.append("orthographic" if self.camera.orthographic else "perspective")
        parts.append(self.renderer.WIRE_MODES[self.renderer.wire_mode]
                     + (" / flat" if self.renderer.flat_shading else " / smooth"))
        if self._fps:
            parts.append(f"{self._fps:.0f} fps")
        glfw.set_window_title(self.window, "  |  ".join(parts))

    def close(self) -> None:
        if self.window is not None:
            self.renderer.clear_mesh()
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
