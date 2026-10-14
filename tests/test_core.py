"""Unit tests for everything that does not need an OpenGL context.

    python -m unittest discover tests        (or simply: python -m pytest)
"""
from __future__ import annotations

import itertools
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from viewer import camera as cam                                                   # noqa: E402
from viewer import loader                                                          # noqa: E402
from viewer.controls import BUTTON_LEFT, BUTTON_RIGHT, MOD_SHIFT, MouseControls   # noqa: E402
from viewer.mesh import Mesh                                                       # noqa: E402


def unit_cube() -> Mesh:
    """Cube from -1..1 with 8 shared vertices and 12 outward-wound triangles."""
    verts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    index = {tuple(v): k for k, v in enumerate(verts.tolist())}
    quads = [
        [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],        # +Z
        [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],    # -Z
        [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)],        # +X
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],    # -X
        [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)],        # +Y
        [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],    # -Y
    ]
    faces = []
    for quad in quads:
        a, b, c, d = [index[tuple(float(x) for x in p)] for p in quad]
        faces += [[a, b, c], [a, c, d]]
    return Mesh(verts, np.array(faces), None, "cube")


def surface_area(mesh: Mesh) -> float:
    return float(np.linalg.norm(mesh.face_normals(normalized=False), axis=1).sum() / 2.0)


# ---------------------------------------------------------------------------- matrices
class MatrixTests(unittest.TestCase):
    def test_look_at_moves_eye_to_origin_and_target_onto_minus_z(self):
        V = cam.look_at((1, 2, 5), (1, 2, 0))
        np.testing.assert_allclose(V @ [1, 2, 5, 1], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(V @ [1, 2, 0, 1], [0, 0, -5, 1], atol=1e-12)

    def test_view_rotation_is_orthonormal(self):
        R = cam.look_at((3, 1, -2), (0, 0.5, 0))[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_perspective_maps_near_and_far_planes_to_ndc_limits(self):
        P = cam.perspective(60, 1.5, 0.1, 100)
        for z, expected in ((-0.1, -1.0), (-100.0, 1.0)):
            c = P @ [0, 0, z, 1]
            self.assertAlmostEqual(c[2] / c[3], expected, places=9)

    def test_intrinsics_matrix_matches_perspective(self):
        w, h, fov = 640, 480, 50.0
        f = (h / 2) / math.tan(math.radians(fov) / 2)
        P1 = cam.perspective(fov, w / h, 0.1, 50)
        P2 = cam.perspective_from_intrinsics(f, f, w / 2, h / 2, w, h, 0.1, 50)
        np.testing.assert_allclose(P1, P2, atol=1e-12)

    def test_project_agrees_with_the_pinhole_formula(self):
        w, h = 800, 600
        c = cam.OrbitCamera(aspect=w / h)
        c.fit((-1, -1, -1), (1, 1, 1))
        K, (R, t) = c.intrinsics(w, h), c.extrinsics()
        p = np.array([0.3, -0.2, 0.1])
        x = R @ p + t                                        # camera space, looking down -z
        u = K[0, 0] * x[0] / -x[2] + K[0, 2]
        v = K[1, 2] - K[1, 1] * x[1] / -x[2]
        px, py, _ = cam.project(p, c.view_matrix(), c.projection_matrix(), w, h)
        self.assertAlmostEqual(px, u, places=6)
        self.assertAlmostEqual(py, v, places=6)


# ---------------------------------------------------------------------------- orbit camera
class OrbitCameraTests(unittest.TestCase):
    def test_fit_puts_the_whole_box_inside_the_view(self):
        c = cam.OrbitCamera(aspect=16 / 9)
        lo, hi = np.array([-1.0, -2.0, -3.0]), np.array([4.0, 5.0, 6.0])
        c.fit(lo, hi)
        V, P = c.view_matrix(), c.projection_matrix()
        for corner in itertools.product(*zip(lo, hi)):
            clip = P @ V @ np.append(corner, 1.0)
            self.assertGreater(clip[3], 0.0)                 # in front of the camera
            self.assertTrue(np.all(np.abs(clip[:3] / clip[3]) <= 1.0), corner)

    def test_pan_moves_the_scene_by_the_dragged_pixel_distance(self):
        w, h = 800, 600
        c = cam.OrbitCamera(aspect=w / h)
        c.fit((-1, -1, -1), (1, 1, 1))
        anchor = c.target.copy()
        x0, y0, _ = cam.project(anchor, c.view_matrix(), c.projection_matrix(), w, h)
        c.pan(100, -40, h)
        x1, y1, _ = cam.project(anchor, c.view_matrix(), c.projection_matrix(), w, h)
        self.assertAlmostEqual(x1 - x0, 100.0, places=6)
        self.assertAlmostEqual(y1 - y0, -40.0, places=6)

    def test_orbit_keeps_the_distance_and_clamps_the_pitch(self):
        c = cam.OrbitCamera(distance=5.0)
        c.orbit(1.0, 10.0)
        self.assertAlmostEqual(np.linalg.norm(c.position - c.target), 5.0)
        self.assertLessEqual(c.pitch, cam.OrbitCamera.MAX_PITCH)
        self.assertTrue(np.all(np.isfinite(c.view_matrix())))

    def test_dolly_is_clamped_and_reset_returns_home(self):
        c = cam.OrbitCamera()
        c.fit((0, 0, 0), (2, 2, 2))
        home = c.state()
        c.dolly(1e-9)
        self.assertEqual(c.distance, c.min_distance)
        c.orbit(0.3, 0.2)
        c.reset()
        self.assertAlmostEqual(c.distance, home["distance"])
        self.assertAlmostEqual(c.yaw, home["yaw"])

    def test_orthographic_projection_has_no_perspective(self):
        c = cam.OrbitCamera(orthographic=True)
        c.fit((-1, -1, -1), (1, 1, 1))
        P = c.projection_matrix()
        self.assertEqual(P[3, 3], 1.0)
        self.assertEqual(P[3, 2], 0.0)


# ---------------------------------------------------------------------------- mouse controls
class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.cam = cam.OrbitCamera()
        self.cam.fit((-1, -1, -1), (1, 1, 1))
        self.ctl = MouseControls(self.cam)
        self.ctl.set_viewport(800, 600)

    def test_left_drag_rotates_without_moving_the_target(self):
        yaw0, target0 = self.cam.yaw, self.cam.target.copy()
        self.ctl.press(BUTTON_LEFT, 100, 100)
        self.ctl.move(160, 100)
        self.ctl.release(BUTTON_LEFT)
        self.assertNotAlmostEqual(self.cam.yaw, yaw0)
        np.testing.assert_allclose(self.cam.target, target0)

    def test_dragging_one_window_height_is_one_full_turn(self):
        yaw0 = self.cam.yaw
        self.ctl.press(BUTTON_LEFT, 0, 0)
        self.ctl.move(600, 0)
        self.assertAlmostEqual(math.cos(self.cam.yaw - yaw0), 1.0)

    def test_right_drag_and_shift_left_drag_pan(self):
        for button, mods in ((BUTTON_RIGHT, 0), (BUTTON_LEFT, MOD_SHIFT)):
            yaw0, target0 = self.cam.yaw, self.cam.target.copy()
            self.ctl.press(button, 10, 10, mods)
            self.ctl.move(50, 30)
            self.ctl.release(button)
            self.assertEqual(self.cam.yaw, yaw0)
            self.assertGreater(np.linalg.norm(self.cam.target - target0), 0.0)

    def test_scroll_zooms_in_and_out(self):
        d0 = self.cam.distance
        self.ctl.scroll(1)
        self.assertLess(self.cam.distance, d0)
        self.ctl.scroll(-1)
        self.assertAlmostEqual(self.cam.distance, d0)

    def test_move_without_a_button_does_nothing(self):
        st = self.cam.state()
        self.ctl.move(300, 300)
        self.assertEqual(self.cam.state()["yaw"], st["yaw"])
        np.testing.assert_allclose(self.cam.target, st["target"])


# ---------------------------------------------------------------------------- mesh
class MeshTests(unittest.TestCase):
    def test_vertex_normals_are_unit_length_and_point_outward(self):
        m = unit_cube()
        n = m.compute_vertex_normals()
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(np.einsum("ij,ij->i", n, m.vertices) > 0))

    def test_face_normals_point_outward(self):
        m = unit_cube()
        centroids = m.vertices[m.faces].mean(axis=1)
        self.assertTrue(np.all(np.einsum("ij,ij->i", m.face_normals(), centroids) > 0))

    def test_bounds_center_radius(self):
        m = unit_cube()
        lo, hi = m.bounds()
        np.testing.assert_allclose(lo, [-1, -1, -1])
        np.testing.assert_allclose(hi, [1, 1, 1])
        np.testing.assert_allclose(m.center(), [0, 0, 0])
        self.assertAlmostEqual(m.radius(), math.sqrt(3))

    def test_bad_face_index_is_rejected(self):
        with self.assertRaises(ValueError):
            Mesh(np.zeros((3, 3)), np.array([[0, 1, 5]]))


# ---------------------------------------------------------------------------- loader
OBJ_TEXT = """# a quad and a triangle using several face syntaxes
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
vt 0 0
vn 0 0 1
f 1/1/1 2/1/1 3/1/1 4/1/1
f -5//1 -3//1 -1//1
"""


class LoaderTests(unittest.TestCase):
    def test_obj_quads_negative_indices_and_normals(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.obj")
            with open(p, "w") as fh:
                fh.write(OBJ_TEXT)
            m = loader.load(p)
        self.assertEqual(m.n_faces, 3)                     # quad -> 2 triangles, + 1 triangle
        self.assertEqual(m.n_vertices, 5)
        self.assertIsNotNone(m.normals)
        np.testing.assert_allclose(m.normals, [[0, 0, 1]] * 5)
        np.testing.assert_allclose(m.bounds()[1], [1, 1, 1])

    def test_obj_without_normals_gets_computed_normals(self):
        text = "\n".join(line for line in OBJ_TEXT.splitlines() if not line.startswith("vn")) + "\n"
        text = text.replace("/1/1", "").replace("//1", "")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.obj")
            with open(p, "w") as fh:
                fh.write(text)
            m = loader.load(p)
        self.assertIsNone(m.normals)
        self.assertEqual(m.ensure_normals().shape, (5, 3))

    def test_round_trip_through_every_format(self):
        m = unit_cube()
        m.ensure_normals()
        cases = [(".obj", {}), (".stl", {"binary": True}), (".stl", {"binary": False}),
                 (".ply", {"binary": False}), (".ply", {"binary": True})]
        for ext, kwargs in cases:
            with self.subTest(ext=ext, **kwargs), tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "cube" + ext)
                loader.save(m, p, **kwargs)
                back = loader.load(p)
                self.assertEqual(back.n_faces, 12)
                self.assertEqual(back.n_vertices, 8)
                np.testing.assert_allclose(back.bounds()[0], [-1, -1, -1], atol=1e-6)
                np.testing.assert_allclose(back.bounds()[1], [1, 1, 1], atol=1e-6)
                self.assertAlmostEqual(surface_area(back), 24.0, places=4)
                if ext != ".stl":                               # STL cannot carry vertex normals
                    self.assertIsNotNone(back.normals)

    def test_unknown_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            loader.load("model.xyz")


if __name__ == "__main__":
    unittest.main(verbosity=2)
