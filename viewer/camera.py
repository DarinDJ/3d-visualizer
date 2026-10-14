"""Camera module.

Turning a 3-D point into a pixel is a chain of matrix products:

    world --[view V]--> camera (eye) space --[projection P]--> clip space --(/w)--> NDC --> pixels

* ``look_at``                    builds V from eye position, target and up vector (the *extrinsics*).
* ``perspective`` / ``orthographic``   build P (the *intrinsics*).
* ``perspective_from_intrinsics``      builds P from a pinhole matrix K = [[fx,0,cx],[0,fy,cy],[0,0,1]],
                                 so the viewer can reproduce a real, calibrated camera.
* ``OrbitCamera``                is the interactive camera used by the viewer: it stores a target
                                 point, a distance and two angles (yaw, pitch) and derives V and P.

Conventions (OpenGL style): right-handed world, +Y up; the camera looks down its own -Z axis.
All matrices are 4x4 numpy arrays in ordinary mathematical (row-major) layout, so
``clip = P @ V @ [x, y, z, 1]``.  OpenGL stores matrices column-major, therefore the
renderer uploads them with ``transpose=GL_TRUE``.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------
# Matrix builders
# ----------------------------------------------------------------------------
def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """View matrix: world -> camera space (the classic gluLookAt).

    The camera axes (right s, up u, backward -f) become the *rows* of the rotation part,
    which is the inverse (= transpose) of the camera's orientation.  The translation
    column moves ``eye`` to the origin.
    """
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    f = normalize(target - eye)                              # forward
    s = normalize(np.cross(f, np.asarray(up, np.float64)))  # right (side)
    u = np.cross(s, f)                                       # true up, orthogonal to f and s
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y_deg, aspect, near, far) -> np.ndarray:
    """Perspective projection (same matrix as gluPerspective)."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0                       # w_clip = -z_eye : this is what makes far things small
    return m


def orthographic(half_height, aspect, near, far) -> np.ndarray:
    """Orthographic projection: no perspective divide, parallel lines stay parallel."""
    m = np.identity(4)
    m[0, 0] = 1.0 / (half_height * aspect)
    m[1, 1] = 1.0 / half_height
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def perspective_from_intrinsics(fx, fy, cx, cy, width, height, near, far) -> np.ndarray:
    """OpenGL projection equivalent to a pinhole camera with intrinsics K.

    Pinhole model:   u = fx * x / (-z) + cx        v = cy - fy * y / (-z)
    (image v grows downwards, OpenGL camera looks down -z).  Substituting into
    x_ndc = 2u/width - 1 and y_ndc = 1 - 2v/height gives the matrix below.
    With fx = fy = (height/2) / tan(fov/2) and the principal point in the image centre
    the result is exactly ``perspective()``.
    """
    m = np.zeros((4, 4))
    m[0, 0] = 2.0 * fx / width
    m[0, 2] = (width - 2.0 * cx) / width
    m[1, 1] = 2.0 * fy / height
    m[1, 2] = (2.0 * cy - height) / height
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def project(point, view, proj, width, height):
    """World point -> pixel (u, v) with the origin in the top-left corner.  Also returns w (depth)."""
    c = proj @ view @ np.append(np.asarray(point, np.float64), 1.0)
    ndc = c[:3] / c[3]
    return (ndc[0] + 1.0) / 2.0 * width, (1.0 - ndc[1]) / 2.0 * height, c[3]


# ----------------------------------------------------------------------------
# Interactive orbit camera
# ----------------------------------------------------------------------------
class OrbitCamera:
    """A 'turntable' camera that orbits around a target point.

    State:
        target        point the camera looks at and rotates around
        distance      distance from the eye to the target (zoom)
        yaw           rotation around the world Y axis                      (radians)
        pitch         elevation above the XZ plane, clamped to +-89 degrees (radians)
        fov_y         vertical field of view in degrees
        aspect        viewport width / height
        orthographic  perspective (False) or orthographic (True) projection
    """

    MAX_PITCH = math.radians(89.0)

    def __init__(self, target=(0.0, 0.0, 0.0), distance=3.0, yaw=math.radians(35.0),
                 pitch=math.radians(25.0), fov_y=45.0, aspect=1.0, orthographic=False):
        self.target = np.array(target, dtype=np.float64)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov_y = float(fov_y)
        self.aspect = float(aspect)
        self.orthographic = orthographic
        self.scene_radius = 1.0
        self.min_distance = 0.01
        self.max_distance = 1e4
        self._home = self.state()

    # -- state ------------------------------------------------------------------
    def state(self) -> dict:
        return {"target": self.target.copy(), "distance": self.distance,
                "yaw": self.yaw, "pitch": self.pitch}

    def restore(self, st: dict) -> None:
        self.target = np.array(st["target"], dtype=np.float64)
        self.distance, self.yaw, self.pitch = st["distance"], st["yaw"], st["pitch"]

    def reset(self) -> None:
        """Go back to the view stored by the last fit()."""
        self.restore(self._home)

    # -- derived quantities -------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """Eye position from spherical coordinates (distance, yaw, pitch) around the target."""
        cp = math.cos(self.pitch)
        direction = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        return self.target + self.distance * direction

    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target)

    def basis(self):
        """Camera right, up and forward axes expressed in world coordinates."""
        v = self.view_matrix()
        return v[0, :3].copy(), v[1, :3].copy(), -v[2, :3].copy()

    def near_far(self) -> tuple[float, float]:
        """Clip planes chosen around the scene so depth precision is not wasted."""
        r = self.scene_radius
        near = max(r * 1e-3, (self.distance - r) * 0.5)
        far = self.distance + 4.0 * r
        return near, far

    def projection_matrix(self) -> np.ndarray:
        near, far = self.near_far()
        if self.orthographic:
            # same framing as the perspective view at the target's depth
            half_h = self.distance * math.tan(math.radians(self.fov_y) / 2.0)
            return orthographic(half_h, self.aspect, near, far)
        return perspective(self.fov_y, self.aspect, near, far)

    def intrinsics(self, width, height) -> np.ndarray:
        """Equivalent pinhole matrix K for a viewport of the given pixel size."""
        fy = (height / 2.0) / math.tan(math.radians(self.fov_y) / 2.0)
        return np.array([[fy, 0.0, width / 2.0],
                         [0.0, fy, height / 2.0],
                         [0.0, 0.0, 1.0]])

    def extrinsics(self):
        """(R, t) such that x_camera = R @ x_world + t."""
        v = self.view_matrix()
        return v[:3, :3].copy(), v[:3, 3].copy()

    # -- manipulation ------------------------------------------------------------
    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        """Rotate around the target.  Pitch is clamped so the camera never flips over the pole."""
        self.yaw = (self.yaw + d_yaw) % (2.0 * math.pi)
        self.pitch = max(-self.MAX_PITCH, min(self.MAX_PITCH, self.pitch + d_pitch))

    def pan(self, dx_px: float, dy_px: float, viewport_height: int) -> None:
        """Slide the target in the screen plane so the scene follows the mouse 1:1.

        At the target's depth the viewport spans 2 * distance * tan(fov/2) world units
        vertically, which converts pixels into world units.
        """
        world_per_px = (2.0 * self.distance * math.tan(math.radians(self.fov_y) / 2.0)
                        / max(int(viewport_height), 1))
        right, up, _ = self.basis()
        self.target = self.target + (-right * dx_px + up * dy_px) * world_per_px

    def dolly(self, factor: float) -> None:
        """Multiply the distance to the target (factor < 1 zooms in)."""
        self.distance = min(self.max_distance, max(self.min_distance, self.distance * factor))

    def fit(self, lo, hi, margin: float = 1.15) -> None:
        """Frame an axis-aligned bounding box and remember the result as the 'home' view."""
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.target = (lo + hi) / 2.0
        r = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
        self.scene_radius = r
        fov = math.radians(self.fov_y)
        if self.aspect < 1.0:                     # portrait window: the horizontal fov is the narrow one
            fov = 2.0 * math.atan(math.tan(fov / 2.0) * self.aspect)
        self.distance = margin * r / math.sin(fov / 2.0)   # bounding sphere just fits in the frustum
        self.min_distance, self.max_distance = r * 0.02, r * 200.0
        self._home = self.state()
