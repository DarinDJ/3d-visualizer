"""Mesh container + pure-numpy geometry helpers (no OpenGL in this file).

A mesh is an *indexed triangle list*:

    vertices : (N, 3) float32   xyz positions
    faces    : (M, 3) int32     three vertex indices per triangle (counter-clockwise = front side)
    normals  : (N, 3) float32   optional per-vertex unit normals, used for smooth shading
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise the rows of v; rows of zero length are left as zero."""
    length = np.linalg.norm(v, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return v / length


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32).reshape(-1, 3)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32).reshape(-1, 3)
            if len(self.normals) != len(self.vertices):
                raise ValueError("normals must have exactly one entry per vertex")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("face index out of range")

    # ------------------------------------------------------------------ stats
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as (min_xyz, max_xyz)."""
        if self.n_vertices == 0:
            return np.zeros(3), np.zeros(3)
        return (self.vertices.min(axis=0).astype(np.float64),
                self.vertices.max(axis=0).astype(np.float64))

    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) / 2.0

    def radius(self) -> float:
        """Radius of the bounding sphere around center() (half the box diagonal)."""
        lo, hi = self.bounds()
        return float(np.linalg.norm(hi - lo) / 2.0) or 1.0

    # ---------------------------------------------------------------- normals
    def face_normals(self, normalized: bool = True) -> np.ndarray:
        """(M, 3) normal of every triangle = cross product of two of its edges."""
        v, f = self.vertices.astype(np.float64), self.faces
        n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        return _unit(n) if normalized else n

    def compute_vertex_normals(self) -> np.ndarray:
        """Smooth per-vertex normals = area-weighted average of the adjacent face normals.

        The *un-normalised* cross product has length 2*area, so simply summing it over
        the faces that touch a vertex gives area weighting for free.  ``np.bincount``
        does that scatter-add without a Python loop.
        """
        fn = self.face_normals(normalized=False)
        idx = self.faces.ravel()
        acc = np.zeros((self.n_vertices, 3))
        for k in range(3):
            acc[:, k] = np.bincount(idx, weights=np.repeat(fn[:, k], 3), minlength=self.n_vertices)
        self.normals = _unit(acc).astype(np.float32)
        return self.normals

    def ensure_normals(self) -> np.ndarray:
        """Return per-vertex normals, computing them if the file did not provide any."""
        if self.normals is None:
            self.compute_vertex_normals()
        return self.normals
