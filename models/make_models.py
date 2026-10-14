#!/usr/bin/env python
"""Generate the procedural sample models (cube, sphere, torus) in this folder as OBJ files.

    python models/make_models.py

Each generator returns a Mesh with analytic normals; the script checks that the triangle
winding agrees with those normals (all faces should be 'outward').
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viewer.loader import save_obj   # noqa: E402
from viewer.mesh import Mesh         # noqa: E402


def cube(size: float = 1.0) -> Mesh:
    h = size / 2.0
    quads = [  # four corners per face, counter-clockwise when seen from outside
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],        # +Z
        [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)],    # -Z
        [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)],        # +X
        [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)],    # -X
        [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)],        # +Y
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],    # -Y
    ]
    verts, norms, faces = [], [], []
    for quad in quads:
        q = np.array(quad, dtype=np.float64)
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        base = len(verts)
        verts.extend(q)
        norms.extend([n] * 4)                       # 24 vertices: each face has its own flat normals
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return Mesh(np.array(verts), np.array(faces), np.array(norms), "cube")


def sphere(radius: float = 1.0, rings: int = 32, segments: int = 64) -> Mesh:
    verts, norms, faces = [], [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings                 # polar angle, 0 at the north pole
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments      # azimuth
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            norms.append(n)
            verts.append(tuple(radius * c for c in n))
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + (j + 1) % segments
            d = (i + 1) * segments + j
            if i > 0:
                faces.append([a, b, d])             # skip the degenerate triangles at the poles
            if i < rings - 1:
                faces.append([b, c, d])
    return Mesh(np.array(verts), np.array(faces), np.array(norms), "sphere")


def torus(major: float = 1.0, minor: float = 0.4, segments: int = 64, sides: int = 32) -> Mesh:
    verts, norms, faces = [], [], []
    for i in range(segments):
        u = 2.0 * math.pi * i / segments            # around the ring
        cu, su = math.cos(u), math.sin(u)
        for j in range(sides):
            v = 2.0 * math.pi * j / sides           # around the tube
            cv, sv = math.cos(v), math.sin(v)
            verts.append(((major + minor * cv) * cu, minor * sv, (major + minor * cv) * su))
            norms.append((cv * cu, sv, cv * su))
    for i in range(segments):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = ((i + 1) % segments) * sides + (j + 1) % sides
            d = ((i + 1) % segments) * sides + j
            faces += [[a, b, d], [b, c, d]]
    return Mesh(np.array(verts), np.array(faces), np.array(norms), "torus")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for mesh in (cube(), sphere(), torus()):
        out = os.path.join(here, mesh.name + ".obj")
        save_obj(mesh, out)
        agree = np.einsum("ij,ij->i", mesh.face_normals(), mesh.normals[mesh.faces[:, 0]]) > 0
        print(f"{out}: {mesh.n_vertices} vertices, {mesh.n_faces} triangles, "
              f"faces wound outward: {agree.mean():.0%}")


if __name__ == "__main__":
    main()
