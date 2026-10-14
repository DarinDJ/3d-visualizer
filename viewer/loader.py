"""Object loader: readers and writers for OBJ, STL and PLY meshes.

    mesh = loader.load("model.obj")       # dispatches on the file extension
    loader.save(mesh, "model.ply")        # same for writing

Only geometry is handled (positions, triangles, optional normals).  Materials, textures
and colours are ignored, which keeps every parser short enough to read in one sitting.
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np

from .mesh import Mesh, _unit


def _weld(corners: np.ndarray):
    """Turn loose triangle corners (3M, 3) into shared vertices + (M, 3) index triples."""
    uniq, inverse = np.unique(np.asarray(corners, dtype=np.float32), axis=0, return_inverse=True)
    return uniq, inverse.reshape(-1, 3).astype(np.int32)


def _triangulate(polygons) -> np.ndarray:
    """Fan-triangulate a list of index polygons (or pass an (M, 3) array straight through)."""
    if isinstance(polygons, np.ndarray) and polygons.ndim == 2 and polygons.shape[1] == 3:
        return polygons.astype(np.int32)
    tris = []
    for poly in polygons:
        for k in range(1, len(poly) - 1):
            tris.append((int(poly[0]), int(poly[k]), int(poly[k + 1])))
    return np.asarray(tris, dtype=np.int32).reshape(-1, 3)


# ============================================================================ OBJ
def load_obj(path: str) -> Mesh:
    """Wavefront OBJ.

    Handles:  ``v`` / ``vn`` / ``f`` records, polygons with any number of corners
              (fan-triangulated), every face token form ``v``, ``v/vt``, ``v//vn``, ``v/vt/vn``
              and negative (relative) indices.
    Ignores:  ``vt``, materials, groups, smoothing groups, lines and curves.
    """
    positions: list[tuple[float, float, float]] = []
    file_normals: list[tuple[float, float, float]] = []
    corners: list[tuple[int, int]] = []          # (vertex index, normal index or -1) per triangle corner
    all_have_normals = True

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "v":
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vn":
                file_normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "f":
                poly = []
                for token in parts[1:]:
                    comps = token.split("/")
                    vi = int(comps[0])
                    vi = vi - 1 if vi > 0 else len(positions) + vi      # negative = relative to the last vertex
                    ni = -1
                    if len(comps) >= 3 and comps[2]:
                        n = int(comps[2])
                        ni = n - 1 if n > 0 else len(file_normals) + n
                    else:
                        all_have_normals = False
                    poly.append((vi, ni))
                for k in range(1, len(poly) - 1):                      # fan triangulation
                    corners.append(poly[0])
                    corners.append(poly[k])
                    corners.append(poly[k + 1])

    name = os.path.splitext(os.path.basename(path))[0]
    verts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if not corners:                                                    # no faces: a point cloud
        return Mesh(verts, np.zeros((0, 3), np.int32), None, name)

    pairs = np.asarray(corners, dtype=np.int64).reshape(-1, 2)
    usable = (all_have_normals and len(file_normals) > 0
              and pairs[:, 1].min() >= 0 and pairs[:, 1].max() < len(file_normals))
    if usable:
        # OBJ indexes positions and normals separately, but a GPU wants one index per vertex.
        # So emit one output vertex for every distinct (position, normal) pair that is used.
        uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
        normals = _unit(np.asarray(file_normals, dtype=np.float64)[uniq[:, 1]])
        return Mesh(verts[uniq[:, 0]], inverse.reshape(-1, 3), normals, name)
    return Mesh(verts, pairs[:, 0].reshape(-1, 3), None, name)


def save_obj(mesh: Mesh, path: str, write_normals: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {mesh.name}: {mesh.n_vertices} vertices, {mesh.n_faces} triangles\n")
        np.savetxt(fh, mesh.vertices, fmt="v %.6f %.6f %.6f")
        f = mesh.faces.astype(np.int64) + 1                            # OBJ indices start at 1
        if write_normals and mesh.normals is not None:
            np.savetxt(fh, mesh.normals, fmt="vn %.6f %.6f %.6f")
            np.savetxt(fh, f[:, [0, 0, 1, 1, 2, 2]], fmt="f %d//%d %d//%d %d//%d")
        else:
            np.savetxt(fh, f, fmt="f %d %d %d")


# ============================================================================ STL
_STL_RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])


def load_stl(path: str) -> Mesh:
    """STL, binary or ASCII.

    STL stores loose triangles with no shared vertices, so identical corner positions are
    *welded* into shared vertices.  That is what makes smooth shading possible and also
    makes the mesh roughly six times smaller in memory.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    name = os.path.splitext(os.path.basename(path))[0]
    is_binary = len(raw) >= 84 and len(raw) == 84 + 50 * struct.unpack("<I", raw[80:84])[0]
    if is_binary:
        n = struct.unpack("<I", raw[80:84])[0]
        tri = np.frombuffer(raw, dtype=_STL_RECORD, count=n, offset=84)["vertices"].reshape(-1, 3)
    elif raw.lstrip().startswith(b"solid"):
        text = raw.decode("ascii", errors="replace")
        nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
        tri = np.asarray(nums, dtype=np.float64).reshape(-1, 3)
    else:
        raise ValueError(f"{path}: not a valid STL file")
    verts, faces = _weld(tri)
    return Mesh(verts, faces, None, name)


def save_stl(mesh: Mesh, path: str, binary: bool = True) -> None:
    v, f = mesh.vertices, mesh.faces
    fn = mesh.face_normals()
    if binary:
        rec = np.zeros(mesh.n_faces, dtype=_STL_RECORD)
        rec["normal"] = fn
        rec["vertices"] = v[f]
        with open(path, "wb") as fh:
            fh.write(f"{mesh.name[:60]} (binary STL)".encode("ascii", "replace").ljust(80, b"\0"))
            fh.write(struct.pack("<I", mesh.n_faces))
            fh.write(rec.tobytes())
    else:
        with open(path, "w", encoding="ascii") as fh:
            fh.write(f"solid {mesh.name}\n")
            for n, tri in zip(fn, v[f]):
                fh.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
                for p in tri:
                    fh.write(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
                fh.write("    endloop\n  endfacet\n")
            fh.write(f"endsolid {mesh.name}\n")


# ============================================================================ PLY
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _parse_ply_header(raw: bytes):
    """Return (format, elements, body_offset).  elements = [(name, count, [(prop, dtype, list_count_dtype)])]."""
    end = raw.find(b"end_header")
    if not raw.startswith(b"ply") or end < 0:
        raise ValueError("not a PLY file")
    body_offset = raw.find(b"\n", end) + 1
    fmt, elements = None, []
    for line in raw[:body_offset].decode("ascii", errors="replace").splitlines()[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if parts[1] == "list":
                elements[-1][2].append((parts[4], _PLY_TYPES[parts[3]], _PLY_TYPES[parts[2]]))
            else:
                elements[-1][2].append((parts[2], _PLY_TYPES[parts[1]], None))
    return fmt, elements, body_offset


def _read_ply_ascii(body: bytes, elements):
    lines = [ln for ln in body.decode("ascii", errors="replace").splitlines() if ln.strip()]
    out, li = {}, 0
    for name, count, props in elements:
        rows, li = lines[li:li + count], li + count
        if not any(p[2] for p in props):                                  # scalars only: vectorised
            arr = np.array([r.split() for r in rows], dtype=np.float64).reshape(len(rows), -1)
            out[name] = {p[0]: arr[:, k].astype(p[1]) for k, p in enumerate(props)}
            continue
        cols = {p[0]: [] for p in props}
        for r in rows:
            toks, t = r.split(), 0
            for pname, dt, count_dt in props:
                if count_dt is None:
                    cols[pname].append(float(toks[t])); t += 1
                else:
                    n = int(toks[t])
                    cols[pname].append(np.array(toks[t + 1:t + 1 + n], dtype=np.float64).astype(dt)); t += 1 + n
        out[name] = {p[0]: (np.array(cols[p[0]], dtype=p[1]) if p[2] is None else cols[p[0]]) for p in props}
    return out


def _read_ply_binary(body: bytes, elements, endian: str):
    out, off = {}, 0
    for name, count, props in elements:
        if not any(p[2] for p in props):                                   # fixed-size records
            dt = np.dtype([(p[0], endian + p[1]) for p in props])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=off)
            off += count * dt.itemsize
            out[name] = {p[0]: arr[p[0]] for p in props}
            continue
        if len(props) == 1:                                                # fast path: all-triangle face list
            pname, dt, count_dt = props[0]
            tri_dt = np.dtype([("n", endian + count_dt), ("i", endian + dt, 3)])
            if off + count * tri_dt.itemsize <= len(body):
                arr = np.frombuffer(body, dtype=tri_dt, count=count, offset=off)
                if count == 0 or np.all(arr["n"] == 3):
                    out[name] = {pname: arr["i"].astype(np.int64)}
                    off += count * tri_dt.itemsize
                    continue
        cols = {p[0]: [] for p in props}                                   # general (slow) path
        for _ in range(count):
            for pname, dt, count_dt in props:
                if count_dt is None:
                    d = np.dtype(endian + dt)
                    cols[pname].append(np.frombuffer(body, d, 1, off)[0]); off += d.itemsize
                else:
                    cd, d = np.dtype(endian + count_dt), np.dtype(endian + dt)
                    n = int(np.frombuffer(body, cd, 1, off)[0]); off += cd.itemsize
                    cols[pname].append(np.frombuffer(body, d, n, off).astype(np.int64)); off += n * d.itemsize
        out[name] = {p[0]: (np.array(cols[p[0]]) if p[2] is None else cols[p[0]]) for p in props}
    return out


def load_ply(path: str) -> Mesh:
    """Stanford PLY (ascii, binary_little_endian or binary_big_endian).

    Reads x y z, optional nx ny nz and the face list.  Faces with more than three corners
    are fan-triangulated; a file without faces is loaded as a point cloud.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    fmt, elements, body_offset = _parse_ply_header(raw)
    body = raw[body_offset:]
    if fmt == "ascii":
        data = _read_ply_ascii(body, elements)
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        data = _read_ply_binary(body, elements, "<" if fmt == "binary_little_endian" else ">")
    else:
        raise ValueError(f"{path}: unknown PLY format {fmt!r}")

    v = data.get("vertex")
    if v is None:
        raise ValueError(f"{path}: PLY file has no vertex element")
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    normals = None
    if all(k in v for k in ("nx", "ny", "nz")):
        normals = _unit(np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float64))
    face = data.get("face", {})
    idx = face.get("vertex_indices", face.get("vertex_index"))
    faces = _triangulate(idx) if idx is not None else np.zeros((0, 3), np.int32)
    return Mesh(verts, faces, normals, os.path.splitext(os.path.basename(path))[0])


def save_ply(mesh: Mesh, path: str, binary: bool = False, write_normals: bool = True) -> None:
    has_n = write_normals and mesh.normals is not None
    header = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
              f"comment {mesh.name}", f"element vertex {mesh.n_vertices}",
              "property float x", "property float y", "property float z"]
    if has_n:
        header += ["property float nx", "property float ny", "property float nz"]
    header += [f"element face {mesh.n_faces}", "property list uchar int vertex_indices", "end_header"]
    vdata = np.hstack([mesh.vertices, mesh.normals]) if has_n else mesh.vertices
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        if binary:
            fh.write(vdata.astype("<f4").tobytes())
            rec = np.zeros(mesh.n_faces, dtype=np.dtype([("n", "u1"), ("i", "<i4", 3)]))
            rec["n"], rec["i"] = 3, mesh.faces
            fh.write(rec.tobytes())
        else:
            np.savetxt(fh, vdata, fmt="%.6f")
            np.savetxt(fh, mesh.faces, fmt="3 %d %d %d")


# ============================================================================ dispatch
_LOADERS = {".obj": load_obj, ".stl": load_stl, ".ply": load_ply}
SUPPORTED = tuple(_LOADERS)


def load(path: str) -> Mesh:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"unsupported file type {ext!r}; supported: {', '.join(SUPPORTED)}")
    return _LOADERS[ext](path)


def save(mesh: Mesh, path: str, binary: bool | None = None) -> None:
    """Write in the format given by the extension.  binary=None picks the usual default
    for each format (binary STL, ASCII PLY)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        save_obj(mesh, path)
    elif ext == ".stl":
        save_stl(mesh, path, binary=True if binary is None else binary)
    elif ext == ".ply":
        save_ply(mesh, path, binary=False if binary is None else binary)
    else:
        raise ValueError(f"unsupported file type {ext!r}; supported: {', '.join(SUPPORTED)}")
