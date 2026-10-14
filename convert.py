#!/usr/bin/env python
"""Convert a mesh between OBJ, STL and PLY using the viewer's own loader.

    python convert.py models/teapot.obj teapot.stl            # binary STL
    python convert.py scan.ply scan.obj --no-normals
    python convert.py part.stl part.ply --binary               # binary PLY
    python convert.py model.obj --info                         # just print statistics
"""
from __future__ import annotations

import argparse
import sys

from viewer import loader


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("destination", nargs="?")
    ap.add_argument("--binary", action="store_true", help="write binary PLY (STL is binary by default)")
    ap.add_argument("--ascii", action="store_true", help="write ASCII STL / PLY")
    ap.add_argument("--no-normals", action="store_true", help="drop vertex normals before writing")
    ap.add_argument("--info", action="store_true", help="only print mesh statistics")
    args = ap.parse_args(argv)

    mesh = loader.load(args.source)
    lo, hi = mesh.bounds()
    print(f"{args.source}: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} triangles, "
          f"normals in file: {'yes' if mesh.normals is not None else 'no'}")
    print(f"  bounds min {np.round(lo, 4).tolist()}  max {np.round(hi, 4).tolist()}")
    if args.info or not args.destination:
        return 0

    if args.no_normals:
        mesh.normals = None
    elif mesh.normals is None:
        mesh.ensure_normals()
    binary = True if args.binary else False if args.ascii else None
    loader.save(mesh, args.destination, binary=binary)
    print(f"wrote {args.destination}")
    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
else:
    import numpy as np
