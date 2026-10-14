#!/usr/bin/env python
"""Entry point of the viewer.

    python main.py                         # opens the default sample model
    python main.py models/bunny.obj        # opens a given OBJ / STL / PLY file
    python main.py model.stl --ortho       # start in orthographic projection
    python main.py model.ply --screenshot shot.png --hidden   # render one frame to a file and exit
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from viewer.app import ViewerApp
from viewer.loader import SUPPORTED

HERE = os.path.dirname(os.path.abspath(__file__))


def default_model() -> str | None:
    models = os.path.join(HERE, "models")
    for name in ("teapot.obj", "bunny.obj", "torus.obj", "sphere.obj", "cube.obj"):
        candidate = os.path.join(models, name)
        if os.path.exists(candidate):
            return candidate
    for ext in SUPPORTED:
        found = sorted(glob.glob(os.path.join(models, "*" + ext)))
        if found:
            return found[0]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A small from-scratch 3-D model viewer.",
                                 epilog="Press H inside the viewer for the mouse/keyboard help.")
    ap.add_argument("model", nargs="?", help="mesh file to open (.obj, .stl or .ply)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--ortho", action="store_true", help="start with an orthographic projection")
    ap.add_argument("--screenshot", metavar="PNG", help="render one frame, save it and exit")
    ap.add_argument("--hidden", action="store_true", help="do not show the window (with --screenshot)")
    args = ap.parse_args(argv)

    model = args.model or default_model()
    app = ViewerApp(args.width, args.height, visible=not args.hidden)
    try:
        if model:
            app.load(model)
        else:
            print("no model given and none found in models/ - drag & drop a file onto the window")
        app.camera.orthographic = args.ortho
        if args.screenshot:
            app.screenshot(args.screenshot)
            return 0
        app.run()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
