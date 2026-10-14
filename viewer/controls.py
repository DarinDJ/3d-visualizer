"""Mouse manipulation.

Raw mouse events (press / move / release / scroll) are turned into camera operations:

    left drag                 orbit  (rotate around the target)
    right or middle drag      pan    (slide the target in the screen plane)
    shift/ctrl + left drag    pan    (for one-button mice and touch-pads)
    wheel                     dolly  (zoom)

The class knows nothing about the windowing library, so it can be unit-tested; app.py
feeds it GLFW events.  Button and modifier constants use GLFW's numeric values.
"""
from __future__ import annotations

import math

from .camera import OrbitCamera

BUTTON_LEFT, BUTTON_RIGHT, BUTTON_MIDDLE = 0, 1, 2
MOD_SHIFT, MOD_CONTROL = 0x0001, 0x0002


class MouseControls:
    def __init__(self, camera: OrbitCamera, zoom_step: float = 0.9, turns_per_height: float = 1.0):
        self.camera = camera
        self.zoom_step = zoom_step                  # distance multiplier per wheel notch
        self.turns_per_height = turns_per_height    # dragging across the whole window height = 1 full turn
        self.viewport = (800, 600)                  # window size in *screen* pixels (mouse units)
        self.action: str | None = None              # "rotate", "pan" or None
        self._last: tuple[float, float] | None = None

    def set_viewport(self, width: int, height: int) -> None:
        self.viewport = (max(int(width), 1), max(int(height), 1))

    # -- events -------------------------------------------------------------------
    def press(self, button: int, x: float, y: float, mods: int = 0) -> None:
        if button == BUTTON_LEFT:
            self.action = "pan" if mods & (MOD_SHIFT | MOD_CONTROL) else "rotate"
        elif button in (BUTTON_RIGHT, BUTTON_MIDDLE):
            self.action = "pan"
        else:
            return
        self._last = (x, y)

    def release(self, button: int) -> None:
        self.action, self._last = None, None

    def move(self, x: float, y: float) -> None:
        if self.action is None or self._last is None:
            return
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        if self.action == "rotate":
            speed = 2.0 * math.pi * self.turns_per_height / self.viewport[1]
            # drag right -> the model turns to the right (camera orbits left);
            # drag down  -> the camera rises and looks down onto the model
            self.camera.orbit(-dx * speed, dy * speed)
        else:
            self.camera.pan(dx, dy, self.viewport[1])

    def scroll(self, wheel_steps: float) -> None:
        """Positive steps (wheel up) zoom in, negative zoom out; exponential so it feels uniform."""
        self.camera.dolly(self.zoom_step ** wheel_steps)
