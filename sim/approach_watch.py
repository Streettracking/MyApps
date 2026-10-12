"""Run timer for the preference index. It does not teach and does not move weights.

The index is (time near the conspecific − time near the cube) / total time.
«новый заход» clears the clock only.
"""

from __future__ import annotations


class ApproachWatch:
    def __init__(self) -> None:
        self.near_dog = 0.0
        self.near_cube = 0.0
        self.total = 0.0
        self.dog = None
        self.cube = None

    def note(self, dt: float, dog: bool, cube: bool, drive) -> None:
        step = float(dt)
        if step < 0.0:
            step = 0.0
        self.total += step
        if dog:
            self.near_dog += step
            if not cube and drive is not None:
                self.dog = float(drive)
        if cube:
            self.near_cube += step
            if not dog and drive is not None:
                self.cube = float(drive)

    def index(self):
        if self.total <= 1e-9:
            return None
        return (self.near_dog - self.near_cube) / self.total

    def reset(self) -> None:
        self.near_dog = 0.0
        self.near_cube = 0.0
        self.total = 0.0
