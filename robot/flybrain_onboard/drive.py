"""SportClient.Move / StopMove. No Damp, no SwitchJoystick.

``Move`` is held by the sport service for about a second, so a dropped
sender still coasts until ``StopMove`` or that timeout. This wrapper
resends while the pilot wants motion and calls ``StopMove`` whenever
the pilot wants none.
"""

from __future__ import annotations

from sim.pilot import clamp_velocity


class SportDrive:
    def __init__(self, client=None):
        self.client = client
        self.moves = []
        self.stops = 0

    def move(self, x: float, z: float) -> None:
        cx, cz = clamp_velocity(x, z)
        self.moves.append((cx, 0.0, cz))
        if self.client is not None:
            self.client.Move(cx, 0.0, cz)

    def stop(self) -> None:
        self.stops += 1
        if self.client is not None:
            self.client.StopMove()
