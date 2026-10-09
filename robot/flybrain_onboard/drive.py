"""SportClient.Move / StopMove / StandUp / StandDown. No Damp, no SwitchJoystick.

``Move`` is held by the sport service for about a second. This wrapper
resends ``Move`` only while the pilot wants motion. ``StopMove`` is the
caller's choice: the brain sends it once, on an edge, not on every tick.
``StandUp``, ``StandDown``, and ``RecoveryStand`` run only when asked.
"""

from __future__ import annotations

from sim.pilot import clamp_velocity


class SportDrive:
    def __init__(self, client=None):
        self.client = client
        self.moves = []
        self.stops = 0
        self.stands = []

    def move(self, x: float, z: float) -> None:
        cx, cz = clamp_velocity(x, z)
        self.moves.append((cx, 0.0, cz))
        if self.client is not None:
            self.client.Move(cx, 0.0, cz)

    def stop(self) -> None:
        self.stops += 1
        if self.client is not None:
            self.client.StopMove()

    def stand_up(self) -> None:
        self.stands.append("StandUp")
        if self.client is not None:
            self.client.StandUp()

    def stand_down(self) -> None:
        self.stands.append("StandDown")
        if self.client is not None:
            self.client.StandDown()

    def recovery_stand(self) -> bool:
        """Go2 ``SportClient.RecoveryStand``. False when this SDK has no such method."""
        if self.client is None:
            self.stands.append("RecoveryStand")
            return True
        fn = getattr(self.client, "RecoveryStand", None)
        if fn is None:
            return False
        self.stands.append("RecoveryStand")
        fn()
        return True
