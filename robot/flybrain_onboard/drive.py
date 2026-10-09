"""SportClient.Move / StopMove / StandUp / StandDown. No Damp, no SwitchJoystick.

``Move`` is held by the sport service for about a second. This wrapper
resends ``Move`` only while the pilot wants motion. ``StopMove`` is the
caller's choice: the brain sends it once, on an edge, not on every tick.

``StandUp`` locks the joints (Unitree: "lock joints and stand tall").
``Move`` is ignored in that pose until ``BalanceStand`` releases the lock.
The laptop bridge sends ``StandUp``, then ``BalanceStand`` about 0.7 s
later. ``stand_up`` only records that deadline. The brain tick fires
``BalanceStand``; this module does not sleep.
"""

from __future__ import annotations

from sim.pilot import clamp_velocity

# Same gap the working laptop bridge leaves between StandUp and BalanceStand.
BALANCE_AFTER_STAND_S = 0.70


class SportDrive:
    def __init__(self, client=None):
        self.client = client
        self.moves = []
        self.stops = 0
        self.stands = []
        # "standup" means StandUp went out and BalanceStand has not.
        self.pose = ""
        self.balance_due = None
        self.last_move_code = None

    def needs_balance(self) -> bool:
        return self.pose == "standup"

    def cancel_balance_timer(self) -> None:
        """Drop a scheduled unlock. A locked stand still unlocks on the next Move."""
        self.balance_due = None

    def stand_up(self, now: float):
        code = self._invoke("StandUp")
        self.pose = "standup"
        self.balance_due = float(now) + BALANCE_AFTER_STAND_S
        return code

    def stand_down(self):
        code = self._invoke("StandDown")
        self.pose = "down"
        self.balance_due = None
        return code

    def recovery_stand(self) -> bool:
        """Go2 ``SportClient.RecoveryStand``. False when this SDK has no such method."""
        if self.client is None:
            self.stands.append("RecoveryStand")
            self.pose = "recovery"
            self.balance_due = None
            return True
        fn = getattr(self.client, "RecoveryStand", None)
        if fn is None:
            return False
        self.stands.append("RecoveryStand")
        self.pose = "recovery"
        self.balance_due = None
        fn()
        return True

    def balance_stand(self):
        """One unlock. None when this SDK has no ``BalanceStand``."""
        self.balance_due = None
        if self.client is not None and getattr(self.client, "BalanceStand", None) is None:
            self.pose = "balance_unavailable"
            return None
        self.stands.append("BalanceStand")
        self.pose = "balance"
        if self.client is None:
            return None
        return self.client.BalanceStand()

    def move(self, x: float, z: float):
        cx, cz = clamp_velocity(x, z)
        self.moves.append((cx, 0.0, cz))
        code = None
        if self.client is not None:
            code = self.client.Move(cx, 0.0, cz)
        self.last_move_code = code
        return code

    def stop(self) -> None:
        self.stops += 1
        if self.client is not None:
            self.client.StopMove()

    def _invoke(self, name: str):
        self.stands.append(name)
        if self.client is None:
            return None
        return getattr(self.client, name)()
