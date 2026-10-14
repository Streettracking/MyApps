import unittest
from types import SimpleNamespace

from sim import recognize_train_live as live
from sim import train_monitor as tm


class OnboardCapsTest(unittest.TestCase):
    def test_sim_always_supported(self):
        s = SimpleNamespace(onboard=False, remote={})
        self.assertTrue(live.onboard_supports(s, "recalibrate"))

    def test_old_onboard_without_caps_is_not(self):
        s = SimpleNamespace(onboard=True, remote={"mode": "manual"})
        self.assertFalse(live.onboard_supports(s, "recalibrate"))
        self.assertFalse(live.onboard_supports(s, "restart"))

    def test_caps_list_enables(self):
        s = SimpleNamespace(onboard=True, remote={"caps": ["recalibrate", "reset_weights"]})
        self.assertTrue(live.onboard_supports(s, "recalibrate"))
        self.assertFalse(live.onboard_supports(s, "restart"))

    def test_view_flag_defaults_and_reads(self):
        mon = SimpleNamespace()
        self.assertTrue(tm._view_flag(mon, "recal_ok"))
        mon._last_view = tm.MonitorView(recal_ok=False)
        self.assertFalse(tm._view_flag(mon, "recal_ok"))
        self.assertTrue(tm._view_flag(mon, "restart_ok"))


if __name__ == "__main__":
    unittest.main()