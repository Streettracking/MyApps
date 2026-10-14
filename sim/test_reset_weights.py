import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sim import recognize_train_live as live
from sim import train_monitor as tm


class ResetWeightsTest(unittest.TestCase):
    def test_move_state_aside_never_deletes(self):
        d = Path(tempfile.mkdtemp())
        f = d / "mb_train_state.npz"
        f.write_bytes(b"abc")
        a = live.move_state_aside(f, "20261012_2200")
        b_src = d / "mb_train_state.npz"
        b_src.write_bytes(b"def")
        b = live.move_state_aside(b_src, "20261012_2200")
        self.assertFalse(f.exists())
        self.assertEqual(a.read_bytes(), b"abc")
        self.assertEqual(b.read_bytes(), b"def")
        self.assertNotEqual(a.parent, b.parent)
        self.assertTrue(a.parent.name.startswith("old_reset_"))
        self.assertIsNone(live.move_state_aside(d / "missing.npz"))

    def test_confirm_needs_two_clicks(self):
        mon = SimpleNamespace(_reset_armed_at=-10.0)
        inp = tm.MonitorInput()
        tm.TrainMonitor._press_reset_weights(mon, inp)
        self.assertFalse(inp.reset_weights)
        tm.TrainMonitor._press_reset_weights(mon, inp)
        self.assertTrue(inp.reset_weights)

    def test_caps_gate(self):
        s = SimpleNamespace(onboard=True, remote={"caps": ["reset_weights"]})
        self.assertTrue(live.onboard_supports(s, "reset_weights"))
        s.remote = {}
        self.assertFalse(live.onboard_supports(s, "reset_weights"))


if __name__ == "__main__":
    unittest.main()