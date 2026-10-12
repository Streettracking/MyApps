import unittest
from types import SimpleNamespace
from unittest import mock

from sim import recognize_train_live as live


def _inp(focused=False):
    return SimpleNamespace(focused=focused, estop=False, stop=False, takeover=False, steer_x=0.0, steer_z=0.0)


def _session(pid=4242, alive=True):
    return SimpleNamespace(_brain_view=SimpleNamespace(alive=alive, proc=SimpleNamespace(pid=pid)))


@unittest.skipUnless(live.os.name == "nt", "foreground check is Windows only")
class BrainFocusTest(unittest.TestCase):
    def test_brain_window_keeps_focus(self):
        with mock.patch.object(live, "_foreground_pid", return_value=4242):
            ours = live.effective_focus(_session(), _inp(), 1.0)
        self.assertTrue(ours)
        self.assertFalse(live.teacher_halt_now(_inp(), focused=ours))

    def test_other_window_halts(self):
        with mock.patch.object(live, "_foreground_pid", return_value=999):
            ours = live.effective_focus(_session(), _inp(), 1.0)
        self.assertFalse(ours)
        self.assertTrue(live.teacher_halt_now(_inp(), focused=ours))

    def test_closed_brain_halts(self):
        with mock.patch.object(live, "_foreground_pid", return_value=4242):
            self.assertFalse(live.effective_focus(_session(alive=False), _inp(), 1.0))

    def test_cached_between_checks(self):
        s = _session()
        with mock.patch.object(live, "_foreground_pid", return_value=4242) as fg:
            live.effective_focus(s, _inp(), 1.0)
            live.effective_focus(s, _inp(), 1.1)
            self.assertEqual(fg.call_count, 1)
            live.effective_focus(s, _inp(), 1.3)
            self.assertEqual(fg.call_count, 2)

    def test_own_window_focused_skips_check(self):
        with mock.patch.object(live, "_foreground_pid", side_effect=AssertionError):
            self.assertTrue(live.effective_focus(_session(), _inp(True), 1.0))


@unittest.skipUnless(live.os.name == "nt", "foreground check is Windows only")
class BrainKeysTest(unittest.TestCase):
    def run_keys(self, held, s=None, now=1.0):
        s = s or _session()
        inp = _inp()
        with mock.patch.object(live, "_foreground_pid", return_value=4242), mock.patch.object(live, "_key_down", side_effect=lambda vk: vk in held):
            live.brain_keys_into(s, inp, now)
        return s, inp

    def test_arrow_in_brain_window_halts(self):
        _s, inp = self.run_keys({live.VK_LEFT})
        self.assertEqual(inp.steer_z, 1.0)
        self.assertTrue(live.teacher_halt_now(inp, focused=True))

    def test_e_is_estop_once(self):
        s, inp = self.run_keys({live.VK_E})
        self.assertTrue(inp.estop)
        _s, again = self.run_keys({live.VK_E}, s, 1.5)
        self.assertFalse(again.estop)

    def test_no_keys_no_halt(self):
        _s, inp = self.run_keys(set())
        self.assertFalse(live.teacher_halt_now(inp, focused=True))

    def test_other_window_keys_ignored(self):
        inp = _inp()
        with mock.patch.object(live, "_foreground_pid", return_value=1), mock.patch.object(live, "_key_down", return_value=True):
            live.brain_keys_into(_session(), inp, 1.0)
        self.assertEqual((inp.steer_x, inp.steer_z, inp.estop), (0.0, 0.0, False))


class FocusFallbackTest(unittest.TestCase):
    def test_non_windows_is_old_rule(self):
        with mock.patch.object(live.os, "name", "posix"):
            self.assertFalse(live.effective_focus(_session(), _inp(), 1.0))


if __name__ == "__main__":
    unittest.main()