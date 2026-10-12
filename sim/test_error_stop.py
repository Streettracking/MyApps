import unittest
from types import SimpleNamespace

from sim import recognize_train_live as live
from sim.pilot import Pilot


class ErrorStopTest(unittest.TestCase):
    def test_onboard_sends_space_only(self):
        posts = []
        onboard = SimpleNamespace(post=lambda op, **kw: posts.append(op))
        s = SimpleNamespace(pilot=Pilot(), _log=lambda t: None)
        live.error_stop(s, onboard)
        self.assertEqual(posts, ["space"])

    def test_local_pilot_stops_with_zero(self):
        s = SimpleNamespace(pilot=Pilot(), _log=lambda t: None)
        s.pilot.start_auto(0.0)
        live.error_stop(s, None)
        cmd = s.pilot.command(0.1, (0.0, 0.0), focused=True, frames_ok=True, link_ok=True, recognized=False, sector=None, dist_m=None, forward_m=None)
        self.assertEqual((cmd.x, cmd.z), (0.0, 0.0))
        self.assertTrue(cmd.stop)


if __name__ == "__main__":
    unittest.main()