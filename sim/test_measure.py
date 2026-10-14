"""«ЗАМЕР ДО» / «ЗАМЕР ПОСЛЕ»: phases, cancel, result lines, verdict, ОБУЧЕНИЕ/АВТО off."""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from sim import recog_measure as rm


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def run(m, clock, log, bez, s, step=0.5):
    """Drive one whole measurement with constant УЗНАЮ values."""
    line = None
    end = clock.t + 2 * rm.PHASE_S + rm.PLACE_S + 1.0
    while clock.t < end and line is None:
        name, _ = m.phase()
        line = m.tick(bez if name == "bez" else (99.0 if name == "place" else s), log.append)
        clock.t += step
    return line


class MeasureTest(unittest.TestCase):
    def test_phases_and_result(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        self.assertTrue(m.press("before", log.append))
        self.assertEqual(m.button_text("before"), "без .128: 15 с")
        self.assertEqual(m.button_text("after"), "ЗАМЕР ПОСЛЕ")
        clock.t += 15.5
        self.assertEqual(m.phase()[0], "place")
        self.assertEqual(m.button_text("before"), "поставь .128:  5 с")
        clock.t += 5.0
        self.assertEqual(m.button_text("before"), "с .128: 15 с")
        clock.t = 100.0
        line = run(m, clock, log, 0.10, 0.40)
        self.assertEqual(line, "замер ДО: без=0.10 с=0.40 разница=+0.30")
        self.assertIn("поставь .128 в кадр", log)
        self.assertEqual(log.count("поставь .128 в кадр"), 1)
        self.assertFalse(m.running)
        self.assertEqual(m.lines()[0], "до:    без 0.10 · с 0.40")
        self.assertFalse(any(l.startswith("итог") for l in log))

    def test_cancel_same_button(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("after", log.append)
        clock.t += 3
        self.assertFalse(m.press("after", log.append))
        self.assertIn("замер отменён", log)
        self.assertFalse(m.running)
        self.assertEqual(m.button_text("after"), "ЗАМЕР ПОСЛЕ")
        self.assertNotIn("after", m.results)

    def test_verdicts(self):
        for b, a, word in (
            ((0.10, 0.30), (0.12, 0.60), "итог: узнаёт"),
            ((0.10, 0.30), (0.40, 0.70), "итог: выучил фон"),
            ((0.10, 0.30), (0.10, 0.31), "итог: без изменений"),
            ((0.10, 0.30), (0.18, 0.60), "итог: без изменений"),
        ):
            clock, log = Clock(), []
            m = rm.Measure(clock)
            m.press("before", log.append)
            run(m, clock, log, *b)
            m.press("after", log.append)
            run(m, clock, log, *a)
            self.assertEqual(log[-1], word, (b, a, log))

    def test_no_samples(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("before", log.append)
        run(m, clock, log, None, None)
        self.assertEqual(log[-1], "замер ДО: без=— с=— разница=—")

    def test_press_turns_learning_and_auto_off_only(self):
        logs = []
        session = SimpleNamespace(learn=True, pilot=SimpleNamespace(autonomy=True), onboard=False, _log=logs.append)
        inp = SimpleNamespace(measure_before=True, measure_after=False, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn and inp.autonomy_toggle)
        session.learn, session.pilot.autonomy = False, False
        inp = SimpleNamespace(measure_before=False, measure_after=True, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)  # switch to ПОСЛЕ: training stays off
        self.assertFalse(inp.pause_learn or inp.autonomy_toggle)
        inp = SimpleNamespace(measure_before=False, measure_after=True, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)  # cancel
        self.assertFalse(inp.pause_learn or inp.autonomy_toggle)
        self.assertIn("замер отменён", logs)

    def test_onboard_value(self):
        s = SimpleNamespace(onboard=True, remote={"confidence": 42.0, "confidence_ready": True})
        self.assertAlmostEqual(rm.uznayu_value(s), 0.42)
        s.remote["confidence_ready"] = False
        self.assertIsNone(rm.uznayu_value(s))


if __name__ == "__main__":
    unittest.main()
