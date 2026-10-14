"""«ЗАМЕР ДО» / «ЗАМЕР ПОСЛЕ»: calibration wait, phases, cancel, verdict, layout."""

import os
import re
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from sim import recog_measure as rm
from sim.trainer_journal import Journal

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}  ")


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def run(m, clock, log, bez, s, step=0.5):
    """Drive one whole measurement. ``bez`` during the wait arms the clock."""
    line = None
    end = clock.t + 2 * rm.PHASE_S + rm.PLACE_S + 2.0
    while clock.t < end and line is None and m.running:
        name, _ = m.phase()
        if name in ("wait", "bez"):
            value = bez
        elif name == "place":
            value = 99.0
        else:
            value = s
        line = m.tick(value, log.append)
        clock.t += step
    return line


class MeasureTest(unittest.TestCase):
    def test_phases_and_result(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        self.assertTrue(m.press("before", log.append))
        self.assertEqual(m.prompt(), "Убери .128 из кадра")
        self.assertEqual(m.button_text("before"), "ЗАМЕР ДО")
        self.assertEqual(m.button_text("after"), "ЗАМЕР ПОСЛЕ")
        m.tick(0.10, log.append)
        self.assertEqual(m.phase()[0], "bez")
        self.assertEqual(m.prompt(), "БЕЗ .128: 15 с")
        clock.t += 15.0
        self.assertEqual(m.phase()[0], "place")
        self.assertEqual(m.prompt(), "ПОСТАВЬ .128 В КАДР: 5 с")
        clock.t += 5.0
        self.assertEqual(m.prompt(), "С .128: 15 с")
        clock.t = 100.0
        m2 = rm.Measure(clock)
        m2.press("before", log.append)
        line = run(m2, clock, log, 0.10, 0.40)
        self.assertEqual(line, "замер ДО: без=10% с=40%")
        self.assertIn("узнаёт", log)
        self.assertFalse(m2.running)
        self.assertEqual(
            m2.line(),
            "до: без %s · с %s   после: без %s · с %s   узнаёт"
            % (rm._pct(0.10), rm._pct(0.40), rm._pct(None), rm._pct(None)),
        )
        self.assertEqual(len(rm._pct(0.10)), len(rm._pct(None)))
        self.assertNotIn("99", "".join(log))

    def test_step_sequence_fake_clock(self):
        """Wait for calibration, then без / пауза / с, with stamped journal lines."""
        clock = Clock()
        journal = Journal(None, clock=lambda: 1700000000.25)
        logged = []

        def log(text):
            logged.append(journal.write(text))

        m = rm.Measure(clock)
        self.assertTrue(m.press("before", log))
        self.assertEqual(m.prompt(), "Убери .128 из кадра")
        for _ in range(4):
            clock.t += 3.0
            self.assertIsNone(m.tick(None, log))
        self.assertEqual(m.phase()[0], "wait")
        self.assertEqual(m.prompt(), "жду калибровку борта…")
        self.assertEqual(logged.count(next(x for x in logged if "жду калибровку" in x)), 1)
        self.assertEqual(m.bez, [])

        self.assertIsNone(m.tick(0.10, log))
        self.assertEqual(m.prompt(), "БЕЗ .128: 15 с")
        clock.t += 7.5
        self.assertEqual(m.prompt(), "БЕЗ .128: 8 с")
        m.tick(0.20, log)
        clock.t += 7.5
        self.assertEqual(m.phase()[0], "place")
        self.assertTrue(m.prompt().startswith("ПОСТАВЬ .128 В КАДР:"))
        m.tick(99.0, log)
        clock.t += 5.0
        self.assertEqual(m.phase()[0], "s")
        self.assertTrue(m.prompt().startswith("С .128:"))
        m.tick(0.50, log)
        clock.t += 15.0
        line = m.tick(0.50, log)
        self.assertEqual(line, "замер ДО: без=15% с=50%")
        self.assertEqual(m.prompt(), "")
        self.assertFalse(m.running)
        self.assertIn("узнаёт", m.line())

        texts = [STAMP.sub("", row) for row in logged]
        self.assertTrue(all(STAMP.match(row) for row in logged))
        for needle in (
            "замер ДО: старт убери .128",
            "замер ДО: жду калибровку борта",
            "замер ДО: конец убери .128",
            "замер ДО: старт без .128",
            "замер ДО: конец без .128 среднее=15%",
            "замер ДО: старт поставь .128",
            "замер ДО: конец поставь .128",
            "замер ДО: старт с .128",
            "замер ДО: конец с .128 среднее=50%",
            "замер ДО: без=15% с=50%",
            "узнаёт",
        ):
            self.assertIn(needle, texts, needle)
        self.assertLess(texts.index("замер ДО: старт без .128"), texts.index("замер ДО: старт поставь .128"))
        self.assertLess(texts.index("замер ДО: конец поставь .128"), texts.index("замер ДО: старт с .128"))

    def test_cancel_same_button(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("after", log.append)
        clock.t += 3
        m.tick(0.2, log.append)
        self.assertFalse(m.press("after", log.append))
        self.assertIn("замер отменён", log)
        self.assertFalse(m.running)
        self.assertEqual(m.prompt(), "")
        self.assertEqual(m.button_text("after"), "ЗАМЕР ПОСЛЕ")
        self.assertNotIn("after", m.results)

    def test_verdicts(self):
        for b, a, word in (
            ((0.10, 0.40), (0.12, 0.60), "узнаёт"),
            ((0.10, 0.40), (0.60, 0.70), "выучил фон"),
            ((0.10, 0.40), (0.20, 0.25), "нет разницы"),
            ((0.10, 0.40), (0.55, 0.90), "выучил фон"),
        ):
            clock, log = Clock(), []
            m = rm.Measure(clock)
            m.press("before", log.append)
            run(m, clock, log, *b)
            m.press("after", log.append)
            run(m, clock, log, *a)
            self.assertEqual(log[-1], word, (b, a, log))
            self.assertIn(word, m.line())

    def test_no_samples_while_calibrating(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("before", log.append)
        line = run(m, clock, log, None, None)
        self.assertIsNone(line)
        self.assertTrue(m.running)
        self.assertEqual(m.phase()[0], "wait")
        self.assertEqual(m.prompt(), "жду калибровку борта…")
        self.assertNotIn("before", m.results)

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


class MeasureLayoutTest(unittest.TestCase):
    def test_block_visible_at_1024_and_1440(self):
        import pygame

        from sim.train_monitor import TrainMonitor, MonitorView
        from sim.ui_theme import apply_theme

        pygame.init()
        mon = TrainMonitor("measure-layout")
        view = MonitorView(
            learner="mb",
            learning_on=True,
            yolo_errors=2,
            yolo_state="смотрит",
            camera=__import__("numpy").zeros((90, 160, 3), dtype="uint8"),
            measure_run="before",
            measure_prompt="БЕЗ .128: 12 с",
            measure_before="ЗАМЕР ДО",
            measure_after="ЗАМЕР ПОСЛЕ",
            measure_line="до: без  10% · с  40%   после: без   — · с   —   узнаёт",
        )
        controls = (
            "learn_rect",
            "auto_rect",
            "take_rect",
            "steer_rect",
            "treat_rect",
            "punish_rect",
            "estop_rect",
        )
        robot = (
            "stand_up_rect",
            "stand_down_rect",
            "recovery_rect",
            "lidar_reset_rect",
            "brain_restart_rect",
            "reset_weights_rect",
        )
        try:
            for theme in ("light", "dark"):
                apply_theme(theme)
                mon._drop_surface_caches()
                for ww, wh in ((1024, 900), (1440, 900)):
                    mon._on_resize(ww, wh)
                    mon.scroll_x = 0
                    mon.scroll_y = 0
                    mon.draw(view)
                    host_c = mon._content_of("controls")
                    host_r = mon._content_of("robot")
                    cam = mon.cam_rect
                    for name in controls:
                        rect = getattr(mon, name)
                        self.assertGreaterEqual(rect.w, 8, (theme, ww, name, rect))
                        self.assertGreaterEqual(rect.h, 20, (theme, ww, name, rect))
                        self.assertTrue(host_c.contains(rect), (theme, ww, name, rect, host_c))
                    for name in robot:
                        rect = getattr(mon, name)
                        self.assertGreaterEqual(rect.w, 8, (theme, ww, name, rect))
                        self.assertGreaterEqual(rect.h, 20, (theme, ww, name, rect))
                        self.assertTrue(host_r.contains(rect), (theme, ww, name, rect, host_r))
                    for name in ("meas_before_rect", "meas_after_rect", "meas_info_rect", "measure_prompt_rect"):
                        rect = getattr(mon, name)
                        self.assertGreaterEqual(rect.w, 8, (theme, ww, name, rect))
                        self.assertGreaterEqual(rect.h, 16, (theme, ww, name, rect))
                        self.assertTrue(cam.contains(rect), (theme, ww, name, rect, cam))
                    self.assertEqual(mon.meas_before_rect.y, mon.meas_after_rect.y)
                    self.assertLess(mon.meas_before_rect.right, mon.meas_after_rect.x)
                    self.assertGreater(mon.measure_prompt_rect.h, 28, (theme, ww))
        finally:
            apply_theme("light")
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
