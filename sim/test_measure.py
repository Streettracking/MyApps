"""«БЕЗ .128» / «С .128»: 15 s average, cancel, board gate, latest line, layout."""

import os
import re
import tempfile
import unittest
from pathlib import Path
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


def finish(m, clock, log, value, step=0.5):
    """Run the open measurement to its 15 s result. No pause is inserted."""
    line = None
    end = clock.t + rm.PHASE_S + 1.0
    while clock.t < end and line is None and m.running:
        line = m.tick(value, log, board=(200, 200))
        if line is None:
            clock.t += step
    return line


class MeasureTest(unittest.TestCase):
    def test_fifteen_seconds_and_latest_line(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        self.assertEqual(m.press("bez", log.append), "start")
        self.assertEqual(m.prompt(), "БЕЗ .128: 15 с")
        self.assertEqual(m.button_text("bez"), "БЕЗ .128  15 с")
        self.assertEqual(m.button_text("s"), "С .128")
        m.tick(0.10, log.append)
        clock.t += 3.0
        self.assertEqual(m.prompt(), "БЕЗ .128: 12 с")
        self.assertEqual(m.button_text("bez"), "БЕЗ .128  12 с")
        line = finish(m, clock, log.append, 0.10)
        self.assertEqual(line, "замер без .128: 10%  без  10% · с    — · разница    —")
        self.assertFalse(m.running)
        self.assertEqual(m.prompt(), "")
        self.assertEqual(m.line(), "без  10% · с    — · разница    —")
        self.assertNotIn("узнаёт", "".join(log))
        self.assertNotIn("ПОСТАВЬ", "".join(log))
        self.assertFalse(hasattr(rm, "PLACE_S"))

        self.assertEqual(m.press("s", log.append), "start")
        self.assertEqual(m.prompt(), "С .128: 15 с")
        line = finish(m, clock, log.append, 0.40)
        self.assertIn("замер с .128: 40%", line)
        self.assertEqual(m.line(), "без  10% · с  40% · разница +30%")

        # A new «БЕЗ» replaces only that side. The previous «С» stays.
        self.assertEqual(m.press("bez", log.append), "start")
        line = finish(m, clock, log.append, 0.20)
        self.assertEqual(m.line(), "без  20% · с  40% · разница +20%")
        self.assertNotIn("до:", m.line())
        self.assertNotIn("после:", m.line())
        self.assertEqual(len(m.lines()), 1)

    def test_step_is_one_button_with_stamped_journal(self):
        clock = Clock()
        journal = Journal(None, clock=lambda: 1700000000.25)
        logged = []

        def log(text):
            logged.append(journal.write(text))

        m = rm.Measure(clock)
        self.assertEqual(m.press("bez", log, board=(12, 200)), "blocked")
        self.assertEqual(m.prompt(), "борт не готов: калибровка 12/200")
        self.assertFalse(m.running)
        self.assertEqual(m.latest, {})
        for _ in range(4):
            clock.t += 3.0
            self.assertIsNone(m.tick(0.9, log, board=(12, 200)))
        self.assertEqual(m.prompt(), "борт не готов: калибровка 12/200")
        self.assertEqual(logged.count(next(x for x in logged if "борт не готов" in x)), 1)

        clock.t += 1.0
        self.assertIsNone(m.tick(0.9, log, board=(200, 200)))
        self.assertEqual(m.prompt(), "")

        self.assertEqual(m.press("s", log, board=(200, 200)), "start")
        self.assertEqual(m.prompt(), "С .128: 15 с")
        clock.t += 7.5
        self.assertEqual(m.prompt(), "С .128: 8 с")
        self.assertEqual(m.button_text("s"), "С .128  8 с")
        m.tick(0.50, log)
        clock.t += 7.5
        line = m.tick(0.50, log)
        self.assertIn("замер с .128: 50%", line)
        self.assertIn("разница", line)
        self.assertEqual(m.prompt(), "")
        self.assertFalse(m.running)

        texts = [STAMP.sub("", row) for row in logged]
        self.assertTrue(all(STAMP.match(row) for row in logged))
        self.assertIn("замер без .128: борт не готов: калибровка 12/200", texts)
        self.assertIn("замер с .128: старт", texts)
        self.assertTrue(any(row.startswith("замер с .128: 50%") for row in texts))
        self.assertNotIn("замер отменён", texts)

    def test_journal_file_gets_the_result(self):
        clock = Clock()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.log"
            journal = Journal(path, clock=lambda: 1700000000.25)
            m = rm.Measure(clock)
            m.press("bez", journal.write)
            clock.t += 15.0
            m.tick(0.15, journal.write)
            text = path.read_text(encoding="utf-8")
        self.assertRegex(text, STAMP.pattern.replace("^", "").replace("  ", "  "))
        self.assertIn("замер без .128: 15%", text)
        self.assertTrue(text.strip().splitlines()[-1].endswith("без  15% · с    — · разница    —"))

    def test_cancel_same_button_keeps_the_previous_value(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("bez", log.append)
        finish(m, clock, log.append, 0.10)
        m.press("s", log.append)
        clock.t += 4.0
        m.tick(0.99, log.append)
        self.assertEqual(m.press("s", log.append), "cancel")
        self.assertIn("замер отменён", log)
        self.assertFalse(m.running)
        self.assertEqual(m.prompt(), "")
        self.assertEqual(m.button_text("s"), "С .128")
        self.assertEqual(m.line(), "без  10% · с    — · разница    —")
        self.assertNotIn("99%", "".join(log))

    def test_other_button_cancels_without_a_pause(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        m.press("bez", log.append)
        finish(m, clock, log.append, 0.10)
        m.press("bez", log.append)
        clock.t += 2.0
        m.tick(0.99, log.append)
        self.assertEqual(m.press("s", log.append), "switch")
        self.assertEqual(m.which, "s")
        self.assertEqual(m.samples, [])
        self.assertEqual(m.prompt(), "С .128: 15 с")
        self.assertEqual(m.line(), "без  10% · с    — · разница    —")
        line = finish(m, clock, log.append, 0.40)
        self.assertIn("замер с .128: 40%", line)
        self.assertEqual(m.line(), "без  10% · с  40% · разница +30%")
        self.assertEqual(log.count("замер отменён"), 1)

    def test_not_ready_does_not_sample(self):
        clock, log = Clock(), []
        m = rm.Measure(clock)
        self.assertEqual(m.press("bez", log.append, board=(4, 200)), "blocked")
        for _ in range(6):
            clock.t += 3.0
            self.assertIsNone(m.tick(0.9, log.append, board=(4, 200)))
        self.assertFalse(m.running)
        self.assertEqual(m.prompt(), "борт не готов: калибровка 4/200")
        self.assertEqual(m.latest, {})
        self.assertEqual(m.line(), "без    — · с    — · разница    —")

    def test_restores_learn_and_auto_and_reads_board_status(self):
        """Board calib, not the trainer counter. Learn and auto come back after end or cancel."""
        logs = []
        trainer = SimpleNamespace(calib_progress=lambda: (200, 200), conf=SimpleNamespace(ready=True, percent=90.0))
        session = SimpleNamespace(
            learn=True,
            pilot=SimpleNamespace(autonomy=True),
            onboard=True,
            mb=trainer,
            remote={
                "calib_n": 40,
                "calib_total": 200,
                "confidence_calib_n": 12,
                "confidence_calib_need": 80,
                "confidence": 90.0,
                "confidence_ready": True,
                "learning": True,
                "autonomy": True,
            },
            _log=logs.append,
        )
        self.assertEqual(rm.board_calib(session), ("window", 12, 80, False))
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "калибровка 12/80")
        self.assertNotIn("40/200", rm.board_phrase(rm.board_calib(session)))
        self.assertNotEqual(rm.board_calib(session), trainer.calib_progress())

        inp = SimpleNamespace(measure_before=True, measure_after=False, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertFalse(inp.pause_learn or inp.autonomy_toggle)
        m = session._measure
        self.assertFalse(m.running)
        self.assertIsNone(m.saved)
        self.assertEqual(m.prompt(), "борт не готов: калибровка 12/80")

        session.remote["confidence_calib_n"] = 80
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "калибровка 80/80")
        self.assertTrue(rm.board_ready(rm.board_calib(session)))
        inp = SimpleNamespace(measure_before=True, measure_after=False, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn and inp.autonomy_toggle)
        session.remote["learning"] = False
        session.remote["autonomy"] = False
        self.assertEqual(m.saved, {"learn": True, "auto": True, "teacher": False})
        self.assertEqual(m.which, "bez")

        # The other button keeps them off and keeps the original snapshot.
        inp = SimpleNamespace(measure_before=False, measure_after=True, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertFalse(inp.pause_learn or inp.autonomy_toggle)
        self.assertEqual(m.saved, {"learn": True, "auto": True, "teacher": False})
        self.assertEqual(m.which, "s")

        inp = SimpleNamespace(measure_before=False, measure_after=True, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn and inp.autonomy_toggle)
        self.assertIsNone(m.saved)
        self.assertIn("замер отменён", logs)

        # A finished run asks for the same restore on the next key pass.
        # Only ОБУЧЕНИЕ was on; АВТО stays off.
        session.remote["learning"] = True
        session.remote["autonomy"] = False
        session.remote["confidence_calib_n"] = 80
        session._measure = rm.Measure(Clock())
        inp = SimpleNamespace(measure_before=True, measure_after=False, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn)
        self.assertFalse(inp.autonomy_toggle)
        session.remote["learning"] = False
        done = session._measure
        clock = done.clock
        line = finish(done, clock, logs.append, 0.10)
        self.assertIsNotNone(line)
        self.assertTrue(done.pending_restore)
        self.assertEqual(done.line(), "без  10% · с    — · разница    —")
        inp = SimpleNamespace(measure_before=False, measure_after=False, pause_learn=False, autonomy_toggle=False)
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn)
        self.assertFalse(inp.autonomy_toggle)
        self.assertIsNone(done.saved)

        session.onboard = False
        session.remote = {"calib_n": 3, "calib_total": 200}
        trainer.calib_progress = lambda: (3, 200)
        self.assertEqual(rm.board_calib(session), ("flag", True))
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "борт: готов")
        view = SimpleNamespace()
        rm.measure_frame(session, view)
        self.assertEqual(view.measure_board, "борт: готов")
        self.assertNotIn("3/200", view.measure_board)
        self.assertEqual(view.measure_before, "БЕЗ .128")
        self.assertEqual(view.measure_after, "С .128")
        session.onboard = True
        session.remote = {"calib_n": 7, "calib_total": 200, "confidence_ready": False}
        rm.measure_frame(session, view)
        self.assertEqual(view.measure_board, "борт: не готов")
        self.assertNotIn("7/200", view.measure_board)
        self.assertNotIn("3/200", view.measure_board)
        inp = SimpleNamespace(measure_before=True, measure_after=False, pause_learn=False, autonomy_toggle=False)
        session._measure = rm.Measure(Clock())
        session.remote["learning"] = True
        session.remote["autonomy"] = False
        rm.drive_measure(session, inp)
        self.assertTrue(inp.pause_learn)
        self.assertTrue(session._measure.running)

    def test_eyes_must_fill_and_old_board_does_not_block(self):
        session = SimpleNamespace(
            onboard=True,
            mb=SimpleNamespace(calib_progress=lambda: (1, 200)),
            remote={
                "calib_n": 1,
                "confidence_calib_n_L": 5,
                "confidence_calib_need_L": 20,
                "confidence_calib_n_R": 20,
                "confidence_calib_need_R": 20,
                "confidence_ready": True,
            },
            _log=lambda _text: None,
        )
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "калибровка 5/20")
        self.assertFalse(rm.board_ready(rm.board_calib(session)))
        session.remote["confidence_calib_n_L"] = 20
        self.assertEqual(rm.board_calib(session), ("window", 20, 20, True))
        self.assertTrue(rm.board_ready(rm.board_calib(session)))

        session.remote = {"confidence_ready": False, "calib_n": 4, "calib_total": 200}
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "борт: не готов")
        self.assertTrue(rm.board_ready(rm.board_calib(session)))
        session.remote["confidence_ready"] = True
        self.assertEqual(rm.board_phrase(rm.board_calib(session)), "борт: готов")

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
            measure_run="bez",
            measure_prompt="БЕЗ .128: 12 с",
            measure_before="БЕЗ .128  12 с",
            measure_after="С .128",
            measure_board="готов",
            measure_line="без  18% · с  61% · разница +43%",
        )
        controls = (
            "learn_rect",
            "auto_rect",
            "take_rect",
            "steer_rect",
            "treat_rect",
            "punish_rect",
            "yolo_err_rect",
            "neg_btn_rect",
            "kin_btn_rect",
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
                    self.assertGreaterEqual(mon.meas_before_rect.w, mon.font_sm.size("БЕЗ .128  12 с")[0] + 32, (theme, ww))
        finally:
            apply_theme("light")
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
