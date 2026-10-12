"""Onboard loop: mushroom body, pilot, and the laptop control port.

Autonomy and learning start off. That is watch-only: this process does
not call SportClient at all. StopMove is one edge, and only after this
process itself sent Move (a link or frame loss, or the step from moving
to stopped). E-STOP sends StopMove once and then blocks. StandUp,
StandDown, and RecoveryStand run only when the operator asks.

StandUp locks the Go2. BalanceStand follows about 0.7 s later, from
tick, so the HTTP handler and the main loop do not sleep. The first
Move after a takeover or an autonomy start sends that unlock itself
when the timer has not fired yet.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from sim.brain_panel import PanelTrace, widget_status
from sim.frame_sense import features_from_frames
from sim.learn_flash import flash_payload
from sim.map_marks import MarkLayer
from sim.mb_train import PAUSE_GRACE_S, MbTrainer, default_npz
from sim.hemifield import DEFAULT_OVERLAP, live_turn
from sim.run_marker import RunMarker
from sim.raw_sense import recog_label
from sim.pilot import LINK_HOLD_S, MANUAL_HOLD_S, DriveCommand, Pilot, eyes_state_label, phase_label
from sim.pilot import forward_clearance, scrub_range
from sim.teacher_lead import (
    AIM_HOLD_S,
    TeacherLead,
    apply_guards,
    attach_recognition_stimuli,
    clamp_weight,
    commands_match,
    lead_label,
    mix_velocity,
    observe_recognition,
)

try:
    from .drive import SportDrive
    from .sensors import cloud_forward
except ImportError:  # python3 /root/flybrain/main.py has no package parent
    from drive import SportDrive
    from sensors import cloud_forward


class BrainLoop:
    def __init__(self, drive: SportDrive, mb: MbTrainer, state_path: str, self_radius: float = 0.6):
        self.drive = drive
        self.mb = mb
        self.state_path = state_path
        self.self_radius = float(self_radius)
        self.pilot = Pilot(self_radius=self.self_radius)
        self.learn = False
        self.mb.learn = False
        self.paused = False
        self.external_stop = False
        self._link_stop = False
        self.marks = MarkLayer()
        self.log = deque(maxlen=16)
        self.run_marker = RunMarker()
        self.t0 = time.monotonic()
        self.last_hb = self.t0
        self.last_frame_t = 0.0
        self.last_manual_t = 0.0
        self.manual_x = 0.0
        self.manual_z = 0.0
        self.frames_ok = False
        self.recognized = False
        self.sector = None
        self.dist_m = None
        self.forward_m = None
        self.readout = 0.0
        self.confidence = 0.0
        self.confidence_ready = False
        self.recognized_l = False
        self.recognized_r = False
        self.confidence_l = 0.0
        self.confidence_r = 0.0
        self.confidence_l_ready = False
        self.confidence_r_ready = False
        self.cloud_ranges = None
        self.cloud_t = 0.0
        self._teach = None
        self._teach_l = None
        self._teach_r = None
        self._moving = False
        self._sent_move = False
        self._last_send = 0.0
        self._move_log_t = -1e9
        self._prev_camera = None
        self._prev_near = None
        self._learn_block_log = 0.0
        self.teacher_weight = 0.0
        self.fly_x = 0.0
        self.fly_z = 0.0
        self.teacher_x = 0.0
        self.teacher_z = 0.0
        self.lead_match = 0.0
        self.lead = TeacherLead()
        attach_recognition_stimuli(self.mb.stimuli)
        self._taught_l: str | None = None
        self._taught_r: str | None = None
        self._pam_event = False
        self._ppl1_event = False
        self._yolo_l = None
        self._yolo_r = None
        self._yolo_t = -1e9
        self._dog_in_frame = False
        self.searching = False
        self.panel = PanelTrace()
        self._lock = threading.Lock()
        self._log("наблюдение: SportClient не трогаем, пока нет автономии или перехвата")

    def _log(self, text: str) -> None:
        self.log.append(f"{time.monotonic() - self.t0:6.1f}s  {text}")

    def note_heartbeat(self, now: float) -> None:
        self.last_hb = float(now)

    def note_manual(self, x: float, z: float, now: float) -> None:
        self.manual_x = float(x)
        self.manual_z = float(z)
        self.last_manual_t = float(now)

    def set_cloud(self, ranges, now: float) -> None:
        self.cloud_ranges = ranges
        self.cloud_t = float(now)

    def _note_dan(self, *kinds: str | None) -> None:
        """Remember which dopamine arrived on the current frame."""
        for kind in kinds:
            if kind == "pam":
                self._pam_event = True
            elif kind == "ppl1":
                self._ppl1_event = True

    def _group_counts(self, kind: str) -> tuple:
        """Active cells and group size for γ-d or α/β-p, both hemispheres."""
        on = 0
        total = 0
        for vision in (self.mb.brain.vision, self.mb.brain_r.vision):
            if kind == "gd":
                on += int(vision.last_gd_on)
                total += int(vision.n_gd)
            else:
                on += int(vision.last_abp_on)
                total += int(vision.n_abp)
        return on, total

    def _kc_active_frac(self) -> float:
        """Active visual Kenyon cells over both groups on the last frame."""
        gd_on, gd_n = self._group_counts("gd")
        ab_on, ab_n = self._group_counts("abp")
        total = gd_n + ab_n
        if total <= 0:
            return 0.0
        return float(gd_on + ab_on) / float(total)

    def _kc_active_frac_group(self, kind: str) -> float:
        on, total = self._group_counts(kind)
        if total <= 0:
            return 0.0
        return float(on) / float(total)

    def driving(self) -> bool:
        """Autonomy, or manual after an explicit takeover. Watch-only is neither."""
        if self.pilot.mode == "auto":
            return True
        return self.pilot.mode == "manual" and bool(self.pilot.took_over)

    def _stop_once(self) -> None:
        """One StopMove if this process has a Move outstanding. Otherwise silence."""
        if not self._sent_move:
            return
        self.drive.stop()
        self._sent_move = False
        self._moving = False

    def halt(self) -> None:
        self._stop_once()

    def on_estop(self, now: float) -> None:
        """One StopMove, then a block. A repeat while the block holds sends nothing."""
        if self.pilot.mode == "estop":
            return
        self.pilot.estop()
        self.drive.cancel_balance_timer()
        self.drive.stop()
        self._sent_move = False
        self._moving = False
        self._refresh_hold(float(now), False)
        self._log("E-STOP")

    def on_space(self, now: float) -> None:
        self._stop_once()
        self.pilot.space()
        self._refresh_hold(float(now), False)
        self._log("стоп")

    def _stop_latched(self, now: float) -> bool:
        """True while an E-STOP, a pause, a space-stop, or an external stop holds.

        A lost link or lost frames while this process is driving is an external
        stop too. It clears once both are back.
        """
        link_ok = (float(now) - self.last_hb) <= LINK_HOLD_S
        frames_ok = self.frames_ok and (float(now) - self.last_frame_t) <= LINK_HOLD_S
        if self.driving() and (not link_ok or not frames_ok):
            self._link_stop = True
        elif link_ok and frames_ok:
            self._link_stop = False
        return bool(
            self.paused
            or self.external_stop
            or self._link_stop
            or self.pilot.mode == "estop"
            or self.pilot.held_stop
        )

    def _refresh_hold(self, now: float, moving: bool) -> None:
        """Freeze dopamine while a stop holds, and for PAUSE_GRACE_S after motion."""
        self.mb.note_motion_hold(self._stop_latched(now), bool(moving), float(now))

    def learning_frozen(self) -> bool:
        return bool(self.mb.learning_hold)

    def on_takeover(self) -> None:
        self._stop_once()
        self.pilot.takeover()
        self._log("перехват: ручное, автономия сама не вернётся")

    def handle(self, data: dict, now: float) -> None:
        self.note_heartbeat(now)
        op = str(data.get("op", ""))
        with self._lock:
            if op == "autonomy_on":
                if self.pilot.start_auto(now):
                    self._log("автономия: поиск в одну сторону, z=+0.35, без качки")
                    self._log("руль: %s" % ("билатерально, z от R_L−R_R" if self.pilot.steer == "bilateral" else "секторы"))
                    self._log("сырой сектор — мозг; sector_smooth — медиана контроллера")
                else:
                    self._log("E-STOP держит стоп. M переводит в ручное")
            elif op == "autonomy_off":
                self._stop_once()
                self.pilot.stop_auto()
                self._log("автономия выключена")
            elif op == "learn_on":
                self.learn = True
                self.mb.learn = True
                self._log("обучение включено")
            elif op == "learn_off":
                self.learn = False
                self.mb.learn = False
                self._log("обучение выключено, T/X веса не меняют")
            elif op == "reinforcement_off":
                off = bool(data.get("off", True))
                self.mb.set_reinforcement_off(off)
                if off:
                    self._log("подкрепление выключено, пластичность заморожена")
                else:
                    self._log("подкрепление включено")
            elif op == "treat":
                self._teach = "pam"
                self._teach_l = None
                self._teach_r = None
            elif op == "punish":
                self._teach = "ppl1"
                self._teach_l = None
                self._teach_r = None
            elif op == "teach_sides":
                if self._teach is None:
                    left = str(data.get("left") or "")
                    right = str(data.get("right") or "")
                    self._teach_l = left if left in ("pam", "ppl1") else None
                    self._teach_r = right if right in ("pam", "ppl1") else None
            elif op == "estop":
                self.on_estop(now)
            elif op == "pause":
                if not self.paused:
                    self.paused = True
                    self._stop_once()
                    self._log("пауза: ход ноль, пластичность заморожена")
                self._refresh_hold(now, False)
            elif op == "resume":
                if self.paused:
                    self.paused = False
                    self._log("ход разрешён, обучение заморожено до движения и ещё %.0f с" % PAUSE_GRACE_S)
                self._refresh_hold(now, False)
            elif op == "external_stop":
                on = bool(data.get("on", True))
                if on and not self.external_stop:
                    self.external_stop = True
                    self._stop_once()
                    self._log("внешний стоп: ход ноль, пластичность заморожена")
                elif not on and self.external_stop:
                    self.external_stop = False
                    self._log("внешний стоп снят")
                else:
                    self.external_stop = on
                self._refresh_hold(now, False)
            elif op == "space":
                self.on_space(now)
            elif op == "takeover":
                self.on_takeover()
            elif op == "stand_up":
                code = self.drive.stand_up(now)
                self._log(f"StandUp code={code}")
            elif op == "stand_down":
                code = self.drive.stand_down()
                self._log(f"StandDown code={code}")
            elif op == "recovery_stand":
                if self.drive.recovery_stand():
                    self._log("RecoveryStand")
                else:
                    self._log("RecoveryStand нет в SDK")
            elif op == "save":
                self.mb.lidar_refresh = 1.5
                self.mb.save(self.state_path)
                self._log("сохранено")
            elif op == "manual":
                if self.driving():
                    self.note_manual(float(data.get("x", 0.0)), float(data.get("z", 0.0)), now)
            elif op == "steer":
                mode = str(data.get("mode", ""))
                if mode in ("bilateral", "sectors"):
                    self.pilot.set_steer(mode)
                    self._log("руль: %s" % ("билатерально" if mode == "bilateral" else "секторы"))
            elif op == "teacher_weight":
                self.teacher_weight = clamp_weight(data.get("weight", 0.0))
                self.lead.weight = self.teacher_weight
            elif op == "teacher_aim":
                self.teacher_weight = clamp_weight(data.get("weight", self.teacher_weight))
                self.lead.weight = self.teacher_weight
                self.lead.set_remote(
                    now,
                    data.get("x", 0.0),
                    data.get("z", 0.0),
                    bool(data.get("hit_l")),
                    bool(data.get("hit_r")),
                    self.teacher_weight,
                    centered=bool(data.get("centered")),
                    approaching=bool(data.get("approaching")),
                    lost=bool(data.get("lost")),
                    too_close=bool(data.get("too_close")),
                    last_l=bool(data.get("last_l")),
                    last_r=bool(data.get("last_r")),
                )
                self._yolo_l = _yolo_vec(data.get("yolo_l"))
                self._yolo_r = _yolo_vec(data.get("yolo_r"))
                self._yolo_t = float(now)
                self._dog_in_frame = (
                    self._yolo_l is not None
                    or self._yolo_r is not None
                    or bool(data.get("hit_l"))
                    or bool(data.get("hit_r"))
                )
            elif op == "run_marker":
                self.run_marker.set_stage(data.get("stage"))
                self.run_marker.set_side(data.get("side"))
            elif op == "heartbeat":
                pass

    def ingest(self, camera: np.ndarray, lidar: np.ndarray, now: float) -> None:
        """One fresh pair. Empty lidar after a reset must not be passed in."""
        feat, _ego, near = features_from_frames(camera, lidar, self._prev_camera, self._prev_near)
        self._prev_camera = camera
        self._prev_near = near
        self.frames_ok = True
        self.last_frame_t = float(now)
        self._pam_event = False
        self._ppl1_event = False
        self.mb.learn = self.learn
        self._refresh_hold(float(now), False)
        yolo_fresh = float(now) - self._yolo_t <= AIM_HOLD_S
        seen_now = bool(yolo_fresh and self._dog_in_frame)
        self.searching = self.pilot.mode == "auto" and not seen_now
        fwd = self.mb.forward(
            feat,
            yolo_l=self._yolo_l if yolo_fresh else None,
            yolo_r=self._yolo_r if yolo_fresh else None,
        )
        value = self.mb.last_readout
        conf = self.mb.observe(feat, value)
        seen = bool(conf.ready and conf.recognized)
        self.marks.consider(self.mb, feat, float(now), recognized=seen, mode="live")
        dist = None
        forward = None
        ranges = self.cloud_ranges
        if ranges is not None and float(now) - self.cloud_t < 0.5:
            if seen and self.marks.aim_sector is not None:
                dist = ranges[int(self.marks.aim_sector)]
            forward = cloud_forward(ranges)
        else:
            if seen:
                dist = self.marks.aim_dist
            forward = forward_clearance(feat, self.self_radius)
        dist = scrub_range(dist, self.self_radius)
        forward = scrub_range(forward, self.self_radius)
        kind = self._teach
        side_l = self._teach_l
        side_r = self._teach_r
        self._teach = None
        self._teach_l = None
        self._teach_r = None
        self._taught_l = None
        self._taught_r = None
        if self.mb.learning_hold or self.searching:
            kind = None
            side_l = None
            side_r = None
        if (kind or side_l or side_r) and not self.learn:
            if float(now) - self._learn_block_log > 1.0:
                self._log("обучение выключено, T/X веса не меняют")
                self._learn_block_log = float(now)
            kind = None
            side_l = None
            side_r = None
        pulsed_l = False
        pulsed_r = False
        if kind:
            self.mb.teach(fwd, kind, float(now))
            self._taught_l = kind
            self._taught_r = kind
            pulsed_l = kind in ("pam", "ppl1")
            pulsed_r = pulsed_l
            self._note_dan(kind)
        elif side_l or side_r:
            self.mb.teach_sides(side_l, side_r, float(now))
            self._taught_l = side_l
            self._taught_r = side_r
            pulsed_l = side_l in ("pam", "ppl1")
            pulsed_r = side_r in ("pam", "ppl1")
            self._note_dan(side_l, side_r)
        if not self.searching:
            self.mb.settle(pulsed_l, pulsed_r)
        with self._lock:
            self.recognized = seen
            self.sector = self.marks.aim_sector if seen else None
            self.dist_m = dist
            self.forward_m = forward
            self.readout = float(value)
            self.confidence = float(conf.percent)
            self.confidence_ready = bool(conf.ready)
            eye_l, eye_r = self.mb.eye_recognized()
            self.recognized_l = eye_l
            self.recognized_r = eye_r
            self.confidence_l = float(self.mb.conf_l.percent)
            self.confidence_r = float(self.mb.conf_r.percent)
            self.confidence_l_ready = bool(self.mb.conf_l.ready)
            self.confidence_r_ready = bool(self.mb.conf_r.ready)
            dog = (float(now) - self._yolo_t) <= AIM_HOLD_S and bool(self._dog_in_frame)
            self.panel.note(float(now), float(value), dog, self.mb)
            self.run_marker.note(float(now), float(self.mb.r_l), float(self.mb.r_r))

    def tick(self, now: float) -> DriveCommand:
        now = float(now)
        # Watch-only still finishes a stand the operator already asked for.
        # The HTTP thread only scheduled it.
        self._release_stand_lock(now)
        link_ok = (now - self.last_hb) <= LINK_HOLD_S
        frames_ok = self.frames_ok and (now - self.last_frame_t) <= LINK_HOLD_S
        recent_manual = (now - self.last_manual_t) <= MANUAL_HOLD_S
        if recent_manual:
            manual = (self.manual_x, self.manual_z)
        else:
            manual = (0.0, 0.0)
        with self._lock:
            recognized = self.recognized
            sector = self.sector
            dist_m = self.dist_m
            forward_m = self.forward_m
            r_l = float(self.mb.r_l)
            r_r = float(self.mb.r_r)
            recognized_l = self.recognized_l
            recognized_r = self.recognized_r
        if self.paused or self.external_stop or self.pilot.mode == "estop" or self.pilot.held_stop:
            self.pilot.track.note_sides(r_l, r_r, self.mb.base_l, self.mb.base_r)
            self._refresh_hold(now, False)
            self._stop_once()
            return DriveCommand(0.0, 0.0, True, "stop", "никто", "")
        if not self.driving():
            # Watch-only: no Move and no StopMove.
            # The fly's two readouts still update so the laptop can see them.
            self.pilot.track.note_sides(r_l, r_r, self.mb.base_l, self.mb.base_r)
            self._refresh_hold(now, False)
            return DriveCommand(0.0, 0.0, True, "stop", "никто", "")
        if self.pilot.mode != "auto":
            axes = manual
        elif abs(manual[0]) + abs(manual[1]) > 1e-6 and recent_manual:
            axes = manual
        else:
            axes = None
        if not link_ok or not frames_ok:
            # The sport service would keep our last Move for about a second.
            # One StopMove covers that. Further ticks stay silent.
            self.pilot.track.note_sides(r_l, r_r, self.mb.base_l, self.mb.base_r)
            self._refresh_hold(now, False)
            self._stop_once()
            self.pilot.phase = "stop"
            self.pilot.who = "никто"
            return DriveCommand(0.0, 0.0, True, "stop", "никто", "")
        cmd = self.pilot.command(
            now,
            (0.0, 0.0),
            focused=True,
            frames_ok=True,
            link_ok=True,
            recognized=recognized,
            sector=sector,
            dist_m=dist_m,
            forward_m=forward_m,
            manual_axes=axes,
            r_l=r_l,
            r_r=r_r,
            recognized_l=recognized_l,
            recognized_r=recognized_r,
            base_l=float(self.mb.base_l),
            base_r=float(self.mb.base_r),
        )
        from sim.search_reflex import cover_command

        seen = (now - self._yolo_t) <= AIM_HOLD_S and bool(self._dog_in_frame)
        if self.pilot.mode == "auto" and cmd.phase != "stop":
            cmd, self.searching = cover_command(
                cmd, seen, r_l, r_r, dist_m, forward_m, self.mb.base_l, self.mb.base_r
            )
            self.pilot.phase = cmd.phase
        elif self.pilot.mode != "auto":
            self.searching = False
        cmd = self._blend_teacher(cmd, now, forward_m)
        self._emit(cmd, now)
        return cmd

    def _command_will_move(self, cmd: DriveCommand, now: float, forward_m) -> bool:
        """True when the command that would leave this tick is not a full stop."""
        if self._stop_latched(now):
            return False
        speed, yaw = float(cmd.x), float(cmd.z)
        if self.pilot.mode == "auto" and self.teacher_weight > 0.005:
            teacher_x, teacher_z, _hit_l, _hit_r = self.lead.remote(now)
            speed, yaw = mix_velocity(cmd.x, cmd.z, teacher_x, teacher_z, self.teacher_weight)
            speed, yaw, _stopped = apply_guards(speed, yaw, self.pilot.mode == "estop", forward_m)
        return abs(float(speed)) > 1e-6 or abs(float(yaw)) > 1e-6

    def _blend_teacher(self, cmd: DriveCommand, now: float, forward_m) -> DriveCommand:
        """Keep the fly command at weight 0. Otherwise mix, then apply the stops."""
        self._refresh_hold(now, self._command_will_move(cmd, now, forward_m))
        teacher_x, teacher_z, hit_l, hit_r = self.lead.remote(now)
        self.fly_x = float(cmd.x)
        self.fly_z = float(cmd.z)
        self.teacher_x = float(teacher_x)
        self.teacher_z = float(teacher_z)
        self.lead.hit_l = bool(hit_l)
        self.lead.hit_r = bool(hit_r)
        self.lead.weight = float(self.teacher_weight)
        matched = commands_match(cmd.x, cmd.z, teacher_x, teacher_z)
        if self.pilot.mode == "auto":
            self.lead.note(now, matched)
            observe_recognition(self.mb.stimuli, self.lead, self.lead.aim_fresh(now))
            kind_l, kind_r = self.mb.reinforce_stimuli(
                now,
                self._taught_l,
                self._taught_r,
                learning=bool(self.learn),
            )
            if kind_l or kind_r:
                self._note_dan(kind_l, kind_r)
        self.lead_match = self.lead.fraction(now)
        if self.pilot.mode != "auto" or self.teacher_weight <= 0.005:
            return cmd
        mixed_x, mixed_z = mix_velocity(cmd.x, cmd.z, teacher_x, teacher_z, self.teacher_weight)
        out_x, out_z, stopped = apply_guards(
            mixed_x,
            mixed_z,
            self.pilot.mode == "estop",
            forward_m,
        )
        who = "учитель" if self.teacher_weight >= 0.995 else "смешанно"
        return DriveCommand(out_x, out_z, stopped, cmd.phase, who, cmd.hint)

    def _release_stand_lock(self, now: float) -> None:
        if self.pilot.mode == "estop":
            self.drive.cancel_balance_timer()
            return
        due = self.drive.balance_due
        if due is None or now < due:
            return
        self._balance_once()

    def _balance_once(self) -> None:
        with self._lock:
            if self.drive.pose == "balance_unavailable":
                self.drive.balance_due = None
                return
            if not self.drive.needs_balance():
                self.drive.balance_due = None
                return
            code = self.drive.balance_stand()
            missing = self.drive.pose == "balance_unavailable"
        if missing:
            self._log("BalanceStand нет в SDK")
        else:
            self._log(f"BalanceStand code={code}")

    def _log_manual_move(self, x: float, z: float, code, now: float) -> None:
        """One line a second, manual axes only, with the sport-service code."""
        if self.pilot.mode != "manual":
            return
        if float(now) - self._move_log_t < 1.0:
            return
        self._move_log_t = float(now)
        self._log(f"Move x={x:.2f} z={z:.2f} code={code}")

    def _emit(self, cmd: DriveCommand, now: float) -> None:
        if cmd.stop or (abs(cmd.x) < 1e-6 and abs(cmd.z) < 1e-6):
            self._stop_once()
            return
        if self.drive.needs_balance():
            self._balance_once()
        if (not self._sent_move) or (now - self._last_send >= 0.10):
            code = self.drive.move(cmd.x, cmd.z)
            sent = self.drive.moves[-1]
            self._sent_move = True
            self._moving = True
            self._last_send = now
            self._log_manual_move(sent[0], sent[2], code, now)

    def status(self) -> dict:
        flash = flash_payload(self.mb.flash)
        link_ok = (time.monotonic() - self.last_hb) <= LINK_HOLD_S
        drive_l = _hemisphere_drive(self.mb.brain, self.mb.last_fwd)
        drive_r = _hemisphere_drive(self.mb.brain_r, self.mb.last_fwd_r)
        with self._lock:
            payload = {
                "mode": self.pilot.mode,
                "label": self.pilot.label(),
                "phase": self.pilot.phase,
                "phase_ru": phase_label(self.pilot.phase, self.pilot.steer, self.pilot.search_sign),
                "last_seen_side": self.pilot.last_seen_side or None,
                "search_sign": 1 if self.pilot.search_sign >= 0 else -1,
                "reinforce": self.mb.reinforce,
                "teacher_pam_l": int(self.mb.teacher_pam_l),
                "teacher_pam_r": int(self.mb.teacher_pam_r),
                "teacher_ppl1_l": int(self.mb.teacher_ppl1_l),
                "teacher_ppl1_r": int(self.mb.teacher_ppl1_r),
                "who": self.pilot.who,
                "took_over": bool(self.pilot.took_over),
                "learning": bool(self.learn),
                "paused": bool(self.paused),
                "learning_frozen": bool(self.mb.learning_hold),
                "reinforcement_off": bool(self.mb.reinforcement_off),
                "object_drives": _object_drives(self.mb.object_drives),
                "autonomy": bool(self.pilot.autonomy),
                "recognized": bool(self.recognized),
                "confidence": self.confidence,
                "confidence_ready": self.confidence_ready,
                "readout": self.readout,
                "sector": self.sector,
                "sector_smooth": self.pilot.track.sector_smooth,
                "hysteresis": self.pilot.track.as_dict(),
                "steer": self.pilot.steer,
                "r_l": self.pilot.track.r_l,
                "r_r": self.pilot.track.r_r,
                "r_diff": self.pilot.track.r_diff,
                "z_fly": self.pilot.track.yaw_z,
                "recognized_L": bool(self.recognized_l),
                "recognized_R": bool(self.recognized_r),
                "eyes_ru": eyes_state_label(bool(self.recognized_l), bool(self.recognized_r)),
                "confidence_L": float(self.confidence_l),
                "confidence_R": float(self.confidence_r),
                "confidence_L_ready": bool(self.confidence_l_ready),
                "confidence_R_ready": bool(self.confidence_r_ready),
                "overlap": float(self.mb.overlap),
                "recog_camera_only": bool(self.mb.camera_only),
                "distance_m": self.dist_m,
                "forward_m": self.forward_m,
                "x": self.drive.moves[-1][0] if self._moving and self.drive.moves else 0.0,
                "z": self.drive.moves[-1][2] if self._moving and self.drive.moves else 0.0,
                "n_pam": int(self.mb.n_pam),
                "n_ppl1": int(self.mb.n_ppl1),
                "n_pam_total": int(self.mb.progress.base_pam + self.mb.n_pam),
                "n_ppl1_total": int(self.mb.progress.base_ppl1 + self.mb.n_ppl1),
                "frames_ok": bool(self.frames_ok),
                "link_ok": bool(link_ok),
                "hint": self.pilot.hint,
                "lead": lead_label(self.teacher_weight),
                "teacher_weight": float(self.teacher_weight),
                "teacher_x": float(self.teacher_x),
                "teacher_z": float(self.teacher_z),
                "fly_x": float(self.fly_x),
                "fly_z": float(self.fly_z),
                "lead_match": float(self.lead_match),
                "lead_match_s": 8.0,
                "kc_on": int(self.mb.last_kc_on),
                "kc_n": int(self.mb.brain.n_kc),
                "kc_active_frac": self._kc_active_frac(),
                "kc_active_frac_gd": self._kc_active_frac_group("gd"),
                "kc_active_frac_abp": self._kc_active_frac_group("abp"),
                "drive_l": drive_l,
                "drive_r": drive_r,
                "turn": float(live_turn(drive_l, drive_r, self.mb.base_l, self.mb.base_r)),
                "pam_event": bool(self._pam_event),
                "ppl1_event": bool(self._ppl1_event),
                "pam_count": int(self.mb.n_pam),
                "ppl1_count": int(self.mb.n_ppl1),
                "stimuli": self.mb.stimuli.status(),
                "kc_l": _compact_kc(None if self.mb.last_fwd is None else self.mb.last_fwd.kc),
                "kc_r": _compact_kc(None if self.mb.last_fwd_r is None else self.mb.last_fwd_r.kc),
                "drift": float(self.mb.last_drift),
                "flash": flash,
                "flash_r": flash_payload(self.mb.flash_r),
                "log": list(self.log),
            }
            payload.update(self.run_marker.status())
            payload.update(widget_status(self.mb, self.panel))
            return payload


def _yolo_vec(value):
    """Four floats, or nothing when the laptop did not send a box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return np.asarray([float(item) for item in value], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def _object_drives(rows) -> list:
    """Per-object approach minus avoidance. Plain floats for JSON."""
    out = []
    for row in list(rows or []):
        if not isinstance(row, dict) or "name" not in row:
            continue
        item = {"name": str(row.get("name", "")), "drive": float(row.get("drive", 0.0))}
        if row.get("when"):
            item["when"] = str(row["when"])
        out.append(item)
    return out


def _hemisphere_drive(brain, fwd) -> float:
    """Approach minus avoidance for one mushroom body. A silent half is zero."""
    if fwd is None:
        return 0.0
    return float(brain.appetitive_drive(fwd.action_scores))


def _compact_kc(vec, cap: int = 400) -> list:
    """Local KC indices for the laptop 3D view. Old clients ignore the key."""
    if vec is None:
        return []
    idx = np.flatnonzero(np.asarray(vec) > 0)
    if idx.size > int(cap):
        step = int(np.ceil(idx.size / float(cap)))
        idx = idx[::step][: int(cap)]
    return [int(i) for i in idx]


def make_brain(
    drive: SportDrive,
    state_path: str,
    npz_path: str | None = None,
    self_radius: float = 0.6,
    overlap: float = DEFAULT_OVERLAP,
    camera_only: bool = True,
) -> BrainLoop:
    mb = MbTrainer(
        npz_path or default_npz(),
        seed=1,
        eta=0.2,
        dan="teacher",
        overlap=overlap,
        camera_only=bool(camera_only),
    )
    mb.learn = False
    loop = BrainLoop(drive, mb, state_path, self_radius=self_radius)
    loop._log(recog_label(mb.camera_only))
    from pathlib import Path

    path = Path(state_path)
    if path.is_file():
        if mb.load(path):
            loop._log("загружены веса KC→MBON")
        else:
            loop._log(mb.load_note)
        mb.learn = False
        loop.learn = False
    return loop


class _Handler(BaseHTTPRequestHandler):
    loop: BrainLoop

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.loop.note_heartbeat(time.monotonic())
        if self.path.split("?", 1)[0] != "/status":
            self.send_response(404)
            self.end_headers()
            return
        self._send(self.loop.status())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        now = time.monotonic()
        path = self.path.split("?", 1)[0]
        if path == "/pause":
            data["op"] = "pause"
        elif path == "/resume":
            data["op"] = "resume"
        elif path != "/cmd":
            self.send_response(404)
            self.end_headers()
            return
        self.loop.handle(data, now)
        self._send(self.loop.status())

    def log_message(self, _format, *_args) -> None:
        return


def serve(loop: BrainLoop, port: int = 8090) -> ThreadingHTTPServer:
    """GET /status, POST /cmd, POST /pause, and POST /resume on this port.

    Pause and resume accept ``{}``. The same ops work on POST /cmd.
    Pause zeros motion and freezes plasticity. After resume, learning stays
    frozen until the dog moves and then for PAUSE_GRACE_S.
    """
    _Handler.loop = loop
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), _Handler)
    threading.Thread(target=server.serve_forever, name="flybrain-http", daemon=True).start()
    return server
