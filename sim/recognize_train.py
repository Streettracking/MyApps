"""One learner, no zones. The fly mushroom body is what learns.

Raw camera/lidar → fixed random projection → PN → KC → MBON.
Only KC→MBON weights change, gated by a DAN. Default DAN is the operator:
T is a treat (appetitive PAM) and X is a punish (aversive PPL1).
Familiarity is optional. The Hebbian layer remains available with --learner hebb.
Scene labels and the D/N keys only choose which monitor curve a sample joins.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

from .frame_sense import sim_previews
from .hemifield import format_fly_line
from .lidar_fresh import DEFAULT_LIDAR_REFRESH, SimLidarBank, mismatch_warning
from .map_marks import MarkLayer
from .pilot import (
    Pilot,
    TeachRepeater,
    cloud_forward,
    ego_sector_ranges,
    format_range_line,
    forward_clearance,
    scrub_range,
)
from .mb_train import MbTrainer, default_npz
from .raw_sense import N_AZ, OFF_LIDAR, render_view
from .recognize import ConspecificRecognizer
from .world import AgentState, ArenaConfig, ArenaWorld, default_agents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "mb_train_state.npz"
WALK = 0.42


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class RecognizeTrainSim:
    def __init__(
        self,
        n_agents: int = 3,
        seed: int = 1,
        state_path: Path | None = None,
        learner: str = "mb",
        dan: str = "teacher",
        npz: Path | None = None,
        eta: float = 0.2,
        auto_teach: bool = False,
        punish: bool = False,
        lidar_refresh: float = DEFAULT_LIDAR_REFRESH,
        return_auto_s: float = 0.0,
        steer: str = "bilateral",
    ):
        if learner not in ("mb", "hebb"):
            raise ValueError(learner)
        self.learner_kind = learner
        self.dan = dan
        self.auto_teach = auto_teach
        self.punish_auto = punish
        self.rng = np.random.default_rng(seed)
        n_agents = max(1, int(n_agents))
        agents = default_agents(n_agents, sense=True)
        agents[0].x, agents[0].y, agents[0].yaw = 1.7, 2.4, 0.2
        self.world = ArenaWorld(ArenaConfig(move_zones=False), [], agents)
        self.world.spawn_distractors(seed)
        self.learner = agents[0]
        self.peers = agents[1:]
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE
        self.learn = True
        self.peer_curve: list[tuple[float, float]] = []
        self.other_curve: list[tuple[float, float]] = []
        self.metric_curve: list[dict] = []
        self.log: deque[str] = deque(maxlen=24)
        self._prev_like = 0.0
        self._last_spike_t = -10.0
        self.last_like = 0.0
        self.last_match = 0.0
        self.step_i = 0
        self.treat_flash = False
        self.operator: str | None = None
        self.scene_dog = False
        self.saw_dog = False
        self.lidar_interval = float(lidar_refresh) if lidar_refresh > 0 else DEFAULT_LIDAR_REFRESH
        self.lidar_fresh_on = float(lidar_refresh) > 0
        self.lidar_bank = SimLidarBank(self.lidar_interval if self.lidar_fresh_on else 0.0)
        self.lidar_bank.enabled = self.lidar_fresh_on
        self.lidar_warning = ""
        self._lidar_warning_full = ""
        self._lidar_warn_logged = ""
        self._weights_from_disk = False
        self._lidar_saved: float | None = None
        self.last_near_max = 0.0
        self.marks = MarkLayer()
        self.pilot = Pilot(return_auto_s)
        self.pilot.set_steer(steer if steer in ("bilateral", "sectors") else "bilateral")
        self.teach_pulse = TeachRepeater()
        self.aim_sector: int | None = None
        self.aim_dist: float | None = None
        self.forward_m: float | None = None
        self.recognized_now = False
        self._hint_logged = ""
        self._learn_block_log = -10.0
        self.mb: MbTrainer | None = None
        self.recognizer: ConspecificRecognizer | None = None
        if learner == "mb":
            self.mb = MbTrainer(npz or default_npz(), seed=seed, eta=eta, dan=dan)
            self._log("грибовидное тело  учитель — клавиша T  учатся только KC→MBON")
            self._log(self._lidar_intro())
        else:
            self.recognizer = ConspecificRecognizer()
            self._log("слой сравнения  без зон")
            self._log(self._lidar_intro())
        self._record_metric()

    def _log(self, text: str) -> None:
        self.log.append(f"{self.world.t:6.1f}s  {text}")

    def _record_metric(self) -> None:
        if self.mb is not None:
            snap = {"t": float(self.world.t), "drift": self.mb.last_drift, "readout": self.mb.last_readout}
        else:
            assert self.recognizer is not None
            snap = self.recognizer.snapshot()
            snap["t"] = float(self.world.t)
        self.metric_curve.append(snap)

    def scripted_steer(self) -> tuple[float, float]:
        """Walk toward peers most of the time, then a distractor. Not a label."""
        phase = int(self.world.t // 3.0) % 4
        moving = [d for d in self.world.distractors if abs(d.vx) + abs(d.vy) > 1e-6]
        if phase < 3 and self.peers:
            pool = self.peers
        elif moving:
            pool = moving
        else:
            return 1.0, 0.0
        target = min(pool, key=lambda o: float(np.hypot(o.x - self.learner.x, o.y - self.learner.y)))
        err = _wrap(float(np.arctan2(target.y - self.learner.y, target.x - self.learner.x)) - self.learner.yaw)
        if abs(err) < 0.35:
            return 1.0, 0.0
        return 0.35, 1.0 if err > 0 else -1.0

    def _move_body(self, agent: AgentState, steer_x: float, steer_z: float, rate: bool = False) -> None:
        dt = self.world.cfg.dt
        if rate:
            agent.yaw = _wrap(agent.yaw + float(steer_z) * dt)
            speed = float(steer_x)
        else:
            agent.yaw = _wrap(agent.yaw + steer_z * 1.6 * dt)
            speed = float(steer_x) * WALK
        agent.vx = speed * float(np.cos(agent.yaw))
        agent.vy = speed * float(np.sin(agent.yaw))
        nxt_x = float(np.clip(agent.x + agent.vx * dt, 0.35, self.world.cfg.width - 0.35))
        nxt_y = float(np.clip(agent.y + agent.vy * dt, 0.35, self.world.cfg.height - 0.35))
        if nxt_x != agent.x + agent.vx * dt or nxt_y != agent.y + agent.vy * dt:
            agent.yaw = _wrap(agent.yaw + np.pi)
        agent.x, agent.y = nxt_x, nxt_y

    def _wander_peers(self) -> None:
        for peer in self.peers:
            peer.yaw = _wrap(peer.yaw + float(self.rng.normal(0.0, 0.18)))
            self._move_body(peer, 1.0, 0.0)

    def _sector_ranges(self):
        xy = self.lidar_bank.held_xy
        if not xy:
            return None
        pts = np.asarray(xy, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            return None
        origin = np.array([float(self.learner.x), float(self.learner.y)], dtype=np.float32)
        return ego_sector_ranges(pts, origin, float(self.learner.yaw), self_radius=self.pilot.self_radius)

    def _note_aim(self, feat: np.ndarray, recognized: bool) -> None:
        self.recognized_now = bool(recognized)
        ranges = self._sector_ranges()
        if ranges is not None:
            self.forward_m = cloud_forward(ranges)
        else:
            self.forward_m = forward_clearance(feat, self.pilot.self_radius)
        if recognized:
            self.aim_sector = self.marks.aim_sector
            if ranges is not None and self.aim_sector is not None:
                self.aim_dist = ranges[int(self.aim_sector) % 8]
            else:
                self.aim_dist = scrub_range(self.marks.aim_dist, self.pilot.self_radius)
        else:
            self.aim_sector = None
            self.aim_dist = None

    def step(self, steer_x: float, steer_z: float, learn: bool | None = None, teach: str | None = None, rate: bool = False) -> None:
        if learn is None:
            learn = self.learn
        self.world.step_environment()
        self._wander_peers()
        self._move_body(self.learner, steer_x, steer_z, rate=rate)
        feat, hit = render_view(self.learner, self.world, include_agents=True)
        self.lidar_bank.update(self.learner, self.world, float(self.world.t))
        self.lidar_bank.apply(self.learner, feat)
        self.last_near_max = float(np.max(feat[OFF_LIDAR : OFF_LIDAR + N_AZ]))
        speed = float(np.hypot(self.learner.vx, self.learner.vy))
        self.treat_flash = False
        self.scene_dog = bool(hit.dog and not hit.distractor)
        self.saw_dog = bool(hit.dog)
        if self.mb is not None:
            self.mb.learn = learn
            fwd = self.mb.forward(feat)
            value = self.mb.last_readout
            # Energy baseline only. Operator labels are applied after, for the panel.
            conf = self.mb.observe(feat, value)
            # Auto-teach stands in for a human pressing T/X. It is a DAN, not a D/N label.
            kind = teach
            if kind is None and self.auto_teach and self.dan == "teacher":
                if hit.dog and not hit.distractor:
                    kind = "pam"
                elif self.punish_auto and hit.distractor and not hit.dog:
                    kind = "ppl1"
            if kind == "pam":
                self.treat_flash = True
            self.mb.teach(fwd, kind if self.dan == "teacher" else None, float(self.world.t))
            self.mb.progress.tick(self.world.cfg.dt, learn)
            self.mb.progress.note_label(self.operator, value, conf.recognized, conf.ready)
            if kind == "pam" and (self.mb.n_pam <= 2 or self.mb.n_pam % 15 == 0):
                self._log(f"лакомство PAM  #{self.mb.n_pam}")
            elif kind == "ppl1" and (self.mb.n_ppl1 <= 2 or self.mb.n_ppl1 % 15 == 0):
                self._log(f"наказание PPL1  #{self.mb.n_ppl1}")
            if (
                conf.ready
                and conf.recognized
                and not getattr(self, "_was_rec", False)
                and self.world.t - getattr(self, "_last_rec_log", -10.0) > 2.0
            ):
                self._log(f"узнаю сородича  {conf.percent:.0f}%")
                self._last_rec_log = float(self.world.t)
            self._was_rec = bool(conf.recognized) if conf.ready else False
            self.last_like = value
            self.last_match = value
            seen = bool(conf.ready and conf.recognized)
            self.marks.consider(
                self.mb,
                feat,
                float(self.world.t),
                recognized=seen,
                mode="sim",
            )
            self._note_aim(feat, seen)
        else:
            rec = self.recognizer
            assert rec is not None
            n_before = rec.n_self
            if learn:
                like = rec.observe(feat, speed)
            else:
                like = rec.likeness(feat)
            match = rec.learned_match(feat)
            self.last_like = float(like)
            self.last_match = float(match)
            if rec.n_self > n_before and (rec.n_self <= 4 or rec.n_self % 8 == 0):
                self._log(f"prototype updated  n={rec.n_self}")
            if like > 0.8 and self._prev_like < 0.45 and self.world.t - self._last_spike_t > 1.5:
                self._log(f"likeness spike  {like:.2f}")
                self._last_spike_t = self.world.t
            self._prev_like = like
            value = match
        point = (float(self.world.t), float(value))
        # ``hit`` bins the monitor curve only.
        if hit.dog:
            self.peer_curve.append(point)
        else:
            self.other_curve.append(point)
        self.step_i += 1
        if self.step_i % 10 == 0:
            self._record_metric()
            if self.mb is not None:
                self.mb.note_drift(float(self.world.t))
        if self.state_path and self.step_i % 300 == 0:
            self.save()
            self._log(f"сохранено  {self.state_path.name}")

    def _effective_refresh(self) -> float:
        return self.lidar_interval if self.lidar_fresh_on else 0.0

    def _lidar_intro(self) -> str:
        if self.lidar_fresh_on:
            return (
                f"свежий лидар {self.lidar_interval:.1f} с, пустой кадр после сброса не идёт в мозг"
            )
        return "лидар копится, как карта без сброса"

    def _sync_lidar_warning(self) -> None:
        if not self._weights_from_disk:
            self.lidar_warning = ""
            self._lidar_warning_full = ""
            return
        full = mismatch_warning(self._lidar_saved, self._effective_refresh())
        self._lidar_warning_full = full or ""
        self.lidar_warning = "веса с другой карты лидара — R сброс, файл не стираю" if full else ""
        if full and full != self._lidar_warn_logged:
            self._log(full)
            self._lidar_warn_logged = full

    def toggle_lidar_fresh(self) -> None:
        self.lidar_fresh_on = not self.lidar_fresh_on
        if self.lidar_fresh_on and self.lidar_interval <= 0:
            self.lidar_interval = DEFAULT_LIDAR_REFRESH
        self.lidar_bank.interval = self.lidar_interval
        self.lidar_bank.set_enabled(self.lidar_fresh_on, float(self.world.t))
        self._log(self._lidar_intro())
        self._sync_lidar_warning()

    def save(self) -> Path:
        if self.mb is not None:
            self.mb.lidar_refresh = self._effective_refresh()
            self.mb.save(self.state_path)
        elif self.recognizer is not None:
            self.recognizer.save(self.state_path)
        self._log(f"сохранено  {self.state_path.name}")
        return self.state_path

    def load(self) -> None:
        if not Path(self.state_path).is_file():
            raise FileNotFoundError(self.state_path)
        if self.mb is not None:
            self.mb.load(self.state_path)
            self.dan = self.mb.dan
            self._weights_from_disk = True
            self._lidar_saved = self.mb.saved_lidar_refresh
            self._sync_lidar_warning()
            self._log(f"загружены веса KC→MBON  дрейф {self.mb.last_drift:.1f}  лакомств всего {self.mb.progress.base_pam}")
        elif self.recognizer is not None:
            self.recognizer.load(self.state_path)
            self._log(f"загружен прототип  n={self.recognizer.n_self}")

    def reset_lidar(self) -> None:
        """Drop the simulated cloud. Current bodies are drawn again next frame."""
        self.lidar_bank.clear(float(self.world.t))
        self._log("карта лидара симулятора очищена")

    def reset(self) -> None:
        if self.mb is not None:
            self.mb.reset()
            self._was_rec = False
            self.marks.alive.clear()
            self._weights_from_disk = False
            self.lidar_warning = ""
            self._lidar_warning_full = ""
            self._log("веса и счётчики обучения сброшены")
        elif self.recognizer is not None:
            self.recognizer.reset()
            self._log("прототип сброшен")

    def view(self, focused: bool, udp_status: str, last_command: str, keys_hint: str):
        from .train_monitor import MonitorView

        cam, lid = sim_previews(self.learner, self.world, self.lidar_bank.display)
        if self.mb is not None:
            caption = "сырой выход: подход − избегание" if self.dan == "teacher" else "сырой выход: минус новизна"
            mode = f"сим · {self.pilot.label()}"
            prog = self.mb.progress
            conf = self.mb.conf
            return MonitorView(
                title=f"тренировка узнавания   {self.learner.agent_id}",
                camera=cam,
                lidar=lid,
                likeness=self.last_like,
                learned_match=self.last_match,
                peer_curve=self._scaled(self.peer_curve),
                other_curve=self._scaled(self.other_curve),
                metric_curve=self.metric_curve[-120:],
                log_lines=list(self.log),
                paused=not self.learn,
                udp_status=udp_status,
                last_command=last_command,
                mode_label=mode,
                operator_label="кривые по кадру симулятора. D и N только для панели точности.",
                t=self.world.t,
                focused=focused,
                keys_hint=keys_hint,
                learner="mb",
                dan_mode=self.dan,
                kc_on=self.mb.last_kc_on,
                kc_n=self.mb.brain.n_kc,
                kc_bins=self.mb.last_kc_bins,
                drift=self.mb.last_drift,
                drift_curve=self.mb.drift_curve[-120:],
                dan_events=self.mb.dan_events[-240:],
                treat_flash=self.treat_flash,
                readout_caption=caption,
                n_pam=self.mb.n_pam,
                n_ppl1=self.mb.n_ppl1,
                recognized=conf.recognized,
                confidence=conf.percent,
                confidence_ready=conf.ready,
                session_time=prog.session_time,
                total_time=prog.base_time + prog.session_time,
                total_pam=prog.base_pam + self.mb.n_pam,
                total_ppl1=prog.base_ppl1 + self.mb.n_ppl1,
                session_sep=prog.session_sep(),
                total_sep=prog.total_sep(),
                session_acc=prog.session_acc(),
                total_acc=prog.total_acc(),
                session_labeled=prog.ses_scored,
                total_labeled=prog.base_scored + prog.ses_scored,
                session_novelty=self.mb.n_novelty,
                total_novelty=prog.base_novelty + self.mb.n_novelty,
                marks=self.marks.visible(float(self.world.t)),
                lidar_mode=self._lidar_caption(),
                lidar_warning=self.lidar_warning,
                lidar_fresh_on=self.lidar_fresh_on,
                learn_flash=self.mb.flash,
                learn_flash_r=self.mb.flash_r,
                mb_layout=self.mb.layout,
                pilot_mode=self.pilot.label(),
                pilot_phase=self.pilot.phase,
                pilot_who=self.pilot.who,
                pilot_hint=self.pilot.hint,
                learning_on=self.learn,
                autonomy_on=self.pilot.autonomy,
                range_line=format_range_line(
                    self.aim_dist,
                    self.forward_m,
                    self.aim_sector,
                    self.pilot.track.sector_smooth,
                    self.pilot.track.as_dict(),
                ),
                fly_line=format_fly_line(
                    self.pilot.track.r_l,
                    self.pilot.track.r_r,
                    self.pilot.track.r_diff,
                    self.pilot.track.yaw_z,
                    self.pilot.steer,
                ),
                steer=self.pilot.steer,
            )
        assert self.recognizer is not None
        return MonitorView(
            title=f"тренировка узнавания   {self.learner.agent_id}",
            camera=cam,
            lidar=lid,
            likeness=self.last_like,
            learned_match=self.last_match,
            peer_curve=self.peer_curve[-400:],
            other_curve=self.other_curve[-400:],
            metric_curve=self.metric_curve[-120:],
            proto_self=self.recognizer.proto[0],
            proto_other=self.recognizer.proto[1],
            log_lines=list(self.log),
            paused=not self.learn,
            n_self=self.recognizer.n_self,
            n_other=self.recognizer.n_other,
            udp_status=udp_status,
            last_command=last_command,
            mode_label="сим · слой сравнения · без зон",
            operator_label="кривые по кадру симулятора, не по ярлыку обучения",
            t=self.world.t,
            focused=focused,
            keys_hint=keys_hint,
            learner="hebb",
            lidar_mode=self._lidar_caption(),
            lidar_warning=self.lidar_warning,
            lidar_fresh_on=self.lidar_fresh_on,
        )

    def _lidar_caption(self) -> str:
        if self.lidar_fresh_on:
            return f"свежий лидар {self.lidar_interval:.1f} с"
        return "лидар копится"

    def _scaled(self, rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Map MBON scores into 0..1 for the shared plot, using this run's own range."""
        if not rows:
            return []
        vals = [v for _, v in self.peer_curve[-400:]] + [v for _, v in self.other_curve[-400:]]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        span = hi - lo if hi > lo else 1.0
        return [(t, (v - lo) / span) for t, v in rows[-400:]]

    def summary(self) -> dict:
        def _window(rows: list[tuple[float, float]], start: float, end: float) -> float | None:
            vals = [v for t, v in rows if start <= t < end]
            if not vals:
                return None
            return float(np.mean(vals))

        t_end = float(self.world.t)
        early_end = min(3.0, t_end)
        late_start = max(0.0, t_end - 5.0)
        out = {
            "t": t_end,
            "learner": self.learner_kind,
            "dan": self.dan,
            "peer_readout_early": _window(self.peer_curve, 0.0, early_end),
            "peer_readout_late": _window(self.peer_curve, late_start, t_end + 1),
            "other_readout_late": _window(self.other_curve, late_start, t_end + 1),
            "peer_samples": len(self.peer_curve),
            "other_samples": len(self.other_curve),
            "state": str(self.state_path),
        }
        if self.mb is not None:
            final = self.mb.probe()
            out.update(
                {
                    "drift": self.mb.last_drift,
                    "probe_init": self.mb.probe_init,
                    "probe_final": final,
                    "sep_dog_minus_dist": self.mb.separation(final),
                    "n_pam": self.mb.n_pam,
                    "n_ppl1": self.mb.n_ppl1,
                    "n_novelty": self.mb.n_novelty,
                    "kc_on": self.mb.last_kc_on,
                }
            )
        elif self.recognizer is not None:
            first = self.metric_curve[0] if self.metric_curve else {}
            last = self.metric_curve[-1] if self.metric_curve else {}
            out.update(
                {
                    "n_self": self.recognizer.n_self,
                    "n_other": self.recognizer.n_other,
                    "invariance_t0": first.get("invariance"),
                    "invariance_final": last.get("invariance"),
                    "sep_moving_final": last.get("sep_moving"),
                    "purity_final": last.get("purity"),
                }
            )
        return out


SIM_KEYS = "T/X учить  A авто  M перехват  Y руль  U запись  P обучение  G вспышка  V свежий  C сброс  B  D/N  R S L F12 Esc"
HEBB_KEYS = "стрелки ход   U запись   C сброс   V свежий   P пауза   R сброс   S/L   F12   Esc"


def _command_name(steer_x: float, steer_z: float, hold: bool) -> str:
    if hold or (steer_x == 0 and steer_z == 0):
        return "стоп"
    bits = []
    if steer_x > 0:
        bits.append("вперёд")
    elif steer_x < 0:
        bits.append("назад")
    if steer_z > 0:
        bits.append("влево")
    elif steer_z < 0:
        bits.append("вправо")
    return " ".join(bits) or "стоп"


def _apply_pilot_keys(session: RecognizeTrainSim, inp, now: float) -> None:
    if inp.estop:
        session.pilot.estop()
        session._log("E-STOP")
    elif inp.stop:
        session.pilot.space()
        session._log("стоп")
    if inp.takeover and not inp.estop:
        session.pilot.takeover()
        session._log("перехват: ручное, автономия сама не вернётся")
    if inp.autonomy_toggle and not inp.estop:
        if session.learner_kind != "mb":
            session._log("автономия только у грибовидного тела")
        elif session.pilot.autonomy:
            session.pilot.stop_auto()
            session._log("автономия выключена")
        elif not session.pilot.start_auto(now):
            session._log("E-STOP держит стоп. M переводит в ручное")
        else:
            session._log("автономия: поиск в одну сторону, руль %s" % _steer_name(session.pilot.steer))
    if inp.steer_toggle and session.learner_kind == "mb":
        mode = session.pilot.toggle_steer()
        session._log("руль: %s" % _steer_name(mode))
    if inp.pause_learn:
        session.learn = not session.learn
        session._log("обучение выключено, T/X веса не меняют" if not session.learn else "обучение включено")


def _steer_name(mode: str) -> str:
    return "билатерально" if mode == "bilateral" else "секторы"


def _readouts(session: RecognizeTrainSim) -> tuple[float, float]:
    if session.mb is None:
        return 0.0, 0.0
    return float(session.mb.r_l), float(session.mb.r_r)


def _drive_sim(session: RecognizeTrainSim, inp, now: float) -> tuple[float, float, bool, str]:
    r_l, r_r = _readouts(session)
    cmd = session.pilot.command(
        now,
        (inp.steer_x, inp.steer_z),
        focused=inp.focused,
        frames_ok=True,
        link_ok=True,
        recognized=session.recognized_now and session.learner_kind == "mb",
        sector=session.aim_sector,
        dist_m=session.aim_dist,
        forward_m=session.forward_m,
        r_l=r_l,
        r_r=r_r,
    )
    if cmd.hint and cmd.hint != session._hint_logged:
        session._log(cmd.hint)
        session._hint_logged = cmd.hint
    elif not cmd.hint:
        session._hint_logged = ""
    if session.pilot.autonomy:
        return cmd.x, cmd.z, True, _phase_name(cmd.phase)
    if abs(inp.steer_x) + abs(inp.steer_z) > 0 and inp.focused:
        return cmd.x, cmd.z, True, _phase_name(cmd.phase)
    if session.pilot.took_over or session.pilot.mode == "estop":
        return 0.0, 0.0, True, "стоп"
    sx, sz = session.scripted_steer()
    return sx, sz, False, "сценарий"


def _phase_name(phase: str) -> str:
    return {
        "search": "поиск",
        "approach": "подход",
        "hold": "стоп 1 м",
        "manual": "ручное",
        "stop": "стоп",
    }.get(phase, phase)


def run_gui(session: RecognizeTrainSim, seconds: float = 0.0, screenshot_path: Path | None = None) -> dict:
    from .frame_record import FrameRecorder, OperatorMarks
    from .train_monitor import TrainMonitor

    mon = TrainMonitor("Go2 recognition trainer — sim")
    shot = Path(screenshot_path) if screenshot_path else None
    rec = FrameRecorder(ROOT / "logs" / "yolo_frames", fps=float(getattr(session, "rec_fps", 2.0)))
    marks = OperatorMarks()
    try:
        return _run_gui(session, mon, shot, seconds, rec, marks)
    finally:
        rec.close()


def _run_gui(session, mon, shot, seconds, rec, marks):
    from .frame_record import note_operator, weak_label_now

    while True:
        inp = mon.pump()
        if inp.quit:
            break
        if inp.record_toggle:
            on = rec.toggle()
            _path = rec.stats()[3]
            session._log("запись кадров %s" % (_path if on else "выключена"))
        if inp.label == "dog":
            session.operator = "dog"
            session._log("метка D — только для панели, в обучение не входит")
        elif inp.label == "none":
            session.operator = "none"
            session._log("метка N — только для панели, в обучение не входит")
        if inp.beep_toggle:
            session._log("звук включён" if mon.beep_on else "звук выключен")
        note_operator(marks, float(session.world.t), inp)
        _apply_pilot_keys(session, inp, float(session.world.t))
        if inp.reset:
            session.reset()
        if inp.lidar_reset:
            session.reset_lidar()
        if inp.lidar_toggle:
            session.toggle_lidar_fresh()
        if inp.flash_toggle:
            if session.learner_kind != "mb":
                mon.flash_open = False
                session._log("вспышка обучения только у грибовидного тела")
            else:
                session._log("схема обучения открыта" if mon.flash_open else "схема обучения скрыта")
        if inp.save:
            session.save()
        if inp.load:
            try:
                session.load()
            except FileNotFoundError:
                session._log(f"нет файла {session.state_path.name}, продолжаем с текущими весами")
        steer_x, steer_z, rate, command_name = _drive_sim(session, inp, float(session.world.t))
        teach = session.teach_pulse.poll(
            float(session.world.t),
            inp.treat,
            inp.punish,
            inp.treat_down,
            inp.punish_down,
        )
        if teach and not session.learn and float(session.world.t) - session._learn_block_log > 1.0:
            session._log("обучение выключено, T/X веса не меняют")
            session._learn_block_log = float(session.world.t)
            teach = None
        elif teach and not session.learn:
            teach = None
        session.step(steer_x, steer_z, teach=teach, rate=rate)
        view = session.view(
            focused=inp.focused,
            udp_status="симулятор. В автономии ведёт мозг, иначе стрелки или сценарий.",
            last_command=command_name,
            keys_hint=SIM_KEYS if session.learner_kind == "mb" else HEBB_KEYS,
        )
        rec_on, rec_n, rec_bytes, _rec_path = rec.stats()
        labelled = weak_label_now(marks, float(session.world.t)) is not None
        view.record_on = rec_on
        view.record_saved = rec_n
        view.record_bytes = rec_bytes
        view.record_idle = rec_on and not labelled
        if rec_on and labelled and view.camera is not None and rec.due():
            _offer_sim_frame(session, rec, marks, view.camera, steer_x, steer_z)
        mon.draw(view)
        if inp.screenshot:
            dest = ROOT / "logs" / "monitor_shot.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            mon.save_screenshot(str(dest))
            session._log(f"снимок  {dest.name}")
        if seconds > 0 and session.world.t >= seconds:
            if shot is not None:
                shot.parent.mkdir(parents=True, exist_ok=True)
                mon.save_screenshot(str(shot))
            break
    return session.summary()


def _offer_sim_frame(session: RecognizeTrainSim, rec, marks, camera, steer_x: float, steer_z: float) -> None:
    from .frame_record import encode_camera_jpeg, fingerprint, make_meta

    try:
        jpeg = encode_camera_jpeg(camera)
    except Exception:
        return
    mb = session.mb
    conf = None if mb is None else mb.conf
    rec.offer(
        jpeg,
        make_meta(
            marks,
            float(session.world.t),
            source="sim",
            mode="auto" if session.pilot.autonomy else "manual",
            speed_m_s=float(np.hypot(session.learner.vx, session.learner.vy)),
            yaw_rad_s=float(steer_z),
            readout=None if mb is None else float(mb.last_readout),
            confidence=None if conf is None else float(conf.percent),
            confidence_ready=False if conf is None else bool(conf.ready),
            recognized=bool(session.recognized_now),
            sector=session.aim_sector,
            r_l=None if mb is None else float(mb.r_l),
            r_r=None if mb is None else float(mb.r_r),
        ),
        fingerprint(camera),
    )


def try_load(session: RecognizeTrainSim) -> None:
    path = Path(session.state_path)
    if not path.is_file():
        session._log(f"нет сохранённого состояния, старт с нуля ({path.name})")
        return
    try:
        session.load()
    except FileNotFoundError:
        session._log(f"нет сохранённого состояния, старт с нуля ({path.name})")


def run_headless(session: RecognizeTrainSim, seconds: float) -> dict:
    session.auto_teach = session.dan == "teacher"
    steps = int(seconds / session.world.cfg.dt)
    for _ in range(steps):
        sx, sz = session.scripted_steer()
        session.step(sx, sz, learn=True)
    session.save()
    return session.summary()


def _bin_delta(a: int, b: int) -> int:
    delta = abs(int(a) - int(b))
    return min(delta, N_AZ - delta)


def seek_trial(seed: int, teach_s: float = 25.0, seek_s: float = 20.0, state: Path | None = None) -> dict:
    """Scripted teacher, then autonomy. Distances are true arena metres, not the lidar estimate."""
    from .raw_sense import bearing_bin

    session = RecognizeTrainSim(
        n_agents=3,
        seed=seed,
        state_path=state or Path(f"/tmp/seek_mb_{seed}.npz"),
        learner="mb",
        dan="teacher",
        auto_teach=True,
        lidar_refresh=DEFAULT_LIDAR_REFRESH,
    )
    dt = float(session.world.cfg.dt)
    for _ in range(int(teach_s / dt)):
        sx, sz = session.scripted_steer()
        session.step(sx, sz, rate=False)
    session.learn = False
    session.auto_teach = False
    session.pilot.start_auto(float(session.world.t))
    phases = {"search": 0, "approach": 0, "hold": 0, "stop": 0, "manual": 0}
    recognized_frames = 0
    hold_events: list[float] = []
    event: list[float] = []
    last_hold = -10.0
    approach_frames = 0
    distractor_approach = 0
    dog_approach = 0
    distractor_hold_frames = 0
    for _ in range(int(seek_s / dt)):
        r_l, r_r = _readouts(session)
        cmd = session.pilot.command(
            float(session.world.t),
            (0.0, 0.0),
            focused=True,
            frames_ok=True,
            link_ok=True,
            recognized=session.recognized_now,
            sector=session.aim_sector,
            dist_m=session.aim_dist,
            forward_m=session.forward_m,
            r_l=r_l,
            r_r=r_r,
        )
        learner = session.learner
        dogs = []
        distractors = []
        bodies = []
        for peer in session.peers:
            dist = float(np.hypot(peer.x - learner.x, peer.y - learner.y))
            bearing = bearing_bin(peer.x - learner.x, peer.y - learner.y, learner.yaw)
            dogs.append(dist)
            bodies.append((dist, bearing, "dog"))
        for obj in session.world.distractors:
            dist = float(np.hypot(obj.x - learner.x, obj.y - learner.y))
            bearing = bearing_bin(obj.x - learner.x, obj.y - learner.y, learner.yaw)
            distractors.append(dist)
            bodies.append((dist, bearing, "distractor"))
        nearest_dog = min(dogs) if dogs else None
        nearest_other = min(distractors) if distractors else None
        phase = cmd.phase
        phases[phase] = phases.get(phase, 0) + 1
        if session.recognized_now:
            recognized_frames += 1
        now_t = float(session.world.t)
        good = phase == "hold" and session.recognized_now and nearest_dog is not None and 0.7 <= nearest_dog <= 1.3
        if good and nearest_dog is not None:
            if event and now_t - last_hold > 1.0:
                hold_events.append(float(np.median(event)))
                event = []
            event.append(float(nearest_dog))
            last_hold = now_t
        elif event and now_t - last_hold > 1.0:
            hold_events.append(float(np.median(event)))
            event = []
        if phase == "approach":
            approach_frames += 1
            if session.aim_sector is not None:
                picked = []
                for dist, bearing, kind in bodies:
                    if bearing is None:
                        continue
                    if _bin_delta(bearing, int(session.aim_sector)) <= 1:
                        picked.append((dist, kind))
                if picked and sorted(picked)[0][1] == "distractor":
                    distractor_approach += 1
                elif picked and sorted(picked)[0][1] == "dog":
                    dog_approach += 1
        if (
            phase == "hold"
            and nearest_other is not None
            and 0.7 <= nearest_other <= 1.3
            and (nearest_dog is None or nearest_other < nearest_dog)
        ):
            distractor_hold_frames += 1
        session.step(cmd.x, cmd.z, learn=False, rate=True)
    if event:
        hold_events.append(float(np.median(event)))
    return {
        "seed": seed,
        "teach_s": teach_s,
        "seek_s": seek_s,
        "pam": 0 if session.mb is None else session.mb.n_pam,
        "phases": phases,
        "recognized_frames": recognized_frames,
        "seek_steps": int(seek_s / dt),
        "hold_events": len(hold_events),
        "hold_median_m": None if not hold_events else float(np.median(hold_events)),
        "approach_frames": approach_frames,
        "distractor_approach_frames": distractor_approach,
        "dog_approach_frames": dog_approach,
        "distractor_hold_frames": distractor_hold_frames,
    }


def _sign_match(z: float, bearing: int | None) -> bool | None:
    """True when the commanded yaw points at this bearing. None if it is outside the view."""
    from .map_marks import bin_angle
    from .pilot import half_sector

    if bearing is None:
        return None
    ang = bin_angle(int(bearing))
    if abs(ang) <= half_sector() + 1e-5:
        return abs(z) < 0.08
    if z > 0.05:
        return ang > 0
    if z < -0.05:
        return ang < 0
    return False


def _aimed_kind(z: float, bodies: list[tuple[float, int | None, str]]) -> str | None:
    from .map_marks import bin_angle
    from .pilot import half_sector

    pool = []
    for dist, bearing, kind in bodies:
        if bearing is None:
            continue
        ang = bin_angle(int(bearing))
        if abs(z) < 0.08:
            if abs(ang) <= half_sector() * 2.0:
                pool.append((dist, kind))
        elif z > 0 and ang > 0:
            pool.append((dist, kind))
        elif z < 0 and ang < 0:
            pool.append((dist, kind))
    if not pool:
        return None
    pool.sort()
    return pool[0][1]


def _fov_bodies(session: RecognizeTrainSim):
    """Bodies inside the camera field. Bearing is None outside it, and those are dropped."""
    from .raw_sense import bearing_bin

    learner = session.learner
    bodies = []
    dogs = []
    for peer in session.peers:
        dist = float(np.hypot(peer.x - learner.x, peer.y - learner.y))
        bearing = bearing_bin(peer.x - learner.x, peer.y - learner.y, learner.yaw)
        if bearing is None:
            continue
        dogs.append((dist, bearing))
        bodies.append((dist, bearing, "dog"))
    for obj in session.world.distractors:
        dist = float(np.hypot(obj.x - learner.x, obj.y - learner.y))
        bearing = bearing_bin(obj.x - learner.x, obj.y - learner.y, learner.yaw)
        if bearing is None:
            continue
        bodies.append((dist, bearing, "distractor"))
    dogs.sort()
    nearest = None if not dogs else dogs[0]
    return nearest, bodies


def _side_hit(z: float, bearing: int | None) -> bool | None:
    return _sign_match(z, bearing)


def compare_steer(seeds: tuple[int, ...] = (1, 2, 3), teach_s: float = 8.0, seek_s: float = 12.0) -> list[dict]:
    """Same teacher, then both steer modes from the moment «УЗНАЮ» is on.

    A 20 s teacher lets the summed readout rise on quiet frames too, and the
    word goes back off (the burst is about 5–11 s). Autonomy started after
    that only searches, in both modes. This comparison therefore teaches for
    8 s with the instantaneous lidar and switches while the word is still on.
    Both modes see the same seed and the same script up to that switch.
    Direction uses the nearest peer that is inside the field of view.
    ``fly_acc`` is the sign of R_L−R_R during the teacher, before either
    controller moves the dog.
    """
    from .hemifield import bilateral_yaw

    rows = []
    for seed in seeds:
        taught = []
        for mode in ("sectors", "bilateral"):
            session = RecognizeTrainSim(
                n_agents=3,
                seed=int(seed),
                state_path=Path("/tmp/bilateral_%s_%s.npz" % (seed, mode)),
                learner="mb",
                dan="teacher",
                auto_teach=True,
                lidar_refresh=0.0,
                steer=mode,
            )
            dt = float(session.world.cfg.dt)
            fly_hit = 0
            fly_n = 0
            for _ in range(int(teach_s / dt)):
                sx, sz = session.scripted_steer()
                session.step(sx, sz, rate=False)
                if session.mb is None:
                    continue
                nearest, _bodies = _fov_bodies(session)
                if nearest is None:
                    continue
                matched = _side_hit(bilateral_yaw(float(session.mb.r_l), float(session.mb.r_r)), nearest[1])
                if matched is None:
                    continue
                fly_n += 1
                if matched:
                    fly_hit += 1
            taught.append((session, dt, fly_hit, fly_n))
        for session, dt, fly_hit, fly_n in taught:
            mode = session.pilot.steer
            session.learn = False
            session.auto_teach = False
            word_at_switch = bool(session.recognized_now)
            session.pilot.start_auto(float(session.world.t))
            zs: list[float] = []
            dir_hit = 0
            dir_n = 0
            approach_frames = 0
            distractor_frames = 0
            episodes = 0
            ended_ok = 0
            in_ep = False
            ep_ok = False
            recognized_frames = 0
            for _ in range(int(seek_s / dt)):
                r_l, r_r = _readouts(session)
                cmd = session.pilot.command(
                    float(session.world.t),
                    (0.0, 0.0),
                    focused=True,
                    frames_ok=True,
                    link_ok=True,
                    recognized=session.recognized_now,
                    sector=session.aim_sector,
                    dist_m=session.aim_dist,
                    forward_m=session.forward_m,
                    r_l=r_l,
                    r_r=r_r,
                )
                nearest, bodies = _fov_bodies(session)
                bearing = None if nearest is None else nearest[1]
                nearest_dog = None if nearest is None else nearest[0]
                phase = cmd.phase
                zs.append(float(cmd.z))
                if session.recognized_now:
                    recognized_frames += 1
                if phase == "approach":
                    approach_frames += 1
                    matched = _side_hit(float(cmd.z), bearing)
                    if matched is not None:
                        dir_n += 1
                        if matched:
                            dir_hit += 1
                    if _aimed_kind(float(cmd.z), bodies) == "distractor":
                        distractor_frames += 1
                if phase in ("approach", "hold") and session.recognized_now:
                    if not in_ep:
                        in_ep = True
                        episodes += 1
                        ep_ok = False
                    if phase == "hold" and nearest_dog is not None and 0.7 <= nearest_dog <= 1.3:
                        ep_ok = True
                else:
                    if in_ep and ep_ok:
                        ended_ok += 1
                    in_ep = False
                    ep_ok = False
                session.step(cmd.x, cmd.z, learn=False, rate=True)
            if in_ep and ep_ok:
                ended_ok += 1
            flips = 0
            prev = 0
            for z in zs:
                sign = 1 if z > 0.05 else (-1 if z < -0.05 else 0)
                if sign != 0 and prev != 0 and sign != prev:
                    flips += 1
                if sign != 0:
                    prev = sign
            pam = 0 if session.mb is None else int(session.mb.n_pam)
            rows.append(
                {
                    "seed": int(seed),
                    "steer": mode,
                    "pam": pam,
                    "teach_s": teach_s,
                    "seek_s": seek_s,
                    "word_at_switch": word_at_switch,
                    "recognized_frames": recognized_frames,
                    "fly_hit": fly_hit,
                    "fly_n": fly_n,
                    "fly_acc": None if fly_n == 0 else float(fly_hit) / float(fly_n),
                    "direction_hit": dir_hit,
                    "direction_n": dir_n,
                    "direction_acc": None if dir_n == 0 else float(dir_hit) / float(dir_n),
                    "z_flips": flips,
                    "z_flips_per_s": float(flips) / float(seek_s),
                    "approach_frames": approach_frames,
                    "episodes": episodes,
                    "episodes_at_1m": ended_ok,
                    "hold_share": None if episodes == 0 else float(ended_ok) / float(episodes),
                    "distractor_approach_frames": distractor_frames,
                    "distractor_share": None if approach_frames == 0 else float(distractor_frames) / float(approach_frames),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sim recognition training: one learner, no zones, no reward")
    p.add_argument("--agents", type=int, default=3, help="1 learner + the rest are scenery peers")
    p.add_argument("--seconds", type=float, default=0.0, help="0 = until the window closes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    p.add_argument("--load", action="store_true", help="Load --state before training")
    p.add_argument("--screenshot", type=Path, default=None)
    p.add_argument("--learner", choices=("mb", "hebb"), default="mb")
    p.add_argument("--dan", choices=("teacher", "familiarity"), default="teacher")
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--auto-teach", action="store_true", help="Headless-style T when a peer is alone in view")
    p.add_argument("--punish", action="store_true", help="With --auto-teach, also send PPL1 on distractor-only views")
    p.add_argument(
        "--lidar-refresh",
        type=float,
        default=DEFAULT_LIDAR_REFRESH,
        help="Seconds between fresh lidar windows. 0 keeps the accumulating map.",
    )
    p.add_argument("--return-auto", type=float, default=0.0, help="Idle seconds after takeover before autonomy returns. 0 stays manual.")
    p.add_argument("--steer", choices=("bilateral", "sectors"), default="bilateral", help="bilateral: yaw from R_L - R_R. sectors: the smoothed camera sector.")
    p.add_argument("--seek", action="store_true", help="Headless teacher then autonomy. Prints find/stop counts.")
    p.add_argument("--compare-steer", action="store_true", help="Train once, then seek with sectors and with bilateral.")
    p.add_argument("--rec-fps", type=float, default=2.0, help="Max camera frames per second written while recording.")
    args = p.parse_args(argv)
    if args.rec_fps <= 0:
        print("--rec-fps must be positive", file=sys.stderr)
        return 2
    if args.compare_steer:
        import json

        print(json.dumps(compare_steer(), indent=2))
        return 0
    if args.seek:
        import json

        rows = [seek_trial(seed, state=Path(args.state).with_name(f"seek_{seed}.npz")) for seed in (1, 2, 3)]
        print(json.dumps(rows, indent=2))
        return 0
    session = RecognizeTrainSim(
        n_agents=args.agents,
        seed=args.seed,
        state_path=args.state,
        learner=args.learner,
        dan=args.dan,
        npz=args.npz,
        eta=args.eta,
        auto_teach=args.auto_teach or args.headless,
        punish=args.punish,
        lidar_refresh=args.lidar_refresh,
        return_auto_s=args.return_auto,
        steer=args.steer,
    )
    session.rec_fps = float(args.rec_fps)
    if args.load or Path(args.state).is_file():
        try_load(session)
    if args.headless:
        summary = run_headless(session, args.seconds or 30.0)
    else:
        summary = run_gui(session, seconds=args.seconds, screenshot_path=args.screenshot)
    import json

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
