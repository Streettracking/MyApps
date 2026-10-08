"""One learner, no zones. The fly mushroom body is what learns.

Raw camera/lidar → fixed random projection → PN → KC → MBON.
Only KC→MBON weights change, gated by a DAN. Default DAN is the operator:
T is a treat (appetitive PAM) and X is a punish (aversive PPL1).
Familiarity is optional. The Hebbian layer remains available with --learner hebb.
Scene labels and the D/N keys only choose which monitor curve a sample joins.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from .frame_sense import sim_previews
from .mb_train import MbTrainer, default_npz
from .raw_sense import render_view
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
        self.mb: MbTrainer | None = None
        self.recognizer: ConspecificRecognizer | None = None
        if learner == "mb":
            self.mb = MbTrainer(npz or default_npz(), seed=seed, eta=eta, dan=dan)
            self._log("грибовидное тело  учитель — клавиша T  учатся только KC→MBON")
        else:
            self.recognizer = ConspecificRecognizer()
            self._log("слой сравнения  без зон")
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

    def _move_body(self, agent: AgentState, steer_x: float, steer_z: float) -> None:
        dt = self.world.cfg.dt
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

    def step(self, steer_x: float, steer_z: float, learn: bool | None = None, teach: str | None = None) -> None:
        if learn is None:
            learn = self.learn
        self.world.step_environment()
        self._wander_peers()
        self._move_body(self.learner, steer_x, steer_z)
        feat, hit = render_view(self.learner, self.world, include_agents=True)
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

    def save(self) -> Path:
        if self.mb is not None:
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
            self._log(f"загружены веса KC→MBON  дрейф {self.mb.last_drift:.1f}  лакомств всего {self.mb.progress.base_pam}")
        elif self.recognizer is not None:
            self.recognizer.load(self.state_path)
            self._log(f"загружен прототип  n={self.recognizer.n_self}")

    def reset(self) -> None:
        if self.mb is not None:
            self.mb.reset()
            self._was_rec = False
            self._log("веса и счётчики обучения сброшены")
        elif self.recognizer is not None:
            self.recognizer.reset()
            self._log("прототип сброшен")

    def view(self, focused: bool, udp_status: str, last_command: str, keys_hint: str):
        from .train_monitor import MonitorView

        cam, lid = sim_previews(self.learner, self.world)
        if self.mb is not None:
            caption = "сырой выход: подход − избегание" if self.dan == "teacher" else "сырой выход: минус новизна"
            mode = "сим · лакомство T · без зон · мозг не рулит" if self.dan == "teacher" else "сим · знакомство · без зон · мозг не рулит"
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
        )

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


SIM_KEYS = "стрелки ход   T лакомство   X наказание   B звук   D/N метка   P R S L F12 Esc"
HEBB_KEYS = "стрелки ход   P пауза   R сброс   S/L   F12   Esc"


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


def run_gui(session: RecognizeTrainSim, seconds: float = 0.0, screenshot_path: Path | None = None) -> dict:
    from .train_monitor import TrainMonitor

    mon = TrainMonitor("Go2 recognition trainer — sim")
    hold = False
    steer_x = steer_z = 0.0
    shot = Path(screenshot_path) if screenshot_path else None
    while True:
        inp = mon.pump()
        if inp.quit:
            break
        if inp.label == "dog":
            session.operator = "dog"
            session._log("метка D — только для панели, в обучение не входит")
        elif inp.label == "none":
            session.operator = "none"
            session._log("метка N — только для панели, в обучение не входит")
        if inp.beep_toggle:
            session._log("звук включён" if mon.beep_on else "звук выключен")
        if inp.pause_learn:
            session.learn = not session.learn
            session._log("обучение на паузе" if not session.learn else "обучение продолжается")
        if inp.reset:
            session.reset()
        if inp.save:
            session.save()
        if inp.load:
            try:
                session.load()
            except FileNotFoundError:
                session._log(f"нет файла {session.state_path.name}, продолжаем с текущими весами")
        if inp.estop:
            hold = True
            steer_x = steer_z = 0.0
            session._log("стоп симулятора")
        if inp.stop:
            hold = True
            steer_x = steer_z = 0.0
        if abs(inp.steer_x) + abs(inp.steer_z) > 0:
            hold = False
            steer_x, steer_z = inp.steer_x, inp.steer_z
        elif not hold:
            steer_x, steer_z = session.scripted_steer()
        else:
            steer_x = steer_z = 0.0
        teach = None
        if inp.treat:
            teach = "pam"
        elif inp.punish:
            teach = "ppl1"
        session.step(steer_x, steer_z, teach=teach)
        view = session.view(
            focused=inp.focused,
            udp_status="симулятор, без UDP. Грибовидное тело собаку не ведёт",
            last_command=_command_name(steer_x, steer_z, hold),
            keys_hint=SIM_KEYS if session.learner_kind == "mb" else HEBB_KEYS,
        )
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
    args = p.parse_args(argv)
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
    )
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
