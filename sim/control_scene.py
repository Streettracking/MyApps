"""Conspecific versus a neutral cube, and a no-reward memory probe.

The cube matches the conspecific's angular size and uses a different colour.
YOLO boxes are built only for conspecifics. The cube is written on the ground
row, so its Kenyon pattern is not that body.

``arrange`` can leave both still (``static``) or give them one shared velocity
(``moving-matched``). The matched vector is the conspecific walk speed, across
the learner's heading, so the paths stay parallel and the speed is the same.
Replacing the conspecific with the cube then traces the same trajectory.
The memory probe uses the matched motion: fly vision is motion-sensitive, and
a still cube would let the learner pick "whatever moves".

The probe yaw is the usual left-minus-right law applied to each side's change
from its own pre-training drive. The live command is ``live_turn``: each
hemisphere is divided by its birth scale, and the learned rise is added
before a tanh, so a symmetric view sits near zero and the command does not
pin at ±1.
"""

from __future__ import annotations

import numpy as np

from .approach_watch import ApproachWatch
from .hemifield import DEFAULT_OVERLAP, fly_turn, live_turn
from .mb_train import MbTrainer, default_npz
from .pilot import STOP_M, X_MAX, Pilot
from .search_reflex import SEARCH_REFLEX_YAW
from .raw_sense import (
    BODY_RADIUS,
    CONSPECIFIC_COLOR,
    CONTROL_COLOR,
    FOV,
    N_AZ,
    SEE_M,
    bearing_bin,
    render_view,
)
from .recognize_train import WALK
from .teacher_lead import PAM_GAP_S, attach_recognition_stimuli, mix_velocity
from .world import AgentState, ArenaConfig, ArenaWorld, ControlCube
from .yolo_teacher import DetBox, hemisphere_hits

# Same angular size as the conspecific body. One place for the probe geometry.
CONTROL_RADIUS = BODY_RADIUS
CONSPECIFIC_RADIUS = BODY_RADIUS
PROBE_DIST_M = 1.4
PROBE_BEARING = 0.55
TRAIN_PULSES = 16
BOX_HALF = 0.5 / float(N_AZ)
MOTION_STATIC = "static"
MOTION_MATCHED = "moving-matched"
# Same speed the conspecific walks in the trainer. One constant, two names.
MATCHED_SPEED = WALK
# Closed-loop memory probe. One revolution of search, then a walk that can
# catch the matched speed before the arena wall stops the target.
APPROACH_S = 20.0
SWEEP_YAW = 1.05
APPROACH_WALK = 0.55
GAIN_MARGIN = 40.0
ARENA_INSET = 0.45
# Both objects start inside the 110° view, on opposite sides of the nose.
OPPOSED_BEARING = 0.50


def _rgb_u8(color) -> tuple:
    return tuple(int(round(float(channel) * 255.0)) for channel in color)


def scene_offsets(seed: int) -> tuple:
    """Extra bearing and distance. Seed 1 is the published pose.

    Any other seed moves the objects, so two calls of the probe are not
    the same feature vector. The shift stays inside the field of view.
    """
    if int(seed) == 1:
        return 0.0, 0.0
    rng = np.random.default_rng(int(seed) + 10007)
    return float(rng.uniform(0.12, 0.28)), float(rng.uniform(0.20, 0.55))


def make_scene(seed: int = 1):
    """One learner, one conspecific, one cube. Nothing else is in the arena."""
    bearing, dist = scene_offsets(seed)
    learner = AgentState(agent_id="go2_1", x=3.0, y=2.5, yaw=0.0, color=(80, 160, 255))
    peer = AgentState(
        agent_id="go2_2",
        x=1.6,
        y=2.5,
        yaw=0.0,
        color=_rgb_u8(CONSPECIFIC_COLOR),
    )
    world = ArenaWorld(ArenaConfig(move_zones=False), [], [learner, peer])
    world.controls.append(ControlCube(x=1.6, y=2.2, radius=CONTROL_RADIUS, color=CONTROL_COLOR))
    world.probe_seed = int(seed)
    world.probe_bearing = float(bearing)
    world.probe_dist = float(dist)
    return world, learner


def _spot(world, learner, side: str | None, park: float) -> tuple:
    bearing = PROBE_BEARING + float(getattr(world, "probe_bearing", 0.0))
    dist = PROBE_DIST_M + float(getattr(world, "probe_dist", 0.0))
    if side == "left":
        ang = float(learner.yaw) + bearing
    elif side == "right":
        ang = float(learner.yaw) - bearing
    else:
        ang = float(learner.yaw) + np.pi + float(park)
    return (
        float(learner.x + dist * np.cos(ang)),
        float(learner.y + dist * np.sin(ang)),
    )


def _matched_velocity(learner) -> tuple:
    """Walk speed, perpendicular to the learner's heading. Same vector for both."""
    yaw = float(learner.yaw)
    return (-MATCHED_SPEED * float(np.sin(yaw)), MATCHED_SPEED * float(np.cos(yaw)))


def arrange(
    world,
    learner,
    conspecific: str | None = None,
    control: str | None = None,
    motion: str = MOTION_STATIC,
):
    """Put each object on ``left``, ``right``, or behind the learner when None.

    ``static`` leaves both still. ``moving-matched`` gives both the same
    velocity, so a later step is the same displacement and the same path
    when one object stands in for the other.
    """
    if motion not in (MOTION_STATIC, MOTION_MATCHED):
        raise ValueError("motion must be static or moving-matched")
    peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
    cube = world.controls[0]
    peer.x, peer.y = _spot(world, learner, conspecific, 0.0)
    cube.x, cube.y = _spot(world, learner, control, 0.35)
    vx, vy = _matched_velocity(learner) if motion == MOTION_MATCHED else (0.0, 0.0)
    peer.vx, peer.vy = float(vx), float(vy)
    cube.vx, cube.vy = float(vx), float(vy)
    return peer, cube


def advance_scene(world, learner, dt: float):
    """Step the conspecific and the cube by the velocity ``arrange`` set.

    Both carry one vector, so the displacement matches. This is not the arena
    bounce: that integrator reflects each object on its own and would split
    a shared trajectory at the wall.
    """
    peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
    cube = world.controls[0]
    step = float(dt)
    for obj in (peer, cube):
        obj.x = float(obj.x + float(obj.vx) * step)
        obj.y = float(obj.y + float(obj.vy) * step)
    return peer, cube


def conspecific_boxes(learner, world) -> list:
    """YOLO boxes for conspecifics in view. The cube is never one of them."""
    boxes = []
    for other in world.agents:
        if other.agent_id == learner.agent_id:
            continue
        dx = other.x - learner.x
        dy = other.y - learner.y
        dist = float(np.hypot(dx, dy))
        bx = bearing_bin(dx, dy, float(learner.yaw))
        if bx is None or dist > SEE_M or dist < 1e-3:
            continue
        cx = (N_AZ - 1 - int(bx) + 0.5) / float(N_AZ)
        boxes.append(DetBox(cx - BOX_HALF, 0.16, cx + BOX_HALF, 0.46, 0.92))
    return boxes


def _view(learner, world):
    return render_view(learner, world, include_agents=True)


def _hemidrives(trainer, feat) -> tuple:
    trainer.forward(feat)
    left = float(trainer.brain.appetitive_drive(trainer.last_fwd.action_scores))
    right = float(trainer.brain_r.appetitive_drive(trainer.last_fwd_r.action_scores))
    return left, right


def _isolated(trainer, world, learner, kind: str, side: str, motion: str):
    arrange(
        world,
        learner,
        conspecific=side if kind == "conspecific" else None,
        control=side if kind == "control" else None,
        motion=motion,
    )
    feat, hit = _view(learner, world)
    drive = float(trainer.score_feature(feat))
    left, right = _hemidrives(trainer, feat)
    return drive, left, right, hit


def _train_side(trainer, world, learner, side: str, t0: float, motion: str) -> float:
    arrange(world, learner, conspecific=side, control=None, motion=motion)
    feat, _hit = _view(learner, world)
    boxes = conspecific_boxes(learner, world)
    stim = trainer.stimuli.get("conspecific")
    overlap = float(trainer.overlap)
    now = float(t0)
    for _ in range(TRAIN_PULSES):
        trainer.forward(feat)
        hit_l, hit_r = hemisphere_hits(boxes, overlap, center_only=True)
        stim.observe(True, hit_l, hit_r, False, True, False, False, hit_l, hit_r)
        kind_l, kind_r = trainer.reinforce_stimuli(now)
        trainer.settle(kind_l in ("pam", "ppl1"), kind_r in ("pam", "ppl1"))
        now += PAM_GAP_S
    return now


def _fly_command(gain_l: float, gain_r: float):
    """Bilateral yaw from the two learned gains, then teacher weight 0."""
    pilot = Pilot()
    pilot.start_auto(0.0)
    pilot.set_steer("bilateral")
    cmd = pilot.command(
        1.0,
        (0.0, 0.0),
        focused=True,
        frames_ok=True,
        link_ok=True,
        recognized=True,
        sector=6,
        dist_m=PROBE_DIST_M,
        forward_m=3.0,
        r_l=float(gain_l),
        r_r=float(gain_r),
        recognized_l=True,
        recognized_r=True,
    )
    _mixed_x, mixed_z = mix_velocity(float(cmd.x), float(cmd.z), 0.0, -float(cmd.z), 0.0)
    return cmd, float(mixed_z)


def object_drive_rows(when: str, conspecific: float, control: float) -> list:
    return [
        {"name": "conspecific", "drive": float(conspecific), "when": when},
        {"name": "control", "drive": float(control), "when": when},
    ]


def run_memory_probe(seed: int = 1, npz=None, motion: str = MOTION_MATCHED) -> dict:
    """Train on the conspecific only, then probe with reinforcement off.

    The default motion matches the conspecific, so the control is not the
    only still object in view. Returns the status fields and the log lines.
    ``z`` is the fly's yaw.
    """
    if motion not in (MOTION_STATIC, MOTION_MATCHED):
        raise ValueError("motion must be static or moving-matched")
    world, learner = make_scene(seed)
    trainer = MbTrainer(npz or default_npz(), seed=int(seed), camera_only=True, overlap=DEFAULT_OVERLAP)
    trainer.learn = True
    attach_recognition_stimuli(trainer.stimuli)
    log = []

    before = {}
    before_hemi = {}
    for kind in ("conspecific", "control"):
        for side in ("left", "right"):
            drive, left, right, _hit = _isolated(trainer, world, learner, kind, side, motion)
            before_hemi[(kind, side)] = (left, right)
            if side == "left":
                before[kind] = drive

    rows = object_drive_rows("before", before["conspecific"], before["control"])
    log.append(
        "до обучения  сородич %+.1f  контроль %+.1f" % (before["conspecific"], before["control"])
    )

    now = 0.0
    for side in ("left", "right"):
        now = _train_side(trainer, world, learner, side, now, motion)

    after = {}
    for kind in ("conspecific", "control"):
        drive, _left, _right, _hit = _isolated(trainer, world, learner, kind, "left", motion)
        after[kind] = drive
    rows.extend(object_drive_rows("after", after["conspecific"], after["control"]))
    trainer.object_drives = list(rows)
    log.append(
        "после %d импульсов на сторону  сородич %+.1f  контроль %+.1f"
        % (TRAIN_PULSES, after["conspecific"], after["control"])
    )

    trainer.set_reinforcement_off(True)
    held_l = np.array(trainer.brain.kc_mbon_w, copy=True)
    held_r = np.array(trainer.brain_r.kc_mbon_w, copy=True)
    stim = trainer.stimuli.get("conspecific")
    probes = {}
    for name, conspecific_side, control_side in (
        ("conspecific_left", "left", "right"),
        ("conspecific_right", "right", "left"),
    ):
        arrange(world, learner, conspecific=conspecific_side, control=control_side, motion=motion)
        feat, _hit = _view(learner, world)
        boxes = conspecific_boxes(learner, world)
        hit_l, hit_r = hemisphere_hits(boxes, float(trainer.overlap), center_only=True)
        stim.observe(True, hit_l, hit_r, False, True, False, False, hit_l, hit_r)
        trainer.forward(feat)
        trainer.reinforce_stimuli(now)
        trainer.teach(trainer.last_fwd, "pam", now)
        trainer.settle(True, True)
        now += PAM_GAP_S
        left = float(trainer.brain.appetitive_drive(trainer.last_fwd.action_scores))
        right = float(trainer.brain_r.appetitive_drive(trainer.last_fwd_r.action_scores))
        if conspecific_side == "left":
            base_l = before_hemi[("conspecific", "left")][0]
            base_r = before_hemi[("control", "right")][1]
        else:
            base_l = before_hemi[("control", "left")][0]
            base_r = before_hemi[("conspecific", "right")][1]
        gain_l = float(left - base_l)
        gain_r = float(right - base_r)
        cmd, mixed_z = _fly_command(gain_l, gain_r)
        yaw = float(fly_turn(gain_l, gain_r))
        probes[name] = {
            "drive_l": left,
            "drive_r": right,
            "gain_l": gain_l,
            "gain_r": gain_r,
            "z": float(cmd.z),
            "yaw": yaw,
            "mixed_z": mixed_z,
            "boxes": len(boxes),
        }
        where = "сородич слева" if conspecific_side == "left" else "сородич справа"
        log.append("проба  %s  z=%+.3f  прирост Л %+.1f  П %+.1f" % (where, cmd.z, gain_l, gain_r))

    frozen = bool(np.array_equal(trainer.brain.kc_mbon_w, held_l) and np.array_equal(trainer.brain_r.kc_mbon_w, held_r))
    log.append("подкрепление выключено  вес учителя 0")
    if motion == MOTION_MATCHED:
        log.append("движение как у сородича  скорость %.2f м/с" % MATCHED_SPEED)
    else:
        log.append("движение статично")
    return {
        "reinforcement_off": bool(trainer.reinforcement_off),
        "teacher_weight": 0.0,
        "motion": motion,
        "train_pulses": TRAIN_PULSES,
        "before": before,
        "after": after,
        "object_drives": rows,
        "probe": probes,
        "weights_frozen": frozen,
        "seed": int(seed),
        "log": log,
    }


def _wrap(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _birth_drives(trainer, birth_l, birth_r, feat) -> tuple:
    """Hemisphere drives of the birth weights. The live weights are put back."""
    cur_l = trainer.brain.kc_mbon_w
    cur_r = trainer.brain_r.kc_mbon_w
    trainer.brain.kc_mbon_w = birth_l
    trainer.brain_r.kc_mbon_w = birth_r
    try:
        return _hemidrives(trainer, feat)
    finally:
        trainer.brain.kc_mbon_w = cur_l
        trainer.brain_r.kc_mbon_w = cur_r


def _hold_inside(obj, cfg) -> None:
    """Stop an object on the arena wall. A 20 s walk cannot chase it out of the map."""
    x0, x1 = ARENA_INSET, float(cfg.width) - ARENA_INSET
    y0, y1 = ARENA_INSET, float(cfg.height) - ARENA_INSET
    x = float(min(max(float(obj.x), x0), x1))
    y = float(min(max(float(obj.y), y0), y1))
    if x != float(obj.x) or y != float(obj.y):
        obj.vx = 0.0
        obj.vy = 0.0
    obj.x, obj.y = x, y


def _place_pair(world, learner, swap: bool, motion: str) -> tuple:
    """Conspecific and cube behind the learner, symmetric, out of the forward view."""
    learner.x, learner.y, learner.yaw = 3.0, 2.5, 0.0
    learner.vx = learner.vy = 0.0
    peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
    cube = world.controls[0]
    sign = -1.0 if swap else 1.0
    bearing = float(getattr(world, "probe_bearing", 0.0))
    peer.x, peer.y = learner.x - 1.5, learner.y + sign * 0.9 + bearing
    cube.x, cube.y = learner.x - 1.5, learner.y - sign * 0.9 - bearing
    speed = MATCHED_SPEED if motion == MOTION_MATCHED else 0.0
    peer.vx, cube.vx = 0.0, 0.0
    peer.vy, cube.vy = speed, speed
    return peer, cube


def _approach_trial(trainer, world, learner, birth_l, birth_r, motion: str, swap: bool) -> dict:
    """Spin until each object has been seen alone, then walk to the higher gain.

    The yaw that chooses a side is the change from that side's own birth drive.
    The raw left-minus-right law is left for every visible pattern, so it is
    recorded and not used to steer. Search keeps turning until the view has
    held each object by itself. A tie (no learned margin) walks to the nearer.
    """
    peer, cube = _place_pair(world, learner, swap, motion)
    dt = float(world.cfg.dt)
    steps = int(round(APPROACH_S / dt))
    half = float(FOV) / 2.0
    best = {"conspecific": None, "control": None}
    aimed = {"conspecific": 0, "control": 0}
    seen = {"conspecific": 0, "control": 0}
    phases = {"search": 0, "approach": 0, "hold": 0}
    min_d = {"conspecific": 99.0, "control": 99.0}
    turned = 0.0
    for _ in range(steps):
        if motion == MOTION_MATCHED:
            advance_scene(world, learner, dt)
            _hold_inside(peer, world.cfg)
            _hold_inside(cube, world.cfg)
        feat, hit = _view(learner, world)
        live_l, live_r = _hemidrives(trainer, feat)
        base_l, base_r = _birth_drives(trainer, birth_l, birth_r, feat)
        gain_l = float(live_l - base_l)
        gain_r = float(live_r - base_r)
        dist_dog = float(np.hypot(peer.x - learner.x, peer.y - learner.y))
        dist_cube = float(np.hypot(cube.x - learner.x, cube.y - learner.y))
        min_d["conspecific"] = min(min_d["conspecific"], dist_dog)
        min_d["control"] = min(min_d["control"], dist_cube)
        ang_dog = _wrap(float(np.arctan2(peer.y - learner.y, peer.x - learner.x)) - learner.yaw)
        ang_cube = _wrap(float(np.arctan2(cube.y - learner.y, cube.x - learner.x)) - learner.yaw)
        dog_on = bool(hit.dog) and abs(ang_dog) <= half
        cube_on = bool(hit.control) and abs(ang_cube) <= half
        if dog_on:
            seen["conspecific"] += 1
        if cube_on:
            seen["control"] += 1
        if dog_on and not cube_on:
            val = gain_l if ang_dog >= 0.0 else gain_r
            cur = best["conspecific"]
            best["conspecific"] = val if cur is None else max(cur, val)
        if cube_on and not dog_on:
            val = gain_l if ang_cube >= 0.0 else gain_r
            cur = best["control"]
            best["control"] = val if cur is None else max(cur, val)
        full = turned >= 2.0 * np.pi - 0.25
        both = best["conspecific"] is not None and best["control"] is not None
        chosen = None
        if both or (full and (best["conspecific"] is not None or best["control"] is not None)):
            score_dog = -1e9 if best["conspecific"] is None else float(best["conspecific"])
            score_cube = -1e9 if best["control"] is None else float(best["control"])
            if score_dog > score_cube + GAIN_MARGIN:
                chosen = "conspecific"
            elif score_cube > score_dog + GAIN_MARGIN:
                chosen = "control"
            else:
                chosen = "conspecific" if dist_dog <= dist_cube else "control"
        if chosen is None:
            yaw, speed, phase = SWEEP_YAW, 0.0, "search"
        else:
            ang = ang_dog if chosen == "conspecific" else ang_cube
            dist = dist_dog if chosen == "conspecific" else dist_cube
            if abs(ang) > half:
                yaw = SWEEP_YAW if ang > 0.0 else -SWEEP_YAW
                speed, phase = 0.0, "search"
            else:
                yaw = float(np.clip(ang / 0.35, -1.0, 1.0))
                if dist <= STOP_M:
                    speed, phase = 0.0, "hold"
                else:
                    speed, phase = APPROACH_WALK, "approach"
            aimed[chosen] += 1
        phases[phase] += 1
        prev = float(learner.yaw)
        learner.yaw = _wrap(prev + yaw * dt)
        turned += abs(_wrap(learner.yaw - prev))
        learner.x = float(learner.x + speed * np.cos(learner.yaw) * dt)
        learner.y = float(learner.y + speed * np.sin(learner.yaw) * dt)
        _hold_inside(learner, world.cfg)
    def _rate(name: str) -> float:
        return float(aimed[name]) / float(steps)

    return {
        "steps": steps,
        "rate_conspecific": _rate("conspecific"),
        "rate_control": _rate("control"),
        "reached_conspecific": bool(min_d["conspecific"] <= STOP_M + 0.05),
        "reached_control": bool(min_d["control"] <= STOP_M + 0.05),
        "min_conspecific": float(min_d["conspecific"]),
        "min_control": float(min_d["control"]),
        "seen_conspecific": int(seen["conspecific"]),
        "seen_control": int(seen["control"]),
        "search_steps": int(phases["search"]),
        "approach_steps": int(phases["approach"]),
        "hold_steps": int(phases["hold"]),
        "score_conspecific": best["conspecific"],
        "score_control": best["control"],
    }


def _isolated_turn(trainer, world, learner, birth_l, birth_r, kind: str, side: str) -> tuple:
    arrange(
        world,
        learner,
        conspecific=side if kind == "conspecific" else None,
        control=side if kind == "control" else None,
        motion=MOTION_STATIC,
    )
    feat, _hit = _view(learner, world)
    live_l, live_r = _hemidrives(trainer, feat)
    base_l, base_r = _birth_drives(trainer, birth_l, birth_r, feat)
    return float(fly_turn(live_l, live_r)), float(fly_turn(live_l - base_l, live_r - base_r))


def conspecific_side(seed: int) -> str:
    """Odd runs put the conspecific on the left. Even runs put it on the right."""
    return "left" if int(seed) % 2 == 1 else "right"


def _run_shift(seed: int, run: int) -> tuple:
    """Extra bearing and distance for a repeat. Run 0 adds nothing.

    The seed's own offset already lives on the world. A later run of that
    seed moves the pair again, so the repeats are not one copied trace.
    """
    if int(run) <= 0:
        return 0.0, 0.0
    rng = np.random.default_rng(10007 * int(seed) + 17 * int(run))
    return float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.08, 0.08))


def _place_opposed(world, learner, side: str, motion: str, run: int = 0) -> tuple:
    """Conspecific and cube ahead, one in each hemifield, both inside the view.

    Seed 1, run 0 is the published pose. Any other seed uses ``scene_offsets``.
    A later run of the same seed adds a small shift.
    """
    learner.x, learner.y, learner.yaw = 3.0, 2.5, 0.0
    learner.vx = learner.vy = 0.0
    peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
    cube = world.controls[0]
    extra_b, extra_d = _run_shift(int(getattr(world, "probe_seed", 1)), int(run))
    magnitude = OPPOSED_BEARING + float(getattr(world, "probe_bearing", 0.0)) + extra_b
    magnitude = float(min(max(magnitude, 0.22), 0.80))
    bearing = magnitude if side == "left" else -magnitude
    dist = PROBE_DIST_M + float(getattr(world, "probe_dist", 0.0)) + extra_d
    dist = float(min(max(dist, 1.15), 2.2))
    peer.x = float(learner.x + dist * np.cos(bearing))
    peer.y = float(learner.y + dist * np.sin(bearing))
    cube.x = float(learner.x + dist * np.cos(-bearing))
    cube.y = float(learner.y + dist * np.sin(-bearing))
    world.opposed_bearing = float(magnitude)
    world.opposed_dist = float(dist)
    speed = MATCHED_SPEED if motion == MOTION_MATCHED else 0.0
    peer.vx, cube.vx = 0.0, 0.0
    peer.vy, cube.vy = speed, speed
    return peer, cube


def _bearing(obj, learner) -> float:
    return _wrap(float(np.arctan2(obj.y - learner.y, obj.x - learner.x)) - learner.yaw)


def _opposed_trial(trainer, world, learner, motion: str, side: str, run: int = 0) -> dict:
    """Start with both targets in view. Empty frames scan. A target uses live_turn.

    The yaw is the normalised live command, not the saturating ratio.
    Alternating ``side`` across runs puts the conspecific on the other hemifield.
    The preference index is time nearer the conspecific minus time nearer the
    cube, over the run. Nearer means in view and closer to the nose.
    """
    peer, cube = _place_opposed(world, learner, side, motion, run)
    dt = float(world.cfg.dt)
    steps = int(round(APPROACH_S / dt))
    half = float(FOV) / 2.0
    open_dog = abs(_bearing(peer, learner)) <= half
    open_cube = abs(_bearing(cube, learner)) <= half
    aimed = {"conspecific": 0, "control": 0}
    seen = {"conspecific": 0, "control": 0}
    phases = {"search": 0, "approach": 0, "hold": 0}
    min_d = {"conspecific": 99.0, "control": 99.0}
    watch = ApproachWatch()
    feat0, _hit0 = _view(learner, world)
    live0_l, live0_r = _hemidrives(trainer, feat0)
    z0 = float(live_turn(live0_l, live0_r, trainer.base_l, trainer.base_r))
    for _ in range(steps):
        if motion == MOTION_MATCHED:
            advance_scene(world, learner, dt)
            _hold_inside(peer, world.cfg)
            _hold_inside(cube, world.cfg)
        feat, hit = _view(learner, world)
        live_l, live_r = _hemidrives(trainer, feat)
        base_l = float(trainer.base_l)
        base_r = float(trainer.base_r)
        dist_dog = float(np.hypot(peer.x - learner.x, peer.y - learner.y))
        dist_cube = float(np.hypot(cube.x - learner.x, cube.y - learner.y))
        min_d["conspecific"] = min(min_d["conspecific"], dist_dog)
        min_d["control"] = min(min_d["control"], dist_cube)
        ang_dog = _bearing(peer, learner)
        ang_cube = _bearing(cube, learner)
        dog_on = bool(hit.dog) and abs(ang_dog) <= half
        cube_on = bool(hit.control) and abs(ang_cube) <= half
        if dog_on:
            seen["conspecific"] += 1
        if cube_on:
            seen["control"] += 1
        if not dog_on and not cube_on:
            yaw, speed, phase = SEARCH_REFLEX_YAW, 0.0, "search"
        else:
            yaw = float(live_turn(live_l, live_r, base_l, base_r))
            near = 99.0
            if dog_on:
                near = min(near, dist_dog)
            if cube_on:
                near = min(near, dist_cube)
            if near <= STOP_M:
                speed, phase = 0.0, "hold"
            else:
                speed, phase = float(X_MAX), "approach"
            if yaw > 0.08:
                chosen = "conspecific" if ang_dog >= ang_cube else "control"
            elif yaw < -0.08:
                chosen = "conspecific" if ang_dog <= ang_cube else "control"
            else:
                chosen = "conspecific" if dist_dog <= dist_cube else "control"
            aimed[chosen] += 1
        # Near means in view and clearly closer to the nose. A tie counts as
        # neither, so an untrained straight walk does not invent a side.
        margin = 0.08
        if dog_on and not cube_on:
            dog_near, cube_near = True, False
        elif cube_on and not dog_on:
            dog_near, cube_near = False, True
        elif dog_on and cube_on and abs(ang_dog) + margin < abs(ang_cube):
            dog_near, cube_near = True, False
        elif dog_on and cube_on and abs(ang_cube) + margin < abs(ang_dog):
            dog_near, cube_near = False, True
        else:
            dog_near, cube_near = False, False
        watch.note(dt, dog_near, cube_near, 0.5 * (float(live_l) + float(live_r)))
        phases[phase] += 1
        learner.yaw = _wrap(float(learner.yaw) + yaw * dt)
        learner.x = float(learner.x + speed * np.cos(learner.yaw) * dt)
        learner.y = float(learner.y + speed * np.sin(learner.yaw) * dt)
        _hold_inside(learner, world.cfg)

    def _rate(name: str) -> float:
        return float(aimed[name]) / float(steps)

    return {
        "steps": steps,
        "side": side,
        "in_view_conspecific": bool(open_dog),
        "in_view_control": bool(open_cube),
        "rate_conspecific": _rate("conspecific"),
        "rate_control": _rate("control"),
        "reached_conspecific": bool(min_d["conspecific"] <= STOP_M + 0.05),
        "reached_control": bool(min_d["control"] <= STOP_M + 0.05),
        "min_conspecific": float(min_d["conspecific"]),
        "min_control": float(min_d["control"]),
        "seen_conspecific": int(seen["conspecific"]),
        "seen_control": int(seen["control"]),
        "search_steps": int(phases["search"]),
        "approach_steps": int(phases["approach"]),
        "hold_steps": int(phases["hold"]),
        "score_conspecific": None,
        "score_control": None,
        "index": None if watch.index() is None else float(watch.index()),
        "z": float(z0),
        "bearing": float(getattr(world, "opposed_bearing", OPPOSED_BEARING)),
        "dist": float(getattr(world, "opposed_dist", PROBE_DIST_M)),
        "seed": int(getattr(world, "probe_seed", 1)),
    }


def run_approach_probe(seed: int = 1, npz=None, motion: str = MOTION_MATCHED, placement: str = "behind") -> dict:
    """20 s with no teacher, before training, after it, and after swapping places.

    ``behind`` keeps both objects out of the forward view and homes on the
    learned gain. ``opposed`` starts with both in view, conspecific on the
    left for an odd seed and on the right for an even seed. The swap trial
    of an opposed run uses the other side, so one call covers both hemifields.
    """
    if motion not in (MOTION_STATIC, MOTION_MATCHED):
        raise ValueError("motion must be static or moving-matched")
    if placement not in ("behind", "opposed"):
        raise ValueError("placement must be behind or opposed")
    world, learner = make_scene(seed)
    trainer = MbTrainer(npz or default_npz(), seed=int(seed), camera_only=True, overlap=DEFAULT_OVERLAP)
    trainer.learn = True
    attach_recognition_stimuli(trainer.stimuli)
    birth_l = np.array(trainer.brain.kc_mbon_w, copy=True)
    birth_r = np.array(trainer.brain_r.kc_mbon_w, copy=True)
    side = conspecific_side(seed)
    other = "right" if side == "left" else "left"
    if placement == "opposed":
        before = _opposed_trial(trainer, world, learner, motion, side)
    else:
        before = _approach_trial(trainer, world, learner, birth_l, birth_r, motion, False)
    now = 0.0
    for train_side in ("left", "right"):
        now = _train_side(trainer, world, learner, train_side, now, motion)
    raw_left, gain_left = _isolated_turn(trainer, world, learner, birth_l, birth_r, "conspecific", "left")
    raw_right, gain_right = _isolated_turn(trainer, world, learner, birth_l, birth_r, "conspecific", "right")
    if placement == "opposed":
        after = _opposed_trial(trainer, world, learner, motion, side)
        swapped = _opposed_trial(trainer, world, learner, motion, other)
    else:
        after = _approach_trial(trainer, world, learner, birth_l, birth_r, motion, False)
        swapped = _approach_trial(trainer, world, learner, birth_l, birth_r, motion, True)
    return {
        "seed": int(seed),
        "motion": motion,
        "placement": placement,
        "side": side if placement == "opposed" else "",
        "seconds": APPROACH_S,
        "before": before,
        "after": after,
        "swap": swapped,
        "turn": {
            "raw_left": raw_left,
            "raw_right": raw_right,
            "gain_left": gain_left,
            "gain_right": gain_right,
        },
    }


def _mean(values) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def run_live_memory_trial(seeds=(1, 2, 3), runs: int = 4, motion: str = MOTION_STATIC, npz=None) -> dict:
    """Both targets in view. Sides alternate. Teacher weight is 0. Reinforcement is off.

    Each seed places the pair on its own bearing and distance. Each run swaps
    which hemifield holds the conspecific. The preference index is time with
    the conspecific in view minus time with the cube in view, averaged across
    the two sides.
    """
    if motion not in (MOTION_STATIC, MOTION_MATCHED):
        raise ValueError("motion must be static or moving-matched")
    if int(runs) < 1:
        raise ValueError("runs must be at least 1")
    rows = []
    for seed in seeds:
        world, learner = make_scene(int(seed))
        trainer = MbTrainer(npz or default_npz(), seed=int(seed), camera_only=True, overlap=DEFAULT_OVERLAP)
        trainer.learn = True
        attach_recognition_stimuli(trainer.stimuli)
        trainer.set_reinforcement_off(True)
        for run in range(int(runs)):
            side = "left" if int(run) % 2 == 0 else "right"
            row = _opposed_trial(trainer, world, learner, motion, side, run)
            row["trained"] = False
            row["run"] = int(run)
            row["teacher_weight"] = 0.0
            row["reinforcement_off"] = True
            rows.append(row)
        trainer.set_reinforcement_off(False)
        now = 0.0
        for train_side in ("left", "right"):
            now = _train_side(trainer, world, learner, train_side, now, motion)
        trainer.set_reinforcement_off(True)
        held_l = np.array(trainer.brain.kc_mbon_w, copy=True)
        held_r = np.array(trainer.brain_r.kc_mbon_w, copy=True)
        for run in range(int(runs)):
            side = "left" if int(run) % 2 == 0 else "right"
            row = _opposed_trial(trainer, world, learner, motion, side, run)
            row["trained"] = True
            row["run"] = int(run)
            row["teacher_weight"] = 0.0
            row["reinforcement_off"] = True
            rows.append(row)
        frozen = bool(
            np.array_equal(trainer.brain.kc_mbon_w, held_l)
            and np.array_equal(trainer.brain_r.kc_mbon_w, held_r)
        )
        for row in rows:
            if int(row["seed"]) == int(seed) and row["trained"]:
                row["weights_frozen"] = frozen

    def _pick(trained: bool, side: str) -> float:
        return _mean([float(row["index"]) for row in rows if row["trained"] is trained and row["side"] == side])

    left_untrained = _pick(False, "left")
    right_untrained = _pick(False, "right")
    left_trained = _pick(True, "left")
    right_trained = _pick(True, "right")
    return {
        "seeds": [int(seed) for seed in seeds],
        "runs": int(runs),
        "motion": motion,
        "teacher_weight": 0.0,
        "reinforcement_off": True,
        "rows": rows,
        "untrained_left": left_untrained,
        "untrained_right": right_untrained,
        "trained_left": left_trained,
        "trained_right": right_trained,
        "untrained": 0.5 * (left_untrained + right_untrained),
        "trained": 0.5 * (left_trained + right_trained),
    }
