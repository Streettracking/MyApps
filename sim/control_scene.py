"""Conspecific versus a neutral cube, and a no-reward memory probe.

The cube matches the conspecific's angular size and uses a different colour.
YOLO boxes are built only for conspecifics. The cube is written on the ground
row, so its Kenyon pattern is not that body.

The probe yaw is the usual left-minus-right law applied to each side's change
from its own pre-training drive. The right mushroom body is born with a
negative approach-minus-avoidance for every pattern, so the raw difference
points left even after that side has learned.
"""

from __future__ import annotations

import numpy as np

from .hemifield import DEFAULT_OVERLAP, fly_turn
from .mb_train import MbTrainer, default_npz
from .pilot import Pilot
from .raw_sense import (
    BODY_RADIUS,
    CONSPECIFIC_COLOR,
    CONTROL_COLOR,
    N_AZ,
    SEE_M,
    bearing_bin,
    render_view,
)
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


def _rgb_u8(color) -> tuple:
    return tuple(int(round(float(channel) * 255.0)) for channel in color)


def make_scene(seed: int = 1):
    """One learner, one conspecific, one cube. Nothing else is in the arena."""
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
    return world, learner


def _spot(learner, side: str | None, park: float) -> tuple:
    if side == "left":
        ang = float(learner.yaw) + PROBE_BEARING
        dist = PROBE_DIST_M
    elif side == "right":
        ang = float(learner.yaw) - PROBE_BEARING
        dist = PROBE_DIST_M
    else:
        ang = float(learner.yaw) + np.pi + float(park)
        dist = PROBE_DIST_M
    return (
        float(learner.x + dist * np.cos(ang)),
        float(learner.y + dist * np.sin(ang)),
    )


def arrange(world, learner, conspecific: str | None = None, control: str | None = None):
    """Put each object on ``left``, ``right``, or behind the learner when None."""
    peer = next(agent for agent in world.agents if agent.agent_id != learner.agent_id)
    cube = world.controls[0]
    peer.x, peer.y = _spot(learner, conspecific, 0.0)
    peer.vx = peer.vy = 0.0
    cube.x, cube.y = _spot(learner, control, 0.35)
    cube.vx = cube.vy = 0.0
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


def _isolated(trainer, world, learner, kind: str, side: str):
    arrange(
        world,
        learner,
        conspecific=side if kind == "conspecific" else None,
        control=side if kind == "control" else None,
    )
    feat, hit = _view(learner, world)
    drive = float(trainer.score_feature(feat))
    left, right = _hemidrives(trainer, feat)
    return drive, left, right, hit


def _train_side(trainer, world, learner, side: str, t0: float) -> float:
    arrange(world, learner, conspecific=side, control=None)
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


def run_memory_probe(seed: int = 1, npz=None) -> dict:
    """Train on the conspecific only, then probe with reinforcement off.

    Returns the status fields and the log lines. ``z`` is the fly's yaw.
    """
    world, learner = make_scene(seed)
    trainer = MbTrainer(npz or default_npz(), seed=int(seed), camera_only=True, overlap=DEFAULT_OVERLAP)
    trainer.learn = True
    attach_recognition_stimuli(trainer.stimuli)
    log = []

    before = {}
    before_hemi = {}
    for kind in ("conspecific", "control"):
        for side in ("left", "right"):
            drive, left, right, _hit = _isolated(trainer, world, learner, kind, side)
            before_hemi[(kind, side)] = (left, right)
            if side == "left":
                before[kind] = drive

    rows = object_drive_rows("before", before["conspecific"], before["control"])
    log.append(
        "до обучения  сородич %+.1f  контроль %+.1f" % (before["conspecific"], before["control"])
    )

    now = 0.0
    for side in ("left", "right"):
        now = _train_side(trainer, world, learner, side, now)

    after = {}
    for kind in ("conspecific", "control"):
        drive, _left, _right, _hit = _isolated(trainer, world, learner, kind, "left")
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
        arrange(world, learner, conspecific=conspecific_side, control=control_side)
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
    return {
        "reinforcement_off": bool(trainer.reinforcement_off),
        "teacher_weight": 0.0,
        "train_pulses": TRAIN_PULSES,
        "before": before,
        "after": after,
        "object_drives": rows,
        "probe": probes,
        "weights_frozen": frozen,
        "log": log,
    }
