"""The single-hand tasks the bimanual pour is split into, for recording and evaluation.

Each task belongs to one hand. A policy for it observes the head camera and
that side's arm and hand joints, and commands the same joints (27 values:
Joint1..7, then the hand's 20 joints in model order). The other arm stays at
HOME. Object poses are read here only to randomize the start and to score.

- cup_bring (left hand): pick the cup up from the table and hold it at
  CUP_GOAL, beside the bottle, where the right hand will pour into it.
- pour (right hand): with the cup already held at CUP_GOAL by the left hand
  (the cup_bring expert runs before the episode starts) and the cap already
  off, grasp the bottle, pour its contents into the cup, and stand it up.

Each task also has a scripted expert (`expert`) that produces a
demonstration from the true object pose; these demonstrations are the
training data for imitation learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Callable, Iterator

import mujoco
import numpy as np

from scipy.spatial.transform import Rotation as R

from .ik import ArmIK
from .ik import SiteIK
from .props import (CAP_HALF_OPEN_CLOSE_TURNS, CAP_HALF_HEIGHT, CAP_STAND_HEIGHT, CAP_HALF_OPEN_RAISE, CAP_HALF_OPEN_TURNS, CAP_RADIUS, CAP_TOP, BottleNames,
                    CapThread, particles_in_cup)
from .scene import ARM_JOINTS, CAP_STAND_POS, Names, build_model, set_home
from .sensors import Observer

FPS = 30
TABLE_Z = 0.70  # table top (objects.TABLE_TOP_Z)
STAND_TOP = CAP_STAND_POS[2] + CAP_STAND_HEIGHT
CUP_GOAL = np.array([0.44, 0.0, 0.76])  # cup bottom centre while held for pouring
# Fingers closed around a cylinder (a power grasp).
_FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
POWER_CLOSED = {
    **{f"{f}_mcp_flex": 1.2 for f in _FINGERS},
    **{f"{f}_pip": 1.4 for f in _FINGERS},
    **{f"{f}_dip": 1.0 for f in _FINGERS},
}


def side_joints(model: mujoco.MjModel, side: str) -> list[str]:
    """Joint1..7 of that arm, then that hand's 20 joints in model order."""
    prefix = Names.hand_prefix(side)
    return [Names.arm_joint(side, j) for j in ARM_JOINTS] + [
        model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith(prefix)
    ]


@dataclass
class Episode:
    model: mujoco.MjModel
    data: mujoco.MjData
    joints: list[str]
    actuators: np.ndarray
    qadr: np.ndarray
    extra: dict = field(default_factory=dict)  # per-episode task state (a cap's thread, say)

    def state(self) -> np.ndarray:
        return self.data.qpos[self.qadr].astype(np.float32)

    def action(self) -> np.ndarray:
        return self.data.ctrl[self.actuators].astype(np.float32)


class Task:
    name: str
    side: str
    instruction: str
    seconds: float  # episode length limit

    def __init__(self, n_particles: int = 0, model: mujoco.MjModel | None = None) -> None:
        self.model = model if model is not None else build_model("pour", n_particles=n_particles)
        self._observer: Observer | None = None
        self.joints = side_joints(self.model, self.side)
        act = {self.model.joint(self.model.actuator_trnid[a, 0]).name: a for a in range(self.model.nu)}
        self.actuators = np.array([act[j] for j in self.joints])
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in self.joints])

    @property
    def observer(self) -> Observer:
        """The head camera, made on first use (headless training runs never render)."""
        if self._observer is None:
            self._observer = Observer(self.model)
        return self._observer

    def reset(self, rng: np.random.Generator) -> Episode:
        data = mujoco.MjData(self.model)
        set_home(self.model, data)
        self.randomize(data, rng)
        mujoco.mj_forward(self.model, data)
        return self.begin(data)

    def begin(self, data: mujoco.MjData) -> Episode:
        """An episode of this task on an existing state (a fresh scene, or where earlier tasks left it)."""
        return Episode(self.model, data, self.joints, self.actuators, self.qadr)

    def randomize(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        pass

    def after_step(self, ep: Episode) -> None:
        """Called after every physics step of an episode (all callers step through `step`): keeps
        the cap's thread (props.CapThread) current in every stage that carries one."""
        thread = ep.extra.get("thread")
        if thread is not None:
            thread.update(ep.data)

    def step(self, ep: Episode) -> None:
        mujoco.mj_step(ep.model, ep.data)
        self.after_step(ep)

    def success(self, ep: Episode) -> bool:
        raise NotImplementedError

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        """Yields (seconds, 27 targets) segments: move linearly to the targets over that time."""
        raise NotImplementedError


def _set_free_xy(model: mujoco.MjModel, data: mujoco.MjData, body: str, xy: np.ndarray) -> None:
    adr = model.joint(f"{body}_free").qposadr[0]
    data.qpos[adr : adr + 2] = xy


class CupBring(Task):
    name = "cup_bring"
    side = "left"
    instruction = "pick up the cup with the left hand and hold it next to the bottle"
    seconds = 8.0
    # Where the cup starts (bottom centre on the table), sampled uniformly.
    START_X = (0.42, 0.50)
    START_Y = (0.10, 0.16)
    # Palm site relative to the cup axis in the power grasp: the bottle's
    # placement mirrored to the left hand, moved out by the larger radius.
    GRASP_OFFSET = np.array([-0.015, 0.030, 0.045])

    def randomize(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        _set_free_xy(self.model, data, "cup", np.array([rng.uniform(*self.START_X), rng.uniform(*self.START_Y)]))

    def success(self, ep: Episode) -> bool:
        cup = ep.data.body("cup")
        upright = cup.xmat[8] > np.cos(np.deg2rad(15))
        return bool(np.linalg.norm(cup.xpos - CUP_GOAL) < 0.03 and upright)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        _, rot = ik.palm_pose(d)
        hand_names = [j.removeprefix(Names.hand_prefix(self.side) + self.side[0] + "_") for j in self.joints[7:]]
        open_hand = np.zeros(20)
        closed = np.array([POWER_CLOSED.get(n, 0.0) for n in hand_names])
        cup = d.body("cup").xpos.copy()
        grasp = cup + self.GRASP_OFFSET
        q = d.qpos.copy()

        def arm(target: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        yield 1.5, np.concatenate([arm(grasp + [0.0, 0.08, 0.04]), open_hand])
        yield 1.0, np.concatenate([arm(grasp), open_hand])
        yield 1.0, np.concatenate([q[ik.qadr], closed])
        yield 0.8, np.concatenate([arm(grasp + [0.0, 0.0, CUP_GOAL[2] - cup[2]]), closed])
        # The expert is a generator, run while the episode plays: by now the
        # cup is in the hand, so carry it by where it actually sits.
        palm, _ = ik.palm_pose(d)
        yield 1.7, np.concatenate([arm(CUP_GOAL + palm - d.body("cup").xpos), closed])
        yield 1.0, np.concatenate([q[ik.qadr], closed])


class Pour(Task):
    name = "pour"
    side = "right"
    instruction = "pour the bottle into the cup held by the left hand"
    seconds = 16.0
    N_PARTICLES = 30
    # Palm site relative to the bottle axis at mid-body, for the power grasp.
    GRASP_OFFSET = np.array([-0.015, -0.020, 0.09])
    BOTTLE_X = (0.43, 0.47)
    BOTTLE_Y = (-0.14, -0.11)
    CAP_PARKED = (0.30, -0.33)
    TILT_DEG = 120.0

    def __init__(self, n_particles: int = N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.bottle = BottleNames("pet", "pet_cap", ("pet_cap_hold_0", "pet_cap_hold_1"),
                                  [f"pet_p{i}" for i in range(n_particles)])
        self.cup_task = CupBring(model=self.model)

    def randomize(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        m = self.model
        self.cup_task.randomize(data, rng)
        # The cap is off: its thread released and the cap parked on the table.
        for eq in self.bottle.eq:
            data.eq_active[m.equality(eq).id] = 0
        adr = m.joint("pet_cap_free").qposadr[0]
        data.qpos[adr : adr + 3] = [*self.CAP_PARKED, 0.72]
        shift = np.array([rng.uniform(*self.BOTTLE_X), rng.uniform(*self.BOTTLE_Y)]) - data.body("pet").xpos[:2]
        # Move the bottle and its contents together (each has one free joint).
        for body in ["pet", *self.bottle.particles]:
            adr = m.jnt_qposadr[m.body(body).jntadr[0]]
            data.qpos[adr : adr + 2] += shift

    def reset(self, rng: np.random.Generator) -> Episode:
        ep = super().reset(rng)
        left = self.cup_task
        cup_ep = Episode(self.model, ep.data, left.joints, left.actuators, left.qadr)
        run_expert(left, cup_ep)
        return ep

    def poured(self, ep: Episode) -> int:
        return particles_in_cup(ep.model, ep.data, self.bottle, "cup")

    def success(self, ep: Episode) -> bool:
        """At least half the contents in the cup, and the cup not tipped over."""
        upright = ep.data.body("cup").xmat[8] > np.cos(np.deg2rad(20))
        return bool(self.poured(ep) >= len(self.bottle.particles) // 2 and upright)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        home = mujoco.MjData(m)
        set_home(m, home)
        mujoco.mj_kinematics(m, home)
        _, rot0 = ik.palm_pose(home)  # the HOME palm orientation, wherever the hand is now
        hand_names = [j.removeprefix(Names.hand_prefix(self.side) + self.side[0] + "_") for j in self.joints[7:]]
        open_hand = np.zeros(20)
        closed = np.array([POWER_CLOSED.get(n, 0.0) for n in hand_names])
        bottle = d.body("pet").xpos.copy()
        grasp = bottle + self.GRASP_OFFSET
        q = d.qpos.copy()

        self.ik_errors: list[tuple[float, float]] = []

        def arm(target: np.ndarray, rot: np.ndarray = rot0) -> np.ndarray:
            sol, e_p, e_r = ik.solve(q, target, rot)
            self.ik_errors.append((e_p, e_r))
            q[ik.qadr] = sol
            return sol

        # From above: coming in low, the hand swept over the cap stand.
        yield 1.0, np.concatenate([arm(grasp + [0.0, -0.08, 0.15]), open_hand])
        yield 1.0, np.concatenate([arm(grasp + [0.0, -0.08, 0.04]), open_hand])
        yield 1.0, np.concatenate([arm(grasp), open_hand])
        yield 1.0, np.concatenate([q[ik.qadr], closed])
        # High enough that the bottle's bottom clears the held cup's rim.
        yield 1.2, np.concatenate([arm(grasp + [0.0, 0.0, 0.20]), closed])
        # Where the mouth is relative to the palm, measured with the bottle in hand.
        palm, _ = ik.palm_pose(d)
        rb = d.xmat[m.body("pet").id].reshape(3, 3)
        mouth = d.body("pet").xpos + rb @ np.array([0, 0, 0.20])
        mouth_in_palm = rot0.T @ (mouth - palm)
        cup = d.body("cup").xpos
        rim_z = cup[2] + 0.09
        # Upright, the bottle stands beside the cup (axes 85 mm apart: the two
        # radii and 10 mm), mouth 60 mm above the rim. Tilting about the world
        # x axis brings the top toward +y, over the cup; the mouth is moved
        # in over the cup and down toward the rim as the bottle tips.
        beside = np.array([cup[0], cup[1] - 0.085, rim_z + 0.06])
        over = np.array([cup[0], cup[1] - 0.015, rim_z + 0.04])
        yield 1.2, np.concatenate([arm(beside - rot0 @ mouth_in_palm), closed])
        steps = 8
        for k in range(1, steps + 1):
            theta = np.deg2rad(self.TILT_DEG) * k / steps
            rot = R.from_euler("x", -theta).as_matrix() @ rot0
            target = beside + min(1.0, theta / np.deg2rad(90)) * (over - beside)
            yield 0.35, np.concatenate([arm(target - rot @ mouth_in_palm, rot), closed])
        yield 2.0, np.concatenate([q[ik.qadr], closed])
        # Back away still tipped (up and out), then stand the bottle up there:
        # standing it up in place swept it through the cup.
        away = over + [0.0, -0.05, 0.08]
        yield 1.0, np.concatenate([arm(away - rot @ mouth_in_palm, rot), closed])
        yield 1.5, np.concatenate([arm(beside + [0.0, -0.02, 0.08] - rot0 @ mouth_in_palm), closed])


class CapOpen(Task):
    """Right hand alone: hold the bottle's shoulder, twist the cap off between thumb and index, keep it.

    The expert follows how a person opened a bottle one-handed in two
    recorded videos (read by eye; one from the side, one along the bottle's
    axis): middle, ring and little fingers hold the neck and shoulder; the cap
    is held between the thumb's pad and the side of the bent index finger,
    on opposite sides; the twist is the two sliding the opposite ways along
    the cap (the same sense of rotation), then easing off, going back and
    gripping again, four or five times; the free cap stays pinched between
    them and is drawn up off the neck.

    Here the index is laid along the cap's far side by two points (its DIP
    joint and its tip) INDEX_SPAN apart around the cap, and the thumb's tip
    is on the near side; a stroke turns all three points STROKE_DEG about the
    cap's axis while pressing in.
    """

    name = "cap_open"
    side = "right"
    instruction = "open the bottle's cap with the right hand and hold the cap"
    seconds = 24.0  # the expert frees the cap at about 18 s and holds it up by 21 s
    N_PARTICLES = 30
    BOTTLE_X = (0.43, 0.47)
    BOTTLE_Y = (-0.14, -0.11)
    # PINCH_AT 120 deg puts the thumb and the index's middle opposite each
    # other where both reach (a reach map of the IK around the cap: the thumb
    # reaches -90..120 deg, the index laid clockwise from DIP to tip reaches
    # around 300 deg).
    APPROACH_YAW, PINCH_AT = -20.0, 120.0
    PALM_AT, PALM_Z = np.array([-0.020, -0.030]), 0.87
    PAD = 0.008  # thumb tip site to its pad
    INDEX_PAD = 0.008  # index DIP/tip sites to the side of the finger
    INDEX_SPAN = 56.0  # deg around the cap between the index's DIP and tip
    INDEX_ORDER = -1.0  # +1: the tip lies counter-clockwise of the DIP (seen from above)
    SQUEEZE, INDEX_SQUEEZE = 0.012, 0.010
    STROKE_DEG = 40.0
    THUMB_LIFT, INDEX_LIFT = 0.012, 0.010
    PULL_DZ, PULL_THUMB_EXTRA, PULL_INDEX_SQUEEZE = 0.018, -0.008, 0.012
    HOLD_THUMB_SQUEEZE, HOLD_INDEX_SQUEEZE = 0.012, 0.016
    WRAP_EXTRA = 0.0  # rad added to the middle, ring and little fingers' flexion on the shoulder
    MAX_STROKES = 12

    def __init__(self, n_particles: int = N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.bottle = BottleNames("pet", "pet_cap", ("pet_cap_hold_0", "pet_cap_hold_1"),
                                  [f"pet_p{i}" for i in range(n_particles)])

    def randomize(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        m = self.model
        shift = np.array([rng.uniform(*self.BOTTLE_X), rng.uniform(*self.BOTTLE_Y)]) - data.body("pet").xpos[:2]
        for body in ["pet", "pet_cap", *self.bottle.particles]:
            adr = m.jnt_qposadr[m.body(body).jntadr[0]]
            data.qpos[adr : adr + 2] += shift

    def begin(self, data: mujoco.MjData) -> Episode:
        ep = super().begin(data)
        ep.extra["thread"] = CapThread(self.model, self.bottle, CAP_HALF_OPEN_TURNS, CAP_HALF_OPEN_CLOSE_TURNS)
        rb = ep.data.body("pet").xmat.reshape(3, 3)
        ep.extra["seat"] = rb.T @ (ep.data.body("pet_cap").xpos - ep.data.body("pet").xpos)
        ep.extra["bottle_start"] = ep.data.body("pet").xpos.copy()
        return ep

    def cap_lift(self, ep: Episode) -> tuple[float, float]:
        """The cap's distance from the bottle's axis and its height above where it started (bottle frame), m."""
        d = ep.data
        rb = d.body("pet").xmat.reshape(3, 3)
        rel = rb.T @ (d.body("pet_cap").xpos - d.body("pet").xpos) - ep.extra["seat"]
        return float(np.hypot(rel[0], rel[1])), float(rel[2])

    def cap_touching(self, ep: Episode) -> set[str]:
        m, d = ep.model, ep.data
        cap = m.body("pet_cap").id
        prefix = Names.hand_prefix(self.side) + self.side[0] + "_"
        return {m.body(m.geom_bodyid[o]).name.removeprefix(prefix) for c in d.contact[: d.ncon]
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)) if m.geom_bodyid[g] == cap}

    def cap_in_fingers(self, ep: Episode) -> bool:
        return any(n.startswith(("thumb", "index")) for n in self.cap_touching(ep))

    def success(self, ep: Episode) -> bool:
        """The cap is off its thread, held by the thumb or index, and clear of the neck (5 mm up or more)."""
        _, up = self.cap_lift(ep)
        return bool(ep.extra["thread"].released and up > 0.005 and self.cap_in_fingers(ep))

    def pinch_hand(self, ep: Episode, base: np.ndarray, center: np.ndarray, alpha: float,
                   thumb_in: float, index_in: float, frame: np.ndarray | None = None) -> np.ndarray:
        """Hand targets (20) that hold the free cap at `center` between thumb and index, the way the
        strokes do (thumb at PINCH_AT + alpha, the index laid opposite); other joints as in `base`.
        The angles are measured in `frame` (the bottle's orientation; world axes if None)."""
        frame = np.eye(3) if frame is None else frame
        m = ep.model
        prefix = Names.hand_prefix(self.side) + self.side[0] + "_"
        hand = self.joints[7:]
        pinch_j = [j for j in hand if "thumb" in j or "index" in j]

        def around(phi: float, r: float) -> np.ndarray:
            a = np.deg2rad(phi)
            return center + frame @ np.array([r * np.cos(a), r * np.sin(a), 0.0])

        mid = self.PINCH_AT + 180 + alpha
        half = self.INDEX_ORDER * self.INDEX_SPAN / 2
        r_i = CAP_RADIUS + self.INDEX_PAD - index_in
        targets = [(prefix + "thumb_tip", around(self.PINCH_AT + alpha, CAP_RADIUS + self.PAD - thumb_in), 1.0),
                   (Names.hand_prefix(self.side) + "index_dip", around(mid - half, r_i), 0.7),
                   (prefix + "index_finger_tip", around(mid + half, r_i), 1.0)]
        sol, _ = SiteIK(m, pinch_j).solve(ep.data.qpos.copy(), targets, posture=1e-4)
        out = base.copy()
        for j, v in zip(pinch_j, sol):
            out[hand.index(j)] = v
        return out

    def plan(self, ep: Episode, cap_z: float) -> SimpleNamespace:
        """The hand placement and finger poses around the cap: the arm's grasp and pre-grasp
        postures, the light hold of the shoulder (`base`), and pinch(alpha, thumb_in, index_in, dz)
        giving hand targets with the cap's centre at cap_z + dz on the bottle's axis."""
        m, d = ep.model, ep.data
        prefix = Names.hand_prefix(self.side) + self.side[0] + "_"
        hand = self.joints[7:]
        hidx = {j: i for i, j in enumerate(hand)}
        pinch_j = [j for j in hand if "thumb" in j or "index" in j]
        wrap = [j for j in hand if j not in pinch_j]
        axis = d.body("pet").xpos[:2].copy()
        bottom = d.body("pet").xpos[2]
        thumb_tip, index_tip = prefix + "thumb_tip", prefix + "index_finger_tip"
        index_dip = Names.hand_prefix(self.side) + "index_dip"

        def around(phi: float, r: float, z: float) -> np.ndarray:
            a = np.deg2rad(phi)
            return np.array([axis[0] + r * np.cos(a), axis[1] + r * np.sin(a), z])

        def shoulder(z: float) -> float:
            h = z - bottom
            return 0.0325 if h < 0.15 else max(0.0145, 0.0325 - (h - 0.15) / 0.03 * 0.018)

        def pinch_targets(alpha: float, thumb_in: float, index_in: float, dz: float = 0.0) -> list:
            mid = self.PINCH_AT + 180 + alpha
            half = self.INDEX_ORDER * self.INDEX_SPAN / 2
            r_i = CAP_RADIUS + self.INDEX_PAD - index_in
            return [(thumb_tip, around(self.PINCH_AT + alpha, CAP_RADIUS + self.PAD - thumb_in, cap_z + dz), 1.0),
                    (index_dip, around(mid - half, r_i, cap_z + dz), 0.7),
                    (index_tip, around(mid + half, r_i, cap_z + dz), 1.0)]

        # Grasp posture: palm by arm IK, then the fingers, then both together.
        ik = ArmIK(m, self.side)
        home = mujoco.MjData(m)
        set_home(m, home)
        mujoco.mj_kinematics(m, home)
        _, rot = ik.palm_pose(home)  # the HOME palm orientation, whatever the arm does now
        yaw = R.from_euler("z", self.APPROACH_YAW, degrees=True)
        rot = yaw.as_matrix() @ rot
        off = yaw.apply([*self.PALM_AT, 0.0])
        palm = np.array([axis[0] + off[0], axis[1] + off[1], self.PALM_Z])
        q = d.qpos.copy()
        q[ik.qadr] = ik.solve(q, palm, rot)[0]
        wrap_at = [(prefix + f + "_tip", around(phi + self.APPROACH_YAW, shoulder(z) + self.PAD, z), 0.7)
                   for f, phi, z in (("middle_finger", 70, 0.872), ("ring_finger", 60, 0.857), ("pinky", 50, 0.842))]
        contact = pinch_targets(-self.STROKE_DEG / 2, 0.0, 0.0) + wrap_at
        hand_ik = SiteIK(m, hand)
        q[hand_ik.qadr] = hand_ik.solve(q, contact, posture=1e-4)[0]
        full = SiteIK(m, self.joints[:7] + hand)
        q[full.qadr], dist = full.solve(q, contact, posture=3e-3)
        self.grasp_error = dist
        grasp_arm = q[ik.qadr].copy()
        q_pre = ik.solve(d.qpos, palm + yaw.apply([0, -0.06, 0]) + [0, 0, 0.03], rot)[0]

        def put(base: np.ndarray, joints: list[str], sol: np.ndarray) -> np.ndarray:
            out = base.copy()
            for j, v in zip(joints, sol):
                out[hidx[j]] = v
            return out

        base = q[[m.joint(j).qposadr[0] for j in hand]].copy()
        base = put(base, wrap, SiteIK(m, wrap).solve(
            q, [(n, p - (p - np.array([*axis, p[2]])) / np.linalg.norm(p[:2] - axis) * self.SQUEEZE, w) for n, p, w in wrap_at],
            posture=1e-4)[0])
        # Hold the shoulder firmly from the start, so the bottle can be lifted later without
        # gripping harder then (tightening while pinching the cap shifted the bottle and lost it).
        for j in wrap:
            if not j.endswith("_abd"):
                base[hidx[j]] += self.WRAP_EXTRA
        pinch_ik = SiteIK(m, pinch_j)
        self.pinch_errors = []

        def pinch(alpha: float, thumb_in: float, index_in: float, dz: float = 0.0) -> np.ndarray:
            sol, err = pinch_ik.solve(q, pinch_targets(alpha, thumb_in, index_in, dz), posture=1e-4)
            self.pinch_errors.append(max(err.values()))
            return put(base, pinch_j, sol)

        alphas = np.linspace(-self.STROKE_DEG / 2, self.STROKE_DEG / 2, 7)
        pressed = [pinch(a, self.SQUEEZE, self.INDEX_SQUEEZE) for a in alphas]
        eased_end = pinch(alphas[-1], -self.THUMB_LIFT, -self.INDEX_LIFT)
        eased_start = pinch(alphas[0], -self.THUMB_LIFT, -self.INDEX_LIFT)

        return SimpleNamespace(grasp_arm=grasp_arm, q_pre=q_pre, palm=palm, rot=rot, base=base, wrap=wrap,
                               hidx=hidx, pinch=pinch, alphas=alphas, pressed=pressed,
                               eased_start=eased_start, eased_end=eased_end)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        d = ep.data
        thread = ep.extra["thread"]
        pl = self.plan(ep, d.body("pet_cap").xpos[2])
        grasp_arm, q_pre, pinch, alphas = pl.grasp_arm, pl.q_pre, pl.pinch, pl.alphas
        pressed, eased_start, eased_end = pl.pressed, pl.eased_start, pl.eased_end

        def arm_hand(h: np.ndarray, arm: np.ndarray = grasp_arm) -> np.ndarray:
            return np.concatenate([arm, h])

        yield 1.5, arm_hand(np.zeros(20), q_pre)
        yield 1.0, arm_hand(np.zeros(20))
        yield 0.8, arm_hand(eased_start)
        k_rel = len(alphas) - 1
        for _ in range(self.MAX_STROKES):
            yield 0.3, arm_hand(pressed[0])
            for k, target in enumerate(pressed[1:], 1):
                yield 0.15, arm_hand(target)
                if thread.released:
                    k_rel = k
                    break
            if thread.released:
                break
            yield 0.2, arm_hand(eased_end)
            yield 0.3, arm_hand(eased_start)
        if not thread.released:
            return
        # Free: it drops onto the neck's lip (CAP_HALF_OPEN_RAISE); follow it
        # down between thumb and index, then draw it up, the index pressing
        # harder than during the strokes (it let go first when it did not).
        # The index has the least reach at a stroke's end: ease off, come back
        # to the stroke's start (the loose cap's angle does not matter) and
        # grip there.
        a = alphas[0]
        drop = -CAP_HALF_OPEN_RAISE
        thumb_in, index_in = self.SQUEEZE + self.PULL_THUMB_EXTRA, self.PULL_INDEX_SQUEEZE
        yield 0.2, arm_hand(pinch(alphas[k_rel], -self.THUMB_LIFT, -self.INDEX_LIFT, drop))
        yield 0.3, arm_hand(pinch(a, -self.THUMB_LIFT, -self.INDEX_LIFT, drop))
        yield 0.4, arm_hand(pinch(a, thumb_in, index_in, drop))
        for k in range(1, 5):
            yield 0.4, arm_hand(pinch(a, thumb_in, index_in, drop + self.PULL_DZ * k / 4))
        yield 0.8, arm_hand(pinch(a, thumb_in, index_in, drop + self.PULL_DZ))
        # Off the neck: pinch harder to keep it (the light pinch that draws it up let it sink back
        # onto the neck within a few seconds).
        yield 0.5, arm_hand(pinch(a, self.HOLD_THUMB_SQUEEZE, self.HOLD_INDEX_SQUEEZE, drop + self.PULL_DZ))


def prepare(cup: "CupBring", cap: "CapOpen", rng: np.random.Generator) -> mujoco.MjData:
    """The first stage of the bimanual sequence by the experts: the left hand brings the cup while the
    right opens the cap and keeps it. cup and cap must share one model. Returns the state they leave."""
    m = cap.model
    d = mujoco.MjData(m)
    set_home(m, d)
    cup.randomize(d, rng)
    cap.randomize(d, rng)
    mujoco.mj_forward(m, d)
    ep_cap = cap.begin(d)
    run_experts([(cup, cup.begin(d)), (cap, ep_cap)])
    cap.last_episode = ep_cap
    return d


class CapPutDown(Task):
    """Right hand: with the cap just off and pinched (as cap_open leaves it), let go of the bottle,
    set the cap down upright on the table to the bottle's right and draw the hand back.

    The fallback for pouring while keeping the cap: the cap waits on the table (the hand then
    takes the bottle with a full grasp to pour, and picks the cap up again to close it).
    """

    name = "cap_put_down"
    side = "right"
    instruction = "put the cap down on its stand"
    seconds = 9.0
    # On the cap stand (props.add_cap_stand), clear of the bottle, the hand's way to it and the
    # cup's. The cap is let go RELEASE_HEIGHT above where it comes to rest.
    SPOT = np.array(CAP_STAND_POS[:2])
    RELEASE_HEIGHT = 0.008

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.cap_task = CapOpen(n_particles, model=self.model)
        self.cup_task = CupBring(model=self.model)

    def reset(self, rng: np.random.Generator) -> Episode:
        d = prepare(self.cup_task, self.cap_task, rng)
        ep = self.begin(d)
        ep.extra.update(self.cap_task.last_episode.extra)
        return ep

    def success(self, ep: Episode) -> bool:
        """Upright on the stand, and no longer touched by the hand."""
        cap = ep.data.body("pet_cap")
        upright = cap.xmat[8] > np.cos(np.deg2rad(20))
        on_stand = abs(cap.xpos[2] - (STAND_TOP + CAP_HALF_HEIGHT)) < 0.006
        near = np.linalg.norm(cap.xpos[:2] - self.SPOT) < 0.025
        return bool(upright and on_stand and near and not self.cap_task.cap_touching(ep) - {"pet", "cap_stand"})

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        c = self.cap_task
        ik = ArmIK(m, self.side)
        palm, rot = ik.palm_pose(d)
        hand = d.ctrl[ep.actuators][7:].copy()
        arm_now = d.ctrl[ep.actuators][:7].copy()
        wrap = [i for i, j in enumerate(self.joints[7:]) if not ("thumb" in j or "index" in j)]
        q = d.qpos.copy()

        def arm(target: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        free = hand.copy()
        free[wrap] = 0.0
        yield 0.8, np.concatenate([arm_now, free])
        cap_in_palm = d.body("pet_cap").xpos - palm  # world offset; the palm keeps its orientation
        # Back off the bottle along the palm's normal before rising: lifting straight up, the
        # opened fingertips caught the shoulder and tipped the bottle over.
        back = palm - 0.03 * rot[:, 1]
        yield 0.6, np.concatenate([arm(back), free])
        up = back + [0, 0, 0.10]
        yield 1.0, np.concatenate([arm(up), free])
        # Across high and in short steps (a single joint-space move dipped and swept the bottle).
        down = np.array([*self.SPOT, STAND_TOP + CAP_HALF_HEIGHT + self.RELEASE_HEIGHT]) - cap_in_palm
        high = np.array([down[0], down[1], up[2]])
        for k in range(1, 5):
            yield 0.35, np.concatenate([arm(up + k / 4 * (high - up)), free])
        for k in range(1, 4):
            yield 0.4, np.concatenate([arm(high + k / 3 * (down - high)), free])
        # Let go by easing thumb and index back toward straight (re-solving a wider pinch swung
        # the thumb's base into the cap and flipped it).
        let_go = free.copy()
        pinch = [i for i, j in enumerate(self.joints[7:]) if "thumb" in j or "index" in j]
        let_go[pinch] *= 0.6
        yield 0.5, np.concatenate([q[ik.qadr], let_go])
        # Straight up off the stand, then back toward the body, high: the next move (to the
        # bottle) otherwise swept low over the stand and knocked the cap off.
        yield 0.8, np.concatenate([arm(down + [0, 0, 0.10]), np.zeros(20)])
        yield 0.8, np.concatenate([arm(down + [-0.06, 0.0, 0.18]), np.zeros(20)])


class PourHeld(Task):
    """Right hand: the bottle held at its shoulder and the cap between thumb and index (as cap_open
    leaves them), pour into the cup the left hand holds (as cup_bring leaves it), stand the bottle up.

    The hand keeps its grip throughout; only the arm moves: lift clear of the cup, bring the bottle
    beside it, tip it over the cup about the world x axis, back away still tipped, stand it up.
    """

    name = "pour_held"
    side = "right"
    instruction = "keeping the cap, pour the bottle into the cup the left hand holds"
    seconds = 14.0
    LIFT = 0.20
    TILT_DEG = 120.0
    CAP_ASIDE = np.array([0.02, 0.0, 0.02])  # where the held cap goes (bottle frame), from where cap_open left it
    GRIP_EXTRA = 0.9  # rad added to the middle, ring and little fingers' flexion before lifting

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None,
                 policy: str | Path | None = None) -> None:
        super().__init__(n_particles, model)
        self.cap_task = CapOpen(n_particles, model=self.model)
        self.cup_task = CupBring(model=self.model)
        self.bottle = self.cap_task.bottle
        # With a trained rl_hold policy (rl_hold.HoldLiftEnv), the policy re-grips and lifts the
        # bottle while keeping the cap pinched, in place of steps 1 and 2 of the script.
        self.policy_path = policy
        self._policy = None

    def policy_lift(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        """The rl_hold policy in closed loop, one control tick per segment, with the arm on the
        schedule it was trained with (hold, then lift straight up). Leaves the palm lifted by
        HoldLiftEnv.LIFT."""
        from stable_baselines3 import PPO

        from .rl_hold import HoldLiftEnv, HoldView

        if self._policy is None:
            self._policy = PPO.load(str(self.policy_path), device="cpu")
        env = HoldLiftEnv
        n_ticks = round(env.EPISODE_S / env.CONTROL_DT)
        view = HoldView(ep.model, self.cap_task, n_ticks)
        d = ep.data
        ik = ArmIK(ep.model, self.side)
        z0 = float(d.xpos[view.bottle_id][2])
        arm0 = d.ctrl[ep.actuators][:7].copy()
        palm, rot = view.palm(d)
        arm_lift, _, _ = ik.solve(d.qpos.copy(), palm + [0, 0, env.LIFT], rot)
        for tick in range(n_ticks):
            action, _ = self._policy.predict(view.obs(d, z0, tick), deterministic=True)
            hand = view.hand_targets(d, action)
            s = np.clip((tick * env.CONTROL_DT - env.HOLD_S) / env.LIFT_S, 0.0, 1.0)
            yield env.CONTROL_DT, np.concatenate([arm0 + s * (arm_lift - arm0), hand])

    def reset(self, rng: np.random.Generator) -> Episode:
        d = prepare(self.cup_task, self.cap_task, rng)
        ep = self.begin(d)
        ep.extra.update(self.cap_task.last_episode.extra)
        return ep

    def poured(self, ep: Episode) -> int:
        return particles_in_cup(ep.model, ep.data, self.bottle, "cup")

    def success(self, ep: Episode) -> bool:
        """At least half the contents in the cup, the cup upright, the cap still in the fingers."""
        upright = ep.data.body("cup").xmat[8] > np.cos(np.deg2rad(20))
        held = self.cap_task.cap_in_fingers(ep)
        return bool(self.poured(ep) >= len(self.bottle.particles) // 2 and upright and held)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        palm, rot0 = ik.palm_pose(d)
        c = self.cap_task
        arm_now = d.ctrl[ep.actuators][:7].copy()
        rb = d.body("pet").xmat.reshape(3, 3)
        if self.policy_path is not None:
            from .rl_hold import HoldLiftEnv

            yield from self.policy_lift(ep)
            yield from self._pour(ep, ik, d.ctrl[ep.actuators][7:].copy(), self.LIFT - HoldLiftEnv.LIFT)
            return
        # 1. Take the free cap well clear of the neck first (up, and off the mouth along +x: the
        #    tilt is about the x axis, so there it stays beside the stream, not over it).
        grip = c.pinch_hand(ep, d.ctrl[ep.actuators][7:].copy(), d.body("pet_cap").xpos + rb @ self.CAP_ASIDE,
                            -c.STROKE_DEG / 2, c.HOLD_THUMB_SQUEEZE, c.HOLD_INDEX_SQUEEZE, rb)
        yield 0.8, np.concatenate([arm_now, grip])
        # 2. Then hold the bottle firmly with the palm and the middle, ring and little fingers: the
        #    cap twist needs only a light hold of the shoulder, and gripping harder shifts the
        #    bottle in the hand, which knocked the cap off while it was still just above the neck.
        for i, name in enumerate(self.joints[7:]):
            if any(f in name for f in ("middle", "ring", "pinky")) and not name.endswith("_abd"):
                grip[i] += self.GRIP_EXTRA
        yield 0.8, np.concatenate([arm_now, grip])
        yield from self._pour(ep, ik, grip, self.LIFT)

    def _pour(self, ep: Episode, ik: ArmIK, grip: np.ndarray, lift: float) -> Iterator[tuple[float, np.ndarray]]:
        """The arm's part with the hand's grip fixed: lift the palm by `lift`, bring the mouth
        beside the cup, tip, pour, back away, stand the bottle up."""
        m, d = ep.model, ep.data
        q = d.qpos.copy()
        palm, rot0 = ik.palm_pose(d)

        def arm(target: np.ndarray, rot: np.ndarray = rot0) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return np.concatenate([sol, grip])

        rb = d.xmat[m.body("pet").id].reshape(3, 3)
        mouth = d.body("pet").xpos + rb @ np.array([0, 0, 0.20])
        mouth_in_palm = rot0.T @ (mouth - palm)
        lifted = palm + [0.0, 0.0, lift]
        yield 1.2, arm(lifted)
        cup = d.body("cup").xpos
        rim_z = cup[2] + 0.09
        beside = np.array([cup[0], cup[1] - 0.085, rim_z + 0.06])
        over = np.array([cup[0], cup[1] - 0.015, rim_z + 0.04])
        # In short steps: one long joint-space move swung the bottle through a wide arc.
        goal = beside - rot0 @ mouth_in_palm
        for k in range(1, 5):
            yield 0.3, arm(lifted + k / 4 * (goal - lifted))
        steps = 8
        rot = rot0
        for k in range(1, steps + 1):
            theta = np.deg2rad(self.TILT_DEG) * k / steps
            rot = R.from_euler("x", -theta).as_matrix() @ rot0
            target = beside + min(1.0, theta / np.deg2rad(90)) * (over - beside)
            yield 0.35, arm(target - rot @ mouth_in_palm, rot)
        yield 2.0, arm(over - rot @ mouth_in_palm, rot)
        away = over + [0.0, -0.05, 0.08]
        yield 1.0, arm(away - rot @ mouth_in_palm, rot)
        yield 1.5, arm(beside + [0.0, -0.02, 0.08] - rot0 @ mouth_in_palm)


def after_pour(pour: "PourHeld", rng: np.random.Generator) -> tuple[mujoco.MjData, dict]:
    """The sequence up to the end of the held pour, by the experts. Returns the state and the cap's
    episode extras (its thread and where it started)."""
    ep = pour.reset(rng)
    run_expert(pour, ep)
    return ep.data, ep.extra


class PlaceBottle(Task):
    """Right hand: after pouring, stand the bottle back where it started, still holding it and the cap."""

    name = "place_bottle"
    side = "right"
    instruction = "put the bottle back on the table, keeping the cap"
    seconds = 5.0
    KEEP_CAP = True  # False when the cap waits on its stand (the full-grasp pour)

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.pour = PourHeld(n_particles, model=self.model)
        self.cap_task = self.pour.cap_task

    def reset(self, rng: np.random.Generator) -> Episode:
        d, extra = after_pour(self.pour, rng)
        ep = self.begin(d)
        ep.extra.update(extra)
        return ep

    def success(self, ep: Episode) -> bool:
        """Upright, standing on the table near where it started, and the cap still in the fingers."""
        b = ep.data.body("pet")
        upright = b.xmat[8] > np.cos(np.deg2rad(10))
        on_table = abs(b.xpos[2] - TABLE_Z) < 0.01
        near = np.linalg.norm(b.xpos[:2] - ep.extra["bottle_start"][:2]) < 0.03
        return bool(upright and on_table and near and (self.cap_task.cap_in_fingers(ep) or not self.KEEP_CAP))

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        palm, rot = ik.palm_pose(d)
        hold = d.ctrl[ep.actuators][7:].copy()
        q = d.qpos.copy()
        # Where the palm must be for the bottle's bottom to sit on the table at its start point.
        rb = d.body("pet").xmat.reshape(3, 3)
        bottom_in_palm = rot.T @ (d.body("pet").xpos - palm)
        # The firmer grip leaves the bottle tipped in the hand (about 13 deg): turn the hand so the
        # bottle's axis is vertical again before it touches down.
        align, _ = R.align_vectors([[0.0, 0.0, 1.0]], [rb[:, 2]])
        rot = align.as_matrix() @ rot
        goal = np.array([*ep.extra["bottle_start"][:2], TABLE_Z + 0.002])
        above = goal + [0, 0, 0.06]

        def arm(bottom: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, bottom - rot @ bottom_in_palm, rot)
            q[ik.qadr] = sol
            return np.concatenate([sol, hold])

        start = d.body("pet").xpos.copy()
        for k in range(1, 4):
            yield 0.4, arm(start + k / 3 * (above - start))
        yield 1.0, arm(goal)
        yield 0.5, arm(goal)


class PlaceCup(Task):
    """Left hand: after the pour, set the cup down on the table, let go and draw the hand back."""

    name = "place_cup"
    side = "left"
    instruction = "put the cup down on the table and let go"
    seconds = 5.0
    PLACE = np.array([0.44, 0.06])  # where the cup goes down, beside where it was held

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.pour = PourHeld(n_particles, model=self.model)

    def reset(self, rng: np.random.Generator) -> Episode:
        d, extra = after_pour(self.pour, rng)
        ep = self.begin(d)
        ep.extra.update(extra)
        return ep

    def success(self, ep: Episode) -> bool:
        """Standing upright on the table and no longer held."""
        m, d = ep.model, ep.data
        cup = d.body("cup")
        upright = cup.xmat[8] > np.cos(np.deg2rad(10))
        on_table = abs(cup.xpos[2] - TABLE_Z) < 0.01
        prefix = Names.hand_prefix(self.side)
        touching = any(m.body(m.geom_bodyid[o]).name.startswith(prefix) for c in d.contact[: d.ncon]
                       for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)) if m.geom_bodyid[g] == m.body("cup").id)
        return bool(upright and on_table and not touching)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        palm, rot = ik.palm_pose(d)
        hold = d.ctrl[ep.actuators][7:].copy()
        q = d.qpos.copy()
        cup_in_palm = rot.T @ (d.body("cup").xpos - palm)
        goal = np.array([*self.PLACE, TABLE_Z + 0.002])

        def arm(cup: np.ndarray, hand: np.ndarray = hold) -> np.ndarray:
            sol, _, _ = ik.solve(q, cup - rot @ cup_in_palm, rot)
            q[ik.qadr] = sol
            return np.concatenate([sol, hand])

        start = d.body("cup").xpos.copy()
        yield 1.2, arm(goal + [0, 0, 0.02])
        yield 0.8, arm(goal)
        yield 0.8, arm(goal, np.zeros(20))
        # Out sideways (away from the cup), then up.
        yield 0.8, arm(goal + [0, 0.07, 0.0], np.zeros(20))
        yield 0.8, arm(goal + [0, 0.07, 0.10], np.zeros(20))


def after_place(place_bottle: "PlaceBottle", place_cup: "PlaceCup", rng: np.random.Generator) -> tuple[mujoco.MjData, dict]:
    """The sequence up to both put down, by the experts (the two share one model and one pour)."""
    d, extra = after_pour(place_bottle.pour, rng)
    eb, ec = place_bottle.begin(d), place_cup.begin(d)
    eb.extra.update(extra)
    ec.extra.update(extra)
    run_experts([(place_bottle, eb), (place_cup, ec)])
    return d, extra


class Recap(Task):
    """Right hand: the bottle standing on the table and the cap held between thumb and index (as
    place_bottle leaves them), put the cap back on and screw it closed, then let go.

    As a person does it: let go of the bottle (it stands), take the hand to the hold used for
    opening (the cap still pinched), set the cap on the neck, hold the shoulder lightly, and turn
    the cap the closing way (clockwise from above) with the same thumb-and-index strokes, run
    backwards, until props.CapThread reports it closed.
    """

    name = "recap"
    side = "right"
    instruction = "put the cap back on the bottle and close it"
    seconds = 20.0
    ABOVE = 0.012  # cap held this far above where it rests on the lip, before setting it down
    MAX_STROKES = 10

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.place_bottle = PlaceBottle(n_particles, model=self.model)
        self.place_cup = PlaceCup(n_particles, model=self.model)
        self.place_cup.pour = self.place_bottle.pour
        self.cap_task = self.place_bottle.cap_task

    def reset(self, rng: np.random.Generator) -> Episode:
        d, extra = after_place(self.place_bottle, self.place_cup, rng)
        ep = self.begin(d)
        ep.extra.update(extra)
        return ep

    def success(self, ep: Episode) -> bool:
        return bool(ep.extra["thread"].closed)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        c = self.cap_task
        thread = ep.extra["thread"]
        rest_z = d.body("pet").xpos[2] + thread.rest_z
        pl = c.plan(ep, rest_z)
        hold = d.ctrl[ep.actuators][7:].copy()
        arm_now = d.ctrl[ep.actuators][:7].copy()
        ik = ArmIK(m, self.side)
        palm, rot_now = ik.palm_pose(d)
        q = d.qpos.copy()

        def arm(target: np.ndarray, rot: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        # Let go of the bottle and the cap: after the pour the cap rests on top of the bottle,
        # near the opening (the fingers only touch it), and the bottle stands on its own.
        yield 0.6, np.concatenate([arm_now, pl.eased_end * 0.0 + np.where(np.isin(np.arange(20), [pl.hidx[j] for j in pl.wrap]), 0.0, pl.eased_end)])
        up = arm(palm + [0, 0, 0.05], rot_now)
        yield 0.8, np.concatenate([up, np.zeros(20)])
        # Into the opening hold from above, thumb and index open around the cap.
        above = arm(pl.palm + [0, 0, 0.05], pl.rot)
        open_around = pl.eased_end.copy()
        yield 1.5, np.concatenate([above, open_around])
        yield 1.2, np.concatenate([pl.grasp_arm, open_around])
        # The cap lies flat on the lip, off centre (its skirt not over the neck), higher than where
        # it sits when on. Squeeze it from both sides about the neck's axis at the height it lies
        # at, so the fingers slide it to the centre, then press it down onto the neck, where the
        # thread takes it again.
        rb = d.body("pet").xmat.reshape(3, 3)
        lying_dz = float((rb.T @ (d.body("pet_cap").xpos - d.body("pet").xpos))[2] - thread.rest_z)
        a0 = pl.alphas[-1]
        yield 0.8, np.concatenate([pl.grasp_arm, pl.pinch(a0, c.HOLD_THUMB_SQUEEZE, c.HOLD_INDEX_SQUEEZE, lying_dz)])
        for k in range(1, 4):
            dz = lying_dz * (1 - k / 3)
            yield 0.4, np.concatenate([pl.grasp_arm, pl.pinch(a0, c.HOLD_THUMB_SQUEEZE, c.HOLD_INDEX_SQUEEZE, dz)])
        yield 0.5, np.concatenate([pl.grasp_arm, pl.pressed[-1]])
        # Close: the opening strokes run backwards (clockwise).
        pressed = pl.pressed[::-1]
        for _ in range(self.MAX_STROKES):
            yield 0.3, np.concatenate([pl.grasp_arm, pressed[0]])
            for target in pressed[1:]:
                yield 0.15, np.concatenate([pl.grasp_arm, target])
                if thread.closed:
                    break
            if thread.closed:
                break
            yield 0.2, np.concatenate([pl.grasp_arm, pl.eased_start])
            yield 0.3, np.concatenate([pl.grasp_arm, pl.eased_end])
        # Let go and draw the hand back.
        yield 0.4, np.concatenate([pl.grasp_arm, pl.eased_start])
        yield 0.6, np.concatenate([pl.grasp_arm, np.zeros(20)])
        yield 1.0, np.concatenate([pl.q_pre, np.zeros(20)])


def after_place_from_stand(put: "CapPutDown", pour: "Pour", place_bottle: "PlaceBottle", place_cup: "PlaceCup",
                           rng: np.random.Generator) -> tuple[mujoco.MjData, dict]:
    """The fallback sequence by the experts up to both put down: cup and cap at once, cap onto its
    stand, full-grasp pour, bottle and cup down at once. All four share one model."""
    ep = put.reset(rng)
    run_expert(put, ep)
    run_expert(pour, pour.begin(ep.data))
    eb, ec = place_bottle.begin(ep.data), place_cup.begin(ep.data)
    eb.extra.update(ep.extra)
    ec.extra.update(ep.extra)
    run_experts([(place_bottle, eb), (place_cup, ec)])
    return ep.data, ep.extra


class RecapFromStand(Task):
    """Right hand: the bottle back on the table (just let go of after the pour) and the cap waiting
    on its stand: pick the cap up, set it on the neck and screw it closed, then let go.

    The cap is picked up with the hold that opened it, shifted from the neck to the stand (the
    stand stands at shoulder height), carried high to above the neck, set on the lip, and turned
    the closing way with the opening strokes run backwards until props.CapThread reports it closed.
    """

    name = "recap"
    side = "right"
    instruction = "put the cap back on the bottle and close it"
    seconds = 50.0
    ABOVE = 0.012
    TWIST_DEG = 30.0  # the hand's turn about the cap's axis per closing stroke
    MOVE_IN_HEIGHT = 0.025  # over the lip while moving in (the supporting finger hangs below the cap)
    ON_LIP = 0.008  # cap centre above its seated height when the skirt first sits on the lip
    CENTRE_TOL = 0.0015
    CENTRE_GAIN = 0.6
    VERBOSE = False
    MAX_STROKES = 12
    CARRY_THUMB_SQUEEZE, CARRY_INDEX_SQUEEZE = 0.012, 0.016  # pinch to take the cap off the stand
    # The middle fingertip's support under the cap's rim: azimuth (deg), distance from the cap's
    # axis, and gap below the rim (the fingertip site is inside the pad).
    SUPPORT_AT, SUPPORT_R, SUPPORT_GAP = 45.0, 0.016, 0.006

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None) -> None:
        super().__init__(n_particles, model)
        self.put = CapPutDown(n_particles, model=self.model)
        self.cap_task = self.put.cap_task
        self.pour = Pour(n_particles, model=self.model)
        self.place_bottle = PlaceBottle(n_particles, model=self.model)
        self.place_bottle.KEEP_CAP = False
        self.place_cup = PlaceCup(n_particles, model=self.model)

    def reset(self, rng: np.random.Generator) -> Episode:
        d, extra = after_place_from_stand(self.put, self.pour, self.place_bottle, self.place_cup, rng)
        ep = self.begin(d)
        ep.extra.update(extra)
        return ep

    def success(self, ep: Episode) -> bool:
        return bool(ep.extra["thread"].closed)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        c = self.cap_task
        thread = ep.extra["thread"]
        ik = ArmIK(m, self.side)
        palm, rot_now = ik.palm_pose(d)
        q = d.qpos.copy()

        def arm(target: np.ndarray, rot: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        # Let go of the bottle gently: loosen halfway and ease back, then open and back off, then
        # rise (opening the full grasp at once flicked the bottle over).
        arm_now = d.ctrl[ep.actuators][:7].copy()
        grasp = d.ctrl[ep.actuators][7:].copy()
        yield 0.6, np.concatenate([arm_now, 0.5 * grasp])
        yield 0.5, np.concatenate([arm(palm - 0.01 * rot_now[:, 1], rot_now), 0.5 * grasp])
        yield 0.6, np.concatenate([q[ik.qadr], np.zeros(20)])
        back = palm - 0.05 * rot_now[:, 1]
        yield 0.6, np.concatenate([arm(back, rot_now), np.zeros(20)])
        yield 0.6, np.concatenate([arm(back + [0, 0, 0.08], rot_now), np.zeros(20)])
        def seat_now() -> np.ndarray:
            """Where the cap's centre sits when on, from the bottle's pose now (the hand may nudge it)."""
            rb = d.body("pet").xmat.reshape(3, 3)
            return d.body("pet").xpos + rb @ np.array([0, 0, thread.rest_z])

        seat = seat_now()
        pl = c.plan(ep, float(seat[2]))
        a0 = pl.alphas[-1]
        hold = pl.pinch(a0, self.CARRY_THUMB_SQUEEZE, self.CARRY_INDEX_SQUEEZE, 0.0)
        opened = pl.pinch(a0, -c.THUMB_LIFT, -c.INDEX_LIFT, 0.0)
        for h in (hold, opened):
            for j in pl.wrap:
                h[pl.hidx[j]] = 0.0
        # The opening hold moved from the neck to the stand.
        shift = d.body("pet_cap").xpos - seat
        pick = pl.palm + shift
        # In from the side the palm faces from, as for opening (from above, the open fingers came
        # down on the cap and pushed it off the stand).
        side = pl.palm - seat
        side[2] = 0.0
        side /= np.linalg.norm(side)
        here, _ = ik.palm_pose(d)
        pre = pick + 0.06 * side
        high = np.array([pre[0], pre[1], max(here[2], pre[2]) + 0.10])
        for k in range(1, 4):
            yield 0.4, np.concatenate([arm(here + k / 3 * (high - here), pl.rot), opened])
        yield 1.0, np.concatenate([arm(pre, pl.rot), opened])
        yield 0.8, np.concatenate([arm(pick, pl.rot), opened])
        yield 0.5, np.concatenate([q[ik.qadr], hold])
        # Up above the bottle's top first, then across, then down over the neck (carried on a
        # slant, the cap met the neck on the way and was knocked out of the fingers).
        over = pl.palm + [0, 0, 0.08]
        lifted = np.array([pick[0], pick[1], over[2] + 0.04])
        yield 0.5, np.concatenate([arm(pick + [0, 0, 0.02], pl.rot), hold])
        # Off the stand: curl the middle finger under the cap's rim to carry it (pinched from the
        # sides alone it slipped out on the way, and squeezing harder pops it out).
        pinch_only = hold
        mid = [j for j in self.joints[7:] if "middle" in j]
        a_sup = np.deg2rad(self.SUPPORT_AT)
        under = d.body("pet_cap").xpos + np.array([self.SUPPORT_R * np.cos(a_sup), self.SUPPORT_R * np.sin(a_sup),
                                                    -(CAP_HALF_HEIGHT + self.SUPPORT_GAP)])
        sol, _ = SiteIK(m, mid).solve(d.qpos.copy(), [(Names.hand_prefix(self.side) + self.side[0] + "_middle_finger_tip", under, 1.0)],
                                      posture=1e-4)
        hold = pinch_only.copy()
        for j, v in zip(mid, sol):
            hold[pl.hidx[j]] = v
        yield 0.6, np.concatenate([q[ik.qadr], hold])
        here2, _ = ik.palm_pose(d)
        for k in range(1, 3):
            yield 0.5, np.concatenate([arm(here2 + k / 2 * (lifted - here2), pl.rot), hold])
        above = np.array([over[0], over[1], lifted[2]])
        for k in range(1, 6):
            yield 0.5, np.concatenate([arm(lifted + k / 5 * (above - lifted), pl.rot), hold])
        # Down beside the neck (the straight fingers clear of the bottle), in from the side the
        # palm faces from as for opening, then onto the lip, steering the hand by where the cap
        # really is at each step (it drifts a few millimetres in the fingers as the arm moves;
        # 12 mm off, the skirt caught the lip and the cap was lost). Coming straight down, the
        # opened fingers met the bottle's top and shook the cap loose.
        # Straight down over the neck, the middle finger under the cap (pinched from the sides
        # alone, it slipped out on the way down), steering by where the cap really is.
        target = above.copy()
        for dz in (0.08, 0.05, 0.03):
            err = seat_now() - d.body("pet_cap").xpos
            target = target + np.array([err[0], err[1], 0.0])
            target[2] = pl.palm[2] + dz
            yield 0.6, np.concatenate([arm(target, pl.rot), hold])
        # Centre it over the neck first, measuring and correcting until it is within CENTRE_TOL
        # (the skirt clears the neck by 1.5 mm: off by more, it lands on the lip and slides off).
        target[2] = pl.palm[2] + 0.015
        for _ in range(6):
            err = seat_now() - d.body("pet_cap").xpos
            if self.VERBOSE:
                palm_now, _ = ik.palm_pose(d)
                hand_on_bottle = any(m.body(m.geom_bodyid[o]).name.startswith("wuji_r_") for cc in d.contact[: d.ncon]
                                     for g, o in ((cc.geom1, cc.geom2), (cc.geom2, cc.geom1)) if m.geom_bodyid[g] == m.body("pet").id)
                rp = ik.palm_pose(d)[1]
                cap_in_palm = rp.T @ (d.body("pet_cap").xpos - palm_now)
                print(f"   centring: off {1000 * np.hypot(err[0], err[1]):.1f} mm; palm off its target "
                      f"{1000 * np.linalg.norm(palm_now - target):.1f} mm; hand on bottle {hand_on_bottle}; "
                      f"cap in palm mm {np.round(1000 * cap_in_palm).astype(int).tolist()}; cap touches {sorted(c.cap_touching(ep))}")
            if np.hypot(err[0], err[1]) < self.CENTRE_TOL:
                break
            target = target + self.CENTRE_GAIN * np.array([err[0], err[1], 0.0])
            yield 0.6, np.concatenate([arm(target, pl.rot), hold])
        err = seat_now() - d.body("pet_cap").xpos
        target = target + np.array([err[0], err[1], 0.0])
        target[2] = pl.palm[2] + self.ON_LIP
        yield 0.6, np.concatenate([arm(target, pl.rot), hold])
        yield 0.4, np.concatenate([q[ik.qadr], pinch_only])  # middle finger out from under it
        hold = pinch_only
        for dz in (self.ON_LIP / 2, 0.0):
            err = seat_now() - d.body("pet_cap").xpos
            target = target + np.array([err[0], err[1], 0.0])
            target[2] = pl.palm[2] + dz
            yield 0.6, np.concatenate([arm(target, pl.rot), hold])
        yield from wrist_close(self, ep, ik, q, arm, seat_now)


def wrist_close(task: Task, ep: Episode, ik: ArmIK, q: np.ndarray, arm: Callable[[np.ndarray, np.ndarray], np.ndarray],
                seat_now: Callable[[], np.ndarray], wait: float = 0.0) -> Iterator[tuple[float, np.ndarray]]:
    """Right hand, the cap set on the neck (the thread engaged): screw it closed and let go. `arm`
    solves the arm for a palm pose and records it in `q`; `seat_now` is where the cap's centre sits
    when on, from the bottle's pose now. Used by RecapFromStand and RecapHeld (task supplies
    cap_task, TWIST_DEG and MAX_STROKES)."""
    c = task.cap_task
    d = ep.data
    thread = ep.extra["thread"]
    # On the neck (the thread has drawn it onto the axis): the opening hold with the fingers
    # eased, planned again from where the bottle is now (the hands nudge it over the stage).
    pl = c.plan(ep, float(seat_now()[2]))
    yield 0.4, np.concatenate([q[ik.qadr], pl.eased_end])
    if wait:  # the left hand is still on its way to the bottle
        yield wait, np.concatenate([q[ik.qadr], pl.eased_end])
    # Screw it closed by turning the whole hand about the cap's axis, the way a wrist turns a
    # cap, while the left hand holds the bottle (bottle_steady): pinch, turn clockwise
    # TWIST_DEG, ease the fingers, turn back, pinch again. Turning with the opening's finger
    # strokes instead, the hand had to hold the shoulder as well and pushed the light, emptied
    # bottle about.
    centre = seat_now()
    pinch = pl.pinch(pl.alphas[0], c.HOLD_THUMB_SQUEEZE, c.HOLD_INDEX_SQUEEZE, 0.0)
    eased = pl.eased_start.copy()
    for h in (pinch, eased):
        for j in pl.wrap:
            h[pl.hidx[j]] = 0.0

    # The plan is for an upright bottle; the left hand's grasp leaves it leaning several degrees,
    # and turning about the vertical then dragged the neck sideways and up at every stroke. So the
    # hold is carried over to the bottle's axis now, and the hand turns about that axis.
    palm_from_centre = pl.palm - (d.body("pet").xpos + [0.0, 0.0, thread.rest_z])

    def turned(deg: float) -> np.ndarray:
        axis = d.body("pet").xmat.reshape(3, 3)[:, 2]
        lean, _ = R.align_vectors([axis], [[0.0, 0.0, 1.0]])
        turn = R.from_rotvec(np.deg2rad(deg) * axis) * lean
        return arm(centre + turn.apply(palm_from_centre), turn.as_matrix() @ pl.rot)

    yield 0.8, np.concatenate([turned(0.0), eased])
    for _ in range(task.MAX_STROKES):
        centre = seat_now()  # turn about where the neck is now
        yield 0.4, np.concatenate([turned(0.0), pinch])
        for k in range(1, 4):
            yield 0.3, np.concatenate([turned(-task.TWIST_DEG * k / 3), pinch])
            if thread.closed:
                break
        yield 0.3, np.concatenate([q[ik.qadr], eased])
        if thread.closed:
            break
        yield 0.5, np.concatenate([turned(0.0), eased])
    q[ik.qadr] = turned(0.0)
    yield 0.6, np.concatenate([q[ik.qadr], np.zeros(20)])
    yield 1.0, np.concatenate([pl.q_pre, np.zeros(20)])


class RecapHeld(Task):
    """Right hand: the bottle back on the table and the cap still pinched between thumb and index
    (as the held pour with the rl_hold policy and place_bottle leave them): let go of the bottle,
    set the cap on the neck and screw it closed, then let go. Run beside bottle_steady, whose left
    hand holds the bottle while the cap is turned.
    """

    name = "recap_held"
    side = "right"
    instruction = "put the cap back on the bottle and close it"
    seconds = 40.0
    TWIST_DEG = RecapFromStand.TWIST_DEG
    MAX_STROKES = 20
    ON_LIP = RecapFromStand.ON_LIP
    CENTRE_TOL, CENTRE_GAIN = RecapFromStand.CENTRE_TOL, RecapFromStand.CENTRE_GAIN

    def __init__(self, n_particles: int = CapOpen.N_PARTICLES, model: mujoco.MjModel | None = None,
                 policy: str | Path | None = None) -> None:
        super().__init__(n_particles, model)
        self.place_bottle = PlaceBottle(n_particles, model=self.model)
        self.place_bottle.pour.policy_path = policy
        self.place_cup = PlaceCup(n_particles, model=self.model)
        self.place_cup.pour = self.place_bottle.pour
        self.cap_task = self.place_bottle.cap_task

    def reset(self, rng: np.random.Generator) -> Episode:
        d, extra = after_place(self.place_bottle, self.place_cup, rng)
        ep = self.begin(d)
        ep.extra.update(extra)
        return ep

    def success(self, ep: Episode) -> bool:
        return bool(ep.extra["thread"].closed)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        thread = ep.extra["thread"]
        ik = ArmIK(m, self.side)
        q = d.qpos.copy()

        def arm(target: np.ndarray, rot: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        def seat_now() -> np.ndarray:
            rb = d.body("pet").xmat.reshape(3, 3)
            return d.body("pet").xpos + rb @ np.array([0, 0, thread.rest_z])

        # Let go of the bottle with the middle, ring and little fingers, keeping the pinch on the
        # cap; bottle_steady's left hand comes for the bottle only once the cap is on the neck.
        arm_now = d.ctrl[ep.actuators][:7].copy()
        pinch = d.ctrl[ep.actuators][7:].copy()
        for i, name in enumerate(self.joints[7:]):
            if any(f in name for f in ("middle_finger", "ring", "pinky")):
                pinch[i] = 0.0
        yield 0.8, np.concatenate([arm_now, pinch])
        palm, rot = ik.palm_pose(d)
        back = palm - 0.03 * rot[:, 1]
        yield 0.6, np.concatenate([arm(back, rot), pinch])
        yield 0.6, np.concatenate([arm(back + [0, 0, 0.05], rot), pinch])
        # Turn the hand so the cap's axis is vertical, keeping where the cap is in the hand.
        palm, rot = ik.palm_pose(d)
        cap_in_palm = rot.T @ (d.body("pet_cap").xpos - palm)
        align, _ = R.align_vectors([[0.0, 0.0, 1.0]], [d.body("pet_cap").xmat.reshape(3, 3)[:, 2]])
        rot = align.as_matrix() @ rot

        def palm_for(cap: np.ndarray) -> np.ndarray:
            return cap - rot @ cap_in_palm

        # Across to above the neck, then down, steering by where the cap really is.
        start = d.body("pet_cap").xpos.copy()
        above = seat_now() + [0, 0, 0.04]
        target = palm_for(start)
        for k in range(1, 5):
            target = palm_for(start + k / 4 * (above - start))
            yield 0.4, np.concatenate([arm(target, rot), pinch])
        for dz in (0.03, 0.02, 0.012):
            err = seat_now() + [0, 0, dz] - d.body("pet_cap").xpos
            target = target + err
            yield 0.6, np.concatenate([arm(target, rot), pinch])
        for _ in range(6):
            err = seat_now() - d.body("pet_cap").xpos
            if np.hypot(err[0], err[1]) < self.CENTRE_TOL:
                break
            target = target + self.CENTRE_GAIN * np.array([err[0], err[1], 0.0])
            yield 0.6, np.concatenate([arm(target, rot), pinch])
        for dz in (self.ON_LIP, self.ON_LIP / 2, 0.0):
            err = seat_now() + [0, 0, dz] - d.body("pet_cap").xpos
            target = target + np.array([err[0], err[1], 0.0])
            target[2] += err[2]
            yield 0.6, np.concatenate([arm(target, rot), pinch])
        yield from wrist_close(self, ep, ik, q, arm, seat_now, wait=BottleSteady.REACH_S)


class BottleSteady(Task):
    """Left hand: hold the bottle still on the table while the right hand screws the cap on (an
    emptied bottle, 25 g, turned with the cap instead of letting the cap turn on it), then let go.

    A full grasp of the body from the bottle's left side, the mirror of the right hand's pouring
    grasp; it holds until the cap's thread reports it closed (or HOLD_MAX_S), then opens and
    draws back.
    """

    name = "bottle_steady"
    side = "left"
    instruction = "hold the bottle still while the cap is closed"
    seconds = 50.0
    # Palm site from the bottle's bottom centre: just under the shoulder (the right hand no longer
    # holds it there while it turns the cap), close to the cap so the bottle does not lean as the
    # cap is turned; any lower, the hand meets the table.
    GRASP_OFFSET = np.array([-0.015, 0.030, 0.135])
    # Reach for the bottle only once the cap is on the neck (recap_held: the right hand's pinch
    # holds the cap deep in the hand, and lowering it, its thumb met the left one at the bottle).
    AFTER_ENGAGE = False
    WAIT_MAX_S = 30.0
    REACH_S = 4.2  # from the start of the reach to the bottle held, and a margin
    HOLD_MAX_S = 45.0

    def success(self, ep: Episode) -> bool:
        b = ep.data.body("pet")
        return bool(b.xmat[8] > np.cos(np.deg2rad(10)) and abs(b.xpos[2] - TABLE_Z) < 0.01)

    def expert(self, ep: Episode) -> Iterator[tuple[float, np.ndarray]]:
        m, d = ep.model, ep.data
        ik = ArmIK(m, self.side)
        home = mujoco.MjData(m)
        set_home(m, home)
        mujoco.mj_kinematics(m, home)
        _, rot = ik.palm_pose(home)
        hand_names = [j.removeprefix(Names.hand_prefix(self.side) + self.side[0] + "_") for j in self.joints[7:]]
        closed = np.array([POWER_CLOSED.get(n, 0.0) for n in hand_names])
        q = d.qpos.copy()

        def arm(target: np.ndarray) -> np.ndarray:
            sol, _, _ = ik.solve(q, target, rot)
            q[ik.qadr] = sol
            return sol

        thread = ep.extra.get("thread")
        waited = 0.0
        while self.AFTER_ENGAGE and thread is not None and thread.state == "released" and waited < self.WAIT_MAX_S:
            yield 0.5, d.ctrl[ep.actuators].copy()
            waited += 0.5
        grasp = d.body("pet").xpos + self.GRASP_OFFSET
        yield 1.2, np.concatenate([arm(grasp + [0.0, 0.08, 0.06]), np.zeros(20)])
        yield 1.0, np.concatenate([arm(grasp + [0.0, 0.08, 0.0]), np.zeros(20)])
        yield 1.0, np.concatenate([arm(grasp), np.zeros(20)])
        yield 0.8, np.concatenate([q[ik.qadr], closed])
        held = 0.0
        while held < self.HOLD_MAX_S and not (thread is not None and thread.closed):
            yield 0.5, np.concatenate([q[ik.qadr], closed])
            held += 0.5
        yield 0.6, np.concatenate([q[ik.qadr], 0.5 * closed])
        yield 0.6, np.concatenate([arm(grasp + [0.0, 0.05, 0.0]), np.zeros(20)])
        yield 0.8, np.concatenate([arm(grasp + [0.0, 0.08, 0.08]), np.zeros(20)])


TASKS: dict[str, Callable[..., Task]] = {
    "cup_bring": CupBring, "pour": Pour, "cap_open": CapOpen, "pour_held": PourHeld,
    "place_bottle": PlaceBottle, "place_cup": PlaceCup, "recap": RecapFromStand, "recap_held": RecapHeld,
    "cap_put_down": CapPutDown, "bottle_steady": BottleSteady,
}


def run_experts(plans: list[tuple[Task, Episode]], on_frame: Callable[[int], None] | None = None) -> list[bool]:
    """Drive several tasks' experts at once on one shared state, each on its own joints.

    Each expert moves through its segments at its own pace; one that has
    finished holds its last targets until all have. on_frame(step) is called
    at FPS before the physics step, with every task's targets already set.
    Returns each task's success.
    """
    d = plans[0][1].data
    m = plans[0][1].model
    every = round(1 / FPS / m.opt.timestep)
    state = []
    for task, ep in plans:
        state.append({"gen": task.expert(ep), "start": d.ctrl[ep.actuators].copy(), "target": None, "n": 0, "k": 0, "done": False})
    step = 0
    while True:
        for (task, ep), st in zip(plans, state):
            if st["done"]:
                continue
            if st["k"] >= st["n"]:
                if st["target"] is not None:
                    st["start"] = st["target"]
                nxt = next(st["gen"], None)
                if nxt is None:
                    st["done"] = True
                    continue
                st["n"], st["target"], st["k"] = int(nxt[0] / m.opt.timestep), nxt[1], 0
            a = min(1.0, (st["k"] + 1) / (0.8 * st["n"]))
            d.ctrl[ep.actuators] = st["start"] + a * (st["target"] - st["start"])
            st["k"] += 1
        if all(st["done"] for st in state):
            break
        if on_frame is not None and step % every == 0:
            on_frame(step)
        mujoco.mj_step(m, d)
        for task, ep in plans:
            task.after_step(ep)
        step += 1
    return [task.success(ep) for task, ep in plans]


def run_expert(task: Task, ep: Episode, on_frame: Callable[[Episode, np.ndarray], None] | None = None) -> bool:
    """Drive the episode with the scripted expert; call on_frame(ep, action) at FPS. Returns success."""
    m, d = ep.model, ep.data
    every = round(1 / FPS / m.opt.timestep)
    start = d.ctrl[ep.actuators].copy()
    step = 0
    for seconds, target in task.expert(ep):
        n = int(seconds / m.opt.timestep)
        for k in range(n):
            a = min(1.0, (k + 1) / (0.8 * n))
            d.ctrl[ep.actuators] = start + a * (target - start)
            if on_frame is not None and step % every == 0:
                on_frame(ep, d.ctrl[ep.actuators].astype(np.float32))
            task.step(ep)
            step += 1
        start = target
    return task.success(ep)
