"""Props for the bimanual task: a PET bottle whose cap unscrews, its contents, and a cup.

- The bottle is a hollow shell (a ring of thin boxes, a bottom, a shoulder and a
  neck), so particles can sit inside and pour out of the neck.
- The cap is a hollow free body (a top disc and a skirt around the neck,
  clear of it) held on the neck by two `connect` equality constraints on the
  bottle axis: they leave it free to spin about the axis and nothing else.
  `CapThread` watches the cap's rotation relative to the bottle and, after
  `turns_to_release` turns in the unscrewing direction, deactivates the
  constraints: from then on the cap is an ordinary loose body that rests on
  the neck's lip and can be lifted off. The thread's pitch (the cap rising as
  it turns) is not modelled.
- The liquid is a set of small spheres (a common proxy for pouring); the
  number that end up inside the cup is the pouring score.

Dimensions are those of a common 500 ml PET bottle and a small cup, not
measurements of particular props.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .objects import OBJECT_CONAFFINITY, OBJECT_CONTYPE

G = mujoco.mjtGeom

BOTTLE_RADIUS = 0.0325
BOTTLE_BODY_HEIGHT = 0.15
SHOULDER_HEIGHT = 0.03
NECK_RADIUS = 0.0125  # inner opening radius
NECK_HEIGHT = 0.02
WALL = 0.002
# Bottoms and the cup wall are thicker than the bottle's wall: the particles
# tunnelled through 2-3 mm plates when the bottle was lifted and when they
# fell into the cup.
BOTTOM = 0.008
CUP_WALL = 0.005
# Hollow cap: a top disc and a skirt whose inside clears the neck's ring of
# boxes (corners out to about 15.5 mm) by 1.5 mm, so a thumb pushing it
# sideways does not press it against the neck while it is still on.
CAP_RADIUS = 0.0185  # outer
# 10 mm tall (a 7 mm skirt): the Wuji's thumb and index, pinching the free
# cap from opposite sides, draw it up only about 5 mm before they slip, and a
# taller skirt stayed on the neck.
CAP_HALF_HEIGHT = 0.005
CAP_TOP = 0.003  # top disc thickness
CAP_SKIRT_WALL = 0.0015

CUP_RADIUS = 0.038
CUP_HEIGHT = 0.09

PARTICLE_RADIUS = 0.003
CAP_FRICTION_NM = 0.01
# The pour task starts from a half-open cap, chosen for the look and an easy
# twist rather than realism: it sits CAP_HALF_OPEN_RAISE above the closed seat,
# turns with CAP_HALF_OPEN_FRICTION_NM, and comes free after
# CAP_HALF_OPEN_TURNS (pass that to CapThread).
# The friction is the lowest of the values tried (0.003 and 0.001 N m): the
# Wuji thumb, sliding over the cap the way a person's thumb does, turned it
# about 32 deg per stroke at 0.001 and about 18 deg at 0.003.
CAP_HALF_OPEN_RAISE = 0.004
CAP_HALF_OPEN_FRICTION_NM = 0.001
CAP_HALF_OPEN_TURNS = 0.5
CAP_HALF_OPEN_CLOSE_TURNS = 0.25  # turned back on this far (clockwise) after engaging, it counts as closed


def _collide(geom: mujoco.MjsGeom, friction: float = 1.0) -> None:
    geom.contype = OBJECT_CONTYPE
    geom.conaffinity = OBJECT_CONAFFINITY
    geom.condim = 4
    geom.friction = [friction, 0.01, 0.001]


def _ring(body: mujoco.MjsBody, name: str, radius: float, z0: float, z1: float, n: int, rgba: list[float],
          wall: float = WALL) -> None:
    """A vertical tube of `n` thin boxes from z0 to z1 at `radius` (to the wall's centre)."""
    half_w = radius * np.tan(np.pi / n) + wall
    for i in range(n):
        a = 2 * np.pi * i / n
        g = body.add_geom(
            name=f"{name}_{i}",
            type=G.mjGEOM_BOX,
            size=[wall / 2, half_w, (z1 - z0) / 2],
            pos=[radius * np.cos(a), radius * np.sin(a), (z0 + z1) / 2],
            quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
            rgba=rgba,
        )
        _collide(g)


def _cone(body: mujoco.MjsBody, name: str, r0: float, r1: float, z0: float, z1: float, n: int, rgba: list[float]) -> None:
    """A conical band of thin boxes from radius r0 at z0 to r1 at z1 (the shoulder)."""
    length = float(np.hypot(r0 - r1, z1 - z0))
    tilt = float(np.arctan2(r0 - r1, z1 - z0))  # inward lean from vertical
    rm, zm = (r0 + r1) / 2, (z0 + z1) / 2
    half_w = r0 * np.tan(np.pi / n) + WALL
    for i in range(n):
        a = 2 * np.pi * i / n
        # Rotate about z by a, then lean inward about the local y axis.
        qz = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])
        qy = np.array([np.cos(-tilt / 2), 0, np.sin(-tilt / 2), 0])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qz, qy)
        g = body.add_geom(
            name=f"{name}_{i}",
            type=G.mjGEOM_BOX,
            size=[WALL / 2, half_w, length / 2],
            pos=[rm * np.cos(a), rm * np.sin(a), zm],
            quat=q.tolist(),
            rgba=rgba,
        )
        _collide(g)


@dataclass
class BottleNames:
    bottle: str
    cap: str
    eq: tuple[str, str]
    particles: list[str]


def add_pet_bottle(spec: mujoco.MjSpec, name: str, pos: list[float], n_particles: int = 40,
                   half_open: bool = False) -> BottleNames:
    """A capped 500 ml-style bottle standing at `pos` (its bottom centre), filled with particles.

    `half_open` seats the cap higher and makes it easier to turn (the CAP_HALF_OPEN_* values).
    """
    world = spec.worldbody
    body = world.add_body(name=name, pos=pos)
    body.add_freejoint(name=f"{name}_free")
    clear = [0.75, 0.9, 1.0, 0.35]
    # Bottom plate and the cylindrical wall.
    g = body.add_geom(name=f"{name}_bottom", type=G.mjGEOM_CYLINDER, size=[BOTTLE_RADIUS, BOTTOM / 2, 0], pos=[0, 0, BOTTOM / 2], rgba=clear)
    _collide(g)
    _ring(body, f"{name}_wall", BOTTLE_RADIUS, 0.0, BOTTLE_BODY_HEIGHT, 20, clear)
    top = BOTTLE_BODY_HEIGHT + SHOULDER_HEIGHT
    _cone(body, f"{name}_shoulder", BOTTLE_RADIUS, NECK_RADIUS + WALL, BOTTLE_BODY_HEIGHT, top, 20, clear)
    _ring(body, f"{name}_neck", NECK_RADIUS + WALL / 2, top, top + NECK_HEIGHT, 12, [0.85, 0.95, 1.0, 0.6])
    # Mass of the empty bottle (about 25 g) spread over its wall geoms.
    for geom in body.geoms:
        geom.mass = 0.025 / len(body.geoms)
    neck_top = top + NECK_HEIGHT

    # The top disc's underside sits on the neck's lip (raised when half open).
    cap = world.add_body(name=f"{name}_cap", pos=[pos[0], pos[1], pos[2] + neck_top - (CAP_HALF_HEIGHT - CAP_TOP)
                                                 + (CAP_HALF_OPEN_RAISE if half_open else 0.0)])
    cap_joint = cap.add_freejoint(name=f"{name}_cap_free")
    # Thread friction: turning takes at least CAP_FRICTION_NM, and the cap
    # spins about 2 turns/s per 0.02 N m beyond that. Without it the 3 g cap
    # spins up without bound and the integration diverges.
    cap_joint.frictionloss = CAP_HALF_OPEN_FRICTION_NM if half_open else CAP_FRICTION_NM
    cap_joint.damping = [0.0015, 0.0, 0.0]
    cap_joint.armature = 1e-5
    blue = [0.1, 0.45, 0.85, 1]
    g = cap.add_geom(name=f"{name}_cap_geom", type=G.mjGEOM_CYLINDER, size=[CAP_RADIUS, CAP_TOP / 2, 0],
                     pos=[0, 0, CAP_HALF_HEIGHT - CAP_TOP / 2], rgba=blue)
    _collide(g, friction=1.2)
    _ring(cap, f"{name}_cap_skirt", CAP_RADIUS - CAP_SKIRT_WALL / 2, -CAP_HALF_HEIGHT, CAP_HALF_HEIGHT - CAP_TOP, 20, blue,
          wall=CAP_SKIRT_WALL)
    for geom in cap.geoms:
        geom.mass = 0.003 / len(cap.geoms)
        # Ribbed plastic held between fingertips: grippy, with torsional friction (condim 4). No
        # rolling friction: with it (condim 6) a cap let go above the table stayed propped on its
        # rim at about 40 deg. Against the fingertips (condim 6) the contact still has it.
        geom.friction = [1.5, 0.02, 0.0]
        geom.condim = 4
    eqs = []
    for k, dz in enumerate((-CAP_HALF_HEIGHT, CAP_HALF_HEIGHT)):
        eq = spec.add_equality(name=f"{name}_cap_hold_{k}", type=mujoco.mjtEq.mjEQ_CONNECT, name1=f"{name}_cap", name2=name,
                               objtype=mujoco.mjtObj.mjOBJ_BODY)
        eq.data[0:3] = [0, 0, dz]  # anchor in the cap's frame, on its axis
        eq.solref = [0.004, 1.0]
        eqs.append(eq.name)

    particles = []
    rng = np.random.default_rng(0)
    per_layer = 9
    for i in range(n_particles):
        layer, slot = divmod(i, per_layer)
        a = 2 * np.pi * slot / per_layer + 0.3 * layer
        r = (0.012 if slot % 2 else 0.022) if slot else 0.0
        p = world.add_body(name=f"{name}_p{i}", pos=[pos[0] + r * np.cos(a), pos[1] + r * np.sin(a),
                                                      pos[2] + BOTTOM + 0.005 + layer * 2.2 * PARTICLE_RADIUS + rng.uniform(0, 1e-4)])
        p.add_freejoint()
        g = p.add_geom(name=f"{name}_p{i}_geom", type=G.mjGEOM_SPHERE, size=[PARTICLE_RADIUS, 0, 0], mass=0.002, rgba=[0.3, 0.6, 1.0, 1])
        # Frictionless (condim 1) so the particles flow through the neck
        # instead of arching and jamming in it, and overdamped so one that
        # falls into the cup settles instead of bouncing out: closer to a
        # liquid than to marbles.
        _collide(g)
        g.condim = 1
        g.solref = [0.004, 1.5]
        g.margin = 0.001
        particles.append(p.name)
    return BottleNames(name, f"{name}_cap", (eqs[0], eqs[1]), particles)


def add_cup(spec: mujoco.MjSpec, name: str, pos: list[float]) -> str:
    """An open cup standing at `pos` (bottom centre)."""
    body = spec.worldbody.add_body(name=name, pos=pos)
    body.add_freejoint(name=f"{name}_free")
    color = [0.95, 0.85, 0.7, 1]
    g = body.add_geom(name=f"{name}_bottom", type=G.mjGEOM_CYLINDER, size=[CUP_RADIUS + CUP_WALL, BOTTOM / 2, 0], pos=[0, 0, BOTTOM / 2], rgba=color)
    _collide(g)
    _ring(body, f"{name}_wall", CUP_RADIUS + CUP_WALL / 2, 0.0, CUP_HEIGHT, 20, color, wall=CUP_WALL)
    for geom in body.geoms:
        geom.mass = 0.08 / len(body.geoms)
    return name


class CapThread:
    """The cap's thread: it comes off after `turns_to_release` turns counter-clockwise (seen from
    above), and goes back on.

    States: "on" (held on the neck by the two connect constraints, free to spin) -> "released"
    (constraints off, a loose body; once it has been taken ARM_DISTANCE away from the seat it can
    go back on) -> "engaged" (set back on the neck: within ENGAGE_OFFSET of its axis, level within
    ENGAGE_TILT_DEG, at about the lip's height; the constraints come back on and draw it onto the
    axis, seated on the lip) -> "closed" (turned `turns_to_close` clockwise since engaging).

    Drawing the cap onto the axis stands in for what the neck's and the cap's chamfers do on a
    real bottle (set down a few millimetres off, a cap finds the thread as it is turned); the
    modelled cap has no thread or chamfer, so without it the cap would have to be placed within
    the 1.5 mm the skirt clears the neck by.

    Re-engaging moves the constraints' anchors on the bottle to where the cap sits, which writes
    the model's eq_data. The thread's resistance is the friction and damping of the cap's free
    joint, which act on every axis; once the cap is loose they are switched off (with them a cap
    let go on the table could not tip over and stayed propped on its rim), and back on when it
    engages. A new CapThread restores what the model was compiled with, so each episode starts from
    the compiled half-open seat.
    """

    ENGAGE_OFFSET = 0.010
    ENGAGE_TILT_DEG = 25.0
    ENGAGE_HEIGHT_TOL = 0.008
    ARM_DISTANCE = 0.008
    _compiled: dict[tuple[int, int], np.ndarray] = {}

    def __init__(self, model: mujoco.MjModel, names: BottleNames, turns_to_release: float = 1.5,
                 turns_to_close: float = 0.4) -> None:
        self.model = model
        self.bottle = model.body(names.bottle).id
        self.cap = model.body(names.cap).id
        self.eqs = [model.equality(n).id for n in names.eq]
        for eq in self.eqs:
            key = (id(model), eq)
            if key not in CapThread._compiled:
                CapThread._compiled[key] = model.eq_data[eq].copy()
            model.eq_data[eq] = CapThread._compiled[key]
        dof = model.jnt_dofadr[model.body_jntadr[self.cap]]
        self.dofs = slice(dof, dof + 6)
        key = (id(model), "cap_dofs")
        if key not in CapThread._compiled:
            CapThread._compiled[key] = (model.dof_frictionloss[self.dofs].copy(), model.dof_damping[self.dofs].copy())
        self.thread_friction, self.thread_damping = CapThread._compiled[key]
        self._set_thread(True)
        self.limit = 2 * np.pi * turns_to_release
        self.close_limit = 2 * np.pi * turns_to_close
        self.turned = 0.0  # counter-clockwise, while on
        self.closing = 0.0  # clockwise, while engaged
        self._last: float | None = None
        self.state = "on"
        self.armed = False  # taken off the neck since it came loose
        # Height of the cap's origin above the bottle's when it rests on the lip.
        self.rest_z = BOTTLE_BODY_HEIGHT + SHOULDER_HEIGHT + NECK_HEIGHT - (CAP_HALF_HEIGHT - CAP_TOP)

    def _set_thread(self, on: bool) -> None:
        self.model.dof_frictionloss[self.dofs] = self.thread_friction if on else 0.0
        self.model.dof_damping[self.dofs] = self.thread_damping if on else 0.0

    @property
    def released(self) -> bool:
        return self.state != "on"

    @property
    def closed(self) -> bool:
        return self.state == "closed"

    def _relative(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rb = data.xmat[self.bottle].reshape(3, 3)
        rc = data.xmat[self.cap].reshape(3, 3)
        return rb.T @ rc, rb.T @ (data.xpos[self.cap] - data.xpos[self.bottle])

    def _yaw_step(self, data: mujoco.MjData) -> float:
        rel, _ = self._relative(data)
        yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        step = 0.0 if self._last is None else (yaw - self._last + np.pi) % (2 * np.pi) - np.pi
        self._last = yaw
        return step

    def can_engage(self, data: mujoco.MjData) -> bool:
        rel, pos = self._relative(data)
        level = rel[2, 2] > np.cos(np.deg2rad(self.ENGAGE_TILT_DEG))
        centred = np.hypot(pos[0], pos[1]) < self.ENGAGE_OFFSET
        resting = abs(pos[2] - self.rest_z) < self.ENGAGE_HEIGHT_TOL
        return bool(level and centred and resting)

    def update(self, data: mujoco.MjData) -> None:
        if self.state == "closed":
            return
        step = self._yaw_step(data)
        if self.state == "on":
            self.turned += step
            if self.turned >= self.limit:
                for eq in self.eqs:
                    data.eq_active[eq] = 0
                self._set_thread(False)
                self.state = "released"
        elif self.state == "released":
            if not self.armed:
                _, pos = self._relative(data)
                self.armed = bool(pos[2] - self.rest_z > self.ARM_DISTANCE or np.hypot(pos[0], pos[1]) > self.ARM_DISTANCE)
            elif self.can_engage(data):
                # On the axis, seated on the lip: the compiled anchors (the half-open seat) lowered
                # by how far that seat is raised above the lip.
                for eq in self.eqs:
                    compiled = CapThread._compiled[(id(self.model), eq)]
                    self.model.eq_data[eq][3:6] = compiled[3:6] - [0.0, 0.0, CAP_HALF_OPEN_RAISE]
                    data.eq_active[eq] = 1
                self._set_thread(True)
                self.state = "engaged"
                self.closing = 0.0
        elif self.state == "engaged":
            self.closing -= step
            if self.closing >= self.close_limit:
                self.state = "closed"


CAP_STAND_RADIUS = 0.03
CAP_STAND_HEIGHT = 0.065  # the pick-up hold reaches no lower: 40 mm left the fingertips 25 mm above the cap


def add_cap_stand(spec: mujoco.MjSpec, pos: list[float]) -> None:
    """A fixed post on the table (centre of its base at `pos`) for the opened cap to wait on.

    With the palm upright, as the hand holds the bottle to open it, the hand's little-finger edge
    meets the table before the pinched cap does; on the post the cap is set down and picked up
    with the same hold that opened it (the post stands where the bottle's shoulder was).
    """
    body = spec.worldbody.add_body(name="cap_stand", pos=[pos[0], pos[1], pos[2] + CAP_STAND_HEIGHT / 2])
    g = body.add_geom(name="cap_stand_geom", type=G.mjGEOM_CYLINDER, size=[CAP_STAND_RADIUS, CAP_STAND_HEIGHT / 2, 0],
                      rgba=[0.35, 0.35, 0.38, 1])
    _collide(g)


def particles_in_cup(model: mujoco.MjModel, data: mujoco.MjData, names: BottleNames, cup: str) -> int:
    body = model.body(cup).id
    rc = data.xmat[body].reshape(3, 3)
    count = 0
    for p in names.particles:
        local = rc.T @ (data.body(p).xpos - data.xpos[body])
        if np.hypot(local[0], local[1]) < CUP_RADIUS and 0.0 < local[2] < CUP_HEIGHT:
            count += 1
    return count
