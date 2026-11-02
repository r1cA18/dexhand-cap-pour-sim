"""The table and the objects, one per grasp type the task uses.

| object | grasp it is meant for |
|---|---|
| bottle | power (wrap) grasp |
| ball | spherical grasp with spread fingers |
| cube | three-finger (tripod) pinch |
| card | lateral (key) pinch, thumb against the side of the index finger |
| bag | hook grasp on the handle |

Sizes and masses are everyday-object guesses, not measurements of the real
props. Positions below are relative to the table top.
"""

from __future__ import annotations

import mujoco

# Bits as in scene.py: an object collides with the robot body, the hands and
# other objects.
OBJECT_CONTYPE = 1
OBJECT_CONAFFINITY = 1 | 2 | 4

# Table top height above the floor, level with the arm base at the scene's
# lift height (scene.LIFT_HOME), in front of the arms.
TABLE_TOP_Z = 0.70
TABLE_CENTER = (0.52, 0.0)
TABLE_HALF = (0.22, 0.45, 0.02)

OBJECTS = ("bottle", "ball", "cube", "card", "bag")


def _collides(geom: mujoco.MjsGeom) -> None:
    geom.contype = OBJECT_CONTYPE
    geom.conaffinity = OBJECT_CONAFFINITY
    geom.condim = 4
    geom.friction = [1.0, 0.01, 0.001]


def add_table(spec: mujoco.MjSpec) -> None:
    table = spec.worldbody.add_body(name="table", pos=[TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP_Z - TABLE_HALF[2]])
    top = table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF), rgba=[0.55, 0.4, 0.28, 1])
    _collides(top)


def add_table_and_objects(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    G = mujoco.mjtGeom
    add_table(spec)

    def free_body(name: str, pos: list[float]) -> mujoco.MjsBody:
        body = world.add_body(name=name, pos=[pos[0], pos[1], pos[2] + TABLE_TOP_Z])
        body.add_freejoint(name=f"{name}_free")
        return body

    bottle = free_body("bottle", [0.48, -0.20, 0.09])
    g = bottle.add_geom(name="bottle_geom", type=G.mjGEOM_CYLINDER, size=[0.032, 0.09, 0], mass=0.30, rgba=[0.2, 0.5, 0.9, 1])
    _collides(g)

    ball = free_body("ball", [0.50, -0.02, 0.035])
    g = ball.add_geom(name="ball_geom", type=G.mjGEOM_SPHERE, size=[0.035, 0, 0], mass=0.10, rgba=[0.95, 0.55, 0.1, 1])
    _collides(g)

    cube = free_body("cube", [0.42, 0.06, 0.0125])
    g = cube.add_geom(name="cube_geom", type=G.mjGEOM_BOX, size=[0.0125] * 3, mass=0.03, rgba=[0.9, 0.2, 0.2, 1])
    _collides(g)

    # The card lies on a pedestal and overhangs its edge toward the robot, so
    # a thumb can reach under it from the side.
    pedestal = world.add_body(name="pedestal", pos=[0.60, 0.20, TABLE_TOP_Z + 0.03])
    g = pedestal.add_geom(name="pedestal_geom", type=G.mjGEOM_BOX, size=[0.04, 0.04, 0.03], rgba=[0.6, 0.6, 0.6, 1])
    _collides(g)
    card = free_body("card", [0.585, 0.20, 0.0625])
    g = card.add_geom(name="card_geom", type=G.mjGEOM_BOX, size=[0.043, 0.027, 0.0025], mass=0.01, rgba=[0.95, 0.95, 0.95, 1])
    _collides(g)

    # A box with an arched handle (three capsules) on top.
    bag = free_body("bag", [0.62, -0.02, 0.06])
    parts = [
        ("bag_box", G.mjGEOM_BOX, [0.05, 0.08, 0.06], [0, 0, 0], None),
        ("bag_post_a", G.mjGEOM_CAPSULE, [0.006, 0.03, 0], [0, -0.045, 0.09], None),
        ("bag_post_b", G.mjGEOM_CAPSULE, [0.006, 0.03, 0], [0, 0.045, 0.09], None),
        ("bag_bar", G.mjGEOM_CAPSULE, [0.006, 0.045, 0], [0, 0, 0.12], [0.7071, 0.7071, 0, 0]),
    ]
    for name, kind, size, pos, quat in parts:
        g = bag.add_geom(name=name, type=kind, size=size, pos=pos, rgba=[0.75, 0.65, 0.45, 1])
        if quat is not None:
            g.quat = quat
        g.mass = 0.25 if name == "bag_box" else 0.01
        _collides(g)
