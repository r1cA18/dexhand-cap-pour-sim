"""A dual-arm upper body with a Wuji Hand 2.1 on each arm flange, as one MuJoCo model.

Sources:
- assets/robot/upper_body.urdf, home_pose.json, head_camera.json: the robot
  description, its home pose and its head camera. They are not included in
  this repository; place them there to run the scene. URDF suffix `_R` is the
  robot's physical LEFT arm, `_L` the physical RIGHT.
- assets/wuji_hand2_1/mjcf/{left,right}.xml: the vendor's Wuji Hand 2.1, rooted
  at its arm-flange mount, 20 position actuators.

Edits to the URDF: the planar base joints are welded; the standoff collision
puck on each flange is removed. The hand's mount face
is seated on the wrist ball. The roll of the hand about the tool
axis is a convention chosen here (HAND_ROLL), not a measured adapter plate.

What a policy may observe is what the robot has: the head camera image and the
robot's own joint states. Object poses are for scoring only (see sensors.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ROBOT_URDF = ASSETS / "robot" / "upper_body.urdf"
HOME_POSE = ASSETS / "robot" / "home_pose.json"
HEAD_CAMERA = ASSETS / "robot" / "head_camera.json"
WELDED = ("base_x", "base_y", "base_yaw")

# Lift height and head tilt the scene starts at. With the lift at 0.187 m the
# arm base (dual_base, 0.513 m above the floor at lift 0) is at 0.70 m, level
# with the table top (objects.TABLE_TOP_Z).
LIFT_HOME = 0.187
# neck_tilt 0.65 rad is the joint's upper limit: the head camera then looks
# down at the table within 12 deg of its centre (searched over the range).
NECK_HOME = {"neck_pan": 0.0, "neck_tilt": 0.65}
WUJI_DIR = ASSETS / "wuji_hand2_1"

# Visual meshes are decimated to at most this many faces each, for rendering
# speed; decimated copies are cached outside the repository.
MESH_MAX_FACES = 3000
MESH_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dexhand-sim" / "meshes"

# Physical side -> URDF suffix and Wuji hand file.
SIDES = {"left": ("R", "left"), "right": ("L", "right")}
ARM_JOINTS = [f"Joint{i}" for i in range(1, 8)]

# Link7 -> TCP_Link in the URDF (JointTCP_S, identical on both arms). MuJoCo's
# fusestatic folds TCP_Link into Link7, so the hand is placed in Link7's frame.
TCP_POS = np.array([0.0, -0.087, 0.0])
TCP_ROT = R.from_euler("xyz", [1.5708, -1.5708, 0.0])
# The wrist ball (TCP_Link cap_a) has radius 0.030 at the TCP origin; seat the
# mount face 2 mm into it, along +z_tcp (the tool axis).
MOUNT_FACE_TCP_Z = 0.028
# Roll of the hand about the tool axis, per physical side, in radians. Chosen
# so that at HOME the palm faces the body midline with the thumb up (the
# convention used for other hands on this arm).
HAND_ROLL = {"left": -np.pi / 2, "right": np.pi / 2}

# Collision groups. The URDF is a motion guard's model: its primitives are
# inflated and overlap between neighbouring links and the torso, so the robot
# body must not collide with itself. It collides with the table and objects.
# Hand geoms collide with each other (fingers against fingers, minus the
# vendor's <exclude> pairs) and with the table and objects.
ROBOT_BIT, OBJECT_BIT, HAND_BIT = 2, 1, 4
HEAD_SHELL_GROUP = 4
FINGERTIP_FRICTION = [1.5, 0.02, 0.002]

# Palm centre in the hand's mount frame: halfway between the wrist and the
# knuckle row (finger proximal bodies at z -0.095), 25 mm out along the palm
# normal (+y on the right hand, -y on the left, from finger flexion in the
# vendor MJCF). A reference point for reaching, not a vendor frame.
PALM_SITE = {"left": [0.0, -0.025, -0.07], "right": [0.0, 0.025, -0.07]}

# Pour task layout: the bottle in front of the right hand (which holds it and
# opens the cap), the cup in front of the left (which brings it over), both on
# the table top (objects.TABLE_TOP_Z).
POUR_BOTTLE_POS = [0.45, -0.12, 0.70]
POUR_CUP_POS = [0.45, 0.12, 0.70]
# A small stand to the bottle's right where the opened cap waits (props.add_cap_stand).
CAP_STAND_POS = [0.32, -0.33, 0.70]  # clear of the way the hand takes to grasp the bottle for pouring

ARM_KP = 400.0
ARM_KV = 40.0


@dataclass(frozen=True)
class Names:
    """Prefixed names inside the composed model."""

    @staticmethod
    def hand_prefix(side: str) -> str:
        return f"wuji_{side[0]}_"

    @staticmethod
    def palm(side: str) -> str:
        return f"wuji_{side[0]}_palm"

    @staticmethod
    def arm_joint(side: str, joint: str) -> str:
        return f"{joint}_{SIDES[side][0]}"


def home_qpos_deg() -> dict[str, list[float]]:
    """HOME per physical side, degrees, Joint1..7 (assets/robot/home_pose.json)."""
    pose = json.loads(HOME_POSE.read_text())["home_pose"]
    return {"left": pose[:7], "right": pose[7:]}


def _prepare_urdf(urdf_text: str, visuals: bool) -> str:
    root = ET.fromstring(urdf_text)
    for el in list(root):
        name = el.get("name", "")
        if el.tag == "joint" and name in WELDED:
            el.set("type", "fixed")
            for child in list(el):
                if child.tag in ("axis", "limit"):
                    el.remove(child)
        if el.tag == "link":
            for vis in el.findall("visual"):
                mesh = vis.find("geometry/mesh")
                # TCP_Link's flange disc is too thin for MuJoCo's mesh volume
                # check, and the hand's mount covers it anyway.
                if not visuals or name.startswith("TCP_Link_") or mesh is not None:
                    el.remove(vis)
        if el.tag == "link" and name.startswith("TCP_Link_"):
            suf = name[-1]
            for col in list(el):
                if col.tag == "collision" and col.get("name", "") in (
                    f"TCP_Link_{suf}_capsule",
                    f"TCP_Link_{suf}_capsule_cap_b",
                ):
                    el.remove(col)
    compiler = ET.SubElement(ET.SubElement(root, "mujoco"), "compiler")
    compiler.set("discardvisual", "false")
    compiler.set("strippath", "false")
    return ET.tostring(root, encoding="unicode")


def _decimated(path: Path, max_faces: int) -> Path:
    """A cached copy of the mesh at `path` with at most `max_faces` faces (the original if it is already small)."""
    import fast_simplification
    import trimesh

    key = hashlib.sha1(f"{path.resolve()}:{path.stat().st_mtime_ns}:{max_faces}:welded".encode()).hexdigest()[:12]
    out = MESH_CACHE / f"{path.stem}-{key}.obj"
    if out.exists():
        return out
    # process=True welds the duplicated vertices of STL triangle soup; the
    # simplifier shreds an unwelded mesh.
    mesh = trimesh.load(path, force="mesh", process=True)
    if len(mesh.faces) <= max_faces:
        return path
    verts, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=1 - max_faces / len(mesh.faces))
    MESH_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.obj")  # per process: parallel builds share the cache
    trimesh.Trimesh(verts, faces, process=False).export(tmp)
    tmp.replace(out)
    return out


def _mount_face_z(hand: mujoco.MjModel, mount: str) -> float:
    """The highest point of the mount mesh along the hand's +z (its arm side)."""
    zs = []
    for g in range(hand.ngeom):
        if hand.geom_bodyid[g] != hand.body(mount).id or hand.geom_dataid[g] < 0:
            continue
        mid = hand.geom_dataid[g]
        v = hand.mesh_vert[hand.mesh_vertadr[mid] : hand.mesh_vertadr[mid] + hand.mesh_vertnum[mid]]
        q = hand.geom_quat[g]
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        zs.append((rot.apply(v) + hand.geom_pos[g])[:, 2].max())
    return float(max(zs))


def _hand_spec(side: str) -> tuple[mujoco.MjSpec, float]:
    path = WUJI_DIR / "mjcf" / f"{SIDES[side][1]}.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    meshdir = (path.parent / spec.meshdir).resolve() if spec.meshdir else path.parent
    for mesh in spec.meshes:
        mesh.file = str(_decimated(meshdir / Path(mesh.file).name, MESH_MAX_FACES))
    face_z = _mount_face_z(mujoco.MjModel.from_xml_path(str(path)), f"{side[0]}_mount")
    return spec, face_z


def hand_mount_in_link7(face_z: float, roll: float) -> tuple[np.ndarray, R]:
    """Pose of the hand root (its mount body) in Link7's frame.

    The vendor hand points its fingers along its local -z with the mount face
    at +face_z, so a half turn about x puts the fingers along +z_tcp.
    """
    rot_in_tcp = R.from_euler("z", roll) * R.from_euler("x", np.pi)
    pos_in_tcp = np.array([0.0, 0.0, MOUNT_FACE_TCP_Z + face_z])
    return TCP_POS + TCP_ROT.apply(pos_in_tcp), TCP_ROT * rot_in_tcp


def _add_world(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = 0.001
    # mujoco 3.3.7 refused this model with a sparse Jacobian once the cap's
    # connect constraints were present ("mj_projectConstraint: pre and
    # post-count of B_rownnz are not equal"), and dense ran the pour scene at
    # 0.023x real time; 3.16.0 takes sparse. Hence the version pin.
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # Grasped objects are kept from slipping by a high impratio (frictional
    # constraints made stiffer than normal ones) rather than by the noslip
    # pass: noslip 3 held the bottle as well but took about 90% of the step
    # time, and without either the bottle slid out of the power grasp. At 10
    # a cap pinched between thumb and index still crept down about 1 mm/s;
    # at 50 (with the fingertip contacts below) it stays put.
    spec.option.impratio = 50
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    world = spec.worldbody
    world.add_light(pos=[0.5, 0, 2.0], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[-0.5, 1.0, 1.5], dir=[0.5, -0.5, -1], diffuse=[0.3, 0.3, 0.3])
    floor = world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, 0])
    floor.rgba = [0.85, 0.85, 0.82, 1]
    world.add_camera(name="overview", pos=[1.6, -1.3, 1.9], xyaxes=[0.63, 0.78, 0, -0.33, 0.27, 0.9], fovy=45)
    _add_head_camera(spec)


def _add_head_camera(spec: mujoco.MjSpec) -> None:
    """The head camera at its design pose, with its vertical field of view.

    The pose is a ROS optical frame (x right, y down, z forward); a
    MuJoCo camera looks along -z with y up, so y and z flip. The principal
    point is not modelled.
    """
    cal = json.loads(HEAD_CAMERA.read_text())
    rot = np.array(cal["rotation_matrix"])
    k = cal["intrinsics"]
    cam = spec.body(cal["parent_link"]).add_camera(name="head")
    cam.pos = cal["position_m"]
    x, y, z, w = R.from_matrix(rot @ np.diag([1.0, -1.0, -1.0])).as_quat()
    cam.quat = [w, x, y, z]
    cam.fovy = float(np.rad2deg(2 * np.arctan(k["height"] / 2 / k["fy"])))
    cam.resolution = [k["width"], k["height"]]


def build_spec(objects: bool | str = True, visuals: bool = True, n_particles: int = 60) -> mujoco.MjSpec:
    """The composed spec: the arms, two Wuji hands, actuators, world, and the props.

    `objects`: True or "grasp" for the five grasp-type objects, "pour" for the
    bottle with a half-open cap (in front of the right hand) and the cup (in
    front of the left hand), False for the table-less robot alone. `visuals=False` drops
    the visual meshes (collision primitives only) for faster loading.
    """
    base = mujoco.MjModel.from_xml_string(_prepare_urdf(ROBOT_URDF.read_text(), visuals))
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Round-trip through MJCF: attaching onto a spec parsed straight from
        # URDF crashed on teardown (mujoco 3.10).
        mujoco.mj_saveLastXML(tmp_path, base)
        spec = mujoco.MjSpec.from_file(tmp_path)
    finally:
        Path(tmp_path).unlink()

    _add_world(spec)
    for mesh in spec.meshes:
        mesh.file = str(_decimated(Path(mesh.file), MESH_MAX_FACES))
    for side, (suf, _) in SIDES.items():
        hand, face_z = _hand_spec(side)
        hand.body(f"{side[0]}_mount").add_site(name="palm", pos=PALM_SITE[side], size=[0.006, 0, 0], group=3)
        # The index's last joint (DIP), a point for placing the finger along an
        # object, not only its tip (the cap twist lays the index along the cap).
        hand.body(f"{side[0]}_index_finger_distal").add_site(name="index_dip", size=[0.004, 0, 0], group=3)
        pos, rot = hand_mount_in_link7(face_z, HAND_ROLL[side])
        x, y, z, w = rot.as_quat()
        frame = spec.body(f"Link7_{suf}").add_frame(pos=pos, quat=[w, x, y, z])
        frame.attach_body(hand.body(f"{side[0]}_mount"), Names.hand_prefix(side), "")
        # The vendor file's <exclude> pairs and actuators come with the body.
        for joint in ARM_JOINTS:
            name = Names.arm_joint(side, joint)
            spec.joint(name).damping = [5.0, 0.0, 0.0]  # linear term first (mujoco >= 3.16)
            spec.joint(name).armature = 0.05
            act = spec.add_actuator(name=f"{side}_{joint}", target=name)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.set_to_position(kp=ARM_KP, kv=ARM_KV)
            act.ctrlrange = spec.joint(name).range
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    for name in ("lift", "neck_pan", "neck_tilt"):
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=2000.0 if name == "lift" else 50.0, kv=200.0 if name == "lift" else 5.0)
        act.ctrlrange = spec.joint(name).range
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            # Visual-only. The head's own shell goes to group HEAD_SHELL_GROUP so
            # the head camera, which sits inside it, can render without it.
            if geom.parent is not None and geom.parent.name == "head_link":
                geom.group = HEAD_SHELL_GROUP
            continue
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if _in_hand(spec, geom):
            geom.contype, geom.conaffinity = HAND_BIT, OBJECT_BIT | HAND_BIT
            if geom.parent.name.endswith(("_distal", "_middle")):
                # Finger pads: grippier than the default, and with torsional
                # friction, so a small object pinched between two fingertips
                # neither slides nor rolls out.
                geom.friction = FINGERTIP_FRICTION
                geom.condim = 6
        else:
            geom.contype, geom.conaffinity = ROBOT_BIT, OBJECT_BIT
        # Collision shapes are not drawn (group 3 is hidden by default): the
        # robot is seen through its meshes, and the guard's inflated
        # primitives would otherwise block the head camera.
        geom.group = 3
    # Arms and hands hold themselves up, as the real position-controlled arm does.
    for body in spec.bodies:
        if body.name.startswith(("Link", "wuji_", "dual_base", "torso", "neck_", "head_")):
            body.gravcomp = 1.0
    if objects in (True, "grasp"):
        from .objects import add_table_and_objects

        add_table_and_objects(spec)
    elif objects == "pour":
        from .objects import add_table
        from .props import add_cap_stand, add_cup, add_pet_bottle

        add_table(spec)
        add_pet_bottle(spec, "pet", POUR_BOTTLE_POS, n_particles=n_particles, half_open=True)
        add_cup(spec, "cup", POUR_CUP_POS)
        add_cap_stand(spec, CAP_STAND_POS)
    elif objects:
        raise ValueError(f"unknown objects {objects!r}")
    return spec


def _in_hand(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> bool:
    body = geom.parent
    return body is not None and body.name.startswith("wuji_")


def build_model(objects: bool | str = True, visuals: bool = True, n_particles: int = 60) -> mujoco.MjModel:
    return build_spec(objects, visuals, n_particles).compile()


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put both arms at HOME, the lift and neck at their start values (qpos and targets), hands open."""
    for name, value in {"lift": LIFT_HOME, **NECK_HOME}.items():
        data.qpos[model.joint(name).qposadr[0]] = value
        data.ctrl[model.actuator(name).id] = value
    for side, deg in home_qpos_deg().items():
        for joint, value in zip(ARM_JOINTS, np.deg2rad(deg)):
            name = Names.arm_joint(side, joint)
            data.qpos[model.joint(name).qposadr[0]] = value
            data.ctrl[model.actuator(f"{side}_{joint}").id] = value
    mujoco.mj_forward(model, data)
