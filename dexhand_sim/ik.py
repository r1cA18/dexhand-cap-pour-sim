"""Damped least-squares IK for one arm's palm site (position and orientation).

Solved on a scratch MjData (kinematics only), so the physics state is never
written; the result is a set of joint targets for the arm's position actuators.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS, Names


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str) -> None:
        self.model = model
        self.side = side
        self.scratch = mujoco.MjData(model)
        self.site = model.site(Names.palm(side)).id
        joints = [model.joint(Names.arm_joint(side, j)) for j in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([j.range[0] for j in joints])
        self.hi = np.array([j.range[1] for j in joints])

    def palm_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(
        self,
        qpos: np.ndarray,
        pos: np.ndarray,
        rot: np.ndarray,
        iters: int = 200,
        damping: float = 1e-3,
        rot_weight: float = 0.3,
    ) -> tuple[np.ndarray, float, float]:
        """Arm joint targets reaching `pos`/`rot` from `qpos`.

        Returns (arm joint values, position error m, rotation error rad).
        """
        m, d = self.model, self.scratch
        d.qpos[:] = qpos
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur_p = d.site_xpos[self.site]
            cur_r = d.site_xmat[self.site].reshape(3, 3)
            e_p = pos - cur_p
            e_r = 0.5 * sum(np.cross(cur_r[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            err = np.concatenate([e_p, rot_weight * e_r])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        cur_r = d.site_xmat[self.site].reshape(3, 3)
        e_r = 0.5 * sum(np.cross(cur_r[:, i], rot[:, i]) for i in range(3))
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(pos - d.site_xpos[self.site])), float(np.linalg.norm(e_r))


class SiteIK:
    """Damped least-squares IK for several sites at once over a chosen set of joints.

    Each target is (site name, world position, weight). Each optional axis
    target is (body name, axis in the body frame, world direction, weight): it
    turns that body axis toward the direction (a fingertip pad toward an
    object, say). Joints outside
    `joints` stay where `qpos` has them. A small pull toward `qpos` keeps the
    solution near the starting posture when the targets leave freedom.
    """

    def __init__(self, model: mujoco.MjModel, joints: list[str]) -> None:
        self.model = model
        self.scratch = mujoco.MjData(model)
        js = [model.joint(n) for n in joints]
        self.qadr = np.array([j.qposadr[0] for j in js])
        self.dadr = np.array([j.dofadr[0] for j in js])
        self.lo = np.array([j.range[0] for j in js])
        self.hi = np.array([j.range[1] for j in js])

    def solve(
        self,
        qpos: np.ndarray,
        targets: list[tuple[str, np.ndarray, float]],
        axes: list[tuple[str, np.ndarray, np.ndarray, float]] = (),
        iters: int = 300,
        damping: float = 1e-4,
        posture: float = 1e-3,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Values of the chosen joints, and the remaining distance per site (m)."""
        m, d = self.model, self.scratch
        d.qpos[:] = qpos
        q0 = qpos[self.qadr].copy()
        sites = [(m.site(n).id, np.asarray(p, float), w) for n, p, w in targets]
        bodies = [(m.body(b).id, np.asarray(a, float), np.asarray(v, float) / np.linalg.norm(v), w) for b, a, v, w in axes]
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        n = len(self.dadr)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errs = [], []
            for sid, p, w in sites:
                mujoco.mj_jacSite(m, d, jacp, None, sid)
                rows.append(w * jacp[:, self.dadr])
                errs.append(w * (p - d.site_xpos[sid]))
            for bid, a, v, w in bodies:
                cur = d.xmat[bid].reshape(3, 3) @ a
                mujoco.mj_jacBody(m, d, None, jacr, bid)
                rows.append(w * jacr[:, self.dadr])
                errs.append(w * np.cross(cur, v))
            jac, err = np.vstack(rows), np.concatenate(errs)
            # Posture term: a soft pull of every joint toward its start value.
            jac = np.vstack([jac, np.sqrt(posture) * np.eye(n)])
            err = np.concatenate([err, np.sqrt(posture) * (q0 - d.qpos[self.qadr])])
            dq = np.linalg.solve(jac.T @ jac + damping * np.eye(n), jac.T @ err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.1, 0.1), self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        dist = {m.site(sid).name: float(np.linalg.norm(p - d.site_xpos[sid])) for sid, p, _ in sites}
        return d.qpos[self.qadr].copy(), dist
