"""Force/torque aggregation for rigid bodies."""

from __future__ import annotations

import numpy as np

from .base import Model
from ..math.quat import quat_to_rotmat
from ..math.vector import cross
from ..state.system import SystemState


class RigidBodyForces(Model):
    """Apply world-frame forces at body-frame points.

    Application points are specified in BODY coordinates relative to the CoM.
    """

    def __init__(self, mass: np.ndarray, inertia_body: np.ndarray) -> None:
        self.mass = np.asarray(mass, dtype=np.float64)
        inertia = np.asarray(inertia_body, dtype=np.float64)
        if self.mass.ndim != 1:
            raise ValueError("mass must have shape (M,)")
        if inertia.ndim == 2:
            inertia = np.broadcast_to(inertia, (self.mass.shape[0], 3, 3)).copy()
        if inertia.shape != (self.mass.shape[0], 3, 3):
            raise ValueError("inertia_body must have shape (M, 3, 3)")
        self.inertia_body = inertia
        self.inertia_body_inv = np.zeros_like(inertia)
        for i in range(self.mass.shape[0]):
            self.inertia_body_inv[i] = np.linalg.pinv(inertia[i])

        self.body_index = np.zeros(0, dtype=np.int64)
        self.forces_world = np.zeros((0, 3), dtype=np.float64)
        self.points_body = np.zeros((0, 3), dtype=np.float64)

    def set_applied_forces(
        self,
        body_index: np.ndarray,
        forces_world: np.ndarray,
        points_body: np.ndarray,
    ) -> None:
        self.body_index = np.asarray(body_index, dtype=np.int64)
        self.forces_world = np.asarray(forces_world, dtype=np.float64)
        self.points_body = np.asarray(points_body, dtype=np.float64)
        if self.forces_world.shape != self.points_body.shape:
            raise ValueError("forces_world and points_body must have shape (K, 3)")
        if self.body_index.ndim != 1 or self.body_index.shape[0] != self.forces_world.shape[0]:
            raise ValueError("body_index must have shape (K,)")

    def acc_particles(self, state: SystemState) -> np.ndarray:
        if state.particles is None:
            return np.zeros((0, 3), dtype=np.float64)
        n = state.particles.pos.shape[0]
        return np.zeros((n, 3), dtype=np.float64)

    def _net_force_torque_world(self, state: SystemState) -> tuple[np.ndarray, np.ndarray]:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

        m = state.rigid_bodies.pos.shape[0]
        if self.forces_world.shape[0] == 0:
            return np.zeros((m, 3), dtype=np.float64), np.zeros((m, 3), dtype=np.float64)

        rot = quat_to_rotmat(state.rigid_bodies.quat)
        rot_forces = rot[self.body_index]
        r_world = np.einsum("bij,bj->bi", rot_forces, self.points_body)
        torque_world = cross(r_world, self.forces_world)

        net_force = np.zeros((m, 3), dtype=np.float64)
        net_torque = np.zeros((m, 3), dtype=np.float64)
        np.add.at(net_force, self.body_index, self.forces_world)
        np.add.at(net_torque, self.body_index, torque_world)
        return net_force, net_torque

    def acc_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        net_force, _ = self._net_force_torque_world(state)
        return net_force / self.mass[:, None]

    def alpha_rigid(self, state: SystemState) -> np.ndarray:
        if state.rigid_bodies is None:
            return np.zeros((0, 3), dtype=np.float64)
        rot = quat_to_rotmat(state.rigid_bodies.quat)
        _, net_torque_world = self._net_force_torque_world(state)
        tau_body = np.einsum("bij,bj->bi", np.transpose(rot, (0, 2, 1)), net_torque_world)

        omega = state.rigid_bodies.omega
        Iw = np.einsum("bij,bj->bi", self.inertia_body, omega)
        omega_cross_Iw = cross(omega, Iw)
        rhs = tau_body - omega_cross_Iw
        return np.einsum("bij,bj->bi", self.inertia_body_inv, rhs)
