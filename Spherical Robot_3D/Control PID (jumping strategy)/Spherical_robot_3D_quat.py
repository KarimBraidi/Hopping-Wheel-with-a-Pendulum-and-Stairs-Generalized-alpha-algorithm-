"""
3D Spherical Robot — Quaternion version.

Mirrors the structure of `Spherical_robot_3D.py` but represents the
sphere's orientation with a unit quaternion and keeps the body-frame
angular velocity as a primary variable in the velocity vector ``u``.

State layout:
    q  (nq = 9) : [x, y, z, p0, p1, p2, p3, alpha_p, beta_p]
    u  (nu = 8) : [vx, vy, vz, wx, wy, wz, alpha_p_dot, beta_p_dot]

The map q_dot = B(q) * u is built from the standard quaternion
kinematic matrix; the trick avoids gimbal lock and (because omega
itself is primary) makes the omega Jacobian a constant block-identity
matrix, sidestepping the normalization issues that plagued earlier
attempts.
"""
import numpy as np
import os
from datetime import datetime
import shutil
import json


# ═══════════════════════════════════════════════════════════════
#  Custom exceptions
# ═══════════════════════════════════════════════════════════════
class MaxNewtonIterAttainedError(Exception):
    def __init__(self, message="Maximum Newton iterations attained."):
        self.message = message
        super().__init__(self.message)

class NoOpenContactError(Exception):
    def __init__(self, message="Contact is not open."):
        self.message = message
        super().__init__(self.message)

class RhoInfInfiniteLoop(Exception):
    def __init__(self, message="Infinite loop through rho_inf update."):
        self.message = message
        super().__init__(self.message)

class MaxHoursAttained(Exception):
    def __init__(self, message="Maximum run time exceeded."):
        self.message = message
        super().__init__(self.message)

class JacobianBlowingUpError(Exception):
    def __init__(self, message="Jacobian is blowing up."):
        self.message = message
        super().__init__(self.message)


# ═══════════════════════════════════════════════════════════════
#  Quaternion utilities
# ═══════════════════════════════════════════════════════════════
def quat_normalize(p):
    n = np.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
    return p if n == 0 else p / n


def quat_to_R(p):
    """Rotation matrix from unit quaternion p = (p0, p1, p2, p3).
    """
    # Note: do NOT normalize inside — complex-step needs linearity in p.
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    return np.array([
        [1 - 2*(p2*p2 + p3*p3),  2*(p1*p2 - p0*p3),      2*(p1*p3 + p0*p2)],
        [2*(p1*p2 + p0*p3),      1 - 2*(p1*p1 + p3*p3),  2*(p2*p3 - p0*p1)],
        [2*(p1*p3 - p0*p2),      2*(p2*p3 + p0*p1),      1 - 2*(p1*p1 + p2*p2)],
    ])

# helpers used by the visualization notebook
get_rotation_matrix_from_quaternion = quat_to_R


def quaternion_to_euler(p):
    """Convert unit quaternion p = (p0, p1, p2, p3) to 3-2-1 Euler angles.

    Returns np.array([psi, theta, phi]) in radians, where
    R = Rz(psi) @ Ry(theta) @ Rx(phi).
    Used by the visualization notebook to extract theta for the rim marker.
    """
    R = quat_to_R(p)
    theta = np.arcsin(np.clip(-float(R[2, 0]), -1.0, 1.0)) # clipping to -1 and 1 since 1.01 is outside the domain
    phi   = np.arctan2(float(R[2, 1]), float(R[2, 2]))
    psi   = np.arctan2(float(R[1, 0]), float(R[0, 0]))
    return np.array([psi, theta, phi])


def B_rot(p):
    """Quaternion kinematic map: q_dot_rot = 0.5 * B_rot(p) * omega_body  (4x3)."""
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]  # from bouncing ball hulahoop edit code (eq 121)
    return 0.5 * np.array([
        [-p1, -p2, -p3],
        [ p0, -p3,  p2],
        [ p3,  p0, -p1],
        [-p2,  p1,  p0],
    ])


def B_transform(q):
    """Full kinematic map B(q): q_dot (9) = B(q) * u (8).

    Block-diagonal: I_3 for translation, B_rot(p) for the quaternion, I_2 for
    the pendulum angles.
    adapted from the bouncing ball hulahoop edit code
    """
    Bm = np.zeros((9, 8), dtype=q.dtype)
    Bm[0, 0] = 1.0
    Bm[1, 1] = 1.0
    Bm[2, 2] = 1.0
    Bm[3:7, 3:6] = B_rot(q[3:7])
    Bm[7, 6] = 1.0
    Bm[8, 7] = 1.0
    return Bm


# ═══════════════════════════════════════════════════════════════
#  Pendulum kinematics
# ═══════════════════════════════════════════════════════════════
def pendulum_dir_body(alpha_p, beta_p):
    """Unit direction (body frame) from sphere centre to pendulum mass."""
    ca, sa = np.cos(alpha_p), np.sin(alpha_p)
    cb, sb = np.cos(beta_p), np.sin(beta_p)
    return np.array([ca*cb, ca*sb, sa])


def _position_pendulum(q, l_pend):
    """Position of pendulum mass D in the world frame."""
    R = quat_to_R(q[3:7])
    d = l_pend * pendulum_dir_body(q[7], q[8])
    return q[0:3] + R @ d


def _velocity_jacobian_pendulum(q, l_pend):
    """J_D (3x8) such that v_D = J_D @ u, in the world frame.

    v_D = v_A + R(p) (omega_b x d) + R(p) (l de_body/dalpha * alpha_dot
                                           + l de_body/dbeta  * beta_dot)
        = v_A - R(p) [d]_x omega_b + R(p) (...)
    where d = l * e_body(alpha, beta).

    Works with complex dtype.
    """
    R = quat_to_R(q[3:7])
    alpha_p, beta_p = q[7], q[8]
    ca, sa = np.cos(alpha_p), np.sin(alpha_p)
    cb, sb = np.cos(beta_p),  np.sin(beta_p)
    d = l_pend * np.array([ca*cb, ca*sb, sa])
    dd_da = l_pend * np.array([-sa*cb, -sa*sb, ca])
    dd_db = l_pend * np.array([-ca*sb,  ca*cb, 0.0])

    # skew-symmetric matrix of d
    sk = np.array([
        [0.0,   -d[2],  d[1]],
        [d[2],   0.0,  -d[0]],
        [-d[1],  d[0],  0.0],
    ])

    J = np.zeros((3, 8), dtype=q.dtype)
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0
    # omega_b columns: v_D contribution = R (omega_b x d) = -R [d]_x omega_b
    J[:, 3:6] = -R @ sk
    # pendulum angle columns
    J[:, 6] = R @ dd_da
    J[:, 7] = R @ dd_db
    return J


# ═══════════════════════════════════════════════════════════════
#  Mass matrix + Coriolis + gravity (velocity space, nu = 8)
# ═══════════════════════════════════════════════════════════════
def compute_mass_matrix(q, m_h, m_p, I_s, l_pend):
    """8x8 mass matrix.
    """
    M = np.zeros((8, 8), dtype=q.dtype)
    M[0, 0] = m_h
    M[1, 1] = m_h
    M[2, 2] = m_h
    M[3, 3] = I_s
    M[4, 4] = I_s
    M[5, 5] = I_s
    J_D = _velocity_jacobian_pendulum(q, l_pend)
    M += m_p * (J_D.T @ J_D)
    return 0.5 * (M + M.T)


def compute_h(q, u, m_h, m_p, I_s, l_pend, gr):
    """Coriolis + gravity in velocity space, h = m_p J_D^T (J_D_dot u) - Q_grav.

    Note for an isotropic sphere there is no  omega_b x (I omega_b) term
    (it vanishes identically). All Coriolis comes from the pendulum.
    """
    J_D = _velocity_jacobian_pendulum(q, l_pend)

    # J_D_dot @ u via complex step on q (q_dot = B(q) u)
    qdot = B_transform(q) @ u
    J_D_dot_u = np.zeros(3)
    hcs = 1e-30
    for k in range(9):
        if qdot[k] == 0.0:
            continue
        qc = q.astype(complex)
        qc[k] += 1j * hcs
        # dJ_D/dq_k @ u
        dJk_u = np.imag(_velocity_jacobian_pendulum(qc, l_pend) @ u) / hcs
        J_D_dot_u += dJk_u * qdot[k]

    h_C = m_p * (J_D.T @ J_D_dot_u)

    # Gravity (world frame, -z direction)
    F_h = np.array([0.0, 0.0, -m_h * gr])
    F_p = np.array([0.0, 0.0, -m_p * gr])
    # Translational direct contribution from sphere mass on (vx, vy, vz)
    Q_grav = np.zeros(8)
    Q_grav[0] += F_h[0]; Q_grav[1] += F_h[1]; Q_grav[2] += F_h[2]
    Q_grav += J_D.T @ F_p

    return h_C - Q_grav


# ═══════════════════════════════════════════════════════════════
#  Proximal maps
# ═══════════════════════════════════════════════════════════════
def prox_R_plus(x):
    return max(0.0, float(x))


def prox_coulomb_2d(x, radius):
    radius = max(0.0, float(radius))
    norm_x = np.linalg.norm(x)
    if norm_x <= radius:
        return x.copy()
    if norm_x > 0.0:
        return radius * x / norm_x
    return np.zeros_like(x)


# ═══════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════
def make_jumping_alpha_profile(
    alpha0,
    t_nd,
    roll_end_s=5.0,
    swing_end_s=5.2,
    jump_direction=1.0,
    jump_same_side=True,
    jump_speed_rad_s=40.0,
    jump_burst_s=0.25,
):
    """Paper-inspired rolling/swing-up/jump command for alpha_p."""
    sign = float(jump_direction)
    t1 = float(roll_end_s)
    t2 = float(swing_end_s)
    t3 = t2 + float(jump_burst_s)
    jump_speed = -abs(float(jump_speed_rad_s)) if jump_same_side else abs(float(jump_speed_rad_s))
    alpha_t1 = alpha0 + sign * (-t1**2)
    alpha_t2 = alpha_t1 + sign * (-40.0 * (t2 - t1))
    alpha_t3 = alpha_t2 + sign * (jump_speed * (t3 - t2))

    def alpha_phys_time(t_phys):
        if t_phys < t1:
            return alpha0 + sign * (-t_phys**2)
        if t_phys < t2:
            return alpha_t1 + sign * (-40.0 * (t_phys - t1))
        if t_phys < t3:
            return alpha_t2 + sign * (jump_speed * (t_phys - t2))
        return alpha_t3

    def alpha_dot_phys_time(t_phys):
        if t_phys < t1:
            return sign * (-2.0 * t_phys)
        if t_phys < t2:
            return sign * (-40.0)
        if t_phys < t3:
            return sign * jump_speed
        return 0.0

    def alpha_ddot_phys_time(t_phys):
        if t_phys < t1:
            return sign * (-2.0)
        return 0.0

    def alpha_func(t):
        return alpha_phys_time(t * t_nd)

    def alpha_dot_func(t):
        return alpha_dot_phys_time(t * t_nd) * t_nd

    def alpha_ddot_func(t):
        return alpha_ddot_phys_time(t * t_nd) * t_nd**2

    return alpha_func, alpha_dot_func, alpha_ddot_func


class StairAwareJumpController:
    """Distance-triggered pendulum motor controller for stair jumping.

    This controller does not prescribe alpha/beta trajectories. It reads the
    next riser location from the simulation stair geometry and returns motor
    torques directly. The jump strategy is still the same three-part idea:
    roll, swing up near the obstacle, drive through the jump, then settle
    before re-arming for the next riser.
    """
    def __init__(
        self,
        beta0=0.0,
        travel_direction=1.0,
        lookahead=0.65,
        profile_trigger_distance=1.0,
        repeat_profile_trigger_distance=1.0,
        swing_distance=0.28,
        jump_distance=0.08,
        settle_distance=0.25,
        settle_tau=0.0,
        hold_time=0.0,
        final_hold=True,
        alpha_hold=None,
        hold_alpha_kp=0.1,
        hold_alpha_kd=0.05,
        settle_velocity_tilt_gain=1.5,
        settle_max_tilt=0.7,
        settle_alpha_dot_per_vx=10.0,
        settle_alpha_dot_max=3.0,
        settle_vx_tau_gain=4.0,
        settle_speed_tol=0.03,
        settle_alpha_tol=0.08,
        settle_alpha_dot_tol=0.08,
        landing_height_tol=0.015,
        roll_end_s=2.0,
        swing_end_s=2.2,
        jump_direction=-1.0,
        jump_same_side=True,
        jump_speed_rad_s=60.0,
        jump_burst_s=0.25,
        k_alpha_vel=0.08,
        tau_alpha_max=2.0,
        tau_beta_max=1.0,
        beta_kp=0.5,
        beta_kd=0.08,
    ):
        self.beta0 = float(beta0)
        self.travel_direction = 1.0 if travel_direction >= 0.0 else -1.0
        self.lookahead = float(lookahead)
        self.profile_trigger_distance = float(profile_trigger_distance)
        self.repeat_profile_trigger_distance = float(repeat_profile_trigger_distance)
        self.swing_distance = float(swing_distance)
        self.jump_distance = float(jump_distance)
        self.settle_distance = float(settle_distance)
        self.settle_tau = float(settle_tau)
        self.hold_time = float(hold_time)
        self.final_hold = bool(final_hold)
        self.alpha_hold = None if alpha_hold is None else float(alpha_hold)
        self.hold_alpha_kp = float(hold_alpha_kp)
        self.hold_alpha_kd = float(hold_alpha_kd)
        self.settle_velocity_tilt_gain = float(settle_velocity_tilt_gain)
        self.settle_max_tilt = float(settle_max_tilt)
        self.settle_alpha_dot_per_vx = float(settle_alpha_dot_per_vx)
        self.settle_alpha_dot_max = float(settle_alpha_dot_max)
        self.settle_vx_tau_gain = float(settle_vx_tau_gain)
        self.settle_speed_tol = float(settle_speed_tol)
        self.settle_alpha_tol = float(settle_alpha_tol)
        self.settle_alpha_dot_tol = float(settle_alpha_dot_tol)
        self.landing_height_tol = float(landing_height_tol)
        self.roll_end_s = float(roll_end_s)
        self.swing_end_s = float(swing_end_s)
        self.jump_direction = float(jump_direction)
        self.jump_same_side = bool(jump_same_side)
        self.jump_speed_rad_s = float(jump_speed_rad_s)
        self.jump_burst_s = float(jump_burst_s)
        self.k_alpha_vel = float(k_alpha_vel)
        self.tau_alpha_max = float(tau_alpha_max)
        self.tau_beta_max = float(tau_beta_max)
        self.beta_kp = float(beta_kp)
        self.beta_kd = float(beta_kd)
        self.sim = None
        self.cleared_risers = set()
        self.hold_until = -np.inf
        self.hold_z_target = None
        self.profile_start_t = None
        self.alpha_dot_ref_func = None

    def bind(self, sim):
        self.sim = sim
        if self.alpha_hold is None:
            self.alpha_hold = float(sim.q_save[7, 0])
        self.t_nd = sim.t_nd
        self.profile_duration = (self.swing_end_s + self.jump_burst_s) / self.t_nd
        return self

    @staticmethod
    def _saturate(value, limit):
        return float(np.clip(value, -limit, limit))

    def _riser_key(self, x_wall, z_top):
        return (round(float(x_wall), 9), round(float(z_top), 9))

    def _next_riser(self, q):
        best = self._next_uncleared_riser_ahead(q)
        if best is None:
            return None
        best_distance = best[3]
        if best is None or best_distance > self.lookahead:
            return None
        return best

    def _next_uncleared_riser_ahead(self, q):
        if self.sim is None or self.sim.n_stairs <= 0:
            return None

        x = float(q[0])
        z = float(q[2])
        best = None
        best_distance = 1e10
        for x1, z1, x2, z2 in self.sim.vertical_segments:
            x_wall = float(x1)
            z_top = max(float(z1), float(z2))
            z_bot = min(float(z1), float(z2))
            if self._riser_key(x_wall, z_top) in self.cleared_risers:
                continue
            signed_distance = self.travel_direction * (x_wall - x)
            if signed_distance < 0.0:
                continue
            if z + self.sim.R < z_bot:
                continue
            if signed_distance < best_distance:
                best_distance = signed_distance
                best = (x_wall, z_bot, z_top, signed_distance)
        return best

    def _has_uncleared_riser_ahead(self, q):
        if self.sim is None or self.sim.n_stairs <= 0:
            return False

        x = float(q[0])
        for x1, z1, x2, z2 in self.sim.vertical_segments:
            x_wall = float(x1)
            z_top = max(float(z1), float(z2))
            if self._riser_key(x_wall, z_top) in self.cleared_risers:
                continue
            if self.travel_direction * (x_wall - x) >= 0.0:
                return True
        return False

    def _recently_cleared_riser(self, q):
        if self.sim is None or self.sim.n_stairs <= 0:
            return False

        x = float(q[0])
        z = float(q[2])
        for x1, z1, x2, z2 in self.sim.vertical_segments:
            x_wall = float(x1)
            z_top = max(float(z1), float(z2))
            if self._riser_key(x_wall, z_top) in self.cleared_risers:
                continue
            signed_distance = self.travel_direction * (x_wall - x)
            clearance = -signed_distance
            if 0.0 <= clearance <= self.settle_distance and z + self.sim.R >= z_top:
                return True
        return False

    def observe(self, t, q, u):
        if self.sim is None or self.sim.n_stairs <= 0:
            return

        x = float(q[0])
        z = float(q[2])
        for x1, z1, x2, z2 in self.sim.vertical_segments:
            x_wall = float(x1)
            z_top = max(float(z1), float(z2))
            key = self._riser_key(x_wall, z_top)
            if key in self.cleared_risers:
                continue

            clearance = self.travel_direction * (x - x_wall)
            landed_height = z_top + self.sim.R
            if clearance >= 0.0 and abs(z - landed_height) <= self.landing_height_tol:
                self.cleared_risers.add(key)
                self.hold_until = max(self.hold_until, float(t) + self.hold_time)
                self.hold_z_target = landed_height
                if self._has_uncleared_riser_ahead(q):
                    self.profile_start_t = None
                    self.alpha_dot_ref_func = None

    @staticmethod
    def _angle_error(angle, target):
        return np.arctan2(np.sin(angle - target), np.cos(angle - target))

    def _settled_after_landing(self, q, u):
        if not self.cleared_risers:
            return False
        alpha_err = self._angle_error(float(q[7]), self.alpha_hold)
        speed_norm = np.linalg.norm([float(u[0]), float(u[1]), float(u[2])])
        return (
            speed_norm <= self.settle_speed_tol and
            abs(alpha_err) <= self.settle_alpha_tol and
            abs(float(u[6])) <= self.settle_alpha_dot_tol
        )

    def _is_holding(self, t, q, u):
        if self.cleared_risers and self._has_uncleared_riser_ahead(q):
            return False
        if float(t) <= self.hold_until:
            return True
        if self.cleared_risers and not self._settled_after_landing(q, u):
            return True
        if self.final_hold and self.cleared_risers and not self._has_uncleared_riser_ahead(q):
            return True
        return False

    def _hold_alpha_tau(self, q, u):
        forward_speed = self.travel_direction * float(u[0])
        alpha_err = self._angle_error(float(q[7]), self.alpha_hold)
        tau = -self.travel_direction * self.settle_vx_tau_gain * forward_speed
        tau += -self.hold_alpha_kp * alpha_err
        tau += -self.hold_alpha_kd * float(u[6])
        return tau

    def _start_profile(self, t, q):
        roll_end_s = self.roll_end_s
        swing_end_s = self.swing_end_s
        _, alpha_dot_ref_func, _ = make_jumping_alpha_profile(
            float(q[7]),
            self.t_nd,
            roll_end_s=roll_end_s,
            swing_end_s=swing_end_s,
            jump_direction=self.jump_direction,
            jump_same_side=self.jump_same_side,
            jump_speed_rad_s=self.jump_speed_rad_s,
            jump_burst_s=self.jump_burst_s,
        )
        self.profile_start_t = float(t)
        self.alpha_dot_ref_func = alpha_dot_ref_func
        self.profile_duration = (swing_end_s + self.jump_burst_s) / self.t_nd

    def _profile_alpha_tau(self, t, q, u):
        if self.alpha_dot_ref_func is None:
            self._start_profile(t, q)
        tau_t = max(0.0, float(t) - self.profile_start_t)
        if self._has_uncleared_riser_ahead(q) and tau_t > self.profile_duration:
            self._start_profile(t, q)
            tau_t = 0.0
        alpha_dot_ref = self.alpha_dot_ref_func(tau_t)
        return self.k_alpha_vel * (alpha_dot_ref - float(u[6]))

    def _cruise_alpha_tau(self, u):
        alpha_dot_ref = self.jump_direction * (-2.0 * self.roll_end_s) * self.t_nd
        return self.k_alpha_vel * (alpha_dot_ref - float(u[6]))

    def alpha_tau(self, t, q, u):
        riser = self._next_riser(q)

        if self.cleared_risers and self._has_uncleared_riser_ahead(q):
            next_riser = self._next_uncleared_riser_ahead(q)
            if next_riser is not None and next_riser[3] > self.repeat_profile_trigger_distance:
                self.profile_start_t = None
                self.alpha_dot_ref_func = None
                tau = self._cruise_alpha_tau(u)
            else:
                tau = self._profile_alpha_tau(t, q, u)
            return self._saturate(tau, self.tau_alpha_max)

        if self.cleared_risers and self.alpha_dot_ref_func is not None and \
                not self._settled_after_landing(q, u):
            tau = self._profile_alpha_tau(t, q, u)
            return self._saturate(tau, self.tau_alpha_max)

        if self._settled_after_landing(q, u):
            self.profile_start_t = None
            self.alpha_dot_ref_func = None

        if self._is_holding(t, q, u) and not self._settled_after_landing(q, u):
            tau = self._hold_alpha_tau(q, u)
            return self._saturate(tau, self.tau_alpha_max)
        elif riser is None:
            if self._recently_cleared_riser(q):
                tau = self.settle_tau * self.travel_direction
            elif self._has_uncleared_riser_ahead(q):
                tau = self._profile_alpha_tau(t, q, u)
            else:
                self.profile_start_t = None
                self.alpha_dot_ref_func = None
                tau = 0.0
        else:
            tau = self._profile_alpha_tau(t, q, u)
        return self._saturate(tau, self.tau_alpha_max)

    def beta_tau(self, t, q, u):
        tau = -self.beta_kp * (float(q[8]) - self.beta0) - self.beta_kd * float(u[7])
        return self._saturate(tau, self.tau_beta_max)


class Simulation:
    def __init__(
        self,
        ntime=5,
        mu_s=0.3, mu_k=0.3,
        eN=0.0, eF=0.0,
        R=0.1,
        m_sphere=1.0,
        m_pendulum=0.1,
        l_pendulum=0.05,
        # initial position
        x0=0.0, y0=0.0, z0=None,
        # initial quaternion (default = identity)
        p0_0=1.0, p1_0=0.0, p2_0=0.0, p3_0=0.0,
        alpha_p0=0.0, beta_p0=0.0,
        # initial velocities (note: omega is BODY-frame)
        x_dot0=0.0, y_dot0=0.0, z_dot0=0.0,
        wx0=0.0, wy0=0.0, wz0=0.0,
        alpha_p_dot0=0.0, beta_p_dot0=0.0,
        # stairs (in x; z is vertical; infinite in y)
        n_stairs=0, stair_width=0.3, stair_height=0.05,
        stair_x_start=0.5, fillet_radius=0,
        # prescribed pendulum
        alpha_p_func=None, alpha_p_dot_func=None, alpha_p_ddot_func=None,
        beta_p_func=None, beta_p_dot_func=None, beta_p_ddot_func=None,
        # pendulum motor torques
        alpha_tau_func=None, beta_tau_func=None,
        torque_controller=None,
    ):
        # ── output directory ──────────────────────────────────
        timestamp = datetime.now().strftime("sphere3dquat_%Y-%m-%d_%H-%M-%S")
        outputs_dir = f"outputs/{timestamp}"
        self.output_path = os.path.join(os.getcwd(), outputs_dir)
        os.makedirs(self.output_path, exist_ok=True)
        try:
            shutil.copy2(os.path.realpath(__file__), self.output_path)
        except Exception:
            pass

        # ── friction / restitution ────────────────────────────
        self.mu_s = float(mu_s)
        self.mu_k = float(mu_k)
        self.eN = float(eN)
        self.eF = float(eF)

        # ── nondimensionalisation ─────────────────────────────
        l_nd = 1.0
        m_nd = 1.0
        a_nd = 9.81
        t_nd = np.sqrt(l_nd / a_nd)
        self.t_nd = t_nd

        self.dtime = 2e-3 / t_nd
        self.dtime_initial = self.dtime
        self.ntime = int(ntime)
        self.tf = self.ntime * self.dtime
        self.t = np.linspace(0, self.tf, self.ntime)

        # ── physical parameters ───────────────────────────────
        self.R = float(R) / l_nd
        self.l_pendulum = float(l_pendulum) / l_nd
        self.m_sphere = float(m_sphere) / m_nd
        self.m_pendulum = float(m_pendulum) / m_nd
        self.m = self.m_sphere + self.m_pendulum
        self.gr = 9.81 / a_nd

        # Isotropic sphere shell inertia (scalar form)
        self.I_s = (2.0 / 3.0) * self.m_sphere * self.R**2

        # ── stairs ────────────────────────────────────────────
        self.n_stairs = int(n_stairs)
        self.stair_width = float(stair_width) / l_nd
        self.stair_height = float(stair_height) / l_nd
        self.stair_x_start = float(stair_x_start) / l_nd
        self.fillet_radius = float(fillet_radius) / l_nd
        self._build_stair_profile()
        self._classify_segments()

        self.torque_controller = torque_controller
        if torque_controller is not None and hasattr(torque_controller, "bind"):
            torque_controller.bind(self)
        if torque_controller is not None:
            alpha_tau_func = getattr(torque_controller, "alpha_tau", alpha_tau_func)
            beta_tau_func = getattr(torque_controller, "beta_tau", beta_tau_func)

        # ── prescribed pendulum ───────────────────────────────
        self.prescribed_alpha = callable(alpha_p_func)
        self.prescribed_beta = callable(beta_p_func)

        if self.prescribed_alpha:
            self.alpha_p_func = alpha_p_func
            self.alpha_p_dot_func = alpha_p_dot_func if callable(alpha_p_dot_func) else (lambda t: 0.0)
            self.alpha_p_ddot_func = alpha_p_ddot_func if callable(alpha_p_ddot_func) else (lambda t: 0.0)
            alpha_p0 = self.alpha_p_func(0.0)
            alpha_p_dot0 = self.alpha_p_dot_func(0.0)
        else:
            self.alpha_p_func = lambda t: alpha_p0
            self.alpha_p_dot_func = lambda t: 0.0
            self.alpha_p_ddot_func = lambda t: 0.0

        if self.prescribed_beta:
            self.beta_p_func = beta_p_func
            self.beta_p_dot_func = beta_p_dot_func if callable(beta_p_dot_func) else (lambda t: 0.0)
            self.beta_p_ddot_func = beta_p_ddot_func if callable(beta_p_ddot_func) else (lambda t: 0.0)
            beta_p0 = self.beta_p_func(0.0)
            beta_p_dot0 = self.beta_p_dot_func(0.0)
        else:
            self.beta_p_func = lambda t: beta_p0
            self.beta_p_dot_func = lambda t: 0.0
            self.beta_p_ddot_func = lambda t: 0.0

        self.alpha_tau_func = alpha_tau_func if callable(alpha_tau_func) else (lambda t, q, u: 0.0)
        self.beta_tau_func = beta_tau_func if callable(beta_tau_func) else (lambda t, q, u: 0.0)

        # ── DOF counts ────────────────────────────────────────
        self.nq = 9  # x,y,z, p0,p1,p2,p3, alpha_p, beta_p
        self.nu = 8  # vx,vy,vz, wx,wy,wz, alpha_p_dot, beta_p_dot
        self.ng = (1 if self.prescribed_alpha else 0) + (1 if self.prescribed_beta else 0)
        self.ngamma = 0

        # contact: tread + riser, each with 2 friction directions
        self.nN = 2
        self.nF = 4
        self.gammaF_lim = np.array([[0, 1], [2, 3]])

        self.nX = 3*self.nu + 3*self.ng + 2*self.ngamma + 3*self.nN + 2*self.nF

        # ── generalized-alpha ─────────────────────────────────
        self.MAXITERn = 300
        self.r = 0.3
        self.rho_inf = 0.5
        self.alpha_m = (2*self.rho_inf - 1) / (self.rho_inf + 1)
        self.alpha_f = self.rho_inf / (self.rho_inf + 1)
        self.gama = 0.5 + self.alpha_f - self.alpha_m
        self.beta = 0.25 * (0.5 + self.gama)**2
        self.tol_n = 1.0e-4

        # ── initial conditions ────────────────────────────────
        if z0 is None:
            z0 = float(R)
        p_init = np.array([p0_0, p1_0, p2_0, p3_0])
        p_init = quat_normalize(p_init)
        q0 = np.array([float(x0), float(y0), float(z0),
                       p_init[0], p_init[1], p_init[2], p_init[3],
                       float(alpha_p0), float(beta_p0)])
        u0 = np.array([float(x_dot0), float(y_dot0), float(z_dot0),
                       float(wx0), float(wy0), float(wz0),
                       float(alpha_p_dot0), float(beta_p_dot0)])

        # ── storage ───────────────────────────────────────────
        self.q_save = np.zeros((self.nq, self.ntime))
        self.u_save = np.zeros((self.nu, self.ntime))
        self.X_save = np.zeros((self.nX, self.ntime))
        self.gNdot_save = np.zeros((self.nN, self.ntime))
        self.gNddot_save = np.zeros((self.nN, self.ntime))
        self.gammaF_save = np.zeros((self.nF, self.ntime))
        self.lambdaN_save = np.zeros((self.nN, self.ntime))
        self.lambdaF_save = np.zeros((self.nF, self.ntime))
        self.tau_pendulum_save = np.zeros((2, self.ntime))
        self.AV_save = np.zeros((self.nu + self.nN + self.nF, self.ntime))
        self.contacts_save = np.zeros((5*self.nN, self.ntime))
        self.rho_inf_save = np.full(self.ntime, self.rho_inf)
        self.dtime_save = np.full(self.ntime, self.dtime)
        self.alpha_save = np.ones(self.ntime)

        self.q_save[:, 0] = q0
        self.u_save[:, 0] = u0

        self.f = open(f"{self.output_path}/log_file.txt", 'a')
        self._save_metadata()

    # ── metadata ──────────────────────────────────────────────
    def _save_metadata(self):
        params = {
            "ndof_q": self.nq, "ndof_u": self.nu,
            "dtime": float(self.dtime), "ntime": self.ntime,
            "R": float(self.R),
            "m_sphere": float(self.m_sphere),
            "m_pendulum": float(self.m_pendulum),
            "l_pendulum": float(self.l_pendulum),
            "mu_s": float(self.mu_s), "mu_k": float(self.mu_k),
            "eN": float(self.eN), "eF": float(self.eF),
            "q0": self.q_save[:, 0].tolist(),
            "u0": self.u_save[:, 0].tolist(),
            "rho_inf": float(self.rho_inf),
            "n_stairs": self.n_stairs,
            "stair_width": float(self.stair_width),
            "stair_height": float(self.stair_height),
            "stair_x_start": float(self.stair_x_start),
            "fillet_radius": float(self.fillet_radius),
            "torque_controller": type(self.torque_controller).__name__
            if self.torque_controller is not None else None,
        }
        with open(os.path.join(self.output_path, "params.json"), "w") as fp:
            json.dump(params, fp, indent=2)

    def save_arrays(self):
        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gNddot_save.npy', self.gNddot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/lambdaN_save.npy', self.lambdaN_save)
        np.save(f'{self.output_path}/lambdaF_save.npy', self.lambdaF_save)
        np.save(f'{self.output_path}/tau_pendulum_save.npy', self.tau_pendulum_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        np.save(f'{self.output_path}/contacts_save.npy', self.contacts_save)
        np.save(f'{self.output_path}/rho_inf_save.npy', self.rho_inf_save)
        np.save(f'{self.output_path}/dtime_save.npy', self.dtime_save)
        np.save(f'{self.output_path}/alpha_save.npy', self.alpha_save)

    # ══════════════════════════════════════════════════════════
    #  Stair geometry (copied verbatim from Euler version)
    # ══════════════════════════════════════════════════════════
    def _build_stair_profile(self):
        FAR = 1e6
        segs = []
        r_f = self.fillet_radius
        self.fillet_arcs = []
        if self.n_stairs == 0:
            segs.append((-FAR, 0.0, FAR, 0.0))
        else:
            segs.append((-FAR, 0.0, self.stair_x_start, 0.0))
            for i in range(self.n_stairs):
                x_left = self.stair_x_start + i * self.stair_width
                h_prev = i * self.stair_height
                h_curr = (i + 1) * self.stair_height
                x_right = x_left + self.stair_width
                if r_f > 0:
                    segs.append((x_left, h_prev, x_left, h_curr - r_f))
                    self.fillet_arcs.append((x_left + r_f, h_curr - r_f, r_f))
                    if i < self.n_stairs - 1:
                        segs.append((x_left + r_f, h_curr, x_right, h_curr))
                    else:
                        segs.append((x_left + r_f, h_curr, FAR, h_curr))
                else:
                    segs.append((x_left, h_prev, x_left, h_curr))
                    if i < self.n_stairs - 1:
                        segs.append((x_left, h_curr, x_right, h_curr))
                    else:
                        segs.append((x_left, h_curr, FAR, h_curr))
        self.stair_segments = segs

    def _classify_segments(self):
        self.horizontal_segments = []
        self.vertical_segments = []
        for seg in self.stair_segments:
            x1, z1, x2, z2 = seg
            dx = abs(x2 - x1)
            dz = abs(z2 - z1)
            if dz < 1e-12:
                self.horizontal_segments.append(seg)
            elif dx < 1e-12:
                self.vertical_segments.append(seg)

    def _compute_tread_gap(self, px, pz):
        best_gap = 1e10
        best_nx, best_nz = 0.0, 1.0
        for seg in self.horizontal_segments:
            x1, z1, x2, z2 = seg
            z_tread = z1
            x_left = min(x1, x2)
            x_right = max(x1, x2)
            if px <= x_left or px >= x_right:
                continue
            dz = pz - z_tread
            gap = abs(dz) - self.R
            if gap < best_gap:
                best_gap = gap
                best_nx = 0.0
                best_nz = 1.0 if dz >= 0 else -1.0
        return best_gap, best_nx, best_nz

    def _compute_riser_gap(self, px, pz):
        best_gap = 1e10
        best_nx, best_nz = 1.0, 0.0
        for seg in self.vertical_segments:
            x1, z1, x2, z2 = seg
            x_wall = x1
            z_bot = min(z1, z2)
            z_top = max(z1, z2)
            cx = x_wall
            cz = max(z_bot, min(z_top, pz))
            dx = px - cx
            dz = pz - cz
            dist = np.sqrt(dx * dx + dz * dz)
            gap = dist - self.R
            if gap < best_gap:
                best_gap = gap
                if dist < 1e-15:
                    best_nx = 1.0 if px >= cx else -1.0
                    best_nz = 0.0
                else:
                    best_nx = dx / dist
                    best_nz = dz / dist
        for cx_a, cz_a, r_f in self.fillet_arcs:
            dx = px - cx_a
            dz = pz - cz_a
            if dx > 0 or dz < 0:
                continue
            dist = np.sqrt(dx * dx + dz * dz)
            gap = dist - r_f - self.R
            if gap < best_gap:
                best_gap = gap
                if dist < 1e-15:
                    best_nx = -1.0
                    best_nz = 0.0
                else:
                    best_nx = dx / dist
                    best_nz = dz / dist
        return best_gap, best_nx, best_nz

    # ══════════════════════════════════════════════════════════
    #  Contact model (in velocity space; omega is body-frame)
    # ══════════════════════════════════════════════════════════
    def _compute_contact(self, q, u, a):
        Rmat = quat_to_R(quat_normalize(q[3:7]))
        omega_b = u[3:6]
        omega_w = Rmat @ omega_b
        # alpha_w = R alpha_b   (since omega_w = R omega_b, time derivative
        # gives alpha_w = R alpha_b + R [omega_b]_x omega_b, and the cross term
        # is zero)
        alpha_b = a[3:6]
        alpha_w = Rmat @ alpha_b
        v_A = u[0:3]
        a_A = a[0:3]

        # Tread (contact 0) and riser (contact 1)
        gap0, nx0, nz0 = self._compute_tread_gap(q[0], q[2])
        n0 = np.array([nx0, 0.0, nz0])
        gap1, nx1, nz1 = self._compute_riser_gap(q[0], q[2])
        n1 = np.array([nx1, 0.0, nz1])

        gN = np.array([gap0, gap1])
        gNdot = np.array([n0 @ v_A, n1 @ v_A])
        gNddot = np.array([n0 @ a_A, n1 @ a_A])

        WN = np.zeros((8, 2))
        WN[0, 0] = n0[0]; WN[1, 0] = n0[1]; WN[2, 0] = n0[2]
        WN[0, 1] = n1[0]; WN[1, 1] = n1[1]; WN[2, 1] = n1[2]
        # Normal forces produce no torque (line of action through centre)

        gammaF = np.zeros(4)
        gammaFdot = np.zeros(4)
        WF = np.zeros((8, 4))

        for k, (n_k, fric_start) in enumerate([(n0, 0), (n1, 2)]):
            t1 = np.array([0.0, 1.0, 0.0])
            t2 = np.array([n_k[2], 0.0, -n_k[0]])

            r_contact = -self.R * n_k  # vector from sphere centre to contact
            v_contact = v_A + np.cross(omega_w, r_contact)
            a_contact = a_A + np.cross(alpha_w, r_contact) + \
                        np.cross(omega_w, np.cross(omega_w, r_contact))

            gammaF[fric_start]     = t1 @ v_contact
            gammaF[fric_start + 1] = t2 @ v_contact
            gammaFdot[fric_start]     = t1 @ a_contact
            gammaFdot[fric_start + 1] = t2 @ a_contact

            for j, t_j in enumerate([t1, t2]):
                col_idx = fric_start + j
                F_j = t_j
                tau_w = np.cross(r_contact, F_j)  # world-frame torque
                tau_b = Rmat.T @ tau_w            # body-frame torque
                WF[0, col_idx] = F_j[0]
                WF[1, col_idx] = F_j[1]
                WF[2, col_idx] = F_j[2]
                WF[3, col_idx] = tau_b[0]
                WF[4, col_idx] = tau_b[1]
                WF[5, col_idx] = tau_b[2]

        return gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF

    # ── X decomposition ───────────────────────────────────────
    def get_X_components(self, X):
        n = self.nu
        ng = self.ng
        ngamma = self.ngamma
        nN = self.nN
        nF = self.nF
        idx = 0
        a = X[idx:idx+n];             idx += n
        U = X[idx:idx+n];             idx += n
        Q = X[idx:idx+n];             idx += n
        Kappa_g = X[idx:idx+ng];      idx += ng
        Lambda_g = X[idx:idx+ng];     idx += ng
        lambda_g = X[idx:idx+ng];     idx += ng
        Lambda_gamma = X[idx:idx+ngamma]; idx += ngamma
        lambda_gamma = X[idx:idx+ngamma]; idx += ngamma
        KappaN = X[idx:idx+nN];       idx += nN
        LambdaN = X[idx:idx+nN];      idx += nN
        lambdaN = X[idx:idx+nN];      idx += nN
        LambdaF = X[idx:idx+nF];      idx += nF
        lambdaF = X[idx:idx+nF];      idx += nF
        return (a, U, Q, Kappa_g, Lambda_g, lambda_g,
                Lambda_gamma, lambda_gamma,
                KappaN, LambdaN, lambdaN, LambdaF, lambdaF)

    # ══════════════════════════════════════════════════════════
    #  RESIDUAL
    # ══════════════════════════════════════════════════════════
    def get_R(self, iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):

        (prev_a, _, _, _, _, _, _, _, _, _, prev_lambdaN, _, prev_lambdaF) = self.get_X_components(prev_X)
        (a, U, Q, Kappa_g, Lambda_g, lambda_g,
         Lambda_gamma, lambda_gamma,
         KappaN, LambdaN, lambdaN, LambdaF, lambdaF) = self.get_X_components(X)

        # AV
        prev_abar = prev_AV[0:self.nu]
        prev_lambdaNbar = prev_AV[self.nu:self.nu+self.nN]
        prev_lambdaFbar = prev_AV[self.nu+self.nN:self.nu+self.nN+self.nF]

        abar = (self.alpha_f*prev_a + (1-self.alpha_f)*a - self.alpha_m*prev_abar) / (1-self.alpha_m)
        lambdaNbar = (self.alpha_f*prev_lambdaN + (1-self.alpha_f)*lambdaN - self.alpha_m*prev_lambdaNbar) / (1-self.alpha_m)
        lambdaFbar = (self.alpha_f*prev_lambdaF + (1-self.alpha_f)*lambdaF - self.alpha_m*prev_lambdaFbar) / (1-self.alpha_m)
        AV = np.concatenate((abar, lambdaNbar, lambdaFbar))

        # Velocity update (nu = 8)
        u = prev_u + self.dtime*((1-self.gama)*prev_abar + self.gama*abar) + U

        # Position update via B(q):  q_dot = B u
        # Use B evaluated at prev_q (cheap, stable) and add a B*Q correction.
        B_prev = B_transform(prev_q)
        qdot_prev = B_prev @ prev_u
        # Acceleration of q for the Newmark-style update:
        # q_ddot = d/dt(B u) = B u_dot + B_dot u  (B_dot u term uses qdot)
        # We mirror the hula-hoop file: take qddot ≈ B * abar_combo + correction
        abar_combo = (1 - 2*self.beta) * prev_abar + 2 * self.beta * abar
        qddot = B_prev @ abar_combo  # ignore the (small) B_dot term for the prediction
        q = prev_q + self.dtime * qdot_prev + 0.5 * self.dtime**2 * qddot + B_prev @ Q

        # Renormalize quaternion in the *current* configuration used by the
        # residual (do NOT mutate q in place because complex-step depends on
        # linearity of M(q), but here we're already inside the residual eval,
        # not the Jacobian-wrt-q path).
        p_norm = np.sqrt(q[3]*q[3] + q[4]*q[4] + q[5]*q[5] + q[6]*q[6])
        if np.abs(p_norm) > 1e-12:
            q = q.copy()
            q[3:7] = q[3:7] / p_norm

        # ── Mass matrix ───────────────────────────────────────
        M = compute_mass_matrix(q, self.m_sphere, self.m_pendulum,
                                self.I_s, self.l_pendulum)
        h = compute_h(q, u, self.m_sphere, self.m_pendulum,
                      self.I_s, self.l_pendulum, self.gr)

        Q_tau = np.zeros(self.nu)
        Q_tau[6] = self.alpha_tau_func(self.t[iter], q, u)
        Q_tau[7] = self.beta_tau_func(self.t[iter], q, u)

        # ── Bilateral constraints (prescribed pendulum) ───────
        # Position level constraint indices in q: alpha_p -> q[7], beta_p -> q[8]
        # Velocity level: corresponding u indices are 6, 7.
        g_list = []; gdot_list = []; gddot_list = []; Wg_cols = []
        if self.prescribed_alpha:
            g_list.append(q[7] - self.alpha_p_func(self.t[iter]))
            gdot_list.append(u[6] - self.alpha_p_dot_func(self.t[iter]))
            gddot_list.append(a[6] - self.alpha_p_ddot_func(self.t[iter]))
            col = np.zeros(self.nu); col[6] = 1.0
            Wg_cols.append(col)
        if self.prescribed_beta:
            g_list.append(q[8] - self.beta_p_func(self.t[iter]))
            gdot_list.append(u[7] - self.beta_p_dot_func(self.t[iter]))
            gddot_list.append(a[7] - self.beta_p_ddot_func(self.t[iter]))
            col = np.zeros(self.nu); col[7] = 1.0
            Wg_cols.append(col)

        if self.ng > 0:
            g = np.array(g_list)
            gdot = np.array(gdot_list)
            gddot = np.array(gddot_list)
            Wg = np.column_stack(Wg_cols)
        else:
            g = np.zeros(0); gdot = np.zeros(0); gddot = np.zeros(0)
            Wg = np.zeros((self.nu, 0))

        gamma = np.zeros(self.ngamma)
        gammadot = np.zeros(self.ngamma)
        Wgamma = np.zeros((self.nu, self.ngamma))

        # ── Contact ───────────────────────────────────────────
        gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF = self._compute_contact(q, u, a)

        ksiN = gNdot + self.eN * prev_gNdot
        PN = LambdaN + self.dtime*((1-self.gama)*prev_lambdaNbar + self.gama*lambdaNbar)
        Kappa_hatN = KappaN + self.dtime**2/2*((1-2*self.beta)*prev_lambdaNbar + 2*self.beta*lambdaNbar)
        ksiF = gammaF + self.eF * prev_gammaF
        PF = LambdaF + self.dtime*((1-self.gama)*prev_lambdaFbar + self.gama*lambdaFbar)

        # Smooth residual
        Rs = np.concatenate((
            M @ a + h - Q_tau - Wg @ lambda_g - Wgamma @ lambda_gamma - WN @ lambdaN - WF @ lambdaF,
            M @ U - Wg @ Lambda_g - Wgamma @ Lambda_gamma - WN @ LambdaN - WF @ LambdaF,
            M @ Q - Wg @ Kappa_g - WN @ KappaN - self.dtime/2*(Wgamma @ Lambda_gamma + WF @ LambdaF),
            g, gdot, gddot,
            gamma, gammadot,
        ))

        # Contact residual (proximal / semi-smooth)
        R_KappaN = np.zeros(self.nN)
        R_LambdaN = np.zeros(self.nN)
        R_lambdaN = np.zeros(self.nN)
        R_LambdaF = np.zeros(self.nF)
        R_lambdaF = np.zeros(self.nF)

        A = np.zeros(self.nN, dtype=int)
        B = np.zeros(self.nN, dtype=int)
        C = np.zeros(self.nN, dtype=int)
        D = np.zeros(self.nN, dtype=int)
        E = np.zeros(self.nN, dtype=int)

        for k in range(self.nN):
            fric_idx = self.gammaF_lim[k, :]
            prox_kappa = prox_R_plus(Kappa_hatN[k] - self.r * gN[k])
            R_KappaN[k] = Kappa_hatN[k] - prox_kappa

            if prox_kappa > 0:
                A[k] = 1
                prox_PN = prox_R_plus(PN[k] - self.r * ksiN[k])
                R_LambdaN[k] = PN[k] - prox_PN
                B[k] = 1 if (PN[k] - self.r * ksiN[k] >= 0) else 0

                prox_lN = prox_R_plus(lambdaN[k] - self.r * gNddot[k])
                R_lambdaN[k] = lambdaN[k] - prox_lN
                C[k] = 1 if (lambdaN[k] - self.r * gNddot[k] >= 0) else 0

                PF_aug = PF[fric_idx] - self.r * ksiF[fric_idx]
                R_LambdaF[fric_idx] = PF[fric_idx] - prox_coulomb_2d(PF_aug, self.mu_k * prox_PN)
                D[k] = 1 if (np.linalg.norm(PF_aug) <= self.mu_k * prox_PN) else 0

                lF_aug = lambdaF[fric_idx] - self.r * gammaFdot[fric_idx]
                R_lambdaF[fric_idx] = lambdaF[fric_idx] - prox_coulomb_2d(lF_aug, self.mu_k * prox_lN)
                E[k] = 1 if (np.linalg.norm(lF_aug) <= self.mu_k * prox_lN) else 0
            else:
                R_LambdaN[k] = PN[k]
                R_lambdaN[k] = lambdaN[k]
                R_LambdaF[fric_idx] = PF[fric_idx]
                R_lambdaF[fric_idx] = lambdaF[fric_idx]

        contacts_nu = np.concatenate((A, B, C, D, E))
        Rc = np.concatenate((R_KappaN, R_LambdaN, R_lambdaN, R_LambdaF, R_lambdaF))
        R_full = np.concatenate((Rs, Rc))

        return R_full, AV, q, u, gNdot, gammaF, contacts_nu

    # ══════════════════════════════════════════════════════════
    #  JACOBIAN
    # ══════════════════════════════════════════════════════════
    def get_R_J(self, iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):
        epsilon = 1e-6
        R, AV, q, u, gNdot, gammaF, contacts_nu = \
            self.get_R(iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF)

        J = np.zeros((self.nX, self.nX))
        I_mat = np.identity(self.nX)
        for i in range(self.nX):
            R_plus, _, _, _, _, _, _ = self.get_R(
                iter, X + epsilon*I_mat[:, i], prev_X, prev_AV,
                prev_q, prev_u, prev_gNdot, prev_gammaF)
            J[:, i] = (R_plus - R) / epsilon

        return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    # ══════════════════════════════════════════════════════════
    #  NEWTON
    # ══════════════════════════════════════════════════════════
    def _newton_solve(self, iter, X0, prev_X, prev_AV, prev_q, prev_u,
                      prev_gNdot, prev_gammaF):
        X = X0.copy()
        R, AV, q, u, gNdot, gammaF, J, contacts_nu = \
            self.get_R_J(iter, X, prev_X, prev_AV, prev_q, prev_u,
                         prev_gNdot, prev_gammaF)
        norm_R = np.linalg.norm(R, np.inf)
        alpha_min = 1.0
        nu = 0
        while norm_R > self.tol_n and nu < self.MAXITERn:
            delta = np.linalg.lstsq(J, R, rcond=None)[0]
            alpha = 1.0
            X_old = X.copy()
            for ls in range(8):
                X = X_old - alpha * delta
                R_trial, _, _, _, _, _, _ = self.get_R(
                    iter, X, prev_X, prev_AV, prev_q, prev_u,
                    prev_gNdot, prev_gammaF)
                norm_trial = np.linalg.norm(R_trial, np.inf)
                if norm_trial < norm_R or alpha < 1e-2:
                    break
                alpha *= 0.5
            alpha_min = min(alpha_min, alpha)
            nu += 1
            R, AV, q, u, gNdot, gammaF, J, contacts_nu = \
                self.get_R_J(iter, X, prev_X, prev_AV, prev_q, prev_u,
                             prev_gNdot, prev_gammaF)
            norm_R = np.linalg.norm(R, np.inf)
        return {
            "converged": norm_R <= self.tol_n,
            "X": X, "AV": AV, "q": q, "u": u,
            "gNdot": gNdot, "gammaF": gammaF,
            "contacts": contacts_nu,
            "alpha_min": alpha_min,
            "norm_R": norm_R,
            "nu": nu,
        }

    def update(self, iter, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):
        result = self._newton_solve(iter, prev_X, prev_X, prev_AV, prev_q, prev_u,
                                    prev_gNdot, prev_gammaF)
        if not result["converged"]:
            msg = f"Newton did not converge: ||R||={result['norm_R']:.3e}"
            print(f"iter {iter}: {msg}")
            try: self.f.write(f"\niter {iter}: {msg}")
            except Exception: pass
            raise MaxNewtonIterAttainedError(msg)

        X = result["X"]; AV = result["AV"]
        q = result["q"]; u = result["u"]
        gNdot = result["gNdot"]; gammaF = result["gammaF"]
        contacts_nu = result["contacts"]
        alpha_min = result["alpha_min"]

        # Hard renormalize the quaternion AFTER convergence (lesson from
        # earlier attempt — keeps the unit-norm constraint without
        # interfering with the Newton iteration).
        p_norm = np.sqrt(q[3]*q[3] + q[4]*q[4] + q[5]*q[5] + q[6]*q[6])
        if p_norm > 1e-12:
            q[3:7] = q[3:7] / p_norm

        self.contacts_save[:, iter] = contacts_nu
        print(f"iter {iter}: CONVERGED, ||R||={result['norm_R']:.3e}, nu={result['nu']}")
        try: self.f.write(f"\niter {iter}: CONVERGED, ||R||={result['norm_R']:.3e}")
        except Exception: pass
        return X, AV, q, u, gNdot, gammaF, alpha_min

    # ══════════════════════════════════════════════════════════
    #  TIME UPDATE
    # ══════════════════════════════════════════════════════════
    def time_update(self, iter):
        prev_X = self.X_save[:, iter-1]
        prev_AV = self.AV_save[:, iter-1]
        prev_q = self.q_save[:, iter-1]
        prev_u = self.u_save[:, iter-1]
        prev_gNdot = self.gNdot_save[:, iter-1]
        prev_gammaF = self.gammaF_save[:, iter-1]

        X, AV, q, u, gNdot, gammaF, alpha_min = self.update(
            iter, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF)

        a = X[0:self.nu]
        _, _, gNddot, _, _, _, _ = self._compute_contact(q, u, a)
        lambda_N_start = 3*self.nu + 3*self.ng + 2*self.ngamma + 2*self.nN
        lambda_N = X[lambda_N_start:lambda_N_start + self.nN]
        lambda_F_start = lambda_N_start + self.nN + self.nF
        lambda_F = X[lambda_F_start:lambda_F_start + self.nF]

        self.tau_pendulum_save[:, iter] = [
            self.alpha_tau_func(self.t[iter], q, u),
            self.beta_tau_func(self.t[iter], q, u),
        ]
        if self.torque_controller is not None and hasattr(self.torque_controller, "observe"):
            self.torque_controller.observe(self.t[iter], q, u)

        self.q_save[:, iter] = q
        self.u_save[:, iter] = u
        self.X_save[:, iter] = X
        self.gNdot_save[:, iter] = gNdot
        self.gNddot_save[:, iter] = gNddot
        self.gammaF_save[:, iter] = gammaF
        self.lambdaN_save[:, iter] = lambda_N
        self.lambdaF_save[:, iter] = lambda_F
        self.AV_save[:, iter] = AV
        self.rho_inf_save[iter] = self.rho_inf
        self.alpha_save[iter] = alpha_min
        self.save_arrays()

    # ══════════════════════════════════════════════════════════
    #  SOLVE  (adaptive dt)
    # ══════════════════════════════════════════════════════════
    def solve(self):
        iter = 1
        self.min_dtime = self.dtime_initial / 16
        min_dtime = self.min_dtime
        dtime_halve_count = 0
        while iter < self.ntime:
            self.t[iter] = self.t[iter-1] + self.dtime
            try:
                self.time_update(iter)
                self.dtime_save[iter] = self.dtime
                if self.dtime < self.dtime_initial - 1e-16:
                    print(f">>> iter {iter}: restoring dtime -> {self.dtime_initial:.6e}")
                    self.dtime = self.dtime_initial
                iter += 1
            except MaxNewtonIterAttainedError:
                if self.dtime > min_dtime + 1e-20:
                    new_dtime = self.dtime / 2
                    print(f"*** iter {iter}: halving dtime {self.dtime:.6e} -> {new_dtime:.6e}")
                    self.dtime = new_dtime
                    dtime_halve_count += 1
                else:
                    print(f"*** iter {iter}: dtime at minimum. Aborting.")
                    break

        print(f"\n{'='*60}")
        print(f"SUMMARY: {iter-1} steps completed")
        print(f"dtime halved {dtime_halve_count} times.")
        print(f"{'='*60}\n")
        try:
            self.f.write(f"\nSUMMARY: {iter-1} steps completed, halved {dtime_halve_count} times.")
            self.f.close()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    #  PLOTTING
    # ══════════════════════════════════════════════════════════
    def plot_results(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available.")
            return

        fig, axes = plt.subplots(3, 4, figsize=(20, 14))
        fig.suptitle('3D Sphere + Pendulum (quaternion)', fontsize=16, fontweight='bold')
        labels = ['x', 'y', 'z', 'p0', 'p1', 'p2', 'p3', 'α_p', 'β_p']
        for i in range(9):
            ax = axes[i // 4, i % 4]
            ax.plot(self.t, self.q_save[i, :], linewidth=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel(labels[i])
            ax.set_title(labels[i])
            ax.grid(True, alpha=0.3)
        ax = axes[2, 3]
        ax.plot(self.t, self.lambdaN_save[0, :], 'g-', linewidth=1.5, label='λN tread')
        ax.plot(self.t, self.lambdaN_save[1, :], 'r-', linewidth=1.5, label='λN riser')
        ax.set_xlabel('Time'); ax.set_ylabel('λN'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_title('Normal contact forces')
        # hide unused
        axes[2, 1].set_visible(True)
        plt.tight_layout()
        out = os.path.join(self.output_path, 'results_3d_quat.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {out}")
        plt.close()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Same-side jumping strategy:
    # 1) roll while searching for the next riser,
    # 2) swing the pendulum up when the riser is inside the approach window,
    # 3) apply the jump torque just before the sphere reaches the step,
    # 4) coast briefly after clearing it, then re-arm for the next riser.
    alpha_p0 = -np.pi / 2
    beta_p0 = 0.0
    t_nd = np.sqrt(1.0 / 9.81)

    jump_controller = StairAwareJumpController(
        beta0=beta_p0,
        travel_direction=1.0,
        lookahead=0.75,
        profile_trigger_distance=1.0,
        repeat_profile_trigger_distance=1.0,
        swing_distance=0.30,
        jump_distance=0.08,
        settle_distance=0.35,
        settle_tau=0.0,
        hold_time=0.4 / t_nd,
        final_hold=True,
        alpha_hold=alpha_p0,
        hold_alpha_kp=0.1,
        hold_alpha_kd=0.05,
        settle_vx_tau_gain=4.0,
        settle_speed_tol=0.03,
        settle_alpha_tol=0.08,
        settle_alpha_dot_tol=0.08,
        landing_height_tol=0.015,
        roll_end_s=2.0,
        swing_end_s=2.2,
        jump_direction=-1.0,
        jump_same_side=True,
        jump_speed_rad_s=60.0,
        jump_burst_s=0.25,
        k_alpha_vel=0.08,
        tau_alpha_max=2.0,
        tau_beta_max=1.0,
    )

    sim = Simulation(
        ntime=9000,
        mu_s=0.6, mu_k=0.45,
        eN=0.3, eF=0.3,
        R=0.18,
        m_sphere=1.0, m_pendulum=0.35, l_pendulum=0.12,
        x0=0.0, y0=0.0, z0=0.18,
        p0_0=1, p1_0=0.0, p2_0=0.0, p3_0=0.0,
        alpha_p0=alpha_p0, beta_p0=beta_p0,
        x_dot0=0.0, y_dot0=0.0, z_dot0=0.0,
        wx0=0.0, wy0=0.0, wz0=0.0,
        alpha_p_dot0=0.0, beta_p_dot0=0.0,
        n_stairs=2, stair_width=1.8, stair_height=0.1,
        stair_x_start=1.0, fillet_radius=0,
        torque_controller=jump_controller,
    )
    sim.solve()
    sim.plot_results()
