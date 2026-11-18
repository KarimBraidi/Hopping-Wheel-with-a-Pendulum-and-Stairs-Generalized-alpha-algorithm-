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
import copy
import time
try:
    # needed only by the hierarchical MPC (StairJumpMPCController)
    import casadi as ca
except ImportError:  # pragma: no cover
    ca = None


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
def _cross3(a, b):
    """Cross product of two 3-vectors without np.cross's axis machinery."""
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


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

    J_D_dot u is the pendulum-point acceleration at constant rates, which
    has the closed form (rigid-body kinematics, body frame quantities):

        J_D_dot u = R [ omega x (omega x d) + 2 omega x d_dot + d_ddot ]

    with d the pendulum offset, d_dot / d_ddot its body-frame derivatives
    at constant (alpha_dot, beta_dot). This replaces a 9-way complex-step
    loop that dominated the residual cost (verified equal to 1e-12).
    """
    J_D = _velocity_jacobian_pendulum(q, l_pend)
    Rm = quat_to_R(q[3:7])
    alpha_p, beta_p = q[7], q[8]
    ca, sa = np.cos(alpha_p), np.sin(alpha_p)
    cb, sb = np.cos(beta_p), np.sin(beta_p)
    d = l_pend * np.array([ca*cb, ca*sb, sa])
    dd_da = l_pend * np.array([-sa*cb, -sa*sb, ca])
    dd_db = l_pend * np.array([-ca*sb, ca*cb, 0.0])
    d2_dab = l_pend * np.array([sa*sb, -sa*cb, 0.0])
    d2_dbb = l_pend * np.array([-ca*cb, -ca*sb, 0.0])
    ad, bd = u[6], u[7]
    omega = u[3:6]
    ddot_b = dd_da * ad + dd_db * bd
    dddot_b = -d * (ad * ad) + 2.0 * d2_dab * (ad * bd) + d2_dbb * (bd * bd)
    J_D_dot_u = Rm @ (_cross3(omega, _cross3(omega, d))
                      + 2.0 * _cross3(omega, ddot_b) + dddot_b)

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
#  NMPC stair-climbing controller
# ═══════════════════════════════════════════════════════════════
class PlanarStairModel:
    """Reduced-order planar (x-z) sphere + internal pendulum, penalty contact.

    Prediction model for the NMPC. The surrogate follows the absolute-angle
    unbalanced-hoop coordinates [x, z, theta, psi], where theta is the shell
    angle in the x-z plane and psi is the pendulum world angle. Because the
    equations only need theta_dot (for contact) and psi itself, the rollout
    state is

        s = [x, z, psi, vx, vz, theta_dot, psi_dot]      (7 per sample)

    and every operation is vectorized over a batch of rollout samples.

    Contact with the stair profile (one tread under the centre + every
    riser, corners included via closest-point normals) is a damped linear
    penalty plus regularized Coulomb friction. The damping ratio is chosen
    to give an effective restitution near the simulator's eN=0.3, and the
    friction coefficient matches the simulator's mu_k. This is only a
    planning model — the receding horizon absorbs the remaining mismatch
    with the rigid nonsmooth contact of the real solver.
    """

    def __init__(self, m_sphere, m_pendulum, l_pendulum, I_sphere, radius,
                 g, stair_edges, stair_heights, mu=0.45,
                 k_contact=1200.0, damping_ratio=0.36, v_eps=0.05,
                 max_pen_frac=0.5):
        self.m_s = float(m_sphere)
        self.m_p = float(m_pendulum)
        self.l_p = float(l_pendulum)
        self.I_s = float(I_sphere)
        self.R = float(radius)
        self.g = float(g)
        self.m_t = self.m_s + self.m_p
        self.edges = np.asarray(stair_edges, dtype=float)
        self.heights = np.asarray(stair_heights, dtype=float)
        self.mu = float(mu)
        self.k_c = float(k_contact)
        # linear Kelvin-Voigt: restitution ~ exp(-pi*zeta/sqrt(1-zeta^2));
        # zeta = 0.36 -> e ~ 0.3, matching the simulator's eN.
        self.c_c = 2.0 * float(damping_ratio) * np.sqrt(self.k_c * self.m_t)
        self.v_eps = float(v_eps)
        self.max_pen = float(max_pen_frac) * self.R
        self.last_riser_force = np.zeros(1)
        self._nx0 = None  # cached constant tread-normal buffers (see _contacts)
        self._nz0 = None
        # Constant Schur complement for the absolute-pendulum-angle mass
        # matrix. Precomputed once for the rollout hot loop.
        ml2 = self.m_p * self.l_p * self.l_p
        k_s = (self.m_p * self.l_p) ** 2 / self.m_t
        self.psi_denom = ml2 - k_s
        self.inv_I_s = 1.0 / self.I_s

    def tread_height(self, x):
        """Tread height directly under x (vectorized). Ground = 0."""
        if len(self.edges) == 0:
            return np.zeros_like(x)
        idx = np.searchsorted(self.edges, x, side="right") - 1
        return np.where(idx < 0, 0.0, self.heights[np.maximum(idx, 0)])

    def _contacts(self, x, z):
        """Yield (gap, nx, nz) for the tread below plus every riser."""
        h_t = self.tread_height(x)
        if self._nx0 is None or self._nx0.shape != x.shape:
            self._nx0 = np.zeros_like(x)
            self._nz0 = np.ones_like(x)
        yield z - (h_t + self.R), self._nx0, self._nz0
        for i in range(len(self.edges)):
            z_bot = self.heights[i - 1] if i > 0 else 0.0
            z_top = self.heights[i]
            cz = np.clip(z, z_bot, z_top)
            dx = x - self.edges[i]
            dz = z - cz
            # 1e-12 floor doubles as the exactly-on-the-wall guard: the
            # normal direction only matters when the gap is near -R there
            dist = np.sqrt(dx * dx + dz * dz)
            inv = 1.0 / np.maximum(dist, 1e-12)
            yield dist - self.R, dx * inv, dz * inv

    def step(self, S, tau, dt):
        """One semi-implicit Euler step for a batch S (N,7) under torque tau (N,)."""
        x, z, psi = S[:, 0], S[:, 1], S[:, 2]
        vx, vz, wtheta, wpsi = S[:, 3], S[:, 4], S[:, 5], S[:, 6]
        sn, cs = np.sin(psi), np.cos(psi)
        mpl = self.m_p * self.l_p

        # RHS in absolute-angle coordinates [x, z, theta, psi].
        F = np.empty((S.shape[0], 4))
        F[:, 0] = mpl * cs * wpsi * wpsi
        F[:, 1] = mpl * sn * wpsi * wpsi - self.m_t * self.g
        F[:, 2] = -tau
        F[:, 3] = -self.m_p * self.g * self.l_p * cs + tau

        riser_force = np.zeros(S.shape[0])
        for c_idx, (gap, nx, nz) in enumerate(self._contacts(x, z)):
            pen = np.minimum(np.maximum(0.0, -gap), self.max_pen)
            gdot = vx * nx + vz * nz
            lam = np.where(pen > 0.0,
                           np.maximum(0.0, self.k_c * pen - self.c_c * gdot),
                           0.0)
            # tangent t = (nz, -nx); theta_dot follows the unbalanced-hoop
            # convention, so flat-ground slip is vx + R*theta_dot.
            slip = (vx + wtheta * self.R * nz) * nz - (vz - wtheta * self.R * nx) * nx
            ft = -self.mu * lam * np.tanh(slip / self.v_eps)
            F[:, 0] += lam * nx + ft * nz
            F[:, 1] += lam * nz - ft * nx
            F[:, 2] += self.R * ft
            if c_idx > 0:  # contacts after the first are risers
                riser_force += lam
        # planning diagnostic: lets the NMPC cost penalize leaning on a riser
        self.last_riser_force = riser_force

        # Closed-form block solve of M qddot = F for [x, z, theta, psi].
        # The theta equation is independent, and the [x,z,psi] block has a
        # constant Schur complement ml^2 - (ml)^2 / m_t.
        # The Schur denominator is precomputed in __init__, so the hot loop
        # avoids a batched 4x4 np.linalg.solve for every sample.
        f1, f2 = F[:, 0], F[:, 1]
        e = mpl * (sn * f1 - cs * f2) / self.m_t
        theta_dd = F[:, 2] * self.inv_I_s
        psi_dd = (F[:, 3] + e) / self.psi_denom
        v1 = (f1 + mpl * sn * psi_dd) / self.m_t
        v2 = (f2 - mpl * cs * psi_dd) / self.m_t

        out = np.empty_like(S)
        V = out[:, 3:7]
        V[:, 0] = vx + dt * v1
        V[:, 1] = vz + dt * v2
        V[:, 2] = wtheta + dt * theta_dd
        V[:, 3] = wpsi + dt * psi_dd
        # rollout safety clamp: keeps a rare penalty blow-up from poisoning
        # the whole rollout batch with NaNs (such samples just score terribly)
        np.clip(V, -60.0, 60.0, out=V)
        out[:, 0] = x + dt * V[:, 0]
        out[:, 1] = z + dt * V[:, 1]
        out[:, 2] = psi + dt * V[:, 3]
        return out


class StairNMPCController:
    """Sampling-based nonlinear MPC for stair jumping. Torque only.

    Replaces the former hand-scripted stair controller. There are no
    behavioral modes, no calibrated launch constants (k_launch etc.) and
    no retry bookkeeping: at ~20 Hz the controller re-optimizes the whole
    pendulum-torque trajectory over a receding horizon by rolling out a
    reduced-order planar model (`PlanarStairModel`) built at run time from
    the physical robot parameters and the actual stair geometry. The
    rollout physics *contains* the launch mechanism — continuous pendulum
    rotation unloads the contact centrifugally until the sphere lifts off —
    so the optimizer rediscovers the jump for whatever riser height and
    tread width it is given, and recovery/retry behavior emerges from
    replanning instead of being scripted.

    Optimizer: iterative best-candidate sampling.
      * decision variables: piecewise-constant torque knots (ZOH, ~50 ms)
        over a ~1.4 s horizon — long enough to contain a full
        roll -> spin-up -> flight -> landing cycle;
      * warm start by time-shifting the previous plan;
      * candidates are Gaussian perturbations around the warm start, then
        around the lowest-cost sequence found so far at each iteration;
        no hand-scripted phases or proportional policy rollouts are injected
        into the optimizer;
      * cost: terminal distance to the next tread's landing zone (derived
        from the geometry) plus running height error, riser-press penalty,
        control effort, a soft actuator speed limit, and terminal
        velocity/spin penalties for calm landings.

    Interface contract with `Simulation`: `alpha_tau_func` / `beta_tau_func`
    stay pure functions of (t, q, u): alpha reads only the stored knot
    sequence, beta is the lateral motor's PD servo to beta = 0
    (make_beta_servo), which keeps the motion in the x-z plane. beta is a
    free DOF (Simulation beta_mode="motor"); beta_mode="constraint" instead
    holds it at 0 with a bilateral constraint and zero lateral torque.
    Replanning happens exclusively in `update_after_step` (converged steps).
    Torque saturation and a hard pendulum overspeed reflex are applied in
    the torque layer, so the actuator respects its physical limits
    regardless of what the optimizer asks for.

    The beta (lateral) motor is not part of the planar problem.
    """

    def __init__(
        self,
        stair_edges,
        stair_heights,
        sphere_radius,
        stair_width,
        m_sphere,
        m_pendulum,
        l_pendulum,
        t_nd,
        g_phys=9.81,
        l_nd=1.0,
        m_nd=1.0,
        # ── actuator limits (PHYSICAL units) ─────────────────
        tau_alpha_max_Nm=4.5,
        pendulum_speed_max_rad_s=55.0,
        approach_speed_max_rad_s=55.0,
        # lateral (beta) motor servo, see make_beta_servo
        beta_kp_Nm_per_rad=49.0,
        beta_kd_Nm_s_per_rad=2.5,
        tau_beta_max_Nm=1.0,
        # ── NMPC discretization ──────────────────────────────
        # budgets sized for wall-clock speed: warm starting carries most of
        # the optimization between replans, so 2 iterations x 96 samples at
        # a 5 ms rollout step recovers the same behavior several times
        # faster than the original 3 x 120 x 4 ms budget
        horizon_s=1.8,
        ctrl_dt_s=0.05,
        model_dt_s=0.005,
        replan_every_s=0.05,
        n_samples=144,
        n_iterations=4,
        noise_sigma_frac=0.25,
        temperature_frac=0.05,  # legacy MPPI option; accepted but unused
        # ── prediction-model contact ─────────────────────────
        mu_model=0.45,
        k_contact=1200.0,
        damping_ratio=0.36,
        v_eps_m_s=0.15,
        # ── cost weights ─────────────────────────────────────
        # Running height reference is location-aware in _rollout(): before
        # the launch zone it is the current rolling height; near/after the
        # riser it transitions to the target tread. That discourages early
        # hops while still allowing a real stair jump.
        w_run_pos=1.5,
        # Small horizontal anchor: without it, when a rise is infeasible
        # in-model the optimizer has no jump worth aiming for and the sphere
        # can drift away backward indefinitely (observed: -89 m at h=0.2).
        # Kept ~10x weaker than the height/press terms so it cannot recreate
        # the press-against-the-riser local minimum.
        w_run_x=0.25,
        w_effort=0.35,
        # Torque-rate (jerk) penalty across knots, anchored to the torque
        # currently being applied. Without it the selected plan inherits the
        # sampling noise and plans swing between +-tau_max knot to knot,
        # which shakes both the shell and the pendulum. Smooth spin-up
        # ramps of a real jump cost almost nothing under this term; only
        # chatter is priced out.
        w_dtau=6.0,
        w_spin=0.002,
        w_overspeed=50.0,
        w_press=6.0,
        w_run_vz=0.7,
        # Penalty on flying above the local terrain's rolling height. Keeps
        # the sphere rolling on approach (spinning the pendulum fast enough
        # to move quickly inevitably hops once centrifugal lift exceeds
        # weight, ~16 rad/s here) — a stair jump still pays for itself
        # through the terminal reward, so only pointless hops are priced out.
        w_airborne=40.0,
        w_prelaunch_airborne=3000.0,
        w_term_pos=90.0,
        w_term_vel=6.0,
        w_term_spin=0.02,
        land_frac=0.25,
        land_offset_min_m=0.25,
        launch_backoff_m=0.45,
        launch_backoff_per_rise=3.0,
        airborne_relief_start_rise_m=0.15,
        airborne_relief_full_rise_m=0.20,
        airborne_launch_relief=0.65,
        # ── misc ─────────────────────────────────────────────
        start_backoff_m=1.0,
        log_every_s=2.0,
        rng_seed=12345,
        verbose=True,
    ):
        # geometry
        self.edges = np.asarray(stair_edges, dtype=float)
        self.heights = np.asarray(stair_heights, dtype=float)
        self.n_stairs = len(self.edges)
        self.R = float(sphere_radius)
        self.stair_width = float(stair_width)

        # robot
        self.m_s = float(m_sphere)
        self.m_p = float(m_pendulum)
        self.l_p = float(l_pendulum)
        # same shell inertia formula as the simulator
        self.I_s = (2.0 / 3.0) * self.m_s * self.R ** 2

        # unit conversion (solver is nondimensional)
        self.t_nd = float(t_nd)
        self.tau_scale = float(m_nd) * float(g_phys) * float(l_nd)
        self.omega_scale = 1.0 / self.t_nd
        self.v_scale = float(l_nd) / self.t_nd
        self.g_nd = float(g_phys) * self.t_nd ** 2 / float(l_nd)

        # actuator limits in solver units
        self.tau_max = float(tau_alpha_max_Nm) / self.tau_scale
        self.omega_max = float(pendulum_speed_max_rad_s) / self.omega_scale
        self.omega_approach_max = min(float(approach_speed_max_rad_s) / self.omega_scale,
                                      self.omega_max)
        self.beta_gains = dict(kp_Nm_per_rad=float(beta_kp_Nm_per_rad),
                               kd_Nm_s_per_rad=float(beta_kd_Nm_s_per_rad),
                               tau_max_Nm=float(tau_beta_max_Nm))
        self._beta_servo = make_beta_servo(beta_kp_Nm_per_rad, beta_kd_Nm_s_per_rad,
                                           tau_beta_max_Nm, self.t_nd, self.tau_scale)

        # discretization (solver time units)
        self.model_dt = float(model_dt_s) / self.t_nd
        self.substeps = max(1, int(round(float(ctrl_dt_s) / float(model_dt_s))))
        self.ctrl_dt = self.substeps * self.model_dt
        self.n_knots = max(1, int(np.ceil(float(horizon_s) / self.t_nd / self.ctrl_dt)))
        self.replan_every = float(replan_every_s) / self.t_nd

        # Sampling parameters
        self.n_samples = int(n_samples)
        self.n_iterations = int(n_iterations)
        self.sigma = float(noise_sigma_frac) * self.tau_max
        self.temperature_frac = float(temperature_frac)  # compatibility only
        self.rng = np.random.default_rng(rng_seed)

        # cost weights
        self.w_run_pos = float(w_run_pos)
        self.w_run_x = float(w_run_x)
        self.w_effort = float(w_effort)
        self.w_dtau = float(w_dtau)
        self.w_spin = float(w_spin)
        self.w_overspeed = float(w_overspeed)
        self.w_press = float(w_press)
        self.w_run_vz = float(w_run_vz)
        self.w_airborne = float(w_airborne)
        self.w_prelaunch_airborne = float(w_prelaunch_airborne)
        self.w_term_pos = float(w_term_pos)
        self.w_term_vel = float(w_term_vel)
        self.w_term_spin = float(w_term_spin)
        self.land_frac = float(land_frac)
        self.land_offset_min = float(land_offset_min_m)
        self.launch_backoff = float(launch_backoff_m)
        self.launch_backoff_per_rise = float(launch_backoff_per_rise)
        self.airborne_relief_start_rise = float(airborne_relief_start_rise_m)
        self.airborne_relief_full_rise = float(airborne_relief_full_rise_m)
        self.airborne_launch_relief = float(airborne_launch_relief)

        self.model = PlanarStairModel(
            self.m_s, self.m_p, self.l_p, self.I_s, self.R, self.g_nd,
            self.edges, self.heights, mu=float(mu_model),
            k_contact=float(k_contact), damping_ratio=float(damping_ratio),
            v_eps=float(v_eps_m_s) / self.v_scale,
        )

        # bookkeeping / Simulation-facing attributes
        self.verbose = bool(verbose)
        self.log_every = float(log_every_s) / self.t_nd
        self.event_log = []
        self.failed = False
        self.suggested_start_x = (self.edges[0] - float(start_backoff_m)
                                  if self.n_stairs else 0.0)
        self.desired_prejump_x = self.suggested_start_x
        self.current_stair_index = 0
        self.next_stair_edge = self.edges[0] if self.n_stairs else np.nan
        self.landing_target_x = np.nan
        self.omega_spin = np.nan          # diagnostic: predicted peak |alpha_dot|
        self.controller_state = "NMPC" if self.n_stairs else "FINISHED"
        self.last_plan = None

        self._plan = None                 # torque knots currently being applied
        self._plan_t0 = 0.0
        self._last_replan_t = -np.inf
        self._last_log_t = -np.inf
        self._target = (np.nan, np.nan)
        if self.n_stairs:
            j, xt, zt = self._target_for(self._support_index(self.suggested_start_x))
            self.current_stair_index = j
            self.next_stair_edge = float(self.edges[j])
            self.landing_target_x = xt
            self._target = (xt, zt)
        if self.verbose:
            self._report_config()

    # ── helpers ───────────────────────────────────────────────
    def _saturate(self, value, limit):
        return float(np.clip(value, -limit, limit))

    def _log(self, t, msg):
        self.event_log.append((float(t) * self.t_nd, msg))

    def _report_config(self):
        print("StairNMPCController (sampling-based NMPC / best-candidate search):")
        if self.n_stairs:
            rise0 = self.heights[0]
            print(f"  {self.n_stairs} stairs, first rise {rise0:.3f} m, tread width {self.stair_width:.3f} m")
        print(f"  horizon {self.n_knots * self.ctrl_dt * self.t_nd:.2f} s "
              f"({self.n_knots} knots x {self.ctrl_dt * self.t_nd * 1e3:.0f} ms), "
              f"model dt {self.model_dt * self.t_nd * 1e3:.1f} ms, "
              f"replan every {self.replan_every * self.t_nd * 1e3:.0f} ms")
        print(f"  {self.n_samples} samples x {self.n_iterations} iterations")
        print(f"  torque cap {self.tau_max * self.tau_scale:.2f} N*m, "
              f"speed cap {self.omega_max * self.omega_scale:.0f} rad/s "
              f"(approach {self.omega_approach_max * self.omega_scale:.0f} rad/s)")

    def _launch_progress_for_x(self, x):
        if not self.n_stairs:
            return np.ones_like(np.asarray(x, dtype=float))
        launch_backoff = max(self._launch_backoff_for_current_target(), 1e-6)
        p = (np.asarray(x, dtype=float) - (float(self.next_stair_edge) - launch_backoff)) / launch_backoff
        p = np.clip(p, 0.0, 1.0)
        return p * p * (3.0 - 2.0 * p)

    def _omega_cap_for_x(self, x):
        progress = self._launch_progress_for_x(x)
        return self.omega_approach_max + progress * (self.omega_max - self.omega_approach_max)

    def _current_rise(self):
        if not self.n_stairs:
            return 0.0
        j = int(np.clip(self.current_stair_index, 0, self.n_stairs - 1))
        h_prev = float(self.heights[j - 1]) if j > 0 else 0.0
        return max(0.0, float(self.heights[j]) - h_prev)

    def _launch_backoff_for_current_target(self):
        rise = self._current_rise()
        adaptive = self.launch_backoff_per_rise * rise
        cap = max(self.launch_backoff, 0.60 * self.stair_width)
        return min(max(self.launch_backoff, adaptive), cap)

    def _airborne_relief_for_current_target(self):
        span = max(self.airborne_relief_full_rise - self.airborne_relief_start_rise, 1e-9)
        return float(np.clip((self._current_rise() - self.airborne_relief_start_rise) / span, 0.0, 1.0))

    # ── stair targeting (pure geometry, no tuning) ────────────
    def _support_index(self, x):
        """Index of the tread under x; -1 = ground before the stairs."""
        if self.n_stairs == 0:
            return -1
        return int(np.searchsorted(self.edges, x, side="right")) - 1

    def _target_for(self, support_idx):
        """Landing-zone target on the next tread (or hold point on the top one)."""
        j = int(min(support_idx + 1, self.n_stairs - 1))
        land_off = max(self.land_offset_min, self.land_frac * self.stair_width)
        land_off = min(land_off, self.stair_width - self.R - 0.05)
        return j, float(self.edges[j] + land_off), float(self.heights[j] + self.R)

    def _planar_state(self, q, u):
        e_w = quat_to_R(q[3:7]) @ pendulum_dir_body(q[7], q[8])
        psi = float(np.arctan2(e_w[2], e_w[0]))
        theta_dot = -float(u[4])
        psi_dot = float(u[6]) + theta_dot
        return np.array([q[0], q[2], psi, u[0], u[2], theta_dot, psi_dot])

    # Iterative best-candidate sampling
    def _rollout(self, s0, knots, tau_anchor=0.0):
        """Roll out candidate knot sequences.

        `tau_anchor` is the torque currently being applied — the rate penalty
        is anchored to it so a fresh plan cannot step the torque
        discontinuously at the replan."""
        Nb, K = knots.shape
        S = np.repeat(s0[None, :], Nb, axis=0)
        cost = np.zeros(Nb)
        alpha_rate = S[:, 6] - S[:, 5]
        peak = np.abs(alpha_rate).copy()
        xt, zt = self._target
        edge = float(self.next_stair_edge) if self.n_stairs else np.inf
        launch_backoff = max(self._launch_backoff_for_current_target(), 1e-6)
        relief = self._airborne_relief_for_current_target()
        guard_power = 1.0 + relief
        dt = self.model_dt
        for k in range(K):
            tau = knots[:, k]
            for _ in range(self.substeps):
                # same overspeed taper the torque layer applies at execution,
                # so rollouts cannot promise speeds the actuator won't deliver
                wa_now = S[:, 6] - S[:, 5]
                outward = tau * wa_now > 0.0
                omega_cap_now = self._omega_cap_for_x(S[:, 0])
                scale = np.clip((omega_cap_now - np.abs(wa_now))
                                / (0.06 * np.maximum(omega_cap_now, 1e-9)), 0.0, 1.0)
                tau_eff = np.where(outward, tau * scale, tau)
                S = self.model.step(S, tau_eff, dt)
                terrain_z = self.model.tread_height(S[:, 0]) + self.R
                linear_prelaunch = np.clip((edge - S[:, 0]) / launch_backoff, 0.0, 1.0)
                smooth_progress = self._launch_progress_for_x(S[:, 0])
                launch_progress = ((1.0 - relief) * (1.0 - linear_prelaunch)
                                   + relief * smooth_progress)
                prelaunch = ((1.0 - relief) * linear_prelaunch
                             + relief * (1.0 - smooth_progress) ** guard_power)
                z_ref = (1.0 - launch_progress) * terrain_z + launch_progress * zt
                dz = S[:, 1] - z_ref
                dxr = S[:, 0] - xt
                wa = S[:, 6] - S[:, 5]
                omega_cap = self._omega_cap_for_x(S[:, 0])
                over = np.maximum(0.0, np.abs(wa) - omega_cap)
                air = np.maximum(0.0, S[:, 1] - terrain_z)
                air_weight = self.w_airborne * (1.0 - self.airborne_launch_relief
                                                 * relief * launch_progress)
                cost += dt * (self.w_run_pos * dz * dz
                              + self.w_run_x * dxr * dxr
                              + self.w_effort * tau * tau
                              + self.w_spin * wa * wa
                              + self.w_overspeed * over * over
                              + self.w_press * self.model.last_riser_force
                              + self.w_run_vz * S[:, 4] * S[:, 4]
                              + air_weight * air * air
                              + self.w_prelaunch_airborne * prelaunch * air * air)
                np.maximum(peak, np.abs(wa), out=peak)
        dx = S[:, 0] - xt
        dz = S[:, 1] - zt
        alpha_rate_end = S[:, 6] - S[:, 5]
        cost += (self.w_term_pos * (dx * dx + dz * dz)
                 + self.w_term_vel * (S[:, 3] ** 2 + S[:, 4] ** 2)
                 + self.w_term_spin * alpha_rate_end ** 2)
        # Torque-rate penalty over the final knot sequences, anchored to the
        # torque in effect at replan time so new plans take over smoothly.
        w_dtau = getattr(self, "_w_dtau_eff", self.w_dtau)
        cost += w_dtau * ((knots[:, 0] - tau_anchor) ** 2
                          + np.sum(np.diff(knots, axis=1) ** 2, axis=1))
        return np.nan_to_num(cost, nan=1e12, posinf=1e12), S, peak

    def _warm_start(self, t):
        if self._plan is None:
            return np.zeros(self.n_knots)
        shift = int(round((t - self._plan_t0) / self.ctrl_dt))
        shift = max(0, min(shift, self.n_knots))
        w = np.empty(self.n_knots)
        w[:self.n_knots - shift] = self._plan[shift:]
        if shift > 0:
            w[self.n_knots - shift:] = self._plan[-1]
        return w

    def _replan(self, t, q, u):
        s0 = self._planar_state(q, u)
        K = self.n_knots
        sampling_center = self._warm_start(t)
        # torque currently in effect: anchor for the rate penalty so the new
        # plan takes over from the old one without a torque step
        if self._plan is None:
            tau_anchor = 0.0
        else:
            k_now = int((t - self._plan_t0) / self.ctrl_dt)
            tau_anchor = float(self._plan[min(max(k_now, 0), K - 1)])
        # Adaptive exploration: when the previous plan already scored well
        # (executing a good jump, or holding at the target) large sampling
        # noise only disrupts it — the applied best-candidate then inherits
        # jitter, which showed up as the sphere fidgeting/hopping on the top
        # tread. Anneal sigma toward 25% as the incumbent cost approaches 0;
        # full exploration returns automatically whenever the cost is high
        # (stuck, or a new stair target appeared).
        prev_cost = self.last_plan["cost"] if self.last_plan else np.inf
        need = min(1.0, prev_cost / 10.0)  # 0 = plan is working, 1 = stuck
        sigma_eff = self.sigma * (0.25 + 0.75 * need)
        self._w_dtau_eff = self.w_dtau
        best_seq, best_cost = None, np.inf
        best_end, best_peak = None, 0.0
        for _ in range(self.n_iterations):
            eps = self.rng.normal(0.0, sigma_eff, size=(self.n_samples, K))
            # two smoothing passes: low-pass the exploration so candidate
            # torque profiles are not chattery
            for _ in range(2):
                eps[:, 1:-1] = 0.25 * eps[:, :-2] + 0.5 * eps[:, 1:-1] + 0.25 * eps[:, 2:]
            cand = np.clip(sampling_center[None, :] + eps, -self.tau_max, self.tau_max)
            # Keep the current center as a candidate so a good plan survives
            # even when none of the new perturbations improves its cost.
            cand = np.vstack([sampling_center[None, :], cand])
            costs, S_ends, peaks = self._rollout(s0, cand, tau_anchor)
            i_best = int(np.argmin(costs))
            if costs[i_best] < best_cost:
                best_cost = float(costs[i_best])
                best_seq = cand[i_best].copy()
                best_end = S_ends[i_best].copy()
                best_peak = float(peaks[i_best])
            sampling_center = best_seq.copy()

        # Execute the best evaluated sequence; its time-shifted values warm
        # start the next replan. The torque-rate penalty keeps plans smooth.
        c_nom, S_end, peak = np.array([best_cost]), best_end[None, :], np.array([best_peak])

        self._plan = best_seq
        self._plan_t0 = float(t)
        self._last_replan_t = float(t)
        self.omega_spin = float(peak[0])
        self.last_plan = {
            "planner": "sampling_nmpc_best_candidate",
            "t_s": float(t) * self.t_nd,
            "target_x": self._target[0],
            "target_z": self._target[1],
            "cost": float(c_nom[0]),
            "predicted_x_end": float(S_end[0, 0]),
            "predicted_z_end": float(S_end[0, 1]),
            "predicted_peak_spin_rad_s": float(peak[0]) * self.omega_scale,
        }
        if self.verbose and t - self._last_log_t >= self.log_every:
            self._last_log_t = float(t)
            self._log(t, f"replan: target stair {self.current_stair_index} "
                         f"(x={self._target[0]:.2f}, z={self._target[1]:.2f}), "
                         f"predicted end x={S_end[0, 0]:.2f} z={S_end[0, 1]:.2f}, "
                         f"peak spin {peak[0] * self.omega_scale:.1f} rad/s, "
                         f"cost {c_nom[0]:.2f}")

    def get_plan_metadata(self):
        return {
            "controller": "StairNMPCController",
            "method": "Best-candidate sampling NMPC on reduced planar model",
            "tau_alpha_max_Nm": self.tau_max * self.tau_scale,
            "beta_control": "lateral_motor_servo",
            "beta_servo": self.beta_gains,
            "pendulum_speed_max_rad_s": self.omega_max * self.omega_scale,
            "approach_speed_max_rad_s": self.omega_approach_max * self.omega_scale,
            "horizon_s": self.n_knots * self.ctrl_dt * self.t_nd,
            "ctrl_dt_s": self.ctrl_dt * self.t_nd,
            "model_dt_s": self.model_dt * self.t_nd,
            "replan_every_s": self.replan_every * self.t_nd,
            "n_samples": self.n_samples,
            "n_iterations": self.n_iterations,
            "weights": {
                "w_run_pos": self.w_run_pos, "w_run_x": self.w_run_x,
                "w_effort": self.w_effort, "w_dtau": self.w_dtau,
                "w_spin": self.w_spin, "w_overspeed": self.w_overspeed,
                "w_press": self.w_press, "w_run_vz": self.w_run_vz,
                "w_airborne": self.w_airborne,
                "w_prelaunch_airborne": self.w_prelaunch_airborne,
                "w_term_pos": self.w_term_pos, "w_term_vel": self.w_term_vel,
                "w_term_spin": self.w_term_spin,
            },
            "model_contact": {
                "mu": self.model.mu, "k_contact": self.model.k_c,
                "c_contact": self.model.c_c, "v_eps": self.model.v_eps,
            },
            "land_frac": self.land_frac,
            "land_offset_min_m": self.land_offset_min,
            "launch_backoff_m": self.launch_backoff,
            "launch_backoff_per_rise": self.launch_backoff_per_rise,
            "airborne_relief_start_rise_m": self.airborne_relief_start_rise,
            "airborne_relief_full_rise_m": self.airborne_relief_full_rise,
            "airborne_launch_relief": self.airborne_launch_relief,
            "last_plan": self.last_plan,
        }

    # ── event-level update (converged steps only) ─────────────
    def update_after_step(self, t, q, u):
        if self.n_stairs == 0:
            return
        x = float(q[0])
        z = float(q[2])
        vz = float(u[2])
        s_idx = self._support_index(x)
        support_target, _, _ = self._target_for(s_idx)
        j = self.current_stair_index
        xt, zt = self._target
        # Advance the target only once actually LANDED on the new tread.
        # Crossing the edge mid-flight used to advance it immediately, so the
        # controller kept spinning for the *next* stair instead of braking
        # for touchdown — overshooting into the following riser and
        # rebounding backward (observed at h=0.15: peak z 0.59 for a 0.33
        # target, then a runaway backward drift). Dropping back to a lower
        # tread retargets immediately.
        if support_target < self.current_stair_index:
            j, xt, zt = self._target_for(s_idx)
        elif self.current_stair_index < self.n_stairs - 1:
            x_tol = max(0.25 * self.R, 0.04 * self.stair_width)
            z_tol = max(0.02, 0.25 * self.R)
            landed_current = (x >= self.landing_target_x - x_tol
                              and abs(z - self._target[1]) <= z_tol
                              and abs(vz) <= 0.20)
            if landed_current:
                j = self.current_stair_index + 1
                _, xt, zt = self._target_for(j - 1)
        if j != self.current_stair_index:
            self._log(t, f"target advanced to stair {j} "
                         f"(x={q[0]:.2f}, z={q[2]:.2f})")
        self.current_stair_index = j
        self.next_stair_edge = float(self.edges[j])
        self.landing_target_x = xt
        self._target = (xt, zt)

        if self._plan is None or t - self._last_replan_t >= self.replan_every:
            self._replan(t, q, u)

    # ── torque laws (pure functions of t, q, u) ───────────────
    def _overspeed_scale(self, wa, tau, omega_cap=None):
        """Fade outward torque to zero over the last ~6% below the speed cap.

        Must be SMOOTH in the pendulum speed: a hard on/off cutoff flips the
        torque discontinuously as u[6] crosses the cap inside the implicit
        solver's Newton iterations and stalls convergence (observed aborting
        with the pendulum pinned at exactly the 55 rad/s cap)."""
        if tau * wa <= 0.0:
            return 1.0  # braking torque is always allowed
        if omega_cap is None:
            omega_cap = self.omega_max
        margin = 0.06 * max(float(omega_cap), 1e-9)
        return float(np.clip((float(omega_cap) - abs(wa)) / margin, 0.0, 1.0))

    def alpha_tau_func(self, t, q, u):
        if self._plan is None:
            tau = 0.0
        else:
            k = int((t - self._plan_t0) / self.ctrl_dt)
            tau = float(self._plan[min(max(k, 0), self.n_knots - 1)])
        omega_cap = float(self._omega_cap_for_x(float(q[0])))
        tau *= self._overspeed_scale(float(u[6]), tau, omega_cap)
        return self._saturate(tau, self.tau_max)

    def beta_tau_func(self, t, q, u):
        return self._beta_servo(t, q, u)


# ═══════════════════════════════════════════════════════════════
#  Hierarchical stair-jumping MPC (second controller choice)
#
#  reference generation (JumpPlanner, once per jump)
#    -> phase-dependent NMPC (TrackingNMPC / FlightNMPC / settle NMPC, CasADi + IPOPT)
#    -> actuator saturation (torque, motor speed, motor acceleration)
#    -> pendulum motor torque into Simulation (via stair_controller.alpha_tau_func)
#
#  Same controller as the planar hoop project ("hoop with a pendulum and
#  stairs (semi-smooth)", frozen v2 in its mpc_stair_jumping/ folder; report
#  in mpc_stair_jumping/report/). Only the prediction-model parameters (thin
#  shell inertia 2/3 m R^2, friction) and the 3D state/torque interface of
#  the controller class differ.
# ═══════════════════════════════════════════════════════════════

# All controller settings (physical SI units); the hoop project's mpc_stair_jumping/config/
# controller_config.json with the sphere's plant parameters and three added settle_nmpc
# entries for narrow treads: elev_slack_l1 (see TrackingNMPC._build), hang_ref_deadband_rad
# and multistart (see _solve_settle). __main__ overrides plant/actuator.
# NOTE: alpha_max_rad_s2 is an ASSUMPTION (no motor-acceleration spec in the project).
DEFAULT_MPC_CONFIG = {'_comment': 'All controller settings. Physical SI units. See the hoop project report '
             '(mpc_stair_jumping/report), Sec. 6-11, for the meaning of every entry.',
 'plant': {'R': 0.18,
           'm_sphere': 1.0,
           'm_pendulum': 0.5,
           'l_pendulum': 0.15,
           'g': 9.81,
           'mu_k': 0.45,
           'mu_s': 0.6,
           'eN': 0.3,
           'eF': 0.3,
           'n_stairs': 3,
           'stair_width': 1.0,
           'stair_height': 0.12,
           'stair_x_start': 1.0},
 'actuator': {'tau_max_Nm': 4.5,
              'omega_max_rad_s': 55.0,
              'alpha_max_rad_s2': 400.0,
              '_alpha_note': 'ASSUMPTION: no acceleration spec exists in the project; alpha_max = '
                             'tau_max/(m_p l^2) = 4.5/(0.5*0.15^2) = 400 rad/s^2',
              'unload_frac': 1.0},
 'model': {'eps_slip_m_s': 0.03},
 'planner': {'collocation_degree': 3,
             'N_approach': 20,
             'N_launch': 28,
             'N_flight': 24,
             'T_bounds': {'A': [0.05, 2.0], 'L': [0.05, 0.6], 'F': [0.08, 1.0]},
             'speed_margin': 0.95,
             'accel_margin': 0.95,
             'lamN_min_frac': {'A': 0.0, 'L': 0.0},
             'approach_phi_halfwidth': 1.5708,
             'landing_unload_eta': 0.5,
             'corner_margin': 0.02,
             'riser_margin_stance': 0.01,
             'touchdown_vy_min': 0.05,
             'landing_min_past_edge': 0.1,
             'v_ref': 1.0,
             'L_ref': 0.1,
             'scales': {'x': 1.0,
                        'y': 0.3,
                        'phi': 3.14159,
                        'vx': 1.0,
                        'vy': 1.0,
                        'thetad': 10.0,
                        'phid': 30.0},
             'weights': {'effort': 1.0,
                         'approach_slip': 1.0,
                         'time': 1.0,
                         'landing_x': 1.0,
                         'landing_vy': 1.0,
                         'landing_slip': 1.0,
                         'landing_energy': 0.1,
                         'dtau': 0.05},
             'max_iter': 1500,
             'tol': 1e-06,
             'nlp_scaling_method': 'none',
             'prelanding_frac': 0.33,
             'prelanding_wrel': 5.0,
             'launch_mode_robust_accel': True,
             'max_cpu_time_s': 30.0},
 'tracking_nmpc': {'Ts': 0.02,
                   'N': 15,
                   'collocation_degree': 3,
                   'speed_margin': 0.97,
                   'accel_margin': 0.97,
                   'corner_margin': 0.02,
                   'weights': {'tau_ref': 1.0, 'effort': 0.01, 'dtau': 0.1},
                   'tau_dev_scale': 1.0,
                   'dtau_scale': 2.0,
                   'slack_l1': 100.0,
                   'slack_l2': 1000.0,
                   'terminal_factor': 10.0,
                   'track_scales': {'x': 0.03,
                                    'y': 0.02,
                                    'phi': 0.15,
                                    'vx': 0.15,
                                    'vy': 0.2,
                                    'thetad': 2.0,
                                    'phid': 4.0},
                   'settle_scales': {'x': 0.1,
                                     'y': 0.02,
                                     'phi': 0.5,
                                     'vx': 0.1,
                                     'vy': 0.2,
                                     'thetad': 1.0,
                                     'phid': 2.0},
                   'max_iter': 100,
                   'tol': 1e-05,
                   'acceptable_tol': 0.001,
                   'nlp_scaling_method': 'none',
                   'warm_start_multipliers': True,
                   'mu_init': 0.001},
 'flight_nmpc': {'N': 12,
                 'collocation_degree': 3,
                 'T_min': 0.01,
                 'T_max': 1.0,
                 'speed_margin': 0.97,
                 'accel_margin': 0.97,
                 'slack_l1': 100.0,
                 'slack_l2': 1000.0,
                 'max_iter': 100,
                 'tol': 1e-05,
                 'acceptable_tol': 0.001,
                 'nlp_scaling_method': 'none',
                 'warm_start_multipliers': True,
                 'mu_init': 0.001},
 'geometry_targets': {'touchdown_past_edge_m': 0.25, 'hold_past_edge_m': 0.45},
 'state_machine': {'takeoff_gap_m': 0.003,
                   'touchdown_gap_m': 0.001,
                   'launch_timeout_s': 0.15,
                   'settle_contact_s': 0.25,
                   'settle_vx': 0.3,
                   'plan_retry_s': 0.3,
                   'min_runup_m': 0.6,
                   'settle_phid': 3.0,
                   'settle_dphi': 0.6},
 'settle_nmpc': {'Ts': 0.066667,
                 'N': 15,
                 'collocation_degree': 3,
                 'speed_margin': 0.97,
                 'accel_margin': 0.97,
                 'corner_margin': 0.02,
                 'weights': {'tau_ref': 1.0, 'effort': 0.01, 'dtau': 0.1},
                 'tau_dev_scale': 1.0,
                 'dtau_scale': 2.0,
                 'slack_l1': 100.0,
                 'slack_l2': 1000.0,
                 'terminal_factor': 10.0,
                 'track_scales': {'x': 0.03,
                                  'y': 0.02,
                                  'phi': 0.15,
                                  'vx': 0.15,
                                  'vy': 0.2,
                                  'thetad': 2.0,
                                  'phid': 4.0},
                 'settle_scales': {'x': 0.1,
                                   'y': 0.02,
                                   'phi': 0.5,
                                   'vx': 0.1,
                                   'vy': 0.2,
                                   'thetad': 1.0,
                                   'phid': 2.0},
                 'max_iter': 100,
                 'tol': 1e-05,
                 'acceptable_tol': 0.001,
                 'nlp_scaling_method': 'none',
                 'warm_start_multipliers': True,
                 'mu_init': 0.001,
                 'max_pendulum_elevation_rad': 0.5236,
                 'elev_slack_l1': 10000.0,
                 'hang_ref_deadband_rad': 0.5,
                 'multistart': True}}


# ---------------------------------------------------------------------------
#  Prediction models (flight / stance with regularized Coulomb friction)  (hoop project: src/models.py)
# ---------------------------------------------------------------------------
# Simplified (planar) prediction models for the stair-jumping NMPC (physical SI units).
#
# Under a stair controller the lateral servo keeps beta_p ~ 0 (or the simulator
# constrains it to 0), and the sphere then moves in the x-z plane.
# The 3D model of ``Simulation`` (compute_mass_matrix,
# compute_h) reduces exactly to a planar unbalanced shell with generalized
# coordinates q = [x, y, theta, phi]: shell centre (y here = the simulator's
# vertical z), shell rolling angle theta (theta_dot = -omega_y) and absolute
# pendulum angle phi = alpha_p + theta (the psi of PlanarStairModel). Mass
# matrix and force vector
#
#     M(phi) = [[m, 0, 0, -m_p l s],
#               [0, m, 0,  m_p l c],
#               [0, 0, I_s, 0     ],
#               [-m_p l s, m_p l c, 0, m_p l^2]]
#     h      = [m_p l c phi_d^2,
#               m_p l s phi_d^2 - m g,
#               -tau,
#               -m_p g l c + tau]
#
# (s = sin phi, c = cos phi, I_s = 2/3 m_sphere R^2 thin shell). The shell is
# axisymmetric, so theta is cyclic and the prediction state is
#
#     s = [x, y, phi, xd, yd, thetad, phid]          (7 states)
#
# with the motor torque tau (pendulum relative to shell, i.e. the alpha_p
# torque) as the only input.
#
# Two smooth phase models are provided:
#
# * flight:  M vdot = h                                   (exact, no contact)
# * stance:  M vdot = h + W_N lam_N + W_F lam_F,  ydd = 0  (tread contact)
#            lam_F = -mu * lam_N * tanh(gamma / eps),  gamma = xd + R thetad
#
# The stance model keeps the sphere on the tread (bilateral during the phase,
# lam_N >= 0 is imposed as a path constraint by the OCPs) and uses a smooth,
# explicit friction law instead of complementarity constraints, because
# complementarity constraints violate LICQ and are unsuitable for standard NLP
# solvers (Neunert et al., RA-L 2018). Rolling-without-slip is NOT assumed: at
# lift-off lam_N -> 0, so the friction available for rolling vanishes and the
# plant necessarily slips (see report, Sec. 4).
NX = 7
IX, IY, IPHI, IVX, IVY, IWT, IWP = range(7)


class PlanarSphereParams:
    """Physical parameters (SI) of the planar model. Defaults = this file's __main__."""

    def __init__(self, R=0.18, m_sphere=1.0, m_pendulum=0.5, l_pendulum=0.15,
                 g=9.81, mu=0.45, eps_slip=0.03):
        self.R = float(R)
        self.m_s = float(m_sphere)
        self.m_p = float(m_pendulum)
        self.l = float(l_pendulum)
        self.g = float(g)
        self.mu = float(mu)
        self.eps = float(eps_slip)
        self.m = self.m_s + self.m_p
        self.I_s = (2.0 / 3.0) * self.m_s * self.R ** 2   # thin shell, as in Simulation
        self.mpl = self.m_p * self.l


def _forces(p, s, tau):
    phi, wp = s[IPHI], s[IWP]
    sp, cp = ca.sin(phi), ca.cos(phi)
    h1 = p.mpl * cp * wp ** 2
    h2 = p.mpl * sp * wp ** 2 - p.m * p.g
    h3 = -tau
    h4 = -p.m_p * p.g * p.l * cp + tau
    return h1, h2, h3, h4, sp, cp


def flight_accel(p, s, tau):
    """Closed-form solution of M vdot = h (same block solve as the plant)."""
    h1, h2, h3, h4, sp, cp = _forces(p, s, tau)
    denom = p.m_p * p.l ** 2 - p.mpl ** 2 / p.m
    e = p.mpl * (sp * h1 - cp * h2) / p.m
    pdd = (h4 + e) / denom
    xdd = (h1 + p.mpl * sp * pdd) / p.m
    ydd = (h2 - p.mpl * cp * pdd) / p.m
    tdd = h3 / p.I_s
    return xdd, ydd, tdd, pdd


def stance_accel(p, s, tau):
    """Stance on a horizontal tread, ydd = 0, regularized Coulomb friction.

    Unknowns z = [xdd, thetadd, phidd, lamN]; the y-equation gives lamN.
      x:   m xdd - mpl s phidd + mu*sig*lamN = h1
      th:  I_s thdd + mu*sig*R*lamN          = h3
      phi: -mpl s xdd + m_p l^2 phidd         = h4
      y:   mpl c phidd - lamN                 = h2
    Returns (xdd, thdd, phidd, lamN).
    """
    h1, h2, h3, h4, sp, cp = _forces(p, s, tau)
    gamma = s[IVX] + p.R * s[IWT]
    sig = ca.tanh(gamma / p.eps)
    a, b = p.mpl, p.m_p * p.l ** 2
    ms = p.mu * sig
    # eliminate lamN = a c phidd - h2, then a 2x2 solve for (xdd, phidd);
    # det = m b - a^2 s^2 + mu sig a^2 s c >= m b - a^2 - mu a^2/2 > 0
    r1 = h1 + ms * h2
    k12 = ms * a * cp - a * sp
    det = p.m * b + a * sp * k12
    xdd = (b * r1 - k12 * h4) / det
    pdd = (p.m * h4 + a * sp * r1) / det
    lamN = a * cp * pdd - h2
    tdd = (h3 - ms * p.R * lamN) / p.I_s
    return xdd, tdd, pdd, lamN


def make_functions(p):
    """CasADi Functions: f_flight, f_stance (state derivative) and outputs."""
    s = ca.SX.sym("s", NX)
    tau = ca.SX.sym("tau")
    xdd, ydd, tdd, pdd = flight_accel(p, s, tau)
    sdot_f = ca.vertcat(s[IVX], s[IVY], s[IWP], xdd, ydd, tdd, pdd)
    alpha_f = pdd - tdd
    xdd_s, tdd_s, pdd_s, lamN = stance_accel(p, s, tau)
    sdot_s = ca.vertcat(s[IVX], 0.0, s[IWP], xdd_s, 0.0, tdd_s, pdd_s)
    alpha_s = pdd_s - tdd_s
    F = {
        "flight": ca.Function("f_flight", [s, tau], [sdot_f], ["s", "tau"], ["sdot"]),
        "stance": ca.Function("f_stance", [s, tau], [sdot_s], ["s", "tau"], ["sdot"]),
        # relative (motor) acceleration phi_dd - theta_dd for each phase model
        "alpha_flight": ca.Function("alpha_f", [s, tau], [alpha_f]),
        "alpha_stance": ca.Function("alpha_s", [s, tau], [alpha_s]),
        "lamN": ca.Function("lamN", [s, tau], [lamN]),
    }
    return F


# ── stair geometry helper (shared by planner and NMPC) ─────
def clearance_terms(x, y, R, h_lo, h_hi, x_edge, delta_corner):
    """Smooth(C^1) clearance constraints of the sphere centre w.r.t. one step.

    Obstacle = lower tread {y <= h_lo} U step block {x >= x_edge, y <= h_hi}.
    Returns expressions that must be >= 0:
      c_tread  = y - (h_lo + R)                         (lower tread)
      c_block  = max(x_edge-x,0)^2 + max(y-h_hi,0)^2 - R^2   (exact distance
                 to the step block, Minkowski-inflated by the radius)
      c_corner = (x-x_edge)^2 + (y-h_hi)^2 - (R+delta)^2  (corner margin)
    """
    c_tread = y - (h_lo + R)
    dx = ca.fmax(x_edge - x, 0.0)
    dy = ca.fmax(y - h_hi, 0.0)
    c_block = dx ** 2 + dy ** 2 - R ** 2
    c_corner = (x - x_edge) ** 2 + (y - h_hi) ** 2 - (R + delta_corner) ** 2
    return c_tread, c_block, c_corner


# ---------------------------------------------------------------------------
#  Radau IIA collocation  (hoop project: src/collocation.py)
# ---------------------------------------------------------------------------
# Radau IIA direct-collocation helpers (CasADi).
#
# Direct collocation is used because the stance model with regularized
# Coulomb friction is stiff (slip time constants of ~1-7 ms); Radau IIA
# collocation is an implicit, stiffly accurate discretization, so the NLP
# transcription stays well-posed at the 5-30 ms interval lengths used here.
def radau_coefficients(d):
    """Collocation matrix C and continuity coefficients D for Radau degree d."""
    tau = np.append(0.0, ca.collocation_points(d, "radau"))
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        D[j] = p(1.0)
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau[r])
    return tau, C, D


# ---------------------------------------------------------------------------
#  NLP assembly helper  (hoop project: src/nlp_tools.py)
# ---------------------------------------------------------------------------
# Small helpers to assemble SX-based NLPs for casadi.nlpsol.
class NLPBuilder:
    def __init__(self):
        self.w, self.lbw, self.ubw, self.w0 = [], [], [], []
        self.g, self.lbg, self.ubg = [], [], []
        self.n_w = 0
        self.slices = {}

    def var(self, name, shape, lb=-np.inf, ub=np.inf, init=0.0):
        rows, cols = shape
        v = ca.SX.sym(name, rows, cols)
        n = rows * cols
        self.w.append(ca.vec(v))
        self.lbw.append(np.broadcast_to(np.asarray(lb, float), (rows, cols)).ravel(order="F"))
        self.ubw.append(np.broadcast_to(np.asarray(ub, float), (rows, cols)).ravel(order="F"))
        self.w0.append(np.broadcast_to(np.asarray(init, float), (rows, cols)).ravel(order="F"))
        self.slices[name] = (self.n_w, self.n_w + n, shape)
        self.n_w += n
        return v

    def con(self, expr, lb, ub):
        expr = ca.vec(expr)
        n = expr.numel()
        self.g.append(expr)
        self.lbg.append(np.broadcast_to(np.asarray(lb, float), (n,)).copy())
        self.ubg.append(np.broadcast_to(np.asarray(ub, float), (n,)).copy())

    def eq(self, expr):
        self.con(expr, 0.0, 0.0)

    def build(self, J, p, solver_name, plugin, opts):
        nlp = {"x": ca.vertcat(*self.w), "f": J, "g": ca.vertcat(*self.g), "p": p}
        solver = ca.nlpsol(solver_name, plugin, nlp, opts)
        self.lbw_v = np.concatenate(self.lbw)
        self.ubw_v = np.concatenate(self.ubw)
        self.w0_v = np.concatenate(self.w0)
        self.lbg_v = np.concatenate(self.lbg) if self.lbg else np.zeros(0)
        self.ubg_v = np.concatenate(self.ubg) if self.ubg else np.zeros(0)
        return solver

    def get(self, wvec, name):
        a, b, shape = self.slices[name]
        return np.asarray(wvec[a:b]).reshape(shape, order="F")

    def put(self, wvec, name, value):
        a, b, shape = self.slices[name]
        wvec[a:b] = np.broadcast_to(np.asarray(value, float), shape).ravel(order="F")


# ---------------------------------------------------------------------------
#  Level 1: stair-aware jump planner  (hoop project: src/planner.py)
# ---------------------------------------------------------------------------
# Level 1 - stair-aware jump trajectory optimization (reference generator).
#
# For the next stair (edge x_e, lower tread h_lo, upper tread h_hi) and the
# current measured state s0, solve a multi-phase optimal control problem with
# a predefined contact sequence
#
#     A (approach, stance) -> L (launch, stance) -> F (flight) -> touchdown
#
# and free phase durations, using the full planar rigid-body model
# (models.py). A and L share the same stance dynamics; they are separate
# phases only so that the fast launch transient receives a finer collocation
# mesh than the approach. The OCP is transcribed by Radau IIA direct
# collocation and solved with IPOPT through CasADi.
#
# Precedents: full-nonlinear-dynamics jump TO with a predefined contact
# schedule, actuator torque/velocity limits, GRF limits, friction cone and
# geometric constraints (Nguyen, Bao & Nguyen, CDC 2022, Eq. (2)); online
# TO tailored to the sensed obstacle (Park, Wensing & Kim, RAS 2021);
# plan-once-then-track (Chignoli & Kim, ICRA 2021). Decision variables and
# constraints are scaled by physical reference values (Ross et al., JGCD
# 2018).
# parameter vector layout
P_S0 = slice(0, NX)
P_XE, P_HLO, P_HHI, P_XL, P_XLMAX, P_TAUPREV = range(NX, NX + 6)
NP = NX + 6


class JumpPlanner:
    def __init__(self, P, cfg, limits):
        self.P = P
        self.cfg = cfg
        self.lim = limits
        self.F = make_functions(P)
        self.d = int(cfg["collocation_degree"])
        self.tau_c, self.C, self.D = radau_coefficients(self.d)
        sc = cfg["scales"]
        self.Sx = np.array([sc["x"], sc["y"], sc["phi"], sc["vx"], sc["vy"],
                            sc["thetad"], sc["phid"]], float)
        self.phases = [("A", "stance", int(cfg["N_approach"])),
                       ("L", "stance", int(cfg["N_launch"])),
                       ("F", "flight", int(cfg["N_flight"]))]
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        P, cfg, lim = self.P, self.cfg, self.lim
        w = cfg["weights"]
        nb = NLPBuilder()
        self.nb = nb
        p = ca.SX.sym("p", NP)
        Sx = ca.DM(self.Sx)
        tau_max = lim["tau_max"]
        om_lim = lim["omega_max"] * cfg["speed_margin"]
        al_lim = lim["alpha_max"] * cfg["accel_margin"]
        mg = P.m * P.g
        R = P.R
        dlt_c = cfg["corner_margin"]
        dlt_r = cfg["riser_margin_stance"]
        x_e, h_lo, h_hi = p[P_XE], p[P_HLO], p[P_HHI]

        J = 0
        self.sym = {}
        u_all = []
        X_prev_end = None
        for name, kind, N in self.phases:
            tb = cfg["T_bounds"][name]
            T = nb.var(f"T_{name}", (1, 1), tb[0], tb[1], 0.5 * (tb[0] + tb[1]))
            Xs = nb.var(f"X_{name}", (NX, N + 1))
            Xcs = nb.var(f"Xc_{name}", (NX, N * self.d))
            Us = nb.var(f"U_{name}", (1, N), -1.0, 1.0)
            X = ca.diag(Sx) @ Xs
            Xc = ca.diag(Sx) @ Xcs
            U = tau_max * Us
            h = T / N
            f = self.F[kind]
            fa = self.F["alpha_" + kind]
            for k in range(N):
                Xk = [X[:, k]] + [Xc[:, k * self.d + j] for j in range(self.d)]
                uk = U[0, k]
                for j in range(1, self.d + 1):
                    xp = 0
                    for r in range(self.d + 1):
                        xp = xp + self.C[r, j] * Xk[r]
                    nb.eq((xp - h * f(Xk[j], uk)) / Sx)
                xend = 0
                for r in range(self.d + 1):
                    xend = xend + self.D[r] * Xk[r]
                nb.eq((X[:, k + 1] - xend) / Sx)
                pre = (name == "F" and k >= int(np.ceil((1.0 - cfg["prelanding_frac"]) * N)))
                for j in range(self.d + 1):
                    sj = Xk[j]
                    nb.con((sj[IWP] - sj[IWT]) / om_lim, -1.0, 1.0)
                    if pre and j > 0:
                        # pre-landing configuration: internal motion stopped before touchdown
                        nb.con((sj[IWP] - sj[IWT]) / cfg["prelanding_wrel"], -1.0, 1.0)
                    nb.con(fa(sj, uk) / al_lim, -1.0, 1.0)
                    if name == "L" and cfg.get("launch_mode_robust_accel", False):
                        # contact may open during the launch: acceleration limit must also
                        # hold for the free-flight inertia (mode-robust constraint)
                        nb.con(self.F["alpha_flight"](sj, uk) / al_lim, -1.0, 1.0)
                    if name == "A":
                        # approach: pendulum below the sphere centre (rolling-drive regime)
                        nb.con((sj[IPHI] + np.pi / 2) / cfg["approach_phi_halfwidth"], -1.0, 1.0)
                    if kind == "stance":
                        nb.con(self.F["lamN"](sj, uk) / mg, cfg["lamN_min_frac"][name], np.inf)
                        nb.con((x_e - R - dlt_r - sj[IX]) / R, 0.0, np.inf)
                    else:
                        ct, cb, cc = clearance_terms(sj[IX], sj[IY], R, h_lo, h_hi, x_e, dlt_c)
                        nb.con(ca.vertcat(ct / R, cb / R ** 2, cc / R ** 2), 0.0, np.inf)
                J = J + w["effort"] * h * (uk / tau_max) ** 2
                if name == "A":
                    gam = Xk[0][IVX] + R * Xk[0][IWT]
                    J = J + w["approach_slip"] * h * (gam / cfg["v_ref"]) ** 2
            if X_prev_end is not None:
                nb.eq((X[:, 0] - X_prev_end) / Sx)
            X_prev_end = X[:, N]
            u_all.append(Us)
            self.sym[name] = dict(T=T, X=X, U=U)
            J = J + w["time"] * T
        XA = self.sym["A"]["X"]
        nb.eq((XA[:, 0] - p[P_S0]) / Sx)
        XL = self.sym["L"]["X"]
        UF = self.sym["F"]["U"]
        nb.con(self.F["lamN"](XL[:, -1], UF[0, 0]) / mg, -np.inf, 0.0)
        sT = self.sym["F"]["X"][:, -1]
        nb.eq((sT[IY] - (h_hi + R)) / R)
        nb.con(sT[IVY], -np.inf, -cfg["touchdown_vy_min"])
        nb.con(sT[IX] - (x_e + cfg["landing_min_past_edge"]), 0.0, np.inf)
        nb.con(p[P_XLMAX] - sT[IX], 0.0, np.inf)
        gam_td = sT[IVX] + R * sT[IWT]
        J = J + w["landing_x"] * ((sT[IX] - p[P_XL]) / cfg["L_ref"]) ** 2
        J = J + w["landing_vy"] * (sT[IVY] / cfg["v_ref"]) ** 2
        J = J + w["landing_slip"] * (gam_td / cfg["v_ref"]) ** 2
        # no centrifugal unloading after touchdown: m_p l phid^2 <= eta m g
        phid_land = np.sqrt(cfg["landing_unload_eta"] * mg / (P.m_p * P.l))
        nb.con(sT[IWP] / phid_land, -1.0, 1.0)
        # residual internal energy of the pendulum at touchdown (to be dissipated)
        E_res = 0.5 * P.m_p * P.l ** 2 * (sT[IWP] - sT[IWT]) ** 2 + P.m_p * P.g * P.l * (1 + ca.sin(sT[IPHI]))
        J = J + w["landing_energy"] * E_res / (P.m_p * P.g * P.l)
        Uc = ca.horzcat(*u_all)
        J = J + w["dtau"] * (Uc[0, 0] - p[P_TAUPREV] / tau_max) ** 2
        J = J + w["dtau"] * ca.sumsqr(Uc[0, 1:] - Uc[0, :-1])
        opts = {"ipopt.print_level": 0, "print_time": False,
                "ipopt.max_iter": int(cfg["max_iter"]),
                "ipopt.tol": float(cfg["tol"]),
                "ipopt.acceptable_tol": 1e-4,
                "ipopt.acceptable_iter": 10,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.nlp_scaling_method": cfg.get("nlp_scaling_method", "none"),
                "ipopt.max_cpu_time": float(cfg.get("max_cpu_time_s", 1e6)),
                "ipopt.sb": "yes"}
        self.solver = nb.build(J, p, "jump_planner", "ipopt", opts)

    # ------------------------------------------------------------------
    def initial_guess(self, s0, geo):
        """Physically motivated initial guess: accelerate, quarter-turn swing, ballistic arc."""
        P = self.P
        R = P.R
        x0 = s0[IX]
        y0 = geo["h_lo"] + R
        x_to = max(x0 + 0.05, geo["x_e"] - R - 0.15)
        v_to = 1.5
        T_A = float(np.clip(2.0 * (x_to - x0) / max(v_to + abs(s0[IVX]), 0.2), 0.3, 2.5))
        T_L = 0.15
        dh = geo["h_hi"] - geo["h_lo"]
        vy = np.sqrt(2 * P.g * (dh + 0.1))
        T_F = vy / P.g + np.sqrt(2 * 0.1 / P.g)
        g = {}
        N = self.phases[0][2]
        t = np.linspace(0, 1, N + 1)
        Xa = np.zeros((NX, N + 1))
        Xa[IX] = x0 + (x_to - x0) * t ** 2 + s0[IVX] * T_A * t * (1 - t)
        Xa[IY] = y0
        Xa[IPHI] = s0[IPHI] + (-np.pi / 2 - s0[IPHI]) * t
        Xa[IVX] = s0[IVX] + (v_to - s0[IVX]) * t
        Xa[IWT] = -Xa[IVX] / R
        g["A"] = (T_A, Xa, 0.2 * np.ones(N))
        N = self.phases[1][2]
        t = np.linspace(0, 1, N + 1)
        Xl = np.zeros((NX, N + 1))
        Xl[IX] = x_to + v_to * T_L * t
        Xl[IY] = y0
        Xl[IPHI] = Xa[IPHI, -1] + (np.pi / 2 + 0.3) * t ** 2
        Xl[IVX] = v_to
        Xl[IWP] = 2 * (np.pi / 2 + 0.3) * t / T_L
        Xl[IWT] = -v_to / R
        Ul = np.where(np.linspace(0, 1, N) < 0.8, 0.8, -0.8)
        g["L"] = (T_L, Xl, Ul)
        N = self.phases[2][2]
        t = np.linspace(0, 1, N + 1)
        tt = t * T_F
        Xf = np.zeros((NX, N + 1))
        x_to2 = Xl[IX, -1]
        Xf[IX] = x_to2 + (geo["x_L"] - x_to2) * t
        Xf[IY] = y0 + vy * tt - 0.5 * P.g * tt ** 2
        Xf[IY, -1] = geo["h_hi"] + R
        Xf[IVX] = (geo["x_L"] - x_to2) / T_F
        Xf[IVY] = vy - P.g * tt
        wp0 = Xl[IWP, -1]
        Xf[IWP] = wp0 * (1 - t)
        Xf[IPHI] = Xl[IPHI, -1] + wp0 * T_F * (t - t ** 2 / 2)
        Xf[IWT] = -v_to / R
        g["F"] = (T_F, Xf, -0.2 * np.ones(N))
        return g

    def _w0_from_guess(self, guess):
        nb = self.nb
        w0 = nb.w0_v.copy()
        for name, _, N in self.phases:
            T, X, U = guess[name]
            nb.put(w0, f"T_{name}", T)
            nb.put(w0, f"X_{name}", X / self.Sx[:, None])
            Xc = np.zeros((NX, N * self.d))
            for k in range(N):
                for j in range(self.d):
                    a = self.tau_c[j + 1]
                    Xc[:, k * self.d + j] = (1 - a) * X[:, k] + a * X[:, k + 1]
            nb.put(w0, f"Xc_{name}", Xc / self.Sx[:, None])
            nb.put(w0, f"U_{name}", np.clip(np.asarray(U, float).reshape(1, -1), -1, 1))
        return w0

    # ------------------------------------------------------------------
    def solve(self, s0, geo, tau_prev=0.0, guess=None):
        """geo: dict(x_e, h_lo, h_hi, x_L, x_land_max). Returns plan dict."""
        s0 = np.asarray(s0, float).copy()
        s0[IY] = geo["h_lo"] + self.P.R      # project onto the tread (stance manifold)
        s0[IVY] = 0.0
        # the model is 2*pi-periodic in phi: plan in the revolution closest to hanging
        k_wrap = float(np.round((s0[IPHI] + np.pi / 2) / (2 * np.pi)))
        s0[IPHI] -= 2 * np.pi * k_wrap
        pv = np.zeros(NP)
        pv[P_S0] = s0
        pv[P_XE], pv[P_HLO], pv[P_HHI] = geo["x_e"], geo["h_lo"], geo["h_hi"]
        pv[P_XL], pv[P_XLMAX], pv[P_TAUPREV] = geo["x_L"], geo["x_land_max"], tau_prev
        nb = self.nb
        w0 = self._w0_from_guess(guess if guess is not None else self.initial_guess(s0, geo))
        Xa = nb.get(w0, "X_A")
        Xa[:, 0] = s0 / self.Sx    # initial node consistent with the measurement
        nb.put(w0, "X_A", Xa)
        t0 = time.perf_counter()
        sol = self.solver(x0=w0, p=pv, lbx=nb.lbw_v, ubx=nb.ubw_v, lbg=nb.lbg_v, ubg=nb.ubg_v)
        wall = time.perf_counter() - t0
        st = self.solver.stats()
        wopt = np.array(sol["x"]).ravel()
        plan = self._extract(wopt)
        plan["X"][IPHI] += 2 * np.pi * k_wrap   # back to the measurement's angle frame
        for ph in plan["phases"]:
            ph["X"][IPHI] += 2 * np.pi * k_wrap
        s0[IPHI] += 2 * np.pi * k_wrap
        plan.update(dict(success=bool(st["success"]), return_status=st["return_status"],
                         iter_count=int(st["iter_count"]), solve_time=wall,
                         s0=s0, geo=dict(geo)))
        return plan

    def _extract(self, wopt):
        nb = self.nb
        out = {"phases": []}
        t0 = 0.0
        ts, Xs, Us, tus = [], [], [], []
        for name, _, N in self.phases:
            T = float(nb.get(wopt, f"T_{name}")[0, 0])
            X = nb.get(wopt, f"X_{name}") * self.Sx[:, None]
            U = nb.get(wopt, f"U_{name}").ravel() * self.lim["tau_max"]
            t = t0 + np.linspace(0, T, N + 1)
            out["phases"].append(dict(name=name, t0=t0, T=T, X=X, U=U))
            ts.append(t if not ts else t[1:])
            Xs.append(X if not Xs else X[:, 1:])
            Us.append(U)
            tus.append(t[:-1])
            t0 += T
        out["t"] = np.concatenate(ts)
        out["X"] = np.concatenate(Xs, axis=1)
        out["t_u"] = np.concatenate(tus)
        out["U"] = np.concatenate(Us)
        out["t_takeoff"] = out["phases"][1]["t0"] + out["phases"][1]["T"]
        out["t_touchdown"] = t0
        return out


# ---------------------------------------------------------------------------
#  Level 2: phase-dependent NMPC (tracking, flight, settle)  (hoop project: src/nmpc.py)
# ---------------------------------------------------------------------------
# Level 2 - phase-dependent nonlinear MPC.
#
# TrackingNMPC  (stance phases: approach/launch tracking, post-impact
#               stabilization). Uniform grid of N intervals of length Ts, Radau
#               IIA collocation, a per-interval phase flag sigma_k in {0,1}
#               (0 = stance on the current tread, 1 = flight) supplied by the
#               contact schedule of the reference (Di Carlo et al., IROS 2018;
#               Nguyen et al., CDC 2022, Eq. (7d)). Cost: scaled tracking error
#               + deviation from the reference (feedforward) torque + torque
#               increments + dominant terminal weight (Nguyen et al. 2022;
#               Qin & Badgwell 2003, Eq. (31)). Torque bounds are hard; state
#               and mixed constraints are softened with exact l1 + quadratic
#               slack penalties (Qin & Badgwell 2003, Sec. 3.4.6).
#
# FlightNMPC    (ballistic phase). The CoM motion is fixed by the take-off
#               state; only the internal (pendulum/shell-spin) motion can be
#               changed. Free-final-time shrinking-horizon OCP from the
#               measured state to touchdown on the target tread (Diehl, Bock &
#               Schloeder, CDC-ECC 2005: shrinking-horizon optimal feedback),
#               with the planner's landing objectives and constraints.
#
# Both are re-solved from the measured state at every sample with the
# previous solution shifted by one interval as initial guess (Diehl et al.
# 2005), using IPOPT (Waechter & Biegler 2006) via CasADi (Andersson et al.
# 2019).
def _ipopt_opts(cfg):
    return {"ipopt.print_level": 0, "print_time": False,
            "ipopt.max_iter": int(cfg["max_iter"]),
            "ipopt.tol": float(cfg["tol"]),
            "ipopt.acceptable_tol": float(cfg.get("acceptable_tol", 1e-4)),
            "ipopt.acceptable_iter": 5,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.warm_start_init_point": "yes" if cfg.get("warm_start_multipliers", False) else "no",
            "ipopt.mu_init": float(cfg.get("mu_init", 0.1)),
            "ipopt.nlp_scaling_method": cfg.get("nlp_scaling_method", "none"),
            "ipopt.sb": "yes"}


class TrackingNMPC:
    """Uniform-grid phase-scheduled tracking NMPC."""

    def __init__(self, P, cfg, limits, scales):
        self.P, self.cfg, self.lim = P, cfg, limits
        self.F = make_functions(P)
        self.N = int(cfg["N"])
        self.Ts = float(cfg["Ts"])
        self.d = int(cfg["collocation_degree"])
        self.tau_c, self.C, self.D = radau_coefficients(self.d)
        self.Sx = np.array([scales[k] for k in ("x", "y", "phi", "vx", "vy", "thetad", "phid")], float)
        self._build()
        self.w_prev = None

    def _build(self):
        P, cfg, lim = self.P, self.cfg, self.lim
        N, d, Ts = self.N, self.d, self.Ts
        nb = NLPBuilder()
        self.nb = nb
        Sx = ca.DM(self.Sx)
        tau_max = lim["tau_max"]
        om_lim = lim["omega_max"] * cfg["speed_margin"]
        al_lim = lim["alpha_max"] * cfg["accel_margin"]
        mg = P.m * P.g
        R = P.R
        # ---- parameters
        s0 = ca.SX.sym("s0", NX)
        Xref = ca.SX.sym("Xref", NX, N + 1)
        Uref = ca.SX.sym("Uref", 1, N)
        sig = ca.SX.sym("sig", 1, N)        # 1 = flight interval, 0 = stance
        lift = ca.SX.sym("lift", 1, N)      # 1 at the first flight interval (lift-off)
        act = ca.SX.sym("act", 1, N)        # 0 masks intervals after the reference touchdown
        rob = ca.SX.sym("rob", 1, N)        # 1: launch interval -> mode-robust acceleration limit
        Wq = ca.SX.sym("Wq", NX)            # 1/scale^2 tracking weights
        Wqf = ca.SX.sym("Wqf", NX)
        geo = ca.SX.sym("geo", 6)           # x_e, h_lo, h_hi, x_min, x_max, lamN_min
        tau_prev = ca.SX.sym("tau_prev")
        p = ca.vertcat(s0, ca.vec(Xref), ca.vec(Uref), ca.vec(sig), ca.vec(lift), ca.vec(act),
                       ca.vec(rob), Wq, Wqf, geo, tau_prev)
        x_e, h_lo, h_hi, x_min, x_max, lam_min = [geo[i] for i in range(6)]
        # ---- variables
        Xs = nb.var("X", (NX, N + 1))
        Xcs = nb.var("Xc", (NX, N * d))
        Us = nb.var("U", (1, N), -1.0, 1.0)
        n_soft = 7  # slacks: speed, accel, lamN, clearance, x-box, lift-off, pendulum elevation
        elev = cfg.get("max_pendulum_elevation_rad", None)
        Ss = nb.var("S", (n_soft, N), 0.0, np.inf)
        X = ca.diag(Sx) @ Xs
        Xc = ca.diag(Sx) @ Xcs
        U = tau_max * Us
        J = 0
        nb.eq((X[:, 0] - s0) / Sx)
        w = cfg["weights"]
        for k in range(N):
            Xk = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]
            uk = U[0, k]
            sg = sig[0, k]
            for j in range(1, d + 1):
                xp = 0
                for r in range(d + 1):
                    xp = xp + self.C[r, j] * Xk[r]
                fdyn = sg * self.F["flight"](Xk[j], uk) + (1 - sg) * self.F["stance"](Xk[j], uk)
                nb.eq((xp - Ts * fdyn) / Sx)
            xend = 0
            for r in range(d + 1):
                xend = xend + self.D[r] * Xk[r]
            nb.eq((X[:, k + 1] - xend) / Sx)
            sk = Ss[:, k]
            ak = act[0, k]
            for j in range(d + 1):
                sj = Xk[j]
                wr = (sj[IWP] - sj[IWT]) / om_lim
                nb.con(wr + sk[0], -1.0, np.inf)
                nb.con(wr - sk[0], -np.inf, 1.0)
                alpha = sg * self.F["alpha_flight"](sj, uk) + (1 - sg) * self.F["alpha_stance"](sj, uk)
                nb.con(alpha / al_lim + sk[1], -1.0, np.inf)
                nb.con(alpha / al_lim - sk[1], -np.inf, 1.0)
                arob = rob[0, k] * (1 - sg) * self.F["alpha_flight"](sj, uk) / al_lim
                nb.con(arob + sk[1], -1.0, np.inf)
                nb.con(arob - sk[1], -np.inf, 1.0)
                # stance: unilateral contact lamN >= lamN_min (inactive in flight)
                lam = self.F["lamN"](sj, uk) / mg
                nb.con((1 - sg) * (lam - lam_min) + sg + sk[2], 0.0, np.inf)
                # flight: stair clearance (inactive in stance / masked intervals)
                ct, cb, cc = clearance_terms(sj[IX], sj[IY], R, h_lo, h_hi, x_e, cfg["corner_margin"])
                on = sg * ak
                nb.con(on * ct / R + (1 - on) + sk[3], 0.0, np.inf)
                nb.con(on * cb / R ** 2 + (1 - on) + sk[3], 0.0, np.inf)
                nb.con(on * cc / R ** 2 + (1 - on) + sk[3], 0.0, np.inf)
                if elev is not None:
                    # pendulum kept at most `elev` above the horizontal (pendulum-driven rolling
                    # regime); keeps the stabilizing NMPC away from the inverted equilibrium
                    nb.con(np.sin(elev) - ca.sin(sj[IPHI]) + sk[6], 0.0, np.inf)
                # stance: stay on the tread between the risers
                nb.con((1 - sg) * (sj[IX] - x_min) / R + sg + sk[4], 0.0, np.inf)
                nb.con((1 - sg) * (x_max - sj[IX]) / R + sg + sk[4], 0.0, np.inf)
            # lift-off consistency at the first flight interval
            lamL = self.F["lamN"](Xk[0], uk) / mg
            nb.con(lift[0, k] * lamL - sk[5], -np.inf, 0.0)
            # cost
            e = X[:, k] - Xref[:, k]
            J = J + ak * ca.dot(Wq, e * e)
            J = J + ak * w["tau_ref"] * ((uk - Uref[0, k]) / cfg["tau_dev_scale"]) ** 2
            J = J + w["effort"] * (uk / tau_max) ** 2
            du = uk - (tau_prev if k == 0 else U[0, k - 1])
            J = J + w["dtau"] * (du / cfg["dtau_scale"]) ** 2
            J = J + cfg["slack_l1"] * ca.sum1(sk) + cfg["slack_l2"] * ca.sumsqr(sk)
            if elev is not None:
                # extra l1 weight on the elevation slack, large enough to make it an exact
                # penalty: with the shared slack_l1 the position-tracking gain outbid it and
                # the settle NMPC planned deliberate over-the-top pendulum swings (sphere,
                # backing up on a 1.0 m tread -> settle limit cycle)
                J = J + cfg.get("elev_slack_l1", 0.0) * sk[6]
        # terminal cost: last active node
        eN = 0
        for k in range(1, N + 1):
            a_k = act[0, k - 1] * (1 - (act[0, k] if k < N else 0))  # 1 at the last active node
            eN_k = X[:, k] - Xref[:, k]
            eN = eN + a_k * ca.dot(Wqf, eN_k * eN_k)
        J = J + eN
        self.solver = nb.build(J, p, "tracking_nmpc", "ipopt", _ipopt_opts(cfg))

    # --------------------------------------------------------------
    def pack(self, s0, Xref, Uref, sig, lift, act, Wq, Wqf, geo, tau_prev, rob=None):
        rob = np.zeros(self.N) if rob is None else rob
        return np.concatenate([np.asarray(s0, float).ravel(),
                               np.asarray(Xref, float).ravel(order="F"),
                               np.asarray(Uref, float).ravel(),
                               np.asarray(sig, float).ravel(), np.asarray(lift, float).ravel(),
                               np.asarray(act, float).ravel(), np.asarray(rob, float).ravel(),
                               np.asarray(Wq, float), np.asarray(Wqf, float),
                               np.asarray(geo, float), [float(tau_prev)]])

    def guess_from_ref(self, Xref, Uref):
        nb = self.nb
        w0 = nb.w0_v.copy()
        nb.put(w0, "X", Xref / self.Sx[:, None])
        Xc = np.zeros((NX, self.N * self.d))
        for k in range(self.N):
            for j in range(self.d):
                a = self.tau_c[j + 1]
                Xc[:, k * self.d + j] = (1 - a) * Xref[:, k] + a * Xref[:, k + 1]
        nb.put(w0, "Xc", Xc / self.Sx[:, None])
        nb.put(w0, "U", np.clip(np.asarray(Uref).reshape(1, -1) / self.lim["tau_max"], -1, 1))
        nb.put(w0, "S", 0.0)
        return w0

    def shifted_guess(self, s0):
        """Shift the previous solution by one interval (Diehl et al. 2005)."""
        nb, d = self.nb, self.d
        w = self.w_prev
        X = nb.get(w, "X")
        Xc = nb.get(w, "Xc")
        U = nb.get(w, "U")
        S = nb.get(w, "S")
        w0 = nb.w0_v.copy()
        Xn = np.hstack([X[:, 1:], X[:, -1:]])
        Xn[:, 0] = s0 / self.Sx
        nb.put(w0, "X", Xn)
        nb.put(w0, "Xc", np.hstack([Xc[:, d:], np.repeat(X[:, -1:], d, axis=1)]))
        nb.put(w0, "U", np.hstack([U[:, 1:], U[:, -1:]]))
        nb.put(w0, "S", np.hstack([S[:, 1:], S[:, -1:]]))
        return w0

    def solve(self, pvec, w0, use_duals=True):
        nb = self.nb
        kw = {}
        if use_duals and self.cfg.get("warm_start_multipliers", False) and getattr(self, "lam_prev", None):
            kw = dict(lam_x0=self.lam_prev[0], lam_g0=self.lam_prev[1])
        sol = self.solver(x0=w0, p=pvec, lbx=nb.lbw_v, ubx=nb.ubw_v, lbg=nb.lbg_v, ubg=nb.ubg_v, **kw)
        wopt = np.array(sol["x"]).ravel()
        self.w_prev = wopt
        self.lam_prev = (np.array(sol["lam_x"]).ravel(), np.array(sol["lam_g"]).ravel())
        U = nb.get(wopt, "U").ravel() * self.lim["tau_max"]
        f = float(sol["f"])
        return dict(U=U, f=f if np.isfinite(f) else np.inf)


class FlightNMPC:
    """Free-final-time shrinking-horizon OCP from the current flight state to touchdown."""

    def __init__(self, P, cfg, limits, scales, planner_cfg):
        self.P, self.cfg, self.lim, self.pcfg = P, cfg, limits, planner_cfg
        self.F = make_functions(P)
        self.N = int(cfg["N"])
        self.d = int(cfg["collocation_degree"])
        self.tau_c, self.C, self.D = radau_coefficients(self.d)
        self.Sx = np.array([scales[k] for k in ("x", "y", "phi", "vx", "vy", "thetad", "phid")], float)
        self._build()
        self.w_prev = None

    def _build(self):
        P, cfg, lim, pc = self.P, self.cfg, self.lim, self.pcfg
        N, d = self.N, self.d
        nb = NLPBuilder()
        self.nb = nb
        Sx = ca.DM(self.Sx)
        tau_max = lim["tau_max"]
        om_lim = lim["omega_max"] * cfg["speed_margin"]
        al_lim = lim["alpha_max"] * cfg["accel_margin"]
        mg = P.m * P.g
        R = P.R
        s0 = ca.SX.sym("s0", NX)
        geo = ca.SX.sym("geo", 5)   # x_e, h_lo, h_hi, x_L, h_td (tread to land on)
        tau_prev = ca.SX.sym("tau_prev")
        p = ca.vertcat(s0, geo, tau_prev)
        x_e, h_lo, h_hi, x_L, h_td = [geo[i] for i in range(5)]
        T = nb.var("T", (1, 1), cfg["T_min"], cfg["T_max"], 0.2)
        Xs = nb.var("X", (NX, N + 1))
        Xcs = nb.var("Xc", (NX, N * d))
        Us = nb.var("U", (1, N), -1.0, 1.0)
        Ss = nb.var("S", (5, N), 0.0, np.inf)
        St = nb.var("St", (2, 1), 0.0, np.inf)
        k_pre = int(np.ceil((1.0 - pc["prelanding_frac"]) * N))
        X = ca.diag(Sx) @ Xs
        Xc = ca.diag(Sx) @ Xcs
        U = tau_max * Us
        h = T / N
        J = 0
        w = pc["weights"]
        nb.eq((X[:, 0] - s0) / Sx)
        for k in range(N):
            Xk = [X[:, k]] + [Xc[:, k * d + j] for j in range(d)]
            uk = U[0, k]
            for j in range(1, d + 1):
                xp = 0
                for r in range(d + 1):
                    xp = xp + self.C[r, j] * Xk[r]
                nb.eq((xp - h * self.F["flight"](Xk[j], uk)) / Sx)
            xend = 0
            for r in range(d + 1):
                xend = xend + self.D[r] * Xk[r]
            nb.eq((X[:, k + 1] - xend) / Sx)
            sk = Ss[:, k]
            for j in range(d + 1):
                sj = Xk[j]
                wr = (sj[IWP] - sj[IWT]) / om_lim
                nb.con(wr + sk[0], -1.0, np.inf)
                nb.con(wr - sk[0], -np.inf, 1.0)
                alpha = self.F["alpha_flight"](sj, uk) / al_lim
                nb.con(alpha + sk[1], -1.0, np.inf)
                nb.con(alpha - sk[1], -np.inf, 1.0)
                if j > 0 and k >= k_pre:
                    wp_pre = (sj[IWP] - sj[IWT]) / pc["prelanding_wrel"]
                    nb.con(wp_pre + sk[4], -1.0, np.inf)
                    nb.con(wp_pre - sk[4], -np.inf, 1.0)
                if j > 0:
                    ct, cb, cc = clearance_terms(sj[IX], sj[IY], R, h_lo, h_hi, x_e, pc["corner_margin"])
                    # clearance is required only while above the landing tread
                    nb.con(cb / R ** 2 + sk[2], 0.0, np.inf)
                    nb.con(cc / R ** 2 + sk[2], 0.0, np.inf)
                    nb.con(ct / R + sk[3], 0.0, np.inf)
            J = J + w["effort"] * h * (uk / tau_max) ** 2
            du = uk - (tau_prev if k == 0 else U[0, k - 1])
            J = J + pc["weights"]["dtau"] * (du / tau_max) ** 2
            J = J + cfg["slack_l1"] * ca.sum1(sk) + cfg["slack_l2"] * ca.sumsqr(sk)
        sT = X[:, N]
        nb.eq((sT[IY] - (h_td + R)) / R)
        nb.con(sT[IVY] + St[0], -np.inf, -pc["touchdown_vy_min"])
        phid_land = np.sqrt(pc["landing_unload_eta"] * mg / (P.m_p * P.l))
        nb.con(sT[IWP] / phid_land + St[1], -1.0, np.inf)
        nb.con(sT[IWP] / phid_land - St[1], -np.inf, 1.0)
        J = J + cfg["slack_l1"] * ca.sum1(St) + cfg["slack_l2"] * ca.sumsqr(St)
        gam_td = sT[IVX] + R * sT[IWT]
        J = J + w["landing_x"] * ((sT[IX] - x_L) / pc["L_ref"]) ** 2
        J = J + w["landing_vy"] * (sT[IVY] / pc["v_ref"]) ** 2
        J = J + w["landing_slip"] * (gam_td / pc["v_ref"]) ** 2
        E_res = 0.5 * P.m_p * P.l ** 2 * (sT[IWP] - sT[IWT]) ** 2 + P.m_p * P.g * P.l * (1 + ca.sin(sT[IPHI]))
        J = J + w["landing_energy"] * E_res / (P.m_p * P.g * P.l)
        self.solver = nb.build(J, p, "flight_nmpc", "ipopt", _ipopt_opts(cfg))

    def guess_ballistic(self, s0, h_td):
        """Initial guess: ballistic CoM until the sphere centre reaches h_td + R."""
        P = self.P
        yc = s0[IY] + P.m_p / P.m * P.l * np.sin(s0[IPHI])
        vyc = s0[IVY] + P.m_p / P.m * P.l * np.cos(s0[IPHI]) * s0[IWP]
        disc = vyc ** 2 + 2 * P.g * max(yc - (h_td + P.R), 0.0)
        T = max((vyc + np.sqrt(max(disc, 0.0))) / P.g, self.cfg["T_min"] * 1.01)
        T = min(T, self.cfg["T_max"] * 0.99)
        N, d = self.N, self.d
        t = np.linspace(0, T, N + 1)
        X = np.zeros((NX, N + 1))
        X[IX] = s0[IX] + s0[IVX] * t
        X[IY] = s0[IY] + s0[IVY] * t - 0.5 * P.g * t ** 2
        X[IY, -1] = h_td + P.R
        X[IVX] = s0[IVX]
        X[IVY] = s0[IVY] - P.g * t
        X[IPHI] = s0[IPHI] + s0[IWP] * t
        X[IWP] = s0[IWP]
        X[IWT] = s0[IWT]
        nb = self.nb
        w0 = nb.w0_v.copy()
        nb.put(w0, "T", T)
        nb.put(w0, "X", X / self.Sx[:, None])
        Xc = np.zeros((NX, N * d))
        for k in range(N):
            for j in range(d):
                a = self.tau_c[j + 1]
                Xc[:, k * d + j] = (1 - a) * X[:, k] + a * X[:, k + 1]
        nb.put(w0, "Xc", Xc / self.Sx[:, None])
        nb.put(w0, "U", 0.0)
        nb.put(w0, "S", 0.0)
        nb.put(w0, "St", 0.0)
        return w0

    def guess_from_traj(self, s0, t_traj, X_traj, t_u, U_traj, T):
        """Initial guess sampled from a reference flight trajectory (the plan)."""
        N, d = self.N, self.d
        T = float(np.clip(T, self.cfg["T_min"] * 1.01, self.cfg["T_max"] * 0.99))
        tg = np.linspace(0.0, T, N + 1)
        X = np.array([np.interp(tg, t_traj, X_traj[i]) for i in range(NX)])
        X[:, 0] = s0
        tc = tg[:-1] + 0.5 * T / N
        U = np.array([U_traj[min(max(int(np.searchsorted(t_u, v, side="right")) - 1, 0), len(U_traj) - 1)]
                      for v in tc])
        nb = self.nb
        w0 = nb.w0_v.copy()
        nb.put(w0, "T", T)
        nb.put(w0, "X", X / self.Sx[:, None])
        Xc = np.zeros((NX, N * d))
        for k in range(N):
            for j in range(d):
                a = self.tau_c[j + 1]
                Xc[:, k * d + j] = (1 - a) * X[:, k] + a * X[:, k + 1]
        nb.put(w0, "Xc", Xc / self.Sx[:, None])
        nb.put(w0, "U", np.clip(U / self.lim["tau_max"], -1, 1).reshape(1, -1))
        nb.put(w0, "S", 0.0)
        nb.put(w0, "St", 0.0)
        return w0

    def solve(self, s0, geo, tau_prev, w0, use_duals=True):
        nb = self.nb
        pv = np.concatenate([s0, geo, [tau_prev]])
        kw = {}
        if use_duals and self.cfg.get("warm_start_multipliers", False) and getattr(self, "lam_prev", None):
            kw = dict(lam_x0=self.lam_prev[0], lam_g0=self.lam_prev[1])
        sol = self.solver(x0=w0, p=pv, lbx=nb.lbw_v, ubx=nb.ubw_v, lbg=nb.lbg_v, ubg=nb.ubg_v, **kw)
        wopt = np.array(sol["x"]).ravel()
        self.w_prev = wopt
        self.lam_prev = (np.array(sol["lam_x"]).ravel(), np.array(sol["lam_g"]).ravel())
        T = float(nb.get(wopt, "T")[0, 0])
        U = nb.get(wopt, "U").ravel() * self.lim["tau_max"]
        return dict(t=np.linspace(0, T, self.N + 1), U=U)

    def shifted_guess(self, s0, dt):
        """Warm start: previous solution with its horizon shortened by the elapsed time."""
        nb = self.nb
        w0 = self.w_prev.copy()
        T = float(nb.get(w0, "T")[0, 0])
        nb.put(w0, "T", float(np.clip(T - dt, self.cfg["T_min"] * 1.01, self.cfg["T_max"] * 0.99)))
        X = nb.get(w0, "X")
        X[:, 0] = s0 / self.Sx
        nb.put(w0, "X", X)
        return w0


# ---------------------------------------------------------------------------
#  Level 3 actuator saturation + controller with phase state machine  (hoop project: src/controller.py)
# ---------------------------------------------------------------------------
# Stair-jumping hierarchical MPC controller for the spherical robot.
#
#     reference generation (JumpPlanner, event-triggered, once per jump)
#         -> phase-dependent NMPC (TrackingNMPC in stance, FlightNMPC in flight)
#         -> actuator saturation layer (torque, acceleration, speed)
#         -> pendulum (alpha_p) motor torque
#
# The class implements the same ``stair_controller`` interface as
# StairNMPCController: ``alpha_tau_func`` / ``beta_tau_func`` (pure functions
# of (t, q, u), evaluated inside every Newton iteration),
# ``update_after_step(t, q, u)`` (called once per converged step; all
# estimation, phase logic and optimization happens here),
# ``get_plan_metadata()``, ``event_log`` and the recorded attributes
# (controller_state, current_stair_index, next_stair_edge, desired_prejump_x,
# landing_target_x, omega_spin, suggested_start_x).
#
# State information: the simulator's converged 3D state (q, u) is mapped to
# the planar state s (full-state measurement); the contact phase is estimated
# from the gap between the sphere and the tread below its centre (no contact
# forces are read).
# solver unit conversion (the simulator is nondimensional: l_nd = 1 m, m_nd = 1 kg, a_nd = 9.81 m/s^2)
G_ND = 9.81
T_ND = np.sqrt(1.0 / G_ND)
TAU_SCALE = 9.81  # N*m per solver torque unit


class ActuatorLimiter:
    """Saturation of torque and motor (relative) acceleration.

    tau_cmd -> clip to [-tau_max, tau_max]
            -> clip to [tau_lo, tau_hi] so that |phi_dd - theta_dd| <= alpha_max,
               where the affine map tau -> alpha_rel is evaluated from the
               phase model (stance/flight) at the last converged state.
    The motor speed limit is enforced by the optimizers only (planner 0.95,
    NMPCs 0.97 omega_max, soft); the former speed taper was removed after it
    showed no measurable effect (motor speed <= 53.6 rad/s without it).
    The bounds are refreshed once per converged step, so tau(t, q, u) stays a
    pure, continuous function of the current Newton iterate.
    """

    def __init__(self, P, tau_max, alpha_max, unload_frac=1.0):
        self.tau_max = float(tau_max)
        self.alpha_max = float(alpha_max)
        self.unload_frac = float(unload_frac)
        self.mg = P.m * P.g
        F = make_functions(P)
        self._af = F["alpha_flight"]
        self._as = F["alpha_stance"]
        self._lamN = F["lamN"]
        self.lo, self.hi = -self.tau_max, self.tau_max

    def _bounds(self, fa, s):
        a0 = float(fa(s, 0.0))
        b = float(fa(s, 1.0)) - a0
        if b > 1e-9:
            return (-self.alpha_max - a0) / b, (self.alpha_max - a0) / b
        return -self.tau_max, self.tau_max

    def refresh(self, s, in_contact):
        if in_contact:
            lo, hi = self._bounds(self._as, s)
            # mode-robust: if the stance normal force at a candidate torque is below the
            # static weight (contact unloading -> it may open within the step), the
            # acceleration limit must also hold for the free-flight inertia
            lam = min(float(self._lamN(s, max(lo, -self.tau_max))),
                      float(self._lamN(s, min(hi, self.tau_max))), float(self._lamN(s, 0.0)))
            if lam < self.unload_frac * self.mg:
                flo, fhi = self._bounds(self._af, s)
                lo, hi = max(lo, flo), min(hi, fhi)
        else:
            lo, hi = self._bounds(self._af, s)
        lo, hi = max(lo, -self.tau_max), min(hi, self.tau_max)
        if lo > hi:  # bounds incompatible: use the torque closest to both
            m = float(np.clip(0.5 * (lo + hi), -self.tau_max, self.tau_max))
            lo = hi = m
        self.lo, self.hi = lo, hi

    def apply(self, tau_cmd):
        tau = min(max(tau_cmd, -self.tau_max), self.tau_max)
        return min(max(tau, self.lo), self.hi)


class StairJumpMPCController:
    """Hierarchical stair-jumping MPC on the planar flight/stance model. Torque only.

    beta_p is a free DOF driven by the lateral motor's PD servo to beta = 0
    (make_beta_servo), so the motion stays planar; Simulation
    beta_mode="constraint" instead holds it at 0 with a bilateral constraint.
    `plan_at_start` asks Simulation.solve() for one update at t = 0, so the
    first jump is planned from the initial rest state (as in the hoop project).
    """
    plan_at_start = True

    def __init__(self, cfg, stair_edges, stair_heights, stair_width, t_nd=T_ND,
                 start_backoff_m=1.0, verbose=False,
                 beta_kp_Nm_per_rad=49.0, beta_kd_Nm_s_per_rad=2.5, tau_beta_max_Nm=1.0):
        if ca is None:
            raise ImportError("StairJumpMPCController needs CasADi: pip install casadi")
        self.cfg = cfg
        pc, ac = cfg["plant"], cfg["actuator"]
        self.P = PlanarSphereParams(R=pc["R"], m_sphere=pc["m_sphere"], m_pendulum=pc["m_pendulum"],
                                    l_pendulum=pc["l_pendulum"], g=pc["g"], mu=pc["mu_k"],
                                    eps_slip=cfg["model"]["eps_slip_m_s"])
        self.lim = dict(tau_max=ac["tau_max_Nm"], omega_max=ac["omega_max_rad_s"],
                        alpha_max=ac["alpha_max_rad_s2"])
        self.edges = np.asarray(stair_edges, float)
        self.heights = np.asarray(stair_heights, float)
        self.n_stairs = len(self.edges)
        self.W = float(stair_width)
        self.t_nd = float(t_nd)
        self.verbose = verbose
        sc = cfg["planner"]["scales"]
        self.planner = JumpPlanner(self.P, cfg["planner"], self.lim)
        self.track = TrackingNMPC(self.P, cfg["tracking_nmpc"], self.lim, sc)
        self.settle = TrackingNMPC(self.P, cfg["settle_nmpc"], self.lim, sc)
        self.flight = FlightNMPC(self.P, cfg["flight_nmpc"], self.lim, sc, cfg["planner"])
        self.act = ActuatorLimiter(self.P, self.lim["tau_max"], self.lim["alpha_max"], ac["unload_frac"])
        self.beta_gains = dict(kp_Nm_per_rad=float(beta_kp_Nm_per_rad),
                               kd_Nm_s_per_rad=float(beta_kd_Nm_s_per_rad),
                               tau_max_Nm=float(tau_beta_max_Nm))
        self._beta_servo = make_beta_servo(beta_kp_Nm_per_rad, beta_kd_Nm_s_per_rad,
                                           tau_beta_max_Nm, self.t_nd, TAU_SCALE)
        g = cfg["geometry_targets"]
        self.td_offset = float(g["touchdown_past_edge_m"])
        self.hold_offset = float(g["hold_past_edge_m"])
        self.sm = cfg["state_machine"]
        # Simulation-facing attributes
        self.suggested_start_x = float(self.edges[0] - start_backoff_m) if self.n_stairs else 0.0
        self.event_log = []
        self.current_stair_index = 0
        self.landing_target_x = self._td_target(0) if self.n_stairs else np.nan
        self.next_stair_edge = float(self.edges[0]) if self.n_stairs else np.nan
        self.desired_prejump_x = np.nan   # planned take-off x of the current jump
        self.omega_spin = np.nan          # planned peak |alpha_p_dot| (solver units)
        self.controller_state = "SETTLE"
        # internal state
        self._phi_prev = None             # for unwrapping the pendulum angle
        self.mode = "SETTLE"
        self.tau_cmd_fn = lambda t: 0.0
        self.tau_cmd_last = 0.0
        self.t_next = 0.0
        self.plan = None
        self.t_plan0 = 0.0
        self.contact_since = 0.0
        self.in_contact = True
        self.target = 0
        self.flight_w = None
        self.t_last_flight = None
        self.t_plan_retry = 0.0
        self._t_takeoff = None

    # ── geometry ───────────────────────────────────────────────
    def _support(self, x):
        return int(np.searchsorted(self.edges, x, side="right")) - 1

    def _tread_h(self, idx):
        return 0.0 if idx < 0 else float(self.heights[idx])

    def _td_target(self, j):
        return float(self.edges[j] + self.td_offset)

    def _hold_target(self, idx, x):
        """Rest position on the current tread; always leaves the minimum run-up."""
        base = x if idx < 0 else self.edges[idx] + self.hold_offset
        if idx + 1 < self.n_stairs:
            base = min(base, self.edges[idx + 1] - self.P.R - self.sm["min_runup_m"] - 0.1)
        return float(base)

    def _geo_for(self, j):
        h_lo = self._tread_h(j - 1)
        x_next_edge = self.edges[j + 1] if j + 1 < self.n_stairs else self.edges[j] + 1e3
        return dict(x_e=float(self.edges[j]), h_lo=h_lo, h_hi=float(self.heights[j]),
                    x_L=self._td_target(j),
                    x_land_max=float(min(x_next_edge, self.edges[j] + self.W) - self.P.R - 0.1))

    def _log(self, t, msg):
        self.event_log.append((float(t), msg))
        if self.verbose:
            print(f"[MPC t={t:.3f}s] {msg}")

    # ── metadata ───────────────────────────────────────────────
    def get_plan_metadata(self):
        return {"controller": "StairJumpMPCController",
                "method": "multi-phase TO reference + phase-dependent NMPC (CasADi/IPOPT)",
                "prediction_model": "planar flight/stance, thin shell I_s = 2/3 m R^2, "
                                    "regularized Coulomb friction",
                "beta_control": "lateral_motor_servo",
                "beta_servo": self.beta_gains,
                "tau_max_Nm": self.lim["tau_max"],
                "pendulum_speed_max_rad_s": self.lim["omega_max"],
                "pendulum_accel_max_rad_s2": self.lim["alpha_max"],
                "config": self.cfg}

    # ── measurement: 3D state -> planar state (SI) ────────────────
    def _planar_state(self, q, u):
        """s = [x, z, phi, vx, vz, theta_dot, phi_dot] from the simulator state.

        phi is the pendulum's world angle in the x-z plane (= alpha_p + theta),
        unwrapped against the previous converged step so that it stays
        continuous over full pendulum revolutions (as the hoop's free phi
        coordinate does); the planner and the settle NMPC handle the 2*pi
        offset themselves.
        """
        e_w = quat_to_R(q[3:7]) @ pendulum_dir_body(q[7], q[8])
        phi = float(np.arctan2(e_w[2], e_w[0]))
        if self._phi_prev is not None:
            phi = self._phi_prev + (phi - self._phi_prev + np.pi) % (2 * np.pi) - np.pi
        self._phi_prev = phi
        theta_dot = -float(u[4])
        phi_dot = float(u[6]) + theta_dot
        return np.array([q[0], q[2], phi, u[0] / self.t_nd, u[2] / self.t_nd,
                         theta_dot / self.t_nd, phi_dot / self.t_nd], float)

    # ── main update (converged steps only) ─────────────────────
    def update_after_step(self, t_nd, q, u):
        t = float(t_nd) * self.t_nd
        s = self._planar_state(q, u)
        sup = self._support(s[IX])
        gap = s[IY] - (self._tread_h(sup) + self.P.R)
        was = self.in_contact
        if self.in_contact:
            self.in_contact = gap < self.sm["takeoff_gap_m"]
        else:
            self.in_contact = gap < self.sm["touchdown_gap_m"]
        if self.in_contact and not was:
            self.contact_since = t
        self._events(t, s, sup, was)
        # MPC sample
        if t >= self.t_next - 1e-9:
            self._mpc_sample(t, s, sup)
            Ts = self.cfg["tracking_nmpc"]["Ts"]
            self.t_next = (np.floor(t / Ts + 1e-6) + 1) * Ts
        self.act.refresh(s, self.in_contact)
        self.controller_state = self.mode

    def _events(self, t, s, sup, was_contact):
        sm = self.sm
        if self.mode == "TRACK" and was_contact and not self.in_contact:
            self.mode = "FLIGHT"
            self.flight_w = None
            self._t_takeoff = t
            self._log(t, f"take-off towards stair {self.target} at x={s[IX]:.3f}, vx={s[IVX]:.2f}")
            self.t_next = t  # re-solve immediately in the new phase
        if self.mode == "FLIGHT" and self.in_contact and not was_contact:
            self._t_takeoff = None
            self._log(t, f"touch-down on tread {sup} at x={s[IX]:.3f} "
                         f"({'target' if sup == self.target else 'NOT target'})")
            self.mode = "SETTLE"
            self.contact_since = t
            self.t_next = t
        if self.mode == "TRACK" and self.plan is not None:
            tau_ref = t - self.t_plan0
            if tau_ref > self.plan["t_takeoff"] + sm["launch_timeout_s"] and self.in_contact:
                self._log(t, "launch did not lift off within the timeout -> stabilize and re-plan")
                self.mode = "SETTLE"
                self.contact_since = t
                self.t_next = t
        if self.mode == "SETTLE":
            dphi = (s[IPHI] + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
            calm = (abs(s[IVX]) <= sm["settle_vx"] and abs(s[IWP]) <= sm["settle_phid"]
                    and abs(dphi) <= sm["settle_dphi"])
            settled = self.in_contact and t - self.contact_since >= sm["settle_contact_s"] and calm
            first = t <= 1e-9 and self.in_contact and calm
            # run-up: at least min_runup_m between the shell centre and riser contact
            runup_ok = (sup >= self.n_stairs - 1 or
                        (self.edges[sup + 1] - self.P.R) - s[IX] >= sm["min_runup_m"])
            if (settled or first) and runup_ok and t >= self.t_plan_retry:
                if sup >= self.n_stairs - 1:
                    if self.mode != "DONE":
                        self._log(t, "on the top tread: holding")
                    self.mode = "DONE"
                else:
                    self._plan_jump(t, s, sup + 1)

    def _translated_guess(self, s, geo):
        """Previous successful plan shifted to the new stair (uniform-stair warm start)."""
        prev = getattr(self, "_last_good_plan", None)
        if prev is None:
            return None
        g0 = prev["geo"]
        dx, dh = geo["x_e"] - g0["x_e"], geo["h_lo"] - g0["h_lo"]
        k = np.round((s[IPHI] - prev["s0"][IPHI]) / (2 * np.pi))
        guess = {}
        for ph in prev["phases"]:
            X = ph["X"].copy()
            X[IX] += dx
            X[IY] += dh
            X[IPHI] += 2 * np.pi * k
            X[IPHI] -= 2 * np.pi * np.round((s[IPHI] + np.pi / 2) / (2 * np.pi))  # planner frame
            guess[ph["name"]] = (ph["T"], X, ph["U"] / self.lim["tau_max"])
        return guess

    def _plan_jump(self, t, s, j):
        geo = self._geo_for(j)
        g_prev = self._translated_guess(s, geo)
        plan = self.planner.solve(s, geo, tau_prev=self.tau_cmd_last, guess=g_prev)
        if not plan["success"] and g_prev is not None:
            plan = self.planner.solve(s, geo, tau_prev=self.tau_cmd_last)
        if not plan["success"]:
            # second attempt with a different, more conservative initial guess
            g2 = self.planner.initial_guess(plan["s0"], geo)
            T_A, Xa, Ua = g2["A"]
            g2["A"] = (min(T_A * 1.5, 1.9), Xa, Ua)
            plan = self.planner.solve(s, geo, tau_prev=self.tau_cmd_last, guess=g2)
        if plan["success"]:
            self._last_good_plan = plan
            self.plan = plan
            self.t_plan0 = t
            self.target = j
            self.current_stair_index = j
            self.landing_target_x = geo["x_L"]
            self.mode = "TRACK"
            self.track.w_prev = None
            self.track.w_prev_was_track = False
            self.t_next = t
            X = plan["X"]
            k_to = int(np.argmin(np.abs(plan["t"] - plan["t_takeoff"])))
            self.next_stair_edge = geo["x_e"]
            self.desired_prejump_x = float(X[IX, k_to])
            self.omega_spin = float(np.max(np.abs(X[IWP] - X[IWT]))) * self.t_nd
            self._log(t, f"plan for stair {j}: take-off at x={X[IX, k_to]:.3f} (t+{plan['t_takeoff']:.2f}s, "
                         f"vx={X[IVX, k_to]:.2f}), apex y={X[IY].max():.3f}, touch-down x={X[IX, -1]:.3f} "
                         f"(t+{plan['t_touchdown']:.2f}s); IPOPT {plan['iter_count']} it, {plan['solve_time']:.2f}s")
        else:
            self.t_plan_retry = t + self.sm["plan_retry_s"]
            self._log(t, f"planner failed ({plan['return_status']}); holding, retry in {self.sm['plan_retry_s']}s")

    # ── MPC sample ─────────────────────────────────────────────
    def _mpc_sample(self, t, s, sup):
        if self.mode == "TRACK":
            self._solve_track(t, s, sup)
        elif self.mode == "FLIGHT":
            self._solve_flight(t, s, sup)
        else:
            self._solve_settle(t, s, sup)

    def _ref_window(self, t):
        N, Ts = self.track.N, self.track.Ts
        plan = self.plan
        tau = (t - self.t_plan0) + Ts * np.arange(N + 1)
        Xref = np.array([np.interp(tau, plan["t"], plan["X"][i]) for i in range(NX)])
        # interval-averaged reference torque (plan is piecewise constant on t_u)
        tu, U = plan["t_u"], plan["U"]
        Uref = np.zeros(N)
        for k in range(N):
            ts = np.linspace(tau[k], tau[k] + Ts, 9)[:-1] + Ts / 16
            idx = np.clip(np.searchsorted(tu, ts, side="right") - 1, 0, len(U) - 1)
            Uref[k] = np.where(ts < plan["t_touchdown"], U[idx], 0.0).mean()
        mid = tau[:-1] + 0.5 * Ts
        sig = (mid > plan["t_takeoff"]).astype(float)
        lift = np.zeros(N)
        if sig.any() and sig[0] == 0:
            lift[int(np.argmax(sig))] = 1.0
        act = (tau[:-1] < plan["t_touchdown"] - 0.25 * Ts).astype(float)
        if act.sum() == 0:
            act[0] = 1.0
        rob = np.zeros(N)
        if self.cfg["planner"].get("launch_mode_robust_accel", False):
            rob = ((tau[:-1] + Ts > plan["phases"][1]["t0"]) & (sig == 0)).astype(float)
        return Xref, Uref, sig, lift, act, rob

    def _weights(self, which):
        sc = self.cfg["tracking_nmpc"][which]
        Wq = 1.0 / np.array([sc[k] for k in ("x", "y", "phi", "vx", "vy", "thetad", "phid")], float) ** 2
        return Wq, self.cfg["tracking_nmpc"]["terminal_factor"] * Wq

    def _solve_track(self, t, s, sup):
        P = self.P
        j = self.target
        geo_p = self._geo_for(j)
        s0 = s.copy()
        s0[IY] = geo_p["h_lo"] + P.R
        s0[IVY] = 0.0
        Xref, Uref, sig, lift, act, rob = self._ref_window(t)
        Wq, Wqf = self._weights("track_scales")
        x_min = (self.edges[sup] + 0.02) if sup >= 0 else -1e3
        x_max = geo_p["x_e"] - P.R - self.cfg["planner"]["riser_margin_stance"]
        geo = [geo_p["x_e"], geo_p["h_lo"], geo_p["h_hi"], x_min, x_max, 0.0]
        pv = self.track.pack(s0, Xref, Uref, sig, lift, act, Wq, Wqf, geo, self.tau_cmd_last, rob)
        w0 = self.track.shifted_guess(s0) if self.track.w_prev is not None else self.track.guess_from_ref(Xref, Uref)
        r = self.track.solve(pv, w0, use_duals=self.track.w_prev_was_track if hasattr(self.track, "w_prev_was_track") else False)
        self.track.w_prev_was_track = True
        tau0 = float(r["U"][0]) if np.isfinite(r["U"][0]) else float(Uref[0])
        # feedforward (plan torque at its native resolution) + NMPC correction,
        # the correction keeping the interval-average torque equal to U[0]
        dtau = tau0 - float(Uref[0])
        plan, tp0 = self.plan, self.t_plan0
        tu, Up = plan["t_u"], plan["U"]

        def fn(tc, dtau=dtau):
            tr = tc - tp0
            k = int(np.searchsorted(tu, tr, side="right")) - 1
            ff = float(Up[min(max(k, 0), len(Up) - 1)]) if tr < plan["t_touchdown"] else 0.0
            return ff + dtau
        self.tau_cmd_fn = fn
        self.tau_cmd_last = tau0
        self._last_mode_solved = "TRACK"

    def _solve_settle(self, t, s, sup):
        P = self.P
        N = self.settle.N
        h = self._tread_h(sup)
        s0 = s.copy()
        s0[IY] = h + P.R
        s0[IVY] = 0.0
        x_hold = self._hold_target(sup, s[IX])
        # hanging-angle reference with hysteresis: re-selected (nearest hanging angle) unless
        # the pendulum is within `hang_ref_deadband_rad` of the inverted position. A
        # per-sample nearest-angle reference switches sides at the inverted position, where
        # "stay inverted" then becomes a stationary point of the NMPC (observed in the hoop
        # project's aborted frozen run v1). The hoop version re-selected only in the lower
        # half-plane: once a transient carried the pendulum over the top, the reference
        # stayed > pi away, and with the exact elevation penalty the NMPC then pinned the
        # pendulum at the elevation limit, driving the sphere into the riser.
        nearest = -np.pi / 2 + 2 * np.pi * np.round((s[IPHI] + np.pi / 2) / (2 * np.pi))
        band = np.pi - self.cfg["settle_nmpc"].get("hang_ref_deadband_rad", np.pi / 2)
        if getattr(self, "_phi_hang_ref", None) is None or abs(s[IPHI] - nearest) < band:
            self._phi_hang_ref = nearest
        phi_hang = self._phi_hang_ref
        ref = np.array([x_hold, h + P.R, phi_hang, 0, 0, 0, 0], float)
        Xref = np.repeat(ref[:, None], N + 1, axis=1)
        Uref = np.zeros(N)
        Wq, Wqf = self._weights("settle_scales")
        x_e_next = self.edges[sup + 1] if sup + 1 < self.n_stairs else self.edges[-1] + 1e3
        x_min = (self.edges[sup] + 0.02) if sup >= 0 else -1e3
        x_max = x_e_next - P.R - self.cfg["planner"]["riser_margin_stance"]
        geo = [x_e_next, h, h + 1.0, x_min, x_max, 0.0]
        pv = self.settle.pack(s0, Xref, Uref, np.zeros(N), np.zeros(N), np.ones(N), Wq, Wqf, geo, self.tau_cmd_last)
        starts = []
        warm = self.settle.w_prev is not None and getattr(self, "_last_mode_solved", None) == "SETTLE"
        if warm:
            # re-solved every Ts_mpc < Ts_settle: previous solution as warm start
            w0 = self.settle.w_prev.copy()
            X0 = self.settle.nb.get(w0, "X")
            X0[:, 0] = s0 / self.settle.Sx
            self.settle.nb.put(w0, "X", X0)
            starts.append((w0, True))
        if not warm or self.cfg["settle_nmpc"].get("multistart", False):
            # Multi-start: also solve from the cold guess and keep the lower cost. On narrow
            # treads the warm-started solution locked into stationary zero-torque branches
            # (robot parked 0.3 m short of the run-up point, or coasting past the hold point
            # to the tread edge) that the cold solve escapes (sphere, w = 1.0 m).
            Xg = np.repeat(s0[:, None], N + 1, axis=1)
            starts.append((self.settle.guess_from_ref(Xg, Uref), False))
        best = None
        for w0, duals in starts:
            r = self.settle.solve(pv, w0, use_duals=duals)
            if best is None or r["f"] < best[0]["f"]:
                best = (r, self.settle.w_prev, self.settle.lam_prev)
        r, self.settle.w_prev, self.settle.lam_prev = best
        self._last_mode_solved = "SETTLE"
        tau0 = float(r["U"][0]) if np.isfinite(r["U"][0]) else 0.0
        self._set_zoh(tau0)

    def _solve_flight(self, t, s, sup):
        P = self.P
        j = self.target
        geo_p = self._geo_for(j)
        # land on the target tread unless the sphere can no longer get above it
        yc = s[IY] + P.m_p / P.m * P.l * np.sin(s[IPHI])
        vyc = s[IVY] + P.m_p / P.m * P.l * np.cos(s[IPHI]) * s[IWP]
        apex_c = yc + max(vyc, 0.0) ** 2 / (2 * P.g)
        reachable = apex_c + P.m_p / P.m * P.l >= geo_p["h_hi"] + P.R
        if not (reachable or s[IX] >= geo_p["x_e"]):
            h_td = geo_p["h_lo"]
        else:
            h_td = geo_p["h_hi"]
        geo = [geo_p["x_e"], geo_p["h_lo"], geo_p["h_hi"], geo_p["x_L"], h_td]
        if self.flight_w is None and h_td == geo_p["h_hi"] and self.plan is not None:
            # first solve of this flight: warm start from the plan's flight phase,
            # time-shifted so that the plan's take-off coincides with the measured one
            plan = self.plan
            t_to = self._t_takeoff if self._t_takeoff is not None else t
            tau_r = plan["t_takeoff"] + (t - t_to)
            T_rem = max(plan["t_touchdown"] - tau_r, 0.05)
            sel = plan["t"] >= tau_r - 1e-9
            tt = np.r_[0.0, plan["t"][sel] - tau_r]
            XX = np.hstack([s[:, None], plan["X"][:, sel]])
            w0 = self.flight.guess_from_traj(s, tt, XX, plan["t_u"] - tau_r, plan["U"], T_rem)
        elif self.flight_w is None or getattr(self, "_flight_htd", None) != h_td:
            w0 = self.flight.guess_ballistic(s, h_td)
        else:
            w0 = self.flight.shifted_guess(s, t - self.t_last_flight)
        r = self.flight.solve(s, np.array(geo, float), self.tau_cmd_last, w0,
                              use_duals=self.flight_w is not None and getattr(self, "_flight_htd", None) == h_td)
        self.flight_w = True
        self._flight_htd = h_td
        self.t_last_flight = t
        tt, U = r["t"], r["U"]
        t0 = t

        def fn(tc, tt=tt, U=U, t0=t0):
            k = int(np.searchsorted(tt, tc - t0, side="right")) - 1
            return float(U[min(max(k, 0), len(U) - 1)])
        self.tau_cmd_fn = fn
        self.tau_cmd_last = float(U[0])
        self._last_mode_solved = "FLIGHT"

    def _set_zoh(self, tau0):
        self.tau_cmd_fn = lambda tc, v=tau0: v
        self.tau_cmd_last = tau0
        if self.mode != "SETTLE":
            self._last_mode_solved = self.mode

    # ── torque laws (pure functions of t, q, u) ────────────────
    def alpha_tau_func(self, t_nd, q, u):
        tau = self.tau_cmd_fn(float(t_nd) * self.t_nd)
        return self.act.apply(tau) / TAU_SCALE

    def beta_tau_func(self, t_nd, q, u):
        return self._beta_servo(t_nd, q, u)


def make_beta_servo(kp_Nm_per_rad=49.0, kd_Nm_s_per_rad=2.5, tau_max_Nm=1.0,
                    t_nd=np.sqrt(1.0 / 9.81), tau_unit_Nm=9.81):
    """Lateral (beta) motor holding the pendulum in the x-z plane: PD to beta = 0.

    The beta_tau_func of both stair controllers, active under
    Simulation(beta_mode="motor"). The pendulum's (alpha, beta) chart is
    singular at alpha = +-pi/2, which a spinning pendulum crosses twice per
    revolution; with beta unactuated those crossings pump the beta axis until
    it runs away (July 2026: beta wound to -10 rad at -220 rad/s). Default
    gains are 10x the July 2026 servo (4.9 N*m/rad, 0.25 N*m*s/rad): with
    those, beta errors of ~4e-3 rad yawed the shell ~0.8 deg during the
    hierarchical MPC's settle manoeuvres, and the next run-up turned that
    into 17 mm of sideways drift (h=0.2, 2026-10-28); 10x gives 0.5 mm with
    |tau_beta| <= 0.003 N*m. Returns a pure, continuous function of (t, q, u)
    in solver units, so it is safe inside the Newton iterations.
    """
    kp = kp_Nm_per_rad / tau_unit_Nm
    kd = kd_Nm_s_per_rad / tau_unit_Nm / t_nd
    tau_max = tau_max_Nm / tau_unit_Nm

    def beta_tau(t, q, u):
        return float(np.clip(-kp * q[8] - kd * u[7], -tau_max, tau_max))
    return beta_tau


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
        stair_controller=None,
        beta_mode="motor",
        print_steps=True,
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
        # Stair-awareness data used by the controller. Each edge is the riser
        # x-location; each height is the landing tread height after that riser.
        self.stair_edges = np.array(
            [self.stair_x_start + i * self.stair_width for i in range(self.n_stairs)],
            dtype=float,
        )
        self.stair_heights = np.array(
            [(i + 1) * self.stair_height for i in range(self.n_stairs)],
            dtype=float,
        )

        # ── lateral pendulum axis (beta) ──────────────────────
        # Under a stair controller beta is held at 0 either by the lateral
        # motor ("motor": beta is a free DOF driven by the controller's
        # beta_tau_func servo, or by beta_tau_func if given) or by a bilateral
        # constraint ("constraint": ideal, infinitely stiff, zero lateral
        # torque). An unactuated beta runs away (see make_beta_servo).
        if beta_mode not in ("constraint", "motor"):
            raise ValueError(f"beta_mode must be 'constraint' or 'motor', got {beta_mode!r}")
        self.beta_mode = beta_mode

        # ── prescribed pendulum ───────────────────────────────
        if stair_controller is not None and beta_p_func is None and beta_mode == "constraint":
            beta_p_func = lambda t: 0.0
            beta_p_dot_func = lambda t: 0.0
            beta_p_ddot_func = lambda t: 0.0

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

        self.stair_controller = stair_controller
        self.print_steps = bool(print_steps)
        if self.stair_controller is not None:
            self.alpha_tau_func = self.stair_controller.alpha_tau_func
            if beta_mode == "constraint":
                self.beta_tau_func = lambda t, q, u: 0.0
            elif callable(beta_tau_func):
                self.beta_tau_func = beta_tau_func
            else:
                self.beta_tau_func = self.stair_controller.beta_tau_func
        else:
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
        self.controller_state_save = np.full(self.ntime, "", dtype=object)
        self.current_stair_index_save = np.full(self.ntime, -1, dtype=int)
        self.next_stair_edge_save = np.full(self.ntime, np.nan)
        self.desired_prejump_x_save = np.full(self.ntime, np.nan)
        self.landing_target_x_save = np.full(self.ntime, np.nan)
        self.jump_omega_save = np.full(self.ntime, np.nan)
        # writing every array to disk each step dominates run time, so
        # save periodically (and once more at the end of solve()).
        self.save_every = 200
        self._printed_events = 0

        self.q_save[:, 0] = q0
        self.u_save[:, 0] = u0
        self._record_controller_state(0)

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
            "stair_edges": self.stair_edges.tolist(),
            "stair_heights": self.stair_heights.tolist(),
            "beta_mode": self.beta_mode,
        }
        if self.stair_controller is not None and hasattr(self.stair_controller, "get_plan_metadata"):
            params["controller"] = self.stair_controller.get_plan_metadata()
            if self.beta_mode == "constraint":
                params["controller"]["beta_control"] = "bilateral_constraint_beta_eq_0"
                params["controller"].pop("beta_servo", None)
        with open(os.path.join(self.output_path, "params.json"), "w") as fp:
            json.dump(params, fp, indent=2)

    def _record_controller_state(self, iter):
        if self.stair_controller is None:
            return
        self.controller_state_save[iter] = self.stair_controller.controller_state
        self.current_stair_index_save[iter] = self.stair_controller.current_stair_index
        self.next_stair_edge_save[iter] = self.stair_controller.next_stair_edge
        self.desired_prejump_x_save[iter] = self.stair_controller.desired_prejump_x
        self.landing_target_x_save[iter] = self.stair_controller.landing_target_x
        self.jump_omega_save[iter] = getattr(self.stair_controller, "omega_spin", np.nan)
        events = getattr(self.stair_controller, "event_log", None)
        if events is not None:
            while self._printed_events < len(events):
                t_s, msg = events[self._printed_events]
                line = f"[controller t={t_s:.2f}s] {msg}"
                print(line)
                try: self.f.write("\n" + line)
                except Exception: pass
                self._printed_events += 1

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
        if self.stair_controller is not None:
            np.save(f'{self.output_path}/controller_state_save.npy', self.controller_state_save)
            np.save(f'{self.output_path}/current_stair_index_save.npy', self.current_stair_index_save)
            np.save(f'{self.output_path}/next_stair_edge_save.npy', self.next_stair_edge_save)
            np.save(f'{self.output_path}/desired_prejump_x_save.npy', self.desired_prejump_x_save)
            np.save(f'{self.output_path}/landing_target_x_save.npy', self.landing_target_x_save)
            np.save(f'{self.output_path}/jump_omega_save.npy', self.jump_omega_save)

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
            # inline 3-vector crosses: np.cross spends ~10x the arithmetic
            # cost in axis-normalization overhead, and this is the residual
            # hot path (called ~40x per Jacobian build)
            wxr = _cross3(omega_w, r_contact)
            v_contact = v_A + wxr
            a_contact = a_A + _cross3(alpha_w, r_contact) + _cross3(omega_w, wxr)

            gammaF[fric_start]     = t1 @ v_contact
            gammaF[fric_start + 1] = t2 @ v_contact
            gammaFdot[fric_start]     = t1 @ a_contact
            gammaFdot[fric_start + 1] = t2 @ a_contact

            for j, t_j in enumerate([t1, t2]):
                col_idx = fric_start + j
                F_j = t_j
                tau_w = _cross3(r_contact, F_j)  # world-frame torque
                tau_b = Rmat.T @ tau_w           # body-frame torque
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
        if self.print_steps:
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

        self.q_save[:, iter] = q
        self.u_save[:, iter] = u
        self.X_save[:, iter] = X
        self.gNdot_save[:, iter] = gNdot
        self.gNddot_save[:, iter] = gNddot
        self.gammaF_save[:, iter] = gammaF
        self.lambdaN_save[:, iter] = lambda_N
        self.lambdaF_save[:, iter] = lambda_F
        self.tau_pendulum_save[:, iter] = [
            self.alpha_tau_func(self.t[iter], q, u),
            self.beta_tau_func(self.t[iter], q, u),
        ]
        self.AV_save[:, iter] = AV
        self.rho_inf_save[iter] = self.rho_inf
        self.alpha_save[iter] = alpha_min
        if self.stair_controller is not None:
            self.stair_controller.update_after_step(self.t[iter], q, u)
            self._record_controller_state(iter)
        if iter % self.save_every == 0 or iter == self.ntime - 1:
            self.save_arrays()

    # ══════════════════════════════════════════════════════════
    #  SOLVE  (adaptive dt)
    # ══════════════════════════════════════════════════════════
    def solve(self):
        if self.stair_controller is not None and getattr(self.stair_controller, "plan_at_start", False):
            # first plan from the initial state, so step 1 already has torque
            self.stair_controller.update_after_step(self.t[0], self.q_save[:, 0], self.u_save[:, 0])
            self._record_controller_state(0)
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

        self.save_arrays()
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
    import argparse

    # ── Controller choice ─────────────────────────────────────
    #   "sampling_nmpc"    StairNMPCController: best-candidate sampling NMPC
    #                      on the penalty-contact planar model (numpy only)
    #   "hierarchical_mpc" StairJumpMPCController: jump trajectory planner +
    #                      phase-dependent NMPC (CasADi/IPOPT) on the planar
    #                      flight/stance model, as in the hoop project
    # Command-line override: python Spherical_robot_3D_quat.py --controller hierarchical_mpc
    controller = "hierarchical_mpc"
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["sampling_nmpc", "hierarchical_mpc"],
                        default=controller)
    controller = parser.parse_known_args()[0].controller

    # ── Scenario: uniform stairs; change freely between runs ──
    # Both controllers derive everything from this geometry at run time, so
    # no retuning is needed when these change. Physical feasibility envelope
    # (measured in this solver at the 55 rad/s / 4.5 N*m actuator limits)
    n_stairs = 3
    stair_width = 1.0
    
    stair_height = 0.2
    stair_x_start = 1.0

    # ── Robot ─────────────────────────────────────────────────
    R = 0.18
    m_sphere = 1.0
    m_pendulum = 0.5
    l_pendulum = 0.18
    alpha_p0 = -np.pi / 2
    beta_p0 = 0.0

    # ── Contact ───────────────────────────────────────────────
    mu_s = 0.6     # static friction (unused by the proximal contact law)
    mu_k = 0.45    # kinetic friction
    eN = 0.3       # normal restitution
    eF = 0.3       # friction restitution

    # ── Actuator limits (physical units) ──────────────────────
    tau_alpha_max_Nm = 4.5           # pendulum drive torque
    pendulum_speed_max_rad_s = 55.0  # continuous spin cap for stair clearance
    # motor acceleration |alpha_p_ddot| limit, used by the hierarchical MPC only
    # (ASSUMPTION: tau_max / (m_p l^2); replace with the motor data-sheet value)
    pendulum_accel_max_rad_s2 = 600.0

    # ── Lateral (beta) axis: how beta is held at 0 ────────────
    #   "motor"       lateral motor: the controller's PD servo torque on beta, saturated
    #   "constraint"  bilateral constraint (ideal, infinitely stiff lateral axis)
    beta_hold = "motor"
    beta_kp_Nm_per_rad = 49.0        # servo stiffness   (beta_hold = "motor" only)
    beta_kd_Nm_s_per_rad = 2.5       # servo damping
    tau_beta_max_Nm = 1.0            # lateral motor torque limit

    t_nd = np.sqrt(1.0 / 9.81)
    stair_edges = [stair_x_start + i * stair_width for i in range(n_stairs)]
    stair_heights = [(i + 1) * stair_height for i in range(n_stairs)]

    if controller == "sampling_nmpc":
        stair_controller = StairNMPCController(
            stair_edges=stair_edges,
            stair_heights=stair_heights,
            sphere_radius=R,
            stair_width=stair_width,
            m_sphere=m_sphere,
            m_pendulum=m_pendulum,
            l_pendulum=l_pendulum,
            t_nd=t_nd,
            tau_alpha_max_Nm=tau_alpha_max_Nm,
            pendulum_speed_max_rad_s=pendulum_speed_max_rad_s,
            beta_kp_Nm_per_rad=beta_kp_Nm_per_rad,
            beta_kd_Nm_s_per_rad=beta_kd_Nm_s_per_rad,
            tau_beta_max_Nm=tau_beta_max_Nm,
        )
    else:
        # prediction model = the parameters above
        cfg = copy.deepcopy(DEFAULT_MPC_CONFIG)
        cfg["plant"].update(R=R, m_sphere=m_sphere, m_pendulum=m_pendulum,
                            l_pendulum=l_pendulum, mu_k=mu_k, mu_s=mu_s, eN=eN, eF=eF,
                            n_stairs=n_stairs, stair_width=stair_width,
                            stair_height=stair_height, stair_x_start=stair_x_start)
        cfg["actuator"].update(tau_max_Nm=tau_alpha_max_Nm,
                               omega_max_rad_s=pendulum_speed_max_rad_s,
                               alpha_max_rad_s2=pendulum_accel_max_rad_s2)
        # events (plans, take-offs, touch-downs) are printed by Simulation from event_log
        stair_controller = StairJumpMPCController(cfg, stair_edges, stair_heights, stair_width,
                                                  t_nd=t_nd, start_backoff_m=1.0,
                                                  beta_kp_Nm_per_rad=beta_kp_Nm_per_rad,
                                                  beta_kd_Nm_s_per_rad=beta_kd_Nm_s_per_rad,
                                                  tau_beta_max_Nm=tau_beta_max_Nm)

    # start behind the first riser (geometry-derived, no tuning)
    x0 = stair_controller.suggested_start_x

    sim = Simulation(
        ntime=8000,
        mu_s=mu_s, mu_k=mu_k,
        eN=eN, eF=eF,
        R=R,
        m_sphere=m_sphere, m_pendulum=m_pendulum, l_pendulum=l_pendulum,
        x0=x0, y0=0.0, z0=R,
        p0_0=1, p1_0=0.0, p2_0=0.0, p3_0=0.0,
        alpha_p0=alpha_p0, beta_p0=beta_p0,
        x_dot0=0.0, y_dot0=0.0, z_dot0=0.0,
        wx0=0.0, wy0=0.0, wz0=0.0,
        alpha_p_dot0=0.0, beta_p_dot0=0.0,
        n_stairs=n_stairs, stair_width=stair_width, stair_height=stair_height,
        stair_x_start=stair_x_start, fillet_radius=0,
        stair_controller=stair_controller,
        beta_mode=beta_hold,
    )
    sim.solve()
    sim.plot_results()
