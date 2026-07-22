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
        # the whole MPPI batch with NaNs (such samples just score terribly)
        np.clip(V, -60.0, 60.0, out=V)
        out[:, 0] = x + dt * V[:, 0]
        out[:, 1] = z + dt * V[:, 1]
        out[:, 2] = psi + dt * V[:, 3]
        return out


class StairNMPCController:
    """Sampling-based nonlinear MPC (MPPI) for stair jumping. Torque only.

    Replaces the former hand-scripted stair controller. There are no
    behavioral modes, no calibrated launch constants (k_launch etc.) and
    no retry bookkeeping: at ~10 Hz the controller re-optimizes the whole
    pendulum-torque trajectory over a receding horizon by rolling out a
    reduced-order planar model (`PlanarStairModel`) built at run time from
    the physical robot parameters and the actual stair geometry. The
    rollout physics *contains* the launch mechanism — continuous pendulum
    rotation unloads the contact centrifugally until the sphere lifts off —
    so the optimizer rediscovers the jump for whatever riser height and
    tread width it is given, and recovery/retry behavior emerges from
    replanning instead of being scripted.

    Optimizer: MPPI (model-predictive path integral).
      * decision variables: piecewise-constant torque knots (ZOH, ~50 ms)
        over a ~1.4 s horizon — long enough to contain a full
        roll -> spin-up -> flight -> landing cycle;
      * warm start by time-shifting the previous plan;
      * candidates are Gaussian perturbations around the warm-started mean;
        no hand-scripted phases or proportional policy rollouts are injected
        into the optimizer;
      * cost: terminal distance to the next tread's landing zone (derived
        from the geometry) plus running height error, riser-press penalty,
        control effort, a soft actuator speed limit, and terminal
        velocity/spin penalties for calm landings.

    Interface contract with `Simulation`: `alpha_tau_func` / `beta_tau_func`
    stay pure functions of (t, q, u): alpha reads only the stored knot
    sequence and beta receives no motor torque. For NMPC runs the simulator
    constrains beta to zero with a bilateral constraint, so the motion
    remains planar without a PD loop. Replanning happens exclusively in
    `update_after_step` (converged steps). Torque saturation and a hard
    pendulum overspeed reflex are applied in the torque layer, so the
    actuator respects its physical limits regardless of what the optimizer
    asks for.

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
        # ── NMPC discretization ──────────────────────────────
        # budgets sized for wall-clock speed: warm starting carries most of
        # the optimization between replans, so 2 iterations x 96 samples at
        # a 5 ms rollout step recovers the same behavior several times
        # faster than the original 3 x 120 x 4 ms budget
        horizon_s=1.8,
        ctrl_dt_s=0.05,
        model_dt_s=0.005,
        replan_every_s=0.12,
        n_samples=144,
        n_iterations=4,
        noise_sigma_frac=0.25,
        temperature_frac=0.05,
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
        # currently being applied. Without it the MPPI mean inherits the
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

        # discretization (solver time units)
        self.model_dt = float(model_dt_s) / self.t_nd
        self.substeps = max(1, int(round(float(ctrl_dt_s) / float(model_dt_s))))
        self.ctrl_dt = self.substeps * self.model_dt
        self.n_knots = max(1, int(np.ceil(float(horizon_s) / self.t_nd / self.ctrl_dt)))
        self.replan_every = float(replan_every_s) / self.t_nd

        # MPPI parameters
        self.n_samples = int(n_samples)
        self.n_iterations = int(n_iterations)
        self.sigma = float(noise_sigma_frac) * self.tau_max
        self.temperature_frac = float(temperature_frac)
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
        print("StairNMPCController (sampling-based NMPC / MPPI):")
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

    # ── MPPI ──────────────────────────────────────────────────
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
        mean = self._warm_start(t)
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
            # torque profiles (and thus the weighted mean) are not chattery
            for _ in range(2):
                eps[:, 1:-1] = 0.25 * eps[:, :-2] + 0.5 * eps[:, 1:-1] + 0.25 * eps[:, 2:]
            cand = np.clip(mean[None, :] + eps, -self.tau_max, self.tau_max)
            cand = np.vstack([mean[None, :], cand])
            costs, S_ends, peaks = self._rollout(s0, cand, tau_anchor)
            i_best = int(np.argmin(costs))
            if costs[i_best] < best_cost:
                best_cost = float(costs[i_best])
                best_seq = cand[i_best].copy()
                best_end = S_ends[i_best].copy()
                best_peak = float(peaks[i_best])
            lam = max(1e-9, self.temperature_frac * (costs.mean() - costs[i_best]))
            wgt = np.exp(-(costs - costs[i_best]) / lam)
            wgt /= wgt.sum()
            mean = np.clip(wgt @ cand, -self.tau_max, self.tau_max)

        # Apply the best EVALUATED candidate instead of re-rolling the
        # weighted mean — saves two rollouts per replan (the mean re-enters
        # the running anyway as the next replan's warm start via the applied
        # plan), and avoids the MPPI average smearing two differently-timed
        # jumps into one that never jumps. The jerk penalty keeps winning
        # candidates smooth.
        mean = best_seq
        c_nom, S_end, peak = np.array([best_cost]), best_end[None, :], np.array([best_peak])

        self._plan = mean
        self._plan_t0 = float(t)
        self._last_replan_t = float(t)
        self.omega_spin = float(peak[0])
        self.last_plan = {
            "planner": "sampling_nmpc_mppi",
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
            "method": "MPPI sampling-based NMPC on reduced planar model",
            "tau_alpha_max_Nm": self.tau_max * self.tau_scale,
            "beta_control": "bilateral_constraint_beta_eq_0",
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
        return 0.0


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

        # ── prescribed pendulum ───────────────────────────────
        if stair_controller is not None and beta_p_func is None:
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
        }
        if self.stair_controller is not None and hasattr(self.stair_controller, "get_plan_metadata"):
            params["controller"] = self.stair_controller.get_plan_metadata()
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
    # ── Scenario: uniform stairs; change freely between runs ──
    # The NMPC derives everything from this geometry at run time, so no
    # retuning is needed when these change. Physical feasibility envelope
    # (measured in this solver at the 55 rad/s / 4.5 N*m actuator limits)
    n_stairs = 3
    stair_width = 1.5
    stair_height = 0.2
    stair_x_start = 1.0

    # ── Robot ─────────────────────────────────────────────────
    R = 0.18
    m_sphere = 1.0
    m_pendulum = 0.5
    l_pendulum = 0.15
    alpha_p0 = -np.pi / 2
    beta_p0 = 0.0

    # ── Actuator limits (physical units) ──────────────────────
    tau_alpha_max_Nm = 4.5           # pendulum drive torque
    pendulum_speed_max_rad_s = 55.0  # continuous spin cap for stair clearance

    t_nd = np.sqrt(1.0 / 9.81)
    stair_edges = [stair_x_start + i * stair_width for i in range(n_stairs)]
    stair_heights = [(i + 1) * stair_height for i in range(n_stairs)]

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
    )

    # start behind the first riser (geometry-derived, no tuning)
    x0 = stair_controller.suggested_start_x

    sim = Simulation(
        ntime=8000,
        mu_s=0.6, mu_k=0.45,
        eN=0.3, eF=0.3,
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
    )
    sim.solve()
    sim.plot_results()
