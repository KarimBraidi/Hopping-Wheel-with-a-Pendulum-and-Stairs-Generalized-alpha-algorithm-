import numpy as np
import os
try:
    import scipy.io as sio
except Exception:
    sio = None
from datetime import datetime
import shutil
import json


# ═══════════════════════════════════════════════════════════════
#  Custom exceptions (same as 2-D version)
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
#  3-D Kinematics  (pure NumPy)
# ═══════════════════════════════════════════════════════════════
def R_z(angle):
    """Rotation matrix about z by *angle*."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])

def R_y(angle):
    """Rotation matrix about y by *angle*."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def R_x(angle):
    """Rotation matrix about x by *angle*."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def rotation_matrix(psi, theta, phi):
    """3-2-1 Euler rotation  body → incline (flat-ground) frame.
    R = Rz(psi) · Ry(theta) · Rx(phi)
    """
    return R_z(psi) @ R_y(theta) @ R_x(phi)


def pendulum_dir_body(alpha_p, beta_p):
    """Unit direction from sphere centre to pendulum mass, in body frame."""
    return np.array([np.cos(alpha_p) * np.cos(beta_p),
                     np.cos(alpha_p) * np.sin(beta_p),
                     np.sin(alpha_p)])


def sphere_angular_velocity_incline(psi, theta, phi,
                                     psi_dot, theta_dot, phi_dot):
    """Sphere angular velocity expressed in the incline (world) frame.

    omega_h = psi_dot * E_z  +  theta_dot * e_y'  +  phi_dot * e_x1
    """
    E_z = np.array([0.0, 0.0, 1.0])
    e_y_prime = R_z(psi) @ np.array([0.0, 1.0, 0.0])
    Rot = rotation_matrix(psi, theta, phi)
    e_x1 = Rot[:, 0]
    return psi_dot * E_z + theta_dot * e_y_prime + phi_dot * e_x1


def sphere_angular_acceleration_incline(psi, theta, phi,
                                        psi_dot, theta_dot, phi_dot,
                                        psi_ddot, theta_ddot, phi_ddot):
    """True angular acceleration d(omega)/dt in the incline frame.

    omega = psi_dot * E_z + theta_dot * e_y'(psi) + phi_dot * e_x1(psi,theta)
    omega_dot = J_omega * qddot  +  dJ_omega/dt * qdot

    The second term (convective / Coriolis-like) is what was previously missing.
    """
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_th, s_th = np.cos(theta), np.sin(theta)

    E_z = np.array([0.0, 0.0, 1.0])
    e_y_prime = np.array([-s_psi, c_psi, 0.0])
    e_x1 = np.array([c_psi * c_th, s_psi * c_th, -s_th])

    # J_omega * qddot  (same as sphere_angular_velocity_incline with ddot rates)
    omega_dot = psi_ddot * E_z + theta_ddot * e_y_prime + phi_ddot * e_x1

    # d(e_y')/dt = psi_dot * d(e_y')/d(psi)
    de_y_prime_dpsi = np.array([-c_psi, -s_psi, 0.0])
    omega_dot += theta_dot * psi_dot * de_y_prime_dpsi

    # d(e_x1)/dt = psi_dot * d(e_x1)/d(psi) + theta_dot * d(e_x1)/d(theta)
    # (e_x1 does not depend on phi)
    de_x1_dpsi = np.array([-s_psi * c_th, c_psi * c_th, 0.0])
    de_x1_dtheta = np.array([-c_psi * s_th, -s_psi * s_th, -c_th])
    omega_dot += phi_dot * (psi_dot * de_x1_dpsi + theta_dot * de_x1_dtheta)

    return omega_dot


def velocity_sphere_center(x_dot, y_dot, z_dot):
    """Translational velocity of sphere centre (incline frame)."""
    return np.array([x_dot, y_dot, z_dot])


def velocity_pendulum(q, qdot, R_radius, l_pend):
    """Velocity of pendulum mass D in the incline frame.

    r_D = r_A + R_total · l · e_x2_body

    We compute v_D = d(r_D)/dt  numerically via central differences on q.
    BUT since v_D is *linear* in qdot, we use a Jacobian approach:
        v_D = J_D(q) · qdot
    where J_D[:,i] = v_D evaluated with qdot = e_i.
    This function returns *just* the velocity for a given (q, qdot).
    """
    # Build r_D(q)
    # q = [x, y, z, psi, theta, phi, alpha_p, beta_p]
    Rot = rotation_matrix(q[3], q[4], q[5])
    e_x2_b = pendulum_dir_body(q[6], q[7])
    # r_D = [x, y, z] + Rot · l · e_x2_b  (position, not needed for velocity directly)

    # Velocity via finite-difference on position w.r.t. time
    # But we know v_D is linear in qdot, so we compute J_D and multiply.
    J_D = _velocity_jacobian_pendulum(q, R_radius, l_pend)
    return J_D @ qdot


def _position_pendulum(q, l_pend):
    """Position of pendulum mass D in incline frame."""
    c_psi, s_psi = np.cos(q[3]), np.sin(q[3])
    c_theta, s_theta = np.cos(q[4]), np.sin(q[4])
    c_phi, s_phi = np.cos(q[5]), np.sin(q[5])
    # R_total = Rz(psi) @ Ry(theta) @ Rx(phi)  (inlined for speed)
    R00 = c_psi*c_theta;  R01 = c_psi*s_theta*s_phi - s_psi*c_phi;  R02 = c_psi*s_theta*c_phi + s_psi*s_phi
    R10 = s_psi*c_theta;  R11 = s_psi*s_theta*s_phi + c_psi*c_phi;  R12 = s_psi*s_theta*c_phi - c_psi*s_phi
    R20 = -s_theta;       R21 = c_theta*s_phi;                       R22 = c_theta*c_phi
    # e_x2_body
    ca, sa = np.cos(q[6]), np.sin(q[6])
    cb, sb = np.cos(q[7]), np.sin(q[7])
    bx = ca*cb;  by = ca*sb;  bz = sa
    # r_D = r_A + Rot @ (l * e_x2_body)
    return np.array([q[0] + l_pend*(R00*bx + R01*by + R02*bz),
                     q[1] + l_pend*(R10*bx + R11*by + R12*bz),
                     q[2] + l_pend*(R20*bx + R21*by + R22*bz)])


def _velocity_jacobian_pendulum(q, R_radius, l_pend):
    """Jacobian J_D (3×8) such that v_D = J_D · qdot.

    Computed analytically:
        r_D = [x,y,z] + R(psi,theta,phi) · l · e_body(alpha,beta)
        J_D[:,i] = dr_D / dq_i
    """
    psi, theta, phi = q[3], q[4], q[5]
    alpha_p, beta_p = q[6], q[7]

    c_psi, s_psi = np.cos(psi), np.sin(psi)
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    ca, sa = np.cos(alpha_p), np.sin(alpha_p)
    cb, sb = np.cos(beta_p), np.sin(beta_p)

    # Body-frame pendulum direction and its partial derivatives
    d = l_pend * np.array([ca*cb, ca*sb, sa])
    dd_da = l_pend * np.array([-sa*cb, -sa*sb, ca])
    dd_db = l_pend * np.array([-ca*sb, ca*cb, 0.0])

    # Rotation matrices
    Rz_mat = np.array([[ c_psi, -s_psi, 0],
                       [ s_psi,  c_psi, 0],
                       [ 0,      0,     1]])
    Ry_mat = np.array([[ c_th, 0, s_th],
                       [ 0,    1, 0   ],
                       [-s_th, 0, c_th]])
    Rx_mat = np.array([[1,  0,     0    ],
                       [0,  c_phi, -s_phi],
                       [0,  s_phi,  c_phi]])

    # Derivatives of individual rotation matrices
    dRz = np.array([[-s_psi, -c_psi, 0],
                    [ c_psi, -s_psi, 0],
                    [ 0,      0,     0]])
    dRy = np.array([[-s_th, 0, c_th],
                    [ 0,    0, 0   ],
                    [-c_th, 0, -s_th]])
    dRx = np.array([[0,  0,      0     ],
                    [0, -s_phi, -c_phi],
                    [0,  c_phi, -s_phi]])

    R_total = Rz_mat @ Ry_mat @ Rx_mat

    J = np.zeros((3, 8), dtype=q.dtype)
    # Columns 0-2: dr_D/d(x,y,z) = I
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0
    # Column 3: dr_D/dpsi = dRz/dpsi @ Ry @ Rx @ d
    J[:, 3] = (dRz @ Ry_mat @ Rx_mat) @ d
    # Column 4: dr_D/dtheta = Rz @ dRy/dtheta @ Rx @ d
    J[:, 4] = (Rz_mat @ dRy @ Rx_mat) @ d
    # Column 5: dr_D/dphi = Rz @ Ry @ dRx/dphi @ d
    J[:, 5] = (Rz_mat @ Ry_mat @ dRx) @ d
    # Column 6: dr_D/dalpha = R_total @ dd/dalpha
    J[:, 6] = R_total @ dd_da
    # Column 7: dr_D/dbeta = R_total @ dd/dbeta
    J[:, 7] = R_total @ dd_db
    return J


def _omega_jacobian_sphere(q):
    """Jacobian J_omega (3×8) such that omega_h = J_omega · qdot.

    Only columns 3,4,5 (psi_dot, theta_dot, phi_dot) are nonzero.
    omega_h = psi_dot*E_z + theta_dot*e_y' + phi_dot*e_x1
    """
    J = np.zeros((3, 8), dtype=q.dtype)
    psi, theta, phi = q[3], q[4], q[5]
    E_z = np.array([0.0, 0.0, 1.0])
    e_y_prime = R_z(psi) @ np.array([0.0, 1.0, 0.0])
    Rot = rotation_matrix(psi, theta, phi)
    e_x1 = Rot[:, 0]
    J[:, 3] = E_z        # d(omega)/d(psi_dot)
    J[:, 4] = e_y_prime  # d(omega)/d(theta_dot)
    J[:, 5] = e_x1       # d(omega)/d(phi_dot)
    return J


# ═══════════════════════════════════════════════════════════════
#  Mass matrix, Coriolis, gravity  (numerical)
# ═══════════════════════════════════════════════════════════════
def compute_mass_matrix(q, m_h, m_p, I_sphere, R_radius, l_pend):
    """8×8 mass matrix  M(q).

    M = m_h * J_A^T J_A  +  J_omega^T I_sphere J_omega  +  m_p * J_D^T J_D

    where:
      J_A (3×8): translational Jacobian of sphere centre (columns 0-2 = I, rest 0)
      J_omega (3×8): angular-velocity Jacobian of sphere
      J_D (3×8):  translational Jacobian of pendulum mass
      I_sphere (3×3): moment-of-inertia tensor of the sphere shell (body frame,
                       but J_omega already maps to incline frame ⇒ rotate)
    """
    # J_A: trivial — v_A = [x_dot, y_dot, z_dot, 0, 0, 0, 0, 0]
    J_A = np.zeros((3, 8))
    J_A[0, 0] = 1.0
    J_A[1, 1] = 1.0
    J_A[2, 2] = 1.0

    # J_omega: sphere angular velocity in incline frame
    J_omega_inc = _omega_jacobian_sphere(q)

    # Convert to body frame for inertia multiplication:
    # omega_body = R^T omega_incline  =>  J_omega_body = R^T J_omega_inc
    Rot = rotation_matrix(q[3], q[4], q[5])
    J_omega_body = Rot.T @ J_omega_inc

    # J_D: pendulum mass translational Jacobian
    J_D = _velocity_jacobian_pendulum(q, R_radius, l_pend)

    # Assemble M
    M = (m_h * J_A.T @ J_A
         + J_omega_body.T @ I_sphere @ J_omega_body
         + m_p * J_D.T @ J_D)

    # Symmetrise (remove FP noise)
    M = 0.5 * (M + M.T)
    return M


def compute_coriolis_and_gravity(q, qdot, m_h, m_p, I_sphere, R_radius, l_pend, gr):
    """Compute  h(q, qdot) = C(q, qdot)*qdot + g(q)  via complex-step differentiation.

    h_i = sum_{j,k} Gamma_{ijk} qdot_j qdot_k  +  dV/dq_i
    Gamma_{ijk} = 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    Complex-step: df/dx = Im(f(x + i*h)) / h   (no subtractive cancellation).
    """
    ndof = 8
    h_cs = 1e-30  # complex-step perturbation (can be tiny — no cancellation)

    # --- Gravity (potential-energy gradient) via complex step ---
    g_vec = np.zeros(ndof)
    for i in range(ndof):
        q_c = q.astype(complex) # convert to complex for perturbation
        q_c[i] += 1j * h_cs  # preturb q[i] in the imaginary direction
        g_vec[i] = np.imag(_potential_energy(q_c, m_h, m_p, l_pend, gr)) / h_cs

    # --- Coriolis via Christoffel symbols (complex step for dM/dq) ---
    dM = np.zeros((ndof, ndof, ndof))  # dM[i,j,k] = dM_ij/dq_k
    for k in range(ndof):
        q_c = q.astype(complex)
        q_c[k] += 1j * h_cs
        M_c = compute_mass_matrix(q_c, m_h, m_p, I_sphere, R_radius, l_pend)
        dM[:, :, k] = np.imag(M_c) / h_cs

    # c_i = sum_{j,k} 0.5*(dM[i,j,k] + dM[i,k,j] - dM[j,k,i]) * u_j * u_k
    c_vec = (0.5 * np.einsum('ijk,j,k->i', dM, qdot, qdot)
             + 0.5 * np.einsum('ikj,j,k->i', dM, qdot, qdot)
             - 0.5 * np.einsum('jki,j,k->i', dM, qdot, qdot))

    return c_vec + g_vec


def _potential_energy(q, m_h, m_p, l_pend, gr):
    """Gravitational potential energy.  V = (m_h + m_p)*gr*z + m_p*gr*(r_D)_z
    where z is measured upward from the ground plane.
    Sign convention: V increases with height ⇒ force = -dV/dq.
    But we return V so that the gradient gives the gravity force with a minus sign
    that we handle in the residual.
    """
    z_A = q[2]
    r_D = _position_pendulum(q, l_pend)
    z_D = r_D[2]
    return m_h * gr * z_A + m_p * gr * z_D


# ═══════════════════════════════════════════════════════════════
#  Flat-ground contact model  (3-D sphere on z = 0 plane)
# ═══════════════════════════════════════════════════════════════
def flat_ground_contact(q, qdot, a, R_radius):
    """Compute contact quantities for a sphere on the z=0 plane.

    Returns
    -------
    gN : (1,) normal gap
    gNdot : (1,) normal gap velocity
    gNddot : (1,) normal gap acceleration
    WN : (8, 1)  normal force direction matrix
    gammaF : (2,) tangential slip velocities (x and y directions)
    gammaFdot : (2,) tangential slip accelerations
    WF : (8, 2)  friction force direction matrix
    """
    # Normal gap: z - R
    gN = np.array([q[2] - R_radius])
    # Normal velocity: z_dot
    gNdot = np.array([qdot[2]])
    # Normal acceleration: z_ddot
    gNddot = np.array([a[2]])

    # Normal force is along +z (incline frame).
    WN = np.zeros((8, 1))
    WN[2, 0] = 1.0  # z DOF

    # Tangential contact velocity:
    # v_contact_point = v_A + omega × r_contact
    # r_contact = [0, 0, -R] in incline frame
    # omega (incline frame) from the current qdot
    psi, theta, phi = q[3], q[4], q[5]
    omega_inc = sphere_angular_velocity_incline(
        psi, theta, phi, qdot[3], qdot[4], qdot[5])
    r_contact = np.array([0.0, 0.0, -R_radius])
    v_contact = np.array([qdot[0], qdot[1], qdot[2]]) + np.cross(omega_inc, r_contact)

    # Tangential components (x and y in incline frame)
    gammaF = np.array([v_contact[0], v_contact[1]])

    # Similarly for acceleration:
    omega_inc_dot = sphere_angular_acceleration_incline(
        psi, theta, phi, qdot[3], qdot[4], qdot[5], a[3], a[4], a[5])
    # Note: full d/dt(omega × r) = omega_dot × r + omega × (omega × r)
    # So a_c = a_A + omega_dot × r_c + omega × (omega × r_c)
    a_contact = np.array([a[0], a[1], a[2]]) + np.cross(omega_inc_dot, r_contact) + np.cross(omega_inc, np.cross(omega_inc, r_contact))
    gammaFdot = np.array([a_contact[0], a_contact[1]])

    # WF: friction force direction matrix (8×2)
    # A unit friction force in x-direction: F = [1, 0, 0]
    #   enters x translation directly.
    #   torque = r_contact × F = [0,0,-R] × [1,0,0] = [0*0 - (-R)*0, (-R)*1 - 0*0, 0*0 - 0*1]
    #          = [0, -R, 0]
    # A unit friction force in y-direction: F = [0, 1, 0]
    #   torque = [0,0,-R] × [0,1,0] = [0*0 - (-R)*1, (-R)*0 - 0*0, 0*0 - 0*0]
    #          = [R, 0, 0]
    # These torques need to be mapped to the generalised coordinates.
    # The generalised force for rotational DOFs (psi, theta, phi) from a torque tau is:
    #   Q_rot = J_omega^T · tau   where J_omega maps qdot to omega (incline frame)
    J_omega = _omega_jacobian_sphere(q)

    WF = np.zeros((8, 2))
    # x-friction: force [1,0,0], torque = r_contact × F
    F_x = np.array([1.0, 0.0, 0.0])
    tau_x = np.cross(r_contact, F_x) 
    WF[0, 0] = 1.0  # x translation
    # J_omega is 3×8, so J_omega.T is 8×3.  But we only want the
    # contribution to the 3 rotational DOFs (psi, theta, phi = indices 3,4,5).
    # The Jacobian maps qdot→omega, so generalised torque = J_omega^T · tau.
    # J_omega only has nonzero columns at indices 3,4,5, so the full 8-vector
    # result will have zeros everywhere except possibly at 3,4,5.
    WF[:, 0] += J_omega.T @ tau_x

    # y-friction: force [0,1,0]
    F_y = np.array([0.0, 1.0, 0.0])
    tau_y = np.cross(r_contact, F_y)  # [R, 0, 0]
    WF[1, 1] = 1.0  # y translation
    WF[:, 1] += J_omega.T @ tau_y

    return gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF


# ═══════════════════════════════════════════════════════════════
#  Proximal maps for semi-smooth Newton contact formulation
# ═══════════════════════════════════════════════════════════════
def prox_R_plus(x):
    """Proximal map onto R_+ (non-negative reals): max(0, x)."""
    return max(0.0, float(x))  # if x>= keep it and if x<0 push it to 0


def prox_coulomb_2d(x, radius):
    """Project 2-D vector *x* onto the disk of given *radius*.

    Returns x if ||x|| <= radius, else radius * x / ||x||.
    """
    radius = max(0.0, float(radius)) # ensures valid radius 
    norm_x = np.linalg.norm(x)  # magnitude of x
    if norm_x <= radius:
        return x.copy()    #inside the disk
    elif norm_x > 0.0:
        return radius * x / norm_x # if x is outside the disk
    else:
        return np.zeros_like(x)


# ═══════════════════════════════════════════════════════════════
#  Simulation class  (3-D sphere + 2-DOF pendulum on flat ground)
# ═══════════════════════════════════════════════════════════════
class Simulation:
    def __init__(
        self,
        ntime=5,
        mu_s=0.3,
        mu_k=0.3,
        eN=0.0,
        eF=0.0,
        R=0.1,
        m_sphere=1.0,
        m_pendulum=0.1,
        l_pendulum=0.05,
        # initial conditions
        x0=0.0, y0=0.0, z0=None,  # z0=None => z0 = R (sitting on ground)
        psi0=0.0, theta0=0.0, phi0=0.0,
        alpha_p0=0.0, beta_p0=0.0,
        # initial velocities
        x_dot0=0.0, y_dot0=0.0, z_dot0=0.0,
        psi_dot0=0.0, theta_dot0=0.0, phi_dot0=0.0,
        alpha_p_dot0=0.0, beta_p_dot0=0.0,
        # stairs (in x-direction, z is vertical)
        n_stairs=0, stair_width=0.3, stair_height=0.05,
        stair_x_start=0.5, fillet_radius=0,
        # prescribed pendulum (callable or None)
        alpha_p_func=None, alpha_p_dot_func=None, alpha_p_ddot_func=None,
        beta_p_func=None, beta_p_dot_func=None, beta_p_ddot_func=None,
    ):
        # ── output directory ──────────────────────────────────
        timestamp = datetime.now().strftime("sphere3d_%Y-%m-%d_%H-%M-%S")
        outputs_dir = f"outputs/{timestamp}"
        self.output_path = os.path.join(os.getcwd(), outputs_dir)
        os.makedirs(self.output_path, exist_ok=True)
        current_file = os.path.realpath(__file__)
        shutil.copy2(current_file, self.output_path)

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

        # ── time stepping ─────────────────────────────────────
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

        # Sphere shell inertia tensor (thin spherical shell: I = (2/3) m R^2)
        I_val = (2.0 / 3.0) * self.m_sphere * self.R**2
        self.I_sphere = np.diag([I_val, I_val, I_val])

        # ── stair geometry (x-direction, z vertical) ─────────
        self.n_stairs = int(n_stairs)
        self.stair_width = float(stair_width) / l_nd
        self.stair_height = float(stair_height) / l_nd
        self.stair_x_start = float(stair_x_start) / l_nd
        self.fillet_radius = float(fillet_radius) / l_nd
        self._build_stair_profile()
        self._classify_segments()

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

        # ── DOF & constraint counts ───────────────────────────
        self.ndof = 8  # x, y, z, psi, theta, phi, alpha_p, beta_p
        self.ng = (1 if self.prescribed_alpha else 0) + (1 if self.prescribed_beta else 0)
        self.ngamma = 0

        # contact: 2 normals (tread + riser), 4 friction (2 per normal)
        self.nN = 2
        self.nF = 4
        # Map: normal 0 (tread) → friction [0, 1]; normal 1 (riser) → friction [2, 3]
        self.gammaF_lim = np.array([[0, 1],
                                    [2, 3]])

        self.nX = 3*self.ndof + 3*self.ng + 2*self.ngamma + 3*self.nN + 2*self.nF

        # ── generalized-alpha parameters ──────────────────────
        self.MAXITERn = 300
        self.MAXITERn_initial = self.MAXITERn
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
        q0 = np.array([float(x0), float(y0), float(z0),
                        float(psi0), float(theta0), float(phi0),
                        float(alpha_p0), float(beta_p0)])
        u0 = np.array([float(x_dot0), float(y_dot0), float(z_dot0),
                        float(psi_dot0), float(theta_dot0), float(phi_dot0),
                        float(alpha_p_dot0), float(beta_p_dot0)])

        # ── storage arrays ────────────────────────────────────
        self.q_save = np.zeros((self.ndof, self.ntime))
        self.u_save = np.zeros((self.ndof, self.ntime))
        self.X_save = np.zeros((self.nX, self.ntime))
        self.gNdot_save = np.zeros((self.nN, self.ntime))
        self.gNddot_save = np.zeros((self.nN, self.ntime))
        self.gammaF_save = np.zeros((self.nF, self.ntime))
        self.lambdaN_save = np.zeros((self.nN, self.ntime))
        self.lambdaF_save = np.zeros((self.nF, self.ntime))
        self.AV_save = np.zeros((self.ndof + self.nN + self.nF, self.ntime))
        self.contacts_save = np.zeros((5*self.nN, self.ntime))
        self.rho_inf_save = np.full(self.ntime, self.rho_inf)
        self.dtime_save = np.full(self.ntime, self.dtime)
        self.alpha_save = np.ones(self.ntime)

        self.q_save[:, 0] = q0
        self.u_save[:, 0] = u0

        # ── log file ──────────────────────────────────────────
        self.f = open(f"{self.output_path}/log_file.txt", 'a')
        self._save_metadata()

    # ── metadata ──────────────────────────────────────────────
    def _save_metadata(self):
        params = {
            "ndof": self.ndof,
            "dtime": float(self.dtime),
            "ntime": self.ntime,
            "R": float(self.R),
            "m_sphere": float(self.m_sphere),
            "m_pendulum": float(self.m_pendulum),
            "l_pendulum": float(self.l_pendulum),
            "mu_s": float(self.mu_s),
            "mu_k": float(self.mu_k),
            "eN": float(self.eN),
            "eF": float(self.eF),
            "q0": self.q_save[:, 0].tolist(),
            "u0": self.u_save[:, 0].tolist(),
            "rho_inf": float(self.rho_inf),
            "n_stairs": self.n_stairs,
            "stair_width": float(self.stair_width),
            "stair_height": float(self.stair_height),
            "stair_x_start": float(self.stair_x_start),
            "fillet_radius": float(self.fillet_radius),
        }
        with open(os.path.join(self.output_path, "params.json"), "w") as fp:
            json.dump(params, fp, indent=2)

    # ── save arrays ───────────────────────────────────────────
    def save_arrays(self):
        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gNddot_save.npy', self.gNddot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/lambdaN_save.npy', self.lambdaN_save)
        np.save(f'{self.output_path}/lambdaF_save.npy', self.lambdaF_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        np.save(f'{self.output_path}/contacts_save.npy', self.contacts_save)
        np.save(f'{self.output_path}/rho_inf_save.npy', self.rho_inf_save)
        np.save(f'{self.output_path}/dtime_save.npy', self.dtime_save)
        np.save(f'{self.output_path}/alpha_save.npy', self.alpha_save)

    # ══════════════════════════════════════════════════════════
    #  Stair geometry  (x-direction, z vertical; infinite in y)
    # ══════════════════════════════════════════════════════════
    def _build_stair_profile(self):
        """Build the stair profile as line segments in the (x, z) plane.

        Stores ``self.stair_segments`` = [(x1, z1, x2, z2), ...].
        When ``fillet_radius > 0``, convex corners are replaced by arcs.
        """
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
        """Classify stair segments into horizontal (treads) and vertical (risers)."""
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
        """Vertical gap to nearest tread whose horizontal span contains the ball centre.

        Returns (gap, nx, nz) with normal pointing from tread toward ball.
        """
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
        """Gap to nearest riser (vertical wall) including corner contacts.

        Returns (gap, nx, nz) with normal pointing from closest point toward ball.
        """
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
    #  Contact model  (tread + riser, each with 2 friction dirs)
    # ══════════════════════════════════════════════════════════
    def _compute_contact(self, q, qdot, a):
        """Compute contact quantities for a sphere on stairs (or flat ground).

        Contact 0: tread (horizontal surface)  – normal in (x,z) plane
        Contact 1: riser (vertical wall/corner) – normal in (x,z) plane

        Returns
        -------
        gN : (2,)  normal gaps
        gNdot : (2,)  normal gap velocities
        gNddot : (2,)  normal gap accelerations
        WN : (8, 2)  normal force direction matrix
        gammaF : (4,)  tangential slip velocities
        gammaFdot : (4,)  tangential slip accelerations
        WF : (8, 4)  friction force direction matrix
        """
        psi, theta, phi = q[3], q[4], q[5]
        omega_inc = sphere_angular_velocity_incline(
            psi, theta, phi, qdot[3], qdot[4], qdot[5])
        omega_inc_dot = sphere_angular_acceleration_incline(
            psi, theta, phi, qdot[3], qdot[4], qdot[5], a[3], a[4], a[5])
        J_omega = _omega_jacobian_sphere(q)
        v_A = np.array([qdot[0], qdot[1], qdot[2]])
        a_A = np.array([a[0], a[1], a[2]])

        # Tread gap (contact 0)
        gap0, nx0_2d, nz0_2d = self._compute_tread_gap(q[0], q[2])
        n0 = np.array([nx0_2d, 0.0, nz0_2d])

        # Riser gap (contact 1)
        gap1, nx1_2d, nz1_2d = self._compute_riser_gap(q[0], q[2])
        n1 = np.array([nx1_2d, 0.0, nz1_2d])

        gN = np.array([gap0, gap1])

        # Normal velocities: gNdot_k = n_k · v_A
        gNdot = np.array([n0 @ v_A, n1 @ v_A])
        gNddot = np.array([n0 @ a_A, n1 @ a_A])

        # WN: normal force enters translational DOFs along n_k, no rotational torque
        WN = np.zeros((8, 2))
        WN[0, 0] = n0[0]; WN[1, 0] = n0[1]; WN[2, 0] = n0[2]
        WN[0, 1] = n1[0]; WN[1, 1] = n1[1]; WN[2, 1] = n1[2]

        # Friction: 2 tangent directions per contact
        # For contact k with normal n_k = (nx, 0, nz):
        #   t_k1 = (0, 1, 0)             (y-lateral, always perp since ny=0)
        #   t_k2 = (nz, 0, -nx)          (tangent in x-z plane)
        gammaF = np.zeros(4)
        gammaFdot = np.zeros(4)
        WF = np.zeros((8, 4))

        for k, (n_k, fric_start) in enumerate([(n0, 0), (n1, 2)]):
            nx_k, nz_k = n_k[0], n_k[2]
            t1 = np.array([0.0, 1.0, 0.0])
            t2 = np.array([nz_k, 0.0, -nx_k])

            r_contact = -self.R * n_k
            v_contact = v_A + np.cross(omega_inc, r_contact)
            a_contact = a_A + np.cross(omega_inc_dot, r_contact) + \
                        np.cross(omega_inc, np.cross(omega_inc, r_contact))

            gammaF[fric_start]     = t1 @ v_contact
            gammaF[fric_start + 1] = t2 @ v_contact
            gammaFdot[fric_start]     = t1 @ a_contact
            gammaFdot[fric_start + 1] = t2 @ a_contact

            # WF columns: friction force + torque mapped to generalized coords
            for j, t_j in enumerate([t1, t2]):
                col_idx = fric_start + j
                F_j = t_j
                tau_j = np.cross(r_contact, F_j)
                WF[0, col_idx] = F_j[0]
                WF[1, col_idx] = F_j[1]
                WF[2, col_idx] = F_j[2]
                WF[:, col_idx] += J_omega.T @ tau_j

        return gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF

    # ── X vector decomposition ────────────────────────────────
    def get_X_components(self, X):
        n = self.ndof
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
    #  RESIDUAL  (semi-smooth proximal formulation)
    # ══════════════════════════════════════════════════════════
    def get_R(self, iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):

        (prev_a, _, _, _, _, _, _, _, _, _, prev_lambdaN, _, prev_lambdaF) = self.get_X_components(prev_X)
        (a, U, Q, Kappa_g, Lambda_g, lambda_g,
         Lambda_gamma, lambda_gamma,
         KappaN, LambdaN, lambdaN, LambdaF, lambdaF) = self.get_X_components(X)

        # AV - Auxiliary Variables [abar, lambdaNbar, lambdaFbar]
        prev_abar = prev_AV[0:self.ndof]
        prev_lambdaNbar = prev_AV[self.ndof:self.ndof+self.nN]
        prev_lambdaFbar = prev_AV[self.ndof+self.nN:self.ndof+self.nN+self.nF]

        # eq. 49
        abar = (self.alpha_f*prev_a + (1-self.alpha_f)*a - self.alpha_m*prev_abar) / (1-self.alpha_m)
        # eq. 96
        lambdaNbar = (self.alpha_f*prev_lambdaN + (1-self.alpha_f)*lambdaN - self.alpha_m*prev_lambdaNbar) / (1-self.alpha_m)
        # eq. 114
        lambdaFbar = (self.alpha_f*prev_lambdaF + (1-self.alpha_f)*lambdaF - self.alpha_m*prev_lambdaFbar) / (1-self.alpha_m)
        AV = np.concatenate((abar, lambdaNbar, lambdaFbar))

        # velocity update (73) 
        u = prev_u + self.dtime*((1-self.gama)*prev_abar + self.gama*abar) + U
        # position update (73)
        q = prev_q + self.dtime*prev_u + self.dtime**2/2*((1-2*self.beta)*prev_abar + 2*self.beta*abar) + Q

        # ── Mass matrix (numerical) ──────────────────────────
        M = compute_mass_matrix(q, self.m_sphere, self.m_pendulum,
                                self.I_sphere, self.R, self.l_pendulum)

        # ── Coriolis + gravity ────────────────────────────────
        h = compute_coriolis_and_gravity(q, u, self.m_sphere, self.m_pendulum,
                                         self.I_sphere, self.R, self.l_pendulum, self.gr)
        # h = C*qdot + dV/dq
        # EOM: M*a + h - W*lambda = 0  (external force = -dV/dq is inside h with correct sign)
        # Actually: M*a = -C*qdot - dV/dq + W*lambda
        #         => M*a + C*qdot + dV/dq - W*lambda = 0
        # h already = C*qdot + dV/dq

        # ── Bilateral constraints ─────────────────────────────
        g_list = []
        gdot_list = []
        gddot_list = []
        Wg_cols = []
        if self.prescribed_alpha:
            g_list.append(q[6] - self.alpha_p_func(self.t[iter]))
            gdot_list.append(u[6] - self.alpha_p_dot_func(self.t[iter]))
            gddot_list.append(a[6] - self.alpha_p_ddot_func(self.t[iter]))
            col = np.zeros(self.ndof); col[6] = 1.0
            Wg_cols.append(col)
        if self.prescribed_beta:
            g_list.append(q[7] - self.beta_p_func(self.t[iter]))
            gdot_list.append(u[7] - self.beta_p_dot_func(self.t[iter]))
            gddot_list.append(a[7] - self.beta_p_ddot_func(self.t[iter]))
            col = np.zeros(self.ndof); col[7] = 1.0
            Wg_cols.append(col)

        if self.ng > 0:
            g = np.array(g_list)
            gdot = np.array(gdot_list)
            gddot = np.array(gddot_list)
            Wg = np.column_stack(Wg_cols)
        else:
            g = np.zeros(0)
            gdot = np.zeros(0)
            gddot = np.zeros(0)
            Wg = np.zeros((self.ndof, 0))

        #  bilateral constraints at velocity level
        gamma = np.zeros(self.ngamma)
        gammadot = np.zeros(self.ngamma)
        Wgamma = np.zeros((self.ndof, self.ngamma))

        # ── Contact (stairs / flat ground) ───────────────────
        gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF = self._compute_contact(
            q, u, a)

        # eq. 44
        ksiN = gNdot + self.eN * prev_gNdot
        # eq. 95
        PN = LambdaN + self.dtime*((1-self.gama)*prev_lambdaNbar + self.gama*lambdaNbar)
        # eq. 102
        Kappa_hatN = KappaN + self.dtime**2/2*((1-2*self.beta)*prev_lambdaNbar + 2*self.beta*lambdaNbar)
        # eq. 48
        ksiF = gammaF + self.eF * prev_gammaF
        # eq. 113
        PF = LambdaF + self.dtime*((1-self.gama)*prev_lambdaFbar + self.gama*lambdaFbar)

        # ── Smooth residual Rs ────────────────────────────────
        Rs = np.concatenate((
            M @ a + h - Wg @ lambda_g - Wgamma @ lambda_gamma - WN @ lambdaN - WF @ lambdaF,
            M @ U - Wg @ Lambda_g - Wgamma @ Lambda_gamma - WN @ LambdaN - WF @ LambdaF,
            M @ Q - Wg @ Kappa_g - WN @ KappaN - self.dtime/2*(Wgamma @ Lambda_gamma + WF @ LambdaF),
            g, gdot, gddot,
            gamma, gammadot,
        ))

        # ── Contact residual Rc (semi-smooth / proximal) ──────
        R_KappaN = np.zeros(self.nN)  # (129)
        R_LambdaN = np.zeros(self.nN)
        R_lambdaN = np.zeros(self.nN)
        R_LambdaF = np.zeros(self.nF)  # (138)
        R_lambdaF = np.zeros(self.nF)  # (142)

        A = np.zeros(self.nN, dtype=int)
        B = np.zeros(self.nN, dtype=int)
        C = np.zeros(self.nN, dtype=int)
        D = np.zeros(self.nN, dtype=int)
        E = np.zeros(self.nN, dtype=int)

        for k in range(self.nN):
            fric_idx = self.gammaF_lim[k, :]

            # Position level: Signorini via proximal
            prox_kappa = prox_R_plus(Kappa_hatN[k] - self.r * gN[k])
            R_KappaN[k] = Kappa_hatN[k] - prox_kappa

            if prox_kappa > 0:  # Contact active
                A[k] = 1

                # Velocity level normal: proximal
                prox_PN = prox_R_plus(PN[k] - self.r * ksiN[k])
                R_LambdaN[k] = PN[k] - prox_PN
                B[k] = 1 if (PN[k] - self.r * ksiN[k] >= 0) else 0

                # Acceleration level normal: proximal
                prox_lN = prox_R_plus(lambdaN[k] - self.r * gNddot[k])
                R_lambdaN[k] = lambdaN[k] - prox_lN
                C[k] = 1 if (lambdaN[k] - self.r * gNddot[k] >= 0) else 0

                # Friction velocity level: Coulomb proximal with mu_k
                PF_aug = PF[fric_idx] - self.r * ksiF[fric_idx]
                R_LambdaF[fric_idx] = PF[fric_idx] - prox_coulomb_2d(PF_aug, self.mu_k * prox_PN)
                D[k] = 1 if (np.linalg.norm(PF_aug) <= self.mu_k * prox_PN) else 0

                # Friction acceleration level: Coulomb proximal with mu_k
                lF_aug = lambdaF[fric_idx] - self.r * gammaFdot[fric_idx]
                R_lambdaF[fric_idx] = lambdaF[fric_idx] - prox_coulomb_2d(lF_aug, self.mu_k * prox_lN)
                E[k] = 1 if (np.linalg.norm(lF_aug) <= self.mu_k * prox_lN) else 0

            else:  # No position contact — all contact forces zero
                R_LambdaN[k] = PN[k]
                R_lambdaN[k] = lambdaN[k]
                R_LambdaF[fric_idx] = PF[fric_idx]
                R_lambdaF[fric_idx] = lambdaF[fric_idx]

        contacts_nu = np.concatenate((A, B, C, D, E))
        Rc = np.concatenate((R_KappaN, R_LambdaN, R_lambdaN, R_LambdaF, R_lambdaF))
        R_full = np.concatenate((Rs, Rc))

        return R_full, AV, q, u, gNdot, gammaF, contacts_nu

    # ══════════════════════════════════════════════════════════
    #  JACOBIAN  (finite differencing of semi-smooth residual)
    # ══════════════════════════════════════════════════════════
    def get_R_J(self, iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):
        epsilon = 1e-6
        R, AV, q, u, gNdot, gammaF, contacts_nu = \
            self.get_R(iter, X, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF)

        J = np.zeros((self.nX, self.nX))
        I_mat = np.identity(self.nX)
        for i in range(self.nX):
            R_plus, _, _, _, _, _, _ = self.get_R(iter, X + epsilon*I_mat[:, i], prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF)
            J[:, i] = (R_plus - R) / epsilon

        return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    # ══════════════════════════════════════════════════════════
    #  SEMI-SMOOTH NEWTON SOLVER  (proximal contact, backtracking)
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
                R_trial, _, _, _, _, _, _ = \
                    self.get_R(iter, X, prev_X, prev_AV, prev_q, prev_u,
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

    # ══════════════════════════════════════════════════════════
    #  UPDATE  (semi-smooth Newton for one time step)
    # ══════════════════════════════════════════════════════════
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
        final_norm_R = result["norm_R"]

        self.contacts_save[:, iter] = contacts_nu

        print(f"iter {iter}: CONVERGED, ||R||={final_norm_R:.3e}, nu={result['nu']}")
        try: self.f.write(f"\niter {iter}: CONVERGED, ||R||={final_norm_R:.3e}")
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

        a = X[0:self.ndof]
        # Recompute gNddot from the converged state (general contact normals)
        _, _, gNddot, _, _, _, _ = self._compute_contact(q, u, a)
        lambda_N_start = 3*self.ndof + 3*self.ng + 2*self.ngamma + 2*self.nN
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
        self.AV_save[:, iter] = AV
        self.rho_inf_save[iter] = self.rho_inf
        self.alpha_save[iter] = alpha_min
        self.save_arrays()

    # ══════════════════════════════════════════════════════════
    #  SOLVE  (main loop with adaptive Δt)
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
                    try: self.f.write(f"\niter {iter}: restoring dtime -> {self.dtime_initial:.6e}")
                    except Exception: pass
                    self.dtime = self.dtime_initial
                iter += 1
            except MaxNewtonIterAttainedError:
                if self.dtime > min_dtime + 1e-20:
                    new_dtime = self.dtime / 2
                    print(f"*** iter {iter}: halving dtime {self.dtime:.6e} -> {new_dtime:.6e}")
                    try: self.f.write(f"\niter {iter}: halving dtime -> {new_dtime:.6e}")
                    except Exception: pass
                    self.dtime = new_dtime
                    dtime_halve_count += 1
                else:
                    print(f"*** iter {iter}: dtime at minimum. Aborting.")
                    try: self.f.write(f"\niter {iter}: dtime at minimum. Aborting.")
                    except Exception: pass
                    break

        # ── Summary ───────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"SUMMARY: {iter-1} steps completed")
        print(f"dtime halved {dtime_halve_count} times.")
        print(f"{'='*60}\n")
        try:
            self.f.write(f"\n{'='*60}")
            self.f.write(f"\nSUMMARY: {iter-1} steps completed")
            self.f.write(f"\ndtime halved {dtime_halve_count} times.")
            self.f.write(f"\n{'='*60}")
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

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('3D Sphere + Pendulum', fontsize=16, fontweight='bold')
        labels = ['x', 'y', 'z', 'ψ', 'θ', 'φ', 'α_p', 'β_p']
        for i in range(8):
            ax = axes[i // 3, i % 3]
            ax.plot(self.t, self.q_save[i, :], linewidth=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel(labels[i])
            ax.set_title(labels[i])
            ax.grid(True, alpha=0.3)
        # last subplot: normal forces (tread + riser)
        ax = axes[2, 2]
        ax.plot(self.t, self.lambdaN_save[0, :], 'g-', linewidth=1.5, label='λN tread')
        ax.plot(self.t, self.lambdaN_save[1, :], 'r-', linewidth=1.5, label='λN riser')
        ax.set_xlabel('Time')
        ax.set_ylabel('λN')
        ax.set_title('Normal Contact Forces')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(self.output_path, 'results_3d.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {out}")
        plt.close()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Prescribed pendulum: constant rotation in alpha_p
    # omega_p is in nondimensional units (rad / nondim_time), same convention as old hoop code
    omega_p = -7.0
    alpha_p0 = 2*np.pi/3  # pendulum starts pointing down (z-direction in body frame)
    beta_p0  = np.pi  # beta_p0 is ignored since beta_p is fixed at 0, but set to same value as alpha_p0 for consistent initial conditions
    def alpha_func(t):
        return alpha_p0 + omega_p * t

    def alpha_dot_func(t):
        return omega_p

    def alpha_ddot_func(t):
        return 0.0

    # beta_p fixed at 0 (pendulum rotates in xz-plane of body frame)
    def beta_func(t):
        return 0.0

    def beta_dot_func(t):
        return 0.0

    def beta_ddot_func(t):
        return 0.0

    sim = Simulation(
        ntime=1000,
        mu_s=0.3,
        mu_k=0.3,
        eN=0.0,
        eF=0.0,
        R=0.18,
        m_sphere=1.0,
        m_pendulum=0.5,
        l_pendulum=0.15,
        x0=-0.5, y0=0.0, z0=0.18,   # sitting on ground
        psi0=0.0, theta0=0.0, phi0=0.0,
        alpha_p0=alpha_p0, beta_p0=beta_p0,
        # stairs (set n_stairs=0 for flat ground)
        n_stairs=2, stair_width=1.8, stair_height=0.1,
        stair_x_start=0.5, fillet_radius=0,
        alpha_p_func=alpha_func,
        alpha_p_dot_func=alpha_dot_func,
        alpha_p_ddot_func=alpha_ddot_func,
        beta_p_func= lambda t: 0.0,  # beta_p fixed at 0
        beta_p_dot_func= lambda t: 0.0,  # beta_p_dot fixed at 0
        beta_p_ddot_func= lambda t: 0.0,  # beta_p_ddot fixed at 0
    )
    sim.solve()
    sim.plot_results()
