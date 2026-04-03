import numpy as np
import os
try:
    import scipy.io as sio  # type: ignore
except Exception:  # pragma: no cover
    sio = None
from datetime import datetime
import shutil
import json


# creating custom exceptions
class MaxNewtonIterAttainedError(Exception):
    """This exception is raised when the maximum number of Newton iterations is attained
      whilst the iterations have not yet converged and the solution was not yet obtained."""
    def __init__(self, message="This exception is raised when the maximum number of Newton iterations is attained."):
        self.message = message
        super().__init__(self.message)

class NoOpenContactError(Exception):
    """Contact is not open."""
    def __init__(self, message="This exception is raised when the contact is not open."):
        self.message = message
        super().__init__(self.message)

class RhoInfInfiniteLoop(Exception):
    """This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."""
    def __init__(self, message="This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."):
        self.message = message
        super().__init__(self.message)

class MaxHoursAttained(Exception):
    """This exception is raised when the maximum number of run hours specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum run time is exceeded."):
        self.message = message
        super().__init__(self.message)

class JacobianBlowingUpError(Exception):
    """This exception is raised when the Jacobian is blowing up."""
    def __init__(self, message="This exception is raised when the Jacobian is blowing up."):
        self.message = message
        super().__init__(self.message)

class Simulation:
    def __init__(
        self,
        ntime=5,
        mu_s=10**9,
        mu_k=0.3,
        eN=0,
        eF=0,
        R=0.1,
        m_hoop=1.0,
        m_pendulum=0.1,
        l_pendulum=0.5,
        theta0=0.0,
        omega0=0.0,
        y0=0.0,
        x0=0.0,
        phi_func=None,
        phi_dot_func=None,
        phi_ddot_func=None,
        n_stairs=0,
        stair_width=0.3,
        stair_height=0.05,
        stair_x_start=0.5,
        fillet_radius=0,
    ):
        # path for outputs
        # Generate timestamp
        timestamp = datetime.now().strftime("hoop_pendulum_%Y-%m-%d_%H-%M-%S")
        outputs_dir = f"outputs/{timestamp}"
        self.output_path = os.path.join(os.getcwd(), outputs_dir)  # Output path
        os.makedirs(self.output_path, exist_ok=True)

        # Path to the current file
        current_file = os.path.realpath(__file__)
        # Copy the file
        shutil.copy2(current_file, self.output_path)

        # friction coefficients
        self.mu_s = mu_s    # Static friction coefficient
        self.mu_k = mu_k    # Kinetic friction coefficient
        # restitution coefficients
        self.eN = eN        # normal coefficient of restitution
        self.eF = eF        # friction coefficient of restitution
        # nondimensionalization parameters
        l_nd = 1       # m, length nondimensionalization paramter
        m_nd = 1       # kg, mass nondimensionalization parameter
        a_nd = 9.81    # m/(s**2), acceleration nondimensionalization parameter
        t_nd = np.sqrt(l_nd/a_nd)   # s, time nondimensionalization parameter
        # simulation (time) parameters
        self.dtime = 2e-3/t_nd # time step duration
        self.dtime_initial = self.dtime  # save original for adaptive stepping
        self.ntime = ntime           # number of iterations
        self.tf = self.ntime*self.dtime            # final time
        self.t = np.linspace(0,self.tf,self.ntime) # time array
        # - geometric radius: R
        # - hoop mass: m_hoop (thin ring)
        self.R = float(R) / l_nd
        self.l_pendulum = float(l_pendulum) / l_nd
        self.m_hoop = float(m_hoop) / m_nd
        self.m_pendulum = float(m_pendulum) / m_nd
        self.m = self.m_hoop + self.m_pendulum

        # Stair properties
        self.n_stairs = int(n_stairs)
        self.stair_width = float(stair_width) / l_nd
        self.stair_height = float(stair_height) / l_nd
        self.stair_x_start = float(stair_x_start) / l_nd
        self.fillet_radius = float(fillet_radius) / l_nd
        self._build_stair_profile()
        self._classify_segments()

        # Inertia about COM (z-axis)
        self.I_h = self.m_hoop * self.R**2
        # Total rotational inertia: hoop + point mass pendulum at distance l_pendulum
        self.I = self.m_hoop * self.R**2
        # Determine if pendulum is prescribed or free.
        # If phi_func is a scalar (or None) and dot funcs are None => free pendulum.
        # If phi_func is callable => prescribed pendulum with bilateral constraint.
        if callable(phi_func):
            # Prescribed mode
            self.prescribed_phi = True
            self.phi_func = phi_func
            self.phi_dot_func = phi_dot_func if callable(phi_dot_func) else (lambda t: 0.0)
            self.phi_ddot_func = phi_ddot_func if callable(phi_ddot_func) else (lambda t: 0.0)
            self.phi0 = self.phi_func(0.0)
            self.phi_dot0 = self.phi_dot_func(0.0)
        else:
            # Free mode: phi_func is a scalar initial angle (or None => 0)
            self.prescribed_phi = False
            self.phi0 = float(phi_func) if phi_func is not None else 0.0
            self.phi_dot0 = float(phi_dot_func) if phi_dot_func is not None else 0.0
            # Dummy functions (not used in residual, but kept for compatibility)
            self.phi_func = lambda t: self.phi0
            self.phi_dot_func = lambda t: 0.0
            self.phi_ddot_func = lambda t: 0.0

        # nondimensional constants
        self.gr = 9.81/a_nd    # gravitational acceleration
        # total degrees of freedom
        self.ndof = 4       # x, y, theta, phi (phi now a full DOF)
        # constraint count
        self.ng = 1 if self.prescribed_phi else 0   # bilateral constraint only when prescribed
        self.ngamma = 0      # number of constraints at velocity level
        # Ground contact model is always present
        self.nN = 2          # number of gap distance constraints (gNx, gNy)
        self.nF = 2          # number of friction constraints
        self.gammaF_lim = np.array([[0], [1]])    # normal 0 (gNx) -> friction 0, normal 1 (gNy) -> friction 1
        self.nX = 3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF     # total number of constraints with their derivative
        # generalized alpha parameters
        self.MAXITERn = 100
        self.MAXITERn_initial = self.MAXITERn   # saving initial value of MAXITERn
        self.r = 0.3
        self.rho_inf = 0.5
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2
        self.tol_n = 1.0e-4     # error tolerance
        # Note: Mass matrix is time-varying and computed in get_R() method
        
        # Initial pendulum values
        phi_init = self.phi0
        
        # applied forces (weight)
        self.q_save = np.zeros((self.ndof,self.ntime))
        self.u_save = np.zeros((self.ndof,self.ntime))
        self.X_save = np.zeros((self.nX,self.ntime))
        self.gNdot_save = np.zeros((self.nN,self.ntime))
        self.gNddot_save = np.zeros((self.nN,self.ntime))
        self.gammaF_save = np.zeros((self.nF,self.ntime))
        self.lambdaN_save = np.zeros((self.nN,self.ntime))
        self.lambdaF_save = np.zeros((self.nF,self.ntime))
        self.AV_save = np.zeros((self.ndof+self.nN+self.nF,self.ntime))
        self.contacts_save = np.zeros((5*self.nN,self.ntime))
        self.gNx_save = np.zeros(self.ntime)
        self.gNy_save = np.zeros(self.ntime)
        self.dual_contact_save = np.zeros(self.ntime, dtype=int)
        self.rho_inf_save = np.full(self.ntime, self.rho_inf)
        self.dtime_save = np.full(self.ntime, self.dtime)
        self.alpha_save = np.ones(self.ntime)  # line-search alpha (min over Newton iters)
        
        # Store prescribed pendulum data
        self.phi_save = np.zeros(self.ntime)
        self.phi_dot_save = np.zeros(self.ntime)
        self.phi_ddot_save = np.zeros(self.ntime)
    
        # initial position (COM coordinates)
        theta0 = float(theta0)
        self.x0_input = float(x0)
        self.y0_input = float(y0)
        self.theta0_input = float(theta0)
        self.omega0_input = float(omega0)
        y0_nd = y0
        x0_nd = x0 
        q0 = np.array([float(x0_nd), float(y0_nd), theta0, self.phi0])
        self.q_save[:,0] = q0
        # initial velocity
        # User-specified initial angular velocity (about +z)
        u0 = np.array([0.0, 0.0, float(omega0), self.phi_dot0])
        self.u_save[:,0] = u0
           
        # Store initial pendulum values
        self.phi_save[0] = self.phi0
        self.phi_dot_save[0] = self.phi_dot0
        self.phi_ddot_save[0] = 0.0
    
        # creating an output file f to log major happenings
        self.f = open(f"{self.output_path}/log_file.txt",'a')

        # Save metadata once for visualization/reproducibility
        self._save_metadata()

    def _build_stair_profile(self):
        """Build the stair profile as a list of line segments [(x1,y1,x2,y2), ...].

        When ``fillet_radius > 0``, each convex corner at the top of a riser
        is replaced by a quarter-circle arc of that radius.  The riser is
        shortened at its top by *r_f* and the tread is shortened at its left
        end by *r_f*.  The arc data is stored in ``self.fillet_arcs``.
        """
        FAR = 1e6  # effectively +/- infinity
        segs = []
        r_f = self.fillet_radius
        self.fillet_arcs = []  # list of (center_x, center_y, fillet_radius)
        if self.n_stairs == 0:
            # Flat ground
            segs.append((-FAR, 0.0, FAR, 0.0))
        else:
            # Ground before stairs
            segs.append((-FAR, 0.0, self.stair_x_start, 0.0))
            for i in range(self.n_stairs): # coordinate construction (every loop build 1 riser and 1 tread)
                x_left = self.stair_x_start + i * self.stair_width
                h_prev = i * self.stair_height # previous height y1
                h_curr = (i + 1) * self.stair_height # current height y2
                x_right = x_left + self.stair_width
                if r_f > 0:
                    # Shortened riser: top reduced by fillet radius
                    segs.append((x_left, h_prev, x_left, h_curr - r_f))
                    # Fillet arc at the convex corner (top of riser / left of tread)
                    # Arc center sits inside the corner, arc spans angle pi to pi/2
                    self.fillet_arcs.append((x_left + r_f, h_curr - r_f, r_f))
                    # Shortened tread: left end pushed right by fillet radius
                    if i < self.n_stairs - 1:
                        segs.append((x_left + r_f, h_curr, x_right, h_curr))
                    else:
                        segs.append((x_left + r_f, h_curr, FAR, h_curr))
                else:
                    # Original sharp corners
                    segs.append((x_left, h_prev, x_left, h_curr))
                    if i < self.n_stairs - 1:
                        segs.append((x_left, h_curr, x_right, h_curr))
                    else:
                        segs.append((x_left, h_curr, FAR, h_curr))
        self.stair_segments = segs  #(x1,y1,x2,y2 ...)

    def _classify_segments(self):
        """Classify stair segments into horizontal (treads) and vertical (risers)."""
        self.horizontal_segments = []
        self.vertical_segments = []
        for seg in self.stair_segments:
            x1, y1, x2, y2 = seg
            dx = abs(x2 - x1)  # horizontal step
            dy = abs(y2 - y1) # vertical step 
            if dy < 1e-12:
                self.horizontal_segments.append(seg)
            elif dx < 1e-12:
                self.vertical_segments.append(seg)

    def _compute_riser_gap(self, px, py):  # px, py = ball center coordinates
        """Compute gap to the nearest riser (vertical wall) and stair corner.

        For each vertical segment, finds the closest point on the finite
        segment by clamping the ball center's y-coordinate to [y_bot, y_top].
        Three cases arise:

        1. Ball center is level with the riser (y_bot <= py <= y_top):
           closest point is directly on the wall, pure horizontal normal.
        2. Ball center is above/below the riser span:
           closest point is the top/bottom corner of the riser 
           normal pointing from the corner toward the ball center.
        3. Ball center coincides with the closest point (dist ≈ 0):
           falls back to a horizontal normal toward the ball side.
           This is just a safety case so we won't divide by zero.

        Returns (gap, nx, ny) where:
            gap      = distance from ball center to closest point, minus R
            (nx, ny) = unit outward normal from closest point toward ball center
        If no riser exists, returns (1e10, 1.0, 0.0) — effectively open.
        """
        best_gap = 1e10 # initialize infinite gap
        best_nx, best_ny = 1.0, 0.0

        # Loop over every vertical (riser) segment and find the one closest to the ball
        for seg in self.vertical_segments:
            x1, y1, x2, y2 = seg

            # The riser is a vertical line at x = x_wall, spanning from y_bot to y_top
            x_wall = x1             # vertical segment: x1 == x2
            y_bot = min(y1, y2)   # classify the endpoints as bottom and top of the riser
            y_top = max(y1, y2)

            # Find the closest point (cx, cy) on this finite segment to the ball center (px, py):
            # cx is always x_wall (the riser is vertical)
            # cy is the ball's y-coordinate clamped to [y_bot, y_top]:
            #  if py is between y_bot and y_top → cy = py  (ball is level with riser)
            #  if py > y_top → cy = y_top  (closest point is the top corner)
            cx = x_wall
            cy = max(y_bot, min(y_top, py))   # clamp to segment span

            # Vector from closest point to ball center
            dx = px - cx
            dy = py - cy

            # Euclidean distance from ball center to closest point
            dist = np.sqrt(dx * dx + dy * dy)

            # Gap = distance minus radius (negative means penetration)
            gap = dist - self.R

            # Keep track of the riser with the smallest gap
            if gap < best_gap:
                best_gap = gap

                if dist < 1e-15:
                    # Degenerate case: ball center is exactly on the closest point.
                    # Use a pure horizontal normal pointing away from the wall.
                    best_nx = 1.0 if px >= cx else -1.0
                    best_ny = 0.0
                else:
                    # Normal case: unit normal = (dx, dy) / dist
                    # Points from the closest point on the riser toward the ball center.
                    # When the ball is level with the riser → pure horizontal (ny=0).
                    # When the ball is above/below the riser → diagonal (corner contact).
                    best_nx = dx / dist
                    best_ny = dy / dist

        # Also check fillet arcs (quarter-circle replacements of sharp corners)
        for cx_a, cy_a, r_f in self.fillet_arcs:
            dx = px - cx_a
            dy = py - cy_a
            # The fillet arc lives in the 2nd quadrant relative to its center
            # (dx <= 0, dy >= 0), spanning angle pi/2 to pi.
            if dx > 0 or dy < 0:
                continue  # ball is outside the fillet zone
            dist = np.sqrt(dx * dx + dy * dy)
            gap = dist - r_f - self.R
            if gap < best_gap:
                best_gap = gap
                if dist < 1e-15:
                    best_nx = -1.0
                    best_ny = 0.0
                else:
                    best_nx = dx / dist
                    best_ny = dy / dist

        return best_gap, best_nx, best_ny

    def _compute_tread_gap(self, px, py):
        """Compute vertical gap to the nearest tread using center-projection.

        Only considers treads where the ball *center* is strictly within
        the tread's horizontal span (x_left < px < x_right).  Edges where
        a riser meets a tread are handled by ``_compute_riser_gap`` instead,
        which owns the corner contact.

        Returns (gap, nx, ny) where:
            gap      = perpendicular distance minus R
            (nx, ny) = (0, +1) normal pointing from tread toward ball
        If no tread is projectable, returns (1e10, 0.0, 1.0) — effectively open.
        """
        # Start with a very large gap (no contact) and a default upward normal
        best_gap = 1e10
        best_nx, best_ny = 0.0, 1.0

        # Loop over every horizontal (tread) segment
        for seg in self.horizontal_segments:
            x1, y1, x2, y2 = seg

            # The tread is a horizontal line at y = y_tread, spanning from x_left to x_right
            y_tread = y1            # horizontal segment: y1 == y2
            x_left = min(x1, x2)
            x_right = max(x1, x2)

            # Only consider this tread if the ball center is strictly inside its horizontal span.
            if px <= x_left or px >= x_right:
                continue

            # Signed vertical distance from tread to ball center
            dy = py - y_tread

            # Gap = perpendicular distance minus radius (negative means penetration)
            gap = abs(dy) - self.R

            # Keep track of the tread with the smallest gap
            if gap < best_gap:
                best_gap = gap
                # Normal is purely vertical (0,1)
                best_nx = 0.0
                best_ny = 1.0 if dy >= 0 else -1.0

        return best_gap, best_nx, best_ny

    def _save_metadata(self):
        """Save run parameters and derived quantities for post-processing."""
        params = {
            "dtime": float(self.dtime),
            "ntime": int(self.ntime),
            "R": float(self.R),
            "m_hoop": float(self.m_hoop),
            "m_pendulum": float(self.m_pendulum),
            "m_total": float(self.m),
            "l_pendulum": float(self.l_pendulum),
            "I": float(self.I),
            "mu_s": float(self.mu_s),
            "mu_k": float(self.mu_k),
            "eN": float(self.eN),
            "eF": float(self.eF),
            "x0_input": float(self.x0_input),
            "y0_input": None if self.y0_input is None else float(self.y0_input),
            "theta0_input": float(self.theta0_input),
            "omega0_input": float(self.omega0_input),
            "phi0_input": float(self.phi_save[0]),
            "q0": [float(self.q_save[0,0]), float(self.q_save[1,0]), float(self.q_save[2,0])],
            "u0": [float(self.u_save[0,0]), float(self.u_save[1,0]), float(self.u_save[2,0])],
            "rho_inf": float(self.rho_inf),
            "alpha_m": float(self.alpha_m),
            "alpha_f": float(self.alpha_f),
            "gamma": float(self.gama),
            "beta": float(self.beta),
            "n_stairs": self.n_stairs,
            "stair_width": float(self.stair_width),
            "stair_height": float(self.stair_height),
            "stair_x_start": float(self.stair_x_start),
            "fillet_radius": float(self.fillet_radius),
        }
        with open(os.path.join(self.output_path, "params.json"), "w", encoding="utf-8") as fp:
            json.dump(params, fp, indent=2)

    def save_arrays(self):
        """Saving arrays."""
        if sio is not None:
            file_name_q = str(f'{self.output_path}/q.mat')
            sio.savemat(file_name_q,dict(q=self.q_save))

            file_name_u = str(f'{self.output_path}/u.mat')
            sio.savemat(file_name_u,dict(u=self.u_save))

            file_name_x_save = str(f'{self.output_path}/x_save.mat')
            sio.savemat(file_name_x_save,dict(X=self.X_save))

            file_name_contacts = str(f'{self.output_path}/contacts.mat')
            sio.savemat(file_name_contacts,dict(contacts=self.contacts_save))
        else:
            try:
                self.f.write("\n  SciPy unavailable; skipping .mat outputs")
            except Exception:
                pass

        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gNddot_save.npy', self.gNddot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/lambdaN_save.npy', self.lambdaN_save)
        np.save(f'{self.output_path}/lambdaF_save.npy', self.lambdaF_save)
        np.save(f'{self.output_path}/phi_save.npy', self.phi_save)
        np.save(f'{self.output_path}/phi_dot_save.npy', self.phi_dot_save)
        np.save(f'{self.output_path}/phi_ddot_save.npy', self.phi_ddot_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        # Save stair geometry as (n_segments, 4) array: [x1, y1, x2, y2]
        np.save(f'{self.output_path}/stair_segments.npy', np.array(self.stair_segments))
        # Save separated gaps and dual contact flag
        np.save(f'{self.output_path}/gNx_save.npy', self.gNx_save)
        np.save(f'{self.output_path}/gNy_save.npy', self.gNy_save)
        np.save(f'{self.output_path}/dual_contact_save.npy', self.dual_contact_save)
        np.save(f'{self.output_path}/rho_inf_save.npy', self.rho_inf_save)
        np.save(f'{self.output_path}/dtime_save.npy', self.dtime_save)
        np.save(f'{self.output_path}/alpha_save.npy', self.alpha_save)
        return
 
    def get_R(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*index_sets):
        """Calculates the residual.
        Note: ndof=3, but we track 4D state with prescribed phi.
        """

        [prev_a,_,_,_,_,_,_,_,_,_,prev_lambdaN,_,prev_lambdaF] = self.get_X_components(prev_X)
        [a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,
            KappaN,LambdaN,lambdaN,LambdaF,lambdaF] = self.get_X_components(X)
        
        # With ndof=4, a, U, Q are already 4D
        a_full = a
        U_full = U
        Q_full = Q
        
        # AV - Auxiliary Variables [abar, lambdaNbar, lambdaFbar]
        prev_abar = prev_AV[0:self.ndof]
        prev_lambdaNbar = prev_AV[self.ndof:self.ndof+self.nN]
        prev_lambdaFbar = prev_AV[self.ndof+self.nN:self.ndof+self.nN+self.nF]

        # auxiliary variables update (extended to 4D)
        # eq. 49
        abar = (self.alpha_f*prev_a+(1-self.alpha_f)*a-self.alpha_m*prev_abar)/(1-self.alpha_m)
        # eq. 96
        lambdaNbar = (self.alpha_f*prev_lambdaN+(1-self.alpha_f)*lambdaN-self.alpha_m*prev_lambdaNbar)/(1-self.alpha_m)
        # eq. 114
        lambdaFbar = (self.alpha_f*prev_lambdaF+(1-self.alpha_f)*lambdaF-self.alpha_m*prev_lambdaFbar)/(1-self.alpha_m)
        
        AV = np.concatenate((abar,lambdaNbar,lambdaFbar),axis=None)

        # velocity update (73) 
        u = prev_u+self.dtime*((1-self.gama)*prev_abar+self.gama*abar)+U
        
        # position update (73)
        q = prev_q+self.dtime*prev_u+self.dtime**2/2*((1-2*self.beta)*prev_abar+2*self.beta*abar)+Q
        
        # Use current iterate values for phi (bilateral constraint will enforce prescribed motion)
        phi_current = q[3]

        # Mass matrix (4x4 including prescribed phi DOF)
        # Includes hoop translation (x, y), rotation (theta), and pendulum angle (phi)
        M = np.array([[self.m, 0, 0, -self.m_pendulum*self.l_pendulum*np.sin(phi_current)],
                      [0, self.m, 0, self.m_pendulum*self.l_pendulum*np.cos(phi_current)],
                      [0, 0, self.m_hoop*self.R**2, 0],
                      [-self.m_pendulum*self.l_pendulum*np.sin(phi_current), self.m_pendulum*self.l_pendulum*np.cos(phi_current), 0, self.m_pendulum*self.l_pendulum**2]])

        # External forces only (constraint reaction handled by bilateral constraint multiplier lambda_g)
        self.force = np.array([[0, -self.m*self.gr, 0, -self.m_pendulum*self.gr*self.l_pendulum*np.cos(phi_current)]])

        # Velocity-dependent (centrifugal/Coriolis) terms
        velocity_res = np.array([u[3]**2*np.cos(q[3])*self.m_pendulum*self.l_pendulum,u[3]**2*np.sin(q[3])*self.m_pendulum*self.l_pendulum,0,0])

        # bilateral constraint at position level
        if self.prescribed_phi:
            # Prescribed mode: phi - phi_prescribed(t) = 0
            g = np.array([q[3] - self.phi_func(self.t[iter])])
            gdot = np.array([u[3] - self.phi_dot_func(self.t[iter])])
            gddot = np.array([a[3] - self.phi_ddot_func(self.t[iter])])
            Wg = np.array([[0, 0, 0, 1]]).T
        else:
            # Free mode: no bilateral constraint
            g = np.zeros((self.ng))
            gdot = np.zeros((self.ng))
            gddot = np.zeros((self.ng))
            Wg = np.zeros((4, self.ng))

        # bilateral constraints at velocity level
        gamma = np.zeros((self.ngamma))
        gammadot = np.zeros((self.ngamma))
        Wgamma = np.zeros((4,self.ngamma))

        # Separated normal gaps: gNx (vertical/riser) and gNy (horizontal/tread)
        # gNx checks all risers and picks the closest (including corner contacts).
        # gNy only considers treads whose horizontal span contains the ball centre.

        # gNx: gap to nearest riser (includes corner contacts)
        gNx, nx_0, ny_0 = self._compute_riser_gap(q[0], q[1])

        # gNy: gap to nearest tread (centre-projection only)
        gNy, nx_1, ny_1 = self._compute_tread_gap(q[0], q[1])

        gN = np.array([gNx, gNy])
        # General normal-velocity projection (works for diagonal normals at corners)
        gNdot = np.array([nx_0 * u[0] + ny_0 * u[1],
                          nx_1 * u[0] + ny_1 * u[1]])
        gNddot = np.array([nx_0 * a[0] + ny_0 * a[1],
                           nx_1 * a[0] + ny_1 * a[1]])
        WN = np.array([[nx_0, ny_0, 0, 0],
                        [nx_1, ny_1, 0, 0]]).T    # (4, 2)
        # Friction tangent for contact k with normal (nx_k, ny_k) is (ny_k, -nx_k).
        # Tangential contact-point velocity = tangent · v_contact, with rolling:
        #   gammaF_k = ny_k * u[0] - nx_k * u[1] + R * u[2]
        gammaF = np.array([ny_0 * u[0] - nx_0 * u[1] + self.R * u[2],
                           ny_1 * u[0] - nx_1 * u[1] + self.R * u[2]])
        gammaFdot = np.array([ny_0 * a[0] - nx_0 * a[1] + self.R * a[2],
                              ny_1 * a[0] - nx_1 * a[1] + self.R * a[2]])
        WF = np.array([[ny_0, -nx_0, self.R, 0],
                        [ny_1, -nx_1, self.R, 0]]).T   # (4, 2)

        # eq. 44
        ksiN = gNdot+self.eN*prev_gNdot
        # discrete normal percussion eq. 95
        PN = LambdaN+self.dtime*((1-self.gama)*prev_lambdaNbar+self.gama*lambdaNbar)
        # eq. 102
        Kappa_hatN = KappaN+self.dtime**2/2*((1-2*self.beta)*prev_lambdaNbar+2*self.beta*lambdaNbar)

        # eq. 48 (use friction restitution coefficient)
        ksiF = gammaF+self.eF*prev_gammaF
        # eq. 113
        PF = LambdaF+self.dtime*((1-self.gama)*prev_lambdaFbar+self.gama*lambdaFbar)
            
        Rs = np.concatenate(([M@a_full-self.force[0,:]-velocity_res-Wg@lambda_g-Wgamma@lambda_gamma-WN@lambdaN-WF@lambdaF],
                [M@U_full-Wg@Lambda_g-Wgamma@Lambda_gamma-WN@LambdaN-WF@LambdaF],
                [M@Q_full-Wg@Kappa_g-WN@KappaN-self.dtime/2*(Wgamma@Lambda_gamma+WF@LambdaF)],
                g,
                gdot,
                gddot,
                gamma,
                gammadot),axis=None)
        
        # Contact residual Rc
        R_KappaN = np.zeros(self.nN)   # (129)
        R_LambdaN = np.zeros(self.nN)
        R_lambdaN = np.zeros(self.nN)
        R_LambdaF = np.zeros(self.nF)  # (138)
        R_lambdaF = np.zeros(self.nF)  # (142)

        if index_sets == ():
            A = np.zeros(self.nN, dtype=int)
            B = np.zeros(self.nN, dtype=int)
            C = np.zeros(self.nN, dtype=int)
            D = np.zeros(self.nN, dtype=int)
            E = np.zeros(self.nN, dtype=int)

            for i in range(self.nN):
                # A: contact active if gap is closed (position-level check)
                if self.r*gN[i] - Kappa_hatN[i] <=0:
                    A[i] = 1
                    # D: stick at impulse level — tangential impulse inside friction cone
                    if np.linalg.norm(self.r*ksiF[self.gammaF_lim[i,:]]-PF[self.gammaF_lim[i,:]])<=self.mu_s*(PN[i]):
                        D[i] = 1
                        # E: stick at force level — tangential force inside friction cone
                        if np.linalg.norm(self.r*gammaFdot[self.gammaF_lim[i,:]]-lambdaF[self.gammaF_lim[i,:]])<=self.mu_s*(lambdaN[i]):
                            E[i] = 1
                    # B: impacting if closing velocity check passes
                    if self.r*ksiN[i]-PN[i] <= 0:
                        B[i] = 1
                        # C: persistent contact if closing acceleration check passes
                        if self.r*gNddot[i]-lambdaN[i] <= 0:
                            C[i] = 1
        else:
            A = index_sets[0]
            B = index_sets[1]
            C = index_sets[2]
            D = index_sets[3]
            E = index_sets[4]

        # Determine dual contact status (both gNx and gNy active)
        both_active = (A[0] == 1 and A[1] == 1)

        # ── CONTACT RESIDUAL ───────────────────────────────────────
        # For each contact point k, the residual equations depend on A/B/C/D/E:
        #
        #   A[k]=1 (active):  R_ΚN = gN          (enforce zero gap)
        #          =0 (open):  R_ΚN = ̂Κ_N         (zero normal impulse)
        #
        #   B[k]=1 (impact):  R_ΛN = ξ_N          (enforce impact law)
        #          =0:         R_ΛN = P_N          (zero percus. impulse)
        #
        #   C[k]=1 (persist): R_λN = ä_N          (enforce zero normal accel)
        #          =0:         R_λN = λ_N          (zero normal force)
        #
        #   D[k]=1 (stick):   R_ΛF = ξ_F          (zero slip velocity)
        #          =0 (slip):  R_ΛF = P_F + μ_k·P_N·sign(ξ_F)  (Coulomb slip)
        #
        #   E[k]=1 (stick):   R_λF = ̇γ_F         (zero slip accel)
        #          =0 (slip):  R_λF = λ_F + μ_k·λ_N·sign(̇γ_F) (Coulomb slip force)
        for k in range(self.nN):
            if A[k]:
                R_KappaN[k] = gN[k]
                if D[k]:
                    R_LambdaF[self.gammaF_lim[k,:]] = ksiF[self.gammaF_lim[k,:]]
                    if E[k]:
                        R_lambdaF[self.gammaF_lim[k,:]] = gammaFdot[self.gammaF_lim[k,:]]
                    else:
                        R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaFdot[self.gammaF_lim[k,:]])                    
                else:
                    R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]+self.mu_k*PN[k]*np.sign(ksiF[self.gammaF_lim[k,:]])
                    R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaF[self.gammaF_lim[k,:]])
            else:
                R_KappaN[k] = Kappa_hatN[k]
                R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]
                R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]
            # (132)
            if B[k]:
                R_LambdaN[k] = ksiN[k]
            else:
                R_LambdaN[k] = PN[k]
            # (135)
            if C[k]:
                R_lambdaN[k] = gNddot[k]
            else:
                R_lambdaN[k] = lambdaN[k]


        Rc = np.concatenate((R_KappaN, R_LambdaN, R_lambdaN, R_LambdaF, R_lambdaF),axis=None)
        
        R = np.concatenate([Rs, Rc],axis=None)


        if index_sets == ():
            # in this case, get_R is called to calculate the actual residual, not as part of calculating the Jacobian
            # print contact region indicators once per residual evaluation
            print(f"A={A}")
            print(f"B={B}")
            print(f"C={C}")
            print(f"D={D}")
            print(f"E={E}")
            if both_active:
                print("DUAL CONTACT")
            # Return 4D q, u now that phi is a full DOF
            return R, AV, q, u, gNdot, gammaF, A, B, C, D, E
        else:
            # Called with fixed index sets (during Jacobian finite differencing)
            return R, AV, q, u, gNdot, gammaF

    def get_R_J(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,forced_sets=None):
        '''Calculate the Jacobian by finite differencing.
        
        For each column i of J, perturb X[i] by epsilon and compute
        J[:,i] = (R(X+ε·e_i) - R(X)) / ε.
        The index sets are frozen to A/B/C/D/E during differencing
        so the Jacobian is consistent with the current contact mode.
        '''

        epsilon = 1e-6
        if forced_sets is not None:
            A, B, C, D, E = forced_sets
            R, AV, q, u, gNdot, gammaF = self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
        else:
            R, AV, q, u, gNdot, gammaF, A, B, C, D, E = self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
        contacts_nu = np.concatenate((A,B,C,D,E),axis=None)

        # Initializing the Jacobian
        J = np.zeros((self.nX,self.nX))
        I = np.identity(self.nX)

        # Constructing the Jacobian column by column
        for i in range(self.nX):
            # print(i)
            R_plus_epsilon,_,_,_,_,_ = self.get_R(iter,X+epsilon*I[:,i],prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
            J[:,i] = (R_plus_epsilon-R)/epsilon

        return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    def _newton_solve(self, iter, X0, prev_X, prev_AV, prev_q, prev_u,
                      prev_gNdot, prev_gammaF, forced_sets=None,
                      detect_chattering=False):
        """Run Newton iterations with backtracking line search.

        Parameters
        ----------
        iter : int
            Current time-step index.
        X0 : ndarray
            Initial guess for the state vector.
        prev_* : ndarray
            Previous time-step state (for Newmark updates).
        forced_sets : tuple or None
            If None, index sets are determined freely each iteration.
            If (A, B, C, D, E), those sets are locked throughout.
        detect_chattering : bool
            If True, monitor D/E sets for 2-cycle chattering and stalls.
            When detected, lock the best candidate configuration and
            continue. Only meaningful when forced_sets is None.

        Returns
        -------
        dict with keys:
            converged : bool
            X, AV, q, u, gNdot, gammaF : ndarray  (state at convergence)
            contacts : ndarray  (final A/B/C/D/E)
            alpha_min : float  (smallest line-search step used)
            norm_R : float  (final residual norm)
            forced_sets : tuple or None  (sets used, may differ from input
                          if chattering forced a lock)
            force_reason : str  (why forcing was triggered, or "")
        """
        X = X0.copy()
        R, AV, q, u, gNdot, gammaF, J, contacts_nu = \
            self.get_R_J(iter, X, prev_X, prev_AV, prev_q, prev_u,
                         prev_gNdot, prev_gammaF, forced_sets=forced_sets)
        norm_R = np.linalg.norm(R, np.inf)
        alpha_min = 1.0
        force_reason = ""

        # Chattering detection history
        nN = self.nN
        idx_history = [(tuple(contacts_nu[3*nN:]), norm_R, contacts_nu.copy())]

        nu = 0
        while norm_R > self.tol_n and nu < self.MAXITERn:
            # Solve J·δ = R
            delta = np.linalg.lstsq(J, R, rcond=None)[0]

            # Backtracking line search
            alpha = 1.0
            X_old = X.copy()
            for ls in range(8):
                X = X_old - alpha * delta
                R_trial, _, _, _, _, _, _, _ = \
                    self.get_R_J(iter, X, prev_X, prev_AV, prev_q, prev_u,
                                 prev_gNdot, prev_gammaF, forced_sets=forced_sets)
                norm_trial = np.linalg.norm(R_trial, np.inf)
                if norm_trial < norm_R or alpha < 1e-2:
                    break
                alpha *= 0.5
            alpha_min = min(alpha_min, alpha)

            nu += 1
            R, AV, q, u, gNdot, gammaF, J, contacts_nu = \
                self.get_R_J(iter, X, prev_X, prev_AV, prev_q, prev_u,
                             prev_gNdot, prev_gammaF, forced_sets=forced_sets)
            norm_R = np.linalg.norm(R, np.inf)

            # Chattering / stall detection
            if detect_chattering:
                de_key = tuple(contacts_nu[3*nN:])
                idx_history.append((de_key, norm_R, contacts_nu.copy()))

                trigger = False
                if forced_sets is None:
                    # 2-cycle detector
                    if len(idx_history) >= 5:
                        k0, k1, k2, k3 = [h[0] for h in idx_history[-4:]]
                        if k0 == k2 and k1 == k3 and k0 != k1:
                            force_reason = "2-cycle chattering in D/E"
                            trigger = True
                    # Stall detector
                    if not trigger and len(idx_history) >= 50:
                        recent = [h[1] for h in idx_history[-50:]]
                        if min(recent) > 0.95 * recent[0]:
                            force_reason = (f"stall (min {min(recent):.3e} vs "
                                            f"start {recent[0]:.3e} over 50 iters)")
                            trigger = True

                if trigger:
                    print(f"*** {force_reason} — forcing best config ***")
                    # Pick best candidate from history (the one with lowest norm_R among unique D/E configs)
                    seen = {}
                    for h_key, h_norm, h_contacts in idx_history:
                        fk = tuple(h_contacts)
                        if fk not in seen or h_norm < seen[fk][0]:
                            seen[fk] = (h_norm, h_contacts)
                    best_norm = np.inf
                    best_contacts = None
                    for fk, (h_norm, h_contacts) in seen.items():
                        cand = (h_contacts[0:nN].astype(int),
                                h_contacts[nN:2*nN].astype(int),
                                h_contacts[2*nN:3*nN].astype(int),
                                h_contacts[3*nN:4*nN].astype(int),
                                h_contacts[4*nN:5*nN].astype(int))
                        R_c, _, _, _, _, _, _, _ = self.get_R_J(
                            iter, X, prev_X, prev_AV, prev_q, prev_u,
                            prev_gNdot, prev_gammaF, forced_sets=cand)
                        nc = np.linalg.norm(R_c, np.inf)
                        if nc < best_norm:
                            best_norm = nc
                            best_contacts = h_contacts
                    forced_sets = (best_contacts[0:nN].astype(int),
                                   best_contacts[nN:2*nN].astype(int),
                                   best_contacts[2*nN:3*nN].astype(int),
                                   best_contacts[3*nN:4*nN].astype(int),
                                   best_contacts[4*nN:5*nN].astype(int))

        return {
            "converged": norm_R <= self.tol_n,
            "X": X, "AV": AV, "q": q, "u": u,
            "gNdot": gNdot, "gammaF": gammaF,
            "contacts": contacts_nu,
            "alpha_min": alpha_min,
            "norm_R": norm_R,
            "forced_sets": forced_sets,
            "force_reason": force_reason,
            "nu": nu,
        }

    def _check_self_consistency(self, iter, X, prev_X, prev_AV, prev_q,
                                prev_u, prev_gNdot, prev_gammaF, used_sets):
        """Check whether the index sets implied by the converged X match
        the sets that were used to obtain it.

        Returns (match: bool, free_sets: tuple of (A,B,C,D,E))
        """
        _, _, _, _, _, _, A, B, C, D, E = \
            self.get_R(iter, X, prev_X, prev_AV, prev_q, prev_u,
                       prev_gNdot, prev_gammaF)
        free_sets = (A, B, C, D, E)

        match = all(np.array_equal(u, f) for u, f in zip(used_sets, free_sets))
        return match, free_sets

    def update(self, iter, prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF):
        """Semi-smooth Newton solver for one time step.

        Algorithm:
        1. NEWTON SOLVE — free index sets, with chattering detection.
           If chattering is detected, the best D/E config is locked.
        2. SELF-CONSISTENCY CHECK — evaluate free sets at the converged X.
           If they match the sets used → VERIFIED, done.
        3. EXHAUSTIVE D/E SEARCH — try all 2^nN × 2^nN D/E combos.
           Each combo: force D/E → Newton → self-consistency check.
           Best self-consistent result wins → RESOLVED.
        4. FALLBACK — no self-consistent combo exists:
           Lock free sets and accept (non-uniqueness proven).
           No Δt halving (only Newton non-convergence triggers that).
        """
        nN = self.nN
        prev_state = (prev_X, prev_AV, prev_q, prev_u, prev_gNdot, prev_gammaF)

        # ── STEP 1: NEWTON SOLVE ─────────────────────────────────
        result = self._newton_solve(
            iter, prev_X, *prev_state,
            forced_sets=None, detect_chattering=True)

        if not result["converged"]:
            msg = (f"Newton did not converge: ||R||={result['norm_R']:.3e} "
                   f"> tol={self.tol_n:.3e}")
            print(f"FINAL Newton residual (iter={iter}): {msg}")
            try:
                self.f.write(f"\niter {iter}: {msg}")
            except Exception:
                pass
            raise MaxNewtonIterAttainedError(msg)

        X = result["X"]
        AV = result["AV"]
        q = result["q"]
        u = result["u"]
        gNdot = result["gNdot"]
        gammaF = result["gammaF"]
        contacts_nu = result["contacts"]
        alpha_min = result["alpha_min"]
        final_norm_R = result["norm_R"]
        used_sets = result["forced_sets"]
        force_reason = result["force_reason"]

        # Determine what sets were actually used (forced or free from last iter)
        if used_sets is None:
            # Free convergence: the final contacts_nu ARE the used sets
            used_sets = (contacts_nu[0:nN].astype(int),
                         contacts_nu[nN:2*nN].astype(int),
                         contacts_nu[2*nN:3*nN].astype(int),
                         contacts_nu[3*nN:4*nN].astype(int),
                         contacts_nu[4*nN:5*nN].astype(int))

        self.contacts_save[:, iter] = contacts_nu

        # ── STEP 2: SELF-CONSISTENCY CHECK ───────────────────────
        match, free_sets = self._check_self_consistency(
            iter, X, *prev_state, used_sets)

        if match:
            # VERIFIED: the converged solution's physics agree with
            # the contact sets that were used. Nothing more to do.
            status = "VERIFIED"
            if force_reason:
                self.correction_log.append(
                    (iter, self.t[iter], force_reason, used_sets, used_sets,
                     final_norm_R, result["nu"], True))
            print(f"FINAL (iter={iter}): ||R||={final_norm_R:.3e}, {status}")
            try:
                self.f.write(f"\niter {iter}: {status}, ||R||={final_norm_R:.3e}")
            except Exception:
                pass
            return X, AV, q, u, gNdot, gammaF, alpha_min

        # ── STEP 3: EXHAUSTIVE D/E SEARCH ────────────────────────
        # The converged solution is NOT self-consistent.
        # Try every D/E combination with A, B, C from the free evaluation.
        print(f"  iter {iter}: self-consistency FAILED — exhaustive D/E search")
        print(f"    used:  D={used_sets[3]} E={used_sets[4]}")
        print(f"    free:  D={free_sets[3]} E={free_sets[4]}")
        try:
            self.f.write(f"\niter {iter}: self-consistency failed, exhaustive D/E search")
        except Exception:
            pass

        A_base, B_base, C_base = free_sets[0], free_sets[1], free_sets[2]
        best = None  # will hold the best self-consistent result dict
        n_de_combos = (1 << nN) * (1 << nN)

        for de_idx in range(n_de_combos):
            d_idx = de_idx >> nN
            e_idx = de_idx & ((1 << nN) - 1)
            D_try = np.array([(d_idx >> k) & 1 for k in range(nN)], dtype=int)
            E_try = np.array([(e_idx >> k) & 1 for k in range(nN)], dtype=int)
            trial_sets = (A_base, B_base, C_base, D_try, E_try)

            res = self._newton_solve(
                iter, X, *prev_state,
                forced_sets=trial_sets, detect_chattering=False)

            if not res["converged"]:
                print(f"    D={D_try} E={E_try}: did not converge")
                continue

            # Self-consistency check (only D and E matter here)
            sc_match, sc_free = self._check_self_consistency(
                iter, res["X"], *prev_state, trial_sets)

            if sc_match:
                if best is None or res["norm_R"] < best["norm_R"]:
                    best = res
                    best["trial_sets"] = trial_sets
                    print(f"    D={D_try} E={E_try}: ||R||={res['norm_R']:.3e}, self-consistent ✓")
                else:
                    print(f"    D={D_try} E={E_try}: ||R||={res['norm_R']:.3e}, self-consistent (worse)")
            else:
                # Log which sets actually mismatched
                mismatch_labels = []
                set_names = ['A', 'B', 'C', 'D', 'E']
                for si, (ts, fs) in enumerate(zip(trial_sets, sc_free)):
                    if not np.array_equal(ts, fs):
                        mismatch_labels.append(f"{set_names[si]}:{ts}->{fs}")
                mismatch_msg = (f"    D={D_try} E={E_try}: ||R||={res['norm_R']:.3e}, NOT self-consistent "
                      f"[{', '.join(mismatch_labels)}]")
                print(mismatch_msg)
                try:
                    self.f.write(f"\n{mismatch_msg}")
                except Exception:
                    pass

        if best is not None:
            # RESOLVED: a self-consistent D/E combo was found.
            X = best["X"]
            AV = best["AV"]
            q = best["q"]
            u = best["u"]
            gNdot = best["gNdot"]
            gammaF = best["gammaF"]
            contacts_nu = best["contacts"]
            alpha_min = min(alpha_min, best["alpha_min"])
            final_norm_R = best["norm_R"]
            self.contacts_save[:, iter] = contacts_nu
            reason = force_reason if force_reason else "free mismatch"
            self.correction_log.append(
                (iter, self.t[iter], reason, used_sets,
                 best["trial_sets"], final_norm_R, -1, True))
            msg = (f"  exhaustive search RESOLVED: D={best['trial_sets'][3]} "
                   f"E={best['trial_sets'][4]}, ||R||={final_norm_R:.3e}")
            print(msg)
            try:
                self.f.write(f"\niter {iter}: {msg}")
            except Exception:
                pass
        else:
            # ── STEP 4: FALLBACK ─────────────────────────────────
            # No self-consistent D/E exists. Proves non-uniqueness.
            # No Δt halving — only Newton non-convergence triggers that.
            # Re-solve with used_sets (original forced sets) locked so
            # the returned X matches the sets from Step 1.
            fb_res = self._newton_solve(
                iter, X, *prev_state,
                forced_sets=used_sets, detect_chattering=False)
            if fb_res["converged"]:
                X = fb_res["X"]
                AV = fb_res["AV"]
                q = fb_res["q"]
                u = fb_res["u"]
                gNdot = fb_res["gNdot"]
                gammaF = fb_res["gammaF"]
                contacts_nu = fb_res["contacts"]
                alpha_min = min(alpha_min, fb_res["alpha_min"])
                final_norm_R = fb_res["norm_R"]
                self.contacts_save[:, iter] = contacts_nu

            reason = force_reason if force_reason else "free mismatch"
            self.correction_log.append(
                (iter, self.t[iter], reason, used_sets,
                 used_sets, final_norm_R, -1, False))
            msg = (f"  exhaustive search found no self-consistent D/E "
                   f"at dt={self.dtime:.3e}. FALLBACK: locking used sets, "
                   f"||R||={final_norm_R:.3e}")
            print(msg)
            try:
                self.f.write(f"\niter {iter}: {msg}")
            except Exception:
                pass

        print(f"FINAL (iter={iter}): ||R||={final_norm_R:.3e} (converged)")
        try:
            self.f.write(f"\niter {iter}: FINAL ||R||={final_norm_R:.3e}")
        except Exception:
            pass

        return X, AV, q, u, gNdot, gammaF, alpha_min
                
    def get_X_components(self,X):
        '''Getting the components of the array X. Now with 3 DOF (x, y, theta).'''
        a = X[0:self.ndof]
        U = X[self.ndof:2*self.ndof]
        Q = X[2*self.ndof:3*self.ndof]
        Kappa_g = X[3*self.ndof:3*self.ndof+self.ng]
        Lambda_g = X[3*self.ndof+self.ng:3*self.ndof+2*self.ng]
        lambda_g = X[3*self.ndof+2*self.ng:3*self.ndof+3*self.ng]
        Lambda_gamma = X[3*self.ndof+3*self.ng:3*self.ndof+3*self.ng+self.ngamma]
        lambda_gamma = X[3*self.ndof+3*self.ng+self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma]
        Kappa_N = X[3*self.ndof+3*self.ng+2*self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma+self.nN]
        Lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+self.nN:3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN]
        lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN]
        Lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF]
        lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF]
        return a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,\
            Kappa_N,Lambda_N,lambda_N,Lambda_F,lambda_F

    def time_update(self, iter):

        prev_X = self.X_save[:,iter-1]
        prev_AV = self.AV_save[:,iter-1]
        prev_q = self.q_save[:,iter-1]
        prev_u = self.u_save[:,iter-1]
        prev_gNdot = self.gNdot_save[:,iter-1]
        prev_gammaF = self.gammaF_save[:,iter-1]

        # try:
        X,AV,q,u,gNdot,gammaF,alpha_min = self.update(iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

        # Extract lambda forces from X
        a = X[0:self.ndof]
        # Compute gNddot for both gap directions using projection-aware gap methods
        gNx_val, nx_0, ny_0 = self._compute_riser_gap(q[0], q[1])
        gNy_val, nx_1, ny_1 = self._compute_tread_gap(q[0], q[1])
        gNddot = np.array([nx_0 * a[0] + ny_0 * a[1],
                           nx_1 * a[0] + ny_1 * a[1]])
        lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN]
        lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF]

        self.q_save[:,iter] = q
        self.u_save[:,iter] = u
        self.X_save[:,iter] = X
        self.gNdot_save[:,iter] = gNdot
        self.gNddot_save[:,iter] = gNddot
        self.gammaF_save[:,iter] = gammaF
        self.lambdaN_save[:,iter] = lambda_N
        self.lambdaF_save[:,iter] = lambda_F
        self.AV_save[:,iter] = AV
        self.phi_save[iter] = q[3]       # actual computed phi
        self.phi_dot_save[iter] = u[3]   # actual computed phi_dot
        self.phi_ddot_save[iter] = a[3]  # actual computed phi_ddot
        # Save separated gaps (projection-aware) and dual contact flag
        self.gNx_save[iter] = gNx_val
        self.gNy_save[iter] = gNy_val
        contacts_final = self.contacts_save[:, iter]
        self.dual_contact_save[iter] = 1 if (contacts_final[0] == 1 and contacts_final[1] == 1) else 0
        self.rho_inf_save[iter] = self.rho_inf
        self.alpha_save[iter] = alpha_min
        self.save_arrays()

        return
        
    def plot_results(self):
        """Create plots for gNdot, gNddot, lambdaN, and lambdaF."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Bouncing Ball Simulation Results', fontsize=16, fontweight='bold')

        # Plot 1: gNdot (normal gap velocity)
        ax = axes[0, 0]
        ax.plot(self.t, self.gNdot_save[0, :], 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('gNdot (m/s)')
        ax.set_title('Normal Gap Velocity')
        ax.grid(True, alpha=0.3)

        # Plot 2: gNddot (normal acceleration)
        ax = axes[0, 1]
        ax.plot(self.t, self.gNddot_save[0, :], 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('gNddot (m/s²)')
        ax.set_title('Normal Acceleration')
        ax.grid(True, alpha=0.3)

        # Plot 3: lambdaN (normal force)
        ax = axes[1, 0]
        ax.plot(self.t, self.lambdaN_save[0, :], 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('λN (N)')
        ax.set_title('Normal Force (Lambda N)')
        ax.grid(True, alpha=0.3)

        # Plot 4: lambdaF (friction force)
        ax = axes[1, 1]
        ax.plot(self.t, self.lambdaF_save[0, :], 'm-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('λF (N)')
        ax.set_title('Friction Force (Lambda F)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(self.output_path, 'contact_forces_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        plt.close()

    def solve(self):
        """Main simulation loop.

        Drives time stepping with adaptive Δt:
        - Calls time_update() for each step (which calls update()).
        - On MaxNewtonIterAttainedError: halve Δt and retry (up to 4 halvings).
        - On success: restore Δt to original and advance to next step.
        - At the end: print forcing summary (3 categories: VERIFIED, RESOLVED, FALLBACK).
        """
        iter = 1
        self.correction_log = []  # (iter, t, reason, old_sets, new_sets, norm_R, n_iters, resolved?)
        self.min_dtime = self.dtime_initial / 16  # minimum Δt (4 halvings)
        min_dtime = self.min_dtime
        dtime_halve_count = 0
        # ── TIME-STEPPING LOOP ─────────────────────────────────
        while iter < self.ntime:
            self.t[iter] = self.t[iter-1] + self.dtime
            try:
                self.time_update(iter)
                self.dtime_save[iter] = self.dtime
                if self.dtime < self.dtime_initial - 1e-16:
                    msg = (f"\n>>> iter {iter}, t={self.t[iter]:.6f}: "
                           f"restoring dtime {self.dtime:.6e} -> {self.dtime_initial:.6e}")
                    print(msg)
                    self.f.write(msg)
                    self.dtime = self.dtime_initial
                iter += 1
            except MaxNewtonIterAttainedError as e:
                self.correction_log = [e for e in self.correction_log if e[0] != iter]
                if self.dtime > min_dtime + 1e-20:
                    new_dtime = self.dtime / 2
                    msg = (f"\n*** iter {iter}, t={self.t[iter]:.6f}: "
                           f"Newton did not converge, halving dtime "
                           f"{self.dtime:.6e} -> {new_dtime:.6e}")
                    print(msg)
                    self.f.write(msg)
                    self.dtime = new_dtime
                    dtime_halve_count += 1
                else:
                    msg = (f"\n*** iter {iter}, t={self.t[iter]:.6f}: "
                           f"dtime at minimum ({min_dtime:.6e}). Aborting.")
                    print(msg)
                    self.f.write(msg)
                    break

        # ── FORCING SUMMARY ─────────────────────────────────────────
        # 3 categories:
        #   VERIFIED:  Self-consistency check passed (free sets = used sets).
        #              Chattering may have been detected, but the locked
        #              config was self-consistent.
        #   RESOLVED:  Self-consistency failed, but exhaustive D/E search
        #              found a self-consistent combo.
        #   FALLBACK:  No self-consistent D/E exists at Δt_min.
        #              Free sets locked and solution accepted.
        n_verified = sum(1 for e in self.correction_log if e[7])
        n_fallback = sum(1 for e in self.correction_log if not e[7])
        n_total = len(self.correction_log)
        # Split verified into "chattering verified" vs "exhaustive resolved"
        n_chattering_verified = sum(1 for e in self.correction_log if e[7] and e[6] >= 0)
        n_exhaustive_resolved = sum(1 for e in self.correction_log if e[7] and e[6] == -1)
        print(f"\n{'='*60}")
        print(f"FORCING SUMMARY: {n_total} events out of {iter-1} time steps")
        if n_total > 0:
            print(f"  {n_chattering_verified} VERIFIED (chattering locked, self-consistent)")
            print(f"  {n_exhaustive_resolved} RESOLVED (exhaustive D/E search)")
            print(f"  {n_fallback} FALLBACK (no self-consistent D/E, locked free sets)")
        print(f"dtime was halved {dtime_halve_count} times during the simulation.")
        print(f"{'='*60}")
        summary_lines = []
        for entry in self.correction_log:
            c_iter, c_t, c_reason, c_old_sets, c_new_sets, c_norm, c_niters, c_match = entry[:8]
            dt_at_iter = self.dtime_save[c_iter] if c_iter < len(self.dtime_save) else self.dtime_initial
            if c_match and c_niters >= 0:
                tag = "VERIFIED"
            elif c_match and c_niters == -1:
                tag = "RESOLVED"
            else:
                tag = "FALLBACK"
            line = (f"iter {c_iter}, t={c_t:.6f}, dt={dt_at_iter:.3e}: {c_reason} -> {tag}, "
                    f"D={c_old_sets[3]}->{c_new_sets[3]} E={c_old_sets[4]}->{c_new_sets[4]}, "
                    f"||R||={c_norm:.3e}")
            print(line)
            summary_lines.append(line)
        if n_total == 0:
            print("No forcing events.")
        print(f"{'='*60}\n")

        try:
            self.f.write(f"\n\n{'='*60}")
            self.f.write(f"\nFORCING SUMMARY: {n_total} events out of {iter-1} time steps")
            if n_total > 0:
                self.f.write(f"\n  {n_chattering_verified} VERIFIED")
                self.f.write(f"\n  {n_exhaustive_resolved} RESOLVED")
                self.f.write(f"\n  {n_fallback} FALLBACK")
            self.f.write(f"\ndtime was halved {dtime_halve_count} times during the simulation.")
            self.f.write(f"\n{'='*60}")
            for line in summary_lines:
                self.f.write(f"\n{line}")
            if n_total == 0:
                self.f.write("\nNo forcing events.")
            self.f.write(f"\n{'='*60}")
        except Exception:
            pass

        # Generate plots
        self.plot_results()

        try:
            self.f.close()
        except Exception:
            pass

if __name__ == "__main__":
    omega_p = -10.0        # slower pendulum , steadier forward drive
    phi0 = 2*np.pi/3      # pendulum starts pointing down

    def phi_func(t):
        return phi0 + omega_p * t 

    def phi_dot_func(t):
        return omega_p 

    def phi_ddot_func(t):
        return 0.0

    # Create and run simulation
    test = Simulation(
        ntime=3000,          
        mu_s=0.3,            # static friction
        mu_k=0.3,            # kinetic friction
        eN=0.3,              # normal restitution
        eF=0.0,              # friction restitution
        R=0.18,             # Radius of the hoop
        m_hoop=1.0,         # mass of the hoop
        m_pendulum=0.1,      # mass of the pendulum
        l_pendulum=0.15,     # length of the pendulum
        theta0=0,
        omega0=0.0,         # Initial sping for the hoop
        y0=0.18,             # geometric center y0
        x0=0.2,             # geometric center x0
        phi_func=phi_func,      # specify phi0 as IC and leave the velocity and acceleration as None for simple pendulum
        phi_dot_func=phi_dot_func,
        phi_ddot_func=phi_ddot_func,
        n_stairs=2,         # number of stairs (0 for flat ground)
        stair_width=1.8,    
        stair_height=0.1,   
        stair_x_start=0.5,   # stairs further to the right
        fillet_radius=0  # no fillet — sharp corners
    )
    test.solve()