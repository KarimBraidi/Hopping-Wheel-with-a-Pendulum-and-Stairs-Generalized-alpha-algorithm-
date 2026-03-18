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
        self._build_stair_profile()

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
        self.nN = 1          # number of gap distance constraints
        self.nF = 1          # number of friction constraints
        self.gammaF_lim = np.array([[0]])    # connectivities of friction and normal forces
        self.nX = 3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF     # total number of constraints with their derivative
        # generalized alpha parameters
        self.MAXITERn = 20
        self.MAXITERn_initial = self.MAXITERn   # saving initial value of MAXITERn
        self.r = 0.3
        self.rho_inf = 0.5
        self.rho_infinity_initial = self.rho_inf
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2
        self.tol_n = 1.0e-6     # error tolerance
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
        """Build the stair profile as a list of line segments [(x1,y1,x2,y2), ...]."""
        FAR = 1e6  # effectively +/- infinity
        segs = []
        if self.n_stairs == 0:
            # Flat ground
            segs.append((-FAR, 0.0, FAR, 0.0))
        else:
            # Ground before stairs
            segs.append((-FAR, 0.0, self.stair_x_start, 0.0))
            for i in range(self.n_stairs):
                x_left = self.stair_x_start + i * self.stair_width
                h_prev = i * self.stair_height
                h_curr = (i + 1) * self.stair_height
                x_right = x_left + self.stair_width
                # Riser (vertical wall)
                segs.append((x_left, h_prev, x_left, h_curr))
                # Tread (horizontal surface)
                if i < self.n_stairs - 1:
                    segs.append((x_left, h_curr, x_right, h_curr))
                else:
                    # Last tread extends to the right
                    segs.append((x_left, h_curr, FAR, h_curr))
        self.stair_segments = segs

    def _closest_point_on_stair(self, px, py):
        """Find the closest point on the stair profile to (px, py).

        Returns (cx, cy, dist, nx, ny) where:
            cx, cy = closest point on profile
            dist   = Euclidean distance from (px, py) to (cx, cy)
            nx, ny = unit normal pointing from the surface toward (px, py)
        """
        min_dist_sq = float('inf')
        best_cx, best_cy = 0.0, 0.0
        for seg in self.stair_segments:
            x1, y1, x2, y2 = seg
            dx_s, dy_s = x2 - x1, y2 - y1
            seg_len_sq = dx_s * dx_s + dy_s * dy_s
            if seg_len_sq < 1e-30:
                cx, cy = x1, y1
            else:
                t = max(0.0, min(1.0, ((px - x1) * dx_s + (py - y1) * dy_s) / seg_len_sq))
                cx = x1 + t * dx_s
                cy = y1 + t * dy_s
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_cx, best_cy = cx, cy
        dist = np.sqrt(min_dist_sq)
        if dist < 1e-15:
            nx, ny = 0.0, 1.0  # default upward if degenerate
        else:
            nx = (px - best_cx) / dist
            ny = (py - best_cy) / dist
        return best_cx, best_cy, dist, nx, ny

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

        # normal gap distance and slip speed constraints (stair-aware)
        cx, cy, dist, nx, ny = self._closest_point_on_stair(q[0], q[1])
        gN = np.array([dist - self.R])
        gNdot = np.array([nx * u[0] + ny * u[1]])
        gNddot = np.array([nx * a[0] + ny * a[1]])
        WN = np.array([[nx, ny, 0, 0]]).T
        # Tangent direction t = (ny, -nx); slip = t . v_contact
        gammaF = np.array([ny * u[0] - nx * u[1] + self.R * u[2]])
        gammaFdot = np.array([ny * a[0] - nx * a[1] + self.R * a[2]])
        WF = np.array([[ny, -nx, self.R, 0]]).T

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
                # check for contact if blocks are not horizontally detached
                if self.r*gN[i] - Kappa_hatN[i] <=0:
                    A[i] = 1
                    if np.linalg.norm(self.r*ksiF[self.gammaF_lim[i,:]]-PF[self.gammaF_lim[i,:]])<=self.mu_s*(PN[i]):
                        # D-stick
                        D[i] = 1
                        if np.linalg.norm(self.r*gammaFdot[self.gammaF_lim[i,:]]-lambdaF[self.gammaF_lim[i,:]])<=self.mu_s*(lambdaN[i]):
                            # E-stick
                            E[i] = 1
                    if self.r*ksiN[i]-PN[i] <= 0:
                        B[i] = 1
                        if self.r*gNddot[i]-lambdaN[i] <= 0:
                            C[i] = 1
        else:
            A = index_sets[0]
            B = index_sets[1]
            C = index_sets[2]
            D = index_sets[3]
            E = index_sets[4]

        # calculating contact residual
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
            # Return 4D q, u now that phi is a full DOF
            return R, AV, q, u, gNdot, gammaF, A, B, C, D, E
        else:
            # in this case, get_R is called as part of calculating the Jacobian for fixed contact regions
            return R, AV, q, u, gNdot, gammaF

    def get_R_J(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF):
        '''Calculate the Jacobian manually.'''

        epsilon = 1e-6
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

    def update(self,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF):
        """Takes components at time t and return values at time t+dt"""

        nu = 0
        
        X = prev_X
        R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

        contacts = np.zeros((self.MAXITERn+1,3*self.nN+2*self.nN),dtype=int)
        contacts[nu,:] = contacts_nu
        self.contacts_save[:,iter] = contacts_nu

        norm_R = np.linalg.norm(R,np.inf)
        print(f"norm(R) = {norm_R}")

        # try:

        while np.abs(np.linalg.norm(R,np.inf))>self.tol_n and nu<self.MAXITERn:
            # Newton Update (lstsq handles singular Jacobian, e.g. m_pendulum=0)
            X = X - np.linalg.lstsq(J, R, rcond=None)[0]
            # Calculate new EOM and residual
            nu = nu+1

            R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

            contacts[nu,:] = contacts_nu
            self.contacts_save[:,iter] = contacts_nu
                
            norm_R = np.linalg.norm(R,np.inf)
            print(f"nu = {nu}")
            print(f"norm(R) = {norm_R}")

        return X,AV,q,u,gNdot,gammaF
                
    def update_rho_inf(self):
        '''Update the numerical parameter rho_inf.'''
        self.rho_inf = self.rho_inf+0.05  #0.01
        print(self.rho_inf)
        self.f.write(f"  Updating rho_inf to {self.rho_inf}")
        if np.abs(self.rho_inf - self.rho_infinity_initial) < 0.001:
            print("possibility of infinite loop")
            self.f.write(f"  Raising RhoInfInfiniteLoop error")
            raise RhoInfInfiniteLoop
        if self.rho_inf > 1.001:
            self.rho_inf = 0
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2

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
        X,AV,q,u,gNdot,gammaF = self.update(iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

        # Extract lambda forces from X
        a = X[0:self.ndof]
        _, _, _, nx, ny = self._closest_point_on_stair(q[0], q[1])
        gNddot = np.array([nx * a[0] + ny * a[1]])
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
        iter = 1
        while iter < self.ntime:
            self.time_update(iter)
            iter += 1

        # Generate plots
        self.plot_results()

        try:
            self.f.close()
        except Exception:
            pass

if __name__ == "__main__":
    omega_p = -13.0        # slower pendulum , steadier forward drive
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
        mu_s=0.5,            # static friction
        mu_k=0.3,            # kinetic friction
        eN=0.5,              # normal restitution
        eF=0.0,              # friction restitution
        R=0.18,             # Radius of the hoop
        m_hoop=1.0,         # mass of the hoop
        m_pendulum=0.1,      # mass of the pendulum
        l_pendulum=0.15,     # length of the pendulum
        theta0=0,
        omega0=0,         # Initial sping for the hoop
        y0=0.18,             # geometric center y0
        x0=0.0,             # geometric center x0
        phi_func=phi0,      # specify phi0 as IC and leave the velocity and acceleration as None for simple pendulum
        phi_dot_func=None,
        phi_ddot_func=None,
        n_stairs=0,         # number of stairs (0 for flat ground)
        stair_width=1.8,    
        stair_height=0.1,   
        stair_x_start=0.5,   # stairs further to the right
    )
    test.solve()