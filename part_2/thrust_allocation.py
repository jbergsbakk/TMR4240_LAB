"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
   
    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        #Extracting 3-DOF commanded wrench [Fx, Fy, Mz]
        tau_c = np.asarray(tau_d, dtype=float)[[0,1,5]]

        #Extended configuration matrix Be, bult from the thruster geometry.
        #Tunnel: one column
        #Azimuth: Two columns
        cols = []
        ext_idx = []
        w = []
        for th in self.thrusters:
            if th.kind == "tunnel":
                c = np.cos(th.alpha0)
                s = np.sin(th.alpha0)
                cols.append([c,s,th.x*s-th.y*c])
                ext_idx.append([len(cols) - 1])
                w += [1.0/th.u_max**2]
            else:
                cols.append([1.0, 0.0, -th.y])
                cols.append([0.0, 1.0, th.x])
                ext_idx.append([len(cols) -2, len(cols)-1])
                w += [1.0/th.u_max**2]*2

        Be = np.array(cols).T

        #Weighted pseudo-inverse, w = diag(1/u_max^2)
        W_inv = np.diag(1.0/np.array(w))
        f_e = np.matmul(np.matmul(W_inv, Be.T), np.linalg.solve(np.matmul(np.matmul(Be, W_inv), Be.T), tau_c)) 

        # Scaling the whole solution by one factor
        u_max = np.array([th.u_max for th in self.thrusters])
        u_mag = np.array([np.linalg.norm(f_e[idx]) for idx in ext_idx])
        ratio = np.max(u_mag/u_max)
        if ratio > 1.0:
            f_e = f_e/ratio

        #mapping extendeed forces back to (u, alpha)
        u_cmd = np.zeros(n)
        alpha_cmd=np.zeros(n)
        for i, (th, idx) in enumerate(zip(self.thrusters, ext_idx)):
            if th.kind == "tunnel":
                u_cmd[i] = f_e[idx[0]]
                alpha_cmd[i] = th .alpha0
                continue
            fx, fy = f_e[idx]
            u= np.hypot(fx,fy)
            a = np.arctan2(fy, fx)

            # (u, a) and (-u, a + pi) will give the same force. We will pick the one that gives the least rotation
            if alpha_now is not None:
                d1 = (a - alpha_now[i] + np.pi) %(2*np.pi)- np.pi
                d2 = (a+np.pi - alpha_now[i] + np.pi) % (2*np.pi) - np.pi
                if abs(d2) < abs(d1):
                    u = -u
                    d1 = d2
                a = alpha_now[i] +d1
            u_cmd[i] = u
            alpha_cmd[i] = a

        return u_cmd, alpha_cmd

