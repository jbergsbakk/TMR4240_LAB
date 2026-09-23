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

        # Extract 3-DOF comanded wrench [Fx, Fy, Mz]
        # We will only allocate surge sway and yaw
        tau_c = tau_d[[0,1,5]]

        #Formulating the configuration matrix Be
        #Chosen variables = [u_tunnel_y, u_azi1_x, u_azi1_y, u_azi2_x, u_azi2_y]
        Be = np.array([
            [0,   1,   0,   1,   0],  # Fx = u_azi1_x + u_azi2_x
            [1,   0,   1,   0,   1],  # Fy = u_tunnel_y + u_azi1_y + u_azi2_y
            [12, -3, -13,   3, -13]   # Mz = x*Fy - y*Fx
        ])

        # Using the Moore-Penrose Pseudo-Inverse to solve
        # This minimizes overall actuator effort (||f||_2)
        Be_pinv = np.linalg.pinv(Be)
        f = Be_pinv @ tau_c             # Matrix multiplication

        # Map cartesian forces back to actuator commands (u, alpha)
        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)

        # Bow tunnel
        u_cmd[0] = f[0]
        alpha_cmd[0] = np.pi / 2        # Fixed at 90 deg (pi/2 rad)

        # Stern azimuth thruster 1
        u_cmd[1] = np.hypot(f[1], f[2])
        alpha_cmd[1] = np.arctan2(f[2], f[1])

        # Stern azimuth thruster 2
        u_cmd[2] = np.hypot(f[3], f[4])
        alpha_cmd[2] = np.arctan2(f[4], f[3])


        # Minimize azimuth thruster rotation
        if alpha_now is not None:
            for i in range(1, 3):  # Only apply to the azimuths
                a1 = alpha_cmd[i]
                u1 = u_cmd[i]

                # Calculate the alternative mathematical representation
                a2 = a1 + np.pi if a1 < 0 else a1 - np.pi
                u2 = -u1

                # Find the shortest angular distance to the current physical angle
                diff1 = (a1 - alpha_now[i] + np.pi) % (2 * np.pi) - np.pi
                diff2 = (a2 - alpha_now[i] + np.pi) % (2 * np.pi) - np.pi

                # Choose the representation that requires the least rotation
                if abs(diff2) < abs(diff1):
                    alpha_cmd[i] = alpha_now[i] + diff2
                    u_cmd[i] = u2
                else:
                    alpha_cmd[i] = alpha_now[i] + diff1

        # Apply saturation limits 
        # Tunnel max: 32 kN. Azimuth max: 80 kN.
        u_max = [32000.0, 80000.0, 80000.0]
        for i in range(n):
            if abs(u_cmd[i]) > u_max[i]:
                u_cmd[i] = np.sign(u_cmd[i]) * u_max[i]

        return u_cmd, alpha_cmd
