"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from typing import Tuple
import numpy as np

# Per-axis tuning parameters live with the rest of the Part 1 configuration.
from part_1.config import RefAxisConfig


class ReferenceModel:
    """
    Template for student reference model.

    The default implementation is pass-through, so eta_ref = eta_cmd and the
    reference velocities/accelerations are zero.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)
        psi_diff = eta_cmd[5] - self.eta_ref[5]
        psi_diff = (psi_diff + np.pi) % (2.0 * np.pi) - np.pi
        psi_target = self.eta_ref[5] + psi_diff

        for i, cfg in ((0, self.cfg_xy), (1, self.cfg_xy), (5, self.cfg_psi)):
            target = psi_target if i == 5 else eta_cmd[i]
            wn, zeta = cfg.wn, cfg.zeta

            acc = wn**2 * (target - self.eta_ref[i]) - 2.0 * zeta * wn * self.nu_ref[i] #The second-order low-pass filter


            vel = self.nu_ref[i] + dt * acc  
            if cfg.rate_limit is not None:
                vel = float(np.clip(vel, -cfg.rate_limit, cfg.rate_limit))
            pos = self.eta_ref[i] + dt * vel

            self.eta_ref[i] = pos
            self.nu_ref[i] = vel
            self.acc_ref[i] = acc

        return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()
