"""
DP controller: PID (surge, sway, yaw) with acceleration feedforward,
computed in BODY frame, with anti-windup.

Interface: 
    compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d (6,).
    reset()
    apply_external_aw(tau_applied, psi, dt). 

Logged:
    last_pid_body {"P","I","D"}
    int_ned (2,)
    int_psi (float).

Control law (BODY frame):
    tau_d = Kp*e_body + Ki*i_body + Kd*e_dot_body + M3 @ acc_ref_body
- e_body:     NED position/heading error (atan2-wrapped heading),
              rotated into BODY via simulation.utils.ned_to_body_xy
              (heading passes through a z-axis rotation unchanged).
- i_body:     integral of that error, kept in NED/heading space and
              rotated the same way at point of use.
- e_dot_body: reference velocity (NED) rotated into BODY once via
              ned_to_body_xy and differenced directly against nu.
- FF:         acceleration feedforward using the FULL M3 (keeps the
              sway<->yaw added-mass coupling, not just its diagonal).

6-DOF<->3-DOF conversion and the BODY<->NED xy rotations are handled
by simulation.utils (to_3dof/to_6dof, ned_to_body_xy/body_to_ned_xy).

Plant model (plant_model.ipynb Sec. 6): M3 = M_RB + M_A (full 3x3),
D3 = D_l (diagonal; not used in the control law, only for tuning
reference). Gains from the frequency-domain method (Sec 6.2), applied
to M3's diagonal only: Kp = m*wc^2, Kd = 2*zeta_c*wc*m, Ki = Kp/Ti.

Anti-windup (Sec 6.3): d(int)/dt = e + Kaw*(tau_applied - tau_unsat),
rotated back into the NED/heading frame the integrators live in.

Constructor contract: check.py, pytest, and the notebook build
DPController() with NO arguments, final tuned values must be these
defaults, not just overrides in run_case_part1.py.
"""

import numpy as np
from simulation.utils import wrap_angle_pi, ned_to_body_xy, body_to_ned_xy, to_3dof, to_6dof
 
# --- Control plant model, plant_model.ipynb Sec. 6 ---------------------
# M3: full rigid-body + added-mass inertia, [surge, sway, yaw]
M3 = np.array([
    [6.007e5,      0.0,       0.0],
    [0.0,      7.067e5,  -4.733e5],
    [0.0,     -5.712e5,   5.456e7],
])
# D3: linear damping only (diagonal)
D3 = np.array([
    [1117.6,     0.0,     0.0],
    [0.0,    2.229e4,     0.0],
    [0.0,        0.0,  1.95e6],
])
 
 
def _design_gains(wc: np.ndarray, zeta: np.ndarray, Ti: np.ndarray):
    """Frequency-domain PID design per independent channel, from the
    diagonal of M3: Kp = m*wc^2, Kd = 2*zeta*wc*m, Ki = Kp / Ti."""
    m_diag = np.diag(M3)
    Kp = m_diag * wc ** 2
    Kd = 2.0 * zeta * wc * m_diag
    Ki = Kp / Ti
    return Kp, Kd, Ki
 
 
class DPController:
    """PID + feedforward DP controller, body-frame implementation."""
 
    def __init__(
        self,
        *,
        # --- Frequency-domain design point (project text, Sec 6.2) ---
        # wc: closed-loop bandwidth [rad/s] per axis [surge, sway, yaw].
        wc=np.array([0.07, 0.07, 0.08]),
        # zeta_c: desired damping ratio per axis. 1.0 = critically damped,
        zeta_c=np.array([1.0, 1.0, 1.0]),
        # Ti: integral time constant [s] per axis (Ki = Kp / Ti). Long Ti
        Ti=np.array([100.0, 100.0, 100.0]),
        Tt_pos=20.0,
        Tt_psi=20.0,

        # I_limit: cap on the INTEGRAL term's contribution to tau_d (N, N, Nm) --
        # NOT the raw error integral. Set to 25% of each axis's thruster budget
        # The 25% split is a placeholder
        I_limit=np.array([40_000.0, 48_000.0, 120_000.0]),
    ):
        self.Kp, self.Kd, self.Ki = _design_gains(
            np.asarray(wc, dtype=float),
            np.asarray(zeta_c, dtype=float),
            np.asarray(Ti, dtype=float),
        )
        Tt = np.array([Tt_pos, Tt_pos, Tt_psi])
        self.Kaw = 1.0 / (Tt * self.Ki)
        self.M = M3  # full matrix, used for feedforward
        self.I_limit = np.asarray(I_limit, dtype=float)
        self._raw_int_safety = 1.0e6
 
        self.reset()
 
    def reset(self) -> None:
        """Called by the engine before each run."""
        self.int_ned = np.zeros(2)   # accumulated [N, E] error (m*s)
        self.int_psi = 0.0           # accumulated heading error (rad*s)
        self._last_tau_unsat_3 = np.zeros(3)   # last [Fx, Fy, Mz] requested
        self._last_psi = 0.0
        self.last_pid_body = {"P": np.zeros(6), "I": np.zeros(6), "D": np.zeros(6)}
 
    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        nu_ref = np.zeros(6) if nu_ref is None else nu_ref
        acc_ref = np.zeros(6) if acc_ref is None else acc_ref
 
        eta3 = to_3dof(eta)            # [N, E, psi]
        nu3 = to_3dof(nu)              # [u, v, r]
        eta_ref3 = to_3dof(eta_ref)    # [N_d, E_d, psi_d]
        nu_ref3 = to_3dof(nu_ref)      # [Ndot_d, Edot_d, psidot_d]
        acc_ref3 = to_3dof(acc_ref)    # [Nddot_d, Eddot_d, psiddot_d]
        psi = eta3[2]
 
        # --- Position / heading error, NED ---
        e_pos_ned = eta_ref3[:2] - eta3[:2]
        e_psi = wrap_angle_pi(eta_ref3[2] - psi)
 
        # --- Velocity error, formed in BODY ---
        nu_ref_body_xy = ned_to_body_xy(nu_ref3[:2], psi)
        e_dot_body = np.array([nu_ref_body_xy[0], nu_ref_body_xy[1], nu_ref3[2]]) - nu3
 
        # --- Acceleration feedforward, rotated into BODY ---
        acc_ref_body_xy = ned_to_body_xy(acc_ref3[:2], psi)
        acc_ref_body = np.array([acc_ref_body_xy[0], acc_ref_body_xy[1], acc_ref3[2]])
 
        # --- Integrator update (NED/heading frame; safety bound only) ---
        self.int_ned = np.clip(
            self.int_ned + e_pos_ned * dt, -self._raw_int_safety, self._raw_int_safety
        )
        self.int_psi = float(
            np.clip(self.int_psi + e_psi * dt, -self._raw_int_safety, self._raw_int_safety)
        )
 
        # --- Rotate error/integral into BODY, apply gains ---
        e_body_xy = ned_to_body_xy(e_pos_ned, psi)
        e_body = np.array([e_body_xy[0], e_body_xy[1], e_psi])
        i_body_xy = ned_to_body_xy(self.int_ned, psi)
        i_body = np.array([i_body_xy[0], i_body_xy[1], self.int_psi])
 
        P = self.Kp * e_body
        I = np.clip(self.Ki * i_body, -self.I_limit, self.I_limit)
        D = self.Kd * e_dot_body
        FF = self.M @ acc_ref_body   # full M3, keeps sway-yaw coupling
 
        tau_3 = P + I + D + FF   # [Fx, Fy, Mz], BODY
 
        # Save for anti-windup comparison in apply_external_aw()
        self._last_tau_unsat_3 = tau_3.copy()
        self._last_psi = psi
 
        # --- Logging (6-DOF, zeros in unused DOFs) ---
        self.last_pid_body = {
            "P": to_6dof(P), "I": to_6dof(I), "D": to_6dof(D),
        }
 
        return to_6dof(tau_3)
 
    def apply_external_aw(self, tau_applied: np.ndarray, psi: float, dt: float) -> None:
        """Back-calculation anti-windup using the wrench actually applied
        after allocation and actuator dynamics.
 
            d(int)/dt = e + Kaw * (tau_sat - tau_unsat)
 
        tau_applied and the stored tau_unsat are both BODY; the correction
        is rotated back into the NED/heading frame the integrator lives in.
        """
        tau_applied_3 = to_3dof(tau_applied)
        diff_body = tau_applied_3 - self._last_tau_unsat_3
        diff_ned_xy = body_to_ned_xy(diff_body[:2], self._last_psi)
        diff_ned = np.array([diff_ned_xy[0], diff_ned_xy[1], diff_body[2]])
 
        correction = self.Kaw * diff_ned * dt
        self.int_ned = np.clip(
            self.int_ned + correction[:2], -self._raw_int_safety, self._raw_int_safety
        )
        self.int_psi = float(
            np.clip(self.int_psi + correction[2], -self._raw_int_safety, self._raw_int_safety)
        )