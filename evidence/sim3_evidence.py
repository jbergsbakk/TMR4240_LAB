"""
Simulation 3 evidence for the report.

    python sim3_evidence.py          # about 20 s

"""
from __future__ import annotations
 
import os
import sys
from pathlib import Path
 
os.environ.setdefault("MPLBACKEND", "Agg")      # no plot windows
 
import numpy as np
 
# Make the repository root importable when run from this folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
 
import run_case_part_2 as rc
from part_2.config import Part2SimConfig, default_thrusters_part_2
 
HOLD = 300.0                 # hold time per leg [s]
LF_WINDOW = 30.0             # moving-average window that defines "LF" [s]
 
THRUSTERS = default_thrusters_part_2()
NAMES = [th.name for th in THRUSTERS]
U_MAX = np.array([th.u_max for th in THRUSTERS])
U_RATE = np.array([th.u_rate for th in THRUSTERS])
A_RATE = np.array([th.rot_speed for th in THRUSTERS])
XYN = [0, 1, 5]              # surge, sway, yaw
 
 
def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))
 
 
def moving_average(x, dt):
    n = int(round(LF_WINDOW / dt))
    return np.convolve(x, np.ones(n) / n, mode="same")
 
 
def four_corner(env):
    cfg = Part2SimConfig(dt=rc.DT, T=5 * HOLD, use_observer=False)
    return rc.run(cfg, rc.four_corner_commands(cfg, HOLD), env)
 
 
def errors(logs):
    e_n = logs.sp[:, 0] - logs.eta[:, 0]
    e_e = logs.sp[:, 1] - logs.eta[:, 1]
    d = logs.sp[:, 5] - logs.eta[:, 5]
    e_psi = np.rad2deg(np.arctan2(np.sin(d), np.cos(d)))
    return e_n, e_e, np.hypot(e_n, e_e), e_psi
 
 
def report(logs, name):
    t, dt = logs.t, rc.DT
    e_n, e_e, e_p, e_psi = errors(logs)
    half = int(round(LF_WINDOW / dt)) // 2
    core = slice(half, -half)                     # drop moving-average edges
    lf_n, lf_e = moving_average(e_n, dt), moving_average(e_e, dt)
    lf_psi = moving_average(e_psi, dt)
 
    print(f"\n=== {name} ===")
    print(f"Tracking: max e_p {e_p.max():.2f} m, RMS e_p {rms(e_p):.2f} m, "
          f"max |e_psi| {np.abs(e_psi).max():.1f} deg, RMS e_psi {rms(e_psi):.2f} deg")
    print(f"  LF part ({LF_WINDOW:.0f} s average): max e_p "
          f"{np.hypot(lf_n, lf_e)[core].max():.2f} m, max |e_psi| "
          f"{np.abs(lf_psi[core]).max():.1f} deg")
    print(f"  WF part: RMS e_p {rms(np.hypot(e_n - lf_n, e_e - lf_e)[core]):.2f} m, "
          f"RMS e_psi {rms((e_psi - lf_psi)[core]):.2f} deg")
    for k in range(5):
        leg = (t >= HOLD * k) & (t < HOLD * (k + 1))
        end = (t >= HOLD * (k + 1) - 100.0) & (t < HOLD * (k + 1))
        print(f"  leg {k + 1}: max e_p {e_p[leg].max():.2f} m, max |e_psi| "
              f"{np.abs(e_psi[leg]).max():.1f} deg | last 100 s: mean e_p "
              f"{e_p[end].mean():.2f} m, mean e_psi {e_psi[end].mean():+.1f} deg")
 
    tau_c = logs.tau_d[:, XYN] / 1e3
    tau_thr = logs.tau_thr[:, XYN] / 1e3
    r_tau = tau_c - tau_thr
    print("Wrench [kN, kN, kNm] (surge, sway, yaw)")
    print(f"  tau_c   RMS {np.round(rms(tau_c), 1)}  max {np.round(np.abs(tau_c).max(0), 1)}")
    print(f"  tau_thr RMS {np.round(rms(tau_thr), 1)}  mean {np.round(tau_thr.mean(0), 1)}")
    print(f"  r_tau   RMS {np.round(rms(r_tau), 1)}  max {np.round(np.abs(r_tau).max(0), 1)}")
    for term, arr in (("P", logs.tau_P), ("I", logs.tau_I), ("D", logs.tau_D)):
        print(f"  {term}-term  RMS {np.round(rms(arr[:, XYN]) / 1e3, 1)}")
 
    du = np.diff(logs.u, axis=0) / dt
    da = np.diff(np.unwrap(logs.alpha, axis=0), axis=0) / dt
    azi = A_RATE > 0
    print(f"Thrusters {NAMES}")
    print(f"  command at thrust limit  [%] {np.round(100 * np.mean(np.abs(logs.u_cmd) >= 0.999 * U_MAX, 0), 1)}")
    print(f"  actual at thrust limit   [%] {np.round(100 * np.mean(np.abs(logs.u) >= 0.999 * U_MAX, 0), 2)}")
    print(f"  thrust-rate limit active [%] {np.round(100 * np.mean(np.abs(du) >= 0.999 * U_RATE, 0), 1)}")
    print(f"  azimuth-rate limit active[%] {np.round(100 * np.mean(np.abs(da[:, azi]) >= 0.999 * A_RATE[azi], 0), 1)} (azimuths)")
    print(f"  mean azimuth rate    [deg/s] {np.round(np.rad2deg(np.abs(da[:, azi]).mean(0)), 1)} (azimuths)")
    print(f"  max |u| [kN] {np.round(np.abs(logs.u).max(0) / 1e3, 1)}")
    print(f"  mean thrust utilisation [%] {100 * np.mean(np.abs(logs.u).sum(1) / U_MAX.sum()):.1f}")
 
 
def environment_report(logs):
    t, dt = logs.t, rc.DT
    wind = logs.tau_w6[:, XYN] / 1e3
    wave = logs.tau_wave[:, XYN] / 1e3
    half = int(round(LF_WINDOW / dt)) // 2
    wave_lf = np.column_stack([moving_average(wave[:, i], dt) for i in range(3)])[half:-half]
    print("\n=== Simulation 3 environment, body frame [kN, kN, kNm] ===")
    print(f"Wind: mean {np.round(wind.mean(0), 1)}, min {np.round(wind.min(0), 1)}, "
          f"max {np.round(wind.max(0), 1)}")
    print(f"Wave: std {np.round(wave.std(0), 1)}, max |.| {np.round(np.abs(wave).max(0), 1)}")
    print(f"Wave LF ({LF_WINDOW:.0f} s average): min {np.round(wave_lf.min(0), 1)}, "
          f"max {np.round(wave_lf.max(0), 1)}")
    for k in range(5):
        leg = (t >= HOLD * k) & (t < HOLD * (k + 1))
        print(f"  leg {k + 1}: wave std {np.round(wave[leg].std(0), 1)} | "
              f"wind mean {np.round(wind[leg].mean(0), 1)}")
    print(f"True velocity: max |u| {np.abs(logs.nu[:, 0]).max():.2f} m/s, "
          f"max |v| {np.abs(logs.nu[:, 1]).max():.2f} m/s, "
          f"max |r| {np.rad2deg(np.abs(logs.nu[:, 5]).max()):.2f} deg/s")
 
 
def main():
    sim2 = four_corner(rc.environment(current=False, wind=False, waves=False))
    sim3 = four_corner(rc.environment())
    report(sim2, "Simulation 2 (calm)")
    report(sim3, "Simulation 3 (Simulation 1 environment)")
    environment_report(sim3)
 
 
if __name__ == "__main__":
    main()