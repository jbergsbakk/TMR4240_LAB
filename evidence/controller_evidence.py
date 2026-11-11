"""
Controller evidence used for the retuning section of the report.

    python controller_evidence.py            # all tables (about 2 min)
    python controller_evidence.py retune     # Part 1 vs Part 2 gains
    python controller_evidence.py zeta       # zeta_c = 1.2 vs 1.0
    python controller_evidence.py aw         # anti-windup on vs off

Changes no files and opens no figures.
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
from part_2.config import ControllerConfig, Part2SimConfig, default_thrusters_part_2
from part_2.controller import DPController
 
# Observer used for the "NPO" rows.  Set explicitly so the tables do not depend
# on SELECTED_OBSERVER in run_case_part_2.py.
OBSERVER = "nonlinear_passive"
 
PART1_WC = (0.07, 0.07, 0.08)       # Part 1 bandwidths [rad/s]
PART1_ZETA = (1.0, 1.0, 1.0)        # Part 1 damping ratios [-]
HOLD = 300.0                        # four-corner hold time per leg [s]
 
U_MAX = np.array([th.u_max for th in default_thrusters_part_2()])
I_LIMIT = np.array(ControllerConfig().I_limit)
 
 
# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
 
def controller(wc=None, zeta_c=None, anti_windup=None) -> DPController:
    """Committed controller with selected fields overridden.
 
    Attributes are set on the instance, so this works whether or not
    ControllerConfig is decorated with @dataclass.
    """
    cfg = ControllerConfig()
    if wc is not None:
        cfg.wc = wc
    if zeta_c is not None:
        cfg.zeta_c = zeta_c
    if anti_windup is not None:
        cfg.anti_windup = anti_windup
    return DPController(cfg)
 
 
def calm():
    return rc.environment(current=False, wind=False, waves=False)
 
 
def design_sea():
    return rc.environment()
 
 
def heavy_sea():
    return rc.environment(hs=rc.ROBUST_HS, tp=rc.ROBUST_TP)
 
 
def four_corner(env, ctl):
    """Simulations 2 and 3: four-corner sequence, raw feedback."""
    cfg = Part2SimConfig(dt=rc.DT, T=5 * HOLD, use_observer=False)
    return rc.run(cfg, rc.four_corner_commands(cfg, HOLD), env, controller=ctl)
 
 
def station_keeping(env, ctl, observer=None, T=600.0):
    """Simulations 4 (closed loop) and 6: hold the origin."""
    cfg = Part2SimConfig(dt=rc.DT, T=T, use_observer=observer is not None,
                         observer_type=observer or OBSERVER)
    return rc.run(cfg, np.zeros(6), env, controller=ctl)
 
 
def metrics(logs, t_end=None) -> dict:
    """Tracking and actuator measures over [0, t_end] (whole run if None)."""
    m = np.ones_like(logs.t, dtype=bool) if t_end is None else logs.t <= t_end
    t = logs.t[m]
    e_n = logs.sp[m, 0] - logs.eta[m, 0]
    e_e = logs.sp[m, 1] - logs.eta[m, 1]
    e_p = np.hypot(e_n, e_e)                                  # horizontal error
    d_psi = logs.sp[m, 5] - logs.eta[m, 5]
    e_psi = np.rad2deg(np.arctan2(np.sin(d_psi), np.cos(d_psi)))
    r = (logs.tau_d[m] - logs.tau_thr[m])[:, [0, 1, 5]]       # residual r_tau
    tau_i = logs.tau_I[m][:, [0, 1, 5]]
    u_cmd, u = logs.u_cmd[m], logs.u[m]
    return {
        "ep_max": e_p.max(),
        "ep_rms": np.sqrt(np.mean(e_p**2)),
        "epsi_max": np.abs(e_psi).max(),
        "eN_mean": e_n.mean(),
        "eE_mean": e_e.mean(),
        # share of time the tunnel COMMAND is at its limit [%]
        "tunnel_sat": 100.0 * np.mean(np.abs(u_cmd[:, 0]) >= 0.999 * U_MAX[0]),
        # peak demanded horizontal force and peak actual thrust [kN]
        "F_cmd_max": np.hypot(logs.tau_d[m, 0], logs.tau_d[m, 1]).max() / 1e3,
        "u_max": np.abs(u).max() / 1e3,
        # mean |du/dt| of the tunnel thruster [kN/s]
        "tunnel_rate": np.abs(np.diff(u[:, 0])).sum() / (t[-1] - t[0]) / 1e3,
        # residual statistics [kN, kN, kNm]
        "r_rms": np.sqrt(np.mean(r**2, axis=0)) / 1e3,
        "r_mean": r.mean(axis=0) / 1e3,
        # share of time each integral term is at its cap [%]
        "I_at_cap": 100.0 * np.mean(np.abs(tau_i) >= 0.999 * I_LIMIT, axis=0),
    }
 
 
# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
 
def table_retune():
    """Table tab:ctrl_retune and the numbers quoted in the three paragraphs."""
    p1 = dict(wc=PART1_WC, zeta_c=PART1_ZETA)
    cases = [
        ("2 (calm, raw)", lambda c: four_corner(calm(), c)),
        ("3 (env., raw)", lambda c: four_corner(design_sea(), c)),
        ("4 (env., NPO)", lambda c: station_keeping(design_sea(), c, OBSERVER)),
    ]
    print("\nTable tab:ctrl_retune: Part 1 vs Part 2 gains in the Part 2 system")
    print(f"{'Sim':15s}{'Gains':8s}{'max e_p':>9s}{'RMS e_p':>9s}{'max e_psi':>11s}"
          f"{'Sat.':>7s}{'F_cmd':>8s}{'u_max':>7s}{'tun.rate':>10s}")
    print(f"{'':15s}{'':8s}{'[m]':>9s}{'[m]':>9s}{'[deg]':>11s}"
          f"{'[%]':>7s}{'[kN]':>8s}{'[kN]':>7s}{'[kN/s]':>10s}")
    for name, sim in cases:
        for label, kw in (("Part 1", p1), ("Part 2", {})):
            q = metrics(sim(controller(**kw)))
            print(f"{name:15s}{label:8s}{q['ep_max']:9.2f}{q['ep_rms']:9.2f}"
                  f"{q['epsi_max']:11.1f}{q['tunnel_sat']:7.1f}{q['F_cmd_max']:8.1f}"
                  f"{q['u_max']:7.1f}{q['tunnel_rate']:10.2f}")
 
 
def table_zeta():
    """Damping-ratio paragraph: committed zeta_c vs zeta_c = 1 in all axes."""
    committed = ControllerConfig().zeta_c
    print("\nDamping ratio check")
    print(f"{'Sim':6s}{'zeta_c':20s}{'max e_p [m]':>13s}{'tunnel sat. [%]':>17s}")
    for name, env in (("2", calm), ("3", design_sea)):
        for zeta in (committed, (1.0, 1.0, 1.0)):
            q = metrics(four_corner(env(), controller(zeta_c=zeta)))
            print(f"{name:6s}{str(tuple(zeta)):20s}{q['ep_max']:13.2f}{q['tunnel_sat']:17.1f}")
 
 
def table_aw():
    """Table tab:ctrl_aw and the Simulation 3 numbers in the paragraph below it."""
    cases = [
        ("2", "raw", lambda c: four_corner(calm(), c), None),
        ("3", "raw", lambda c: four_corner(design_sea(), c), None),
        ("4", "raw", lambda c: station_keeping(design_sea(), c), None),
        ("4", "NPO", lambda c: station_keeping(design_sea(), c, OBSERVER), None),
        # Simulation 6 runs 1000 s; evaluated over its first 600 s
        ("6", "NPO", lambda c: station_keeping(heavy_sea(), c, OBSERVER, T=1000.0), 600.0),
    ]
    print("\nTable tab:ctrl_aw: position error e_p, max / RMS [m]")
    print(f"{'Sim':5s}{'Feedback':10s}{'AW on':>14s}{'AW off':>14s}")
    sim3 = {}
    for name, fb, sim, t_end in cases:
        row = {}
        for label, aw in (("on", True), ("off", False)):
            row[label] = metrics(sim(controller(anti_windup=aw)), t_end)
        print(f"{name:5s}{fb:10s}"
              f"{row['on']['ep_max']:8.2f} /{row['on']['ep_rms']:5.2f}"
              f"{row['off']['ep_max']:8.2f} /{row['off']['ep_rms']:5.2f}")
        if name == "3":
            sim3 = row
 
    ki = DPController().Ki
    tt = ControllerConfig().Tt_pos
    print("\nSimulation 3 detail (surge, sway, yaw)")
    for label in ("on", "off"):
        q = sim3[label]
        print(f"  AW {label:3s}: mean e_N, e_E = {q['eN_mean']:.2f}, {q['eE_mean']:.2f} m"
              f" | I at cap = {np.round(q['I_at_cap'], 1)} %"
              f" | r_tau RMS = {np.round(q['r_rms'], 1)}"
              f" | r_tau mean = {np.round(q['r_mean'], 1)} [kN, kN, kNm]")
    print(f"  Offset held per kN of mean residual, 1/(Tt*Ki): "
          f"surge {1e3 / (tt * ki[0]):.2f} m, sway {1e3 / (tt * ki[1]):.2f} m")
 
 
TABLES = {"retune": table_retune, "zeta": table_zeta, "aw": table_aw}
 
 
def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv) or list(TABLES)
    unknown = [a for a in args if a not in TABLES]
    if unknown:
        raise SystemExit(f"unknown table {unknown}; choose from {list(TABLES)}")
    cfg = ControllerConfig()
    print(f"Committed tuning: wc = {tuple(cfg.wc)}, zeta_c = {tuple(cfg.zeta_c)}, "
          f"anti_windup = {cfg.anti_windup}, dt = {rc.DT} s, seed = {rc.SEED}, "
          f"observer = {OBSERVER}")
    for name in args:
        TABLES[name]()
 
 
if __name__ == "__main__":
    main()
