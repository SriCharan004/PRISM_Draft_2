"""
reproduce_numbers.py
====================
Regenerates every number reported in Annexure E (PRISM in action on CAS ActSim data)
from fixed seeds. It calls the exact modules that produced the paper numbers, so the
output matches the annexure tables.

REQUIREMENTS
------------
Run this from inside the PRISM_Draft_2 repository, which must contain:
    prism_sim.py      (the detector core, imported unchanged)
    prism_noext.py    (ActSim bridge: claim-level panel generation)
    prism_robust.py   (PRISM(adjusted): Hampel filter + head-to-head)
    prism_curve.py    (curvature sweep)
Plus:
    pip install numpy pandas scipy
    pip install git+https://github.com/casact/actsim      # CAS ActSim

USAGE
-----
    python reproduce_numbers.py            # full reproduction (paper n; ~20-30 min)
    python reproduce_numbers.py --quick    # small n for a fast sanity check (numbers will differ)

Each block prints a labelled table. All data are synthetic.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from dataclasses import replace
from scipy import stats

import prism_sim as ps            # detector core (BOCPD, evaluate, config) -- unchanged
import prism_noext as px          # ActSim claim-level panel bridge
import prism_robust as rb         # PRISM(adjusted): Hampel filter, head-to-head
import prism_curve as pc          # curvature sweep

warnings.filterwarnings("ignore")
cfg0 = replace(ps.CFG, channel="severity")
K, T_BURN = cfg0.K, 4
QUICK = "--quick" in sys.argv
N   = 5   if QUICK else 80        # replications per regime for the paired runs
NC  = 20  if QUICK else 100       # in-control reference / diagnostic panels
NCAL= 10  if QUICK else 60        # calibration panels


# ======================================================================================
# Table E1 -- calibration of the ActSim in-control panel
# ======================================================================================
def table_E1_calibration():
    logrets, ac = [], []
    for i in range(NCAL):
        ls = rb.gen(990_000 + i, "none", cfg0)["logsev"]      # in-control ActSim panel
        r = np.diff(ls); logrets.append(r)
        ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    r = np.concatenate(logrets)
    z = (r - r.mean()) / r.std()
    print("\n=== Table E1: ActSim in-control calibration (%d panels) ===" % NCAL)
    print(f"  mean severity log-return : {r.mean():.5f}   (target 0.00850)")
    print(f"  sd severity log-return   : {r.std():.5f}   (target 0.01600)")
    print(f"  KS test vs Normal p-value: {stats.kstest(z, 'norm').pvalue:.3f}   (cannot reject if > 0.05)")
    print(f"  lag-1 autocorrelation    : {np.mean(ac):+.3f}   (native ~ -0.02)")


# ======================================================================================
# Table E2 -- why the loss-only benchmark is blind to a gradual turn (native vs ActSim)
# ======================================================================================
def _bocpd_detect(panels, cfg):
    det = 0; peak = []
    for p in panels:
        A = ps.bocpd(p["logret"], cfg.h0, cfg)                # severity-only, flat hazard
        fa = ps.first_alarm(A, cfg)
        if fa is not None and K - 2 <= fa <= K + 6: det += 1
        peak.append(A[T_BURN:].max())
    return det / len(panels), float(np.mean(peak))

def table_E2_blindness():
    # ActSim panels
    act_grad = [px.panel_actsim(880_000 + i, "gradual", True, cfg0) for i in range(NC)]
    act_abr  = [px.panel_actsim(880_000 + i, "abrupt",  True, cfg0) for i in range(NC)]
    # native (prism_sim) panels
    nat_grad = [ps.generate_panel(770_000 + i, "gradual", True, cfg0) for i in range(NC)]
    nat_abr  = [ps.generate_panel(770_000 + i, "abrupt",  True, cfg0) for i in range(NC)]
    # in-control return sd for each generator (BOCPD's known variance)
    sd_act = float(np.std(np.concatenate([px.panel_actsim(905_000 + i, "gradual", False, cfg0)["logret"][1:] for i in range(30)])))
    sd_nat = float(np.std(np.concatenate([ps.generate_panel(905_000 + i, "gradual", False, cfg0)["logret"][1:] for i in range(30)])))
    cfg_act, cfg_nat = replace(cfg0, sig_d=sd_act), replace(cfg0, sig_d=sd_nat)
    def ac(panels): return np.mean([np.corrcoef(p["logret"][1:][:-1], p["logret"][1:][1:])[0,1] for p in panels])
    ng_d, ng_p = _bocpd_detect(nat_grad, cfg_nat); ag_d, ag_p = _bocpd_detect(act_grad, cfg_act)
    na_d, _    = _bocpd_detect(nat_abr,  cfg_nat); aa_d, _    = _bocpd_detect(act_abr,  cfg_act)
    print("\n=== Table E2: severity-only BOCPD, native vs ActSim (%d panels each) ===" % NC)
    print(f"  {'metric':34s} {'native':>10s} {'ActSim':>10s}")
    print(f"  {'Abrupt  detection':34s} {na_d:10.3f} {aa_d:10.3f}")
    print(f"  {'Gradual detection':34s} {ng_d:10.3f} {ag_d:10.3f}")
    print(f"  {'Gradual peak alarm prob (mean)':34s} {ng_p:10.3f} {ag_p:10.3f}")
    print(f"  {'Return lag-1 autocorrelation':34s} {ac(nat_grad):+10.3f} {ac(act_grad):+10.3f}")


# ======================================================================================
# Table E3 -- primary results: loss-only BOCPD vs PRISM(adjusted), with paired McNemar
# ======================================================================================
def _wilson(kk, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = kk / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - h)/d, (c + h)/d

def table_E3_headtohead():
    rb.run(n=N, n_ref=NC)                                     # writes results/h2h_summary.json + h2h_runs.csv
    S = json.load(open("results/h2h_summary.json"))["summary"]
    df = pd.read_csv("results/h2h_runs.csv")
    def detect(col, g): f = g[col]; return (f >= K - 2) & (f <= K + 6)
    print("\n=== Table E3: on-time detection, BOCPD vs PRISM(adjusted) (%d paired reps) ===" % N)
    print(f"  {'regime':20s} {'BOCPD':>7s} {'PRISM(adj)':>11s} {'95% CI':>16s}  McNemar(PRISM-only vs BOCPD-only, p)")
    for scen in ["abrupt", "gradual", "subtle"]:
        b = S[scen]["BOCPD"]["detect"]; p = S[scen]["PRISM(adjusted)"]["detect"]
        g = df[df.scenario == scen]; n = len(g); lo, hi = _wilson(int(round(p*n)), n)
        da, db = detect("prism_adj_first", g).values, detect("bocpd_first", g).values
        n10, n01 = int((da & ~db).sum()), int((~da & db).sum())
        pv = stats.binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
        print(f"  {scen:20s} {b:7.3f} {p:11.3f}   [{lo:.3f},{hi:.3f}]   {n10} vs {n01}, p={pv:.2e}")
    print("  false-alarm rate (no real change):")
    for scen in ["none", "transient_up", "transient_dn"]:
        print(f"    {scen:16s} BOCPD={S[scen]['BOCPD']['false_alarm']:.3f}  PRISM(adjusted)={S[scen]['PRISM(adjusted)']['false_alarm']:.3f}")


# ======================================================================================
# Table E4 -- detectability across the curvature of a gradual inflection
# ======================================================================================
def table_E4_curvature():
    pc.run(n=N, n_ref=NC)                                     # writes results/curve_summary.json
    S = json.load(open("results/curve_summary.json"))["summary"]
    slope = {"gentle": "5.1%/yr", "mild": "6.8%/yr", "steep": "8.5%/yr", "very_steep": "10.2%/yr"}
    print("\n=== Table E4: detection and lag across curvature (%d reps each) ===" % N)
    print(f"  {'curvature':12s} {'growth':>9s} {'BOCPD':>7s} {'PRISM(adj)':>11s} {'median lag':>11s}")
    for name in ["gentle", "mild", "steep", "very_steep"]:
        r = S[name]
        print(f"  {name:12s} {slope[name]:>9s} {r['bocpd']['detect']:7.3f} {r['prism']['detect']:11.3f} {str(r['prism']['median_lag']):>11s}")


if __name__ == "__main__":
    if QUICK: print(">>> QUICK MODE: small n, numbers are a sanity check only, NOT the paper values.")
    else:     print(">>> FULL REPRODUCTION: paper n; this takes ~20-30 minutes.")
    table_E1_calibration()
    table_E2_blindness()
    table_E3_headtohead()
    table_E4_curvature()
    print("\n[done] All data synthetic. Detector core prism_sim.py imported unchanged.")
