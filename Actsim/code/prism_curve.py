"""
Gradual shifts of different CURVATURE (steepness) on ActSim data.

The pre-inflection growth is held fixed; after the inflection the growth rate ramps up to
g0 * ratio over `ramp` quarters. Larger ratio = sharper upward bend = higher curvature.
Matches the user's "2,2,2 -> 2,2,3 / 2,2,4 / 2,2,5" idea (ratios 1.5 / 2.0 / 2.5 ...).

  flat        ratio 1.0   no change (baseline / false-alarm check)
  gentle      ratio 1.5   2 -> 3
  mild        ratio 2.0   2 -> 4
  steep       ratio 2.5   2 -> 5
  very_steep  ratio 3.0   2 -> 6

Methods (no external): plain BOCPD (raw, flat hazard) vs PRISM(adjusted) = Hampel filter
+ internal-signal hazard. Reported per curvature: eventual detection, on-time detection,
and detection lag from onset K.
"""
import contextlib, io, json, logging, sys, time, warnings, os
from dataclasses import replace, asdict
from multiprocessing import Pool
import numpy as np, pandas as pd
import prism_sim as ps, prism_noext as px, prism_robust as rb

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
os.makedirs("results", exist_ok=True)

CURVES = [("flat", 1.0), ("gentle", 1.5), ("mild", 2.0), ("steep", 2.5), ("very_steep", 3.0)]
RATIO = dict(CURVES)
SEED0 = 840_000

def curve_mean_path(ratio, cfg):
    T, K = cfg.T, cfg.K
    g1 = cfg.g0 * ratio
    g = np.full(T, cfg.g0)
    for j in range(cfg.ramp):
        if K + j < T: g[K + j] = cfg.g0 + (g1 - cfg.g0) * (j + 1) / cfg.ramp
    g[K + cfg.ramp:] = g1
    g[0] = 0.0
    return np.log(cfg.sev0) + np.cumsum(g)

def gen(seed, name, cfg, book=px.BOOK):
    from actsim import ClaimSimulator
    native = ps.generate_panel(seed, "gradual", False, cfg)
    exposure = native["exposure"] * (book.claims_q0 / (cfg.freq0 * 1000.0))
    lms = curve_mean_path(RATIO[name], cfg)
    pol = px._policies(lms, exposure, book, cfg)
    logging.getLogger("actsim").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = ClaimSimulator(pol, random_seed=px.ACTSIM_OFFSET + seed); sim.simulate_claims()
    cl = sim.claim_data; T = cfg.T
    if len(cl):
        q = (cl["policy_id"].to_numpy().astype(int)) // 1000; amt = cl["amount"].to_numpy(float)
        counts = np.bincount(q, minlength=T); total = np.bincount(q, weights=amt, minlength=T)
    else:
        counts = np.zeros(T, int); total = np.zeros(T)
    sev = np.full(T, np.nan); nz = counts > 0; sev[nz] = total[nz] / counts[nz]
    if not nz[0]: sev[0] = np.exp(lms[0])
    for t in range(1, T):
        if not nz[t]: sev[t] = sev[t - 1]
    return dict(scenario=name, logsev=np.log(sev))

def _gen(a):
    i, name = a; return gen(SEED0 + i, name, replace(ps.CFG, channel="severity"))

def run(n=80, n_ref=100, procs=2):
    cfg0 = replace(ps.CFG, channel="severity"); t0 = time.time()
    jobs = [(i, name) for name, _ in CURVES for i in range(n)]
    with Pool(procs) as pool:
        panels = pool.map(_gen, jobs, chunksize=8)
    print(f"[gen] {len(panels)} panels {time.time()-t0:.0f}s", flush=True)
    with Pool(procs) as pool:
        refp = pool.map(_gen, [(970_000 + i, "flat") for i in range(n_ref)], chunksize=8)
    ref_raw = [p["logsev"] for p in refp]; ref_filt = [rb.hampel(x) for x in ref_raw]
    print(f"[gen] ref {time.time()-t0:.0f}s", flush=True)
    ref_f = rb.build_ref(ref_filt, cfg0, use_filter=True)
    sd_raw = float(np.std(np.concatenate([np.diff(x) for x in ref_raw])))
    cfg_raw = replace(cfg0, sig_d=sd_raw); cfg_flt = replace(cfg0, sig_d=ref_f["ret_sd"]); K = cfg0.K
    rows = []
    for p in panels:
        raw = p["logsev"]; flt = rb.hampel(raw)
        lr_raw = np.concatenate([[0.0], np.diff(raw)]); lr_flt = np.concatenate([[0.0], np.diff(flt)])
        A_b = ps.bocpd(lr_raw, cfg_raw.h0, cfg_raw)
        s = rb.signal(flt, cfg_flt, ref_f)
        haz = ps.sigmoid(ps.logit(cfg_flt.h0) + cfg_flt.beta_int * s)
        A_p = ps.bocpd(lr_flt, haz, cfg_flt)
        rows.append(dict(scenario=p["scenario"],
                         bocpd=rb.first_A(A_b, cfg_raw) or -1,
                         prism=rb.first_A(A_p, cfg_flt) or -1))
    df = pd.DataFrame(rows); df.to_csv("results/curve_runs.csv", index=False)
    def summ(g, col, hascp):
        f = g[col]; na = f < 0; ff = f[~na]; nn = len(g)
        within = ((ff >= K - 2) & (ff <= K + 6)).sum(); ev = (ff >= K - 2).sum(); anyal = (~na).sum()
        lag = float(np.median(ff[ff >= K - 2] - K)) if ev else None
        d = dict(anyalarm=round(anyal / nn, 3))
        if hascp:
            d.update(detect=round(within / nn, 3), eventual=round(ev / nn, 3), median_lag=lag)
        else:
            d.update(false_alarm=round(anyal / nn, 3))
        return d
    out = {}
    for name, ratio in CURVES:
        g = df[df.scenario == name]; hascp = name != "flat"
        out[name] = dict(ratio=ratio, bocpd=summ(g, "bocpd", hascp), prism=summ(g, "prism", hascp))
    json.dump(dict(meta=dict(n=n, n_ref=n_ref, sd_raw=sd_raw, sd_flt=ref_f["ret_sd"],
                             thr=ref_f["thr"], ramp=cfg0.ramp, note="PRISM(adjusted)=Hampel+internal hazard"),
                   summary=out), open("results/curve_summary.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
