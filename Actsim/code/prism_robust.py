"""
Robust PRISM on ActSim data: a reversion-aware (Hampel) pre-filter so PRISM ignores
one-off outlier quarters but still detects genuine trend inflections.

Two goals, tested together:
  (1) "If we cap the outliers, PRISM should not respond" -- a transient one-quarter
      severity spike (a catastrophic-claim quarter) should NOT trigger an alarm.
  (2) "Make PRISM work" -- gradual, subtle and abrupt real inflections should still be
      detected.

A one-off outlier is a single anomalous quarter that reverts. A Hampel filter compares
each quarter to its local median and replaces only isolated spikes, so a transient outlier
is removed while a sustained shift (several quarters at a new level) survives.

Scenarios (severity channel, K=24):
  none          no change                    (false-alarm test)
  transient_up  +one-quarter spike at K       (outlier; PRISM should stay quiet)
  transient_dn  -one-quarter dip at K          (outlier; PRISM should stay quiet)
  gradual       gradual upward inflection      (real change; should detect)
  subtle        weaker gradual inflection      (real change; should detect)
  abrupt        permanent jump at K            (real change; filter must NOT remove it)

Detectors (internal-driven, no external):
  baseline   internal signal on the raw series
  robust     internal signal on the Hampel-filtered series
"""
import contextlib, io, json, logging, os, sys, time, warnings
from dataclasses import replace, asdict
from multiprocessing import Pool
import numpy as np, pandas as pd
import prism_sim as ps, prism_noext as px, prism_reco as pr

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
os.makedirs("results", exist_ok=True)
SCEN = ["none", "transient_up", "transient_dn", "gradual", "subtle", "abrupt"]
HASCP = {"gradual", "subtle", "abrupt"}
SPIKE = 0.16          # one-quarter log spike (~17% severity blip)
SEED0 = 820_000; T_BURN = 4

def base_kind(k):
    if k in ("none", "transient_up", "transient_dn"): return "none"
    if k == "subtle":  return "subtle_path"
    return k          # gradual, abrupt

def scen_mean_path(kind, cfg):
    if kind == "subtle_path":
        T, K = cfg.T, cfg.K; g = np.full(T, cfg.g0)
        hi = cfg.g0 + 0.55 * (cfg.g1 - cfg.g0)
        for j in range(cfg.ramp):
            if K + j < T: g[K + j] = cfg.g0 + (hi - cfg.g0) * (j + 1) / cfg.ramp
        g[K + cfg.ramp:] = hi; g[0] = 0.0
        return np.log(cfg.sev0) + np.cumsum(g)
    return pr.log_mean_sev(kind, cfg)

def gen(seed, kind, cfg, book=px.BOOK):
    from actsim import ClaimSimulator
    native = ps.generate_panel(seed, "gradual", False, cfg)
    exposure = native["exposure"] * (book.claims_q0 / (cfg.freq0 * 1000.0))
    lms = scen_mean_path(base_kind(kind), cfg)
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
    logsev = np.log(sev)
    if kind == "transient_up": logsev[cfg.K] += SPIKE      # one-quarter blip, does NOT propagate
    if kind == "transient_dn": logsev[cfg.K] -= SPIKE
    return dict(seed=seed, scenario=kind, logsev=logsev)

def _gen(a):
    i, k = a; return gen(SEED0 + i, k, replace(ps.CFG, channel="severity"))

def hampel(x, h=2, k=3.0):
    """Centered Hampel filter: replace isolated spikes with the local median.
    Uses up to h future quarters, i.e. an h-quarter confirmation lag."""
    y = x.copy(); n = len(x)
    for t in range(n):
        lo, hi = max(0, t - h), min(n, t + h + 1)
        w = x[lo:hi]; med = np.median(w); mad = np.median(np.abs(w - med)) + 1e-12
        if abs(x[t] - med) > k * 1.4826 * mad:
            y[t] = med
    return y

def signal(logsev, cfg, ref, w=6):
    S1, S2 = pr.raw_signals_w(logsev, cfg, w)
    z1 = (S1 - ref["s1_mu"]) / ref["s1_sd"]; z2 = (S2 - ref["s2_mu"]) / ref["s2_sd"]
    return np.nan_to_num(np.nanmean(np.vstack([z1, z2]), axis=0), nan=0.0)

def build_ref(ref_logsev, cfg, use_filter, w=6):
    proc = [hampel(x) if use_filter else x for x in ref_logsev]
    s1s, s2s = [], []
    for ls in proc:
        S1, S2 = pr.raw_signals_w(ls, cfg, w); s1s.append(S1); s2s.append(S2)
    s1s, s2s = np.array(s1s), np.array(s2s)
    ref = dict(s1_mu=np.nanmean(s1s), s1_sd=np.nanstd(s1s) + 1e-9,
               s2_mu=np.nanmean(s2s), s2_sd=np.nanstd(s2s) + 1e-9)
    zmax = [np.nanmax(signal(ls, cfg, ref, w)[T_BURN:]) for ls in proc]
    ref["thr"] = float(np.percentile(zmax, 95))
    ref["ret_sd"] = float(np.std(np.concatenate([np.diff(ls) for ls in proc])))
    return ref

def first_ge(s, thr):
    idx = np.where(s[T_BURN:] >= thr)[0]; return int(idx[0] + T_BURN) if len(idx) else None
def first_A(A, cfg):
    idx = np.where(A[T_BURN:] >= cfg.tau)[0]; return int(idx[0] + T_BURN) if len(idx) else None

def evaluate(panels, ref_logsev, cfg0, use_filter):
    ref = build_ref(ref_logsev, cfg0, use_filter)
    cfg = replace(cfg0, sig_d=ref["ret_sd"]); rows = []
    for p in panels:
        ls = hampel(p["logsev"]) if use_filter else p["logsev"]
        s = signal(ls, cfg, ref)
        haz = ps.sigmoid(ps.logit(cfg.h0) + cfg.beta_int * s)
        lr = np.concatenate([[0.0], np.diff(ls)])
        A_int = ps.bocpd(lr, haz, cfg)
        rows.append(dict(scenario=p["scenario"], sig_first=first_ge(s, ref["thr"]) or -1,
                         haz_first=first_A(A_int, cfg) or -1))
    return pd.DataFrame(rows), ref

def evaluate_headtohead(panels, ref_logsev_raw, ref_logsev_filt, cfg0):
    """Plain BOCPD (raw, flat hazard) vs PRISM(adjusted) = Hampel filter + internal hazard.
    Also BOCPD on the filtered series, to isolate the filter's own effect."""
    ref_f = build_ref(ref_logsev_filt, cfg0, use_filter=True)      # filtered internal reference
    sd_raw = float(np.std(np.concatenate([np.diff(x) for x in ref_logsev_raw])))
    sd_flt = ref_f["ret_sd"]
    cfg_raw = replace(cfg0, sig_d=sd_raw); cfg_flt = replace(cfg0, sig_d=sd_flt)
    rows = []
    for p in panels:
        raw = p["logsev"]; flt = hampel(raw)
        lr_raw = np.concatenate([[0.0], np.diff(raw)]); lr_flt = np.concatenate([[0.0], np.diff(flt)])
        A_bocpd = ps.bocpd(lr_raw, cfg_raw.h0, cfg_raw)                    # plain BOCPD, raw
        A_bocpd_f = ps.bocpd(lr_flt, cfg_flt.h0, cfg_flt)                  # BOCPD on filtered
        s = signal(flt, cfg_flt, ref_f)
        haz = ps.sigmoid(ps.logit(cfg_flt.h0) + cfg_flt.beta_int * s)
        A_padj = ps.bocpd(lr_flt, haz, cfg_flt)                           # PRISM(adjusted)
        rows.append(dict(scenario=p["scenario"],
                         bocpd_first=first_A(A_bocpd, cfg_raw) or -1,
                         bocpd_filt_first=first_A(A_bocpd_f, cfg_flt) or -1,
                         prism_adj_first=first_A(A_padj, cfg_flt) or -1,
                         prism_adj_sig_first=first_ge(s, ref_f["thr"]) or -1))
    return pd.DataFrame(rows), dict(sd_raw=sd_raw, sd_flt=sd_flt, thr=ref_f["thr"])

def summarise(df, cfg):
    K = cfg.K; out = {}
    for scen, g in df.groupby("scenario"):
        hascp = scen in HASCP; out[scen] = {}
        for col, name in (("sig_first", "internal_signal"), ("haz_first", "internal_hazard")):
            f = g[col]; na = f < 0; ff = f[~na]; n = len(g)
            within = ((ff >= K - 2) & (ff <= K + 6)).sum(); anyal = (~na).sum()
            d = dict(anyalarm=round(anyal / n, 3))
            if hascp:
                d.update(detect=round(within / n, 3), eventual=round((ff >= K - 2).sum() / n, 3))
            else:
                d.update(false_alarm=round(anyal / n, 3))
            out[scen][name] = d
    return out

METHODS = ["bocpd_first", "bocpd_filt_first", "prism_adj_first", "prism_adj_sig_first"]
MNAME = {"bocpd_first": "BOCPD", "bocpd_filt_first": "BOCPD+filter",
         "prism_adj_first": "PRISM(adjusted)", "prism_adj_sig_first": "PRISM(adjusted,signal)"}

def summarise_h2h(df, cfg):
    K = cfg.K; out = {}
    for scen, g in df.groupby("scenario"):
        hascp = scen in HASCP; out[scen] = {}
        for col in METHODS:
            f = g[col]; na = f < 0; ff = f[~na]; n = len(g)
            within = ((ff >= K - 2) & (ff <= K + 6)).sum(); anyal = (~na).sum()
            d = dict(anyalarm=round(anyal / n, 3))
            if hascp:
                d.update(detect=round(within / n, 3), eventual=round((ff >= K - 2).sum() / n, 3),
                         late=round((ff > K + 6).sum() / n, 3))
            else:
                d.update(false_alarm=round(anyal / n, 3))
            out[scen][MNAME[col]] = d
    return out

def run(n=80, n_ref=100, procs=2):
    cfg0 = replace(ps.CFG, channel="severity"); t0 = time.time()
    jobs = [(i, k) for k in SCEN for i in range(n)]
    with Pool(procs) as pool:
        panels = pool.map(_gen, jobs, chunksize=8)
    print(f"[gen] {len(panels)} panels {time.time()-t0:.0f}s", flush=True)
    with Pool(procs) as pool:
        refp = pool.map(_gen, [(950_000 + i, "none") for i in range(n_ref)], chunksize=8)
    ref_raw = [p["logsev"] for p in refp]
    ref_filt = [hampel(x) for x in ref_raw]
    print(f"[gen] ref {time.time()-t0:.0f}s", flush=True)
    df, meta = evaluate_headtohead(panels, ref_raw, ref_filt, cfg0)
    df.to_csv("results/h2h_runs.csv", index=False)
    res = summarise_h2h(df, replace(cfg0, sig_d=meta["sd_flt"]))
    json.dump(dict(meta=dict(n=n, n_ref=n_ref, spike=SPIKE, filter="centered Hampel h=2 k=3", **meta),
                   summary=res), open("results/h2h_summary.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s sd_raw={meta['sd_raw']:.4f} sd_flt={meta['sd_flt']:.4f} thr={meta['thr']:.2f}", flush=True)
    return res

if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
