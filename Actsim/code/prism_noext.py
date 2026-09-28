"""
PRISM x ActSim, external-free study.

Data: loss side simulated claim-by-claim with CAS ActSim, aggregated to a quarterly
mean-severity series (severity channel). Indicators are NOT used anywhere.

Scenarios:
    gradual  : trend steepens over `ramp` quarters from K   (inflect=True, "gradual")
    none     : no inflection at all                          (inflect=False)
    sudden   : one-time jump + steeper trend at K            (inflect=True, "abrupt")

Views (no external anywhere):
    severity_only : plain BOCPD on the loss series, flat hazard
    internal_only : BOCPD on the loss series, hazard driven by internal predictors
                    (= PRISM with external removed; "severity + internal")
    internal_sig  : the internal predictor thresholded on its own, no BOCPD
                    (= "internal predictors only")
"""
import contextlib, io, json, logging, os, sys, time, warnings
from dataclasses import dataclass, replace, asdict
import numpy as np, pandas as pd
from scipy import stats
import prism_sim as ps

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
os.makedirs("results", exist_ok=True)

SEED = dict(gradual=200_000, none=210_000, sudden=220_000, ref=910_000)
ACTSIM_OFFSET = 50_000_000
T_BURN = 4

@dataclass(frozen=True)
class Book:
    claims_q0: float = 4000.0
    sev_sigma: float = 0.5
    class_rel: tuple = (0.6, 1.0, 1.8)
    class_mix: tuple = (0.50, 0.35, 0.15)
    blocks: int = 8

BOOK = Book()

# ----- ActSim bridge (severity channel) -------------------------------------------------
def _import():
    from actsim import ClaimSimulator
    return ClaimSimulator

def _mean_path(scenario, inflect, cfg):
    T, K = cfg.T, cfg.K
    g = ps._trend_path(cfg, scenario, cfg.g0, cfg.g1) if inflect else np.full(T, cfg.g0)
    drift = g.copy()
    if inflect and scenario == "abrupt":
        drift[K] += cfg.jump
    drift[0] = 0.0
    return np.log(cfg.sev0) + np.cumsum(drift)

def _policies(log_mean_sev, exposure, book, cfg):
    rel = np.array(book.class_rel); mix = np.array(book.class_mix)
    rel = rel / np.sum(rel * mix)
    lam = cfg.freq0
    rows = []
    for t in range(cfg.T):
        start = pd.Timestamp("2015-01-01") + pd.DateOffset(months=3 * t)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        for c in range(len(rel)):
            m = lam * exposure[t] * mix[c] / book.blocks
            mu = log_mean_sev[t] + np.log(rel[c]) - 0.5 * book.sev_sigma ** 2
            for b in range(book.blocks):
                rows.append((t * 1000 + c * 100 + b, "poisson", (float(m),), "lognormal",
                             (float(mu), float(book.sev_sigma)), start, end))
    return pd.DataFrame(rows, columns=["policy_id", "freq_dist", "freq_params", "sev_dist",
                                       "sev_params", "start_date", "end_date"])

def panel_actsim(seed, scenario, inflect, cfg, book=BOOK):
    ClaimSimulator = _import()
    native = ps.generate_panel(seed, scenario if scenario != "none" else "gradual", inflect, cfg)
    exposure = native["exposure"] * (book.claims_q0 / (cfg.freq0 * 1000.0))
    log_mean_sev = _mean_path(scenario if scenario != "none" else "gradual", inflect, cfg)
    pol = _policies(log_mean_sev, exposure, book, cfg)
    logging.getLogger("actsim").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = ClaimSimulator(pol, random_seed=ACTSIM_OFFSET + seed); sim.simulate_claims()
    cl = sim.claim_data; T = cfg.T
    if len(cl):
        q = (cl["policy_id"].to_numpy().astype(int)) // 1000
        amt = cl["amount"].to_numpy(float)
        counts = np.bincount(q, minlength=T); total = np.bincount(q, weights=amt, minlength=T)
    else:
        counts = np.zeros(T, int); total = np.zeros(T)
    sev = np.full(T, np.nan); nz = counts > 0; sev[nz] = total[nz] / counts[nz]
    if not nz[0]: sev[0] = np.exp(log_mean_sev[0])
    for t in range(1, T):
        if not nz[t]: sev[t] = sev[t - 1]
    logsev = np.log(sev); logret = np.concatenate([[0.0], np.diff(logsev)])
    return dict(seed=seed, scenario=scenario, inflect=inflect, channel="severity", K=cfg.K,
                logret=logret, severity=sev, logsev=logsev, mon_loglevel=logsev,
                exposure=exposure, counts=counts)

# ----- reference + calibration ----------------------------------------------------------
def build_reference(n_ref, cfg, book):
    s1s, s2s, rets, zmax = [], [], [], []
    for i in range(n_ref):
        p = panel_actsim(SEED["ref"] + i, "none", False, cfg, book)
        S1, S2 = ps.raw_internal_signals(p, cfg)
        s1s.append(S1); s2s.append(S2); rets.append(p["logret"][1:])
    s1s, s2s = np.array(s1s), np.array(s2s)
    ref = dict(s1_mu=float(np.nanmean(s1s)), s1_sd=float(np.nanstd(s1s) + 1e-9),
               s2_mu=float(np.nanmean(s2s)), s2_sd=float(np.nanstd(s2s) + 1e-9))
    ret_sd = float(np.std(np.concatenate(rets)))
    # threshold for the standalone internal-signal detector: 95th pct of in-control max z
    for i in range(n_ref):
        p = panel_actsim(SEED["ref"] + 500 + i, "none", False, cfg, book)
        z = ps.internal_composite(p, ref, cfg)
        zmax.append(np.nanmax(z[T_BURN:]))
    thr = float(np.percentile(zmax, 95))
    return ref, ret_sd, thr

# ----- the three views ------------------------------------------------------------------
def first_ge(series, thr, t_burn=T_BURN):
    idx = np.where(series[t_burn:] >= thr)[0]
    return int(idx[0] + t_burn) if len(idx) else None

def alarms(panel, ref, cfg, thr_int):
    z = ps.internal_composite(panel, ref, cfg)
    haz_flat = cfg.h0
    haz_int = ps.sigmoid(ps.logit(cfg.h0) + cfg.beta_int * z)
    A_sev = ps.bocpd(panel["logret"], haz_flat, cfg)
    A_int = ps.bocpd(panel["logret"], haz_int, cfg)
    return {"severity_only": ps.first_alarm(A_sev, cfg),
            "internal_only": ps.first_alarm(A_int, cfg),
            "internal_sig":  first_ge(z, thr_int)}

def score(fa, K, inflected):
    if fa is None:
        return dict(first=None, detect=False, early=False, false=False, lead=np.nan)
    if inflected:
        detect = (K - 2) <= fa <= (K + 6)
        return dict(first=fa, detect=detect, early=(K - 2) <= fa <= K,
                    false=4 <= fa <= (K - 4), lead=float(K - fa) if detect else np.nan)
    else:  # no-inflection: any alarm after burn-in is a false alarm
        return dict(first=fa, detect=False, early=False, false=True, lead=np.nan)

VIEWS = ["severity_only", "internal_only", "internal_sig"]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

def run(n=120, n_ref=100):
    cfg0 = replace(ps.CFG, channel="severity")
    t0 = time.time()
    ref, ret_sd, thr = build_reference(n_ref, cfg0, BOOK)
    cfg = replace(cfg0, sig_d=ret_sd)  # calibrate BOCPD variance to ActSim noise
    print(f"[ref] built {n_ref} panels in {time.time()-t0:.0f}s  ret_sd={ret_sd:.4f} thr_int={thr:.2f}", flush=True)
    rows = []
    for scen in ("gradual", "none", "sudden"):
        base = "abrupt" if scen == "sudden" else ("gradual" if scen == "gradual" else "none")
        inflected = scen != "none"
        for i in range(n):
            p = panel_actsim(SEED[scen] + i, base, inflected, cfg, BOOK)
            fa = alarms(p, ref, cfg, thr)
            for v in VIEWS:
                rows.append(dict(run=i, scenario=scen, view=v, **score(fa[v], cfg.K, inflected)))
            if (i + 1) % 40 == 0:
                print(f"[{scen}] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/noext_runs.csv", index=False)
    summary = {}
    for scen, g in df.groupby("scenario"):
        inflected = scen != "none"
        summary[scen] = {}
        for v in VIEWS:
            s = g[g.view == v]; nn = len(s)
            det = int(s.detect.sum()); fls = int(s["false"].sum())
            lo, hi = wilson(det, nn)
            summary[scen][v] = dict(
                n=nn,
                detect=round(det / nn, 3) if inflected else None,
                detect_lo=round(lo, 3) if inflected else None,
                detect_hi=round(hi, 3) if inflected else None,
                early=round(float(s.early.mean()), 3) if inflected else None,
                false_alarm=round(fls / nn, 3),
                mean_lead=round(float(s.lead.dropna().mean()), 2) if (inflected and det) else None)
    meta = dict(n=n, n_ref=n_ref, ret_sd=ret_sd, thr_int=thr, detector_sig_d=cfg.sig_d,
                book=asdict(BOOK), external_used=False,
                note="Loss data from CAS ActSim, aggregated to quarterly mean severity. No external indicators used.")
    out = dict(meta=meta, summary=summary)
    json.dump(out, open("results/noext_summary.json", "w"), indent=2)
    print("[done]", round(time.time()-t0, 0), "s", flush=True)
    return out

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run(n)
