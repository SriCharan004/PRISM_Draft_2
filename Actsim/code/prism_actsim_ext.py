"""
Draft 2's test suite, re-run on ActSim data (with external indicators paired in).

The loss side of every panel is simulated claim-by-claim with CAS ActSim and aggregated to
quarterly mean severity; the external leading indicators and the exposure path are taken,
seed-for-seed, from prism_sim.generate_panel (which builds indicators that lead onset by
1-2 quarters, exactly as Draft 2 specifies). All four hazard variants are then scored with
prism_sim.evaluate, so the metrics (Detect / Early / False / Lead, window K-2..K+6) and the
four-variant framing are identical to Draft 2 Table 3.
"""
import contextlib, io, json, logging, sys, time, warnings, os
from dataclasses import replace, asdict
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
import prism_sim as ps, prism_noext as px

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
os.makedirs("results", exist_ok=True)
SEED = dict(abrupt=300_000, gradual=310_000, ref=915_000)

def panel_actsim_ext(seed, scenario, inflect, cfg, book=px.BOOK):
    """ActSim loss side + native indicators/exposure (paired), for prism_sim.evaluate."""
    from actsim import ClaimSimulator
    native = ps.generate_panel(seed, scenario, inflect, cfg)         # indicators + exposure + RNG stream
    exposure = native["exposure"] * (book.claims_q0 / (cfg.freq0 * 1000.0))
    lms = px._mean_path(scenario, inflect, cfg) if hasattr(px, "_mean_path") else None
    # build the deterministic severity mean path exactly as prism_noext does
    T, K = cfg.T, cfg.K
    g = ps._trend_path(cfg, scenario, cfg.g0, cfg.g1) if (inflect and cfg.channel=="severity") else np.full(T, cfg.g0)
    drift = g.copy()
    if inflect and cfg.channel=="severity" and scenario=="abrupt": drift[K] += cfg.jump
    drift[0] = 0.0
    log_mean_sev = np.log(cfg.sev0) + np.cumsum(drift)
    pol = px._policies(log_mean_sev, exposure, book, cfg)
    logging.getLogger("actsim").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = ClaimSimulator(pol, random_seed=px.ACTSIM_OFFSET + seed); sim.simulate_claims()
    cl = sim.claim_data
    if len(cl):
        q = (cl["policy_id"].to_numpy().astype(int)) // 1000; amt = cl["amount"].to_numpy(float)
        counts = np.bincount(q, minlength=T); total = np.bincount(q, weights=amt, minlength=T)
    else:
        counts = np.zeros(T, int); total = np.zeros(T)
    sev = np.full(T, np.nan); nz = counts > 0; sev[nz] = total[nz] / counts[nz]
    if not nz[0]: sev[0] = np.exp(log_mean_sev[0])
    for t in range(1, T):
        if not nz[t]: sev[t] = sev[t-1]
    logsev = np.log(sev); logret = np.concatenate([[0.0], np.diff(logsev)])
    freq = counts / exposure; logfreq = np.log(np.clip(freq, 1e-9, None))
    return dict(seed=seed, scenario=scenario, inflect=inflect, channel=cfg.channel, K=K,
                logret=logret, severity=sev, logsev=logsev, exposure=exposure, counts=counts,
                freq=freq, logfreq=logfreq,
                mon_loglevel=logsev if cfg.channel=="severity" else logfreq,
                indicators=native["indicators"])

def _genref(a):
    i, cfg = a; return panel_actsim_ext(SEED["ref"]+i, "gradual", False, cfg)
def _genpan(a):
    i, scen, cfg = a; return panel_actsim_ext(SEED[scen]+i, scen, True, cfg)

def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)

def run(n=400, n_ref=150, procs=2, channel="severity"):
    cfg0 = replace(ps.CFG, channel=channel); t0=time.time()
    with Pool(procs) as pool:
        refp = pool.map(_genref, [(i,cfg0) for i in range(n_ref)], chunksize=8)
    ref = px.build_reference_from_panels(refp, cfg0) if hasattr(px,"build_reference_from_panels") else None
    # internal reference moments + calibrated sd from ActSim in-control panels
    S1=[]; S2=[]; rets=[]
    for p in refp:
        s1,s2 = ps.raw_internal_signals(p, cfg0); S1.append(s1); S2.append(s2); rets.append(p["logret"][1:])
    S1=np.array(S1); S2=np.array(S2)
    ref = dict(s1_mu=float(np.nanmean(S1)), s1_sd=float(np.nanstd(S1)+1e-9),
               s2_mu=float(np.nanmean(S2)), s2_sd=float(np.nanstd(S2)+1e-9))
    ret_sd = float(np.std(np.concatenate(rets)))
    cfg = replace(cfg0, sig_d=ret_sd)
    print(f"[ref] {n_ref} panels {time.time()-t0:.0f}s ret_sd={ret_sd:.4f}", flush=True)
    rows=[]
    for scen in ("abrupt","gradual"):
        with Pool(procs) as pool:
            pans = pool.map(_genpan, [(i,scen,cfg0) for i in range(n)], chunksize=8)
        for i,p in enumerate(pans):
            res,_ = ps.evaluate(p, ref, cfg)
            for v in ps.VARIANTS:
                rows.append(dict(run=i, scenario=scen, variant=v, **res[v]))
        print(f"[{scen}] {n} panels ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"results/actsimext_{channel}_runs.csv", index=False)
    # summarise into Table-3 form
    out={}
    for scen,g in df.groupby("scenario"):
        out[scen]={}
        for v in ps.VARIANTS:
            s=g[g.variant==v]; nn=len(s); det=int(s.detect.sum()); lo,hi=wilson(det,nn)
            out[scen][v]=dict(n=nn, detect=round(det/nn,3), lo=round(lo,3), hi=round(hi,3),
                              early=round(float(s.early.mean()),3), false=round(float(s["false"].mean()),3),
                              lead=round(float(s.lead.dropna().mean()),2) if det else None)
        # McNemar PRISM vs severity-only
        a=g[g.variant=="prism"].set_index("run").detect.astype(bool)
        b=g[g.variant=="severity_only"].set_index("run").detect.astype(bool)
        n10=int((a&~b).sum()); n01=int((~a&b).sum())
        p=stats.binomtest(n10,n10+n01,0.5).pvalue if (n10+n01) else 1.0
        out[scen]["mcnemar_prism_vs_severity"]=dict(prism_only=n10, sev_only=n01, p=p)
    json.dump(dict(meta=dict(n=n,n_ref=n_ref,ret_sd=ret_sd,channel=channel,
                             note="ActSim loss + native indicators; four variants via prism_sim.evaluate"),
                   summary=out), open(f"results/actsimext_{channel}_summary.json","w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return out

if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 400
    ch=sys.argv[2] if len(sys.argv)>2 else "severity"
    run(n, channel=ch)
