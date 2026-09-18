"""Regenerates every supplementary number cited in Sections 4.2-4.4 of the PRISM paper.
Each function is named after the claim it backs and writes a JSON under results/.
All seeds are fixed; all data is synthetic. Usage:  python extra_analyses.py [name ...]
Names: roc, tuned, naive, arl, seeds, frontier, confidence, s3   (default: confidence s3)
"""
import json, os, sys
import numpy as np
from dataclasses import replace
from scipy import stats
import prism_sim as ps

cfg = ps.CFG; T, K = ps.CFG.T, ps.CFG.K
os.makedirs("results", exist_ok=True)


def _first(series, thr, t_burn=4):
    idx = np.where(series[t_burn:] >= thr)[0]
    return int(idx[0] + t_burn) if len(idx) else None


def _bump(ind, cfg=cfg, K=K):
    for k in range(cfg.n_ind):
        start = K - cfg.ind_lead - k; peak = K + 1
        for t in range(max(start, 0), cfg.T):
            frac = (t - start + 1) / max(peak - start + 1, 1)
            ind[k, t] += cfg.ind_peak * np.clip(frac, 0, 1) if t <= peak \
                else cfg.ind_peak * max(0.0, 1 - (t - peak) / cfg.ind_decay)
    return ind


def roc_benchmark(n=500):
    """Fig 5: PRISM vs CUSUM vs constant-hazard BOCPD, detection vs false alarm (gradual)."""
    def cusum(logret, k=0.5):
        z = (logret - cfg.g0) / cfg.sig_d
        C = np.zeros(len(z))
        for t in range(1, len(z)):
            C[t] = max(0.0, C[t - 1] + z[t] - k)
        return C
    ref = ps.build_incontrol_reference(200, "gradual", cfg)
    A_s, A_p, CU, A_s0, A_p0, CU0 = [], [], [], [], [], []
    for i in range(n):
        p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfg)
        A_s.append(ps.bocpd(p["logret"], cfg.h0, cfg)); A_p.append(ps.run_alarm("prism", p, ref, cfg)); CU.append(cusum(p["logret"]))
        q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
        A_s0.append(ps.bocpd(q["logret"], cfg.h0, cfg)); A_p0.append(ps.run_alarm("prism", q, ref, cfg)); CU0.append(cusum(q["logret"]))
    def curve(Ain, Anull, thrs):
        out = []
        for th in thrs:
            det = np.mean([(lambda fa: fa is not None and (K - 2) <= fa <= (K + 6))(_first(a, th)) for a in Ain])
            fal = np.mean([_first(a, th) is not None for a in Anull])
            out.append((fal, det))
        return out
    roc = {"severity_bocpd": curve(A_s, A_s0, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
           "prism": curve(A_p, A_p0, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
           "cusum": curve(CU, CU0, [3, 4, 5, 6, 7, 8, 10, 12])}
    json.dump(roc, open("results/roc_benchmark.json", "w"), indent=2)
    return roc


def tuned_baseline(n=500):
    """§4.4(tuned): best constant hazard for the loss-only monitor (gradual)."""
    out = []
    for h0 in [0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.40]:
        det = fal = 0
        for i in range(n):
            p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfg)
            fa = ps.first_alarm(ps.bocpd(p["logret"], h0, cfg), cfg)
            if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
            if ps.first_alarm(ps.bocpd(q["logret"], h0, cfg), cfg) is not None: fal += 1
        out.append(dict(h0=h0, detect=det / n, false=fal / n))
    json.dump(out, open("results/tuned_baseline.json", "w"), indent=2)
    return out


def naive_rule(n=500):
    """Table 5: indicator-threshold rule vs PRISM, incl. adversarial fooled/blind rates."""
    def nf(ind, z=2.0, consec=2, t_burn=4):
        hits = (ind > z).any(axis=0); run = 0
        for t in range(t_burn, ind.shape[1]):
            run = run + 1 if hits[t] else 0
            if run >= consec: return t
        return None
    res = {"sweep": []}
    for z, c in [(2.0, 2), (2.0, 3), (2.5, 2), (3.0, 2)]:
        det = fal = 0
        for i in range(n):
            p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfg)
            fa = nf(p["indicators"], z, c)
            if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
            if nf(q["indicators"], z, c) is not None: fal += 1
        res["sweep"].append(dict(z=z, consec=c, detect=det / n, false=fal / n))
    fooled = blind = 0
    for i in range(n):
        rng = np.random.default_rng(12000 + i)
        if nf(_bump(rng.normal(0, cfg.ind_noise, (cfg.n_ind, T)))) is not None: fooled += 1
        rng = np.random.default_rng(13000 + i)
        fa = nf(rng.normal(0, cfg.ind_noise, (cfg.n_ind, T)))
        if fa is not None and (K - 2) <= fa <= (K + 6): blind += 1
    res.update(fooled=fooled / n, blind=blind / n)
    json.dump(res, open("results/naive_extra.json", "w"), indent=2)
    return res


def confidence_sweep(n=400):
    """Table 6: the beta (confidence) dial -- detection vs false alarms vs fooled rate."""
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    a0 = ps.logit(cfg.h0)
    sweep = []
    for beta in [0.5, 1.0, 2.0, 4.0]:
        det = fal = fool = 0
        for i in range(n):
            p = ps.generate_panel(3000 + i, "gradual", inflect=True, cfg=cfg)
            haz = ps.sigmoid(a0 + beta * ps.external_composite(p, cfg) + cfg.beta_int * ps.internal_composite(p, ref, cfg))
            fa = ps.first_alarm(ps.bocpd(p["logret"], haz, cfg), cfg)
            if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
            hq = ps.sigmoid(a0 + beta * ps.external_composite(q, cfg) + cfg.beta_int * ps.internal_composite(q, ref, cfg))
            if ps.first_alarm(ps.bocpd(q["logret"], hq, cfg), cfg) is not None: fal += 1
            rng = np.random.default_rng(12000 + i)
            qq = dict(q); qq["indicators"] = _bump(rng.normal(0, cfg.ind_noise, (cfg.n_ind, T)))
            hqq = ps.sigmoid(a0 + beta * ps.external_composite(qq, cfg) + cfg.beta_int * ps.internal_composite(q, ref, cfg))
            if ps.first_alarm(ps.bocpd(q["logret"], hqq, cfg), cfg) is not None: fool += 1
        sweep.append(dict(beta=beta, detect=det / n, false=fal / n, fooled=fool / n))
    json.dump(sweep, open("results/confidence_sweep.json", "w"), indent=2)
    return sweep


def delay_arl(n=500):
    """§4.4(ARL): detection-delay distribution and average run lengths (gradual)."""
    ref = ps.build_incontrol_reference(200, "gradual", cfg)
    out = {}
    for v in ["severity_only", "prism"]:
        delays, arl0 = [], []
        cens = 0
        for i in range(n):
            p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfg)
            fa = ps.first_alarm(ps.run_alarm(v, p, ref, cfg), cfg)
            if fa is not None and fa >= K - 2: delays.append(fa - K)
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
            fq = ps.first_alarm(ps.run_alarm(v, q, ref, cfg), cfg)
            arl0.append(T if fq is None else fq); cens += fq is None
        d = np.array(delays)
        out[v] = dict(median=float(np.median(d)), q25=float(np.percentile(d, 25)), q75=float(np.percentile(d, 75)),
                      arl1=float(d.mean()), arl0_lb=float(np.mean(arl0)), arl0_censored=cens / n)
    json.dump(out, open("results/delay_arl.json", "w"), indent=2)
    return out


def seed_stability(n=400):
    """§4.4(seeds): headline under five independent master seeds (gradual)."""
    import pandas as pd
    out = []
    for s0 in [1000, 11000, 21000, 31000, 41000]:
        rows, _ = ps.run_experiment(n=n, scenario="gradual", cfg=cfg, seed0=s0)
        d = pd.DataFrame(rows)
        out.append(dict(seed0=s0, prism=float(d[d.variant == "prism"].detect.mean()),
                        sev=float(d[d.variant == "severity_only"].detect.mean())))
    json.dump(out, open("results/seed_stability.json", "w"), indent=2)
    return out


def frontier(n=300):
    """§4.4(frontier): detection as the gradual drift shrinks."""
    out = []
    for f in [0.25, 0.50, 0.75, 1.00]:
        cfgf = replace(cfg, g1=cfg.g0 + f * (cfg.g1 - cfg.g0))
        ref = ps.build_incontrol_reference(120, "gradual", cfgf)
        dp = dsv = 0
        for i in range(n):
            p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfgf)
            for v in ("prism", "severity_only"):
                fa = ps.first_alarm(ps.run_alarm(v, p, ref, cfgf), cfgf)
                hit = fa is not None and (K - 2) <= fa <= (K + 6)
                if v == "prism" and hit: dp += 1
                if v == "severity_only" and hit: dsv += 1
        out.append(dict(frac=f, prism=dp / n, sev=dsv / n))
    json.dump(out, open("results/frontier.json", "w"), indent=2)
    return out


def s3_demo(n=400):
    """§4.4(S3): multi-predictor extension -- signal rises on reshuffle; adds no power on short panels."""
    ref = ps.build_incontrol_reference_s3(150, "gradual", cfg)
    p = ps.generate_panel_cov(1004, "gradual", inflect=True, reshuffle=True, cfg=cfg)
    S3 = ps.raw_s3(p, cfg); z3 = (S3 - ref["s3_mu"]) / ref["s3_sd"]
    res = dict(pre_z=float(np.nanmean(z3[cfg.B:K])), post_z=float(np.nanmean(z3[K + 4:K + 13])))
    for use_s3 in (False, True):
        det = {"internal_only": 0, "prism": 0}
        for i in range(n):
            pp = ps.generate_panel_cov(3000 + i, "gradual", inflect=True, reshuffle=True, cfg=cfg)
            for v in det:
                haz = ps.hazard_for_s3(v, pp, ref, cfg, use_s3=use_s3)
                fa = ps.first_alarm(ps.bocpd(pp["logret"], haz, cfg), cfg)
                if fa is not None and (K - 2) <= fa <= (K + 6): det[v] += 1
        res["with_s3" if use_s3 else "s1s2_only"] = {v: det[v] / n for v in det}
    json.dump(res, open("results/s3_demo.json", "w"), indent=2)
    return res


ALL = dict(roc=roc_benchmark, tuned=tuned_baseline, naive=naive_rule, arl=delay_arl,
           seeds=seed_stability, frontier=frontier, confidence=confidence_sweep, s3=s3_demo)

if __name__ == "__main__":
    names = sys.argv[1:] or ["confidence", "s3"]
    for nm in names:
        print(f"[{nm}] ...", flush=True)
        print(json.dumps(ALL[nm](), indent=1)[:400])


def lead_ablation(n=400):
    """Fig 6 / §4.3(1): early-detection rate vs indicator lead (gradual, PRISM & external-only)."""
    out = []
    for L in (0, 1, 2, 3):
        cfgL = replace(cfg, ind_lead=L)
        ref = ps.build_incontrol_reference(150, "gradual", cfgL)
        row = {"ind_lead": L}
        for v in ("external_only", "prism"):
            det = early = 0
            for i in range(n):
                p = ps.generate_panel(3000 + i, "gradual", inflect=True, cfg=cfgL)
                fa = ps.first_alarm(ps.run_alarm(v, p, ref, cfgL), cfgL)
                if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
                if fa is not None and (K - 2) <= fa <= K: early += 1
            row[f"{v}_detect"] = det / n
            row[f"{v}_early"] = early / n
        out.append(row)
    json.dump(out, open("results/lead_ablation.json", "w"), indent=2)
    return out


def snr_sensitivity(n=400):
    """§4.3(2): PRISM detection vs indicator peak signal strength (gradual)."""
    out = []
    for peak in (0.6, 1.2, 1.8):
        cfgP = replace(cfg, ind_peak=peak)
        ref = ps.build_incontrol_reference(150, "gradual", cfgP)
        det = early = fal = 0
        for i in range(n):
            p = ps.generate_panel(3000 + i, "gradual", inflect=True, cfg=cfgP)
            fa = ps.first_alarm(ps.run_alarm("prism", p, ref, cfgP), cfgP)
            if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
            if fa is not None and (K - 2) <= fa <= K: early += 1
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfgP)
            if ps.first_alarm(ps.run_alarm("prism", q, ref, cfgP), cfgP) is not None: fal += 1
        out.append(dict(ind_peak=peak, detect=det / n, early=early / n, false=fal / n))
    json.dump(out, open("results/snr_sensitivity.json", "w"), indent=2)
    return out


ALL["lead"] = lead_ablation
ALL["snr"] = snr_sensitivity


# ======================================================================================
# SCENARIO BATTERY (Annexure B): nine scenarios, including adversarial ones, on the
# severity channel. Reconstructed here so the annexure is reproducible like everything
# else -- it was originally built with one-off code that was never shipped.
# ======================================================================================
SCEN = ["none", "outlier_up", "outlier_down", "ind_false", "subtle", "gradual", "blind", "abrupt", "heavy"]
HASCP = {"subtle", "gradual", "blind", "abrupt", "heavy"}   # scenarios with a real inflection


def _scn_bump(ind, cfg=cfg, K=K):
    for k in range(cfg.n_ind):
        start = K - cfg.ind_lead - k; peak = K + 1
        for t in range(max(start, 0), cfg.T):
            frac = (t - start + 1) / max(peak - start + 1, 1)
            ind[k, t] += cfg.ind_peak * np.clip(frac, 0, 1) if t <= peak \
                else cfg.ind_peak * max(0.0, 1 - (t - peak) / cfg.ind_decay)
    return ind


def _scn_panel(seed, kind, cfg=cfg, K=K):
    """Custom severity-channel panel for one of the nine stress-test scenarios."""
    rng = np.random.default_rng(seed)
    T = cfg.T
    g = np.full(T, cfg.g0); extra = np.zeros(T); ind_bump = False
    if kind == "abrupt":
        g[K:] = cfg.g1; extra[K] += cfg.jump; ind_bump = True
    elif kind == "heavy":
        g[K:] = cfg.g1 * 2.0; extra[K] += cfg.jump * 2.0; ind_bump = True
    elif kind == "gradual":
        for j in range(cfg.ramp):
            if K + j < T: g[K + j] = cfg.g0 + (cfg.g1 - cfg.g0) * (j + 1) / cfg.ramp
        g[K + cfg.ramp:] = cfg.g1; ind_bump = True
    elif kind == "subtle":
        hi = cfg.g0 + 0.55 * (cfg.g1 - cfg.g0)
        for j in range(cfg.ramp):
            if K + j < T: g[K + j] = cfg.g0 + (hi - cfg.g0) * (j + 1) / cfg.ramp
        g[K + cfg.ramp:] = hi; ind_bump = True
    elif kind == "outlier_up":
        extra[K] += 0.06
    elif kind == "outlier_down":
        extra[K] -= 0.06
    elif kind == "ind_false":
        ind_bump = True
    elif kind == "blind":
        for j in range(cfg.ramp):
            if K + j < T: g[K + j] = cfg.g0 + (cfg.g1 - cfg.g0) * (j + 1) / cfg.ramp
        g[K + cfg.ramp:] = cfg.g1
    logret = g + rng.normal(0, cfg.sig_d, T) + extra; logret[0] = 0.0
    logsev = np.log(cfg.sev0) + np.cumsum(logret); severity = np.exp(logsev)
    exposure = np.exp(np.cumsum(rng.normal(cfg.exp_g, cfg.exp_sd, T))) * 1000.0
    indicators = rng.normal(0, cfg.ind_noise, (cfg.n_ind, T))
    if ind_bump: indicators = _scn_bump(indicators, cfg, K)
    return dict(seed=seed, scenario=kind, channel="severity", K=K, logret=logret, severity=severity,
                logsev=logsev, exposure=exposure, counts=np.zeros(T), freq=np.zeros(T), logfreq=np.zeros(T),
                mon_loglevel=logsev, indicators=indicators)


def scenario_battery(n=500):
    """Table 5 / Annexure B: detection (real-inflection scenarios) or false-alarm rate
    (no-change scenarios) for each of the four hazard variants, across nine scenarios."""
    ref = ps.build_incontrol_reference(200, "gradual", cfg)
    res = {}
    for kind in SCEN:
        agg = {v: dict(detect=0, early=0, anyalarm=0) for v in ps.VARIANTS}
        for i in range(n):
            p = _scn_panel(7000 + i, kind)
            out, _ = ps.evaluate(p, ref, cfg)
            for v in ps.VARIANTS:
                o = out[v]
                if o["detect"]: agg[v]["detect"] += 1
                if o["early"]: agg[v]["early"] += 1
                if o["first"] is not None: agg[v]["anyalarm"] += 1
        res[kind] = {v: dict(detect=agg[v]["detect"] / n, early=agg[v]["early"] / n,
                             anyalarm=agg[v]["anyalarm"] / n) for v in ps.VARIANTS}
    json.dump(res, open("results/scenario_battery.json", "w"), indent=2)
    return res


def outlier_recovery(n=400):
    """§ Annexure B: does a single outlier's alarm persist, or recover? PRISM only."""
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    out = {}
    for kind in ("outlier_up", "outlier_down"):
        react = persist = 0
        for i in range(n):
            p = _scn_panel(9000 + i, kind)
            A = ps.run_alarm("prism", p, ref, cfg)
            reacted = bool(np.any(A[K - 1:K + 2] >= cfg.tau))
            later = bool(np.any(A[K + 5:K + 9] >= cfg.tau))
            if reacted:
                react += 1
                if later: persist += 1
        out[kind] = dict(reacted=react / n, persist_given_reacted=persist / max(react, 1))
    json.dump(out, open("results/outlier_recovery.json", "w"), indent=2)
    return out


ALL["scenarios"] = scenario_battery
ALL["outlier_recovery"] = outlier_recovery
