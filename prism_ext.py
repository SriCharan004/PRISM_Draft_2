"""
PRISM Draft-2 extensions. Separate module; prism_sim.py is untouched so every
Draft-1 number still reproduces exactly.

Three additions, each targeting a stated gap in Draft 1:

  E1. Lognormal-Normal-Inverse-Gamma severity UPM (Student-t predictive).
      Draft 1 promised this in 2.4 but shipped only the variance-fixed special
      case. Implementing it lets us test the untested limitation: does a
      heavier-tailed severity process detect later?

  E2. Two-stream corroboration gate. Draft 1's stress test found PRISM
      false-alarms ~40% of the time when indicators fire with no real change.
      The gate requires the internal predictor-stability stream to agree before
      an alarm escalates, and we measure whether it removes false alarms
      without removing detection.

  E3. Empirical-Bayes / hierarchical estimation of the hazard weight beta.
      Draft 1 fixed beta = 1 a priori and listed weak identification as its
      main theoretical limitation. Here beta is estimated by maximising the
      BOCPD marginal likelihood, pooled across synthetic "lines of business".

All data is synthetic; all seeds are fixed. Usage:
    python prism_ext.py [e1 e2 e3 ...]      (default: all three)
"""
import json, os, sys
import numpy as np
from dataclasses import replace
from scipy import stats
import prism_sim as ps

cfg = ps.CFG
T, K = cfg.T, cfg.K
os.makedirs("results", exist_ok=True)


# ======================================================================================
# E1.  Full Normal-Inverse-Gamma severity UPM (Student-t predictive)
# ======================================================================================
def bocpd_nig(x, hazard, cfg=cfg, m0=None, kappa0=1.0, a0=6.0, b0=None):
    """BOCPD with a full NIG within-segment model: x | mu, s2 ~ N(mu, s2), unknown s2.

    Conjugate prior NIG(m0, kappa0, a0, b0) on (mu, s2). Per run length the
    posterior predictive is Student-t with 2a degrees of freedom, location m and
    scale^2 = b(kappa+1)/(a*kappa). Unlike the variance-fixed model, this one
    learns the segment variance -- which is why a shift can be partly absorbed
    into an inflated variance rather than declared a changepoint.

    a0 is set moderately informative (6.0) so early segments do not have runaway
    variance; b0 defaults to the in-control variance scaled to that prior.
    """
    n = len(x)
    haz = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard)
    haz = np.clip(haz, 1e-4, 0.99)
    if m0 is None:
        m0 = cfg.g0
    if b0 is None:
        b0 = cfg.sig_d ** 2 * (a0 - 1)          # E[s2] = b0/(a0-1) = sig_d^2

    m = np.array([m0]); kap = np.array([kappa0])
    a = np.array([a0]); b = np.array([b0])
    post = np.array([1.0])
    A = np.zeros(n); cap = 80
    for t in range(n):
        xt = x[t]
        nu = 2.0 * a
        scale = np.sqrt(np.maximum(b * (kap + 1.0) / (a * kap), 1e-300))
        pred = stats.t.pdf(xt, df=nu, loc=m, scale=scale)
        growth = post * pred * (1 - haz[t])
        cp = np.sum(post * pred * haz[t])
        new_post = np.concatenate([[cp], growth])
        ssum = new_post.sum()
        new_post = new_post / ssum if ssum > 0 else np.ones_like(new_post) / len(new_post)
        # NIG conjugate update for grown runs
        kap_g = kap + 1.0
        m_g = (kap * m + xt) / kap_g
        a_g = a + 0.5
        b_g = b + 0.5 * kap * (xt - m) ** 2 / kap_g
        m = np.concatenate([[m0], m_g]); kap = np.concatenate([[kappa0], kap_g])
        a = np.concatenate([[a0], a_g]); b = np.concatenate([[b0], b_g])
        post = new_post
        if len(post) > cap:
            post = post[:cap]; post = post / post.sum()
            m, kap, a, b = m[:cap], kap[:cap], a[:cap], b[:cap]
        A[t] = post[:cfg.r_star].sum()
    return A


def generate_panel_heavy(seed, scenario="gradual", inflect=True, cfg=cfg, df=4.0):
    """Panel whose severity log-returns are heavy-tailed (Student-t, df), rescaled
    to the same standard deviation sig_d so only the TAIL differs from Draft 1."""
    p = ps.generate_panel(seed, scenario, inflect, cfg)
    rng = np.random.default_rng(20_000_000 + seed)
    g = np.full(T, cfg.g0)
    if inflect:
        if scenario == "abrupt":
            g[K:] = cfg.g1
        else:
            for j in range(cfg.ramp):
                if K + j < T:
                    g[K + j] = cfg.g0 + (cfg.g1 - cfg.g0) * (j + 1) / cfg.ramp
            g[K + cfg.ramp:] = cfg.g1
    raw = rng.standard_t(df, T)
    raw = raw / np.sqrt(df / (df - 2.0))          # unit variance
    logret = g + cfg.sig_d * raw
    if inflect and scenario == "abrupt":
        logret[K] += cfg.jump
    logret[0] = 0.0
    logsev = np.log(cfg.sev0) + np.cumsum(logret)
    p.update(logret=logret, logsev=logsev, severity=np.exp(logsev), mon_loglevel=logsev)
    return p


def e1_nig_severity(n=400):
    """Compare the variance-fixed UPM against the full NIG UPM, on Gaussian-tailed
    and heavy-tailed (t_4) severity, for the gradual regime."""
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    out = {}
    for tail, gen in (("gaussian", lambda s, inf: ps.generate_panel(s, "gradual", inf, cfg)),
                      ("heavy_t4", lambda s, inf: generate_panel_heavy(s, "gradual", inf, cfg))):
        out[tail] = {}
        for upm, fn in (("fixed_variance", ps.bocpd), ("full_nig", bocpd_nig)):
            for variant in ("severity_only", "prism"):
                det = early = fal = 0
                delays = []
                for i in range(n):
                    p = gen(5000 + i, True)
                    haz = ps.hazard_for(variant, p, ref, cfg)
                    fa = ps.first_alarm(fn(p["logret"], haz, cfg), cfg)
                    if fa is not None and (K - 2) <= fa <= (K + 6):
                        det += 1; delays.append(fa - K)
                    if fa is not None and (K - 2) <= fa <= K: early += 1
                    q = gen(860000 + i, False)
                    hq = ps.hazard_for(variant, q, ref, cfg)
                    if ps.first_alarm(fn(q["logret"], hq, cfg), cfg) is not None: fal += 1
                out[tail][f"{upm}|{variant}"] = dict(
                    detect=det / n, early=early / n, false=fal / n,
                    median_delay=float(np.median(delays)) if delays else None)
    json.dump(out, open("results/ext_e1_nig.json", "w"), indent=2)
    return out


# ======================================================================================
# E2.  Two-stream corroboration gate
# ======================================================================================
def corroborated_first_alarm(panel, ref, cfg=cfg, gate=0.0, t_burn=4, window=2):
    """PRISM alarm that only escalates when the INTERNAL stream also agrees.

    The Bayesian alarm A_t >= tau is computed exactly as in Draft 1. It escalates
    only if the internal predictor-stability composite is at or above `gate`
    (in z units) at the alarm quarter or within the preceding `window` quarters --
    i.e. the actuary's own model must also show degradation, not just the
    external indicators. gate = -inf reproduces ungated PRISM.
    """
    A = ps.run_alarm("prism", panel, ref, cfg)
    z_int = ps.internal_composite(panel, ref, cfg)
    for t in range(t_burn, len(A)):
        if A[t] >= cfg.tau:
            lo = max(0, t - window)
            if np.nanmax(z_int[lo:t + 1]) >= gate:
                return t
    return None


def _bumped_indicator_panel(seed, cfg=cfg):
    """No real inflection, but indicators fire -- the adversarial scenario."""
    q = ps.generate_panel(860000 + seed, "gradual", inflect=False, cfg=cfg)
    rng = np.random.default_rng(12000 + seed)
    ind = rng.normal(0, cfg.ind_noise, (cfg.n_ind, T))
    for k in range(cfg.n_ind):
        start = K - cfg.ind_lead - k; peak = K + 1
        for t in range(max(start, 0), T):
            frac = (t - start + 1) / max(peak - start + 1, 1)
            ind[k, t] += cfg.ind_peak * np.clip(frac, 0, 1) if t <= peak \
                else cfg.ind_peak * max(0.0, 1 - (t - peak) / cfg.ind_decay)
    qq = dict(q); qq["indicators"] = ind
    return qq


def e2_corroboration(n=400):
    """Does requiring internal agreement remove the misleading-indicator false
    alarms without removing genuine detection? Sweep the gate threshold."""
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    gates = [-np.inf, 0.0, 0.5, 1.0, 1.5]
    out = []
    for gate in gates:
        det = fooled = fal = 0
        for i in range(n):
            p = ps.generate_panel(5000 + i, "gradual", inflect=True, cfg=cfg)
            fa = corroborated_first_alarm(p, ref, cfg, gate=gate)
            if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
            q = ps.generate_panel(860000 + i, "gradual", inflect=False, cfg=cfg)
            if corroborated_first_alarm(q, ref, cfg, gate=gate) is not None: fal += 1
            qq = _bumped_indicator_panel(i, cfg)
            if corroborated_first_alarm(qq, ref, cfg, gate=gate) is not None: fooled += 1
        out.append(dict(gate=("none" if gate == -np.inf else gate),
                        detect=det / n, false_null=fal / n, fooled=fooled / n))
    json.dump(out, open("results/ext_e2_corroboration.json", "w"), indent=2)
    return out


# ======================================================================================
# E3.  Empirical-Bayes estimation of the hazard weight beta
# ======================================================================================
def bocpd_logevidence(x, hazard, cfg=cfg):
    """Log marginal likelihood log p(x_1:T) under the BOCPD model with the given
    hazard sequence. This is the sum of the per-step normalising constants of the
    run-length recursion, so maximising it over beta is exact empirical Bayes --
    no changepoint labels are used, which is what makes beta identifiable from
    ordinary operating data."""
    n = len(x)
    haz = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard)
    haz = np.clip(haz, 1e-4, 0.99)
    s2 = cfg.sig_d ** 2
    v0 = cfg.nig_kappa * s2
    m = np.array([cfg.g0]); v = np.array([v0]); post = np.array([1.0])
    ll = 0.0; cap = 80
    for t in range(n):
        xt = x[t]
        pred = np.exp(-0.5 * (xt - m) ** 2 / (v + s2)) / np.sqrt(2 * np.pi * (v + s2))
        growth = post * pred * (1 - haz[t])
        cp = np.sum(post * pred * haz[t])
        joint = np.concatenate([[cp], growth])
        Z = joint.sum()
        if Z <= 0 or not np.isfinite(Z):
            return -np.inf
        ll += np.log(Z)
        post = joint / Z
        v_g = 1.0 / (1.0 / v + 1.0 / s2); m_g = v_g * (m / v + xt / s2)
        m = np.concatenate([[cfg.g0], m_g]); v = np.concatenate([[v0], v_g])
        if len(post) > cap:
            post = post[:cap]; post = post / post.sum(); m, v = m[:cap], v[:cap]
    return ll


def _hazard_with_beta(panel, ref, beta_ext, beta_int, cfg=cfg):
    a0 = ps.logit(cfg.h0)
    z_ext = ps.external_composite(panel, cfg)
    z_int = ps.internal_composite(panel, ref, cfg)
    return ps.sigmoid(a0 + beta_ext * z_ext + beta_int * z_int)


def e3_empirical_bayes(n_lines=6, panels_per_line=25, grid=None):
    """Estimate beta by maximising pooled BOCPD marginal likelihood.

    Six synthetic 'lines of business' differ in how informative their indicators
    are (ind_peak from 0.4 to 1.6) -- so the TRUE best beta differs by line. We
    estimate beta per line, and pooled across lines, and compare against the
    a-priori beta = 1 used throughout Draft 1.
    """
    if grid is None:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    peaks = np.linspace(0.4, 1.6, n_lines)
    per_line = []
    pooled = np.zeros(len(grid))
    for li, peak in enumerate(peaks):
        cfgL = replace(cfg, ind_peak=float(peak))
        ref = ps.build_incontrol_reference(120, "gradual", cfgL)
        lls = np.zeros(len(grid))
        for i in range(panels_per_line):
            p = ps.generate_panel(30000 + li * 1000 + i, "gradual", inflect=True, cfg=cfgL)
            for gi, b in enumerate(grid):
                haz = _hazard_with_beta(p, ref, b, b, cfgL)
                lls[gi] += bocpd_logevidence(p["logret"], haz, cfgL)
        pooled += lls
        per_line.append(dict(line=li, ind_peak=float(peak),
                             beta_hat=float(grid[int(np.argmax(lls))]),
                             loglik=[float(v) for v in lls]))
    beta_pooled = float(grid[int(np.argmax(pooled))])

    # Does the estimated beta beat the a-priori beta = 1 out of sample?
    val = {}
    for label, getb in (("beta_prior_1", lambda li: 1.0),
                        ("beta_hat_per_line", lambda li: per_line[li]["beta_hat"]),
                        ("beta_hat_pooled", lambda li: beta_pooled)):
        det = fal = 0; tot = 0
        for li, peak in enumerate(peaks):
            cfgL = replace(cfg, ind_peak=float(peak))
            ref = ps.build_incontrol_reference(120, "gradual", cfgL)
            b = getb(li)
            for i in range(panels_per_line):
                p = ps.generate_panel(70000 + li * 1000 + i, "gradual", inflect=True, cfg=cfgL)
                fa = ps.first_alarm(ps.bocpd(p["logret"], _hazard_with_beta(p, ref, b, b, cfgL), cfgL), cfgL)
                if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
                q = ps.generate_panel(880000 + li * 1000 + i, "gradual", inflect=False, cfg=cfgL)
                fq = ps.first_alarm(ps.bocpd(q["logret"], _hazard_with_beta(q, ref, b, b, cfgL), cfgL), cfgL)
                if fq is not None: fal += 1
                tot += 1
        val[label] = dict(detect=det / tot, false=fal / tot)

    out = dict(grid=grid, per_line=per_line, beta_pooled=beta_pooled, validation=val)
    json.dump(out, open("results/ext_e3_empirical_bayes.json", "w"), indent=2)
    return out


ALL = {"e1": e1_nig_severity, "e2": e2_corroboration, "e3": e3_empirical_bayes}

if __name__ == "__main__":
    names = sys.argv[1:] or ["e1", "e2", "e3"]
    for nm in names:
        print(f"[{nm}] running ...", flush=True)
        r = ALL[nm]()
        print(json.dumps(r, indent=1)[:1500], flush=True)


def e3b_decision_calibration(n_lines=4, panels_per_line=40, grid=None, lambdas=(1.0, 2.0, 4.0)):
    """Constructive follow-up to E3.

    E3 shows beta is not identifiable from the loss-data marginal likelihood: the
    hazard only shapes the run-length prior, one inflection per panel carries
    almost no information about it, and a covariate-driven hazard slightly HURTS
    one-step-ahead predictive density because it tracks noisy indicators. So the
    marginal likelihood prefers beta ~ 0.

    The resolution is that beta is a DECISION parameter, not a statistical one:
    it encodes how the actuary trades a missed inflection against a false alarm.
    Here we calibrate beta by maximising a stated utility
        U(beta) = detection(beta) - lambda * false_alarm(beta)
    where lambda is the cost of a false alarm relative to a missed detection.
    """
    if grid is None:
        grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    peaks = np.linspace(0.4, 1.6, n_lines)
    curves = []
    for b in grid:
        det = fal = tot = 0
        for li, peak in enumerate(peaks):
            cfgL = replace(cfg, ind_peak=float(peak))
            ref = ps.build_incontrol_reference(120, "gradual", cfgL)
            for i in range(panels_per_line):
                p = ps.generate_panel(40000 + li * 1000 + i, "gradual", inflect=True, cfg=cfgL)
                fa = ps.first_alarm(ps.bocpd(p["logret"], _hazard_with_beta(p, ref, b, b, cfgL), cfgL), cfgL)
                if fa is not None and (K - 2) <= fa <= (K + 6): det += 1
                q = ps.generate_panel(890000 + li * 1000 + i, "gradual", inflect=False, cfg=cfgL)
                if ps.first_alarm(ps.bocpd(q["logret"], _hazard_with_beta(q, ref, b, b, cfgL), cfgL), cfgL) is not None:
                    fal += 1
                tot += 1
        curves.append(dict(beta=b, detect=det / tot, false=fal / tot))
    best = {}
    for lam in lambdas:
        u = [c["detect"] - lam * c["false"] for c in curves]
        bi = int(np.argmax(u))
        best[f"lambda_{lam}"] = dict(beta_star=curves[bi]["beta"], utility=float(u[bi]),
                                     detect=curves[bi]["detect"], false=curves[bi]["false"])
    out = dict(curves=curves, optimal_by_lambda=best)
    json.dump(out, open("results/ext_e3b_decision.json", "w"), indent=2)
    return out


ALL["e3b"] = e3b_decision_calibration


# ======================================================================================
# E4.  Posterior calibration -- is A_t a probability, or just a score?
# ======================================================================================
def e4_calibration(n=500, nbins=10):
    """Reliability check on the alarm statistic.

    A_t = P(r_t < r* | x_1:t) claims to be the posterior probability that a
    changepoint occurred within the last r* quarters. We test that claim
    directly: pool every (A_t, truth) pair over many panels, bin by predicted
    probability, and compare the bin's mean prediction against the observed
    frequency of the event it predicts.

    Ground truth at quarter t is: did the true inflection onset K fall within
    the last r* quarters, i.e. 0 <= t - K < r*. Panels are half with an
    inflection and half without, so the event has a realistic base rate.
    Burn-in quarters are excluded (A_t is trivially ~1 there by construction).

    Reported: per-bin observed frequency, plus Expected Calibration Error (ECE)
    and Brier score for PRISM and the loss-only monitor.
    """
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    out = {}
    for variant in ("severity_only", "prism"):
        preds, truth = [], []
        for i in range(n):
            inflect = (i % 2 == 0)
            p = ps.generate_panel(50000 + i, "gradual", inflect=inflect, cfg=cfg)
            A = ps.run_alarm(variant, p, ref, cfg)
            for t in range(4, T):                      # skip burn-in
                y = 1 if (inflect and 0 <= (t - K) < cfg.r_star) else 0
                preds.append(float(A[t])); truth.append(y)
        preds = np.array(preds); truth = np.array(truth)
        edges = np.linspace(0.0, 1.0, nbins + 1)
        bins = []
        ece = 0.0
        for b in range(nbins):
            lo, hi = edges[b], edges[b + 1]
            m = (preds >= lo) & (preds < hi) if b < nbins - 1 else (preds >= lo) & (preds <= hi)
            cnt = int(m.sum())
            if cnt == 0:
                bins.append(dict(lo=float(lo), hi=float(hi), n=0, mean_pred=None, obs_freq=None))
                continue
            mp = float(preds[m].mean()); of = float(truth[m].mean())
            bins.append(dict(lo=float(lo), hi=float(hi), n=cnt, mean_pred=mp, obs_freq=of))
            ece += (cnt / len(preds)) * abs(mp - of)
        brier = float(np.mean((preds - truth) ** 2))
        # AUC separates DISCRIMINATION (does a higher A_t mean a likelier change?)
        # from CALIBRATION (does A_t = 0.6 mean 60%?). Computed via the rank identity.
        pos = preds[truth == 1]; neg = preds[truth == 0]
        if len(pos) and len(neg):
            r = stats.rankdata(np.concatenate([pos, neg]))
            auc = float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
        else:
            auc = None
        out[variant] = dict(bins=bins, ece=float(ece), brier=brier, auc=auc,
                            base_rate=float(truth.mean()), n_points=int(len(preds)))
    json.dump(out, open("results/ext_e4_calibration.json", "w"), indent=2)
    return out


# ======================================================================================
# E5.  Multivariate hazard -- monitor frequency and severity jointly
# ======================================================================================
def bocpd_joint(panel, hazard, cfg=cfg):
    """Run-length recursion driven by BOTH channels at once.

    A single run-length posterior is shared by two conditionally-independent
    within-segment models: the Gaussian UPM on severity log-returns and the
    Poisson-Gamma UPM on counts with an exposure offset. The per-step predictive
    is the product of the two channel predictives, so evidence for a regime
    change accumulates from either channel (or both).
    """
    x = panel["logret"]; N = panel["counts"]; E = panel["exposure"]
    n = len(x)
    haz = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard)
    haz = np.clip(haz, 1e-4, 0.99)
    s2 = cfg.sig_d ** 2; v0 = cfg.nig_kappa * s2
    m = np.array([cfg.g0]); v = np.array([v0])
    a0, b0 = cfg.gam_a, cfg.gam_b
    a = np.array([a0]); b = np.array([b0])
    post = np.array([1.0]); A = np.zeros(n); cap = 80
    for t in range(n):
        pred_s = np.exp(-0.5 * (x[t] - m) ** 2 / (v + s2)) / np.sqrt(2 * np.pi * (v + s2))
        pr = b / (b + E[t])
        pred_f = np.exp(stats.nbinom.logpmf(N[t], a, pr))
        pred = pred_s * pred_f
        growth = post * pred * (1 - haz[t]); cp = np.sum(post * pred * haz[t])
        joint = np.concatenate([[cp], growth]); Z = joint.sum()
        post = joint / Z if Z > 0 else np.ones_like(joint) / len(joint)
        v_g = 1.0 / (1.0 / v + 1.0 / s2); m_g = v_g * (m / v + x[t] / s2)
        m = np.concatenate([[cfg.g0], m_g]); v = np.concatenate([[v0], v_g])
        a = np.concatenate([[a0], a + N[t]]); b = np.concatenate([[b0], b + E[t]])
        if len(post) > cap:
            post = post[:cap]; post = post / post.sum()
            m, v, a, b = m[:cap], v[:cap], a[:cap], b[:cap]
        A[t] = post[:cfg.r_star].sum()
    return A


def generate_panel_both(seed, scenario="gradual", inflect=True, cfg=cfg):
    """Panel where the inflection moves BOTH channels (severity trend and claim
    rate), which is the realistic case a single-channel monitor half-misses."""
    ps_sev = ps.generate_panel(seed, scenario, inflect, replace(cfg, channel="severity"))
    ps_frq = ps.generate_panel(seed, scenario, inflect, replace(cfg, channel="frequency"))
    p = dict(ps_sev)
    p["counts"] = ps_frq["counts"]; p["freq"] = ps_frq["freq"]; p["logfreq"] = ps_frq["logfreq"]
    return p


def e5_joint_channels(n=400):
    """Compare single-channel monitors against the joint recursion when the
    inflection moves both frequency and severity."""
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    cfg_f = replace(cfg, channel="frequency")
    out = {}
    for scenario in ("gradual", "abrupt"):
        out[scenario] = {}
        for variant in ("severity_only", "prism"):
            res = {k: 0 for k in ("sev", "frq", "joint")}
            fals = {k: 0 for k in ("sev", "frq", "joint")}
            for i in range(n):
                p = generate_panel_both(60000 + i, scenario, True, cfg)
                q = generate_panel_both(870000 + i, scenario, False, cfg)
                haz_p = ps.hazard_for(variant, p, ref, cfg)
                haz_q = ps.hazard_for(variant, q, ref, cfg)
                cand = {
                    "sev": (ps.bocpd(p["logret"], haz_p, cfg), ps.bocpd(q["logret"], haz_q, cfg)),
                    "frq": (ps.bocpd_poisson(p["counts"], p["exposure"], haz_p, cfg_f),
                            ps.bocpd_poisson(q["counts"], q["exposure"], haz_q, cfg_f)),
                    "joint": (bocpd_joint(p, haz_p, cfg), bocpd_joint(q, haz_q, cfg)),
                }
                for k, (Ap, Aq) in cand.items():
                    fa = ps.first_alarm(Ap, cfg)
                    if fa is not None and (K - 2) <= fa <= (K + 6): res[k] += 1
                    if ps.first_alarm(Aq, cfg) is not None: fals[k] += 1
            out[scenario][variant] = {k: dict(detect=res[k] / n, false=fals[k] / n) for k in res}
    json.dump(out, open("results/ext_e5_joint.json", "w"), indent=2)
    return out


ALL["e4"] = e4_calibration
ALL["e5"] = e5_joint_channels
