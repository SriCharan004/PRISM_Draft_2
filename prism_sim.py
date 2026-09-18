"""
PRISM simulator: covariate-driven Bayesian online changepoint detection for
loss-cost trend-inflection monitoring.

Synthetic experiment ONLY. No real claims data is used anywhere in this file.
All data-generating parameters are declared in CONFIG and are fixed ("pre-registered")
before any results are computed; nothing here is tuned to the outcome.

The inflection can be placed on EITHER channel and at ANY quarter:
  * channel = "severity"  -> Gaussian (known-variance) UPM on severity log-returns
  * channel = "frequency" -> Poisson-Gamma UPM on claim counts with exposure offset
  * K (inflection quarter) is a free parameter.

Two streams drive the hazard, regardless of channel:
  * external leading indicators (noisy series that LEAD the inflection)
  * internal predictor-stability signals from a baseline model frozen pre-cutoff
"""
import numpy as np
from dataclasses import dataclass
from scipy import stats

# --------------------------------------------------------------------------------------
# CONFIG  (all design choices fixed in advance)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    T: int = 40
    K: int = 24                 # inflection quarter (free parameter)
    B: int = 16                 # pre-cutoff quarters used to freeze the baseline model
    channel: str = "severity"   # "severity" or "frequency" -- which channel inflects & is monitored
    # severity stylized facts
    sev0: float = 15000.0
    g0: float = 0.0085          # pre-inflection severity log-growth (~3.4%/yr)
    g1: float = 0.0190          # post-inflection severity log-growth (~7.8%/yr)
    sig_d: float = 0.016        # sd of severity log-returns
    jump: float = 0.085         # abrupt one-time severity log-jump at K
    ramp: int = 8               # quarters over which a gradual inflection ramps g0 -> g1
    # exposure
    exp_g: float = 0.008
    exp_sd: float = 0.0015
    # frequency stylized facts
    freq0: float = 0.06         # baseline claim frequency per unit exposure
    freq_g0: float = 0.005      # pre-inflection frequency log-growth (keeps RNG stream stable)
    freq_g1: float = 0.018      # post-inflection frequency log-growth
    freq_jump: float = 0.16     # abrupt one-time frequency log-jump at K
    # external indicators (lead the inflection, noisy -- NOT oracles)
    ind_lead: int = 2
    ind_peak: float = 1.2
    ind_noise: float = 1.0
    ind_decay: int = 6
    n_ind: int = 4
    # hazard (pre-registered)
    h0: float = 0.04
    beta_ext: float = 1.0
    beta_int: float = 1.0
    # alarm
    r_star: int = 4
    tau: float = 0.50
    win_roll: int = 6
    # severity Gaussian-UPM prior (variance-fixed NIG special case)
    nig_kappa: float = 1.0
    nig_a: float = 2.0
    # frequency Poisson-Gamma prior on the rate
    gam_a: float = 60.0
    gam_b: float = 1000.0


CFG = Config()
sigmoid = lambda u: 1.0 / (1.0 + np.exp(-u))
logit = lambda p: np.log(p / (1.0 - p))


def _trend_path(cfg, scenario, base, hi):
    """Quarterly growth path with an abrupt or gradual inflection at K."""
    g = np.full(cfg.T, base)
    if scenario == "abrupt":
        g[cfg.K:] = hi
    elif scenario == "gradual":
        for j in range(cfg.ramp):
            if cfg.K + j < cfg.T:
                g[cfg.K + j] = base + (hi - base) * (j + 1) / cfg.ramp
        g[cfg.K + cfg.ramp:] = hi
    return g


# --------------------------------------------------------------------------------------
# DATA GENERATION
# --------------------------------------------------------------------------------------
def generate_panel(seed, scenario="abrupt", inflect=True, cfg=CFG):
    """Synthetic quarterly panel. The inflection is placed on cfg.channel at quarter cfg.K.
    RNG draw order is fixed so the severity-channel path is identical regardless of channel."""
    rng = np.random.default_rng(seed)
    T, K, ch = cfg.T, cfg.K, cfg.channel

    # --- severity ---
    g = _trend_path(cfg, scenario, cfg.g0, cfg.g1) if (inflect and ch == "severity") else np.full(T, cfg.g0)
    logret = g + rng.normal(0.0, cfg.sig_d, T)            # RNG draw 1
    if inflect and ch == "severity" and scenario == "abrupt":
        logret[K] += cfg.jump
    logret[0] = 0.0
    logsev = np.log(cfg.sev0) + np.cumsum(logret)
    severity = np.exp(logsev)

    # --- exposure ---
    exposure = np.exp(np.cumsum(rng.normal(cfg.exp_g, cfg.exp_sd, T))) * 1000.0   # RNG draw 2

    # --- frequency ---
    fg = _trend_path(cfg, scenario, cfg.freq_g0, cfg.freq_g1) if (inflect and ch == "frequency") else np.full(T, cfg.freq_g0)
    log_lam = np.log(cfg.freq0) + np.cumsum(fg)
    if inflect and ch == "frequency" and scenario == "abrupt":
        log_lam[K:] += cfg.jump if False else cfg.freq_jump   # one-time level shift carried forward
    lam = np.exp(log_lam)
    counts = rng.poisson(lam * exposure)                  # RNG draw 3
    freq = counts / exposure
    logfreq = np.log(np.clip(freq, 1e-9, None))

    # --- external indicators: in-control N(0,1) noise + a leading bump around K ---
    indicators = rng.normal(0.0, cfg.ind_noise, (cfg.n_ind, T))   # RNG draw 4
    if inflect:
        for k in range(cfg.n_ind):
            start = K - cfg.ind_lead - k
            peak = K + 1
            for t in range(max(start, 0), T):
                if t <= peak:
                    frac = (t - start + 1) / max(peak - start + 1, 1)
                    bump = cfg.ind_peak * np.clip(frac, 0, 1)
                else:
                    bump = cfg.ind_peak * max(0.0, 1 - (t - peak) / cfg.ind_decay)
                indicators[k, t] += bump

    mon_loglevel = logsev if ch == "severity" else logfreq
    return dict(seed=seed, scenario=scenario, inflect=inflect, channel=ch, K=K,
                logret=logret, severity=severity, logsev=logsev,
                exposure=exposure, counts=counts, freq=freq, logfreq=logfreq,
                mon_loglevel=mon_loglevel, indicators=indicators)


# --------------------------------------------------------------------------------------
# INTERNAL PREDICTOR-STABILITY SIGNALS  (channel-agnostic: monitors the frozen model's
# out-of-sample residual on the monitored channel's log-level)
# --------------------------------------------------------------------------------------
def _ols_line(t, y):
    A = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def raw_internal_signals(panel, cfg=CFG):
    y = panel["mon_loglevel"]; T = cfg.T; B = cfg.B
    t_idx = np.arange(T)
    b0, b1 = _ols_line(t_idx[:B], y[:B])
    resid = y - (b0 + b1 * t_idx)
    S1 = np.full(T, np.nan); S2 = np.full(T, np.nan)
    w = cfg.win_roll
    for t in range(B, T):
        seg = resid[t - w + 1:t + 1]
        S1[t] = np.mean(seg)
        if len(seg) > 2 and np.std(seg) > 1e-9:
            ac = np.corrcoef(seg[:-1], seg[1:])[0, 1]
            S2[t] = 0.0 if np.isnan(ac) else ac
        else:
            S2[t] = 0.0
    return S1, S2

def build_incontrol_reference(n_panels=200, scenario="abrupt", cfg=CFG, seed0=900000):
    s1s, s2s = [], []
    for i in range(n_panels):
        p = generate_panel(seed0 + i, scenario=scenario, inflect=False, cfg=cfg)
        S1, S2 = raw_internal_signals(p, cfg)
        s1s.append(S1); s2s.append(S2)
    s1s = np.array(s1s); s2s = np.array(s2s)
    return dict(s1_mu=np.nanmean(s1s), s1_sd=np.nanstd(s1s) + 1e-9,
                s2_mu=np.nanmean(s2s), s2_sd=np.nanstd(s2s) + 1e-9)

def internal_composite(panel, ref, cfg=CFG):
    S1, S2 = raw_internal_signals(panel, cfg)
    z1 = (S1 - ref["s1_mu"]) / ref["s1_sd"]
    z2 = (S2 - ref["s2_mu"]) / ref["s2_sd"]
    z = np.nanmean(np.vstack([z1, z2]), axis=0)
    return np.nan_to_num(z, nan=0.0)

def external_composite(panel, cfg=CFG):
    return panel["indicators"].mean(axis=0)


# --------------------------------------------------------------------------------------
# BOCPD -- severity (Gaussian known-variance) and frequency (Poisson-Gamma)
# --------------------------------------------------------------------------------------
def bocpd(x, hazard, cfg=CFG):
    """Gaussian known-variance BOCPD on returns x with a per-step covariate hazard."""
    n = len(x)
    haz = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard)
    haz = np.clip(haz, 1e-4, 0.99)
    s2 = cfg.sig_d ** 2
    v0 = cfg.nig_kappa * s2
    m = np.array([cfg.g0]); v = np.array([v0]); post = np.array([1.0])
    A = np.zeros(n); cap = 80
    for t in range(n):
        xt = x[t]
        pred = np.exp(-0.5 * (xt - m) ** 2 / (v + s2)) / np.sqrt(2 * np.pi * (v + s2))
        growth = post * pred * (1 - haz[t]); cp = np.sum(post * pred * haz[t])
        new_post = np.concatenate([[cp], growth]); new_post /= new_post.sum()
        v_g = 1.0 / (1.0 / v + 1.0 / s2); m_g = v_g * (m / v + xt / s2)
        m = np.concatenate([[cfg.g0], m_g]); v = np.concatenate([[v0], v_g]); post = new_post
        if len(post) > cap:
            post = post[:cap]; post /= post.sum(); m, v = m[:cap], v[:cap]
        A[t] = post[:cfg.r_star].sum()
    return A

def bocpd_poisson(counts, exposure, hazard, cfg=CFG):
    """Poisson-Gamma BOCPD on claim counts with an exposure offset.
    Within-segment: N_t ~ Poisson(lambda * E_t), conjugate Gamma(a,b) prior on the rate;
    posterior predictive for the next count is Negative-Binomial. Detects a shift in the rate."""
    n = len(counts)
    haz = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard)
    haz = np.clip(haz, 1e-4, 0.99)
    a0, b0 = cfg.gam_a, cfg.gam_b
    a = np.array([a0]); b = np.array([b0]); post = np.array([1.0])
    A = np.zeros(n); cap = 80
    for t in range(n):
        N = counts[t]; E = exposure[t]
        p = b / (b + E)                                   # NB success prob
        pred = np.exp(stats.nbinom.logpmf(N, a, p))       # predictive per run length
        growth = post * pred * (1 - haz[t]); cp = np.sum(post * pred * haz[t])
        new_post = np.concatenate([[cp], growth]); new_post /= new_post.sum()
        a_g = a + N; b_g = b + E                          # Gamma conjugate update
        a = np.concatenate([[a0], a_g]); b = np.concatenate([[b0], b_g]); post = new_post
        if len(post) > cap:
            post = post[:cap]; post /= post.sum(); a, b = a[:cap], b[:cap]
        A[t] = post[:cfg.r_star].sum()
    return A


# --------------------------------------------------------------------------------------
# VARIANTS & METRICS
# --------------------------------------------------------------------------------------
VARIANTS = ["severity_only", "external_only", "internal_only", "prism"]

def hazard_for(variant, panel, ref, cfg=CFG):
    a0 = logit(cfg.h0)
    z_ext = external_composite(panel, cfg)
    z_int = internal_composite(panel, ref, cfg)
    if variant == "severity_only":
        return cfg.h0
    if variant == "external_only":
        return sigmoid(a0 + cfg.beta_ext * z_ext)
    if variant == "internal_only":
        return sigmoid(a0 + cfg.beta_int * z_int)
    if variant == "prism":
        return sigmoid(a0 + cfg.beta_ext * z_ext + cfg.beta_int * z_int)
    raise ValueError(variant)

def run_alarm(variant, panel, ref, cfg=CFG):
    haz = hazard_for(variant, panel, ref, cfg)
    if cfg.channel == "severity":
        return bocpd(panel["logret"], haz, cfg)
    return bocpd_poisson(panel["counts"], panel["exposure"], haz, cfg)

def first_alarm(A, cfg=CFG, t_burn=4):
    idx = np.where(A[t_burn:] >= cfg.tau)[0]
    return int(idx[0] + t_burn) if len(idx) else None

def evaluate(panel, ref, cfg=CFG):
    K = panel["K"]; out = {}; A_store = {}
    for v in VARIANTS:
        A = run_alarm(v, panel, ref, cfg)
        fa = first_alarm(A, cfg); A_store[v] = A
        if fa is None:
            out[v] = dict(first=None, detect=False, early=False, false=False, lead=np.nan)
        else:
            detect = (K - 2) <= fa <= (K + 6)
            early = (K - 2) <= fa <= K
            false = 4 <= fa <= (K - 4)
            out[v] = dict(first=fa, detect=detect, early=early, false=false,
                          lead=float(K - fa) if detect else np.nan)
    return out, A_store

def run_experiment(n=1000, scenario="abrupt", cfg=CFG, seed0=1000):
    ref = build_incontrol_reference(200, scenario=scenario, cfg=cfg)
    rows = []
    for i in range(n):
        p = generate_panel(seed0 + i, scenario=scenario, inflect=True, cfg=cfg)
        res, _ = evaluate(p, ref, cfg)
        for v in VARIANTS:
            rows.append(dict(run=i, scenario=scenario, channel=cfg.channel, variant=v, **res[v]))
    return rows, ref


if __name__ == "__main__":
    import pandas as pd
    rows, _ = run_experiment(n=60, scenario="gradual")
    df = pd.DataFrame(rows)
    print(df.groupby("variant")[["detect", "early", "false"]].mean())


# ======================================================================================
# S3 EXTENSION: multi-predictor panels + Kendall-tau ranking-distance signal
# --------------------------------------------------------------------------------------
# Default paths above are untouched (zero regression risk). These functions add an
# optional covariate block to the DGP so predictor-importance rankings exist and can
# reshuffle at the inflection -- which is what S3 monitors.
# ======================================================================================
N_COV = 4
COV_SHARE = 0.5              # fraction of log-return variance explained by covariates
W_PRE  = np.array([0.55, 0.30, 0.12, 0.03])   # pre-inflection importance ordering 1>2>3>4
W_POST = np.array([0.05, 0.15, 0.32, 0.48])   # post-inflection: ordering reverses

def generate_panel_cov(seed, scenario="gradual", inflect=True, reshuffle=True, cfg=CFG):
    """Panel whose log-returns are partly driven by N_COV observable rating covariates.
    Total return variance is held at sig_d^2 (calibration preserved):
      logret_t = g_t + x_t' w(t) + eps_t,  var(x'w)=COV_SHARE*sig_d^2, var(eps)=(1-COV_SHARE)*sig_d^2.
    At the inflection the weight vector shifts W_PRE -> W_POST (over the ramp for 'gradual'),
    reshuffling which covariates matter -- the structural change S3 is designed to catch."""
    p = generate_panel(seed, scenario, inflect, cfg)          # consumes the same RNG draws
    rng = np.random.default_rng(10_000_000 + seed)            # separate stream for the extension
    T, K = cfg.T, cfg.K
    X = rng.normal(0.0, 1.0, (N_COV, T))
    # weight path
    Wt = np.tile(W_PRE, (T, 1))
    if inflect and reshuffle:
        if scenario == "abrupt":
            Wt[K:] = W_POST
        else:
            for j in range(cfg.ramp):
                if K + j < T:
                    lam = (j + 1) / cfg.ramp
                    Wt[K + j] = (1 - lam) * W_PRE + lam * W_POST
            Wt[K + cfg.ramp:] = W_POST
    # scale so covariate part has variance COV_SHARE*sig_d^2 each quarter
    scale = np.sqrt(COV_SHARE) * cfg.sig_d / np.linalg.norm(W_PRE)
    cov_part = np.einsum("kt,tk->t", X, Wt) * scale
    resid = rng.normal(0.0, np.sqrt(1 - COV_SHARE) * cfg.sig_d, T)
    logret = np.where(np.arange(T) == 0, 0.0, p["logret"] * 0)  # rebuild deterministically
    g = p["logret"] * 0 + cfg.g0
    # reconstruct trend path identical to generate_panel
    if inflect and cfg.channel == "severity":
        if scenario == "abrupt":
            g[K:] = cfg.g1
        else:
            for j in range(cfg.ramp):
                if K + j < T: g[K + j] = cfg.g0 + (cfg.g1 - cfg.g0) * (j + 1) / cfg.ramp
            g[K + cfg.ramp:] = cfg.g1
    logret = g + cov_part + resid
    if inflect and cfg.channel == "severity" and scenario == "abrupt":
        logret[K] += cfg.jump
    logret[0] = 0.0
    logsev = np.log(cfg.sev0) + np.cumsum(logret)
    p.update(logret=logret, logsev=logsev, severity=np.exp(logsev),
             mon_loglevel=logsev, X=X)
    return p

def raw_s3(panel, cfg=CFG):
    """S3_t: Kendall-tau ranking distance between the current predictor-importance ordering
    and the frozen pre-cutoff ordering. Importance_j = |corr(X_j, logret)| over a rolling
    window (robust with short windows). Distance in [0,1]; 0 = same ordering."""
    X = panel["X"]; y = panel["logret"]; T = cfg.T; B = cfg.B; w = cfg.win_roll + 2
    def importance(lo, hi):
        return np.array([abs(np.corrcoef(X[j, lo:hi], y[lo:hi])[0, 1]) for j in range(N_COV)])
    imp0 = importance(1, B)                     # frozen ranking (skip t=0)
    rank0 = np.argsort(np.argsort(-imp0))
    S3 = np.full(T, np.nan)
    for t in range(B, T):
        imp = importance(t - w + 1, t + 1)
        rank = np.argsort(np.argsort(-imp))
        tau = stats.kendalltau(rank0, rank).statistic
        S3[t] = (1 - (0.0 if np.isnan(tau) else tau)) / 2.0
    return S3

def build_incontrol_reference_s3(n_panels=200, scenario="gradual", cfg=CFG, seed0=900000):
    """In-control reference for S1, S2 AND S3, from no-inflection covariate panels."""
    s1s, s2s, s3s = [], [], []
    for i in range(n_panels):
        p = generate_panel_cov(seed0 + i, scenario, inflect=False, cfg=cfg)
        S1, S2 = raw_internal_signals(p, cfg); S3 = raw_s3(p, cfg)
        s1s.append(S1); s2s.append(S2); s3s.append(S3)
    s1s, s2s, s3s = map(np.array, (s1s, s2s, s3s))
    return dict(s1_mu=np.nanmean(s1s), s1_sd=np.nanstd(s1s) + 1e-9,
                s2_mu=np.nanmean(s2s), s2_sd=np.nanstd(s2s) + 1e-9,
                s3_mu=np.nanmean(s3s), s3_sd=np.nanstd(s3s) + 1e-9)

def internal_composite_s3(panel, ref, cfg=CFG, use_s3=True):
    S1, S2 = raw_internal_signals(panel, cfg)
    z = [(S1 - ref["s1_mu"]) / ref["s1_sd"], (S2 - ref["s2_mu"]) / ref["s2_sd"]]
    if use_s3:
        S3 = raw_s3(panel, cfg)
        z.append((S3 - ref["s3_mu"]) / ref["s3_sd"])
    zc = np.nanmean(np.vstack(z), axis=0)
    return np.nan_to_num(zc, nan=0.0)

def hazard_for_s3(variant, panel, ref, cfg=CFG, use_s3=True):
    a0 = logit(cfg.h0)
    z_ext = external_composite(panel, cfg)
    z_int = internal_composite_s3(panel, ref, cfg, use_s3=use_s3)
    if variant == "severity_only": return cfg.h0
    if variant == "external_only": return sigmoid(a0 + cfg.beta_ext * z_ext)
    if variant == "internal_only": return sigmoid(a0 + cfg.beta_int * z_int)
    if variant == "prism":         return sigmoid(a0 + cfg.beta_ext * z_ext + cfg.beta_int * z_int)
    raise ValueError(variant)
