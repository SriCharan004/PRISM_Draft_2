"""Regenerates every figure used in the PRISM paper into figures/.
Run AFTER run_all.py and extra_analyses.py (it reads their results/*.json).
Usage: python make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
import prism_sim as ps

INK, ACC, ACC2, RED, GRN, GRID, LIGHT = (
    "#1f2a37", "#2E5A88", "#7A8B99", "#B23A48", "#3F7D5C", "#D8DEE6", "#EAF0F6")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": INK, "axes.linewidth": 0.8})
os.makedirs("figures", exist_ok=True)
cfg = ps.CFG; K = cfg.K


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fig1_architecture():
    """Schematic; illustrative only, not data-driven."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    def box(x, y, w, h, t, fc=LIGHT, ec=ACC, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if bold else "normal")
    def ar(x1, y1, x2, y2, c=ACC2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.4, color=c))
    box(0.2, 4.5, 3.0, 1.1, "External leading\nindicators  z(ext)", fc="#F3F7FB")
    box(0.2, 2.6, 3.0, 1.1, "Internal predictor-\nstability  z(int) (S1,S2,S3)", fc="#F3F7FB")
    box(0.2, 0.7, 3.0, 1.1, "Loss data: frequency\n& severity  xt", fc="#F3F7FB")
    box(4.0, 3.4, 2.5, 1.3, "Covariate-driven\nhazard\nh(t)=sigma(a+b.z)", fc=LIGHT, bold=True)
    box(4.0, 0.7, 2.5, 1.3, "Within-segment\nactuarial likelihood\n(Gaussian / Poisson-Gamma)", fc=LIGHT)
    box(7.2, 2.0, 2.6, 1.6, "Run-length posterior\nrecursion  P(rt|x1:t)\n-> alarm At", fc="#E4ECF4", bold=True)
    ar(3.2, 5.0, 4.0, 4.3); ar(3.2, 3.15, 4.0, 3.9); ar(3.2, 1.25, 4.0, 1.35); ar(6.5, 4.0, 7.3, 3.4); ar(6.5, 1.35, 7.4, 2.4)
    box(7.2, 0.2, 2.6, 1.0, "Credentialed actuary\n(human-in-the-loop)", fc="#F3F7FB", ec=ACC2); ar(8.5, 2.0, 8.5, 1.2)
    plt.tight_layout(); plt.savefig("figures/fig1_architecture.png", dpi=200, bbox_inches="tight"); plt.close()


def fig2_worked(seed=1004):
    ref = ps.build_incontrol_reference(150, "gradual", cfg)
    p = ps.generate_panel(seed, "gradual", inflect=True, cfg=cfg)
    Ap = ps.run_alarm("prism", p, ref, cfg).copy()
    As = ps.bocpd(p["logret"], cfg.h0, cfg).copy()
    Ap[:4] = np.nan; As[:4] = np.nan
    q = np.arange(cfg.T); fa = ps.first_alarm(ps.run_alarm("prism", p, ref, cfg), cfg)
    fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
    ax1.plot(q, p["severity"] / 1000, color=INK, lw=2, label="Severity ($000)")
    ax1.set_xlabel("Quarter index"); ax1.set_ylabel("Mean severity ($000)", color=INK)
    ax1.axvline(K, ls=":", color=INK, alpha=0.7); ax1.text(K + 0.2, ax1.get_ylim()[0] + 1, "onset", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(q, Ap, color=ACC, lw=2, label="PRISM alarm At")
    ax2.plot(q, As, color=RED, lw=1.6, ls="--", label="Severity-only At")
    ax2.axhline(0.5, color=ACC2, lw=1, ls=":"); ax2.set_ylabel("Alarm statistic At", color=ACC); ax2.set_ylim(0, 1.05)
    if fa is not None:
        ax2.annotate(f"PRISM fires (q{fa})", xy=(fa, 0.5), xytext=(fa - 9, 0.78), fontsize=8.5, color=ACC,
                     arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=8, frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig2_worked.png", dpi=200, bbox_inches="tight"); plt.close()


def fig3_detection():
    summary = json.load(open("results/summary.json"))
    vs = ps.VARIANTS; labels = ["Severity-only", "External only", "Internal only", "PRISM"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8)); x = np.arange(len(vs)); w = 0.38
    for sc, col, off in [("abrupt", ACC2, -w / 2), ("gradual", ACC, w / 2)]:
        det = [summary["severity"][sc][v]["detect"] for v in vs]
        lo = [summary["severity"][sc][v]["detect"] - summary["severity"][sc][v]["lo"] for v in vs]
        hi = [summary["severity"][sc][v]["hi"] - summary["severity"][sc][v]["detect"] for v in vs]
        ax.bar(x + off, det, w, yerr=[lo, hi], capsize=3, color=col, label=sc.capitalize(), edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("Detection rate (95% CI)")
    ax.set_ylim(0, 1.0); ax.grid(axis="y", color=GRID, lw=0.7); ax.legend(frameon=False, fontsize=9)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig3_detection.png", dpi=200, bbox_inches="tight"); plt.close()


def fig4_calibration():
    pre = np.concatenate([ps.generate_panel(700000 + s, "gradual", inflect=False, cfg=cfg)["logret"][1:K] for s in range(300)])
    ks = stats.kstest((pre - cfg.g0) / cfg.sig_d, "norm")
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.hist(pre, bins=40, density=True, color=LIGHT, edgecolor=ACC2, lw=0.6, label="Synthetic")
    xs = np.linspace(pre.min(), pre.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, cfg.g0, cfg.sig_d), color=RED, lw=2, label="Target N(g0, sig^2)")
    ax.set_xlabel("Pre-inflection severity log-return"); ax.set_ylabel("Density")
    ax.text(0.02, 0.95, f"KS p = {ks.pvalue:.2f}", transform=ax.transAxes, fontsize=9, va="top")
    ax.legend(frameon=False, fontsize=9)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig4_calibration.png", dpi=200, bbox_inches="tight"); plt.close()


def fig5_benchmark_roc():
    roc = json.load(open("results/roc_benchmark.json"))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    def plot_curve(key, label, color, marker):
        pts = sorted(roc[key]); ax.plot([p[0] for p in pts], [p[1] for p in pts], marker + "-", color=color, lw=2, label=label, ms=5)
    plot_curve("prism", "PRISM (two-stream hazard)", ACC, "o")
    plot_curve("cusum", "CUSUM control chart (classical)", GRN, "s")
    plot_curve("severity_bocpd", "Constant-hazard BOCPD (loss-only)", RED, "^")
    ax.set_xlabel("False-alarm rate (per in-control decade)"); ax.set_ylabel("Detection rate (gradual inflection)")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig5_benchmark_roc.png", dpi=200, bbox_inches="tight"); plt.close()


def fig6_lead():
    lead = json.load(open("results/lead_ablation.json"))
    L = [r["ind_lead"] for r in lead]
    pe = [r["prism_early"] for r in lead]; ee = [r["external_only_early"] for r in lead]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(L, pe, "-o", color=ACC, lw=2, label="PRISM early detection")
    ax.plot(L, ee, "-s", color=ACC2, lw=1.6, label="External-only early detection")
    ax.set_xlabel("Indicator lead (quarters)"); ax.set_ylabel("Early-detection rate"); ax.set_xticks(L)
    ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8.5)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig6_lead.png", dpi=200, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    for name, fn in [("fig1_architecture", fig1_architecture), ("fig2_worked", fig2_worked),
                      ("fig3_detection", fig3_detection), ("fig4_calibration", fig4_calibration),
                      ("fig5_benchmark_roc", fig5_benchmark_roc), ("fig6_lead", fig6_lead)]:
        try:
            fn(); print(f"  [ok] {name}")
        except FileNotFoundError as e:
            print(f"  [skip] {name}: missing {e.filename} -- run run_all.py / extra_analyses.py first")
    print("Done. See figures/.")
