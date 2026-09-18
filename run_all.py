"""Regenerate every result, table, and figure in the PRISM paper from fixed seeds.
Usage:  python run_all.py
Outputs: results/ (CSV + JSON), figures/ (PNG), and a SHA-256 results hash for verification.
All data is synthetic; no real claims data is used."""
import os, json, hashlib
import numpy as np, pandas as pd
from dataclasses import replace
from scipy import stats
import prism_sim as ps

os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

def wilson(k, n, z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)

def main():
    summary = {}
    frames = []
    # ---- primary Monte Carlo: severity + frequency, abrupt + gradual ----
    for channel in ("severity", "frequency"):
        cfg = replace(ps.CFG, channel=channel)
        summary[channel] = {}
        for sc in ("abrupt", "gradual"):
            rows, _ = ps.run_experiment(n=1000, scenario=sc, cfg=cfg)
            df = pd.DataFrame(rows); frames.append(df)
            summary[channel][sc] = {}
            for v in ps.VARIANTS:
                s = df[df.variant==v]; det=int(s.detect.sum()); n=len(s); lo,hi=wilson(det,n)
                summary[channel][sc][v] = dict(detect=det/n, lo=lo, hi=hi,
                    early=float(s.early.mean()), false=float(s.false.mean()),
                    lead=float(s.lead.dropna().mean()) if det else float("nan"))
    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv("results/prism_results.csv", index=False)

    # ---- calibration ----
    pre = np.concatenate([ps.generate_panel(700000+s,"gradual",inflect=False)["logret"][1:ps.CFG.K] for s in range(300)])
    ks = stats.kstest((pre-ps.CFG.g0)/ps.CFG.sig_d, "norm")
    summary["calibration"] = dict(mean=float(pre.mean()), sd=float(pre.std()),
                                  ks_p=float(ks.pvalue), ks_stat=float(ks.statistic))
    json.dump(summary, open("results/summary.json","w"), indent=2)

    # ---- reproducibility hash over the canonical results table ----
    h = hashlib.sha256(pd.util.hash_pandas_object(alldf, index=True).values.tobytes()).hexdigest()
    open("results/RESULTS_HASH.txt","w").write(h + "\n")
    print("Severity gradual  PRISM detection:", round(summary["severity"]["gradual"]["prism"]["detect"],3))
    print("Frequency gradual PRISM detection:", round(summary["frequency"]["gradual"]["prism"]["detect"],3))
    print("Calibration KS p-value:", round(summary["calibration"]["ks_p"],3))
    print("SHA-256 results hash:", h)

    # ---- supplementary analyses + figures (so figures/ is never empty) ----
    import extra_analyses as ea
    for name in ("tuned", "naive", "arl", "seeds", "frontier", "confidence", "s3", "lead", "snr", "roc", "scenarios", "outlier_recovery"):
        print(f"[extra] {name} ...")
        ea.ALL[name]()
    import make_figures as mf
    for name, fn in [("fig1_architecture", mf.fig1_architecture), ("fig2_worked", mf.fig2_worked),
                      ("fig3_detection", mf.fig3_detection), ("fig4_calibration", mf.fig4_calibration),
                      ("fig5_benchmark_roc", mf.fig5_benchmark_roc), ("fig6_lead", mf.fig6_lead)]:
        fn(); print(f"[figure] {name}")
    print("Done. See results/ and figures/.")

if __name__ == "__main__":
    main()
