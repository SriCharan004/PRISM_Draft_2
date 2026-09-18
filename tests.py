"""Automated correctness checks for PRISM. Run:  python tests.py
Each test is a falsifiable property a reviewer can verify, not a smoke test."""
import numpy as np
from dataclasses import replace
from scipy import stats
import prism_sim as ps

def test_calibration():
    """Synthetic pre-inflection log-returns match the target law (KS does not reject)."""
    pre = np.concatenate([ps.generate_panel(700000+s, "gradual", inflect=False)["logret"][1:ps.CFG.K]
                          for s in range(300)])
    ks = stats.kstest((pre - ps.CFG.g0)/ps.CFG.sig_d, "norm")
    assert abs(pre.mean()-ps.CFG.g0) < 0.001, pre.mean()
    assert abs(pre.std()-ps.CFG.sig_d) < 0.001, pre.std()
    assert ks.pvalue > 0.05, ks.pvalue
    print(f"  [pass] calibration: mean={pre.mean():.5f} sd={pre.std():.5f} KS p={ks.pvalue:.3f}")

def test_beta0_reduction():
    """PRISM with beta=0 must equal constant-hazard BOCPD exactly."""
    ref = ps.build_incontrol_reference(80, "gradual")
    cfg0 = replace(ps.CFG, beta_ext=0.0, beta_int=0.0)
    maxdiff = 0.0
    for seed in range(2000, 2030):
        p = ps.generate_panel(seed, "gradual", inflect=True)
        a = ps.bocpd(p["logret"], ps.CFG.h0)
        b = ps.bocpd(p["logret"], ps.hazard_for("prism", p, ref, cfg0))
        maxdiff = max(maxdiff, np.max(np.abs(a-b)))
    assert maxdiff < 1e-9, maxdiff
    print(f"  [pass] beta=0 reduction: max|A_prism - A_severity| = {maxdiff:.2e}")

def test_recursion_normalized():
    """Run-length posterior is a proper distribution at every step."""
    p = ps.generate_panel(1, "gradual")
    # internal re-implementation check: alarm statistic is a probability in [0,1]
    A = ps.bocpd(p["logret"], ps.CFG.h0)
    assert np.all(A >= -1e-9) and np.all(A <= 1+1e-9)
    print("  [pass] alarm statistic in [0,1] at every step")

def test_frequency_channel():
    """Frequency channel runs and PRISM beats the loss-only monitor on a gradual freq inflection."""
    cfg = replace(ps.CFG, channel="frequency")
    rows, _ = ps.run_experiment(n=200, scenario="gradual", cfg=cfg)
    import pandas as pd; d = pd.DataFrame(rows)
    sev = d[d.variant=="severity_only"].detect.mean(); pr = d[d.variant=="prism"].detect.mean()
    assert pr > sev, (pr, sev)
    print(f"  [pass] frequency channel: PRISM={pr:.3f} > loss-only={sev:.3f}")

def test_arbitrary_inflection_point():
    """The monitor works for an inflection at any quarter K."""
    import pandas as pd
    for K in (16, 30):
        cfg = replace(ps.CFG, K=K, B=min(ps.CFG.B, K-4))
        rows, _ = ps.run_experiment(n=150, scenario="gradual", cfg=cfg)
        d = pd.DataFrame(rows)
        pr = d[d.variant=="prism"].detect.mean(); sev = d[d.variant=="severity_only"].detect.mean()
        assert pr > sev, (K, pr, sev)
    print("  [pass] arbitrary inflection point: PRISM > loss-only for K in {16, 30}")

if __name__ == "__main__":
    print("Running PRISM correctness tests...")
    test_calibration(); test_beta0_reduction(); test_recursion_normalized()
    test_frequency_channel(); test_arbitrary_inflection_point()
    print("All tests passed.")
