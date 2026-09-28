# PRISM on ActSim data

This folder holds the ActSim part of the PRISM project. Here the loss data is **not**
drawn from PRISM's own model — it is simulated claim by claim with **CAS ActSim**
(github.com/casact/actsim) and then added up into a quarterly severity series, the way
an actuary builds a trend exhibit from a claims listing. The point is to check that
PRISM still works when the data comes from somewhere independent of the detector.

All data here is synthetic. No real claims are used.

## What's inside

- **`code/`** — the scripts that generate the ActSim panels and run the tests.
  - `prism_noext.py` — builds the ActSim book and turns claims into a quarterly severity series.
  - `prism_robust.py` — PRISM with the reversion (Hampel) filter, and the head-to-head vs plain BOCPD.
  - `prism_curve.py` — tests gradual changes of different steepness.
  - `prism_actsim_ext.py` — the four-variant test (severity / external / internal / PRISM) on ActSim.
  - `diag_bocpd.py` — shows why plain BOCPD is blind to gradual turns.
  - `gen_workbook_panel.py` — makes the single worked example behind the Excel workbook.
  - `reproduce_numbers.py` — runs everything and prints every number in the report from fixed seeds.
- **`results/`** — the output files (JSON summaries, per-run CSVs, figures) and the workbook panel data.
- **`documents/`** — the write-ups: the ActSim report, the two annexures, and the Excel workbook.

Note: the scripts use `prism_sim.py` (the detector) from the main repo. Keep them next to
that file, or copy `prism_sim.py` into `code/`, before running.

## How to run

```
pip install numpy pandas scipy
pip install git+https://github.com/casact/actsim
python code/reproduce_numbers.py --quick   # quick check (numbers are approximate)
python code/reproduce_numbers.py           # full run, matches the report (~20-30 min)
```

## What the results show (short version)

- **Abrupt jumps:** plain BOCPD already catches them, so PRISM adds nothing.
- **Gradual / subtle turns:** plain BOCPD is completely blind (0%); PRISM's internal
  signals catch them. This is the case the method exists for.
- **Flat data:** the same sensitivity costs about 6% false alarms.

## One thing to keep in mind

ActSim gives a single severity series, not a set of rating predictors. So the "internal
predictors" here are a reduced, two-signal version (drift off a fixed baseline, and
residual correlation), not the full three-signal pricing-model monitor the paper
describes. It works, but it is a proxy — testing the full version would need a claims
dataset with rating variables.
