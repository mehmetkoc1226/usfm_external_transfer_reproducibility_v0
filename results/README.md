# Result archives

`archived/` contains the original study outputs grouped by experimental block.
These files are the evidentiary source for the manuscript tables and figures.
They include configurations, training histories/status records, target-unit
scores, point-effect tables, and the complete 3,000-draw bootstrap arrays.

`canonical/` contains an independent reconstruction of the RQ4 change in USFM
advantage. The reconstruction uses archived full-fine-tuning and linear-probe
target-unit scores. Because it uses a clean implementation and the same planned
random-seed sequence, small Monte Carlo differences from the original interval
endpoints are expected; all point estimates and interval-sign classifications
are preserved. See `old_vs_recomputed.csv`.

Running `scripts/recompute_rq4_adaptation.py` writes a fresh copy to
`recomputed/` without modifying either archived or canonical evidence.
