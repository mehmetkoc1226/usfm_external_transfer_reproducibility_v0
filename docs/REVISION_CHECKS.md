# Revision-stage checks

These analyses were added during scientific review after the original RQ1–RQ4
sequence. All original notebooks, target-unit scores and primary numerical
results are retained. They are exploratory sensitivity checks, not new
preregistered research questions or optimized model-performance estimates.

## Archived results and training histories

`submission_additional_checks.py` recalculates 72 external model–seed–protocol–
scenario evaluations, each with AUROC and average precision (144 scalars). It
summarizes the three two-seed means formed by omitting each completed seed.
These values do not form confidence intervals or a population jackknife.

URFM's LP and FT margins remain positive under every omission. UltraFedFM's
positive-LP/negative-FT direction survives all omissions in three scenarios.
For BrEaST→BUSI, omitting seed 44 yields FT +0.009 and LP +0.028, whereas the
original three-seed FT mean is −0.016. The primary four-scenario reversal
describes three-seed means rather than every individual run.

All 18 primary LP histories contain 30 epochs. Eleven best validation
checkpoints occur at epoch 30 and fifteen within epochs 28–30. This motivates
an optimization sensitivity check but does not prove inadequate convergence.

## Frozen logistic-head control

The same CLS features and archived source splits are used. BUS-BRA image
features are averaged by patient. Source training contains 851 BUS-BRA patients
or 201 BrEaST cases; validation contains 213 patients or 51 cases.

StandardScaler is fitted on source training only. LogisticRegression uses
normalized inverse-frequency source-training class weights, L2 regularization,
L-BFGS with tolerance 1e-8 and at most 5,000 iterations. The common candidate
grid is C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]. Maximum source-validation
AUROC selects C; exact ties prefer the smallest C. One deterministic fit is
selected per model–source pair. Repeating the same fit three times would not
constitute three independent training seeds.

The protocol and fitted source parameters were saved before external
evaluation. `selection_seal.json` records their hashes. All 42 candidate fits
converged without warnings, with at most 737 iterations and maximum objective
gradient infinity norm below 9.6e-8. Source-selected C values were:

| Source | URFM | UltraFedFM | MAE |
| --- | ---: | ---: | ---: |
| BUS-BRA | 0.01 | 0.01 | 0.01 |
| BrEaST | 0.001 | 0.1 | 1 |

| External scenario | URFM minus MAE | UltraFedFM minus MAE |
| --- | ---: | ---: |
| BUS-BRA→BrEaST | +0.141 | −0.070 |
| BUS-BRA→BUSI | +0.098 | −0.041 |
| BrEaST→BUS-BRA | +0.077 | +0.005 |
| BrEaST→BUSI | +0.160 | −0.125 |

The four-scenario median margins are +0.120 for URFM and −0.056 for UltraFedFM.
The control changes feature scaling, regularization selection and optimization
together. It does not identify which change causes the difference. The same
small source validation sets and fixed source splits remain limitations.
No alternate full-FT recipe was evaluated. Control values are descriptive
point estimates; primary three-seed conditional target-bootstrap intervals
must not be attached to them.

## Feature row correspondence

BUSI features were extracted in the original notebook after concatenating
manifest train rows followed by val rows. The public manifest retains lexical
image order. The control reconstructs the cached order using `split` before
attaching identities and labels. This does not use target scores to select a
model or change source fits. `feature_alignment_correction.json` documents
the correction to intermediate revision estimates, not any primary study result.

Six original seed-42 LP heads, covering both sources and all models, were
reapplied to all nine caches. All 12 external AUROC and AP pairs exactly match
the archived LP score table; the maximum absolute logit-margin discrepancy is
1.2e-6. Small float32 differences arise from BLAS order and the equivalent
averaging of patient features or image logits. The comparison hashes and
results are in `results/submission_checks/feature_cache_head_correspondence.*`.

For a separate correspondence check, provide the original small checkpoints:

```bash
python scripts/verify_feature_cache_heads.py --features /path/to/caches --heads /path/to/heads
```

The restricted checkpoint reader accepts only contiguous float32 storage and
tensor reconstruction plus OrderedDict. It rejects arbitrary pickle globals.
Feature arrays and pretrained encoders are not included in this release.
