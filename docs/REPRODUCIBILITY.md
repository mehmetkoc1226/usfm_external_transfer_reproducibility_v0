# Reproducibility and audit boundaries

## What can be audited without retraining

The released target-unit score tables and full bootstrap arrays permit direct
verification of the manuscript's AUROC contrasts, seed-balanced effects,
percentile intervals, directional bootstrap support, and RQ4 adaptation-change
identity. Run `python scripts/verify_package.py` for the fast checks and
`python scripts/recompute_rq4_adaptation.py` for the independent RQ4 rebuild.

All primary conclusions were taken from archived CSV/JSON packages rather than
remembered console output. The release also includes training-status histories,
fixed source splits, nested RQ2 subsets, experiment configurations, and model
loading audits.

## What requires retraining

Reproducing feature extraction and optimization requires the raw BUS-BRA,
BrEaST, and BUSI images, official URFM and UltraFedFM checkpoints, the public
ImageNet-MAE weights loaded through `timm`, and a compatible GPU environment.
Raw medical images and third-party model weights are not included because their
licenses and access conditions are controlled by their original providers.

The output-cleared notebooks preserve the original Colab root and block-to-block
artifact layout. They are released as an audit trail and as executable research
code once the documented external assets are supplied. They are not presented
as a containerized, hardware-independent reproduction workflow.

## Statistical scope

The paired bootstrap represents target-sampling uncertainty conditional on the
completed runs. It does not capture uncertainty from drawing a different source
sample, choosing different checkpoints, or repeating optimization beyond seeds
42-44. Scenario intervals are unadjusted and should not be interpreted as
simultaneous confidence intervals.

Private components of ultrasound-model pretraining corpora cannot be inspected
independently. The release therefore documents checkpoint provenance but cannot
prove complete pretraining/evaluation non-overlap.
