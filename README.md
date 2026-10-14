# External Breast Lesion Classification with Ultrasound Foundation Models

This repository accompanies the manuscript **"External Breast Lesion
Classification with Ultrasound Foundation Models: Frozen Probing versus Shared
Fine-Tuning."** It preserves
the experiment notebooks, fixed split manifests, target-unit score tables,
bootstrap arrays, and audit utilities used for the reported analyses.

## Repository scope

- `notebooks/original/`: the four output-cleared Google Colab notebooks used for
  RQ1-RQ4. Their code-cell hashes are recorded in `audit/notebook_provenance.json`.
- `data/manifests/`: de-identified analysis-unit manifests with machine-specific
  image paths removed.
- `data/splits/`: fixed source splits and nested label-efficiency subsets.
- `results/archived/`: the original CSV/JSON result packages produced by the
  study notebooks.
- `results/canonical/`: independently recomputed RQ4 adaptation-change outputs
  retained for manuscript auditing.
- `scripts/verify_package.py`: fast numerical and notebook-integrity checks.
- `scripts/recompute_rq4_adaptation.py`: deterministic RQ4 reanalysis from the
  archived target-unit score tables, without model training.
- `figures/data/`: numeric inputs used by the manuscript figures.
- `results/submission_checks/`: revision-stage metric recomputation, descriptive
  seed omission, primary LP histories and feature-cache correspondence checks.
- `results/lp_convergence_control/`: a post-hoc, source-selected, converged
  logistic-head control; it does not replace any primary result.
- `scripts/lp_convergence_control.py`: separate source selection and evaluation
  stages for that control.
- `scripts/verify_feature_cache_heads.py`: optional correspondence check using
  the original small LP heads and feature caches, supplied separately.

Raw ultrasound images and third-party pretrained checkpoints are not
redistributed. Obtain them from their official sources and follow
`data/README.md` and `docs/REPRODUCIBILITY.md`.

## Quick audit

The audit path does not require a GPU or the raw images.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python scripts/verify_package.py
python scripts/verify_revision_checks.py
python scripts/recompute_rq4_adaptation.py
```

`verify_package.py` checks output removal and code-cell hashes for all four
notebooks, repository file checksums, seed-balanced point calculations, archived
bootstrap summaries, and the exact reuse of the RQ1 full-label results in RQ2.
The RQ4 script reconstructs
the adaptation-change contrasts from target-unit scores and validates a sample
of its optimized AUROC calculations against scikit-learn.

## Revision-stage sensitivity analysis

The primary 30-epoch frozen probe remains unchanged. An additional deterministic
logistic classifier uses source-training-only feature scaling, L2 regularization
selected on source validation, and converged L-BFGS optimization. URFM has positive
MAE-relative point margins in four external scenarios; UltraFedFM in one. These
are protocol-specific descriptive results. The control changes scaling,
regularization selection and optimization jointly and cannot isolate the effect
of the primary training budget.

If the original 768-dimensional feature arrays and completion masks are available:

```bash
python scripts/lp_convergence_control.py --features /path/to/caches --stage select
python scripts/lp_convergence_control.py --features /path/to/caches --stage evaluate
```

Keep the source selection seal intact between stages. BUSI caches follow the
original notebook's train-then-validation order; the script reconstructs that
order from `split`, rather than using the public manifest's lexical image order.
Six archived seed-42 heads reproduced all 12 external AUROC/AP pairs after this
alignment. The cache/head hashes and numerical comparison are included; neither
features nor checkpoints are redistributed. See `docs/REVISION_CHECKS.md`.

## Training reproduction

The original training environment was Google Colab with Google Drive mounted at
`/content/drive/MyDrive/SPATIOSPECTRAL_PEFT_PILOT`. Install the training
dependencies, place datasets and checkpoints as documented, and execute the
notebooks in the order listed in `notebooks/README.md`. The notebooks retain
their original path assumptions to preserve the audited code; a different root
requires editing the `ROOT` assignment near the start of each notebook.

Training reproduction requires authorized access to all datasets, the official
URFM and UltraFedFM checkpoints, and GPU resources. It is therefore distinct
from the CPU-only audit above.

## Citation

Please cite the associated article. Machine-readable provisional citation
metadata are provided in `CITATION.cff`; update the journal and DOI fields after
publication.

## License

Repository-authored code is released under the MIT License. Dataset images,
dataset annotations, pretrained checkpoints, and third-party code retain their
original licenses and terms.
