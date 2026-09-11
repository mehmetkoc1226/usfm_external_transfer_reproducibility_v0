# Adaptation-Dependent External Transfer of Ultrasound Foundation Models

This repository accompanies the manuscript **"Adaptation-Dependent External
Transfer of Ultrasound Foundation Models in Breast Ultrasound."** It preserves
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
python scripts/recompute_rq4_adaptation.py
```

`verify_package.py` checks output removal and code-cell hashes for all four
notebooks, repository file checksums, seed-balanced point calculations, archived
bootstrap summaries, and the exact reuse of the RQ1 full-label results in RQ2.
The RQ4 script reconstructs
the adaptation-change contrasts from target-unit scores and validates a sample
of its optimized AUROC calculations against scikit-learn.

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
