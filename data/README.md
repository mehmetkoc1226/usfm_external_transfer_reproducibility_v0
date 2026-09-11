# Data and checkpoint setup

The repository does not redistribute raw medical images or third-party model
weights. Download each dataset from its official distribution and comply with
its terms of use.

The study used these lesion-only analysis cohorts:

| Dataset | Analysis unit | Benign | Malignant | Total |
| --- | ---: | ---: | ---: | ---: |
| BUS-BRA | patient | 721 | 343 | 1,064 |
| BrEaST | case | 154 | 98 | 252 |
| BUSI | image | 437 | 210 | 647 |

Normal BUSI images were excluded. BUS-BRA images were aggregated within patient
for frozen source fitting and external evaluation. BrEaST was analyzed by case;
BUSI was analyzed by image because reliable patient identifiers were not
available.

The original notebooks expect the following Colab/Drive root:

```text
/content/drive/MyDrive/SPATIOSPECTRAL_PEFT_PILOT/
```

Within it, place the extracted datasets under `data/` and the official weights
under:

```text
external_models/URFM/mae_vit_base_patch16_dec768d8b_all_biomedclip_1199.pth
external_models/UltraFedFM/official_pretrained_ultrafedfm_checkpoint.pth
```

URFM is resolved from the official `QingboKang/URFM` release when access permits.
The ImageNet-MAE comparator is loaded by `timm` as
`vit_base_patch16_224.mae`. The UltraFedFM checkpoint used in the experiment has
SHA-256:

```text
837fe48d140cf62fa2a9efbf056939d80d1714311e3c1b5fa26da7d65018e549
```

Files in `manifests/` preserve row identity, labels, and analysis grouping but
omit local paths. Files in `splits/` preserve the fixed source assignments and
nested RQ2 subsets.
