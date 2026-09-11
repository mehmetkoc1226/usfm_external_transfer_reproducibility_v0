# Notebook execution order

The notebooks are preserved in the order in which the study blocks were run:

1. `FM_External_Transfer_Phase0_v1_1.ipynb`: URFM, ImageNet-MAE, and contextual
   DINOv2 frozen-feature evaluation; produces the fixed source splits and the
   primary URFM-versus-MAE RQ1 archive.
2. `UltraFedFM_Phase0B_Replication_v1_0.ipynb`: adds the second ultrasound
   foundation model and completes the second RQ1 comparison.
3. `FullStudy_Block1_LabelEfficiency_Pooling_v1_0.ipynb`: RQ2 label fractions
   and RQ3 common mean-patch pooling sensitivity.
4. `FullStudy_Final_Controlled_FullFT_v1_1.ipynb`: RQ4 common full fine-tuning,
   target-unit score export, and paired-bootstrap analyses.

All notebook outputs and execution counts were removed for repository release;
the code cells were not changed. Their hashes are in
`../audit/notebook_provenance.json`.

These are Colab notebooks, not four independent scripts. Later notebooks read
artifacts written by earlier blocks under the shared `ROOT`. Run each notebook
top to bottom and do not change the fixed seeds, splits, training budgets, or
decision rules when reproducing the reported analysis.
