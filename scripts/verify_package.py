"""Verify notebook integrity and archived numerical summaries.

This script does not train models. Run it from any working directory with:
    python scripts/verify_package.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "archived"
TOL = 1e-12


def code_hash(notebook: dict) -> str:
    payload = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOL):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def verify_checksums() -> int:
    manifest = ROOT / "audit" / "checksums.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"File checksum mismatch: {relative}")
        checked += 1
    return checked


def verify_notebooks() -> int:
    provenance = json.loads((ROOT / "audit" / "notebook_provenance.json").read_text())
    checked = 0
    for record in provenance:
        path = ROOT / "notebooks" / "original" / record["file"]
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if any(cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"):
            raise AssertionError(f"Notebook contains outputs: {path.name}")
        if any(cell.get("execution_count") is not None for cell in notebook["cells"] if cell.get("cell_type") == "code"):
            raise AssertionError(f"Notebook contains execution counts: {path.name}")
        if code_hash(notebook) != record["code_sha256"]:
            raise AssertionError(f"Code-cell hash mismatch: {path.name}")
        checked += 1
    return checked


def verify_effect_points() -> int:
    tables = [
        ("FM_EXTERNAL_TRANSFER_PHASE0_RESULTS/PRIMARY_URFM_VS_MAE_EXTERNAL_EFFECTS.csv", "urfm_minus_mae_auc_point", "delta"),
        ("ULTRAFEDFM_PHASE0B_RESULTS/PRIMARY_ULTRAFED_VS_MAE_EXTERNAL_EFFECTS.csv", "ultrafed_minus_mae_auc_point", "delta"),
        ("FULLSTUDY_BLOCK1_LABEL_POOL_RESULTS/RQ2_USFM_MINUS_MAE_EXTERNAL_EFFECTS.csv", "point", "delta"),
        ("FULLSTUDY_FINAL_FULLFT_RESULTS/RQ4_FULLFT_USFM_VS_MAE_EFFECTS.csv", "point", "delta"),
        ("FULLSTUDY_FINAL_FULLFT_RESULTS/RQ4_ADAPTATION_CHANGE_IN_USFM_ADVANTAGE.csv", "point", "change"),
    ]
    checked = 0
    for relative, point_col, suffix in tables:
        frame = pd.read_csv(ARCHIVE / relative, float_precision="round_trip")
        seed_cols = [f"seed{seed}_{suffix}" for seed in (42, 43, 44)]
        for index, row in frame.iterrows():
            expected = float(np.mean([row[col] for col in seed_cols]))
            assert_close(float(row[point_col]), expected, f"{relative} row {index} point")
            checked += 1
    return checked


def verify_bootstrap_summary(
    effect_path: str,
    bootstrap_path: str,
    keys: list[str],
    value_col: str,
    support_col: str,
) -> int:
    effects = pd.read_csv(ARCHIVE / effect_path, float_precision="round_trip")
    bootstrap = pd.read_csv(ARCHIVE / bootstrap_path, float_precision="round_trip")
    checked = 0
    for index, row in effects.iterrows():
        mask = np.ones(len(bootstrap), dtype=bool)
        for key in keys:
            mask &= bootstrap[key].astype(str).to_numpy() == str(row[key])
        values = bootstrap.loc[mask, value_col].to_numpy(float)
        if len(values) != 3000:
            raise AssertionError(f"{bootstrap_path} row {index}: expected 3000 draws, found {len(values)}")
        assert_close(float(row["bootstrap_median"]), float(np.median(values)), f"{effect_path} row {index} median")
        assert_close(float(row["ci_lo"]), float(np.quantile(values, 0.025)), f"{effect_path} row {index} lower CI")
        assert_close(float(row["ci_hi"]), float(np.quantile(values, 0.975)), f"{effect_path} row {index} upper CI")
        assert_close(float(row[support_col]), float(np.mean(values > 0)), f"{effect_path} row {index} support")
        checked += 1
    return checked


def verify_reused_full_label_results() -> int:
    rq2 = pd.read_csv(
        ARCHIVE / "FULLSTUDY_BLOCK1_LABEL_POOL_RESULTS/RQ2_USFM_MINUS_MAE_EXTERNAL_EFFECTS.csv",
        float_precision="round_trip",
    )
    specs = [
        ("URFM", "FM_EXTERNAL_TRANSFER_PHASE0_RESULTS/PRIMARY_URFM_VS_MAE_EXTERNAL_EFFECTS.csv", "urfm_minus_mae_auc_point"),
        ("UltraFedFM", "ULTRAFEDFM_PHASE0B_RESULTS/PRIMARY_ULTRAFED_VS_MAE_EXTERNAL_EFFECTS.csv", "ultrafed_minus_mae_auc_point"),
    ]
    checked = 0
    for model, relative, point_col in specs:
        primary = pd.read_csv(ARCHIVE / relative, float_precision="round_trip")
        reused = rq2[(rq2["usfm"] == model) & (rq2["fraction"] == 1.0)]
        merged = primary.merge(reused, on=["source", "target"], validate="one_to_one")
        if len(merged) != 4:
            raise AssertionError(f"{model}: expected four full-label scenarios")
        for _, row in merged.iterrows():
            assert_close(float(row[point_col]), float(row["point"]), f"{model} reused 100% point")
            for seed in (42, 43, 44):
                assert_close(float(row[f"seed{seed}_delta_x"]), float(row[f"seed{seed}_delta_y"]), f"{model} seed {seed}")
            checked += 1
    return checked


def main() -> None:
    checksum_count = verify_checksums()
    notebook_count = verify_notebooks()
    point_count = verify_effect_points()
    bootstrap_count = 0
    bootstrap_count += verify_bootstrap_summary(
        "FM_EXTERNAL_TRANSFER_PHASE0_RESULTS/PRIMARY_URFM_VS_MAE_EXTERNAL_EFFECTS.csv",
        "FM_EXTERNAL_TRANSFER_PHASE0_RESULTS/PRIMARY_URFM_VS_MAE_BOOTSTRAP_3000.csv",
        ["source", "target"], "urfm_minus_mae_auc", "p_urfm_better",
    )
    bootstrap_count += verify_bootstrap_summary(
        "ULTRAFEDFM_PHASE0B_RESULTS/PRIMARY_ULTRAFED_VS_MAE_EXTERNAL_EFFECTS.csv",
        "ULTRAFEDFM_PHASE0B_RESULTS/PRIMARY_ULTRAFED_VS_MAE_BOOTSTRAP_3000.csv",
        ["source", "target"], "ultrafed_minus_mae_auc", "p_ultrafed_better",
    )
    bootstrap_count += verify_bootstrap_summary(
        "FULLSTUDY_BLOCK1_LABEL_POOL_RESULTS/RQ2_USFM_MINUS_MAE_EXTERNAL_EFFECTS.csv",
        "FULLSTUDY_BLOCK1_LABEL_POOL_RESULTS/RQ2_BOOTSTRAP_3000.csv",
        ["usfm", "fraction", "source", "target"], "fm_minus_mae_auc", "p_a_better",
    )
    bootstrap_count += verify_bootstrap_summary(
        "FULLSTUDY_FINAL_FULLFT_RESULTS/RQ4_FULLFT_USFM_VS_MAE_EFFECTS.csv",
        "FULLSTUDY_FINAL_FULLFT_RESULTS/RQ4_FULLFT_USFM_VS_MAE_BOOTSTRAP.csv",
        ["usfm", "source", "target"], "ft_fm_minus_mae_auc", "p_a_better",
    )
    reused_count = verify_reused_full_label_results()
    report = {
        "status": "PASS",
        "file_checksums_verified": checksum_count,
        "notebooks_verified": notebook_count,
        "effect_rows_verified": point_count,
        "bootstrap_summaries_verified": bootstrap_count,
        "reused_full_label_scenarios_verified": reused_count,
    }
    output = ROOT / "audit" / "verification_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
