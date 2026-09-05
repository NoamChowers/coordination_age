#!/usr/bin/env python3
"""Regenerate Figure 1 from saved predictions without fitting any models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILENAME = "figure_1_train_loocv_model_comparison.png"


def create_model_comparison(comparison_dir=None, output_dir=None):
    comparison_dir = Path(comparison_dir or ROOT / "outputs/model_comparison")
    output_dir = Path(output_dir or ROOT / "outputs/visualizations")
    predictions = pd.read_csv(comparison_dir / "comparison_oof_predictions.csv")
    metrics = pd.read_csv(comparison_dir / "comparison_metrics.csv")
    if predictions.duplicated(["procedure", "SubjectNumber"]).any():
        raise ValueError("Each participant must have one prediction per procedure.")
    if not metrics.procedure.is_unique or set(predictions.procedure) != set(metrics.procedure):
        raise ValueError("Metric and prediction procedures do not match.")
    errors = predictions.assign(absolute_error_years=(
        predictions.chronological_age_years - predictions.oof_prediction_years
    ).abs())
    order = metrics.sort_values("rmse_rank").procedure.tolist()
    lookup = metrics.set_index("procedure")
    values, labels = [], []
    for name in order:
        group = errors.loc[errors.procedure.eq(name)]
        if len(group) != 115 or not np.isfinite(group.absolute_error_years).all():
            raise ValueError(f"Expected 115 finite participant errors for {name}.")
        rmse = float(np.sqrt(np.mean(group.absolute_error_years ** 2)))
        if not np.isclose(rmse, lookup.loc[name, "rmse_years"], atol=1e-10, rtol=1e-10):
            raise ValueError(f"Saved RMSE disagrees with predictions for {name}.")
        labels.append(f"{name}  (RMSE {rmse:.3f})")
        values.append(group.absolute_error_years.to_numpy())
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 200}):
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        boxes = ax.boxplot(
            values[::-1], tick_labels=labels[::-1], vert=False,
            patch_artist=True, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
            medianprops={"color": "black"}, flierprops={"marker": ".", "alpha": 0.45},
        )
        for patch, name in zip(boxes["boxes"], order[::-1]):
            patch.set_facecolor("#0072B2" if name == "RBF Kernel Ridge" else "#A7A9AC")
            patch.set_alpha(0.85)
        ax.set_xlabel("Absolute outer-LOOCV prediction error (years)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / FILENAME
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    entry = {
        "filename": FILENAME,
        "description": "Final candidate comparison: absolute outer-LOOCV errors, ordered by RMSE.",
        "sources": "; ".join(
            str((comparison_dir / name).relative_to(ROOT))
            if (comparison_dir / name).is_relative_to(ROOT) else str(comparison_dir / name)
            for name in ("comparison_oof_predictions.csv", "comparison_metrics.csv")),
    }
    # Keep unrelated figure entries, including the user's cohort overview.
    manifest_path = output_dir / "figure_manifest.csv"
    existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=entry)
    existing = existing.loc[existing.filename.ne(FILENAME)]
    pd.concat([existing, pd.DataFrame([entry])], ignore_index=True).to_csv(manifest_path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=ROOT / "outputs/model_comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/visualizations")
    args = parser.parse_args()
    print(create_model_comparison(args.comparison_dir, args.output_dir))


if __name__ == "__main__":
    main()
