#!/usr/bin/env python3
"""Create a compact overview of the modeling cohort and historical split.

The figure combines the within-split chronological-age distributions with
predictor-cell missingness summarized by measurement group. Raw summary tables
are saved beside the PNG so every plotted value is auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_COLOR = "#0072B2"
TEST_COLOR = "#D55E00"
MISSING_COLOR = "#6B7280"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-csv", type=Path, default=root / "Data" / "X.csv")
    parser.add_argument("--y-csv", type=Path, default=root / "Data" / "y.csv")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=root / "outputs" / "train115_jkplus" / "recovered_train_test_split.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "visualizations"
    )
    return parser.parse_args()


def predictor_groups(columns: list[str]) -> dict[str, list[str]]:
    fixed = {
        "Demographic (sex)": ["Sex_0Female_1Male"],
        "Box and Blocks": ["boxAndBlocks"],
        "Jebsen–Taylor": [column for column in columns if column.startswith("jebsen_")],
        "Finger pressing": ["MVC", "deltaV", "V_UCM", "V_ORT"],
        "Reach-to-grasp": [
            "peakvel",
            "movementtime",
            "maxaperturePercent",
            "meandistance",
        ],
    }
    assigned = {column for group in fixed.values() for column in group}
    fixed["Object lifting"] = [column for column in columns if column not in assigned]

    flattened = [column for group in fixed.values() for column in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(columns):
        raise ValueError("Predictor groups do not partition the columns of X.csv.")
    return fixed


def age_summary(y: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(y.columns) != ["AgeInYears"]:
        raise ValueError("y.csv must contain exactly the AgeInYears column.")

    positions = split["production_row_position"].astype(int)
    expected = np.arange(len(y))
    if len(split) != len(y) or not np.array_equal(np.sort(positions), expected):
        raise ValueError("The split file must cover every production row exactly once.")

    cohort = split.copy()
    cohort["chronological_age_years"] = y.iloc[positions.to_numpy()]["AgeInYears"].to_numpy()
    cohort["partition"] = pd.Categorical(
        cohort["partition"], categories=["train", "test"], ordered=True
    )

    edges = np.arange(4, 20, 1, dtype=float)
    labels = [f"[{int(left)}, {int(right)})" for left, right in zip(edges[:-1], edges[1:])]
    cohort["age_bin"] = pd.cut(
        cohort["chronological_age_years"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    if cohort["age_bin"].isna().any():
        bad = cohort.loc[cohort["age_bin"].isna(), "chronological_age_years"].tolist()
        raise ValueError(f"Ages fall outside the supported [4, 19) plotting range: {bad}")

    counts = (
        cohort.groupby(["age_bin", "partition"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=labels, columns=["train", "test"], fill_value=0)
    )
    counts.index.name = "age_bin_years"
    counts = counts.reset_index()
    counts["total"] = counts["train"] + counts["test"]
    return cohort.sort_values(["partition", "chronological_age_years"]), counts


def missingness_summary(x: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group, columns in predictor_groups(list(x.columns)).items():
        missing = int(x[columns].isna().sum().sum())
        cells = int(x.shape[0] * len(columns))
        records.append(
            {
                "predictor_group": group,
                "n_predictors": len(columns),
                "missing_cells": missing,
                "total_cells": cells,
                "missing_percent": 100.0 * missing / cells,
            }
        )
    return pd.DataFrame(records)


def make_figure(
    cohort: pd.DataFrame,
    age_counts: pd.DataFrame,
    missingness: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    fig, (age_ax, missing_ax) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    x_positions = np.arange(len(age_counts))
    width = 0.39
    split_sizes = cohort["partition"].value_counts()
    for offset, partition, label, color in (
        (-width / 2, "train", f"Train (n={split_sizes['train']})", TRAIN_COLOR),
        (width / 2, "test", f"Test (n={split_sizes['test']})", TEST_COLOR),
    ):
        raw_counts = age_counts[partition].to_numpy()
        percentages = 100.0 * raw_counts / split_sizes[partition]
        bars = age_ax.bar(
            x_positions + offset,
            percentages,
            width=width,
            label=label,
            color=color,
            alpha=0.88,
        )
        age_ax.bar_label(bars, labels=[str(value) if value else "" for value in raw_counts], padding=2, fontsize=8)

    age_ax.set_title("A. Chronological-age distribution", loc="left", fontweight="bold")
    age_ax.set_xlabel("Age bin (years); labels above bars are participant counts")
    age_ax.set_ylabel("Participants within split (%)")
    age_ax.set_xticks(x_positions)
    age_ax.set_xticklabels([str(age) for age in range(4, 19)])
    age_ax.set_ylim(0, max(age_ax.get_ylim()[1], 16.5))
    age_ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    age_ax.set_axisbelow(True)
    age_ax.legend(frameon=False, ncols=2, loc="upper right")

    ordered = missingness.sort_values("missing_percent", ascending=True)
    bars = missing_ax.barh(
        ordered["predictor_group"], ordered["missing_percent"], color=MISSING_COLOR, alpha=0.9
    )
    missing_ax.set_title("B. Missing predictor cells", loc="left", fontweight="bold")
    missing_ax.set_xlabel("Missing cells within predictor group (%)")
    missing_ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    missing_ax.set_axisbelow(True)
    maximum = max(float(ordered["missing_percent"].max()), 1.0)
    missing_ax.set_xlim(0, maximum * 1.42)
    labels = [
        f"{row.missing_cells}/{row.total_cells} ({row.missing_percent:.1f}%)"
        for row in ordered.itertuples()
    ]
    missing_ax.bar_label(bars, labels=labels, padding=4, fontsize=8)

    total_missing = int(missingness["missing_cells"].sum())
    total_cells = int(missingness["total_cells"].sum())
    fig.suptitle(
        "Modeling cohort and historical train–test split",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.965,
        f"N={len(cohort)} participants; {total_missing}/{total_cells} predictor cells missing "
        f"({100.0 * total_missing / total_cells:.1f}%) before fold-specific imputation",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3.0)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def update_manifest(output_dir: Path, filename: str) -> None:
    manifest_path = output_dir / "figure_manifest.csv"
    columns = ["filename", "description", "sources"]
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=columns)
    manifest = manifest.loc[manifest["filename"] != filename].copy()
    record = pd.DataFrame(
        [
            {
                "filename": filename,
                "description": "Chronological-age distributions in the historical train/test split and predictor-group missingness.",
                "sources": "X.csv; y.csv; recovered_train_test_split.csv",
            }
        ]
    )
    pd.concat([record, manifest], ignore_index=True)[columns].to_csv(manifest_path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.x_csv)
    y = pd.read_csv(args.y_csv)
    split = pd.read_csv(args.split_csv)
    if len(x) != len(y):
        raise ValueError("X.csv and y.csv must have the same number of rows.")

    cohort, age_counts = age_summary(y, split)
    missingness = missingness_summary(x)

    figure_name = "dataset_split_overview.png"
    age_counts.to_csv(args.output_dir / "dataset_split_age_bin_counts.csv", index=False)
    missingness.to_csv(args.output_dir / "dataset_missingness_by_group.csv", index=False)
    make_figure(cohort, age_counts, missingness, args.output_dir / figure_name)
    update_manifest(args.output_dir, figure_name)

    print(args.output_dir / figure_name)


if __name__ == "__main__":
    main()
