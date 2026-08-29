"""Construct the frozen 32-feature production matrix and age target."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ID_COLUMN = "SubjectNumber"
AGE_MONTHS_COLUMN = "AgeInMonths"
TARGET_COLUMN = "AgeInYears"

FINAL_FEATURE_COLUMNS = [
    "Sex_0Female_1Male",
    "boxAndBlocks",
    "jebsen_PageTurning",
    "jebsen_SmallObjects",
    "jebsen_Beans",
    "jebsen_Checkers",
    "jebsen_LightCans",
    "jebsen_HeavyCans",
    "MVC",
    "deltaV",
    "V_UCM",
    "V_ORT",
    "peakvel",
    "movementtime",
    "maxaperturePercent",
    "meandistance",
    "peakGripForceLight",
    "peakGripForceHeavy",
    "peakLoadForceLight",
    "peakLoadForceHeavy",
    "peakGripForceRateLight",
    "peakGripForceRateHeavy",
    "peakLoadForceRateLight",
    "peakLoadForceRateHeavy",
    "T1T3Light",
    "T1T3Heavy",
    "T1T5Light",
    "T1T5Heavy",
    "liftMeanDifferencesLight",
    "liftMeanDifferencesHeavy",
    "numPeaksLight",
    "numPeaksHeavy",
]

INVALID_NEGATIVE_TIMING_COLUMNS = [
    "T1T3Light",
    "T1T3Heavy",
    "T1T5Light",
    "T1T5Heavy",
]


def build_model_inputs(
    aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return model-ready X and y from the finalized aggregate table."""

    required = [ID_COLUMN, AGE_MONTHS_COLUMN, *FINAL_FEATURE_COLUMNS]
    missing = [column for column in required if column not in aggregate.columns]
    if missing:
        raise ValueError(f"Aggregate data is missing required columns: {missing}")
    if aggregate[ID_COLUMN].isna().any() or not aggregate[ID_COLUMN].is_unique:
        raise ValueError("SubjectNumber must be unique and nonmissing.")

    X = aggregate[FINAL_FEATURE_COLUMNS].copy()
    X[INVALID_NEGATIVE_TIMING_COLUMNS] = X[
        INVALID_NEGATIVE_TIMING_COLUMNS
    ].mask(X[INVALID_NEGATIVE_TIMING_COLUMNS] < 0, np.nan)
    y = pd.DataFrame(
        {TARGET_COLUMN: pd.to_numeric(aggregate[AGE_MONTHS_COLUMN], errors="raise") / 12}
    )
    if not np.isfinite(y[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Age target must be finite.")
    if X.columns.tolist() != FINAL_FEATURE_COLUMNS or X.shape[1] != 32:
        raise RuntimeError("The production feature schema is not exactly the frozen 32 columns.")

    metadata = {
        "subjects": int(len(X)),
        "features": int(X.shape[1]),
        "missing_feature_cells": int(X.isna().sum().sum()),
        "negative_timing_cells_censored": int(
            sum(
                (aggregate[column] < 0).fillna(False).sum()
                for column in INVALID_NEGATIVE_TIMING_COLUMNS
            )
        ),
        "subject_order": aggregate[ID_COLUMN].astype(int).tolist(),
    }
    return X, y, metadata


__all__ = [
    "FINAL_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "build_model_inputs",
]
