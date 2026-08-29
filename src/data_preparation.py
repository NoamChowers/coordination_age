"""Auditable construction of valid all-trials and aggregate datasets."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


ID_COLUMN = "SubjectNumber"
MINIMUM_VALID_TRIALS = 5
MINIMUM_BLOCK_MATCHES = 3
MINIMUM_BLOCK_MATCH_FRACTION = 0.80

# Repeated feature family -> raw task and condition.
FAMILY_SOURCE: dict[str, tuple[str, str | None]] = {
    "deltaV": ("pressing", None),
    "V_UCM": ("pressing", None),
    "V_ORT": ("pressing", None),
    "peakvel": ("reachtograsp", None),
    "movementtime": ("reachtograsp", None),
    "maxaperturePercent": ("reachtograsp", None),
}
for _feature in (
    "peakGripForce",
    "peakLoadForce",
    "peakGripForceRate",
    "peakLoadForceRate",
    "T1T2",
    "T1T3",
    "T1T5",
    "numpeaks",
):
    FAMILY_SOURCE[f"{_feature}Light"] = ("liftobject", "light")
    FAMILY_SOURCE[f"{_feature}Heavy"] = ("liftobject", "heavy")

AGGREGATION = {
    family: "mean" if task == "pressing" else "median"
    for family, (task, _) in FAMILY_SOURCE.items()
}

PASSTHROUGH_COLUMNS = [
    ID_COLUMN,
    "Adult_0Child_1",
    "AgeInMonths",
    "Sex_0Female_1Male",
    "boxAndBlocks",
    "jebsen_PageTurning",
    "jebsen_SmallObjects",
    "jebsen_Beans",
    "jebsen_Checkers",
    "jebsen_LightCans",
    "jebsen_HeavyCans",
    "jebsenTaylor",
    "MVC",
]

AGGREGATE_COLUMNS = PASSTHROUGH_COLUMNS + [
    "deltaV",
    "V_UCM",
    "V_ORT",
    "peakvel",
    "movementtime",
    "maxaperturePercent",
    "meandistance",
    "peakGripForceLight",
    "peakGripForceHeavy",
    "peakGripForceDifference",
    "peakLoadForceLight",
    "peakLoadForceHeavy",
    "peakLoadForceDifference",
    "peakGripForceRateLight",
    "peakGripForceRateHeavy",
    "peakGripForceRateDifference",
    "peakLoadForceRateLight",
    "peakLoadForceRateHeavy",
    "peakLoadForceRateDifference",
    "T1T2Light",
    "T1T2Heavy",
    "T1T2Difference",
    "T1T3Light",
    "T1T3Heavy",
    "T1T3Difference",
    "T1T5Light",
    "T1T5Heavy",
    "T1T5Difference",
    "liftMeanDifferencesLight",
    "liftMeanDifferencesHeavy",
    "numPeaksLight",
    "numPeaksHeavy",
]

DIFFERENCE_SOURCES = {
    f"{stem}Difference": (f"{stem}Heavy", f"{stem}Light")
    for stem in (
        "peakGripForce",
        "peakLoadForce",
        "peakGripForceRate",
        "peakLoadForceRate",
        "T1T2",
        "T1T3",
        "T1T5",
    )
}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}; use CSV or Parquet.")


def repetition_columns(dataframe: pd.DataFrame, family: str) -> list[str]:
    """Return a family's numbered trial columns in numeric trial order."""

    pattern = re.compile(rf"{re.escape(family)}(\d+)")
    matches = [column for column in dataframe.columns if pattern.fullmatch(column)]
    return sorted(matches, key=lambda column: int(pattern.fullmatch(column).group(1)))


def aggregate_column(dataframe: pd.DataFrame, family: str) -> str:
    """Resolve case differences such as numpeaks -> numPeaks."""

    matches = [column for column in dataframe.columns if column.lower() == family.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one embedded aggregate for {family!r}; found {matches}."
        )
    return matches[0]


def _validate_subject_table(dataframe: pd.DataFrame, *, name: str) -> None:
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(f"{name} is missing {ID_COLUMN!r}.")
    if dataframe[ID_COLUMN].isna().any() or not dataframe[ID_COLUMN].is_unique:
        raise ValueError(f"{name} must contain unique, nonmissing subject identifiers.")


def normalize_trial_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate the normalized blacklist/QC manifest used by stage one."""

    manifest = manifest.copy()
    if "subject" not in manifest.columns and ID_COLUMN in manifest.columns:
        manifest = manifest.rename(columns={ID_COLUMN: "subject"})
    required = {"subject", "task", "condition", "trial", "blacklisted"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            f"The blacklist manifest is missing columns: {missing}."
        )

    manifest["subject"] = pd.to_numeric(manifest["subject"], errors="raise").astype(int)
    manifest["trial"] = pd.to_numeric(manifest["trial"], errors="raise").astype(int)
    manifest["blacklisted"] = pd.to_numeric(
        manifest["blacklisted"], errors="coerce"
    )
    relevant = manifest["task"].isin({task for task, _ in FAMILY_SOURCE.values()})
    if manifest.loc[relevant, "blacklisted"].isna().any():
        raise ValueError("Every modeled-task trial must have a blacklist decision.")
    bad_flags = set(manifest.loc[relevant, "blacklisted"].dropna().unique()) - {0, 1}
    if bad_flags:
        raise ValueError(f"Blacklist flags must be 0/1; found {sorted(bad_flags)}.")

    manifest["condition_key"] = manifest["condition"].fillna("<none>").astype(str)
    key = ["subject", "task", "condition_key", "trial"]
    if manifest.loc[relevant].duplicated(key).any():
        raise ValueError("The blacklist/QC manifest contains duplicate trial keys.")
    return manifest


def _same_observed_vector(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether two nonempty task vectors match exactly, including NaNs."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if not np.isfinite(left_array).any() or not np.isfinite(right_array).any():
        return False
    if not np.array_equal(np.isnan(left_array), np.isnan(right_array)):
        return False
    observed = ~np.isnan(left_array)
    return bool(np.array_equal(left_array[observed], right_array[observed]))


def detect_copied_trials_from_features(
    all_trials: pd.DataFrame,
) -> tuple[dict[tuple[int, str, str | None, int], set[str]], list[dict[str, Any]]]:
    """Detect direct and repeated-block copies using only layer-2 features.

    An isolated later trial is flagged when its complete task-specific feature
    vector exactly equals an earlier visible trial. A full second-half block is
    inferred only when the task has an even maximum trial number, at least
    ``MINIMUM_BLOCK_MATCHES`` visible pairs match at the half-table offset, and
    at least ``MINIMUM_BLOCK_MATCH_FRACTION`` of comparable pairs agree.
    """

    indexed = all_trials.set_index(ID_COLUMN)
    detections: dict[tuple[int, str, str | None, int], set[str]] = {}
    block_evidence: list[dict[str, Any]] = []
    task_conditions = list(dict.fromkeys(FAMILY_SOURCE.values()))

    for task, condition in task_conditions:
        families = [
            family
            for family, source in FAMILY_SOURCE.items()
            if source == (task, condition)
        ]
        numbered: dict[str, dict[int, str]] = {}
        for family in families:
            family_columns = repetition_columns(all_trials, family)
            pattern = re.compile(rf"{re.escape(family)}(\d+)")
            numbered[family] = {
                int(pattern.fullmatch(column).group(1)): column
                for column in family_columns
            }
        trial_numbers = sorted(
            {trial for family_map in numbered.values() for trial in family_map}
        )
        if not trial_numbers:
            continue
        maximum_trial = max(trial_numbers)

        for subject, row in indexed.iterrows():
            subject = int(subject)
            vectors = {
                trial: tuple(
                    row[numbered[family][trial]]
                    if trial in numbered[family]
                    else np.nan
                    for family in families
                )
                for trial in range(1, maximum_trial + 1)
            }

            # Isolated exact copies: retain the earliest matching trial.
            for later in range(2, maximum_trial + 1):
                for earlier in range(1, later):
                    if _same_observed_vector(vectors[earlier], vectors[later]):
                        key = (subject, task, condition, later)
                        detections.setdefault(key, set()).add("direct_vector_match")
                        break

            # Conservative block-copy inference, designed for n -> 2n filling.
            if maximum_trial % 2:
                continue
            offset = maximum_trial // 2
            comparable = 0
            matching = 0
            for source_trial in range(1, offset + 1):
                destination_trial = source_trial + offset
                left = np.asarray(vectors[source_trial], dtype=float)
                right = np.asarray(vectors[destination_trial], dtype=float)
                if not np.isfinite(left).any() or not np.isfinite(right).any():
                    continue
                comparable += 1
                matching += int(_same_observed_vector(left, right))
            match_fraction = matching / comparable if comparable else 0.0
            if (
                matching >= MINIMUM_BLOCK_MATCHES
                and match_fraction >= MINIMUM_BLOCK_MATCH_FRACTION
            ):
                for destination_trial in range(offset + 1, maximum_trial + 1):
                    key = (subject, task, condition, destination_trial)
                    detections.setdefault(key, set()).add("inferred_block_copy")
                block_evidence.append(
                    {
                        ID_COLUMN: subject,
                        "task": task,
                        "condition": condition,
                        "maximum_trial": maximum_trial,
                        "offset": offset,
                        "matching_pairs": matching,
                        "comparable_pairs": comparable,
                        "match_fraction": match_fraction,
                    }
                )
    return detections, block_evidence


def correct_all_trials(
    all_trials: pd.DataFrame,
    trial_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply blacklist truth and remove raw-file copies from a wide trial table."""

    _validate_subject_table(all_trials, name="all-trials input")
    original = all_trials.set_index(ID_COLUMN)
    corrected = original.copy()
    manifest = normalize_trial_manifest(trial_manifest)
    duplicate_methods, block_evidence = detect_copied_trials_from_features(all_trials)

    def detection_for_row(row: pd.Series) -> str:
        condition = None if row["condition_key"] == "<none>" else row["condition_key"]
        methods = duplicate_methods.get(
            (int(row["subject"]), row["task"], condition, int(row["trial"])),
            set(),
        )
        return "+".join(sorted(methods))

    manifest["duplicate_method"] = manifest.apply(detection_for_row, axis=1)
    manifest["invalid_reason"] = manifest.apply(
        lambda row: (
            "blacklisted+" + row["duplicate_method"]
            if row["blacklisted"] == 1 and row["duplicate_method"]
            else "blacklisted"
            if row["blacklisted"] == 1
            else row["duplicate_method"]
        ),
        axis=1,
    )

    audit_rows: list[dict[str, Any]] = []
    changed_pairs: set[tuple[int, str]] = set()
    invalid = manifest[manifest["invalid_reason"].ne("")]
    for row in invalid.itertuples(index=False):
        condition = None if row.condition_key == "<none>" else row.condition_key
        for family, (task, family_condition) in FAMILY_SOURCE.items():
            if task != row.task or family_condition != condition:
                continue
            column = f"{family}{row.trial}"
            if column not in corrected.columns or row.subject not in corrected.index:
                continue
            previous = corrected.at[row.subject, column]
            changed = pd.notna(previous)
            corrected.at[row.subject, column] = np.nan
            if changed:
                changed_pairs.add((int(row.subject), family))
            audit_rows.append(
                {
                    ID_COLUMN: int(row.subject),
                    "task": row.task,
                    "condition": condition,
                    "trial": int(row.trial),
                    "family": family,
                    "column": column,
                    "reason": row.invalid_reason,
                    "previous_value": previous,
                    "cell_changed": bool(changed),
                }
            )

    # Keep each embedded aggregate synchronized when its contributing cells move.
    for subject, family in sorted(changed_pairs):
        columns = repetition_columns(corrected, family)
        embedded = aggregate_column(corrected, family)
        method = AGGREGATION[family]
        shipped = original.at[subject, embedded]
        reproduced = getattr(original.loc[subject, columns], method)(skipna=True)
        if pd.notna(shipped) and not np.isclose(shipped, reproduced, rtol=1e-6):
            raise ValueError(
                f"Cannot safely update {embedded} for subject {subject}: "
                f"embedded value {shipped!r} is not reproduced by {method} "
                f"over its uncorrected trial columns ({reproduced!r})."
            )
        corrected.at[subject, embedded] = getattr(
            corrected.loc[subject, columns], method
        )(skipna=True)

    audit = pd.DataFrame(audit_rows)
    summary = {
        "manifest_trials": int(len(manifest)),
        "blacklisted_trials": int(manifest["blacklisted"].eq(1).sum()),
        "direct_vector_duplicate_trials": int(
            sum("direct_vector_match" in methods for methods in duplicate_methods.values())
        ),
        "block_inferred_duplicate_trials": int(
            sum("inferred_block_copy" in methods for methods in duplicate_methods.values())
        ),
        "duplicate_trials": int(len(duplicate_methods)),
        "inferred_duplicate_blocks": int(len(block_evidence)),
        "invalid_trial_family_rows": int(len(audit)),
        "cells_changed_to_nan": int(audit.get("cell_changed", pd.Series(dtype=bool)).sum()),
        "embedded_aggregates_recomputed": int(len(changed_pairs)),
    }
    return corrected.reset_index(), audit, summary


def _family_output_name(family: str) -> str:
    return "numPeaks" + family[len("numpeaks") :] if family.startswith("numpeaks") else family


def build_aggregate_dataset(
    valid_all_trials: pd.DataFrame,
    supplemental_aggregates: pd.DataFrame,
    *,
    minimum_valid_trials: int = MINIMUM_VALID_TRIALS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build aggregate features and counts, then mask families below the threshold.

    ``meandistance`` is the sole aggregate feature not represented in the legacy
    all-trials schema, so it is copied from ``supplemental_aggregates``. Pressing
    aggregates use their embedded values because visible layer-2 repetitions
    begin at trial 3 while some shipped aggregates also used trials 1 and 2.
    """

    if minimum_valid_trials < 1:
        raise ValueError("minimum_valid_trials must be positive.")
    _validate_subject_table(valid_all_trials, name="valid all-trials input")
    _validate_subject_table(supplemental_aggregates, name="supplemental aggregates")
    all_trials = valid_all_trials.set_index(ID_COLUMN)
    supplemental = supplemental_aggregates.set_index(ID_COLUMN)
    if set(all_trials.index) != set(supplemental.index):
        raise ValueError("All-trials and supplemental aggregate subject sets differ.")
    supplemental = supplemental.loc[all_trials.index]
    if "meandistance" not in supplemental.columns:
        raise ValueError("Supplemental aggregates must provide meandistance.")

    missing_passthrough = [c for c in PASSTHROUGH_COLUMNS[1:] if c not in all_trials]
    if missing_passthrough:
        raise ValueError(f"All-trials input lacks passthrough columns: {missing_passthrough}")

    result = all_trials[PASSTHROUGH_COLUMNS[1:]].copy()
    counts: dict[str, pd.Series] = {}
    for family, (task, _) in FAMILY_SOURCE.items():
        columns = repetition_columns(all_trials, family)
        if not columns:
            raise ValueError(f"No numbered trial columns found for {family}.")
        output_name = _family_output_name(family)
        counts[family] = all_trials[columns].notna().sum(axis=1).astype(int)
        if task == "pressing":
            embedded = aggregate_column(all_trials, family)
            result[output_name] = all_trials[embedded]
        else:
            result[output_name] = all_trials[columns].median(axis=1, skipna=True)

    result["meandistance"] = supplemental["meandistance"]
    reach_count_families = ["peakvel", "movementtime", "maxaperturePercent"]
    reach_counts = pd.concat(
        [counts[family].rename(family) for family in reach_count_families],
        axis=1,
    )
    if not reach_counts.nunique(axis=1).eq(1).all():
        mismatched_subjects = reach_counts.index[
            reach_counts.nunique(axis=1).ne(1)
        ].tolist()
        raise ValueError(
            "Cannot apply the reach-to-grasp trial-count proxy to meandistance: "
            "observable reach feature counts disagree for subjects "
            f"{mismatched_subjects}."
        )
    shared_reach_count = reach_counts.iloc[:, 0]
    result.loc[shared_reach_count < minimum_valid_trials, "meandistance"] = np.nan
    for column in ("liftMeanDifferencesLight", "liftMeanDifferencesHeavy"):
        if column not in all_trials.columns:
            raise ValueError(f"All-trials input lacks {column!r}.")
        result[column] = all_trials[column]
    for output, (heavy, light) in DIFFERENCE_SOURCES.items():
        result[output] = result[heavy] - result[light]

    for family, family_counts in counts.items():
        output_name = _family_output_name(family)
        result.loc[family_counts < minimum_valid_trials, output_name] = np.nan
        result[f"n_trials_{family}"] = family_counts

    result = result.reset_index()
    expected_columns = AGGREGATE_COLUMNS + [f"n_trials_{family}" for family in FAMILY_SOURCE]
    result = result[expected_columns]
    summary = {
        "subjects": int(len(result)),
        "aggregate_columns": int(len(AGGREGATE_COLUMNS)),
        "trial_count_columns": int(len(FAMILY_SOURCE)),
        "minimum_valid_trials": int(minimum_valid_trials),
        "censored_aggregate_cells": int(
            sum((family_counts < minimum_valid_trials).sum() for family_counts in counts.values())
            + (shared_reach_count < minimum_valid_trials).sum()
        ),
        "meandistance_proxy_censored_subjects": int(
            (shared_reach_count < minimum_valid_trials).sum()
        ),
    }
    return result, summary


__all__ = [
    "AGGREGATE_COLUMNS",
    "FAMILY_SOURCE",
    "ID_COLUMN",
    "MINIMUM_VALID_TRIALS",
    "build_aggregate_dataset",
    "correct_all_trials",
    "detect_copied_trials_from_features",
    "read_table",
]
