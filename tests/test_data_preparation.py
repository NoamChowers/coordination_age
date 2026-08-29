import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    FAMILY_SOURCE,
    PASSTHROUGH_COLUMNS,
    build_aggregate_dataset,
    correct_all_trials,
)


def test_blacklist_and_direct_duplicates_are_removed_and_aggregate_is_updated():
    all_trials = pd.DataFrame(
        {
            "SubjectNumber": [101],
            "deltaV": [2.0],
            "deltaV3": [1.0],
            "deltaV4": [2.0],
            "deltaV5": [1.0],
            "deltaV6": [4.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "subject": [101] * 4,
            "task": ["pressing"] * 4,
            "condition": [None] * 4,
            "trial": [3, 4, 5, 6],
            "blacklisted": [0, 0, 0, 1],
        }
    )

    corrected, audit, summary = correct_all_trials(all_trials, manifest)

    assert corrected.loc[0, "deltaV"] == 1.5
    assert np.isnan(corrected.loc[0, "deltaV5"])
    assert np.isnan(corrected.loc[0, "deltaV6"])
    assert summary["duplicate_trials"] == 1
    assert summary["cells_changed_to_nan"] == 2
    assert set(audit.loc[audit["cell_changed"], "reason"]) == {
        "direct_vector_match",
        "blacklisted",
    }


def test_supported_half_table_copy_infers_hidden_destination_trials():
    row = {"SubjectNumber": 141}
    for family, multiplier in [("deltaV", 1.0), ("V_UCM", 10.0), ("V_ORT", 100.0)]:
        values = {
            3: 1.0,
            4: 2.0,
            5: 3.0,
            6: 6.0,
            7: 7.0,
            8: 8.0,
            9: 9.0,
            10: 10.0,
            11: 101.0,
            12: 102.0,
            13: 1.0,
            14: 2.0,
            15: 3.0,
            16: np.nan,
            17: np.nan,
            18: np.nan,
            19: np.nan,
            20: np.nan,
        }
        for trial, value in values.items():
            row[f"{family}{trial}"] = multiplier * value
        row[family] = np.mean([multiplier * value for value in values.values()])
    all_trials = pd.DataFrame([row])
    manifest = pd.DataFrame(
        {
            "subject": [141] * 20,
            "task": ["pressing"] * 20,
            "condition": [None] * 20,
            "trial": list(range(1, 21)),
            "blacklisted": [1, 1] + [0] * 18,
        }
    )

    corrected, audit, summary = correct_all_trials(all_trials, manifest)

    assert summary["inferred_duplicate_blocks"] == 1
    assert summary["block_inferred_duplicate_trials"] == 10
    for family in ("deltaV", "V_UCM", "V_ORT"):
        assert corrected.loc[0, [f"{family}{trial}" for trial in range(11, 21)]].isna().all()
    hidden = audit[(audit["trial"].isin([11, 12])) & audit["cell_changed"]]
    assert set(hidden["reason"]) == {"inferred_block_copy"}


def complete_valid_all_trials() -> pd.DataFrame:
    row = {column: 1.0 for column in PASSTHROUGH_COLUMNS}
    row["SubjectNumber"] = 101
    for family, (task, _) in FAMILY_SOURCE.items():
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        if family in {"peakvel", "movementtime", "maxaperturePercent"}:
            values[-1] = np.nan
        for trial, value in enumerate(values, start=1):
            row[f"{family}{trial}"] = value
        embedded = "numPeaks" + family[len("numpeaks") :] if family.startswith("numpeaks") else family
        row[embedded] = np.nanmean(values) if task == "pressing" else np.nanmedian(values)
    row["liftMeanDifferencesLight"] = 0.25
    row["liftMeanDifferencesHeavy"] = 0.50
    return pd.DataFrame([row])


def test_aggregation_counts_and_censors_below_five():
    all_trials = complete_valid_all_trials()
    supplemental = pd.DataFrame(
        {"SubjectNumber": [101], "meandistance": [9.5]}
    )

    aggregate, summary = build_aggregate_dataset(all_trials, supplemental)

    assert np.isnan(aggregate.loc[0, "meandistance"])
    assert aggregate.loc[0, "n_trials_peakvel"] == 4
    assert np.isnan(aggregate.loc[0, "peakvel"])
    assert aggregate.loc[0, "n_trials_movementtime"] == 4
    assert np.isnan(aggregate.loc[0, "movementtime"])
    assert aggregate.loc[0, "n_trials_peakGripForceLight"] == 5
    assert aggregate.loc[0, "peakGripForceLight"] == 3.0
    assert aggregate.loc[0, "peakGripForceDifference"] == 0.0
    assert summary["minimum_valid_trials"] == 5
    assert summary["censored_aggregate_cells"] == 4
    assert summary["meandistance_proxy_censored_subjects"] == 1


def test_meandistance_proxy_rejects_disagreeing_reach_counts():
    all_trials = complete_valid_all_trials()
    all_trials.loc[0, "movementtime5"] = 5.0
    supplemental = pd.DataFrame(
        {"SubjectNumber": [101], "meandistance": [9.5]}
    )

    with pytest.raises(ValueError, match="reach feature counts disagree"):
        build_aggregate_dataset(all_trials, supplemental)
