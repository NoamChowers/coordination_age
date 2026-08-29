import numpy as np
import pandas as pd

from src.model_inputs import (
    FINAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model_inputs,
)


def test_build_model_inputs_freezes_schema_target_and_timing_rule():
    row = {column: 1.0 for column in FINAL_FEATURE_COLUMNS}
    row.update(
        {
            "SubjectNumber": 17,
            "AgeInMonths": 126,
            "T1T3Light": -0.1,
        }
    )

    X, y, metadata = build_model_inputs(pd.DataFrame([row]))

    assert X.columns.tolist() == FINAL_FEATURE_COLUMNS
    assert X.shape == (1, 32)
    assert np.isnan(X.loc[0, "T1T3Light"])
    assert y.columns.tolist() == [TARGET_COLUMN]
    assert y.loc[0, TARGET_COLUMN] == 10.5
    assert metadata["negative_timing_cells_censored"] == 1
    assert metadata["subject_order"] == [17]
