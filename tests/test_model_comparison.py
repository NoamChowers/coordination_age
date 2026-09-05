import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from scripts.evaluate_model_candidates import combine_results, evaluate_candidate


def test_checkpoint_resumes_without_refitting_and_rejects_changed_data(tmp_path, monkeypatch):
    X = pd.DataFrame({"feature": np.arange(8.0)}, index=np.arange(101, 109))
    y = pd.Series(np.arange(8.0), index=X.index)
    spec = {"estimator": Pipeline([("model", DummyRegressor())]), "grid": None}
    first, manifest = evaluate_candidate(X, y, "Dummy", spec, tmp_path, 1)
    expected = [(y.sum() - value) / (len(y) - 1) for value in y]
    np.testing.assert_allclose(first.oof_prediction_years, expected)

    def fail_fit(*args, **kwargs):
        raise AssertionError("A completed checkpoint must not refit.")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)
    second, _ = evaluate_candidate(X, y, "Dummy", spec, tmp_path, 1)
    pd.testing.assert_frame_equal(first, second, check_like=True)
    changed = y.copy()
    changed.iloc[0] += 1
    with pytest.raises(ValueError, match="Stale configuration"):
        evaluate_candidate(X, changed, "Dummy", spec, tmp_path, 1)
    assert json.loads(manifest.read_text())["participant_ids"] == X.index.tolist()


def test_combining_replaces_only_requested_candidate_and_checks_participants():
    baseline = pd.DataFrame({
        "SubjectNumber": [101, 102, 101, 102],
        "procedure": ["KRR", "KRR", "RF", "RF"],
        "chronological_age_years": [5.0, 7.0, 5.0, 7.0],
        "oof_prediction_years": [5.1, 6.9, 4.0, 8.0],
    })
    y = pd.Series([5.0, 7.0], index=[101, 102])
    update = baseline.query("procedure == 'RF'").copy()
    update["oof_prediction_years"] = [4.5, 7.5]
    combined, metrics = combine_results(baseline, {"RF": update}, y)
    pd.testing.assert_frame_equal(
        baseline.query("procedure == 'KRR'").reset_index(drop=True),
        combined.query("procedure == 'KRR'").reset_index(drop=True),
    )
    assert metrics.set_index("procedure").loc["RF", "rmse_years"] == 0.5
    with pytest.raises(ValueError, match="Baseline outcomes/participants"):
        combine_results(baseline, {}, y.rename(index={102: 103}))


def test_cli_can_start_without_saved_comparison_and_preserves_other_candidates(tmp_path, monkeypatch):
    import sys
    import scripts.evaluate_model_candidates as runner
    X = pd.DataFrame({'feature': np.arange(8.)}, index=np.arange(101, 109))
    y = pd.Series(np.arange(8.), index=X.index)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/krr_pipeline.py').write_text('# test preprocessing fingerprint\n')
    monkeypatch.setattr(runner, 'load_training_data', lambda: (X, y))
    specs = {name: {'estimator': Pipeline([('model', DummyRegressor())]), 'grid': None}
             for name in ['First', 'Second']}
    monkeypatch.setattr(runner, 'candidate_specs', lambda features: specs)
    output = tmp_path / 'outputs/model_comparison'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--models', 'First', 'Second',
                                    '--output-dir', str(output), '--n-jobs', '1'])
    runner.main()
    first = pd.read_csv(output / 'comparison_oof_predictions.csv')
    assert set(first.procedure) == {'First', 'Second'}
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--models', 'Second',
                                    '--output-dir', str(output), '--n-jobs', '1'])
    runner.main()
    pd.testing.assert_frame_equal(first, pd.read_csv(output / 'comparison_oof_predictions.csv'))
