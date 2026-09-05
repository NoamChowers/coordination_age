import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.fit_jackknife_plus_krr import (
    fit_jackknife_plus_krr, save_jackknife_plus_artifacts,
)
from scripts.infer_jackknife_plus_krr import modified_jackknife_plus_bounds
from src.inference import CoordinationAgeModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return CoordinationAgeModel()


def test_held_out_predictions_and_intervals_reproduce_saved_results():
    model = CoordinationAgeModel(ROOT / 'artifacts/train115_jackknife_plus_krr')
    split = pd.read_csv(ROOT / 'Data/train_test_split.csv')
    test = split.query("partition == 'test'").sort_values('partition_position')
    X = pd.read_csv(ROOT / 'Data/X.csv').iloc[test.production_row_position].copy()
    X.index = test.SubjectNumber.to_numpy()
    result = model.predict(X)
    expected = pd.read_csv(ROOT / 'outputs/train115_jkplus/train115_jkplus_test29_predictions.csv').set_index('SubjectNumber').loc[X.index]
    for actual, saved in [('prediction', 'predicted_age_years'),
                          ('jkplus_lower', 'jkplus_lower'), ('jkplus_upper', 'jkplus_upper')]:
        np.testing.assert_allclose(result[actual], expected[saved], atol=1e-10, rtol=0)
    assert result.requested_coverage.eq(0.95).all()
    assert np.sqrt(np.mean((expected.chronological_age_years-result.prediction)**2)) == pytest.approx(1.403451450722289)


def test_full_and_reduced_artifacts_support_batch_and_single_inference(model):
    X = pd.read_csv(ROOT / 'Data/X.csv').iloc[:2].copy()
    X.index = ['first', 'second']
    batch = model.predict(X)
    single = model.predict(X.iloc[:1])
    pd.testing.assert_frame_equal(batch.iloc[:1], single, atol=1e-10, rtol=0)
    assert batch.index.tolist() == ['first', 'second']
    reduced = CoordinationAgeModel(ROOT / 'artifacts/jackknife_plus_krr_without_box_and_blocks')
    assert len(reduced.feature_columns) == 31
    assert np.isfinite(reduced.predict(X.drop(columns='boxAndBlocks')).prediction).all()


@pytest.mark.parametrize('case,match', [
    ('order', 'fitted order'), ('missing_column', 'fitted order'),
    ('extra_column', 'fitted order'), ('infinite', 'infinity'),
    ('nonnumeric', 'numeric'), ('empty', 'at least one'),
    ('all_missing', 'every predictor missing'), ('sex', 'must be 0, 1'),
    ('duplicate_ids', 'identifiers must be unique'),
])
def test_invalid_observations_fail_clearly(model, case, match):
    X = pd.read_csv(ROOT / 'Data/X.csv').iloc[:2].copy()
    if case == 'order': X = X[X.columns[::-1]]
    if case == 'missing_column': X = X.drop(columns=X.columns[-1])
    if case == 'extra_column': X['AgeInYears'] = 8
    if case == 'infinite': X.iloc[0, 1] = np.inf
    if case == 'nonnumeric': X[X.columns[1]] = 'invalid'
    if case == 'empty': X = X.iloc[:0]
    if case == 'all_missing': X.loc[X.index[0], :] = np.nan
    if case == 'sex': X.iloc[0, 0] = 2
    if case == 'duplicate_ids': X.index = ['same', 'same']
    with pytest.raises(ValueError, match=match):
        model.predict(X)


def test_modified_ranks_are_order_statistics_and_small_samples_are_unbounded():
    predictions = np.array([[10.], [20.], [30.], [40.]])
    residuals = np.array([1., 2., 3., 4.])
    lower, upper, ranks = modified_jackknife_plus_bounds(predictions, residuals, alpha=0.5)
    assert lower[0] == 9 and upper[0] == 44
    assert ranks['lower_rank_one_based'] == 1 and ranks['upper_rank_one_based'] == 4
    lower, upper, _ = modified_jackknife_plus_bounds(predictions, residuals, alpha=0.05)
    assert lower[0] == -np.inf and upper[0] == np.inf
    with pytest.raises(ValueError, match='alpha'):
        modified_jackknife_plus_bounds(predictions, residuals, alpha=0)


def test_small_fit_serialization_and_integrity_validation(tmp_path):
    X = pd.DataFrame({'Sex_0Female_1Male': [0, 1]*4, 'measurement': np.arange(8.)},
                     index=np.arange(101, 109))
    y = pd.Series(np.arange(8.)+4, index=X.index, name='AgeInYears')
    fitted = fit_jackknife_plus_krr(
        X, y, inner_folds=2, n_jobs=1, progress_every=100,
        param_grid={'model__regressor__alpha': [0.1], 'model__regressor__gamma': [0.01]},
    )
    paths = save_jackknife_plus_artifacts(
        loo_artifact=fitted[0], production_model=fitted[1],
        residuals=fitted[2], selections=fitted[3], output_dir=tmp_path,
    )
    loaded = CoordinationAgeModel(tmp_path)
    np.testing.assert_allclose(loaded.predict(X, alpha=0.5).prediction, fitted[1].predict(X))
    manifest = json.loads(paths['manifest'].read_text())
    assert manifest['n_loo_models'] == 8
    manifest['software']['scikit_learn'] = 'incompatible'
    paths['manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='require scikit-learn'):
        CoordinationAgeModel(tmp_path)
    import sklearn
    manifest['software']['scikit_learn'] = sklearn.__version__
    paths['manifest'].write_text(json.dumps(manifest))
    with paths['production_model'].open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        CoordinationAgeModel(tmp_path)
