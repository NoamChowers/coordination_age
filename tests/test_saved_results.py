"""Check consistency of the bundled results without repeating the large fits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_all_artifact_manifests_match_including_selection_tables():
    for path in (ROOT / 'artifacts').glob('*/*manifest.json'):
        manifest = json.loads(path.read_text())
        for artifact in manifest['artifacts'].values():
            data = (path.parent / artifact['filename']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == artifact['sha256']
            assert len(data) == artifact['bytes']


def test_ablation_metrics_and_frozen_path_match_saved_predictions():
    folder = ROOT / 'outputs/task_ablation'
    metrics = pd.read_csv(folder / 'task_subset_metrics.csv').set_index('subset_id')
    predictions = pd.read_csv(folder / 'task_subset_oof_predictions.csv')
    assert not predictions.duplicated(['subset_id', 'SubjectNumber']).any()
    assert len(metrics) == 21
    for key, group in predictions.groupby('subset_id'):
        assert len(group) == 115
        rmse = np.sqrt(np.mean((group.observed_age-group.oof_prediction)**2))
        np.testing.assert_allclose(rmse, metrics.loc[key, 'train_oof_rmse'], atol=1e-12)
    manifest = json.loads((folder / 'task_subset_manifest.json').read_text())
    digest = hashlib.sha256((folder / 'train_selected_backward_path.json').read_bytes()).hexdigest()
    assert digest == manifest['backward_path_sha256']
    assert digest == (folder / 'train_selected_backward_path.sha256').read_text().split()[0]
    test = pd.read_csv(folder / 'task_subset_test_predictions.csv')
    for key, group in test.groupby('subset_id'):
        assert len(group) == 29
        rmse = np.sqrt(np.mean((group.observed_age-group.test_prediction)**2))
        np.testing.assert_allclose(rmse, metrics.loc[key, 'test_rmse'], atol=1e-12)
