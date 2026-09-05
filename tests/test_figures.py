from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.visualizations.create_all_figures import create_all_figures


def test_all_figures_generate_from_saved_tables(tmp_path, monkeypatch):
    from sklearn.model_selection import GridSearchCV
    def forbidden_fit(*args, **kwargs):
        raise AssertionError('Figure generation must not fit models.')
    monkeypatch.setattr(GridSearchCV, 'fit', forbidden_fit)
    manifest_path = create_all_figures(tmp_path)
    manifest = pd.read_csv(manifest_path)
    assert set(manifest.filename) == {
        'dataset_split_overview.png', 'figure_1_train_loocv_model_comparison.png',
        'figure_3_test_jkplus_interval_diagnostics.png',
        'figure_4_full_data_loocv_krr_diagnostics.png', 'task_ablation_rmse.png',
    }
    for filename in manifest.filename:
        with Image.open(tmp_path / filename) as image:
            assert image.width > 500 and image.height > 300
            image.verify()
    summary = pd.read_csv(tmp_path / 'visualization_summary_metrics.csv')
    np.testing.assert_allclose(summary.rmse_years, [1.42149926715, 1.40345145072, 1.3862], atol=5e-5)
    age = pd.read_csv(tmp_path / 'full_data_age_bin_metrics.csv')
    assert age.n.tolist() == [38, 43, 47, 16]
