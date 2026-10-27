"""Forecasting lab: every stage renders, and the backtest results the page relies on hold."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from lab.forecast import data as D
from lab.forecast import models as F

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["demand", "model-bake-off", "reconcile", "uncertainty", "order", "monitor"]


@pytest.fixture(scope="module")
def study():
    return F.run()


@pytest.mark.parametrize("stage", STAGES)
def test_stage_renders(stage):
    at = AppTest.from_file(str(ROOT / "views/forecast_lab.py"), default_timeout=180)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception


def test_deterministic():
    assert D.simulate().hash == D.simulate().hash


def test_hierarchy_sums():
    c = D.simulate()
    nodes, S = D.hierarchy(list(c.wide.columns))
    assert len(nodes) == 112 and S.shape == (112, 72)
    assert np.allclose((c.wide.values @ S.T)[:, 0], c.wide.values.sum(1))


def test_global_model_beats_baselines(study):
    s = study.scores.groupby("model").WAPE.mean()
    assert s["Global gradient boosting"] < s["Holt-Winters (damped)"] < s["Seasonal naive"]


def test_reconciled_forecasts_are_coherent(study):
    fd = study.folds[-1]
    for k, R in fd["rec"].items():
        assert np.allclose(R[0], R[-72:].sum(0), rtol=1e-6), k


def test_intervals_cover(study):
    assert 0.74 <= study.coverage[.8]["coverage"] <= 0.86
    assert 0.90 <= study.coverage[.95]["coverage"] <= 0.98


def test_better_forecast_cheaper_orders(study):
    inv = study.inventory.set_index("method").cost
    assert inv["Global gradient boosting"] < inv["Seasonal naive"]


def test_drift_alert_finds_the_planted_store(study):
    al = F.alerts(study).dropna()
    assert list(al.store) == [D.BREAK_STORE]
    assert D.BREAK_WEEK <= al.iloc[0]["first alert week"] <= D.BREAK_WEEK + 8
