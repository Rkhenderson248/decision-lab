"""Smoke tests: every page renders without an exception.

Runs on every push (see .github/workflows/ci.yml), so a broken commit is
caught before Streamlit Community Cloud redeploys the live app.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    "streamlit_app.py",  # the home page needs the navigation context
    "views/lending.py",
    "views/decision_value.py",
    "views/goodhart.py",
    "views/market_intel.py",
    "views/experiments.py",
    "views/human_ai.py",
    "views/copilot.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_pipeline_capacity_change():
    at = AppTest.from_file(str(ROOT / "views/lending.py"), default_timeout=60).run()
    at.slider[0].set_value(300).run()
    assert not at.exception


def test_product_fit_no_eligible_products():
    at = AppTest.from_file(str(ROOT / "views/lending.py"), default_timeout=60)
    at.query_params["view"] = "product"
    at.run()
    # Drop the credit score below every programme minimum.
    credit = next(s for s in at.slider if s.label == "Credit score")
    credit.set_value(500).run()
    assert not at.exception


def test_market_engine_contract():
    from lab import market_engine as me
    from lab.market_sample import sample_markets

    raw = sample_markets()
    scored = me.score(me.prepare(raw))
    assert scored["market_key"].is_unique
    assert scored["strategic_mortgage_opportunity_score"].between(0, 100).all()
    assert set(scored["mortgage_risk_basis"]) <= {"Full", "HMDA-only", "Not available"}
    # Custom weights change the ranking but not the population of markets.
    tilted = me.score(me.prepare(raw), {"demographic": 0, "demand": 0, "capacity": 0, "collateral": 1, "openness": 0})
    assert len(tilted) == len(scored)
    assert not tilted["strategic_mortgage_opportunity_score"].equals(scored["strategic_mortgage_opportunity_score"])


def test_market_page_weight_change():
    at = AppTest.from_file(str(ROOT / "views/market_intel.py"), default_timeout=120).run()
    assert not at.exception
    at.segmented_control(key="mi_area").set_value("Micropolitan").run()
    assert not at.exception


def test_human_ai_full_run():
    at = AppTest.from_file(str(ROOT / "views/human_ai.py"), default_timeout=60).run()
    for _ in range(10):
        for _step in range(3):
            at.button[0].click().run()
            assert not at.exception
    assert any("Brier" in m.value for m in at.markdown)


@pytest.mark.parametrize("stage", ["frame", "data", "model", "decide", "prove", "run", "value"])
def test_copilot_stage(stage):
    at = AppTest.from_file(str(ROOT / "views/copilot.py"), default_timeout=90)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception


def test_copilot_next_button_advances():
    at = AppTest.from_file(str(ROOT / "views/copilot.py"), default_timeout=90).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["cp_stage"] == "Data"


def test_elasticity_recovers_truth():
    from lab import copilot as cp

    el = cp.elasticity_table()
    iv = el[el["method"].eq("Instrumented by rate tests")]
    # The instrumented interval covers the truth; the raw correlation does not.
    assert ((iv["lo"] <= iv["truth"]) & (iv["truth"] <= iv["hi"])).all()
    raw = el[el["method"].eq("Raw correlation")]
    assert ((raw["lo"] > raw["truth"]) | (raw["hi"] < raw["truth"])).all()


def test_backtest_cache_matches_fresh_run():
    import numpy as np

    from lab import demand as dm

    cached = dm.backtest()
    fresh = dm.compute_backtest()
    for df in (cached, fresh):
        df["ape"] = df["err"].abs() / df["final"]
    a = cached.groupby("method")["ape"].mean()
    b = fresh.groupby("method")["ape"].mean()
    assert np.allclose(a.sort_index().values, b.sort_index().values, rtol=1e-4)


def test_copilot_stage_survives_reselect():
    """Clicking the selected stage again must not send the visitor back to Frame."""
    at = AppTest.from_file(str(ROOT / "views/copilot.py"), default_timeout=90)
    at.query_params["stage"] = "run"
    at.run()
    at.session_state["cp_stage"] = None  # what Streamlit sends when the active option is clicked again
    at.run()
    assert not at.exception
    assert at.session_state["cp_stage__last"] == "Run"
    assert any("Run it" in m.value for m in at.markdown)
