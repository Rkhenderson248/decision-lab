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
