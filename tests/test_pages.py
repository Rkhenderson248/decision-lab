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
    "views/pipeline.py",
    "views/product_fit.py",
    "views/decision_value.py",
    "views/goodhart.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_pipeline_capacity_change():
    at = AppTest.from_file(str(ROOT / "views/pipeline.py"), default_timeout=60).run()
    at.slider[0].set_value(300).run()
    assert not at.exception


def test_product_fit_no_eligible_products():
    at = AppTest.from_file(str(ROOT / "views/product_fit.py"), default_timeout=60).run()
    # Drop the credit score below every programme minimum.
    credit = next(s for s in at.slider if s.label == "Credit score")
    credit.set_value(500).run()
    assert not at.exception
