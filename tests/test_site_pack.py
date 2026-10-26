"""The site pack: the exported weights must reproduce the production model, so a score in the website's assistant
matches the score the platform returns. build_pack() asserts that internally; these tests pin the contract."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import export_site_pack as esp  # noqa: E402

PACK = esp.build_pack()


def _predict(model, x):
    return model["intercept"] + sum((x[f["key"]] - f["mean"]) / f["scale"] * f["coef"] for f in model["features"])


def test_shape():
    assert set(PACK) == {"generated", "source", "model", "markets", "metrics"}
    m = PACK["model"]
    assert [f["key"] for f in m["features"]] == list(esp.ml.FEATURES)
    assert m["scored"] == len(PACK["markets"]["rows"]) > 100
    assert len(m["quantiles"]) == 101 and len(m["deciles"]) == 10
    assert {"by_year", "by_year_area", "by_state", "defs", "latest"} <= set(PACK["metrics"])


def test_browser_arithmetic_matches_model():
    m, cols = PACK["model"], PACK["markets"]["cols"]
    worst = 0.0
    for r in PACK["markets"]["rows"]:
        o = dict(zip(cols, r))
        x = {"growth": o["growth"], "origination_rate": o["orig_rate"], "denial_rate": o["denial"], "refinance_share": o["refi"],
             "log_applications": math.log1p(o["apps"]), "log_avg_loan": math.log(max(o["avg_loan"], 10_000)), "is_metro": o["metro"]}
        worst = max(worst, abs(_predict(m, x) - o["pred"]))
    assert worst < 5e-4  # inputs are rounded to 4 decimals in the pack


def test_ranked_and_public_only():
    preds = [r[-1] for r in PACK["markets"]["rows"]]
    assert preds == sorted(preds, reverse=True)
    assert np.isfinite(preds).all()
    assert "HMDA" in PACK["source"]
