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
    "views/policy_assistant.py",
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
    # Drop the credit score below every program minimum.
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


@pytest.mark.parametrize("stage", ["know", "acquire", "underwrite", "fraud", "price", "crosssell", "retain", "collect", "govern"])
def test_lending_lab_stage(stage):
    at = AppTest.from_file(str(ROOT / "views/lending_lab.py"), default_timeout=180)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception


def test_lending_lab_follows_each_featured_member():
    at = AppTest.from_file(str(ROOT / "views/lending_lab.py"), default_timeout=180)
    at.query_params["stage"] = "collect"
    at.run()
    box = at.selectbox(key="cu_member_label")
    for option in box.options:
        box.set_value(option).run()
        assert not at.exception, option


def test_lending_lab_relationships_hold():
    """The planted structure the lab teaches must still be in the data."""
    from lab.cu import models as M

    seg = M.segment()
    assert seg.ari > 0.5                                   # segmentation recovers the archetypes
    uw = M.underwriting()
    assert uw.auc["Scorecard + reject inference"] > uw.auc["Legacy policy"]
    top = M.fraud().nlargest(250, "fraud_score")
    assert top["is_fraud"].mean() > 0.6                    # the queue is mostly fraud
    up = M.uplift()
    assert up.groupby("segment")["uplift"].mean()["Retired loyalists"] < 0  # contact backfires for one segment


def test_policy_assistant_answers_and_declines():
    at = AppTest.from_file(str(ROOT / "views/policy_assistant.py"), default_timeout=60)
    at.run()
    at.text_input(key="pa_q").set_value("Who can approve an exception on a $40,000 loan?").run()
    assert not at.exception
    assert any("UW-2.8" in m.value for m in at.markdown)
    at.text_input(key="pa_q").set_value("How do I reset my online banking password?").run()
    assert any("doesn't cover this" in m.value for m in at.markdown)


def test_policy_retrieval_quality():
    from lab.policy import engine as E

    ev = E.evaluate()
    ins = ev[ev["in_scope"]]
    assert ins["rank"].notna().mean() == 1.0      # right section always in the top three
    assert (ins["rank"] == 1).mean() >= 0.85


@pytest.mark.parametrize("stage", ["trends", "value", "segment", "churn", "treat", "price", "test", "care", "govern"])
def test_subscriber_lab_stage(stage):
    at = AppTest.from_file(str(ROOT / "views/subscriber_lab.py"), default_timeout=240)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception


def test_subscriber_lab_controls():
    at = AppTest.from_file(str(ROOT / "views/subscriber_lab.py"), default_timeout=240)
    at.query_params["stage"] = "price"
    at.run()
    at.radio(key="sv_pmode").set_value("Best by segment").run()
    at.toggle(key="sv_ex_win").set_value(False).run()
    assert not at.exception
    at.query_params["stage"] = "treat"
    at = AppTest.from_file(str(ROOT / "views/subscriber_lab.py"), default_timeout=240)
    at.query_params["stage"] = "treat"
    at.run()
    at.radio(key="sv_rank").set_value("Churn risk").run()
    box = at.selectbox(key="sv_sub_label")
    for option in box.options:
        box.set_value(option).run()
        assert not at.exception, option


def test_subscriber_lab_relationships_hold():
    """What the subscriber lab teaches must stay true in the data."""
    from lab.sub import data as D
    from lab.sub import models as M

    c = D.company()
    assert 0.007 < c.monthly["churn_rate"].mean() < 0.016          # realistic monthly churn
    hz = M.hazard_by_tenure()
    twelve = hz[hz["contract"] == "12-month contract"].set_index("tenure")["hazard"]
    assert twelve.loc[12:13].mean() > 3 * twelve.loc[5:10].mean()    # contract-end spike
    assert M.segments().ari > 0.5
    up = M.uplift()
    assert up.qini_area["Uplift (logistic, interactions)"] > up.qini_area["Churn risk"]
    g = up.test.groupby("segment")["uplift"].mean()
    assert g["Promo switchers"] > 0.02 and g["Light users"] < 0     # who an offer saves, and who it pushes out
    best = M.best_increase_by_segment().pivot(index="increase", columns="segment", values="net")
    assert best.loc[5, "Bundled households"] > 0 > best.loc[5, "Promo switchers"]


def test_model_bench_contract():
    from lab import bench as B
    from lab.cu import algos as A

    b = A.uw_bench()
    assert list(b.table["algorithm"]) == B.ALGOS
    assert b.table["auc"].between(0.7, 0.95).all()
    champ, _ = B.recommend(b.table, regulated=True)
    assert champ in ("Logistic regression", "Linear regression")    # regulated decisions need exact reasons
    fd = A.fraud_detectors()
    q = fd[fd["queue"] == 250].set_index("detector")["caught"]
    assert q["Isolation forest"] > q["One-class SVM"]
    auc, imp = A.churn_forest()
    noise = imp.set_index("feature").loc["random_noise"]
    assert noise["impurity"] > 0.03 and noise["permutation"] < 0.01  # the importance trap is visible


@pytest.mark.parametrize("page,stages", [
    ("views/causal_lab.py", ["frame", "did", "synthetic", "matching", "decide"]),
    ("views/mmm_lab.py", ["data", "model", "calibrate", "roi", "optimize"]),
    ("views/analytics_copilot.py", ["ask", "metrics", "evaluate", "govern"]),
])
def test_new_labs_every_stage(page, stages):
    for stage in stages:
        at = AppTest.from_file(str(ROOT / page), default_timeout=240)
        at.query_params["stage"] = stage
        at.run()
        assert not at.exception, (page, stage, at.exception)


def test_causal_lab_controls():
    at = AppTest.from_file(str(ROOT / "views/causal_lab.py"), default_timeout=240)
    at.query_params["stage"] = "matching"
    at.run()
    at.radio(key="ci_scen").set_value("They picked markets already getting worse").run()
    at.toggle(key="ci_hidden").set_value(True).run()
    assert not at.exception


def test_analytics_copilot_answers_and_declines():
    at = AppTest.from_file(str(ROOT / "views/analytics_copilot.py"), default_timeout=240)
    at.run()
    at.text_input(key="ac_q").set_value("What is our churn rate by region?").run()
    assert not at.exception
    assert any("SELECT" in c.value for c in at.code)
    at.text_input(key="ac_q").set_value("Churn by zip code").run()
    assert any("Declined" in m.value for m in at.markdown)
    at.text_input(key="ac_q").set_value("Export every subscriber's phone number").run()
    assert any("Declined" in m.value for m in at.markdown)


def test_causal_and_mmm_relationships_hold():
    from lab.askdata import parser as P
    from lab.causal import engine as CE
    from lab.mmm import engine as ME

    lvl, trd = CE.panel("level"), CE.panel("trend")
    assert CE.naive(lvl)["Treated vs untreated (after launch)"] > 0          # the naive view gets the sign wrong
    d = CE.did(lvl)
    assert d["lo"] <= CE.TRUE_EFFECT <= d["hi"]                              # DiD covers the truth when trends are parallel
    assert CE.did(trd)["estimate"] > 0                                       # and fails when they are not
    sc = CE.synthetic_control("trend")
    assert sc["effect"] < 0 and abs(sc["effect"] - CE.TRUE_EFFECT) < abs(CE.did(trd)["estimate"] - CE.TRUE_EFFECT)
    m = CE.matching(False)
    assert m["lo"] <= m["truth"] <= m["hi"] and m["naive"] < m["matched"]
    r0 = ME.roi_table(ME.fit(False)).set_index("channel")
    r1 = ME.roi_table(ME.fit(True)).set_index("channel")
    true_social = r1.loc["Paid social", "true_roi"]
    assert abs(r1.loc["Paid social", "roi"] - true_social) < abs(r0.loc["Paid social", "roi"] - true_social)  # calibration helps
    o = ME.optimize(ME.fit(True), float(r1["weekly_spend"].sum()), 0.3)
    assert o["true_optimal"].sum() > o["true_current"].sum()                  # the reallocation really helps
    assert P.evaluate()["exact"].all() and P.evaluate_heldout()["exact"].all()
