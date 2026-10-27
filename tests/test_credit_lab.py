"""Credit risk & loss lab: every stage renders, and the statistics the audit stage relies on hold."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from lab.credit import data as D
from lab.credit import loss as X
from lab.credit import models as M

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["portfolio", "pd-model", "loss-reserve", "scenarios", "backtest", "audit-defense", "ownership"]


@pytest.fixture(scope="module")
def built():
    book = D.simulate()
    champ, bench, (dev, oot, bk) = M.train(book.loans)
    return book, champ, bench, dev, oot


@pytest.mark.parametrize("stage", STAGES)
def test_stage_renders(stage):
    at = AppTest.from_file(str(ROOT / "views/credit_lab.py"), default_timeout=120)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception


def test_deterministic(built):
    assert D.simulate().hash == built[0].hash


def test_sql_matches_pandas(built):
    book = built[0]
    con = D.warehouse(book)
    assert con.execute("SELECT COUNT(*) FROM loans").fetchone()[0] == len(book.loans)
    roll = con.execute(D.SQL_ROLL).fetchall()
    shares = {}
    for frm, _, _, share in roll:
        shares[frm] = shares.get(frm, 0) + share
    assert all(abs(v - 1) < 1e-3 for v in shares.values())


def test_champion_beats_benchmark_out_of_time(built):
    _, champ, bench, _, _ = built
    assert champ.auc_oot > bench.auc_oot >= 0.75


def test_monotone_constraints_hold(built):
    _, champ, _, dev, _ = built
    assert M.monotone_check(champ, dev.sample(600, random_state=0))["Holds at every step"].all()


def test_overlay_fixes_calibration(built):
    _, champ, _, dev, oot = built
    ov = M.macro_overlay(champ, dev)
    adj = ov.apply(champ.p_oot, M.forward_du(oot))
    assert ov.beta_du > 0
    assert M.hosmer_lemeshow(oot.default_12m.values, adj)[1] > champ.hl_p
    assert not (M.backtest_grades(oot.default_12m.values, adj)["Traffic light"] == "Red").any()


def test_reserve_orders_and_scales(built):
    book, champ, _, _, _ = built
    lgd = X.fit_lgd(book.loans)
    assert lgd.mae < lgd.mae_base
    curve = X.marginal_curve(book.loans, book.perf)
    beta, _ = X.macro_beta(book.loans, book.perf)
    act = M.frame(book.loans[book.loans.end_state == "Active"])
    p = champ.model.predict_proba(act[M.COLS])[:, 1]
    r = X.reserve(act, p, lgd.predict(act), curve, beta)
    assert 0 < r["Baseline"] < r["Adverse"] < r["Severe"]
    e = X.ecl(act, p, lgd.predict(act), curve, beta)
    assert np.all(e.ecl <= e.ead_now + 1e-6) and 0.005 < e.ecl.sum() / e.ead_now.sum() < 0.15
