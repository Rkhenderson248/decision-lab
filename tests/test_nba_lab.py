"""Next-best-action lab: the simulation, the policy, the off-policy estimates and the page."""

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from lab.nba import data as D
from lab.nba import models as M


@pytest.fixture(scope="module")
def study():
    return M.run()


def test_deterministic():
    assert D.simulate().hash == D.simulate().hash


def test_propensities_are_probabilities():
    log = D.simulate().log
    assert log.propensity.between(D.EPS / len(D.ACTIONS) - 1e-12, 1).all()
    assert set(log.action_id.unique()) == set(range(len(D.ACTIONS)))     # exploration reached every action


def test_constraints_hold(study):
    for name in ["Model, within limits", "Shipped (with guardrails)"]:
        a = study.policies[name].action
        for k, cap in M.CAPACITY.items():
            assert (a == D.ACTIONS.index(k)).sum() <= cap
        assert (a > 0).mean() <= M.CONTACT_CAP + 1e-9


def test_new_policy_truly_beats_rules(study):
    truth = study.ope.groupby("policy").truth.first()
    assert truth["Shipped (with guardrails)"] > truth["Current rules"] + 2
    assert truth["Model, within limits"] <= truth["Model, no limits"] + 1e-9


def test_dr_gain_interval_excludes_zero(study):
    est, lo, hi = study.gain
    truth = study.ope.groupby("policy").truth.first()
    real = truth["Shipped (with guardrails)"] - truth["Current rules"]
    assert lo > 0 and lo <= real <= hi


def test_dr_closer_than_direct_method(study):
    o = study.ope.assign(err=lambda d: (d.est - d.truth).abs())
    e = o.groupby("estimator").err.mean()
    assert e["Doubly robust"] < e["Direct method"]


def test_guardrail_holds_back_lapsing(study):
    sg = study.segments.set_index("segment")
    assert sg.loc["Lapsing", "fallback"] and not sg.loc["Core", "fallback"]
    lapsing = M.segment_of(study.world.customers) == "Lapsing"
    rules = study.world.log.rule_action.values
    assert (study.policies["Shipped (with guardrails)"].action[lapsing] == rules[lapsing]).all()


def test_sample_size_monotone():
    assert M.sample_size(30, 1) > M.sample_size(30, 2) > 0


@pytest.mark.parametrize("stage", ["the-log", "response-model", "choose-within-limits", "judge-before-launch", "guardrails", "roll-out"])
def test_stage_renders(stage):
    at = AppTest.from_file("../views/nba_lab.py", default_timeout=120)
    at.query_params["stage"] = stage
    at.run()
    assert not at.exception, at.exception
