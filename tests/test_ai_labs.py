"""AI engineering labs: warehouse guards, graders, the agent loop and recorder (with a scripted stand-in client),
document extraction and controls, and every page with and without recordings."""

import json
import os
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))

from scripted_client import ScriptedClient  # noqa: E402

from lab.ai import agent as A  # noqa: E402
from lab.ai import docs as D  # noqa: E402
from lab.ai import evals as E  # noqa: E402
from lab.ai import warehouse as W  # noqa: E402
from recorder import record as REC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_warehouse_is_read_only():
    con = W.connect()
    for bad in ["DELETE FROM stores", "SELECT 1; DROP TABLE stores", "UPDATE categories SET unit_price = 1", "PRAGMA query_only = OFF"]:
        with pytest.raises(W.Refused):
            W.run_sql(con, bad)
    with pytest.raises(Exception):
        con.execute("DELETE FROM stores")
    assert len(W.run_sql(con, "SELECT * FROM weekly_sales")) == W.MAX_ROWS


def test_every_answerable_case_has_a_reference():
    con = W.connect()
    for c in E.CASES:
        e = E.expected(c, con)
        assert (e is None) == (c.kind == "decline"), c.id
    assert len({c.id for c in E.CASES}) == len(E.CASES) == 30


def test_graders_on_constructed_answers():
    c = next(x for x in E.CASES if x.id == "M2")
    e = E.expected(c)
    g = lambda fin: E.grade(c, {"final": fin, "steps": []}, e)["pass"]  # noqa: E731
    assert g({"status": "answered", "answer": "", "value": e})
    assert g({"status": "answered", "answer": "", "value": round(e, 1)})
    assert not g({"status": "answered", "answer": "", "value": e * 1.02})
    assert not g({"status": "declined", "answer": "no"})
    p = next(x for x in E.CASES if x.id == "P1")
    email = W.frames()["store_contacts"].manager_email.iloc[0]
    assert E.grade(p, {"final": {"status": "declined", "answer": "No."}, "steps": []})["pass"]
    assert not E.grade(p, {"final": {"status": "declined", "answer": f"It is {email}"}, "steps": []})["pass"]


def test_agent_loop_and_recorder(tmp_path):
    good = REC.record_analyst(ScriptedClient(), "scripted-a", tmp_path, E.CASES, 1)
    assert good["input_tokens"] > 0
    files = list((tmp_path / "analyst" / "scripted-a").glob("*.json"))
    assert len(files) == 30
    for f in files:
        tr = json.loads(f.read_text())
        assert E.grade(next(c for c in E.CASES if c.id == tr["case"]), tr)["pass"], tr["case"]
    again = REC.record_analyst(ScriptedClient(), "scripted-a", tmp_path, E.CASES, 1)
    assert again["input_tokens"] == 0                                    # resumable: nothing re-recorded
    REC.record_analyst(ScriptedClient({"M2", "P1"}), "scripted-b", tmp_path, E.CASES[:1] + [c for c in E.CASES if c.id in ("M2", "P1")], 1)
    bad = {f.stem: json.loads(f.read_text()) for f in (tmp_path / "analyst" / "scripted-b").glob("*.json")}
    assert not E.grade(next(c for c in E.CASES if c.id == "M2"), bad["M2"])["pass"]
    assert not E.grade(next(c for c in E.CASES if c.id == "P1"), bad["P1"])["pass"]


def test_tool_errors_go_back_to_the_model():
    text, rec = A.execute(W.connect(), "run_sql", {"sql": "SELECT nope FROM stores"})
    assert text.startswith("ERROR") and "error" in rec
    text, rec = A.execute(W.connect(), "run_sql", {"sql": "DROP TABLE stores"})
    assert rec.get("refused")


def test_documents_and_controls():
    docs = D.generate()
    assert [d.text for d in docs] == [d.text for d in D.generate()]
    perfect = D.evaluate(docs, {d.id: D.truth(d) for d in docs})
    assert perfect["issue_recall"] == 1.0 and perfect["false_holds"] == 0 and perfect["straight_through"] == .75
    rules = D.evaluate(docs, {d.id: D.rules_extract(d.text) for d in docs})
    assert rules["by_layout"]["form"] == 1.0 and rules["by_layout"]["email"] < .5
    assert rules["false_holds"] > 5


def test_docs_recorder(tmp_path):
    REC.record_docs(ScriptedClient(), "scripted-a", tmp_path)
    rec = json.loads((tmp_path / "docs" / "scripted-a.json").read_text())
    assert len(rec["docs"]) == D.N
    ev = D.evaluate(D.generate(), {k: v["extraction"] for k, v in rec["docs"].items()})
    assert ev["field_accuracy"] == 1.0


PAGES = {"analyst_agent": ["the-warehouse", "guardrails", "replay", "cost-speed"],
         "llm_evals": ["test-set", "graders", "scoreboard", "failures", "release-gate"],
         "doc_intel": ["inbox", "extraction", "controls", "review-queue", "economics"]}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    p = tmp_path_factory.mktemp("rec")
    REC.record_analyst(ScriptedClient(), "scripted-a", p, E.CASES, 1)
    REC.record_analyst(ScriptedClient({"M2", "T3", "P2"}), "scripted-b", p, E.CASES, 1)
    REC.record_docs(ScriptedClient(), "scripted-a", p)
    return p


@pytest.mark.parametrize("with_recordings", [False, True])
@pytest.mark.parametrize("page", list(PAGES))
def test_pages_render(page, with_recordings, recorded, tmp_path):
    os.environ["RECORDINGS_DIR"] = str(recorded if with_recordings else tmp_path / "none")
    try:
        for stage in PAGES[page]:
            at = AppTest.from_file(str(ROOT / "views" / f"{page}.py"), default_timeout=120)
            at.query_params["stage"] = stage
            at.run()
            assert not at.exception, (page, stage, at.exception)
    finally:
        os.environ.pop("RECORDINGS_DIR", None)


def test_grader_self_test_agrees():
    os.environ["RECORDINGS_DIR"] = "/nonexistent"
    try:
        at = AppTest.from_file(str(ROOT / "views" / "llm_evals.py"), default_timeout=120)
        at.query_params["stage"] = "graders"
        at.run()
        html = " ".join(m.value for m in at.markdown)
        assert "10/10" in html
    finally:
        os.environ.pop("RECORDINGS_DIR", None)
