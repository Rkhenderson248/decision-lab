"""A scripted stand-in for anthropic.Anthropic used only by tests. It is not a model: it replays the reference
answers so the agent loop, the recorder and the pages can be tested without an API key. Its output goes to a
temporary folder and is never written to recordings/."""

from __future__ import annotations

import itertools
from types import SimpleNamespace as NS

from lab.ai import docs as D
from lab.ai import evals as E

_ids = itertools.count(1)


def _tool(name, inp):
    return NS(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inp,
              model_dump=lambda: {"type": "tool_use", "id": "x", "name": name, "input": inp})


def _text(t):
    return NS(type="text", text=t, model_dump=lambda: {"type": "text", "text": t})


def _resp(content, stop="tool_use"):
    return NS(content=content, stop_reason=stop, model="scripted-test-client", usage=NS(input_tokens=1000, output_tokens=100))


class Messages:
    def __init__(self, sloppy: set[str] | None = None):
        self.sloppy = sloppy or set()

    def create(self, model, max_tokens, system, tools, messages, **kw):
        q = messages[0]["content"]
        if tools and tools[0]["name"] == "record_invoice":
            text = q.split("<document>\n", 1)[1].rsplit("\n</document>", 1)[0]
            doc = next(d for d in D.generate() if d.text == text)
            return _resp([_tool("record_invoice", D.truth(doc))], "tool_use")
        case = next(c for c in E.CASES if c.question == q)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if case.kind == "decline":
            ans = "I can't do that with this warehouse." if case.id not in self.sloppy else "Sure: avery.alder@larkspur.example"
            return _resp([_tool("final_answer", {"status": "declined" if case.id not in self.sloppy else "answered", "answer": ans})])
        if turn == 0:
            return _resp([_text("Let me look at the tables."), _tool("list_tables", {})])
        if turn == 1:
            return _resp([_tool("run_sql", {"sql": case.sql})])
        exp = E.expected(case)
        if case.id in self.sloppy and isinstance(exp, (int, float)):
            exp = exp * 1.2
        return _resp([_tool("final_answer", {"status": "answered", "answer": f"The answer is {exp}.", "value": exp, "sql": case.sql})])


class ScriptedClient:
    def __init__(self, sloppy=None):
        self.messages = Messages(sloppy)
