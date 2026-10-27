"""The analyst agent: a tool-use loop over the warehouse. Used only by recorder/record.py (and by tests with a scripted
stand-in client). Every step is written to a trace so the site can replay it exactly, without calling a model.

Tools
  list_tables()                      -> table names and row counts
  describe_table(table)              -> columns, types and three sample rows
  run_sql(sql)                       -> up to 200 rows (read-only; refusals come back as tool errors)
  final_answer(status, answer, value, unit, sql, chart)
                                     -> ends the run; status is "answered" or "declined"
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pandas as pd

from lab.ai import warehouse as W

SYSTEM = """You are a careful data analyst for Larkspur Market, a grocery chain. You answer questions using only the
SQL warehouse available through your tools. Rules:
- Look before you query: list tables and describe the ones you need.
- Every number in your answer must come from a query you ran. Never estimate or invent figures.
- The warehouse is read-only. If asked to change, delete or insert data, decline.
- store_contacts holds personal data. Never reveal a manager's name, email or phone number; decline such requests.
- If the data cannot answer the question (other companies, the future, facts not in the tables), decline and say why.
- Revenue is in dollars. Weeks are numbered 1 to 156 starting 2023-01-02; week_start is the Monday of each week.
- "In 2024" (or any year) means the weeks whose week_start falls in that calendar year.
- Finish by calling final_answer exactly once. Put the single headline number in `value` when there is one."""

TOOLS = [
    {"name": "list_tables", "description": "List the warehouse tables and their row counts.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "describe_table", "description": "Columns, types and three sample rows for one table.",
     "input_schema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}},
    {"name": "run_sql", "description": "Run one read-only SQLite SELECT (or WITH ... SELECT). Returns at most 200 rows as JSON.",
     "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "final_answer", "description": "Finish. status='answered' with the answer, or status='declined' with the reason in answer.",
     "input_schema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["answered", "declined"]},
         "answer": {"type": "string", "description": "One to three sentences for a business reader."},
         "value": {"type": ["number", "string", "null"], "description": "The headline number or label, if any."},
         "unit": {"type": ["string", "null"], "description": "e.g. dollars, units, percent, store"},
         "sql": {"type": ["string", "null"], "description": "The query the answer rests on."},
         "chart": {"type": ["object", "null"], "description": "Optional: {type: 'bar'|'line', x: column, y: column} over the last query's rows."},
     }, "required": ["status", "answer"]}},
]

PROMPT_HASH = hashlib.sha256((SYSTEM + json.dumps(TOOLS, sort_keys=True)).encode()).hexdigest()[:12]


def execute(con, name: str, args: dict) -> tuple[str, dict]:
    """Run one tool. Returns (text for the model, structured record for the trace)."""
    try:
        if name == "list_tables":
            rows = [{"table": t, "rows": len(df)} for t, df in W.frames().items()]
            return json.dumps(rows), {"rows": rows}
        if name == "describe_table":
            t = str(args.get("table", ""))
            if t not in W.frames():
                raise W.Refused(f"No table named {t!r}.")
            df = W.frames()[t]
            out = {"columns": [{"name": c, "type": ty} for c, ty in W.SCHEMA[t]], "sample": df.head(3).to_dict("records")}
            return json.dumps(out, default=str), out
        if name == "run_sql":
            df = W.run_sql(con, str(args.get("sql", "")))
            rows = json.loads(df.to_json(orient="records"))
            return json.dumps({"columns": list(df.columns), "rows": rows, "truncated": len(df) == W.MAX_ROWS}), {"columns": list(df.columns), "rows": rows}
        if name == "final_answer":
            return "ok", {}
        raise W.Refused(f"Unknown tool {name}.")
    except W.Refused as e:
        return f"ERROR: {e}", {"error": str(e), "refused": True}
    except Exception as e:  # SQL errors go back to the model so it can correct itself
        return f"ERROR: {type(e).__name__}: {e}", {"error": f"{type(e).__name__}: {e}"}


def _block(b) -> dict:
    return b.model_dump() if hasattr(b, "model_dump") else dict(b)


def run(client, model: str, question: str, max_turns: int = 12, max_tokens: int = 4000) -> dict:
    """One question end to end. `client` is anthropic.Anthropic() or anything with the same messages.create."""
    con = W.connect()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps, final = [], None
    usage = {"input_tokens": 0, "output_tokens": 0}
    t0 = time.perf_counter()
    served = None
    for turn in range(max_turns):
        t1 = time.perf_counter()
        resp = client.messages.create(model=model, max_tokens=max_tokens, system=SYSTEM, tools=TOOLS, messages=messages)
        dt = time.perf_counter() - t1
        served = getattr(resp, "model", served)
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
            usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
        blocks = [_block(b) for b in resp.content]
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in blocks:
            if b.get("type") == "text" and b.get("text", "").strip():
                steps.append({"kind": "text", "turn": turn, "text": b["text"], "seconds": round(dt, 2)})
            elif b.get("type") == "tool_use":
                name, args = b["name"], b.get("input", {}) or {}
                text, rec = execute(con, name, args)
                steps.append({"kind": "tool", "turn": turn, "tool": name, "input": args, "result": rec, "seconds": round(dt, 2)})
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": text, **({"is_error": True} if text.startswith("ERROR") else {})})
                if name == "final_answer":
                    final = args
        if final is not None or getattr(resp, "stop_reason", None) != "tool_use":
            break
        messages.append({"role": "user", "content": results})
    return {"question": question, "model_requested": model, "model_served": served, "final": final, "steps": steps,
            "turns": turn + 1, "usage": usage, "seconds": round(time.perf_counter() - t0, 2), "prompt_hash": PROMPT_HASH}
