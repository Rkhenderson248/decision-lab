"""Record real model runs once, so the site can replay them for free, forever.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...            (Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-...")
    python -m recorder.record                      (everything: about 15-25 minutes, a few dollars)
    python -m recorder.record --only docs          (just the invoices)
    python -m recorder.record --models claude-haiku-5-5 --cases L1,M2,P1

Writes JSON under recordings/ (commit that folder). Re-running skips anything already recorded; delete a file to
record it again. Nothing here runs when a page is viewed. The pages read recordings/ and grade them deterministically.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.ai import agent as A  # noqa: E402
from lab.ai import docs as D  # noqa: E402
from lab.ai import evals as E  # noqa: E402

DEFAULT_MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5"]


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the SDK first: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (your own key; it is never written to disk).")
    return anthropic.Anthropic(max_retries=4), anthropic.__version__


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def record_analyst(client, model: str, out: Path, cases: list[E.Case], repeats: int) -> dict:
    tok = {"input_tokens": 0, "output_tokens": 0}
    for c in cases:
        for r in range(1, repeats + 1):
            f = out / "analyst" / model / (f"{c.id}.json" if r == 1 else f"{c.id}_r{r}.json")
            if f.exists():
                continue
            t0 = time.perf_counter()
            try:
                tr = A.run(client, model, c.question)
            except Exception as e:  # keep going; the failure itself is recorded and graded as a failure
                tr = {"question": c.question, "model_requested": model, "final": None, "steps": [], "error": f"{type(e).__name__}: {e}",
                      "usage": {"input_tokens": 0, "output_tokens": 0}, "seconds": round(time.perf_counter() - t0, 2), "prompt_hash": A.PROMPT_HASH}
            tr.update({"case": c.id, "repeat": r, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            _write(f, tr)
            for k in tok:
                tok[k] += tr["usage"][k]
            g = E.grade(c, tr)
            print(f"  {model}  {c.id:<3} {'PASS' if g['pass'] else 'FAIL'}  {g['why']:<40} {tr['seconds']:>6.1f}s  {sum(tr['usage'].values()):>7,} tok")
    return tok


def _extract(client, model: str, text: str) -> tuple[dict | None, dict, float, str | None]:
    t0 = time.perf_counter()
    resp = client.messages.create(model=model, max_tokens=2000, system=D.EXTRACT_SYSTEM, tools=[D.EXTRACT_TOOL],
                                  messages=[{"role": "user", "content": "Extract this invoice by calling record_invoice once.\n\n<document>\n" + text + "\n</document>"}])
    dt = time.perf_counter() - t0
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    for b in resp.content:
        if getattr(b, "type", "") == "tool_use" and b.name == "record_invoice":
            return dict(b.input), usage, dt, getattr(resp, "model", None)
    return None, usage, dt, getattr(resp, "model", None)


def record_docs(client, model: str, out: Path) -> dict:
    f = out / "docs" / f"{model}.json"
    rec = json.loads(f.read_text()) if f.exists() else {"model_requested": model, "prompt_hash": D.PROMPT_HASH, "docs": {}}
    tok = {"input_tokens": 0, "output_tokens": 0}
    for d in D.generate():
        if d.id in rec["docs"]:
            continue
        try:
            ext, usage, dt, served = _extract(client, model, d.text)
            rec["docs"][d.id] = {"extraction": ext, "usage": usage, "seconds": round(dt, 2)}
            rec["model_served"] = served
        except Exception as e:
            rec["docs"][d.id] = {"extraction": None, "error": f"{type(e).__name__}: {e}", "usage": {"input_tokens": 0, "output_tokens": 0}, "seconds": 0}
        _write(f, rec)
        u = rec["docs"][d.id]["usage"]
        for k in tok:
            tok[k] += u[k]
        ok = sum(D.field_ok(k, (rec["docs"][d.id]["extraction"] or {}).get(k), D.truth(d)[k]) for k in D.FIELDS)
        print(f"  {model}  {d.id} {d.layout:<9} {ok}/10 fields  {rec['docs'][d.id]['seconds']:>5.1f}s")
    rec["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write(f, rec)
    return tok


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model IDs")
    p.add_argument("--only", choices=["analyst", "docs"], help="record one part only")
    p.add_argument("--cases", help="comma-separated case IDs (default: all 30)")
    p.add_argument("--repeats", type=int, default=1, help="runs per analyst case (adds consistency to the eval page)")
    p.add_argument("--out", default=str(ROOT / "recordings"))
    a = p.parse_args(argv)
    client, sdk = _client()
    out = Path(a.out)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    cases = [c for c in E.CASES if not a.cases or c.id in a.cases.split(",")]
    totals = {}
    for m in models:
        print(f"\n== {m}")
        t = {"input_tokens": 0, "output_tokens": 0}
        if a.only in (None, "analyst"):
            for k, v in record_analyst(client, m, out, cases, a.repeats).items():
                t[k] += v
        if a.only in (None, "docs"):
            for k, v in record_docs(client, m, out).items():
                t[k] += v
        totals[m] = t
    man = out / "manifest.json"
    meta = json.loads(man.read_text()) if man.exists() else {}
    meta.update({"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sdk": f"anthropic {sdk}",
                 "python": platform.python_version(), "models": sorted(set(meta.get("models", [])) | set(models)),
                 "analyst_prompt_hash": A.PROMPT_HASH, "docs_prompt_hash": D.PROMPT_HASH})
    _write(man, meta)
    print("\nTokens this run:", json.dumps(totals))
    print(f"Done. Commit the folder: git add recordings && git commit -m 'Record model runs' && git push")


if __name__ == "__main__":
    main()
