"""Read recordings/ (written once by recorder/record.py). Pages only ever read; they never call a model.

If recordings/ is missing or empty, every function returns empty results and the pages say what to run."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# List prices per million tokens (input, output) as published when this lab was built (October 2026). They are only
# used to turn recorded token counts into dollars; check current pricing before quoting them.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00)}
LABELS = {"claude-sonnet-5-5": "Sonnet 5.5", "claude-haiku-5-5": "Haiku 5.5", "claude-opus-5-5": "Opus 5.5"}
COMMAND = "python -m recorder.record"


def folder() -> Path:
    return Path(os.environ.get("RECORDINGS_DIR", ROOT / "recordings"))


def label(model: str) -> str:
    return LABELS.get(model, model)


def cost(model: str, usage: dict) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    return usage.get("input_tokens", 0) / 1e6 * p[0] + usage.get("output_tokens", 0) / 1e6 * p[1]


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    base = Path(path)
    out = {"manifest": {}, "analyst": {}, "docs": {}}
    if not base.exists():
        return out
    m = base / "manifest.json"
    if m.exists():
        out["manifest"] = json.loads(m.read_text())
    for d in sorted((base / "analyst").glob("*")) if (base / "analyst").exists() else []:
        runs = {}
        for f in sorted(d.glob("*.json")):
            tr = json.loads(f.read_text())
            runs.setdefault(tr.get("case", f.stem.split("_")[0]), []).append(tr)
        if runs:
            out["analyst"][d.name] = runs
    for f in sorted((base / "docs").glob("*.json")) if (base / "docs").exists() else []:
        out["docs"][f.stem] = json.loads(f.read_text())
    return out


def load() -> dict:
    return _load(str(folder()))


def analyst_models() -> list[str]:
    return list(load()["analyst"])


def doc_models() -> list[str]:
    return list(load()["docs"])
