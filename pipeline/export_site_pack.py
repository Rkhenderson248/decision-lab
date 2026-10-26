"""Export the Decision platform's production model and governed metrics for richardhenderson.io's "Ask me" assistant.

    python pipeline/export_site_pack.py [out.json]      # default: site/decision-pack.json

Copy the output into the WordPress theme at assets/data/decision-pack.json after a retrain.

It trains exactly what the Streamlit app trains (lab.platform.ml), takes the promoted ridge pipeline apart into its
scaler and coefficients, and writes them with the training medians (for reason codes), the scoring-year prediction
distribution (for ranks), every scored market and HMDA metric aggregates. The browser then scores any scenario with
the same arithmetic the model uses: no server, no model call. Public HMDA aggregates only.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab.platform import metrics, ml, models  # noqa: E402


def build_pack() -> dict:
    """Train exactly what the app trains and return the pack. Raises if the export does not reproduce the model."""

    run = models.build()
    versions, info = ml.train(run.con)
    champ = info["champion"]
    if not hasattr(champ.model, "steps"):
        raise SystemExit(f"champion {champ.name} is not a linear pipeline; nothing to export")
    scaler, ridge = champ.model.steps[0][1], champ.model.steps[-1][1]
    cols = list(ml.FEATURES)

    # Check the export reproduces the model to the last digit before writing it.
    sc = info["score"].copy()
    z = (sc[cols].values - scaler.mean_) / scaler.scale_
    mine = z @ ridge.coef_ + ridge.intercept_
    theirs = champ.model.predict(sc[cols])
    assert np.allclose(mine, theirs, atol=1e-9), "exported weights do not reproduce the model"

    names = pd.read_sql_query("SELECT market_key, market_name FROM gold_markets", run.con)
    sc = sc.merge(names, on="market_key", how="left")
    sc["pred"] = theirs
    sc = sc.sort_values("pred", ascending=False).reset_index(drop=True)
    n = len(sc)
    sc["rank"] = np.arange(1, n + 1)

    med = info["train"][cols].median()
    q = np.quantile(theirs, np.linspace(0, 1, 101))
    te = info["test"]
    p_te = champ.model.predict(te[cols])
    dec = pd.qcut(pd.Series(p_te).rank(method="first"), 10, labels=False) + 1
    by_dec = pd.DataFrame({"d": dec, "y": te[ml.TARGET].values}).groupby("d")["y"].mean()

    def r(v, k=4):
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), k)

    model = {
        "name": champ.name, "version": champ.version, "data_hash": champ.data_hash,
        "decision": "Which markets to prioritize for purchase lending next year",
        "target": f"Change in purchase originations, {ml.SCORE_YEAR} to {ml.SCORE_YEAR + 1}",
        "train_year": ml.TRAIN_YEAR, "test_year": ml.TEST_YEAR, "score_year": ml.SCORE_YEAR,
        "train_rows": champ.train_rows, "min_apps": ml.MIN_APPS,
        "features": [{"key": k, "label": ml.FEATURES[k][0], "lo": ml.FEATURES[k][1], "hi": ml.FEATURES[k][2],
                      "mean": r(scaler.mean_[i], 6), "scale": r(scaler.scale_[i], 6), "coef": r(ridge.coef_[i], 6),
                      "median": r(med[k], 6)} for i, k in enumerate(cols)],
        "intercept": r(ridge.intercept_, 6),
        "gate": {"spearman": r(champ.spearman, 3), "lift": r(champ.lift, 4), "mae": r(champ.mae, 4),
                 "baseline_mae": r(info["baseline_mae"], 4), "rule": "rank corr ≥ 0.10 and top-decile lift ≥ 1 pt"},
        "registry": [{"v": v.version, "name": v.name, "stage": v.stage, "spearman": r(v.spearman, 3),
                      "lift": r(v.lift, 4), "mae": r(v.mae, 4)} for v in versions],
        "deciles": [r(by_dec.get(i), 4) for i in range(1, 11)],
        "test_mean": r(te[ml.TARGET].mean(), 4),
        "quantiles": [r(v, 5) for v in q],
        "scored": n,
        "tiers": [{"min_pct": 90, "label": "Prioritize", "note": "top decile of markets"},
                  {"min_pct": 60, "label": "Invest selectively", "note": "above the median, below the top decile"},
                  {"min_pct": 30, "label": "Maintain", "note": "middle of the pack"},
                  {"min_pct": 0, "label": "Hold", "note": "bottom 30% of markets"}],
    }

    def row(x):
        return [x["market_name"] or str(x["market_key"]), x["state"] or "", 1 if x["area_type"] == "Metro" else 0,
                int(x["applications"]), r(x["growth"], 4), r(x["origination_rate"], 4), r(x["denial_rate"], 4),
                r(x["refinance_share"], 4), int(round(x["avg_loan_amount"])), r(x["pred"], 4)]

    markets = {"cols": ["name", "state", "metro", "apps", "growth", "orig_rate", "denial", "refi", "avg_loan", "pred"],
               "rows": [row(x) for _, x in sc.iterrows()]}

    # Governed metrics, computed the way lab.platform.metrics defines them (summed numerators and denominators).
    def table(by=None, where=""):
        df = metrics.compute(run.con, by=by, where=where)
        return json.loads(df.round(5).to_json(orient="records"))

    latest = int(pd.read_sql_query("SELECT MAX(year) y FROM gold_market_year_features", run.con)["y"][0])
    states = metrics.compute(run.con, by="state", where=f"year = {latest} AND state IS NOT NULL")
    states = states.sort_values("originations", ascending=False)
    met = {
        "defs": [{"key": m.key, "label": m.label, "owner": m.owner, "fmt": m.fmt, "definition": m.definition} for m in metrics.METRICS],
        "latest": latest,
        "by_year": table("year"),
        "by_year_area": table("year, area_type"),
        "by_state": json.loads(states.round(5).to_json(orient="records")),
    }

    return {"generated": date.today().isoformat(), "source": "Public HMDA aggregates (CFPB), Census and FHFA via the decision-lab pipeline",
            "model": model, "markets": markets, "metrics": met}


def main(out: str) -> None:
    pack = build_pack()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pack, separators=(",", ":"), ensure_ascii=False))
    m = pack["model"]
    print(f"wrote {out}: {Path(out).stat().st_size / 1024:.0f} KB, {m['scored']} markets, champion {m['version']} {m['name']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "site" / "decision-pack.json"))
