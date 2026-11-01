"""Lab DNA: what each lab is made of, measured from its code.

Every lab's own code (its page plus the lab modules it imports, transitively; shared site chrome such as theme,
widgets and layout helpers is left out) is cut into blocks: each function, and each statement inside a page's
top-level stage blocks. Each block is scored against a fixed keyword list per discipline and all of its lines go
to the best-scoring discipline. Shares are line-weighted. Comments, docstrings and blank lines do not count.
Deterministic: the same code always gives the same DNA.

    python scripts/lab_dna.py            # prints a table and writes data/lab_dna.json
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GROUPS = ["Data engineering", "Math & statistics", "Machine learning", "AI & language", "Decision science",
          "Visualization & design", "Storytelling", "BI engineering", "Product engineering", "MLOps & governance"]

# Keyword patterns per discipline. A block scores one point per match; ties go to the earlier group in GROUPS.
PATTERNS = {
    "Data engineering": r"read_csv|read_sql|to_sql|sqlite|(?-i:\bSELECT\b|\bJOIN\b|\bFROM\b|\bWHERE\b|GROUP BY)|groupby|\.merge\(|pivot|melt|concat|dropna|fillna|json\.load|requests\.|urlopen|download|ingest|schema|parquet|\bcon\.execute|default_rng|rng\.|simulate|to_datetime|resample|warehouse|pipeline|etl|dedup",
    "Math & statistics": r"np\.(?:mean|std|var|log|exp|sqrt|percentile|quantile|cov|corrcoef|polyfit|cumsum|diff)|scipy\.stats|stats\.|norm\.|\.ppf|binom|chi2|regress|\bols\b|lstsq|bootstrap|confidence|interval|elasticit|hazard|surviv|conformal|p_value|pvalue|hosmer|decompos|shapley|wilson|variance|std\(|percentile|quantile|seasonal|holt|smoothing|reconcil|wape|mape|mase|correlat|\bsd\b|z_?score",
    "Machine learning": r"sklearn|xgboost|XGB|LogisticRegression|RandomForest|GradientBoosting|HistGradientBoosting|\.fit\(|predict_proba|\.predict\(|roc_auc|\bauc\b|calibrat|cross_val|train_test|shap|permutation_importance|feature_importance|uplift|KMeans|cluster|PCA|SVC|LinearRegression|isotonic|n_estimators|max_depth|learning_rate|classifier|regressor|monotone",
    "AI & language": r"anthropic|claude|\bllm\b|prompt|embedding|retriev|tfidf|Tfidf|bm25|cosine|(?<!refinance )intent|synonym|tokeni[sz]|parse_question|grounded|refus|hallucinat|natural language|citation|semantic layer|grounded",
    "Decision science": r"linprog|minimize|optimi[sz]|argmax|shadow|policy|threshold|expected_value|expected value|profit|budget|allocat|newsvendor|critical ratio|\bips\b|doubly|off.?policy|causal|diff.?in.?diff|synthetic control|matching|propensity|counterfactual|scenario|sensitivity|tornado|cutoff|frontier|marginal|roi\b|payback|price_opt|capacity|knapsack|greedy",
    "Visualization & design": r"\bgo\.|px\.|add_trace|add_scatter|add_bar|add_hline|add_vline|add_vrect|add_hrect|Heatmap|update_layout|update_xaxes|update_yaxes|t\.chart|st\.plotly|marker=|line=dict|colorscale|svg|<div|<span|css|style=",
    "Storytelling": r"t\.header|ui\.call|cui\.call|stage_head|t\.insight|t\.footnote|ui\.cards|cards\(|narrat|headline|st\.caption|st\.markdown\(f?[\"']#|member_note",
    "BI engineering": r"lvdash|databricks|power ?bi|tableau|dax|semantic layer|dashboard|kpi|metric_def|measure",
    "Product engineering": r"st\.(button|selectbox|slider|radio|columns|tabs|expander|number_input|text_input|toggle|checkbox|form|dataframe|metric|download_button|chat_input|chat_message|data_editor|select_slider|multiselect)|session_state|query_params|w\.segmented|_go\(|PG\.bar|cache_resource|cache_data|st\.rerun|st\.tabs",
    "MLOps & governance": r"monitor|drift|\bpsi\b|mlflow|registry|lineage|audit|validat|test plan|model card|governance|fairness|disparate|adverse action|reason code|guardrail|backtest|freshness|alert|\bsla\b|rate.?limit|champion|challenger|retrain|inventory|overlay|override",
}
RX = {g: re.compile(p, re.I) for g, p in PATTERNS.items()}

OVERRIDE = {"lab/policy/manual.py": "AI & language"}
# Each lab counts only its own modules; another lab's package it borrows data from is left out.
OWN = {
    "lending-lab": ["lab/cu/", "lab/bench"], "subscriber-lab": ["lab/sub/", "lab/bench"], "credit-risk": ["lab/credit/"],
    "forecasting": ["lab/forecast/"], "next-best-action": ["lab/nba/"], "analytics-copilot": ["lab/askdata/"],
    "policy-assistant": ["lab/policy/"], "decision-platform": ["lab/platform/"], "pricing-copilot": ["lab/copilot", "lab/demand"],
    "causal-impact": ["lab/causal/", "lab/bench"], "marketing-mix": ["lab/mmm/"], "market-intelligence": ["lab/market_", "lab/ask.py", "pipeline/"],
    "product-strategy": ["lab/strategy_engine"], "lending": ["lab/view_", "lab/products", "lab/pipeline_model"],
}
SHARED = {"lab.theme", "lab.widgets", "lab.progress", "lab.cu.ui", "lab.ratelimit", "lab.sub.ui"}

LABS = {
    "lending-lab": "views/lending_lab.py", "subscriber-lab": "views/subscriber_lab.py", "credit-risk": "views/credit_lab.py",
    "forecasting": "views/forecast_lab.py", "next-best-action": "views/nba_lab.py", "analytics-copilot": "views/analytics_copilot.py",
    "policy-assistant": "views/policy_assistant.py", "decision-platform": "views/decision_platform.py", "pricing-copilot": "views/copilot.py",
    "causal-impact": "views/causal_lab.py", "marketing-mix": "views/mmm_lab.py", "market-intelligence": "views/market_intel.py",
    "product-strategy": "views/product_strategy.py", "lending": "views/lending.py",
}
EXTRA = {"market-intelligence": ["pipeline/build_markets.py", "pipeline/databricks_export.py"]}


def module_path(mod: str) -> Path | None:
    p = ROOT / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = ROOT / mod.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def imports(path: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("lab"):
            for a in n.names:
                sub = f"{n.module}.{a.name}"
                out.add(sub if module_path(sub) else n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("lab"):
                    out.add(a.name)
    return {m for m in out if m not in SHARED and m != "lab"}


def files_for(view: str, slug: str) -> list[Path]:
    seen, stack = [], [ROOT / view] + [ROOT / e for e in EXTRA.get(slug, [])]
    while stack:
        p = stack.pop()
        if p in seen or not p.exists():
            continue
        seen.append(p)
        for m in sorted(imports(p)):
            mp = module_path(m)
            if mp and any(str(mp.relative_to(ROOT)).startswith(o) for o in OWN.get(slug, [])):
                stack.append(mp)
    return seen


def code_lines(src: str) -> set[int]:
    """Line numbers that carry code: not blank, not comments, not docstrings."""
    lines = src.split("\n")
    keep = {i + 1 for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("#")}
    tree = ast.parse(src)
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            keep -= set(range(body[0].lineno, body[0].end_lineno + 1))
    return keep


def blocks(src: str) -> list[tuple[int, int]]:
    """Functions whole; inside top-level if/with/for blocks, each statement; other top-level statements whole."""
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.If, ast.With, ast.For)) and (n.end_lineno - n.lineno) > 12:
            def walk(stmts):
                for s in stmts:
                    if isinstance(s, (ast.If, ast.With, ast.For)) and (s.end_lineno - s.lineno) > 12:
                        walk(s.body)
                        walk(getattr(s, "orelse", []) or [])
                    else:
                        out.append((s.lineno, s.end_lineno))
            walk([n])
        else:
            out.append((n.lineno, n.end_lineno))
    return out


def classify(text: str) -> str:
    scores = {g: len(RX[g].findall(text)) for g in GROUPS}
    best = max(GROUPS, key=lambda g: (scores[g], -GROUPS.index(g)))
    return best if scores[best] else ""


def dna(slug: str, view: str) -> dict:
    counts, detail = Counter(), defaultdict(Counter)
    files = files_for(view, slug)
    for p in files:
        src = p.read_text()
        lines = src.split("\n")
        keep = code_lines(src)
        rel = str(p.relative_to(ROOT))
        default = "Product engineering" if p.parts[-2] == "views" else "Data engineering" if re.search(r"data|sample|pipeline|build_", p.name) else "Math & statistics"
        for a, b in blocks(src):
            span = [i for i in range(a, b + 1) if i in keep]
            if not span:
                continue
            text = "\n".join(lines[i - 1] for i in span)
            g = OVERRIDE.get(rel) or classify(text) or default
            counts[g] += len(span)
            for kw in RX[g].findall(text):
                detail[g][kw.lower() if isinstance(kw, str) else kw[0].lower()] += 1
    total = sum(counts.values()) or 1
    return {"slug": slug, "lines": total, "files": [str(p.relative_to(ROOT)) for p in files],
            "share": {g: round(counts[g] / total * 100, 1) for g in GROUPS},
            "signals": {g: [k for k, _ in detail[g].most_common(8)] for g in GROUPS if counts[g]}}


def main() -> None:
    res = {slug: dna(slug, v) for slug, v in LABS.items()}
    out = ROOT / "data" / "lab_dna.json"
    out.write_text(json.dumps({"method": __doc__.strip().split("\n\n")[1], "groups": GROUPS, "labs": res}, indent=1))
    w = max(len(s) for s in res)
    print(" " * (w + 7) + "  ".join(g.split()[0][:6].rjust(6) for g in GROUPS))
    for s, r in res.items():
        print(f"{s:<{w}} {r['lines']:>5}  " + "  ".join(f"{r['share'][g]:>6.1f}" for g in GROUPS))
    print(f"\nwrote {out.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
