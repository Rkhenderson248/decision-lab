"""From a plain-English question to a governed intent: rules first, an optional language model second.

The rules handle the common phrasings exactly and predictably; anything they cannot place is declined, never guessed.
With an API key, Claude maps the question to the same intent schema and the result is validated against the
semantic layer before any SQL runs.
"""

from __future__ import annotations

import json
import re

import streamlit as st

from lab.askdata import semantic as S

PII = ["name", "names", "email", "e-mail", "phone number", "phone numbers", "address", "addresses", "ssn", "social security",
       "credit card", "account number", "customer list", "list of customers", "customer ids", "sub_id", "date of birth"]
WRITE = ["delete", "drop", "update", "insert", "alter", "truncate", "remove all"]
UNGOVERNED = {"nps": "Net promoter score", "net promoter": "Net promoter score", "profit": "Profit", "margin": "Margin",
              "satisfaction": "Customer satisfaction", "csat": "Customer satisfaction", "lifetime value": "Lifetime value",
              "clv": "Lifetime value", "ltv": "Lifetime value", "acquisition cost": "Acquisition cost", "cac": "Acquisition cost",
              "marketing spend": "Marketing spend", "ad spend": "Marketing spend", "handle time": "Average handle time", "aht": "Average handle time",
              "employees": "Headcount", "headcount": "Headcount", "weather": "Weather", "stock price": "Stock price"}

VALUE_SYNONYMS = {
    "region": {"texas": "Texas", "tx": "Texas", "pacific": "Pacific", "west coast": "Pacific", "northeast": "Northeast", "north east": "Northeast",
               "great lakes": "Great Lakes", "midwest": "Great Lakes", "southeast": "Southeast", "south east": "Southeast", "mountain": "Mountain"},
    "channel": {"online": "Online", "web": "Online", "digital channel": "Online", "retail": "Retail store", "store": "Retail store", "stores": "Retail store",
                "phone sales": "Phone sales", "telesales": "Phone sales", "partner": "Partner", "partners": "Partner"},
    "plan": {"300 mbps": "300 Mbps", "300mbps": "300 Mbps", "entry plan": "300 Mbps", "basic plan": "300 Mbps", "1 gig": "1 Gig", "1gig": "1 Gig",
             "one gig": "1 Gig", "2 gig": "2 Gig", "2gig": "2 Gig", "multi-gig": "2 Gig", "multi gig": "2 Gig"},
    "segment": {"bundled": "Bundled households", "bundled households": "Bundled households", "promo switchers": "Promo switchers", "promo": "Promo switchers",
                "streamers": "Streamers & gamers", "gamers": "Streamers & gamers", "light users": "Light users", "movers": "Movers & renters",
                "renters": "Movers & renters"},
    "customer_type": {"small business": "Small business", "business customers": "Small business", "b2b": "Small business", "businesses": "Small business",
                      "consumer": "Consumer", "consumers": "Consumer", "residential": "Consumer"},
    "reason": {"outage": "Outage or slow speed", "outages": "Outage or slow speed", "slow speed": "Outage or slow speed", "billing": "Billing question",
               "price increase": "Price increase", "equipment": "Equipment", "router": "Equipment", "moving": "Moving",
               "cancel request": "Cancel request", "cancel requests": "Cancel request", "upgrade": "Upgrade inquiry", "upgrades": "Upgrade inquiry"},
}
GENERIC = {"customers", "subscribers", "how many customers", "how many subscribers", "contacts", "calls"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _has(q: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", q) is not None


def _month_id(year: int, month: int) -> int:
    return (year - S.FIRST_MONTH.year) * 12 + (month - S.FIRST_MONTH.month)


def parse(question: str) -> S.Intent:
    q = " " + question.lower().strip().rstrip("?.!") + " "
    it = S.Intent()
    if any(_has(q, w) for w in WRITE):
        it.refusal = "The copilot is read-only: it can answer questions, never change data."
        return it
    if any(_has(q, w) for w in PII) or re.search(r"\b(which|what|list|show)\b.*\b(customers|subscribers)\b.*\b(are|by name|named)\b", q):
        it.refusal = "Customer-level data (names, contact details, account lists) is not available here; the copilot answers with governed aggregates only."
        return it

    # Metric: the longest matching synonym wins ("churn rate" beats "churn", "contacts per 100" beats "contacts").
    best = None
    for key, m in S.METRICS.items():
        for syn in m.synonyms + [m.label.lower()]:
            score = 1 if syn in GENERIC else len(syn)   # "customers" only counts when nothing more specific matched
            if _has(q, syn) and (best is None or score > best[1]):
                best = (key, score)
    if best is None:
        hit = next((v for k, v in UNGOVERNED.items() if _has(q, k)), None)
        it.refusal = (f"{hit} is not a governed metric yet, so there is no agreed definition to answer from." if hit else
                      "That question does not match a governed metric.") + " Available: " + ", ".join(m.label.lower() for m in S.METRICS.values()) + "."
        return it
    it.metric = best[0]
    m = S.METRICS[it.metric]

    # Filters: explicit values.
    for dim, table in VALUE_SYNONYMS.items():
        for phrase in sorted(table, key=len, reverse=True):
            if _has(q, phrase):
                if dim == "reason" and m.source != "care":
                    continue
                if dim == "customer_type" and phrase == "small business" and _has(q, "segment"):
                    it.filters["segment"] = "Small business"
                    break
                it.filters.setdefault(dim, table[phrase])
                break

    # Dimensions: a dimension word ("by region", "which plan", "segments").
    for dim, d in S.DIMS.items():
        if dim in it.filters:
            continue
        if any(_has(q, syn) for syn in d["synonyms"]):
            it.dims.append(dim)

    # Any "by <something>" must be a governed breakdown or a time grain; otherwise decline rather than quietly ignore it.
    known = {syn for d in S.DIMS.values() for syn in d["synonyms"]} | {"month", "quarter", "year", "plan", "region"}
    for mby in re.finditer(r"\b(?:by|per|across|split by|broken down by)\s+([a-z0-9][a-z0-9 \-]*?)(?=\s+(?:in|for|over|during|since|last|this|from|and)\b|\s*$)", q):
        phrase = mby.group(1).strip()
        words = phrase.split()
        if not phrase or phrase.isdigit() or any(_has(" " + phrase + " ", k) for k in known):
            continue
        if any(phrase.startswith(w) for w in ("100", "subscriber", "customer", "the ")):
            continue
        it.refusal = (f"“{phrase}” is not a governed breakdown, so the copilot will not guess. Breakdowns available for "
                      f"{m.label.lower()}: " + ", ".join(S.DIMS[d]["label"].lower() for d in m.dims) + ", or by month, quarter or year.")
        return it

    # Time grain.
    if re.search(r"\b(by month|monthly|each month|per month|month by month|over time|trend|trending)\b", q):
        it.grain = "month"
    elif re.search(r"\b(by quarter|quarterly|each quarter|per quarter)\b", q):
        it.grain = "quarter"
    elif re.search(r"\b(by year|yearly|annually|each year|per year|year by year)\b", q):
        it.grain = "year"

    # Period.
    last = S.N_MONTHS - 1
    mm = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(month|months|quarter|quarters|year|years)\b", q)
    yr = re.search(r"\b(?:in|during|for)\s+(20\d\d)\b", q) or re.search(r"\b(20\d\d)\b", q)
    since = re.search(r"\bsince\s+([a-z]{3})[a-z]*\s+(20\d\d)\b", q)
    if mm:
        n = int(mm.group(1)) * {"m": 1, "q": 3, "y": 12}[mm.group(2)[0]]
        it.start, it.end = max(0, last - n + 1), last
    elif since:
        it.start, it.end = max(0, _month_id(int(since.group(2)), MONTHS.get(since.group(1), 1))), last
    elif re.search(r"\b(this year|year to date|ytd)\b", q):
        it.start, it.end = max(0, _month_id(2026, 1)), last
    elif yr:
        y = int(yr.group(1))
        it.start, it.end = max(0, _month_id(y, 1)), min(last, _month_id(y, 12))
        if it.start > last or it.end < 0:
            it.refusal = f"The warehouse covers {S.month_label(0)} to {S.month_label(last)}; {y} is outside it."
            return it
    elif re.search(r"\b(last year|past year|last 12 months|past 12 months)\b", q):
        it.start, it.end = last - 11, last
    elif re.search(r"\b(last quarter|latest quarter|this quarter)\b", q):
        it.start, it.end = last - 2, last
    elif re.search(r"\b(last month|latest month|this month|right now|today|currently|current)\b", q):
        it.start, it.end = last, last
    elif re.search(r"\b(all time|ever|since launch|all months|history)\b", q):
        it.start, it.end = 0, last
    elif it.grain == "quarter":
        it.start, it.end = last - 23, last
    elif it.grain == "year":
        it.start, it.end = 0, last

    # Ranking.
    tp = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    if tp:
        it.top = int(tp.group(2))
        it.order = "asc" if tp.group(1) == "bottom" else "desc"
    elif re.search(r"\b(lowest|least|fewest|smallest|bottom)\b", q):
        it.order = "asc"
    elif re.search(r"\b(highest|most|largest|biggest|top|best)\b", q):
        it.order = "desc"

    # Governance: only the breakdowns each metric supports.
    bad = [d for d in it.dims + list(it.filters) if d not in m.dims]
    if bad:
        it.refusal = (f"{m.label} cannot be broken down by {S.DIMS[bad[0]]['label'].lower()}; it is recorded by "
                      + ", ".join(S.DIMS[d]["label"].lower() for d in m.dims) + ".")
        return it
    if len(it.dims) > 2 or (it.grain and len(it.dims) > 1):
        it.notes.append("Kept the first breakdown only, so the chart stays readable.")
        it.dims = it.dims[:1] if it.grain else it.dims[:2]
    return it


SYSTEM = """You translate business questions into a JSON intent for a governed semantic layer. Reply with JSON only.
Schema: {"metric": one of METRICS or null, "group_by": [up to two of DIMS], "time_grain": "month"|"quarter"|"year"|null,
"filters": {dim: value}, "last_n_months": integer 1-36, "top": integer or null, "order": "desc"|"asc"|null, "refusal": string or null}.
Refuse (metric null, refusal set) for anything not in METRICS, any request for customer-level or personal data, or any request to change data.
"""


def model_parse(question: str) -> S.Intent | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        model = st.secrets.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    except Exception:  # noqa: BLE001
        return None
    if not key:
        return None
    try:
        import anthropic

        layer = {"METRICS": {k: m.description for k, m in S.METRICS.items()},
                 "DIMS": {k: d["values"] for k, d in S.DIMS.items()}}
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=300, system=SYSTEM + json.dumps(layer),
            messages=[{"role": "user", "content": question}])
        raw = json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group(0))
        it = S.Intent()
        if raw.get("refusal") or not raw.get("metric"):
            it.refusal = raw.get("refusal") or "Not answerable from the governed metrics."
            return it
        if raw["metric"] not in S.METRICS:
            return None
        it.metric = raw["metric"]
        m = S.METRICS[it.metric]
        it.dims = [d for d in raw.get("group_by") or [] if d in m.dims][:2]
        it.grain = raw.get("time_grain") if raw.get("time_grain") in S.GRAINS else None
        it.filters = {d: v for d, v in (raw.get("filters") or {}).items() if d in m.dims and v in S.DIMS[d]["values"]}
        n = int(raw.get("last_n_months") or 12)
        it.start, it.end = max(0, S.N_MONTHS - n), S.N_MONTHS - 1
        it.top = raw.get("top")
        it.order = raw.get("order")
        return it
    except Exception:  # noqa: BLE001 - fall back to the rules
        return None


# Test set: question -> expected (metric, dims, grain, filters) or "REFUSE".
GOLDEN = [
    ("What is our churn rate by region?", ("churn_rate", ["region"], None, {})),
    ("Monthly churn rate trend for the last 24 months", ("churn_rate", [], "month", {})),
    ("How many new subscribers did we get by channel last year?", ("new_subscribers", ["channel"], None, {})),
    ("Show MRR by quarter", ("mrr", [], "quarter", {})),
    ("Which segment has the highest churn?", ("churn_rate", ["segment"], None, {})),
    ("ARPU by plan for small business customers", ("arpu", ["plan"], None, {"customer_type": "Small business"})),
    ("How many active subscribers do we have in Texas?", ("active_subscribers", [], None, {"region": "Texas"})),
    ("Net adds by month this year", ("net_adds", [], "month", {})),
    ("Top 3 contact reasons", ("care_contacts", ["reason"], None, {})),
    ("Care contacts per 100 subscribers by segment", ("contacts_per_100", ["segment"], None, {})),
    ("How many customers did we lose in 2025?", ("churned_subscribers", [], None, {})),
    ("Gross adds from partners by quarter", ("new_subscribers", [], "quarter", {"channel": "Partner"})),
    ("Churn rate for promo switchers by month", ("churn_rate", [], "month", {"segment": "Promo switchers"})),
    ("Outage calls by region", ("care_contacts", ["region"], None, {"reason": "Outage or slow speed"})),
    ("Average revenue per subscriber in the Pacific region", ("arpu", [], None, {"region": "Pacific"})),
    ("Disconnects by plan in the last 6 months", ("churned_subscribers", ["plan"], None, {})),
    ("Revenue by customer type", ("mrr", ["customer_type"], None, {})),
    ("Which channel brings the most new customers?", ("new_subscribers", ["channel"], None, {})),
    ("What is our NPS by region?", "REFUSE"),
    ("List the names and emails of customers who churned", "REFUSE"),
    ("Delete all churned customers from the table", "REFUSE"),
    ("Care contacts by acquisition channel", "REFUSE"),
    ("What is the profit margin on 2 Gig plans?", "REFUSE"),
    ("How's the weather in Texas?", "REFUSE"),
    ("Churn by zip code", "REFUSE"),   # added after the held-out set caught it
]


@st.cache_data(show_spinner=False)
def evaluate():
    return _eval(GOLDEN)


def describe(metric, dims, grain, filters) -> str:
    parts = [S.METRICS[metric].label.lower() if metric in S.METRICS else str(metric)]
    if dims:
        parts.append("by " + " & ".join(S.DIMS[d]["label"].lower() for d in dims))
    if grain:
        parts.append({"month": "monthly", "quarter": "quarterly", "year": "yearly"}[grain])
    if filters:
        parts.append("; ".join(f"{v}" for v in filters.values()))
    return " · ".join(parts)


def _eval(cases):
    import pandas as pd

    rows = []
    for q, exp in cases:
        it = parse(q)
        if exp == "REFUSE":
            ok_metric = ok_all = it.refusal is not None
            got = "Declined" if it.refusal else describe(it.metric, it.dims, it.grain, it.filters)
        else:
            em, ed, eg, ef = exp
            ok_metric = it.refusal is None and it.metric == em
            ok_all = ok_metric and sorted(it.dims) == sorted(ed) and it.grain == eg and it.filters == ef
            got = "Declined" if it.refusal else describe(it.metric, it.dims, it.grain, it.filters)
        rows.append({"question": q, "expected": "Decline" if exp == "REFUSE" else describe(*exp),
                     "got": got, "metric_ok": ok_metric, "exact": ok_all, "should_refuse": exp == "REFUSE"})
    return pd.DataFrame(rows)


# Written after the rules were tuned on GOLDEN. First run: 11 of 12; the miss ("churn by zip code") was fixed and added to GOLDEN.
HELDOUT_FIRST_RUN = (11, 12)
HELDOUT = [
    ("What does attrition look like across plans?", ("churn_rate", ["plan"], None, {})),
    ("Give me sign ups per quarter from retail stores", ("new_subscribers", [], "quarter", {"channel": "Retail store"})),
    ("How much recurring revenue comes from consumers right now?", ("mrr", [], None, {"customer_type": "Consumer"})),
    ("Cancellation rate in the Great Lakes over time", ("churn_rate", [], "month", {"region": "Great Lakes"})),
    ("Which markets have the lowest average bill?", ("arpu", ["region"], None, {})),
    ("Call volume about billing by segment", ("care_contacts", ["segment"], None, {"reason": "Billing question"})),
    ("Net growth by acquisition channel since Jan 2025", ("net_adds", ["channel"], None, {})),
    ("How many 2 Gig subscribers are active today?", ("active_subscribers", [], None, {"plan": "2 Gig"})),
    ("Contact rate for movers by region", ("contacts_per_100", ["region"], None, {"segment": "Movers & renters"})),
    ("What's our customer acquisition cost by channel?", "REFUSE"),
    ("Export every subscriber's phone number", "REFUSE"),
    ("Churn by zip code", "REFUSE"),
]


@st.cache_data(show_spinner=False)
def evaluate_heldout():
    return _eval(HELDOUT)
