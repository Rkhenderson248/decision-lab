"""Document intelligence: 40 synthetic supplier invoices sent to Larkspur Market, with the true fields known.

Four layouts (a clean form, a two-column statement, a forwarded email in prose, and a faxed copy with scanning
errors) and planted problems: totals that do not add up, missing purchase orders, a due date before the invoice
date, and duplicates of earlier invoices. Extraction pulls ten fields; deterministic checks decide whether a document
can be paid straight through or goes to a person. Two extractors are compared: a rules baseline (regular expressions,
shipped and run live) and a model with a forced JSON schema (recorded once by recorder/record.py, replayed here).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

SEED = 2611
N = 40
VENDORS = ["Northfield Dairy Co-op", "Bramble & Rye Bakers", "Coldwater Frozen Foods", "Sunvale Produce Partners",
           "Tidewell Beverage Supply", "Hearthstone Household Goods", "Copperleaf Packaging", "Greenmeadow Growers"]
FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date", "po_number", "currency", "subtotal", "tax", "total", "line_count"]
ITEMS = ["cases whole milk 1gal", "cases greek yogurt", "sourdough loaves", "frozen peas 2lb", "sparkling water 12pk",
         "paper towels 6pk", "apples gala 40lb", "bananas 40lb", "shrink film rolls", "cold brew 6pk", "dish soap 24oz", "baguettes"]
LAYOUTS = ["form", "statement", "email", "fax"]


@dataclass
class Doc:
    id: str
    layout: str
    text: str
    truth: dict
    issues: list = field(default_factory=list)   # planted: mismatch, missing_po, due_before_issue, duplicate


def _fmt_date(d: date, style: int) -> str:
    return [d.isoformat(), d.strftime("%m/%d/%Y"), d.strftime("%B %d, %Y").replace(" 0", " "), d.strftime("%d %b %Y")][style]


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _fax(text: str, rng) -> str:
    """Scanning noise: some O for 0, l for 1, S for 5 inside words and numbers, and broken spacing."""
    sub = {"0": "O", "1": "l", "5": "S"}
    out = []
    for ch in text:
        if ch in sub and rng.random() < .07:
            out.append(sub[ch])
        elif ch == " " and rng.random() < .03:
            out.append("  ")
        else:
            out.append(ch)
    return "".join(out)


def generate(seed: int = SEED, n: int = N) -> list[Doc]:
    rng = np.random.default_rng(seed)
    docs: list[Doc] = []
    issue_plan = {3: ["mismatch"], 9: ["missing_po"], 12: ["mismatch"], 17: ["due_before_issue"], 21: ["missing_po"],
                  24: ["duplicate"], 28: ["mismatch"], 31: ["missing_po", "mismatch"], 35: ["duplicate"], 38: ["due_before_issue"]}
    for i in range(n):
        layout = LAYOUTS[i % 4]
        issues = issue_plan.get(i, [])
        if "duplicate" in issues:
            src = docs[{24: 5, 35: 14}.get(i, 0)]
            t = dict(src.truth)
            d = Doc(f"INV-{i + 1:02d}", layout, "", t, ["duplicate"])
        else:
            v = VENDORS[int(rng.integers(len(VENDORS)))]
            issued = date(2025, 9, 1) + timedelta(days=int(rng.integers(0, 50)))
            terms = int(rng.choice([15, 30, 45]))
            due = issued - timedelta(days=int(rng.integers(3, 10))) if "due_before_issue" in issues else issued + timedelta(days=terms)
            nl = int(rng.integers(2, 6))
            lines = []
            for _ in range(nl):
                q = int(rng.integers(2, 40))
                p = round(float(rng.uniform(3, 60)), 2)
                lines.append((ITEMS[int(rng.integers(len(ITEMS)))], q, p, round(q * p, 2)))
            sub = round(sum(x[3] for x in lines), 2)
            tax = round(sub * (0.0825 if v != "Sunvale Produce Partners" else 0.0), 2)
            total = round(sub + tax, 2)
            stated_total = round(total + float(rng.choice([-1, 1])) * float(rng.choice([10, 25.5, 100])), 2) if "mismatch" in issues else total
            po = None if "missing_po" in issues else f"PO-{int(rng.integers(40000, 49999))}"
            t = {"vendor_name": v, "invoice_number": f"{v.split()[0][:3].upper()}-{int(rng.integers(10000, 99999))}",
                 "invoice_date": issued.isoformat(), "due_date": due.isoformat(), "po_number": po, "currency": "USD",
                 "subtotal": sub, "tax": tax, "total": stated_total, "line_count": nl, "_lines": lines, "_terms": terms}
            d = Doc(f"INV-{i + 1:02d}", layout, "", t, issues)
        d.text = render(d, rng)
        docs.append(d)
    return docs


def render(d: Doc, rng) -> str:
    t = d.truth
    ds = int(rng.integers(4))
    lines = t["_lines"]
    if d.layout in ("form", "fax"):
        body = [f"{t['vendor_name'].upper()}", "INVOICE", "", f"Invoice No: {t['invoice_number']}", f"Invoice Date: {_fmt_date(date.fromisoformat(t['invoice_date']), ds)}",
                f"Due Date: {_fmt_date(date.fromisoformat(t['due_date']), ds)}", f"PO Number: {t['po_number'] or ''}", "Bill To: Larkspur Market, Accounts Payable", "",
                "QTY  DESCRIPTION                     UNIT      AMOUNT"]
        body += [f"{q:>3}  {name:<30}  {_money(p):>8}  {_money(a):>10}" for name, q, p, a in lines]
        body += ["", f"Subtotal: ${_money(t['subtotal'])}", f"Sales tax: ${_money(t['tax'])}", f"TOTAL DUE (USD): ${_money(t['total'])}"]
    elif d.layout == "statement":
        body = [f"{t['vendor_name']}  |  Statement of charges", f"Ref {t['invoice_number']}    Dated {_fmt_date(date.fromisoformat(t['invoice_date']), ds)}",
                f"Customer: Larkspur Market    Your order: {t['po_number'] or 'n/a'}", f"Terms: Net {t['_terms']}  (pay by {_fmt_date(date.fromisoformat(t['due_date']), ds)})", ""]
        body += [f"- {name} x{q} @ {_money(p)} = {_money(a)}" for name, q, p, a in lines]
        body += ["", f"Net amount {_money(t['subtotal'])}  +  tax {_money(t['tax'])}", f"Amount payable USD {_money(t['total'])}"]
    else:
        items = "; ".join(f"{q} {name} at ${_money(p)}" for name, q, p, _ in lines)
        po = f"against your PO {t['po_number']}" if t["po_number"] else "(we didn't get a PO number for this one)"
        body = ["From: billing@" + re.sub(r"[^a-z]", "", t["vendor_name"].lower())[:14] + ".example", "To: ap@larkspur.example",
                f"Subject: Invoice {t['invoice_number']}", "",
                "Hi team,", "",
                f"Please find our invoice {t['invoice_number']} dated {_fmt_date(date.fromisoformat(t['invoice_date']), ds)} {po}. "
                f"This covers {items}. That comes to ${_money(t['subtotal'])} before tax; tax is ${_money(t['tax'])}, "
                f"so the total is ${_money(t['total'])} USD, due {_fmt_date(date.fromisoformat(t['due_date']), ds)}.",
                "", "Thanks,", f"Accounts receivable, {t['vendor_name']}"]
    text = "\n".join(body)
    if d.layout == "fax":
        text = "*** FAX RECEIVED ***  PAGE 1/1\n" + _fax(text, rng)
    return text


def truth(d: Doc) -> dict:
    return {k: d.truth[k] for k in FIELDS}


# ---------------------------------------------------------------------------------------------------------------
# Rules baseline: what a team writes before it has a model. Run live; deterministic.
# ---------------------------------------------------------------------------------------------------------------
_DATE = r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{2} [A-Z][a-z]{2} \d{4})"


def _parse_date(s: str | None):
    if not s:
        return None
    from datetime import datetime
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s.strip(), f).date().isoformat()
        except ValueError:
            pass
    return None


def _amt(s):
    try:
        return float(s.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def rules_extract(text: str) -> dict:
    g = lambda p: (m.group(1) if (m := re.search(p, text)) else None)  # noqa: E731
    vendor = next((v for v in VENDORS if v.lower() in text.lower()), None)
    return {
        "vendor_name": vendor,
        "invoice_number": g(r"Invoice No:\s*([A-Z]{3}-\d{5})") or g(r"Ref\s+([A-Z]{3}-\d{5})"),
        "invoice_date": _parse_date(g(r"Invoice Date:\s*" + _DATE) or g(r"Dated\s+" + _DATE)),
        "due_date": _parse_date(g(r"Due Date:\s*" + _DATE) or g(r"pay by\s+" + _DATE)),
        "po_number": g(r"PO Number:\s*(PO-\d{5})") or g(r"Your order:\s*(PO-\d{5})"),
        "currency": "USD" if "USD" in text else None,
        "subtotal": _amt(g(r"Subtotal:\s*\$([\d,]+\.\d{2})") or g(r"Net amount\s+([\d,]+\.\d{2})")),
        "tax": _amt(g(r"Sales tax:\s*\$([\d,]+\.\d{2})") or g(r"tax\s+([\d,]+\.\d{2})")),
        "total": _amt(g(r"TOTAL DUE \(USD\):\s*\$([\d,]+\.\d{2})") or g(r"Amount payable USD\s+([\d,]+\.\d{2})")),
        "line_count": (len(re.findall(r"^\s*\d+\s{2}\S", text, re.M)) or len(re.findall(r"^- ", text, re.M))) or None,
    }


# ---------------------------------------------------------------------------------------------------------------
# Model extraction contract (used by the recorder) and deterministic grading + routing (used by the page).
# ---------------------------------------------------------------------------------------------------------------
EXTRACT_SYSTEM = """You extract fields from supplier invoices for an accounts-payable team. Copy what the document says;
do not correct arithmetic or fill gaps. Use null for anything the document does not state. Dates as YYYY-MM-DD.
Amounts as plain numbers. Scanned copies may contain O for 0, l for 1 and S for 5 inside numbers: read them as digits.
line_count is the number of distinct line items billed."""

EXTRACT_TOOL = {
    "name": "record_invoice",
    "description": "Record the fields of one invoice.",
    "input_schema": {"type": "object", "properties": {
        "vendor_name": {"type": ["string", "null"]}, "invoice_number": {"type": ["string", "null"]},
        "invoice_date": {"type": ["string", "null"]}, "due_date": {"type": ["string", "null"]},
        "po_number": {"type": ["string", "null"]}, "currency": {"type": ["string", "null"]},
        "subtotal": {"type": ["number", "null"]}, "tax": {"type": ["number", "null"]}, "total": {"type": ["number", "null"]},
        "line_count": {"type": ["integer", "null"]},
    }, "required": FIELDS},
}
PROMPT_HASH = hashlib.sha256((EXTRACT_SYSTEM + json.dumps(EXTRACT_TOOL, sort_keys=True)).encode()).hexdigest()[:12]


def field_ok(name: str, got, want) -> bool:
    if want is None:
        return got in (None, "", "null")
    if got is None:
        return False
    if name in ("subtotal", "tax", "total"):
        try:
            return abs(float(got) - float(want)) < .005
        except (TypeError, ValueError):
            return False
    if name == "line_count":
        try:
            return int(got) == int(want)
        except (TypeError, ValueError):
            return False
    return str(got).strip().lower() == str(want).strip().lower()


def checks(ext: dict, seen: set) -> list[str]:
    """Deterministic controls an AP team would run on any extraction before paying."""
    out = []
    for k in ("vendor_name", "invoice_number", "invoice_date", "total"):
        if ext.get(k) in (None, ""):
            out.append(f"missing {k.replace('_', ' ')}")
    if not ext.get("po_number"):
        out.append("no purchase order")
    try:
        if abs(float(ext["subtotal"]) + float(ext["tax"]) - float(ext["total"])) > .01:
            out.append("total does not add up")
    except (TypeError, ValueError, KeyError):
        out.append("amounts unreadable")
    if ext.get("invoice_date") and ext.get("due_date") and str(ext["due_date"]) < str(ext["invoice_date"]):
        out.append("due before issued")
    key = (str(ext.get("vendor_name") or "").lower(), str(ext.get("invoice_number") or "").upper())
    if all(key) and key in seen:
        out.append("possible duplicate")
    seen.add(key)
    return out


def evaluate(docs: list[Doc], extractions: dict[str, dict]) -> dict:
    """Field accuracy, document accuracy, routing and planted-issue recall for one extractor."""
    rows, seen = [], set()
    for d in docs:
        ext = extractions.get(d.id)
        tr = truth(d)
        fields = {k: (field_ok(k, (ext or {}).get(k), tr[k]) if ext is not None else False) for k in FIELDS}
        flags = checks(ext or {}, seen) if ext is not None else ["not extracted"]
        rows.append({"id": d.id, "layout": d.layout, "issues": d.issues, "fields": fields, "all_fields": all(fields.values()),
                     "flags": flags, "held": bool(flags), "planted": bool(d.issues)})
    n = len(rows)
    planted = [r for r in rows if r["planted"]]
    clean = [r for r in rows if not r["planted"]]
    return {
        "rows": rows,
        "field_accuracy": float(np.mean([v for r in rows for v in r["fields"].values()])),
        "by_field": {k: float(np.mean([r["fields"][k] for r in rows])) for k in FIELDS},
        "by_layout": {lay: float(np.mean([v for r in rows if r["layout"] == lay for v in r["fields"].values()])) for lay in LAYOUTS},
        "doc_accuracy": float(np.mean([r["all_fields"] for r in rows])),
        "issue_recall": float(np.mean([r["held"] for r in planted])) if planted else 0.0,
        "false_holds": int(sum(r["held"] for r in clean)),
        "straight_through": float(np.mean([not r["held"] for r in rows])),
        "n": n,
    }
