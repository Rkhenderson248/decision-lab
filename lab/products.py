"""An illustrative mortgage product catalogue and a transparent fit engine.

The rules are deliberately simplified stand-ins for real programme
guidelines. They exist to show the shape of a decision system: hard
eligibility checks first, then a ranking on what the borrower actually cares
about, with every exclusion explained. They are not underwriting guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONFORMING_LIMIT = 800_000  # illustrative
FHA_LIMIT = 550_000  # illustrative
USDA_INCOME_LIMIT = 125_000  # illustrative, annual household income


@dataclass
class Borrower:
    price: float
    down_pct: float
    credit: int
    income_monthly: float
    debts_monthly: float
    occupancy: str  # Primary residence, Second home, Investment
    military: bool
    rural: bool
    first_time: bool
    horizon_years: int
    market_rate: float  # % for a conventional 30-year fixed, the reference rate
    tax_ins_rate: float  # annual taxes + insurance as % of price
    arm_reset_bump: float  # assumed rate change at ARM reset, percentage points


@dataclass
class Check:
    rule: str
    passed: bool
    detail: str


@dataclass
class Result:
    name: str
    family: str
    eligible: bool
    checks: list[Check] = field(default_factory=list)
    rate: float = 0.0
    term_years: int = 30
    loan: float = 0.0
    ltv: float = 0.0
    upfront_fee: float = 0.0
    pi: float = 0.0
    mi: float = 0.0
    tax_ins: float = 0.0
    housing: float = 0.0
    dti: float = 0.0
    horizon_cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def payment(principal: float, annual_rate: float, years: int) -> float:
    r = annual_rate / 100 / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


def _horizon_cost(loan, rate, years, months, mi_monthly, price, mi_cancels, arm_reset=None):
    """Interest + mortgage insurance paid over the borrower's horizon."""
    bal = loan
    pmt = payment(loan, rate, years)
    interest = 0.0
    mi_paid = 0.0
    cur_rate = rate
    for m in range(1, months + 1):
        if arm_reset and m == arm_reset[0] + 1:
            cur_rate = rate + arm_reset[1]
            remaining = years * 12 - arm_reset[0]
            pmt = bal * (cur_rate / 1200) / (1 - (1 + cur_rate / 1200) ** -remaining)
        i = bal * cur_rate / 1200
        interest += i
        bal -= pmt - i
        if mi_monthly and not (mi_cancels and bal <= 0.78 * price):
            mi_paid += mi_monthly
        if bal <= 0:
            break
    return interest + mi_paid


def _pmi_rate(credit: int) -> float:
    if credit >= 760:
        return 0.30
    if credit >= 720:
        return 0.45
    if credit >= 680:
        return 0.70
    return 1.00


def evaluate(b: Borrower) -> list[Result]:
    base_loan = b.price * (1 - b.down_pct / 100)
    base_ltv = base_loan / b.price * 100
    tax_ins = b.price * b.tax_ins_rate / 100 / 12
    primary = b.occupancy == "Primary residence"
    occ_bump = {"Primary residence": 0.0, "Second home": 0.25, "Investment": 0.60}[b.occupancy]
    months = b.horizon_years * 12

    results: list[Result] = []

    def finish(res: Result, max_dti: float, mi_annual_pct: float, mi_cancels: bool, financed_fee_pct: float = 0.0, arm=None):
        res.upfront_fee = base_loan * financed_fee_pct / 100
        res.loan = base_loan + res.upfront_fee
        res.ltv = base_ltv
        res.pi = payment(res.loan, res.rate, res.term_years)
        res.mi = res.loan * mi_annual_pct / 100 / 12
        res.tax_ins = tax_ins
        res.housing = res.pi + res.mi + tax_ins
        res.dti = (res.housing + b.debts_monthly) / b.income_monthly * 100 if b.income_monthly else 999
        res.checks.append(
            Check("Debt-to-income", res.dti <= max_dti, f"{res.dti:.1f}% vs {max_dti:.0f}% maximum")
        )
        res.eligible = all(c.passed for c in res.checks)
        res.horizon_cost = res.upfront_fee + _horizon_cost(
            res.loan, res.rate, res.term_years, min(months, res.term_years * 12), res.mi, b.price, mi_cancels, arm
        )
        results.append(res)

    # Conventional 30 and 15 ------------------------------------------------
    for name, term, offset in (("Conventional 30-year fixed", 30, 0.0), ("Conventional 15-year fixed", 15, -0.65)):
        max_ltv = {"Primary residence": 97 if b.first_time else 95, "Second home": 90, "Investment": 85}[b.occupancy]
        r = Result(name, "Conventional", False, rate=b.market_rate + offset + occ_bump, term_years=term)
        r.checks += [
            Check("Credit score", b.credit >= 620, f"{b.credit} vs 620 minimum"),
            Check("Loan-to-value", base_ltv <= max_ltv, f"{base_ltv:.1f}% vs {max_ltv}% maximum for {b.occupancy.lower()}"),
            Check("Loan size", base_loan <= CONFORMING_LIMIT, f"${base_loan:,.0f} vs ${CONFORMING_LIMIT:,.0f} conforming limit"),
        ]
        mi = _pmi_rate(b.credit) if base_ltv > 80 else 0.0
        if mi:
            r.notes.append("Private mortgage insurance until the balance reaches 78% of value.")
        finish(r, 50 if b.credit >= 700 else 45, mi, True)

    # FHA -----------------------------------------------------------------------
    r = Result("FHA 30-year fixed", "Government", False, rate=b.market_rate - 0.25, term_years=30)
    fha_credit_ok = (b.credit >= 580 and b.down_pct >= 3.5) or (b.credit >= 500 and b.down_pct >= 10)
    r.checks += [
        Check("Occupancy", primary, "Primary residence required"),
        Check("Credit score", fha_credit_ok, f"{b.credit} vs 580 (3.5% down) or 500 (10% down)"),
        Check("Loan-to-value", base_ltv <= 96.5, f"{base_ltv:.1f}% vs 96.5% maximum"),
        Check("Loan size", base_loan <= FHA_LIMIT, f"${base_loan:,.0f} vs ${FHA_LIMIT:,.0f} illustrative area limit"),
    ]
    r.notes.append("Upfront premium of 1.75% financed into the loan, plus an annual premium.")
    finish(r, 50, 0.55, False, financed_fee_pct=1.75)

    # VA -------------------------------------------------------------------------
    r = Result("VA 30-year fixed", "Government", False, rate=b.market_rate - 0.30, term_years=30)
    fee = 2.15 if b.down_pct < 5 else (1.5 if b.down_pct < 10 else 1.25)
    r.checks += [
        Check("Military service", b.military, "Eligible service history required"),
        Check("Occupancy", primary, "Primary residence required"),
        Check("Credit score", b.credit >= 580, f"{b.credit} vs 580 illustrative lender minimum"),
    ]
    r.notes.append(f"No monthly mortgage insurance. Funding fee of {fee}% financed (first use).")
    finish(r, 50, 0.0, False, financed_fee_pct=fee)

    # USDA -------------------------------------------------------------------------
    r = Result("USDA 30-year fixed", "Government", False, rate=b.market_rate - 0.20, term_years=30)
    r.checks += [
        Check("Eligible rural area", b.rural, "Property must be in an eligible area"),
        Check("Occupancy", primary, "Primary residence required"),
        Check("Household income", b.income_monthly * 12 <= USDA_INCOME_LIMIT, f"${b.income_monthly * 12:,.0f} vs ${USDA_INCOME_LIMIT:,.0f} illustrative limit"),
        Check("Credit score", b.credit >= 640, f"{b.credit} vs 640 minimum"),
    ]
    r.notes.append("1% upfront guarantee fee financed, plus a 0.35% annual fee.")
    finish(r, 41, 0.35, False, financed_fee_pct=1.0)

    # 7/6 ARM ------------------------------------------------------------------------
    r = Result("7/6 adjustable-rate", "Conventional", False, rate=b.market_rate - 0.55 + occ_bump, term_years=30)
    r.checks += [
        Check("Credit score", b.credit >= 640, f"{b.credit} vs 640 minimum"),
        Check("Loan-to-value", base_ltv <= 90, f"{base_ltv:.1f}% vs 90% maximum"),
        Check("Loan size", base_loan <= CONFORMING_LIMIT, f"${base_loan:,.0f} vs ${CONFORMING_LIMIT:,.0f} conforming limit"),
    ]
    mi = _pmi_rate(b.credit) if base_ltv > 80 else 0.0
    if b.horizon_years > 7:
        r.notes.append(f"Your horizon passes the year-7 reset. Costs assume the rate rises {b.arm_reset_bump:.1f} points.")
    else:
        r.notes.append("Fixed for the first seven years, which covers your stated horizon.")
    finish(r, 45, mi, True, arm=(84, b.arm_reset_bump))

    # Jumbo -------------------------------------------------------------------------
    r = Result("Jumbo 30-year fixed", "Jumbo", False, rate=b.market_rate + 0.15 + occ_bump, term_years=30)
    r.checks += [
        Check("Loan size", base_loan > CONFORMING_LIMIT, f"${base_loan:,.0f} must exceed ${CONFORMING_LIMIT:,.0f}"),
        Check("Credit score", b.credit >= 700, f"{b.credit} vs 700 minimum"),
        Check("Loan-to-value", base_ltv <= 90, f"{base_ltv:.1f}% vs 90% maximum"),
    ]
    finish(r, 43, 0.0, False)

    return results


def rank(results: list[Result], by: str = "horizon") -> list[Result]:
    key = (lambda r: r.housing) if by == "monthly" else (lambda r: r.horizon_cost)
    eligible = sorted([r for r in results if r.eligible], key=key)
    rest = sorted([r for r in results if not r.eligible], key=lambda r: len(r.failed))
    return eligible + rest


def total_checks(results: list[Result]) -> int:
    return int(np.sum([len(r.checks) for r in results]))
