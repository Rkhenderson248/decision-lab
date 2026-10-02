"""Synthetic stand-in for the public market table.

Used only when data/markets.csv is absent, so the page can be developed and
tested without the real export. Market names are deliberately fictional
("Sample Metro 014") and the page labels the data as a sample, so no reader
can mistake it for published statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STATES = {
    "CA": 9, "TX": 9, "FL": 7, "NY": 5, "PA": 5, "OH": 5, "IL": 4, "GA": 4, "NC": 4, "MI": 4,
    "TN": 3, "VA": 3, "WA": 3, "IN": 3, "MO": 3, "WI": 3, "AL": 3, "SC": 3, "KY": 3, "LA": 3,
    "AZ": 2, "CO": 2, "MN": 2, "OK": 2, "OR": 2, "IA": 2, "KS": 2, "AR": 2, "MS": 2, "UT": 2,
    "NE": 1, "NM": 1, "ID": 1, "WV": 1, "NV": 1, "ME": 1, "NH": 1, "MT": 1, "ND": 1, "SD": 1,
    "WY": 1, "MD": 2, "MA": 2, "NJ": 2, "CT": 1, "DE": 1, "RI": 1, "VT": 1, "AK": 1, "HI": 1,
}


def sample_markets(seed: int = 11, metros: int = 387, micros: int = 541) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.array(list(_STATES))
    p_state = np.array(list(_STATES.values()), dtype=float)
    p_state /= p_state.sum()
    rows = []
    for area, n, prefix, key0 in (("Metropolitan", metros, "Sample Metro", 10000),
                                  ("Micropolitan", micros, "Sample Micro", 60000)):
        g = rng.normal(size=n)            # demographic growth
        p = 0.55 * g + rng.normal(size=n) * 0.85  # price momentum, partly growth-driven
        c = rng.normal(size=n)            # credit quality
        o = rng.normal(size=n)            # competitive openness
        st = rng.choice(states, size=n, p=p_state)
        metro = area == "Metropolitan"
        pop = np.clip(np.exp(rng.normal(12.55 if metro else 10.55, 0.95 if metro else 0.45, n)),
                      55_000 if metro else 10_000, 19_000_000 if metro else 240_000).round()
        apps = pop * np.clip(rng.normal(0.0085, 0.0018, n) * (1 + 0.14 * g), 0.003, None)
        orate = np.clip(63 + 3.5 * c + rng.normal(0, 3, n), 45, 80)
        value = np.exp(rng.normal(np.log(330_000 if metro else 215_000), 0.33, n) + 0.12 * p)
        income = np.exp(rng.normal(np.log(108 if metro else 86), 0.22, n) + 0.05 * c)
        hpi1 = 3.6 + 2.4 * p + rng.normal(0, 1.2, n)
        hpi3 = 4.8 + 2.0 * p + rng.normal(0, 1.0, n)
        has_hpi = rng.random(n) > (0.03 if metro else 0.28)
        lenders = (115 if metro else 38) * (pop / (300_000 if metro else 40_000)) ** (0.33 if metro else 0.28)
        lenders = np.clip(lenders * np.exp(0.15 * o + rng.normal(0, 0.12, n)), 8, 900).round()
        hhi = np.clip((480 if metro else 760) - 160 * o + rng.normal(0, 120, n), 150, 3200)
        for i in range(n):
            hp = bool(has_hpi[i])
            rows.append({
                "market_key": f"{key0 + i * 7:05d}"[-5:],
                "market_name": f"{prefix} {i + 1:03d}, {st[i]}",
                "area_type": area,
                "primary_state": st[i],
                "population_latest": pop[i],
                "population_cagr_pct": 0.45 + 0.75 * g[i] + rng.normal(0, 0.15),
                "domestic_migration_rate_pct": 0.2 + 0.6 * g[i] + rng.normal(0, 0.2),
                "international_migration_rate_pct": max(0.0, 0.35 + 0.1 * g[i] + rng.normal(0, 0.15)),
                "natural_change_rate_pct": 0.1 + rng.normal(0, 0.18),
                "demographic_opportunity_score": float(np.clip(50 + 19 * g[i] + rng.normal(0, 8), 0, 100)),
                "hmda_first_year": 2022,
                "hmda_latest_year": 2024,
                "purchase_applications_latest": apps[i].round(),
                "purchase_originations_latest": (apps[i] * orate[i] / 100).round(),
                "purchase_application_cagr_pct": -7.5 + 3.5 * g[i] + rng.normal(0, 3),
                "purchase_origination_cagr_pct": -8.0 + 3.8 * g[i] + 1.0 * c[i] + rng.normal(0, 3),
                "purchase_origination_rate_pct": orate[i],
                "purchase_approval_rate_pct": float(np.clip(orate[i] + 13 + rng.normal(0, 2), 50, 95)),
                "purchase_denial_rate_pct": float(np.clip(12 - 3.2 * c[i] + rng.normal(0, 2), 3, 35)),
                "refinance_share_pct": float(np.clip(19 + 5 * rng.normal() - 2 * g[i], 5, 45)),
                "cashout_share_pct": float(np.clip(8 + 2.5 * rng.normal(), 1, 25)),
                "government_purchase_share_pct": float(np.clip(23 - 5 * c[i] + rng.normal(0, 5), 4, 60)),
                "avg_loan_amount": value[i] * np.clip(rng.normal(0.82, 0.04), 0.6, 0.97),
                "avg_property_value": value[i],
                "avg_applicant_income_k": income[i],
                "high_cltv_share_pct": float(np.clip(19 - 3.5 * c[i] + rng.normal(0, 3), 2, 50)),
                "avg_dti_pct": float(np.clip(37.5 - 1.6 * c[i] + rng.normal(0, 1.2), 25, 48)),
                "high_dti_share_pct": float(np.clip(23 - 4.5 * c[i] + 2 * p[i] + rng.normal(0, 3), 4, 55)),
                "avg_interest_rate_pct": 6.65 + rng.normal(0, 0.12),
                "active_lenders": lenders[i],
                "lender_hhi": hhi[i],
                "top5_lender_share_pct": float(np.clip(36 + hhi[i] / 70 + rng.normal(0, 4), 15, 85)),
                "applicant_income_growth_pct": 9 + 2.5 * rng.normal(),
                "fhfa_latest_year": 2025 if hp else np.nan,
                "hpi_1y_pct": hpi1[i] if hp else np.nan,
                "hpi_3y_cagr_pct": hpi3[i] if hp else np.nan,
                "hpi_5y_cagr_pct": (7.2 + 1.8 * p[i] + rng.normal(0, 0.8)) if hp else np.nan,
                "hpi_volatility_pct": max(0.6, 2.8 + 0.6 * abs(p[i]) + rng.normal(0, 0.8)) if hp else np.nan,
                "hpi_drawdown_pct": min(0.0, -0.8 - 1.4 * max(0.0, -hpi1[i] / 3) + rng.normal(0, 0.7)) if hp else np.nan,
                "hpi_acceleration_pp": (rng.normal(-0.6, 1.4)) if hp else np.nan,
            })
    return pd.DataFrame(rows)
