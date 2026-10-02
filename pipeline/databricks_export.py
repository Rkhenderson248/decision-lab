# Databricks notebook source
# MAGIC %md
# MAGIC ## Export · public market table for the Decision Lab
# MAGIC
# MAGIC Paste this as the **last cell** of the Mortgage Market Intelligence notebook and run it
# MAGIC after the notebook has finished. It writes one row per CBSA containing **public-source
# MAGIC metrics only** (Census population estimates, HMDA, FHFA HPI). Nothing internal leaves:
# MAGIC no footprint, territories, lender names, lender position or branch data.
# MAGIC
# MAGIC The Decision Lab recomputes every score, archetype and risk reading from these inputs,
# MAGIC so the website and the notebook stay methodologically identical.
# MAGIC
# MAGIC Output: `markets.csv` plus `markets_meta.json`. Commit both to the decision-lab repo under
# MAGIC `data/` (or attach them in the chat and they will be committed for you).

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

DECISION_LAB_COLUMNS = {
    # identity
    "market_key": "market_key", "NAME": "market_name", "area_type": "area_type", "primary_state": "primary_state",
    # Census
    "population_latest": "population_latest", "population_cagr_pct": "population_cagr_pct",
    "domestic_migration_rate_pct": "domestic_migration_rate_pct",
    "international_migration_rate_pct": "international_migration_rate_pct",
    "natural_change_rate_pct": "natural_change_rate_pct",
    "demographic_opportunity_score": "demographic_opportunity_score",
    # HMDA (market-level aggregates only)
    "hmda_first_year": "hmda_first_year", "hmda_latest_year": "hmda_latest_year",
    "purchase_applications_latest": "purchase_applications_latest",
    "purchase_originations_latest": "purchase_originations_latest",
    "purchase_application_cagr_pct": "purchase_application_cagr_pct",
    "purchase_origination_cagr_pct": "purchase_origination_cagr_pct",
    "purchase_origination_rate_pct": "purchase_origination_rate_pct",
    "purchase_approval_rate_pct": "purchase_approval_rate_pct",
    "purchase_denial_rate_pct": "purchase_denial_rate_pct",
    "refinance_share_pct": "refinance_share_pct", "cashout_share_pct": "cashout_share_pct",
    "government_purchase_share_pct": "government_purchase_share_pct",
    "avg_loan_amount": "avg_loan_amount", "avg_property_value": "avg_property_value",
    "avg_applicant_income_k": "avg_applicant_income_k", "high_cltv_share_pct": "high_cltv_share_pct",
    "avg_dti_pct": "avg_dti_pct", "high_dti_share_pct": "high_dti_share_pct",
    "avg_interest_rate_pct": "avg_interest_rate_pct", "active_lenders": "active_lenders",
    "lender_hhi": "lender_hhi", "top5_lender_share_pct": "top5_lender_share_pct",
    "applicant_income_growth_pct": "applicant_income_growth_pct",
    # FHFA
    "fhfa_latest_year": "fhfa_latest_year", "hpi_1y_pct": "hpi_1y_pct", "hpi_3y_cagr_pct": "hpi_3y_cagr_pct",
    "hpi_5y_cagr_pct": "hpi_5y_cagr_pct", "hpi_volatility_pct": "hpi_volatility_pct",
    "hpi_drawdown_pct": "hpi_drawdown_pct", "hpi_acceleration_pp": "hpi_acceleration_pp",
}

_src = RESULTS_MMI["mortgage_markets"]
_present = [c for c in DECISION_LAB_COLUMNS if c in _src.columns]
_missing = sorted(set(DECISION_LAB_COLUMNS) - set(_present))
DECISION_LAB_EXPORT = (_src[_present].rename(columns=DECISION_LAB_COLUMNS)
                       .sort_values(["area_type", "population_latest"], ascending=[True, False])
                       .reset_index(drop=True))
DECISION_LAB_EXPORT["market_key"] = DECISION_LAB_EXPORT["market_key"].astype(str).str.zfill(5)

# Guard rails: nothing internal, one row per market.
_forbidden = [c for c in DECISION_LAB_EXPORT.columns
              if any(t in c.lower() for t in ("territory", "footprint", "branch", "fairway", "lender_name", "lei"))]
assert not _forbidden, f"Internal columns in export: {_forbidden}"
assert DECISION_LAB_EXPORT["market_key"].is_unique, "Duplicate market keys in export"

DECISION_LAB_META = {
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "markets": int(len(DECISION_LAB_EXPORT)),
    "hmda_years": [int(DECISION_LAB_EXPORT["hmda_first_year"].min()), int(DECISION_LAB_EXPORT["hmda_latest_year"].max())]
                  if "hmda_latest_year" in DECISION_LAB_EXPORT and DECISION_LAB_EXPORT["hmda_latest_year"].notna().any() else None,
    "fhfa_latest_year": int(DECISION_LAB_EXPORT["fhfa_latest_year"].max())
                        if "fhfa_latest_year" in DECISION_LAB_EXPORT and DECISION_LAB_EXPORT["fhfa_latest_year"].notna().any() else None,
    "sources": ["U.S. Census Bureau Population Estimates", "HMDA Loan Application Register (FFIEC/CFPB)",
                "FHFA House Price Index"],
    "missing_columns": _missing,
}

# Write next to the notebook if possible; otherwise to a Unity Catalog volume or DBFS.
_targets = [
    os.path.join(os.getcwd(), "decision_lab_export"),
    "/dbfs/FileStore/decision_lab",
]
_written = None
for _dir in _targets:
    try:
        os.makedirs(_dir, exist_ok=True)
        DECISION_LAB_EXPORT.to_csv(os.path.join(_dir, "markets.csv"), index=False, float_format="%.6g")
        with open(os.path.join(_dir, "markets_meta.json"), "w") as _fh:
            json.dump(DECISION_LAB_META, _fh, indent=2)
        _written = _dir
        break
    except Exception as _exc:  # read-only location; try the next one
        print(f"Could not write to {_dir}: {_exc}")

print(f"{len(DECISION_LAB_EXPORT):,} markets · {len(DECISION_LAB_EXPORT.columns)} columns")
print("Missing (optional) columns:", _missing or "none")
print("Written to:", _written or "nowhere. Use the download button on the table below instead.")
display(DECISION_LAB_EXPORT)  # the table's download button also produces markets.csv
