"""Build data/markets.csv from public sources. Runs in GitHub Actions.

One row per Core Based Statistical Area, holding only public metrics:

* U.S. Census Bureau population estimates (CBSA totals and components)
* American Community Survey 5-year: median household income and home value
* FHFA House Price Index (traditional, all-transactions, quarterly, metro level)
* HMDA via the FFIEC Data Browser API: purchase and refinance activity by
  action taken and loan type (aggregations endpoint) and active lenders
  (filers endpoint)

Every step logs what it found to data/build_log.json, so a run that half
works is diagnosable from the repository alone.

Usage:  python pipeline/build_markets.py --out data
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

UA = {"User-Agent": "decision-lab-pipeline/1.0 (+https://richardhenderson.io)"}
HMDA_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view"
FHFA_URL = "https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv"
POPEST_CANDIDATES = [
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-{y}/metro/totals/cbsa-est{y}-alldata.csv"
]
ACS_URL = ("https://api.census.gov/data/{y}/acs/acs5?get=NAME,B19013_001E,B25077_001E"
           "&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*")
# The Census API now requires a (free) key: https://api.census.gov/data/key_signup.html
CENSUS_API_KEY = __import__("os").environ.get("CENSUS_API_KEY", "").strip()

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}

LOG: dict = {"steps": {}, "warnings": []}


def log(step: str, **info) -> None:
    LOG["steps"].setdefault(step, {}).update(info)
    print(f"[{step}] " + json.dumps(info, default=str)[:600], flush=True)


def warn(msg: str) -> None:
    LOG["warnings"].append(msg)
    print("WARNING:", msg, flush=True)


def fetch(url: str, timeout: int = 120, retries: int = 4, binary: bool = False):
    last = None
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", errors="replace")
        except HTTPError as exc:
            last = exc
            if exc.code in (400, 401, 403, 404):
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:  # noqa: BLE001
                    body = ""
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            time.sleep(2 ** attempt + 1)
        except (URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2 ** attempt + 1)
    raise last  # type: ignore[misc]


def safe_cagr(start, end, years):
    if not (pd.notna(start) and pd.notna(end)) or start <= 0 or end <= 0 or years <= 0:
        return np.nan
    return float(((end / start) ** (1 / years) - 1) * 100)


# ---------------------------------------------------------------------------
# 1. Census population estimates: CBSA totals + county crosswalk
# ---------------------------------------------------------------------------
def load_popest():
    year = datetime.now().year
    frame, used = None, None
    for y in range(year, year - 4, -1):
        for template in POPEST_CANDIDATES:
            url = template.format(y=y)
            try:
                frame = pd.read_csv(io.BytesIO(fetch(url, binary=True)), encoding="latin-1", dtype=str)
                used = (url, y)
                break
            except Exception as exc:  # noqa: BLE001
                LOG["steps"].setdefault("popest_attempts", {})[url] = str(exc)[:120]
        if frame is not None:
            break
    if frame is None:
        raise RuntimeError("No Census CBSA population estimates file could be downloaded.")

    url, vintage = used
    frame.columns = [c.strip().upper() for c in frame.columns]
    for c in ("CBSA", "MDIV", "STCOU"):
        if c in frame.columns:
            frame[c] = frame[c].fillna("").str.strip()
    pop_cols = sorted(c for c in frame.columns if c.startswith("POPESTIMATE") and c[11:].isdigit())
    years = [int(c[11:]) for c in pop_cols]
    latest, first = max(years), min(years)
    lsad = frame["LSAD"].fillna("")

    totals = frame[lsad.str.contains("Statistical Area", na=False) & frame["MDIV"].eq("") & frame["STCOU"].eq("")].copy()
    divisions = frame[lsad.str.contains("Division", na=False)][["CBSA", "MDIV", "NAME", f"POPESTIMATE{latest}"]].copy()
    counties = frame[frame["STCOU"].ne("")][["CBSA", "MDIV", "STCOU", "NAME", f"POPESTIMATE{latest}"]].copy()

    def num(col):
        return pd.to_numeric(totals.get(col), errors="coerce")

    out = pd.DataFrame({
        "market_key": totals["CBSA"].str.zfill(5),
        "market_name": totals["NAME"].str.replace(r"\s+(Metro|Micro)politan Statistical Area$", "", regex=True),
        "area_type": np.where(lsad.loc[totals.index].str.startswith("Metro"), "Metropolitan", "Micropolitan"),
        "population_latest": num(f"POPESTIMATE{latest}"),
        "population_first": num(f"POPESTIMATE{first}"),
        "population_prior": num(f"POPESTIMATE{latest - 1}"),
        "natural_change_rate_pct": num(f"RNATURALCHG{latest}") / 10,
        "international_migration_rate_pct": num(f"RINTERNATIONALMIG{latest}") / 10,
        "domestic_migration_rate_pct": num(f"RDOMESTICMIG{latest}") / 10,
    })
    # Rate columns are not published in every vintage; derive them from components when absent.
    for rate, comp in (("natural_change_rate_pct", "NATURALCHG"), ("international_migration_rate_pct", "INTERNATIONALMIG"),
                       ("domestic_migration_rate_pct", "DOMESTICMIG")):
        if out[rate].isna().all():
            out[rate] = num(f"{comp}{latest}") / out["population_prior"] * 100
    out["population_cagr_pct"] = [safe_cagr(a, b, latest - first) for a, b in zip(out["population_first"], out["population_latest"])]
    out["latest_growth_pct"] = (out["population_latest"] / out["population_prior"] - 1) * 100

    counties["pop"] = pd.to_numeric(counties[f"POPESTIMATE{latest}"], errors="coerce")
    counties["state"] = counties["STCOU"].str[:2].map(STATE_FIPS)
    anchor = counties.sort_values("pop", ascending=False).drop_duplicates("CBSA").set_index("CBSA")["state"]
    out["primary_state"] = out["market_key"].map(anchor)
    parsed = out["market_name"].str.extract(r",\s*([A-Z]{2})")[0]
    out["primary_state"] = out["primary_state"].fillna(parsed)

    divisions["pop"] = pd.to_numeric(divisions[f"POPESTIMATE{latest}"], errors="coerce")
    log("census_popest", url=url, vintage=vintage, years=[first, latest], markets=len(out),
        rate_columns=[c for c in frame.columns if c.startswith("R") and c.endswith(str(latest))][:12],
        migration_coverage=int(out["domestic_migration_rate_pct"].notna().sum()),
        metros=int((out["area_type"] == "Metropolitan").sum()), divisions=len(divisions), counties=len(counties))
    return out, divisions[["CBSA", "MDIV", "pop"]], counties[["CBSA", "MDIV", "STCOU", "pop"]]


def demographic_score(m: pd.DataFrame) -> pd.Series:
    def pct(col):
        return pd.to_numeric(m[col], errors="coerce").groupby(m["area_type"]).rank(pct=True) * 100
    parts = {"population_cagr_pct": 0.35, "latest_growth_pct": 0.25, "domestic_migration_rate_pct": 0.20,
             "international_migration_rate_pct": 0.10, "natural_change_rate_pct": 0.10}
    values = pd.DataFrame({c: pct(c) for c in parts})
    w = pd.Series(parts)
    return values.mul(w, axis=1).sum(axis=1, min_count=1) / values.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)


# ---------------------------------------------------------------------------
# 2. ACS 5-year: income and home value, latest and three years earlier
# ---------------------------------------------------------------------------
def load_acs():
    if not CENSUS_API_KEY:
        warn("CENSUS_API_KEY not set; ACS income and home-value inputs skipped.")
        return pd.DataFrame(columns=["market_key"])
    year = datetime.now().year
    got = {}
    for y in range(year - 1, year - 8, -1):
        try:
            text = fetch(ACS_URL.format(y=y) + f"&key={CENSUS_API_KEY}", timeout=60)
            rows = json.loads(text)
        except json.JSONDecodeError:
            LOG["steps"].setdefault("acs_attempts", {})[str(y)] = "non-JSON: " + text[:300]
            continue
        except Exception as exc:  # noqa: BLE001
            LOG["steps"].setdefault("acs_attempts", {})[str(y)] = str(exc)[:300]
            continue
        head, body = rows[0], rows[1:]
        df = pd.DataFrame(body, columns=head)
        geo = [c for c in df.columns if "statistical area" in c.lower()][0]
        df = df.rename(columns={geo: "market_key", "B19013_001E": "income", "B25077_001E": "home_value"})
        for c in ("income", "home_value"):
            df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda s: s > 0)
        got[y] = df[["market_key", "income", "home_value"]]
        if len(got) == 1:
            continue
        if max(got) - y >= 3:
            break
    if not got:
        warn("ACS unavailable; income and home-value inputs left empty.")
        return pd.DataFrame(columns=["market_key"])
    latest = max(got)
    base = got[latest].rename(columns={"income": "avg_applicant_income_k", "home_value": "avg_property_value"})
    base["avg_applicant_income_k"] = base["avg_applicant_income_k"] / 1000
    earlier = [y for y in got if latest - y >= 3]
    if earlier:
        prior = got[max(earlier)][["market_key", "income"]].rename(columns={"income": "income_prior"})
        base = base.merge(prior, on="market_key", how="left")
        base["applicant_income_growth_pct"] = (base["avg_applicant_income_k"] * 1000 / base["income_prior"] - 1) * 100
        base = base.drop(columns=["income_prior"])
    log("acs", latest=latest, compared_with=max(earlier) if earlier else None, markets=len(base))
    return base


# ---------------------------------------------------------------------------
# 3. FHFA HPI: metro features (divisions rolled up to their CBSA)
# ---------------------------------------------------------------------------
def hpi_features(group: pd.DataFrame) -> dict:
    group = group.sort_values(["yr", "period"]).copy()
    ppy = 4
    group["seq"] = group["yr"].astype(int) * ppy + group["period"].fillna(1).astype(int)
    group = group.drop_duplicates("seq", keep="last")
    latest = group.iloc[-1]
    lseq = int(latest["seq"])

    def change(periods):
        t = group[group["seq"] <= lseq - periods]
        if t.empty or t.iloc[-1]["index_nsa"] <= 0:
            return np.nan
        return float((latest["index_nsa"] / t.iloc[-1]["index_nsa"] - 1) * 100)

    one, three, five = change(ppy), change(3 * ppy), change(5 * ppy)
    recent = group.tail(3 * ppy + 1)
    ch = recent["index_nsa"].pct_change(fill_method=None) * 100
    vol = float(ch.std(ddof=0) * math.sqrt(ppy)) if ch.notna().sum() >= 3 else np.nan
    trailing = group.tail(5 * ppy + 1)
    peak = trailing["index_nsa"].max()
    prev = group[group["seq"] <= lseq - ppy]
    prior_one = np.nan
    if not prev.empty:
        pl = prev.iloc[-1]
        pt = group[group["seq"] <= int(pl["seq"]) - ppy]
        if not pt.empty and pt.iloc[-1]["index_nsa"] > 0:
            prior_one = float((pl["index_nsa"] / pt.iloc[-1]["index_nsa"] - 1) * 100)
    return {
        "fhfa_latest_year": int(latest["yr"]),
        "hpi_1y_pct": one,
        "hpi_3y_cagr_pct": ((1 + three / 100) ** (1 / 3) - 1) * 100 if pd.notna(three) and three > -100 else np.nan,
        "hpi_5y_cagr_pct": ((1 + five / 100) ** (1 / 5) - 1) * 100 if pd.notna(five) and five > -100 else np.nan,
        "hpi_volatility_pct": vol,
        "hpi_drawdown_pct": float((latest["index_nsa"] / peak - 1) * 100) if peak > 0 else np.nan,
        "hpi_acceleration_pp": one - prior_one if pd.notna(one) and pd.notna(prior_one) else np.nan,
    }


def load_fhfa(divisions: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(fetch(FHFA_URL, timeout=300, binary=True)), dtype=str)
    raw.columns = [c.strip().lower() for c in raw.columns]
    for c in ("hpi_type", "hpi_flavor", "frequency", "level"):
        raw[c] = raw[c].fillna("").str.lower().str.strip()
    raw["place_id"] = raw["place_id"].fillna("").str.strip().str.zfill(5)
    for c in ("yr", "period", "index_nsa"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    msa = raw[raw["hpi_type"].str.contains("traditional") & raw["hpi_flavor"].str.contains("all")
              & raw["frequency"].eq("quarterly") & raw["level"].str.contains("msa")].dropna(subset=["yr", "index_nsa"])
    feats = pd.DataFrame([{"place_id": pid, **hpi_features(g)} for pid, g in msa.groupby("place_id")])
    if feats.empty:
        warn("FHFA metro series not found with the expected filters.")
        return pd.DataFrame(columns=["market_key"])

    direct = feats.rename(columns={"place_id": "market_key"})
    # Divided metros are published by division; roll divisions up, population-weighted.
    div = divisions.merge(feats, left_on="MDIV", right_on="place_id", how="inner")
    rolled = []
    for cbsa, g in div.groupby("CBSA"):
        w = g["pop"].fillna(0)
        rec = {"market_key": cbsa.zfill(5), "fhfa_latest_year": int(g["fhfa_latest_year"].max())}
        for c in [c for c in feats.columns if c.startswith("hpi_")]:
            vals = g[c]
            ok = vals.notna() & (w > 0)
            rec[c] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else np.nan
        rolled.append(rec)
    rolled = pd.DataFrame(rolled)
    out = pd.concat([direct[~direct["market_key"].isin(rolled["market_key"])] if not rolled.empty else direct, rolled],
                    ignore_index=True)
    log("fhfa", series=int(msa["place_id"].nunique()), direct=len(direct), divisions_rolled=len(rolled),
        latest_year=int(out["fhfa_latest_year"].max()))
    return out


# ---------------------------------------------------------------------------
# 4. HMDA via the Data Browser API
# ---------------------------------------------------------------------------
# The Data Browser accepts at most two HMDA data filters per request, so the
# cohort cannot also be narrowed to first-lien, site-built, 1-4 unit lending.
# Shares and rates are computed on all closed-end applications in the market.
COHORT: dict = {}


def hmda_json(endpoint: str, params: dict):
    url = f"{HMDA_BASE}/{endpoint}?" + urlencode(params, safe=",")
    return json.loads(fetch(url, timeout=90, retries=5))


PROBE_VARIANTS = {"full": {"actions_taken": "1", "loan_purposes": "1"}}


def hmda_latest_year() -> int:
    year = datetime.now().year
    found = None
    for y in range(year - 1, year - 4, -1):
        for name, variant in PROBE_VARIANTS.items():
            params = {k: v for k, v in variant.items() if k != "_geo"}
            kind, code = variant.get("_geo", ("msamds", "19124"))
            params.update({"years": y, kind: code})
            try:
                data = hmda_json("aggregations", params)
                total = sum(int(a.get("count", 0)) for a in data.get("aggregations", []))
                LOG["steps"].setdefault("hmda_probe", {})[f"{y}:{name}"] = {"total": total, "sample": data.get("aggregations", [])[:2]}
                if total > 0 and name == "full" and found is None:
                    found = y
            except Exception as exc:  # noqa: BLE001
                LOG["steps"].setdefault("hmda_probe", {})[f"{y}:{name}"] = str(exc)[:300]
        if found:
            return found
    raise RuntimeError("Could not find a published HMDA year through the Data Browser API.")


def hmda_geo_activity(kind: str, code: str, year: int) -> list[dict]:
    """Two requests (two-filter limit): outcomes by purpose, then purchase originations by loan type."""
    outcomes = hmda_json("aggregations", {"years": year, kind: code, "actions_taken": "1,2,3,4,5",
                                          "loan_purposes": "1,31,32"}).get("aggregations", [])
    types = hmda_json("aggregations", {"years": year, kind: code, "actions_taken": "1",
                                       "loan_types": "1,2,3,4"}).get("aggregations", [])
    # Loan-type rows cover all purposes; tag them so they are counted only for the government share.
    for row in types:
        row["_types"] = True
    return outcomes + types


def hmda_geo_filers(kind: str, code: str, year: int) -> list[dict]:
    data = hmda_json("filers", {"years": year, kind: code})
    return data.get("institutions", [])


def summarise_activity(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    for c in ("count", "sum"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)
    is_types = df["_types"].fillna(False).astype(bool) if "_types" in df.columns else pd.Series(False, index=df.index)
    a = df.get("actions_taken", pd.Series("", index=df.index)).astype(str)
    p = df.get("loan_purposes", pd.Series("", index=df.index)).astype(str)
    t = df.get("loan_types", pd.Series("", index=df.index)).astype(str)
    types_orig = df.loc[is_types & a.eq("1")]
    tt = t.loc[types_orig.index]
    gov_share = (types_orig.loc[tt.isin(["2", "3", "4"]), "count"].sum() / types_orig["count"].sum()
                 if types_orig["count"].sum() > 0 else np.nan)
    df, a, p = df.loc[~is_types], a.loc[~is_types], p.loc[~is_types]
    purch = p.eq("1")
    refi = p.isin(["31", "32"])
    orig = a.eq("1")
    return {
        "purchase_applications": float(df.loc[purch, "count"].sum()),
        "purchase_originations": float(df.loc[purch & orig, "count"].sum()),
        "purchase_approved": float(df.loc[purch & a.isin(["1", "2"]), "count"].sum()),
        "purchase_denied": float(df.loc[purch & a.eq("3"), "count"].sum()),
        "purchase_decided": float(df.loc[purch & a.isin(["1", "2", "3"]), "count"].sum()),
        "purchase_volume": float(df.loc[purch & orig, "sum"].sum()),
        "gov_share_all_purposes": float(gov_share) if pd.notna(gov_share) else np.nan,
        "_gov_weight": float(types_orig["count"].sum()),
        "refi_originations": float(df.loc[refi & orig, "count"].sum()),
        "cashout_originations": float(df.loc[p.eq("32") & orig, "count"].sum()),
    }


def load_hmda(markets: pd.DataFrame, divisions: pd.DataFrame, counties: pd.DataFrame, workers: int = 6):
    latest = hmda_latest_year()
    first = latest - 3
    metros = markets[markets["area_type"].eq("Metropolitan")]["market_key"]
    divided = set(divisions["CBSA"].str.zfill(5))
    geos = []  # (market_key, kind, code)
    for key in metros:
        if key in divided:
            for mdiv in divisions.loc[divisions["CBSA"].str.zfill(5).eq(key), "MDIV"]:
                geos.append((key, "msamds", mdiv.zfill(5)))
        else:
            geos.append((key, "msamds", key))
    micro_keys = set(markets.loc[markets["area_type"].eq("Micropolitan"), "market_key"])
    for _, r in counties[counties["CBSA"].str.zfill(5).isin(micro_keys)].iterrows():
        geos.append((r["CBSA"].zfill(5), "counties", r["STCOU"].zfill(5)))

    years = list(range(first, latest + 1))
    jobs = [(g, y) for g in geos for y in years]
    activity, failures = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(hmda_geo_activity, g[1], g[2], y): (g, y) for g, y in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            (key, kind, code), y = futs[f]
            try:
                s = summarise_activity(f.result())
                bucket = activity.setdefault((key, y), {})
                gs, gw = s.pop("gov_share_all_purposes", np.nan), s.pop("_gov_weight", 0.0)
                if pd.notna(gs) and gw > 0:
                    bucket["_gov_num"] = bucket.get("_gov_num", 0.0) + gs * gw
                    bucket["_gov_den"] = bucket.get("_gov_den", 0.0) + gw
                for k, v in s.items():
                    bucket[k] = bucket.get(k, 0.0) + v
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{kind}={code} {y}: {str(exc)[:80]}")
            if i % 250 == 0:
                print(f"  HMDA activity {i}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)

    # Lenders: latest year and the first year, for share and momentum.
    lender_counts: dict[str, dict[str, float]] = {}
    lender_counts_first: dict[str, dict[str, float]] = {}
    lender_names: dict[str, str] = {}
    filer_fail = 0
    sample_filers = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(hmda_geo_filers, g[1], g[2], y): (g, y) for g in geos for y in (first, latest)}
        for f in as_completed(futs):
            (key, _, _), y = futs[f]
            try:
                inst = f.result()
                if sample_filers is None and inst:
                    sample_filers = inst[:3]
                agg = (lender_counts if y == latest else lender_counts_first).setdefault(key, {})
                for row in inst:
                    lei = str(row.get("lei", ""))
                    agg[lei] = agg.get(lei, 0.0) + float(row.get("count", 0) or 0)
                    if row.get("name") and (y == latest or lei not in lender_names):
                        lender_names[lei] = str(row["name"]).strip()
            except Exception:  # noqa: BLE001
                filer_fail += 1

    # Yearly market series for trend charts.
    year_rows = []
    for (key, y), a in activity.items():
        apps, origs = a.get("purchase_applications", 0), a.get("purchase_originations", 0)
        decided = a.get("purchase_decided", 0)
        allorig = origs + a.get("refi_originations", 0)
        year_rows.append({
            "market_key": key, "year": y, "purchase_applications": apps, "purchase_originations": origs,
            "purchase_denial_rate_pct": a.get("purchase_denied", 0) / decided * 100 if decided else np.nan,
            "refinance_share_pct": a.get("refi_originations", 0) / allorig * 100 if allorig else np.nan,
            "avg_loan_amount": a.get("purchase_volume", 0) / origs if origs else np.nan,
        })
    LOG["_market_years"] = pd.DataFrame(year_rows)

    # Lender table: top 40 lenders per market by latest-year records, with first-year counts.
    lrows = []
    for key, counts in lender_counts.items():
        total = sum(counts.values()) or 1.0
        first_counts = lender_counts_first.get(key, {})
        total_first = sum(first_counts.values()) or np.nan
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:40]
        for rank, (lei, c) in enumerate(top, 1):
            c0 = first_counts.get(lei, 0.0)
            lrows.append({"market_key": key, "lei": lei, "lender": lender_names.get(lei, lei), "rank": rank,
                          "records": c, "share_pct": c / total * 100, "records_first": c0,
                          "share_first_pct": c0 / total_first * 100 if total_first == total_first else np.nan})
    LOG["_lenders"] = pd.DataFrame(lrows)

    recs = []
    for key in markets["market_key"]:
        a1, a0 = activity.get((key, latest), {}), activity.get((key, first), {})
        if not a1 or a1.get("purchase_applications", 0) <= 0:
            continue
        apps, origs = a1["purchase_applications"], a1["purchase_originations"]
        decided = a1.get("purchase_decided", 0)
        allorig = origs + a1.get("refi_originations", 0)
        rec = {
            "market_key": key, "hmda_first_year": first if a0 else latest, "hmda_latest_year": latest,
            "purchase_applications_latest": apps, "purchase_originations_latest": origs,
            "purchase_application_cagr_pct": safe_cagr(a0.get("purchase_applications"), apps, 3),
            "purchase_origination_cagr_pct": safe_cagr(a0.get("purchase_originations"), origs, 3),
            "purchase_origination_rate_pct": origs / apps * 100 if apps else np.nan,
            "purchase_approval_rate_pct": a1.get("purchase_approved", 0) / decided * 100 if decided else np.nan,
            "purchase_denial_rate_pct": a1.get("purchase_denied", 0) / decided * 100 if decided else np.nan,
            "refinance_share_pct": a1.get("refi_originations", 0) / allorig * 100 if allorig else np.nan,
            "cashout_share_pct": a1.get("cashout_originations", 0) / allorig * 100 if allorig else np.nan,
            "government_purchase_share_pct": a1["_gov_num"] / a1["_gov_den"] * 100 if a1.get("_gov_den") else np.nan,
            "avg_loan_amount": a1.get("purchase_volume", 0) / origs if origs else np.nan,
        }
        counts = pd.Series(lender_counts.get(key, {}), dtype=float)
        counts = counts[counts > 0]
        if len(counts):
            share = counts / counts.sum() * 100
            rec["active_lenders"] = float(len(counts))
            rec["lender_hhi"] = float((share ** 2).sum())
            rec["top5_lender_share_pct"] = float(share.sort_values(ascending=False).head(5).sum())
        recs.append(rec)
    out = pd.DataFrame(recs)
    log("hmda", years=[first, latest], geographies=len(geos), requests=len(jobs) + len(geos),
        markets=len(out), activity_failures=len(failures), filer_failures=filer_fail,
        failure_examples=failures[:5], sample_filers=sample_filers, seconds=round(time.time() - t0))
    if len(failures) > 0.2 * len(jobs):
        warn(f"{len(failures)} of {len(jobs)} HMDA activity requests failed.")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-hmda", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = "ok"
    try:
        markets, divisions, counties = load_popest()
        markets["demographic_opportunity_score"] = demographic_score(markets)

        parts = []
        for name, loader in (("acs", load_acs), ("fhfa", lambda: load_fhfa(divisions))):
            try:
                parts.append(loader())
            except Exception as exc:  # noqa: BLE001
                warn(f"{name} failed: {exc}")
                LOG["steps"].setdefault(name, {})["error"] = traceback.format_exc()[-1500:]
        if not args.skip_hmda:
            try:
                parts.append(load_hmda(markets, divisions, counties, args.workers))
            except Exception as exc:  # noqa: BLE001
                warn(f"hmda failed: {exc}")
                LOG["steps"].setdefault("hmda", {})["error"] = traceback.format_exc()[-1500:]

        for p in parts:
            if not p.empty and "market_key" in p.columns:
                p["market_key"] = p["market_key"].astype(str).str.zfill(5)
                markets = markets.merge(p, on="market_key", how="left")

        keep = [c for c in markets.columns if c not in ("population_first", "population_prior")]
        final = markets[keep].sort_values(["area_type", "population_latest"], ascending=[True, False])
        has_hmda = "purchase_applications_latest" in final.columns and final["purchase_applications_latest"].notna().sum() > 100
        if not has_hmda:
            status = "incomplete"
            warn("HMDA coverage too thin; markets.csv not replaced.")
        else:
            final.to_csv(out_dir / "markets.csv", index=False, float_format="%.6g")
            meta = {
                "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "markets": int(len(final)),
                "hmda_years": LOG["steps"].get("hmda", {}).get("years"),
                "fhfa_latest_year": LOG["steps"].get("fhfa", {}).get("latest_year"),
                "acs_year": LOG["steps"].get("acs", {}).get("latest"),
                "census_vintage": LOG["steps"].get("census_popest", {}).get("vintage"),
                "sources": ["U.S. Census Bureau Population Estimates", "American Community Survey 5-year",
                            "HMDA via FFIEC Data Browser API", "FHFA House Price Index"],
                "coverage": {c: int(final[c].notna().sum()) for c in final.columns if c not in ("market_key", "market_name")},
            }
            (out_dir / "markets_meta.json").write_text(json.dumps(meta, indent=2))
            years_df = LOG.pop("_market_years", None)
            if isinstance(years_df, pd.DataFrame) and not years_df.empty:
                years_df.sort_values(["market_key", "year"]).to_csv(out_dir / "market_years.csv", index=False, float_format="%.6g")
            lenders_df = LOG.pop("_lenders", None)
            if isinstance(lenders_df, pd.DataFrame) and not lenders_df.empty:
                lenders_df.to_csv(out_dir / "lenders.csv", index=False, float_format="%.5g")
            # Monthly snapshot for "what changed" comparisons.
            hist_dir = out_dir / "history"
            hist_dir.mkdir(exist_ok=True)
            final.to_csv(hist_dir / f"markets_{datetime.now(timezone.utc):%Y-%m}.csv", index=False, float_format="%.6g")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        LOG["fatal"] = traceback.format_exc()[-3000:]
        print(LOG["fatal"], file=sys.stderr)
    LOG.pop("_market_years", None)
    LOG.pop("_lenders", None)
    LOG["status"] = status
    LOG["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG["seconds"] = round(time.time() - started)
    (out_dir / "build_log.json").write_text(json.dumps(LOG, indent=2, default=str))
    print("STATUS:", status)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
