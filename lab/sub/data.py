"""Corvane Connect (fictional): a home internet and mobile provider with consumer and small-business plans.

Everything here is synthetic. Subscribers join over four years, each with a plan, a contract and a care history; a
monthly churn hazard (with spikes at contract and promotion ends) decides who leaves. Three randomized experiments
are planted so the lab can learn causal effects honestly: a price-increase test (month 18), a retention-offer
campaign (month 30) and a payment-reminder test (month 35). Truth is known here, which a real company never has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

NAME = "Corvane Connect"
N = 60_000
FIRST, LAST = -48, 35           # simulated calendar months; the reporting window is months 0–35
TODAY = 36

PERSONAS = {
    # share, monthly fee (mean, sd), base hazard logit, $-increase sensitivity, retention-offer effect (logit),
    # contract mix (none, 12, 24), autopay, mobile lines, usage GB, care calls / 90d, B2B, age
    "Bundled households": dict(share=.21, fee=(142, 22), a=-4.95, inc=.040, tau=-0.02, contract=(.15, .25, .60), autopay=.85, lines=2.6, usage=620, care=.7, b2b=0, age=44),
    "Promo switchers":    dict(share=.18, fee=(62, 9), a=-3.95, inc=.130, tau=-0.85, contract=(.30, .60, .10), autopay=.45, lines=.4, usage=380, care=.9, b2b=0, age=36),
    "Small business":     dict(share=.12, fee=(186, 40), a=-4.75, inc=.030, tau=-0.45, contract=(.05, .25, .70), autopay=.70, lines=3.4, usage=900, care=1.6, b2b=1, age=47),
    "Streamers & gamers": dict(share=.17, fee=(96, 14), a=-4.25, inc=.070, tau=-0.40, contract=(.40, .40, .20), autopay=.65, lines=.8, usage=1450, care=1.0, b2b=0, age=31),
    "Light users":        dict(share=.17, fee=(54, 8), a=-4.6, inc=.090, tau=+0.45, contract=(.75, .15, .10), autopay=.55, lines=.3, usage=120, care=1.1, b2b=0, age=66),
    "Movers & renters":   dict(share=.15, fee=(68, 10), a=-3.5, inc=.060, tau=-0.05, contract=(.70, .25, .05), autopay=.40, lines=.6, usage=520, care=.8, b2b=0, age=27),
}
REGIONS = ["Northeast", "Great Lakes", "Southeast", "Texas", "Mountain", "Pacific"]
REGION_OUTAGE = dict(zip(REGIONS, [3.0, 4.2, 5.5, 4.8, 2.6, 3.4]))
CHANNELS = ["Online", "Retail store", "Phone sales", "Partner"]
CAC = dict(zip(CHANNELS, [190, 330, 270, 420]))       # acquisition cost incl. install, $
TIERS = ["300 Mbps", "1 Gig", "2 Gig"]
DRIVERS = ["Billing question", "Price increase", "Outage or slow speed", "Equipment", "Moving", "Cancel request", "Upgrade inquiry"]
DRIVER_EFFECT = dict(zip(DRIVERS, [0.15, 0.55, 0.35, 0.10, 0.70, 1.25, -0.30]))
MARGIN = 0.55                                         # contribution margin on the monthly fee
CARE_COST = 9.0                                       # $ per agent-handled contact
PRICE_TEST_MONTH, CAMPAIGN_MONTH, REMINDER_MONTH = 18, 30, 35
INCREASES = [0, 3, 5, 8]
OFFER_COST = 90.0                                     # $15 a month off for six months


@dataclass
class Company:
    subs: pd.DataFrame        # one row per subscriber ever, with outcome history summaries
    monthly: pd.DataFrame     # calendar month: starts, churners, active, revenue
    price_test: pd.DataFrame
    campaign: pd.DataFrame
    reminder: pd.DataFrame
    notes: pd.DataFrame       # care-contact notes with their true driver


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


NOTE_BITS = {
    "Billing question": ["question about my bill", "charged twice this month", "why is my bill different", "autopay did not go through",
                         "explain the fees on my statement", "billing date change", "late fee on account"],
    "Price increase": ["my price went up", "rate increase notice", "bill went up again", "promo ended and price jumped",
                       "can you lower my monthly price", "new price is too high", "saw a better price elsewhere"],
    "Outage or slow speed": ["internet keeps dropping", "outage in my area", "speeds are slow at night", "wifi not working",
                             "service down since morning", "slow speed test results", "connection unstable during calls"],
    "Equipment": ["router not working", "need a new modem", "return equipment", "replace the gateway", "device lights blinking",
                  "mesh extender setup", "equipment fee question"],
    "Moving": ["moving to a new address", "transfer service to new apartment", "relocating next month", "new home setup",
               "moving out of state", "can I take service with me", "schedule install at new place"],
    "Cancel request": ["want to cancel service", "please cancel my account", "close my account", "switching providers",
                       "how do I cancel", "cancel at end of contract", "stop service"],
    "Upgrade inquiry": ["faster plan options", "upgrade to gig", "add a mobile line", "better wifi package", "what plans are faster",
                        "interested in higher speed", "add streaming bundle"],
}
FILLER = ["hi", "hello", "customer called", "chat:", "agent note:", "pls help", "asap", "thanks", "again", "today", "", "", ""]


@st.cache_resource(show_spinner="Building the synthetic subscriber base…")
def company(seed: int = 11) -> Company:
    rng = np.random.default_rng(seed)
    names = list(PERSONAS)
    persona = rng.choice(names, N, p=[PERSONAS[n]["share"] for n in names])
    P = {k: np.array([PERSONAS[p][k] for p in persona]) for k in ("a", "inc", "tau", "autopay", "lines", "usage", "care", "b2b", "age")}
    fee_mu = np.array([PERSONAS[p]["fee"][0] for p in persona])
    fee_sd = np.array([PERSONAS[p]["fee"][1] for p in persona])
    # Growth: more subscribers joined recently.
    months = np.arange(FIRST, LAST + 1)
    w = np.exp(0.012 * (months - FIRST))
    start = rng.choice(months, N, p=w / w.sum())
    contract = np.array([rng.choice([0, 12, 24], p=PERSONAS[p]["contract"]) for p in persona])
    promo = (persona == "Promo switchers") | (rng.random(N) < 0.08)
    region = rng.choice(REGIONS, N, p=[.18, .17, .2, .17, .12, .16])
    outage = rng.gamma(2.0, np.array([REGION_OUTAGE[r] for r in region]) / 2.0)
    autopay = rng.random(N) < P["autopay"]
    lines = rng.poisson(P["lines"])
    usage = rng.lognormal(np.log(P["usage"]), 0.45)
    tier_p = np.clip((np.log(usage) - np.log(250)) / 2.2, 0.02, 0.9)
    tier = np.where(rng.random(N) < tier_p, np.where(rng.random(N) < 0.35, "2 Gig", "1 Gig"), "300 Mbps")
    care = rng.poisson(P["care"] * (1 + outage / 10))
    repeat = rng.binomial(care, 0.25)
    late = rng.poisson(np.where(persona == "Movers & renters", 0.9, np.where(persona == "Promo switchers", 0.7, 0.35)))
    age = np.clip(rng.normal(P["age"], 9), 19, 90).round()
    channel = np.array([rng.choice(CHANNELS, p=(.45, .2, .2, .15) if b == 0 else (.2, .15, .4, .25)) for b in P["b2b"]])
    fee = np.clip(rng.normal(fee_mu, fee_sd), 30, 420) + 12 * (tier == "1 Gig") + 25 * (tier == "2 Gig")
    paperless = rng.random(N) < np.clip(0.9 - (age - 25) / 70, 0.15, 0.95)
    # Last care-contact driver (only for subscribers who called).
    wdrv = np.column_stack([
        0.5 + 0.6 * late + 0.4 * ~autopay, 0.3 + 1.5 * promo, 0.2 + outage / 3, np.full(N, 0.45),
        0.15 + 2.2 * (persona == "Movers & renters"), 0.15 + 0.12 * care, 0.2 + 1.4 * (usage > 1000) * (tier == "300 Mbps")])
    wdrv = wdrv / wdrv.sum(1, keepdims=True)
    drv_idx = (wdrv.cumsum(1) > rng.random(N)[:, None]).argmax(1)
    driver = np.where(care > 0, np.array(DRIVERS)[drv_idx], "No contact")
    drv_eff = np.array([DRIVER_EFFECT.get(d_, 0.0) for d_ in driver])

    # Experiments: who is in them is decided as the simulation reaches that month.
    increase = np.zeros(N)
    in_price_test = np.zeros(N, bool)
    in_campaign = np.zeros(N, bool)
    treated = np.zeros(N, bool)

    static = (P["a"] + 0.20 * np.minimum(care, 6) + 0.6 * ((care >= 3) & (outage > 5)) + 0.05 * outage * (1 + 1.6 * P["b2b"])
              + 0.12 * repeat - 0.45 * autopay - 0.22 * np.minimum(lines, 4) + 0.30 * np.minimum(late, 3) + drv_eff
              + 0.55 * ((usage > 1000) & (tier == "300 Mbps")) - 0.20 * paperless + 0.004 * (fee - fee_mu))
    alive = np.ones(N, bool)
    churn_month = np.full(N, 10_000)
    starts_by, churn_by, active_by, rev_by = [], [], [], []
    for m in months:
        live = alive & (start <= m)
        if m == PRICE_TEST_MONTH:
            pool = np.flatnonzero(live)
            pick = rng.choice(pool, int(len(pool) * 0.50), replace=False)
            in_price_test[pick] = True
            increase[pick] = rng.choice(INCREASES, len(pick))
        if m == CAMPAIGN_MONTH:
            pool = np.flatnonzero(live)
            pick = rng.choice(pool, int(len(pool) * 0.50), replace=False)
            in_campaign[pick] = True
            treated[pick] = rng.random(len(pick)) < 0.5
        k = m - start
        z = static - 0.33 * np.log1p(np.maximum(k, 0))
        z += 1.55 * ((contract > 0) & ((k == contract) | (k == contract + 1)))
        z += 1.35 * (promo & ((k == 12) | (k == 13)))
        if PRICE_TEST_MONTH <= m < PRICE_TEST_MONTH + 6:
            z += P["inc"] * increase * (1 + 0.15 * (persona == "Promo switchers") * (increase >= 5))
        if CAMPAIGN_MONTH <= m < CAMPAIGN_MONTH + 6:
            z += P["tau"] * treated
        h = _sigmoid(z)
        leave = live & (k >= 1) & (rng.random(N) < h)
        alive &= ~leave
        churn_month[leave] = m
        if m >= 0:
            starts_by.append(int((start == m).sum()))
            churn_by.append(int(leave.sum()))
            active_by.append(int((alive & (start <= m)).sum()))
            rev_by.append(float(fee[alive & (start <= m)].sum()))
    churned = churn_month <= LAST
    upgrade_z = -3.4 + 1.6 * (usage > 1000) * (tier == "300 Mbps") + 0.8 * (persona == "Streamers & gamers") + 0.4 * (tier == "1 Gig") - 0.02 * (age - 40)
    upgraded = rng.random(N) < _sigmoid(upgrade_z)

    subs = pd.DataFrame({
        "sub_id": [f"C-{i:05d}" for i in range(N)], "persona": persona, "start_month": start, "contract": contract, "promo": promo,
        "region": region, "channel": channel, "b2b": P["b2b"].astype(bool), "tier": tier, "fee": fee.round(2), "mobile_lines": lines,
        "usage_gb": usage.round(0), "autopay": autopay, "paperless": paperless, "care_calls_90d": care, "repeat_contacts": repeat,
        "outage_hrs_90d": outage.round(1), "late_payments_12m": late, "age": age, "last_driver": driver,
        "churn_month": np.where(churned, churn_month, -1), "churned": churned, "upgraded_12m": upgraded,
        "price_test": in_price_test, "increase": increase, "in_campaign": in_campaign, "treated": treated,
    })
    subs["tenure_today"] = np.where(churned, churn_month, TODAY) - start
    subs["cac"] = subs["channel"].map(CAC) * np.where(subs["b2b"], 1.4, 1.0)

    monthly = pd.DataFrame({"month": np.arange(0, LAST + 1), "starts": starts_by, "churners": churn_by, "active": active_by, "revenue": rev_by})
    active_start = monthly["active"].shift(1).fillna(monthly["active"].iloc[0])
    monthly["churn_rate"] = monthly["churners"] / active_start

    def churned_between(a, b):
        return (subs["churn_month"] >= a) & (subs["churn_month"] < b)

    pt = subs[subs["price_test"]].copy()
    pt["churned_6m"] = churned_between(PRICE_TEST_MONTH, PRICE_TEST_MONTH + 6)[pt.index]
    cp_ = subs[subs["in_campaign"]].copy()
    cp_["churned_6m"] = churned_between(CAMPAIGN_MONTH, CAMPAIGN_MONTH + 6)[cp_.index]

    # Payment reminders: past-due subscribers at month 35, half sent a reminder at random.
    act = subs[~subs["churned"]].copy()
    p_due = _sigmoid(-3.0 + 0.75 * act["late_payments_12m"] + 0.006 * (act["fee"] - 80) - 0.6 * act["autopay"])
    due = act[rng.random(len(act)) < p_due].copy()
    due["reminded"] = rng.random(len(due)) < 0.5
    lp = due["late_payments_12m"].to_numpy()
    z0 = -1.25 + 0.42 * np.minimum(lp, 5) + 0.004 * (due["fee"].to_numpy() - 80)
    eff = np.where(lp <= 2, -0.75, np.where(lp <= 3, -0.35, -0.05))
    due["disconnected_60d"] = rng.random(len(due)) < _sigmoid(z0 + eff * due["reminded"].to_numpy())
    due["balance_due"] = (due["fee"] * rng.uniform(1.0, 2.5, len(due))).round(2)

    # Care notes: the agent's free-text summary of the last contact.
    called = subs[subs["last_driver"] != "No contact"].sample(9000, random_state=1)
    notes = []
    for drv in called["last_driver"]:
        parts = list(rng.choice(NOTE_BITS[drv], 1 + int(rng.random() < 0.35), replace=False))
        if rng.random() < 0.18:  # some contacts mention a second topic, which is what makes classification hard
            other = rng.choice([d_ for d_ in DRIVERS if d_ != drv])
            parts.append(rng.choice(NOTE_BITS[other]))
        notes.append(" ".join([rng.choice(FILLER)] + parts).strip())
    notes = pd.DataFrame({"sub_id": called["sub_id"].to_numpy(), "note": notes, "driver": called["last_driver"].to_numpy()})
    return Company(subs, monthly, pt.reset_index(drop=True), cp_.reset_index(drop=True), due.reset_index(drop=True), notes)
