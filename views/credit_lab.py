"""Credit risk & loss lab: a fictional property-management lender, from SQL warehouse to a reserve that survives audit."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import progress as PG
from lab import theme as t
from lab import widgets as w
from lab.credit import data as D
from lab.credit import loss as X
from lab.credit import models as M
from lab.cu import ui

ui.css()
STAGES = ["Portfolio", "PD model", "Loss & reserve", "Scenarios", "Backtest", "Audit defense", "Ownership"]
ROOT = Path(__file__).resolve().parent.parent

t.header(
    "Flagship product · Calder Lane (fictional)",
    "Credit risk & loss: from the warehouse to a reserve that survives audit",
    "Calder Lane Residential manages about 120 apartment properties; its lending arm makes small installment loans "
    "to residents for move-in costs, deposits and furnishing. Nineteen thousand synthetic loans, four years of "
    "monthly performance and one question carried end to end: how much will this book lose, and can we prove it "
    "to an examiner?",
)


# ---------------------------------------------------------------------------
# Everything is computed once per session and cached: deterministic, seeded, no external calls.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the book, the warehouse and the models…")
def lab() -> dict:
    t0 = time.perf_counter()
    book = D.simulate()
    con = D.warehouse(book)
    vint = pd.read_sql(D.SQL_VINTAGE, con)
    roll = pd.read_sql(D.SQL_ROLL, con)
    dq = pd.read_sql(D.SQL_DQ, con)
    sql_n = int(con.execute("SELECT COUNT(*) FROM loans").fetchone()[0])
    sql_bal = float(con.execute(f"SELECT SUM(balance) FROM performance WHERE month = {D.CUTOFF}").fetchone()[0] or 0)
    champ, bench, (dev, oot, bk) = M.train(book.loans)
    overlay = M.macro_overlay(champ, dev)
    p_oot_adj = overlay.apply(champ.p_oot, M.forward_du(oot))
    lgd = X.fit_lgd(book.loans)
    curve = X.marginal_curve(book.loans, book.perf)
    beta_u, roll_series = X.macro_beta(book.loans, book.perf)
    act = M.frame(book.loans[book.loans.end_state == "Active"])
    p_act = champ.model.predict_proba(act[M.COLS])[:, 1]
    lgd_act = lgd.predict(act)
    base = X.ecl(act, p_act, lgd_act, curve, beta_u)
    rc, contrib = M.reason_codes(champ, bk.head(400))
    mono = M.monotone_check(champ, dev.sample(1500, random_state=1))
    return dict(book=book, vint=vint, roll=roll, dq=dq, sql_n=sql_n, sql_bal=sql_bal, champ=champ, bench=bench,
                dev=dev, oot=oot, bk=bk, overlay=overlay, p_oot_adj=p_oot_adj, lgd=lgd, curve=curve, beta_u=beta_u,
                roll_series=roll_series, act=act, p_act=p_act, lgd_act=lgd_act, base=base, rc=rc, contrib=contrib,
                mono=mono, build_s=time.perf_counter() - t0)


L = lab()
book, champ, bench, dev, oot = L["book"], L["champ"], L["bench"], L["dev"], L["oot"]


@st.cache_data(show_spinner=False)
def reserve_at(shock: float, lgd_mult: float = 1.0, pd_mult: float = 1.0) -> float:
    return float(X.ecl(L["act"], L["p_act"], L["lgd_act"], L["curve"], L["beta_u"], shock=shock,
                       lgd_mult=lgd_mult, pd_mult=pd_mult)["ecl"].sum())


# Stage state (linkable with ?stage=).
q = str(st.query_params.get("stage", "")).lower().replace("-", "").replace(" ", "")
lookup = {s.lower().replace(" ", "").replace("&", "").replace("-", ""): s for s in STAGES}
lookup.update({"pd": "PD model", "loss": "Loss & reserve", "reserve": "Loss & reserve", "audit": "Audit defense"})
if "cr_stage" not in st.session_state:
    st.session_state.cr_stage = lookup.get(q, STAGES[0])
    st.session_state.cr_stage__last = st.session_state.cr_stage

stage = w.segmented("Stage", STAGES, key="cr_stage", label_visibility="collapsed")
st.query_params["stage"] = stage.lower().replace(" & ", "-").replace(" ", "-")
idx = STAGES.index(stage)
PG.bar(idx, len(STAGES), stage)
st.write("")

loans = book.loans
active = loans[loans.end_state == "Active"]
base = L["base"]
ead_now = float(base["ead_now"].sum())


def fmt(x, dp=1):
    return f"{x * 100:.{dp}f}%"


def md_table(df: pd.DataFrame) -> str:
    """A Markdown table without the optional tabulate dependency."""
    cell = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def sql_block(title: str, sql: str, df: pd.DataFrame, rows: int = 8) -> None:
    with st.expander(f"SQL · {title}"):
        st.code(sql, language="sql")
        st.caption(f"{len(df):,} rows returned from the in-memory SQLite warehouse. First {rows}:")
        st.dataframe(df.head(rows), hide_index=True, width="stretch")


# ===========================================================================
if stage == "Portfolio":
    ui.stage_head(1, "Know the book", "What have we lent, and how is it performing?",
                  "Every number on this stage comes from SQL against a small warehouse (loans, monthly performance, "
                  "macro), the way a servicing team would query it. Vintage curves and roll rates are the two views a "
                  "credit committee reads first.", "diagnostic")
    co_rate = (loans.end_state == "Charged off").mean()
    t.tiles([
        {"label": "Loans originated", "value": f"{len(loans):,}", "note": "Jan 2021 – Dec 2024, three products"},
        {"label": "Open at cutoff", "value": f"{len(active):,}", "note": f"{t.money(ead_now)} outstanding, Jun 2025"},
        {"label": "Charged off to date", "value": fmt(co_rate), "accent": True, "note": f"{(loans.end_state == 'Charged off').sum():,} loans"},
        {"label": "Mean loss given default", "value": fmt(L['lgd'].mean_lgd, 0), "note": "after deposit and workout recovery"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Vintage curves: cumulative charge-off by months on book")
        v = L["vint"]
        fig = go.Figure()
        cols = {2021: t.S2, 2022: t.S3, 2023: t.BASE, 2024: t.S1}
        for yr, g in v.groupby("vintage"):
            fig.add_scatter(x=g.mob, y=g.cum_co_rate, name=str(yr), mode="lines", line=dict(width=2.4, color=cols.get(yr)))
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Months on book", yaxis_title="Cumulative charge-off",
                          legend=dict(orientation="h", y=1.1))
        t.chart(fig, 330)
    with b:
        st.markdown("##### Delinquency rate and unemployment")
        d = L["dq"][L["dq"].month >= 3]
        fig = go.Figure()
        fig.add_scatter(x=d.date, y=d.dq_rate, name="Balance delinquent", line=dict(color=t.S1, width=2.4))
        fig.add_scatter(x=d.date, y=d.unemployment / 100, name="Unemployment", yaxis="y2", line=dict(color=t.S2, width=2, dash="dot"))
        fig.update_layout(yaxis_tickformat=".0%", yaxis2=dict(overlaying="y", side="right", tickformat=".1%", showgrid=False),
                          legend=dict(orientation="h", y=1.1))
        t.chart(fig, 330)
    st.markdown("##### Roll rates: where each state goes next month")
    r = L["roll"].pivot_table(index="from_status", columns="to_status", values="share", fill_value=0)
    order = [s for s in D.STATES if s in r.columns]
    r = r.reindex(index=[s for s in ["Current", "30 DPD", "60 DPD"] if s in r.index], columns=order)
    fig = go.Figure(go.Heatmap(z=r.values, x=r.columns, y=r.index, colorscale=[[0, t.CARD], [0.15, "#2E4A2A"], [1, t.S1]],
                               text=[[f"{v:.1%}" for v in row] for row in r.values], texttemplate="%{text}", showscale=False))
    fig.update_layout(yaxis_autorange="reversed", xaxis_title="Next month", yaxis_title="This month", margin=dict(t=10))
    t.chart(fig, 230)
    roll30 = float(r.loc["30 DPD", "60 DPD"]) if "60 DPD" in r.columns else 0
    t.insight(f"Once a loan reaches 30 days late, <b>{roll30:.0%}</b> roll to 60 the next month; the 2021 vintage, booked "
              "while unemployment was still high, runs above every later vintage at the same age. That gap is what the "
              "PD model and the macro overlay have to explain.")
    sql_block("vintage curves (window function)", D.SQL_VINTAGE, L["vint"])
    sql_block("roll rates (partitioned shares)", D.SQL_ROLL, L["roll"])
    sql_block("servicing dashboard", D.SQL_DQ, L["dq"])

# ===========================================================================
elif stage == "PD model":
    ui.stage_head(2, "Probability of default", "How likely is each loan to charge off within 12 months?",
                  "A gradient-boosted model is only worth running if it beats a transparent benchmark out of time and "
                  "still behaves like credit sense. Here XGBoost carries monotone constraints on every risk driver, is "
                  "calibrated on held-back data, and is tested on a vintage it never saw.", "build")
    ub = champ.extra["uncalibrated"]
    t.tiles([
        {"label": "Out-of-time AUC · champion", "value": f"{champ.auc_oot:.3f}", "accent": True, "note": f"Gini {champ.gini_oot:.2f}, KS {champ.ks_oot:.2f}"},
        {"label": "Logistic benchmark", "value": f"{bench.auc_oot:.3f}", "note": f"KS {bench.ks_oot:.2f}; the challenger to beat"},
        {"label": "Brier score", "value": f"{champ.brier_oot:.4f}", "note": f"benchmark {bench.brier_oot:.4f}; lower is better"},
        {"label": "Train / out-of-time gap", "value": f"{champ.auc_dev - champ.auc_oot:+.3f}", "note": "AUC drop; small means not over-fitted"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### ROC, 2023 vintage (out of time)")
        from sklearn.metrics import roc_curve
        fig = go.Figure()
        for f, c in [(champ, t.S1), (bench, t.S2)]:
            fx, fy, _ = roc_curve(oot.default_12m, f.p_oot)
            fig.add_scatter(x=fx, y=fy, name=f"{f.name.split(' (')[0]} · {f.auc_oot:.3f}", line=dict(color=c, width=2.4))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="No model", line=dict(color=t.BASE, dash="dot"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 330)
    with b:
        st.markdown("##### Calibration by decile, out of time")
        fig = go.Figure()
        for f, c, nm in [(champ, t.S1, "Champion (calibrated)"), (ub, t.S3, "Before calibration")]:
            cal = f.calib
            fig.add_scatter(x=cal.exp_rate, y=cal.obs_rate, mode="lines+markers", name=nm, line=dict(color=c, width=2))
        mx = float(max(champ.calib.obs_rate.max(), champ.calib.exp_rate.max())) * 1.05
        fig.add_scatter(x=[0, mx], y=[0, mx], name="Perfect", line=dict(color=t.BASE, dash="dot"))
        fig.update_layout(xaxis_title="Predicted PD", yaxis_title="Observed default rate", xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 330)
    st.markdown("##### Monotone constraints, tested on real loans")
    st.caption("Each driver is swept from its 5th to 95th percentile for 1,500 development loans; the average PD must move "
               "the stated way at every step. A reviewer can check each direction against credit sense.")
    mono = L["mono"].copy()
    mono["PD at low end"] = mono["PD at low end"].map(lambda v: f"{v:.1%}")
    mono["PD at high end"] = mono["PD at high end"].map(lambda v: f"{v:.1%}")
    mono["Holds at every step"] = mono["Holds at every step"].map({True: "✓ holds", False: "✗ breaks"})
    st.dataframe(mono, hide_index=True, width="stretch")
    st.markdown("##### Reason codes for a 2024 loan (TreeSHAP, log-odds)")
    bk = L["bk"].head(400).reset_index(drop=True)
    pb = champ.p_book[:400]
    order = np.argsort(-pb)
    opts = {f"Loan {int(bk.loan_id[i])} · PD {pb[i]:.1%} · grade {M.grade([pb[i]])[0]}": i for i in order[[0, 5, 40, 150, 399]]}
    pick = st.selectbox("Pick a loan", list(opts), key="cr_loan")
    i = opts[pick]
    c = L["contrib"][i]
    o = np.argsort(-np.abs(c))[:7][::-1]
    fig = go.Figure(go.Bar(x=c[o], y=[M.LABEL[M.COLS[j]] for j in o], orientation="h",
                           marker_color=[t.S2 if v > 0 else t.S1 for v in c[o]]))
    fig.update_layout(xaxis_title="Contribution to log-odds of default (coral raises risk)")
    t.chart(fig, 300)
    top = [f"{n}" for n, v in L["rc"][i] if v > 0][:3]
    ui.call("Adverse-action reasons come from the same contributions the model used, ranked; here: "
            f"<b>{t.esc(', '.join(top) or 'none above zero')}</b>. Because every driver is monotone, a reason can never "
            "say a better credit score raised the risk.")

# ===========================================================================
elif stage == "Loss & reserve":
    ui.stage_head(3, "Loss modeling", "How much will the open book lose over its life?",
                  "Lifetime expected loss (CECL-style) for every open loan: the marginal default curve by month on book, "
                  "scaled to the loan's own PD, times loss given default, times the scheduled balance, discounted at the "
                  "loan's rate. Each component is a model with its own evidence.", "build")
    lg = L["lgd"]
    t.tiles([
        {"label": "Lifetime ECL · baseline", "value": t.money(base.ecl.sum()), "accent": True, "note": f"on {len(base):,} open loans"},
        {"label": "Reserve rate", "value": fmt(base.ecl.sum() / ead_now, 2), "note": f"of {t.money(ead_now)} exposure"},
        {"label": "Average lifetime PD", "value": fmt(base.lifetime_pd.mean()), "note": "remaining life of each loan"},
        {"label": "LGD model error", "value": f"{lg.mae:.3f}", "note": f"MAE out of time; mean-only {lg.mae_base:.3f}"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Default timing: cumulative default by term")
        cv = L["curve"]
        fig = go.Figure()
        for term_k, c in zip([12, 24, 36], [t.S2, t.S3, t.S1]):
            g = cv[cv.term == term_k]
            fig.add_scatter(x=g.mob, y=g.cum_pd, name=f"{term_k}-month loans", line=dict(color=c, width=2.4))
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Cumulative default", xaxis_title="Months on book",
                          legend=dict(orientation="h", y=1.12))
        t.chart(fig, 320)
    with b:
        st.markdown("##### Loss given default by property class")
        dfl = loans[loans.lgd.notna()]
        fig = go.Figure()
        for k, c in zip(D.CLASSES, [t.S1, t.S3, t.S2]):
            fig.add_box(y=dfl.loc[dfl.property_class == k, "lgd"], name=f"Class {k}", marker_color=c, boxmean=True)
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Share of exposure lost", showlegend=False)
        t.chart(fig, 320)
    st.markdown("##### Reserve by grade")
    base_g = base.assign(grade=M.grade(base.pd12)).groupby("grade").agg(loans=("loan_id", "size"), ead=("ead_now", "sum"),
                                                                          ecl=("ecl", "sum"), pd12=("pd12", "mean"),
                                                                          lgd=("lgd_hat", "mean")).reset_index()
    base_g["rate"] = base_g.ecl / base_g.ead
    fig = go.Figure()
    fig.add_bar(x=base_g.grade, y=base_g.ecl, name="Lifetime ECL", marker_color=t.S1,
                text=[t.money(v) for v in base_g.ecl], textposition="outside")
    fig.add_scatter(x=base_g.grade, y=base_g.rate, name="Reserve rate", yaxis="y2", mode="lines+markers", line=dict(color=t.S2, width=2))
    fig.update_layout(yaxis_title="ECL", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False),
                      legend=dict(orientation="h", y=1.12), xaxis_title="Grade at cutoff")
    t.chart(fig, 300)
    with st.expander("How each component is built"):
        st.markdown(f"""
- **PD (12-month):** the calibrated, monotone XGBoost champion from stage 2.
- **Timing:** discrete hazard by term and month on book from the 2021–2022 vintages, smoothed over three months; each
  loan's curve is scaled in odds so its first-12-month cumulative matches its own PD.
- **LGD:** fractional logit (logistic quasi-likelihood on the share lost) on deposit coverage, score, verification and
  property class; trained on {lg.train_n:,} 2021–22 workouts, tested on {lg.test_n:,} later ones.
- **EAD:** scheduled amortizing balance in each future month (level-payment annuity), no further draws.
- **Discounting:** at each loan's APR, month by month.
""")
    ui.call(f"Class C properties lose the most when a loan defaults: thinner deposits and weaker workouts. The LGD model cuts "
            f"error by <b>{1 - lg.mae / lg.mae_base:.0%}</b> versus using the average, so the reserve moves with the mix of "
            "the book, not just its size.")

# ===========================================================================
elif stage == "Scenarios":
    ui.stage_head(4, "Scenarios & sensitivity", "What happens to the reserve if the economy turns?",
                  "The unemployment sensitivity is estimated from the book's own history (monthly roll from current to 30 "
                  "days late, fitted on aggregated counts). Scenarios hold for a 12-month reasonable-and-supportable "
                  "period, then revert straight-line to the long-run average over the next 12.", "advisory")
    with st.container(border=True):
        a, b, c = st.columns(3)
        wb = a.slider("Baseline weight", 0, 100, 60, 5, key="cr_wb")
        wa = b.slider("Adverse weight (+2.0 pts)", 0, 100, 30, 5, key="cr_wa")
        ws = c.slider("Severe weight (+4.5 pts)", 0, 100, 10, 5, key="cr_ws")
    tot = max(wb + wa + ws, 1)
    res = {s: reserve_at(sh) for s, sh in X.SCENARIOS.items()}
    weighted = (res["Baseline"] * wb + res["Adverse"] * wa + res["Severe"] * ws) / tot
    t.tiles([
        {"label": "Baseline", "value": t.money(res["Baseline"]), "note": fmt(res["Baseline"] / ead_now, 2) + " of exposure"},
        {"label": "Adverse", "value": t.money(res["Adverse"]), "note": fmt(res["Adverse"] / ead_now, 2)},
        {"label": "Severe", "value": t.money(res["Severe"]), "note": fmt(res["Severe"] / ead_now, 2)},
        {"label": "Probability-weighted reserve", "value": t.money(weighted), "accent": True,
         "note": f"weights {wb / tot:.0%} / {wa / tot:.0%} / {ws / tot:.0%}"},
    ])
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Sensitivity of the weighted reserve")
        def wres(**kw):
            r = {s: reserve_at(sh, **kw) for s, sh in X.SCENARIOS.items()}
            return (r["Baseline"] * wb + r["Adverse"] * wa + r["Severe"] * ws) / tot
        bars = []
        for name, lo, hi in [("PD ±20%", dict(pd_mult=0.8), dict(pd_mult=1.2)),
                             ("LGD ±20%", dict(lgd_mult=0.8), dict(lgd_mult=1.2))]:
            bars.append((name, wres(**lo) - weighted, wres(**hi) - weighted))
        sw_lo = (res["Baseline"] * (wb + 10) + res["Adverse"] * wa + res["Severe"] * max(ws - 10, 0)) / tot - weighted
        sw_hi = (res["Baseline"] * max(wb - 10, 0) + res["Adverse"] * wa + res["Severe"] * (ws + 10)) / tot - weighted
        bars.append(("Severe weight ±10 pts", sw_lo, sw_hi))
        bars.sort(key=lambda r: abs(r[2] - r[1]))
        fig = go.Figure()
        fig.add_bar(y=[r[0] for r in bars], x=[r[1] for r in bars], orientation="h", name="Down", marker_color=t.S1)
        fig.add_bar(y=[r[0] for r in bars], x=[r[2] for r in bars], orientation="h", name="Up", marker_color=t.S2)
        fig.update_layout(barmode="overlay", xaxis_title="Change in reserve ($)", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 300)
    with b:
        st.markdown("##### The macro link, estimated from the book")
        rs = L["roll_series"]
        fig = go.Figure(go.Scatter(x=rs.u, y=rs.rate, mode="markers", marker=dict(color=t.S1, size=8, opacity=.8),
                                   text=[D.month_label(m) for m in rs.month], hovertemplate="%{text}: %{y:.2%}<extra></extra>"))
        fig.update_layout(xaxis_title="Unemployment (%)", yaxis_title="Current → 30 DPD roll", yaxis_tickformat=".1%")
        t.chart(fig, 300)
    ui.call(f"Each point of unemployment multiplies the odds of a current loan going late by <b>{np.exp(L['beta_u']):.2f}×</b> "
            "(controlling for the age mix of the book). Weights are a management judgment, so the page shows how much "
            "the reserve moves with them: an auditor will ask, and the answer should already be on file.")

# ===========================================================================
elif stage == "Backtest":
    ui.stage_head(5, "Backtest", "Did the predictions come true?",
                  "The 2023 vintage has a full 12 months of outcomes. Each grade's predicted PD is tested against what "
                  "happened with a one-sided binomial test, and the whole curve with Hosmer–Lemeshow. Stability is "
                  "checked with the population stability index.", "advisory")
    y = oot.default_12m.values
    adj = L["p_oot_adj"]
    hl0 = (champ.hl_chi, champ.hl_p)
    hl1 = M.hosmer_lemeshow(y, adj)[:2]
    psi_b = M.psi(champ.p_dev, champ.p_book)
    t.tiles([
        {"label": "Hosmer–Lemeshow, as built", "value": f"p = {hl0[1]:.3f}", "note": f"χ² {hl0[0]:.1f}; fails at 5%"},
        {"label": "After macro overlay", "value": f"p = {hl1[1]:.3f}", "accent": True, "note": f"χ² {hl1[0]:.1f}"},
        {"label": "Predicted vs actual, 2023", "value": f"{adj.mean():.1%} / {y.mean():.1%}", "note": f"as built {champ.p_oot.mean():.1%}"},
        {"label": "PSI, development → 2024 book", "value": f"{psi_b:.3f}", "note": "below 0.10 is stable"},
    ])
    view = w.segmented("Version", ["After macro overlay", "As built"], key="cr_bt", label_visibility="collapsed")
    p_show = adj if view == "After macro overlay" else champ.p_oot
    bt = M.backtest_grades(y, p_show)
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        fig = go.Figure()
        fig.add_bar(x=bt.Grade, y=bt["Predicted PD"], name="Predicted", marker_color=t.S3)
        fig.add_bar(x=bt.Grade, y=bt["Observed"], name="Observed", marker_color=t.S1)
        fig.update_layout(barmode="group", yaxis_tickformat=".0%", xaxis_title="Grade", legend=dict(orientation="h", y=1.12))
        t.chart(fig, 320)
    with b:
        light = {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}
        show = bt.assign(**{"Predicted PD": bt["Predicted PD"].map(lambda v: f"{v:.1%}"),
                            "Observed": bt["Observed"].map(lambda v: f"{v:.1%}"),
                            "p-value": bt["p (under-prediction)"].map(lambda v: f"{v:.3f}"),
                            "Light": bt["Traffic light"].map(lambda v: f"{light[v]} {v}")})
        st.dataframe(show[["Grade", "Loans", "Predicted PD", "Observed", "p-value", "Light"]],
                     hide_index=True, width="stretch", height=35 * (len(show) + 1) + 3)
        st.caption("One-sided binomial test: the chance of seeing this many defaults if the predicted PD were right. "
                   "Green ≥ 0.05, amber ≥ 0.01, red below.")
    ov = L["overlay"]
    t.insight(f"As built, the model is point-in-time: it sees unemployment on the day of origination, not the softening "
              f"that followed in 2023. The overlay re-fits its log-odds with the forward change in unemployment on "
              f"<b>development data only</b> (coefficient {ov.beta_du:.3f} per point, slope on the score {ov.slope:.2f}), "
              "and the 2023 test then passes without ever having been used to fit.")

# ===========================================================================
elif stage == "Audit defense":
    ui.stage_head(6, "Audit defense", "Can we defend this model to an external reviewer?",
                  "Model risk management in practice: a standing test plan computed live from the code on this page, a "
                  "model inventory record, an assumptions log, and the findings an external validator would raise, each "
                  "with its evidence and remediation.", "advisory")
    y = oot.default_12m.values
    hl1 = M.hosmer_lemeshow(y, L["p_oot_adj"])[:2]
    bt_adj = M.backtest_grades(y, L["p_oot_adj"])
    psi_b = M.psi(champ.p_dev, champ.p_book)
    re_hash = D.simulate().hash
    tests = [
        ("Discrimination", "Out-of-time AUC ≥ 0.75", f"{champ.auc_oot:.3f}", champ.auc_oot >= 0.75),
        ("Benchmark", "Champion beats logistic out of time", f"{champ.auc_oot:.3f} vs {bench.auc_oot:.3f}", champ.auc_oot > bench.auc_oot),
        ("Over-fitting", "Train → out-of-time AUC drop < 0.03", f"{champ.auc_dev - champ.auc_oot:.3f}", champ.auc_dev - champ.auc_oot < 0.03),
        ("Calibration", "Hosmer–Lemeshow p ≥ 0.05 (with overlay)", f"{hl1[1]:.3f}", hl1[1] >= 0.05),
        ("Grade backtest", "No grade Red (with overlay)", f'{int((bt_adj["Traffic light"] == "Green").sum())}/{len(bt_adj)} grades Green',
         not (bt_adj["Traffic light"] == "Red").any()),
        ("Conceptual soundness", "Every monotone constraint holds", f"{int(L['mono']['Holds at every step'].sum())}/{len(L['mono'])}",
         bool(L["mono"]["Holds at every step"].all())),
        ("Stability", "Score PSI development → book < 0.10", f"{psi_b:.3f}", psi_b < 0.10),
        ("LGD", "Beats mean-only benchmark out of time", f"{L['lgd'].mae:.3f} vs {L['lgd'].mae_base:.3f}", L["lgd"].mae < L["lgd"].mae_base),
        ("Data reconciliation", "SQL loan count = model data", f"{L['sql_n']:,} = {len(loans):,}", L["sql_n"] == len(loans)),
        ("Reproducibility", "Re-simulated data hash matches", f"{re_hash} = {book.hash}", re_hash == book.hash),
    ]
    passed = sum(1 for r in tests if r[3])
    t.tiles([
        {"label": "Validation tests passing", "value": f"{passed}/{len(tests)}", "accent": True, "note": "recomputed on every load"},
        {"label": "Open findings", "value": "0", "note": "4 raised, 4 remediated with evidence"},
        {"label": "Data fingerprint", "value": book.hash, "note": "SHA-256 of the loan tape"},
        {"label": "Rebuild time", "value": f"{L['build_s']:.1f}s", "note": "data, SQL, models and reserve, end to end"},
    ])
    st.markdown("##### Standing test plan")
    tp = pd.DataFrame([{"Area": a, "Test": b, "Result": c, "Status": "✓ Pass" if d else "✗ Fail"} for a, b, c, d in tests])
    st.dataframe(tp, hide_index=True, width="stretch", height=35 * (len(tp) + 1) + 3)

    st.markdown("##### External validation findings")
    findings = [
        ("F-1 · High", "Out-of-time calibration fails",
         f"Hosmer–Lemeshow p = {champ.hl_p:.3f} on the 2023 vintage; predicted {champ.p_oot.mean():.1%} vs actual {y.mean():.1%}.",
         "Root cause: point-in-time PD cannot see post-origination unemployment. Added a forward-unemployment overlay "
         f"fitted on development data only; re-test p = {hl1[1]:.3f}, all grades Green (stage 5)."),
        ("F-2 · Medium", "Black-box concern on gradient boosting",
         "A boosted model can learn relationships that contradict credit policy and cannot produce adverse-action reasons.",
         "Monotone constraints on all nine risk drivers, swept and tested on real loans; TreeSHAP reason codes; a "
         f"logistic benchmark kept as a standing challenger (AUC {bench.auc_oot:.3f})."),
        ("F-3 · Medium", "LGD relies on a short workout history",
         f"Only {L['lgd'].train_n:,} development workouts; recoveries on recent defaults are still maturing.",
         "LGD ±20% sensitivity disclosed with the reserve (stage 4); mean-only benchmark tracked; refit scheduled "
         "when 2023 workouts close."),
        ("F-4 · Low", "Reproducibility and lineage",
         "Reviewer must be able to re-create every figure from source.",
         "Seeded simulation, data fingerprint, SQL reconciliation test and a one-file validation pack (below); "
         "same inputs give the same reserve to the cent."),
    ]
    for code, title, obs, rem in findings:
        st.markdown(
            f'<div class="lab-card" style="margin-bottom:10px"><h4 style="margin:0 0 6px;color:{t.S2 if "High" in code else t.MUTED};'
            f'font-size:.72rem;letter-spacing:.14em;text-transform:uppercase">{t.esc(code)} · Closed</h4>'
            f'<h3 style="font-size:1.1rem">{t.esc(title)}</h3>'
            f'<p><b style="color:{t.INK}">Observation.</b> {t.esc(obs)}</p>'
            f'<p style="margin-top:6px"><b style="color:{t.S1}">Remediation and evidence.</b> {t.esc(rem)}</p></div>',
            unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Model inventory record")
        st.markdown(f"""
- **Model:** CL-PD-01, resident installment loan PD (12-month), with CL-LGD-01 and CL-ECL-01
- **Use:** origination grade, pricing tier, lifetime loss reserve
- **Tier:** 1 (feeds the financial statements)
- **Technique:** XGBoost, monotone, Platt-calibrated; logistic challenger; fractional-logit LGD
- **Development data:** {len(dev):,} loans, 2021–22 vintages; out of time {len(oot):,} (2023)
- **Owner:** credit risk analytics · **Validator:** independent, annual · **Monitoring:** monthly PSI and backtest
""")
    with b:
        st.markdown("##### Assumptions log")
        st.markdown(f"""
- 12-month default = charge-off at 90+ days past due within 12 months on book
- Reasonable-and-supportable period {X.SUPPORTABLE} months, straight-line reversion over the next {X.SUPPORTABLE}
- Scenario shocks: adverse +{X.SCENARIOS['Adverse']:.1f} pts, severe +{X.SCENARIOS['Severe']:.1f} pts unemployment
- Deposits applied to the balance before workout recovery
- Exposure is the scheduled balance; no prepayment assumed in the reserve (conservative)
- All data synthetic; Calder Lane is fictional
""")
    pack = [f"# Calder Lane · validation pack\n\nData fingerprint: {book.hash}\n\n## Test plan\n",
            md_table(tp), "\n\n## Grade backtest (2023, with overlay)\n", md_table(bt_adj),
            "\n\n## Monotone check\n", md_table(L["mono"]),
            "\n\n## Findings\n", "\n".join(f"- **{c} {ti}.** {o} Remediation: {r}" for c, ti, o, r in findings),
            "\n\n## Reserve (baseline)\n", f"Lifetime ECL {base.ecl.sum():,.2f} on exposure {ead_now:,.2f}\n"]
    st.download_button("Download the validation pack (Markdown)", "".join(pack), file_name=f"calder-lane-validation-{book.hash}.md",
                       mime="text/markdown", type="primary")

# ===========================================================================
elif stage == "Ownership":
    ui.stage_head(7, "End-to-end ownership", "Who built this, and what did it take?",
                  "One person, no direct reports, every layer: data design, SQL, statistics, the boosted model, the loss "
                  "engine, the validation and the audit response. The point is technical depth carried all the way to "
                  "a number the business books.", "build")

    def loc(paths, ext):
        n = 0
        for p in paths:
            for f in (ROOT / p).glob(f"*{ext}") if (ROOT / p).is_dir() else [ROOT / p]:
                n += sum(1 for line in f.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))
        return n
    py = loc(["lab/credit", "views/credit_lab.py"], ".py")
    sql_lines = sum(len([x for x in s.splitlines() if x.strip()]) for s in [D.SQL_VINTAGE, D.SQL_ROLL, D.SQL_DQ])
    t.tiles([
        {"label": "Python", "value": f"{py:,} lines", "note": "pandas, NumPy, SciPy, scikit-learn, XGBoost"},
        {"label": "SQL", "value": f"{sql_lines} lines", "note": "CTEs, window functions, partitioned shares"},
        {"label": "Statistical tests", "value": "8", "note": "AUC, KS, Brier, HL, binomial, PSI, monotone, MAE"},
        {"label": "Team", "value": "1", "accent": True, "note": "no direct reports; pure technical delivery"},
    ])
    steps = [
        ("Data & warehouse", "Designed the loan tape and monthly performance, loaded a SQL warehouse, wrote the vintage, roll-rate and servicing queries."),
        ("PD model", "Monotone XGBoost with a held-out calibration step, a logistic challenger and an out-of-time vintage."),
        ("Loss engine", "Hazard timing, fractional-logit LGD, amortizing exposure, discounting: lifetime ECL per loan."),
        ("Scenarios", "Macro sensitivity estimated from the book, probability-weighted scenarios with reversion, sensitivity on every input."),
        ("Validation", "A standing test plan that recomputes on every load, with a reproducibility hash and SQL reconciliation."),
        ("Audit response", "Findings answered with root cause, remediation and evidence; a validation pack an examiner can keep."),
    ]
    ui.cards(steps)
    ui.call("The impact is not the AUC. It is a reserve the controller can book, a reason code a declined resident can "
            "read, and a file that answers the examiner's questions before they are asked.")

# ---------------------------------------------------------------------------
st.write("")
prev_col, _, next_col = st.columns([1, 2, 1])


def _go(s: str) -> None:
    st.session_state.cr_stage = s


if idx > 0:
    prev_col.button(f"← {STAGES[idx - 1]}", on_click=_go, args=(STAGES[idx - 1],), width="stretch")
if idx < len(STAGES) - 1:
    next_col.button(f"{STAGES[idx + 1]} →", on_click=_go, args=(STAGES[idx + 1],), type="primary", width="stretch")

t.footnote("Calder Lane is fictional. All loans, residents, properties and outcomes are synthetic; the models, reserve and "
           "audit findings are illustrative and are not accounting, credit or regulatory advice.")
