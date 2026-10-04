import time

import plotly.graph_objects as go
import streamlit as st

from lab import products as pr
from lab import theme as t


def render() -> None:
    t.header(
        "Decision systems · Product recommendation",
        "Which loan fits, and why?",
        "Eligibility first, then the comparison that matters to the borrower. Change the scenario and "
        "the engine re-checks every product, ranks the eligible ones by what they cost over the time you "
        "expect to keep the home, and explains every exclusion.",
    )

    with st.container(border=True):
        a, b, c, d = st.columns(4)
        price = a.number_input("Home price", 80_000, 3_000_000, 425_000, step=5_000, format="%d")
        down = b.slider("Down payment or equity (%)", 0.0, 50.0, 10.0, 0.5)
        credit = c.slider("Credit score", 500, 850, 712, 1)
        horizon = d.slider("Years you expect to keep the loan", 2, 30, 8)

        e, f, g, h = st.columns(4)
        income = e.number_input("Gross monthly income", 1_000, 100_000, 9_500, step=250, format="%d")
        debts = f.number_input("Other monthly debt payments", 0, 30_000, 650, step=50, format="%d")
        occupancy = g.selectbox("Occupancy", ["Primary residence", "Second home", "Investment"])
        market = h.slider("Reference 30-year fixed rate (%)", 3.0, 9.0, 6.4, 0.05)

        i, j, k, l = st.columns(4)
        military = i.toggle("Eligible military service", value=False)
        rural = j.toggle("Eligible rural area", value=False)
        first_time = k.toggle("First-time buyer", value=True)
        with l.popover("Assumptions", width="stretch"):
            goal = st.radio("Recommend the product with the", ["Lowest cost over my horizon", "Lowest monthly payment"], index=0)
            tax_ins = st.slider("Taxes and insurance (% of price per year)", 0.5, 4.0, 1.8, 0.1)
            arm_bump = st.slider("ARM rate change at reset (points)", -1.0, 4.0, 1.5, 0.25)

    borrower = pr.Borrower(
        price=price, down_pct=down, credit=credit, income_monthly=income, debts_monthly=debts,
        occupancy=occupancy, military=military, rural=rural, first_time=first_time,
        horizon_years=horizon, market_rate=market, tax_ins_rate=tax_ins, arm_reset_bump=arm_bump,
    )

    started = time.perf_counter()
    results = pr.rank(pr.evaluate(borrower), by="monthly" if goal == "Lowest monthly payment" else "horizon")
    elapsed_ms = (time.perf_counter() - started) * 1000
    eligible = [r for r in results if r.eligible]
    blocked = [r for r in results if not r.eligible]

    loan_base = price * (1 - down / 100)
    t.tiles(
        [
            {"label": "Base loan amount", "value": t.money(loan_base), "note": f"{100 - down:.1f}% loan-to-value"},
            {"label": "Products eligible", "value": f"{len(eligible)} of {len(results)}", "accent": True},
            {"label": "Rules checked", "value": f"{pr.total_checks(results)}", "note": f"in {elapsed_ms:.1f} ms"},
            {"label": "Best monthly housing payment", "value": t.money(min((r.housing for r in eligible), default=0)) if eligible else "—", "note": "principal, interest, MI, taxes, insurance"},
        ]
    )

    if not eligible:
        st.markdown(
            '<div class="lab-reco"><div class="e">No eligible product</div>'
            '<div class="t">Nothing qualifies as entered.</div>'
            '<div class="d">The closest options and what blocks them are listed below. Usually it is one '
            "rule: a little more down payment, lower monthly debts, or a different occupancy changes the answer.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        best = eligible[0]
        runner = eligible[1] if len(eligible) > 1 else None
        if runner and goal == "Lowest monthly payment":
            edge = f"About {t.money(runner.housing - best.housing)} a month less than {t.esc(runner.name)}."
        elif runner:
            edge = f"About {t.money(runner.horizon_cost - best.horizon_cost)} less than {t.esc(runner.name)} over {horizon} years."
        else:
            edge = "The only product that clears every rule for this scenario."
        cheapest_monthly = min(eligible, key=lambda r: r.housing)
        if goal != "Lowest monthly payment" and cheapest_monthly is not best and best.housing - cheapest_monthly.housing > 150:
            edge += f" It costs {t.money(best.housing - cheapest_monthly.housing)} more a month than {t.esc(cheapest_monthly.name)}. Switch the goal under Assumptions if cash flow matters more."
        note = t.esc(" ".join(best.notes))
        st.markdown(
            f'<div class="lab-reco"><div class="e">Recommended for this scenario</div>'
            f'<div class="t">{t.esc(best.name)}</div>'
            f'<div class="d">{t.money(best.housing)}/month all-in at {best.rate:.2f}% · '
            f"{t.money(best.horizon_cost)} in interest, insurance and fees over {horizon} years. {edge} {note}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        if eligible:
            names = [r.name for r in eligible][::-1]
            monthly_goal = goal == "Lowest monthly payment"
            costs = [(r.housing if monthly_goal else r.horizon_cost) for r in eligible][::-1]
            colors = [t.S1 if r is eligible[0] else "#B9C5BF" for r in eligible][::-1]
            fig = go.Figure(
                go.Bar(
                    y=names,
                    x=costs,
                    orientation="h",
                    marker=dict(color=colors, cornerradius=4),
                    text=[t.money(v) for v in costs],
                    textposition="outside",
                    textfont=dict(size=12, color=t.GRAPHITE),
                    customdata=[[r.housing, r.rate] for r in eligible][::-1],
                    hovertemplate="%{y}<br>%{x:$,.0f}<br>%{customdata[0]:$,.0f}/month at %{customdata[1]:.2f}%<extra></extra>",
                    width=0.6,
                )
            )
            fig.update_layout(
                title=("Monthly housing payment · eligible products" if monthly_goal else f"Cost of borrowing over {horizon} years · eligible products"),
                xaxis_title=("Principal, interest, insurance, taxes" if monthly_goal else "Interest + mortgage insurance + financed fees"),
                xaxis=dict(tickprefix="$", tickformat="~s", range=[0, max(costs) * 1.22]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                bargap=0.3,
            )
            t.chart(fig, height=max(240, 70 + 52 * len(eligible)))

            rows = [
                {
                    "Product": r.name,
                    "Rate": f"{r.rate:.2f}%",
                    "Loan incl. fees": f"${r.loan:,.0f}",
                    "Monthly all-in": f"${r.housing:,.0f}",
                    "Debt-to-income": f"{r.dti:.1f}%",
                    f"Cost over {horizon} yrs": f"${r.horizon_cost:,.0f}",
                }
                for r in eligible
            ]
            st.dataframe(rows, hide_index=True)

    with right:
        st.markdown("**Why not the others**")
        if not blocked:
            st.caption("Every product in the catalog is eligible for this scenario.")
        for r in blocked:
            items = "".join(
                f'<li><span class="no">✕</span><span><b>{t.esc(c.rule)}.</b> {t.esc(c.detail)}</span></li>' for c in r.failed
            )
            st.markdown(
                f'<div class="lab-card" style="margin-bottom:12px;height:auto"><span class="lab-pill off">{t.esc(r.family)}</span>'
                f'<h3 style="font-size:1.12rem">{t.esc(r.name)}</h3><ul class="lab-checks">{items}</ul></div>',
                unsafe_allow_html=True,
            )

    with st.expander("See every check for every product"):
        for r in results:
            status = "Eligible" if r.eligible else "Not eligible"
            st.markdown(f"**{r.name}** · {status}")
            items = "".join(
                f'<li><span class="{"ok" if c.passed else "no"}">{"✓" if c.passed else "✕"}</span>'
                f"<span><b>{t.esc(c.rule)}.</b> {t.esc(c.detail)}</span></li>"
                for c in r.checks
            )
            st.markdown(f'<ul class="lab-checks">{items}</ul>', unsafe_allow_html=True)

    t.footnote(
        "Illustrative rules and rates only. This is not a quote, a pre-approval or underwriting guidance; real "
        "program limits, pricing and overlays differ by lender, location and date."
    )

