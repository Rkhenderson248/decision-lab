"""Human–AI decision lab: a judge–advisor experiment you take yourself."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lab import theme as t

ROUNDS = 10
FEATURES = ["Deal size", "Stage", "Days in stage", "Champion engaged", "Competitor in the deal", "Existing customer"]

t.header(
    "Human–AI collaboration · Research in practice",
    "Do you know when to trust the model?",
    "Ten sales deals. For each, estimate the chance it closes this quarter, then see a model's estimate "
    "and decide whether to change your mind. At the end you get your reliance profile: when you followed "
    "the model and should have, when you followed it and shouldn't have, and how much weight you gave it.",
)


def make_cases(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(ROUNDS):
        size = float(np.round(np.exp(rng.normal(np.log(60_000), 0.7)), -3))
        stage = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]))
        days = int(np.clip(rng.gamma(2.2, 14), 2, 120))
        champ = bool(rng.random() < 0.55)
        comp = bool(rng.random() < 0.45)
        existing = bool(rng.random() < 0.35)
        logit = (-1.6 + 0.75 * stage - 0.022 * days + 1.0 * champ - 0.7 * comp + 0.6 * existing
                 - 0.25 * np.log(size / 60_000) + rng.normal(0, 0.7))
        p_true = 1 / (1 + np.exp(-logit))
        closed = bool(rng.random() < p_true)
        # The model sees the same signals but not everything; occasionally it is confidently wrong.
        model_logit = logit + rng.normal(0, 0.55) + (1.4 * (1 if not closed else -1) if rng.random() < 0.18 else 0)
        p_ai = float(np.clip(1 / (1 + np.exp(-model_logit)), 0.03, 0.97))
        drivers = sorted([
            ("Late stage" if stage >= 3 else "Early stage", 0.75 * (stage - 2.5)),
            ("Champion engaged" if champ else "No champion", 1.0 * (1 if champ else -1) * 0.5),
            ("Competitor present" if comp else "No competitor", -0.35 if comp else 0.35),
            ("Existing customer" if existing else "New logo", 0.3 if existing else -0.3),
            (f"{days} days in stage", -0.022 * (days - 30)),
        ], key=lambda x: -abs(x[1]))[:3]
        rows.append({
            "id": i + 1, "size": size, "stage": stage, "days": days, "champ": champ, "comp": comp,
            "existing": existing, "p_true": p_true, "closed": closed, "p_ai": p_ai,
            "explained": bool(i % 2 == 1), "drivers": drivers,
        })
    return pd.DataFrame(rows)


def reset():
    seed = int(np.random.default_rng().integers(1, 1_000_000))
    st.session_state.hai = {"seed": seed, "cases": make_cases(seed), "i": 0, "stage": "initial", "answers": []}


if "hai" not in st.session_state:
    reset()
state = st.session_state.hai

tab_play, tab_theory = st.tabs(["Take the experiment", "Why reliance is a design problem"])

STAGES = {1: "Discovery", 2: "Proposal", 3: "Negotiation", 4: "Verbal yes"}

with tab_play:
    cases = state["cases"]
    if state["i"] < ROUNDS:
        case = cases.iloc[state["i"]]
        st.progress(state["i"] / ROUNDS, text=f"Deal {state['i'] + 1} of {ROUNDS}")
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            facts = [
                ("Deal size", f"${case['size']:,.0f}"), ("Stage", STAGES[int(case["stage"])]),
                ("Days in current stage", f"{int(case['days'])}"), ("Champion engaged", "Yes" if case["champ"] else "No"),
                ("Competitor in the deal", "Yes" if case["comp"] else "No"), ("Existing customer", "Yes" if case["existing"] else "No"),
            ]
            items = "".join(f'<li><span class="ok">·</span><span><b>{k}:</b> {t.esc(v)}</span></li>' for k, v in facts)
            st.markdown(f'<div class="lab-card" style="height:auto"><p class="lab-eyebrow">Deal {int(case["id"])}</p>'
                        f'<h3 style="font-size:1.3rem">Will it close this quarter?</h3><ul class="lab-checks">{items}</ul></div>',
                        unsafe_allow_html=True)
        with right:
            if state["stage"] == "initial":
                st.markdown("**Your estimate first, before any advice.**")
                guess = st.slider("Chance this deal closes (%)", 0, 100, 50, 5, key=f"init_{state['i']}")
                if st.button("Lock in my estimate", type="primary"):
                    state["initial"] = guess
                    state["stage"] = "advised"
                    st.rerun()
            elif state["stage"] == "advised":
                ai = case["p_ai"] * 100
                st.markdown(f"Your estimate: **{state['initial']}%**")
                expl = ""
                if case["explained"]:
                    expl = "<br>".join(f"{'↑' if v > 0 else '↓'} {t.esc(k)}" for k, v in case["drivers"])
                    expl = f'<div class="d" style="margin-top:8px">Why: <br>{expl}</div>'
                st.markdown(f'<div class="lab-reco"><div class="e">Model estimate</div><div class="t">{ai:.0f}% chance of closing</div>'
                            f'{expl}</div>', unsafe_allow_html=True)
                st.write("")
                final = st.slider("Your final estimate (%)", 0, 100, int(state["initial"]), 5, key=f"final_{state['i']}")
                if st.button("Submit final estimate", type="primary"):
                    state["answers"].append({"round": int(case["id"]), "initial": state["initial"] / 100, "ai": case["p_ai"],
                                             "final": final / 100, "closed": bool(case["closed"]), "explained": bool(case["explained"])})
                    state["stage"] = "feedback"
                    st.rerun()
            else:
                last = state["answers"][-1]
                outcome = "closed" if last["closed"] else "did not close"
                st.markdown(f"The deal **{outcome}**. You said **{last['final'] * 100:.0f}%**; the model said **{last['ai'] * 100:.0f}%**.")
                if st.button("Next deal" if state["i"] < ROUNDS - 1 else "See my results", type="primary"):
                    state["i"] += 1
                    state["stage"] = "initial"
                    st.rerun()
    else:
        ans = pd.DataFrame(state["answers"])
        y = ans["closed"].astype(float)
        brier = {k: float(((ans[k] - y) ** 2).mean()) for k in ("initial", "ai", "final")}
        moved = ans["ai"] - ans["initial"]
        woa = np.where(moved.abs() > 0.02, (ans["final"] - ans["initial"]) / moved.where(moved.abs() > 0.02), np.nan)
        ans["woa"] = np.clip(woa, -0.5, 1.5)
        mean_woa = float(np.nanmean(ans["woa"])) if np.isfinite(ans["woa"]).any() else float("nan")

        # Binary view at 50%: did you end up on the model's side, and was the model right?
        ai_right = (ans["ai"] >= 0.5) == ans["closed"]
        human_init_side = ans["initial"] >= 0.5
        ai_side = ans["ai"] >= 0.5
        final_side = ans["final"] >= 0.5
        disagree = human_init_side != ai_side
        switched = disagree & (final_side == ai_side)
        cats = pd.Series("Agreed from the start", index=ans.index)
        cats[disagree & switched & ai_right] = "Appropriate reliance"
        cats[disagree & switched & ~ai_right] = "Over-reliance"
        cats[disagree & ~switched & ai_right] = "Under-reliance"
        cats[disagree & ~switched & ~ai_right] = "Appropriate self-reliance"
        ans["pattern"] = cats

        best = min(brier, key=brier.get)
        verdict = {"final": "You plus the model beat both of you alone.",
                   "ai": "The model alone would have done better than your final answers.",
                   "initial": "Your first instincts beat both the model and your revised answers."}[best]
        t.tiles([
            {"label": "Your first estimates", "value": f"{brier['initial']:.3f}", "note": "Brier score, lower is better"},
            {"label": "The model", "value": f"{brier['ai']:.3f}"},
            {"label": "Your final estimates", "value": f"{brier['final']:.3f}", "accent": True},
            {"label": "Weight you gave the model", "value": f"{mean_woa * 100:.0f}%" if np.isfinite(mean_woa) else "n/a",
             "note": "people typically give advice 20–40%"},
        ])
        t.insight(f"<b>{verdict}</b> A weight of 0% means you ignored the model; 100% means you adopted its number.")

        left, right = st.columns([1.2, 1], gap="large")
        with left:
            fig = go.Figure()
            x = ans["round"]
            fig.add_trace(go.Scatter(x=x, y=ans["initial"] * 100, mode="markers", name="Your first estimate",
                                     marker=dict(size=10, color=t.BASE, symbol="circle-open", line=dict(width=2))))
            fig.add_trace(go.Scatter(x=x, y=ans["ai"] * 100, mode="markers", name="Model",
                                     marker=dict(size=10, color=t.S2, symbol="diamond")))
            fig.add_trace(go.Scatter(x=x, y=ans["final"] * 100, mode="markers", name="Your final",
                                     marker=dict(size=11, color=t.S1)))
            for _, r in ans.iterrows():
                fig.add_annotation(x=r["round"], y=104, text="✓" if r["closed"] else "✕", showarrow=False,
                                   font=dict(size=13, color=t.PETROL if r["closed"] else t.MUTED))
            fig.add_hline(y=50, line=dict(color=t.GRID, width=1))
            fig.update_layout(title="Each deal: your first call, the model, your final call (✓ closed)",
                              xaxis=dict(title="Deal", dtick=1), yaxis=dict(title="Chance of closing (%)", range=[-2, 110]))
            t.chart(fig, height=380)
        with right:
            order = ["Appropriate reliance", "Appropriate self-reliance", "Over-reliance", "Under-reliance", "Agreed from the start"]
            counts = ans["pattern"].value_counts().reindex(order).fillna(0).astype(int)
            fig2 = go.Figure(go.Bar(y=counts.index[::-1], x=counts.values[::-1], orientation="h",
                                    marker=dict(color=[t.S1, t.S1, t.S2, t.S2, t.BASE][::-1], cornerradius=4),
                                    text=counts.values[::-1], textposition="outside", width=0.6,
                                    hovertemplate="%{y}: %{x}<extra></extra>"))
            fig2.update_layout(title=dict(text="Your reliance profile", x=0, xref="container"),
                               xaxis=dict(visible=False, range=[0, max(counts.max(), 1) * 1.3]),
                               yaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(l=8, r=8, t=60, b=8))
            t.chart(fig2, height=380)

        ex = ans.groupby("explained")["woa"].mean()
        if True in ex.index and False in ex.index and np.isfinite(ex).all():
            more = "more" if ex[True] > ex[False] else "less"
            st.markdown(f"**Explanations changed your behaviour.** On deals where the model showed its reasons you gave it "
                        f"**{ex[True] * 100:.0f}%** weight, against **{ex[False] * 100:.0f}%** without: {more} reliance. "
                        "Explanations raise trust, but they raise it whether or not the model is right. That is the "
                        "design problem.")
        with st.expander("What these terms mean"):
            st.markdown("""
- **Appropriate reliance:** you disagreed, switched to the model, and the model was right.
- **Over-reliance:** you switched to the model and it was wrong.
- **Under-reliance:** you kept your view and the model was right.
- **Weight of advice:** how far you moved toward the model, as a share of the gap between you.
- **Brier score:** mean squared error of a probability. 0 is perfect; always saying 50% scores 0.25.
""")
        if st.button("Take it again with new deals"):
            reset()
            st.rerun()

with tab_theory:
    st.markdown("Combining a person and a model only helps if each is weighted by what they bring. The optimal weight "
                "depends on how accurate each is and how correlated their mistakes are. This is why simply adding a "
                "model to a workflow rarely delivers its standalone accuracy.")
    with st.container(border=True):
        a, b, c = st.columns(3)
        sh = a.slider("Human error (spread)", 0.05, 0.40, 0.20, 0.01)
        sa = b.slider("Model error (spread)", 0.05, 0.40, 0.14, 0.01)
        r = c.slider("Correlation of their errors", -0.5, 0.95, 0.30, 0.05)
    w = np.linspace(0, 1, 101)
    mse = (1 - w) ** 2 * sh ** 2 + w ** 2 * sa ** 2 + 2 * w * (1 - w) * r * sh * sa
    denom = sh ** 2 + sa ** 2 - 2 * r * sh * sa
    w_star = float(np.clip((sh ** 2 - r * sh * sa) / denom, 0, 1)) if denom > 0 else 0.5
    best = float((1 - w_star) ** 2 * sh ** 2 + w_star ** 2 * sa ** 2 + 2 * w_star * (1 - w_star) * r * sh * sa)
    typical = 0.30
    mse_typ = float((1 - typical) ** 2 * sh ** 2 + typical ** 2 * sa ** 2 + 2 * typical * (1 - typical) * r * sh * sa)
    t.tiles([
        {"label": "Best weight on the model", "value": f"{w_star * 100:.0f}%", "accent": True},
        {"label": "Error at best weight vs model alone", "value": t.pct(best / sa ** 2 - 1, 0, signed=True)},
        {"label": "Error at a typical 30% weight vs best", "value": t.pct(mse_typ / best - 1, 0, signed=True)},
    ])
    fig = go.Figure(go.Scatter(x=w * 100, y=np.sqrt(mse), mode="lines", line=dict(color=t.S1, width=2.5),
                               hovertemplate="Weight %{x:.0f}% → error %{y:.3f}<extra></extra>", showlegend=False))
    fig.add_vline(x=w_star * 100, line=dict(color=t.PETROL, width=1))
    fig.add_vline(x=typical * 100, line=dict(color=t.S2, width=1))
    fig.add_annotation(x=typical * 100, y=1, yref="paper", text="typical person ", showarrow=False, xanchor="right",
                       yanchor="top", font=dict(size=12, color=t.S2))
    fig.add_annotation(x=w_star * 100, y=1, yref="paper", text=" optimal", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(size=12, color=t.PETROL))
    fig.update_layout(title="Combined error by the weight given to the model", xaxis_title="Weight on the model (%)",
                      yaxis_title="Error of the combined estimate (RMSE)")
    t.chart(fig, height=360)
    st.caption("Grounded in judge–advisor research (advice discounting, Bonaccio & Dalal 2006), algorithm aversion "
               "(Dietvorst, Simmons & Massey 2015) and forecast combination theory. This is the subject of R.K. "
               "Henderson's doctoral research on human–AI collaboration in managerial decisions.")

t.footnote("Deals are synthetic and generated fresh for each visitor. Nothing you enter is stored or sent anywhere.")
