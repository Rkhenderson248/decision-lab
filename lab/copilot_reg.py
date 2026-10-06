"""The linear regressions behind the copilot's price response, laid out in full (new module for clean redeploys)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lab import copilot as cp


def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    return beta, resid, s2


@st.cache_data(show_spinner=False)
def regression(segment: str = "Leisure", seed: int = 21) -> dict:
    """Two-stage least squares written out as the two linear regressions it is.

    Stage 1: log rate on the random test offset plus calendar controls (how much of the price was set at random).
    Stage 2: log rooms on the stage-1 fitted log rate plus the same controls. Its slope is the elasticity.
    """
    h = cp.rate_history(seed)
    d = h[~h["sold_out"] & (h[segment.lower()] > 0)].copy()
    y = np.log(d[segment.lower()].to_numpy(float))
    lp = np.log(d["rate"].to_numpy())
    dow = pd.get_dummies(d["dow"], prefix="d", drop_first=True, dtype=float)
    C = np.column_stack([np.ones(len(d)), dow.to_numpy(), d[["sin", "cos", "t"]].to_numpy()])
    cnames = ["Intercept"] + [f"Weekday {c.split('_')[1]}" for c in dow.columns] + ["Season (sin)", "Season (cos)", "Trend (per year)"]

    Z = np.column_stack([C, d["offset"].to_numpy()])
    g, r1, s21 = _ols(lp, Z)
    _, r0, _ = _ols(lp, C)
    q = 1
    f_stat = ((r0 @ r0 - r1 @ r1) / q) / (r1 @ r1 / (len(lp) - Z.shape[1]))
    lp_hat = Z @ g

    X2 = np.column_stack([C, lp_hat])
    b2, _, _ = _ols(y, X2)
    resid = y - np.column_stack([C, lp]) @ b2          # structural residuals use the actual rate
    s2 = resid @ resid / (len(y) - X2.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X2.T @ X2)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    # Frisch–Waugh: strip the controls from both sides, leaving the pure price relationship to plot.
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    rx = lp_hat - C @ np.linalg.lstsq(C, lp_hat, rcond=None)[0]
    coefs = pd.DataFrame({"term": cnames + ["log(rate), instrumented"], "estimate": b2, "se": se, "t": b2 / se})
    coefs = pd.concat([coefs.iloc[[-1]], coefs.iloc[:-1]], ignore_index=True)
    return {"coefs": coefs, "r2": float(r2), "n": len(y), "rx": rx, "ry": ry, "slope": float(b2[-1]),
            "first_stage": float(g[-1]), "f_stat": float(f_stat), "truth": cp.SEGMENTS[segment][1]}
