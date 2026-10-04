"""Build the sample Decision Diagnostic report (PDF) from the lending lab's synthetic data.

    python scripts/sample_diagnostic.py OUTPUT.pdf FONT_DIR

Every figure in the report is computed here from Kestrel Valley Credit Union
(fictional), so the sample stays consistent with the live lab.
"""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

from lab.cu import data as d  # noqa: E402
from lab.cu import models as M  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "sample-decision-diagnostic.pdf")
FONTS = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/fonts")

PETROL = colors.HexColor("#0F4640")
INK = colors.HexColor("#0E1311")
GRAPHITE = colors.HexColor("#4A534D")
MUTED = colors.HexColor("#6B756E")
GRID = colors.HexColor("#E1E4DE")
MIST = colors.HexColor("#F1F2EE")
LUME = colors.HexColor("#7FD0BE")
ORANGE = "#D9772B"
TEAL = "#008A73"

pdfmetrics.registerFont(TTFont("Display", str(FONTS / "Bodoni-500.ttf")))
pdfmetrics.registerFont(TTFont("DisplayBold", str(FONTS / "Bodoni-600.ttf")))
pdfmetrics.registerFont(TTFont("DisplayItalic", str(FONTS / "BodoniItalic-500.ttf")))
pdfmetrics.registerFont(TTFont("Text", str(FONTS / "Schibsted-400.ttf")))
pdfmetrics.registerFont(TTFont("TextBold", str(FONTS / "Schibsted-600.ttf")))
pdfmetrics.registerFontFamily("Text", normal="Text", bold="TextBold", italic="Text", boldItalic="TextBold")
for f in ("Schibsted-400.ttf", "Schibsted-600.ttf"):
    font_manager.fontManager.addfont(str(FONTS / f))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONTS / "Schibsted-400.ttf")).get_name(),
                     "font.size": 9, "axes.edgecolor": "#E1E4DE", "axes.labelcolor": "#6B756E", "xtick.color": "#6B756E",
                     "ytick.color": "#6B756E", "axes.spines.top": False, "axes.spines.right": False})

S = {
    "eyebrow": ParagraphStyle("eyebrow", fontName="TextBold", fontSize=8, leading=11, textColor=PETROL, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Display", fontSize=30, leading=33, textColor=INK, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Display", fontSize=19, leading=23, textColor=INK, spaceBefore=6, spaceAfter=8),
    "h3": ParagraphStyle("h3", fontName="TextBold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Text", fontSize=10, leading=15, textColor=GRAPHITE, spaceAfter=7, alignment=TA_LEFT),
    "lede": ParagraphStyle("lede", fontName="Text", fontSize=12, leading=18, textColor=GRAPHITE, spaceAfter=10),
    "small": ParagraphStyle("small", fontName="Text", fontSize=8, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Text", fontSize=8.8, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="TextBold", fontSize=8.8, leading=12, textColor=INK),
    "big": ParagraphStyle("big", fontName="DisplayBold", fontSize=22, leading=24, textColor=PETROL),
    "callout": ParagraphStyle("callout", fontName="Text", fontSize=10, leading=15, textColor=INK),
}


def money(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def fig_to_image(fig, width_in: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_in * inch, width_in * inch * ratio
    return img


def table(rows, widths, header=True, zebra=False):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], hAlign="LEFT")
    style = [("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("LINEABOVE", (0, 0), (-1, 0), 1, PETROL), ("BACKGROUND", (0, 0), (-1, 0), MIST)]
    t.setStyle(TableStyle(style))
    return t


def callout(text: str) -> Table:
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 2.5, PETROL), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F8F5")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return t


def kpis(items) -> Table:
    cells = [[Paragraph(v, S["big"]) for _, v, _ in items], [Paragraph(k, S["cellb"]) for k, _, _ in items],
             [Paragraph(n, S["small"]) for _, _, n in items]]
    t = Table(cells, colWidths=[6.5 / len(items) * inch] * len(items))
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.5, GRID), ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
                           ("LINEBEFORE", (1, 0), (-1, -1), 0.5, GRID), ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    return t


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
NIM, LGD, OPEX, ADOPTION = 0.045, 0.6, 120, 0.7
U = M.underwriting()
te = U.test
ct = M.cutoff_table(te, nim=NIM, lgd=LGD, opex=OPEX)


def policy(mask, lgd=LGD):
    amt, bad = te.loc[mask, "amount"], te.loc[mask, "default_12m"]
    ga, gb = te["group_b"], ~te["group_b"]
    return {"approval": mask.mean(), "bad": bad.mean(), "n": int(mask.sum()),
            "profit": (amt * ~bad).sum() * NIM - (amt * bad).sum() * lgd - OPEX * mask.sum(),
            "air": ((mask & ga).sum() / ga.sum()) / max((mask & gb).sum() / gb.sum(), 1e-9)}


legacy = policy((te["credit_score"] >= 640) & (te["dti"] <= 0.45))
fair = ct[ct["air"] >= 0.8]
rec_row = fair.loc[fair["profit"].idxmax()]
rec = policy(te["score"] >= rec_row["cutoff"])
vol_cut = int(ct.iloc[(ct["approval_rate"] - legacy["approval"]).abs().argmin()]["cutoff"])
vol = policy(te["score"] >= vol_cut)
annual = 2.0  # the test window is six months
gain_rec = (rec["profit"] - legacy["profit"]) * annual
gain_vol = (vol["profit"] - legacy["profit"]) * annual
cal = U.calib
members = d.bank().members
seg = M.segments()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 2.4))
ax.plot(ct["cutoff"], ct["profit"] * annual / 1e6, color=TEAL, lw=2)
ax.axhline(legacy["profit"] * annual / 1e6, color=ORANGE, lw=1.2, ls="--")
ax.text(ct["cutoff"].min() + 2, legacy["profit"] * annual / 1e6 + 0.05, "Current policy", color=ORANGE, fontsize=8)
ax.axvline(rec_row["cutoff"], color="#0E1311", lw=0.8)
ax.text(rec_row["cutoff"] + 2, ax.get_ylim()[1] * 0.92, f"Recommended cut-off {int(rec_row['cutoff'])}", fontsize=8)
ax.axvline(vol_cut, color="#A3ABA5", lw=0.8, ls=":")
ax.set_xlabel("Score cut-off")
ax.set_ylabel("Annual profit ($M)")
ax.grid(axis="y", color="#EDEFEA")
chart_profit = fig_to_image(fig, 6.5)

fig, ax = plt.subplots(figsize=(3.1, 2.5))
ax.plot(cal["score"], cal["actual"] * 100, color="#0E1311", lw=1.6, marker="o", ms=3, label="Actual")
ax.plot(cal["score"], cal["kgb"] * 100, color=ORANGE, lw=1.6, marker="o", ms=3, label="Approved-only model")
ax.plot(cal["score"], cal["ri"] * 100, color=TEAL, lw=1.6, marker="o", ms=3, label="With reject inference")
ax.set_xlabel("Score")
ax.set_ylabel("Bad rate (%)")
ax.legend(frameon=False, fontsize=7)
ax.grid(axis="y", color="#EDEFEA")
chart_cal = fig_to_image(fig, 3.15)

fig, ax = plt.subplots(figsize=(3.1, 2.5))
ca = ct[ct["cutoff"] <= 660]
ax.plot(ca["cutoff"], ca["air"], color=TEAL, lw=1.8)
ax.axhspan(0, 0.8, color=ORANGE, alpha=0.08)
ax.axhline(0.8, color=ORANGE, lw=1, ls="--")
ax.axvline(rec_row["cutoff"], color="#0E1311", lw=0.8)
ax.set_ylim(0.5, 1.02)
ax.set_xlabel("Score cut-off")
ax.set_ylabel("Adverse impact ratio")
ax.grid(axis="y", color="#EDEFEA")
chart_air = fig_to_image(fig, 3.15)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def on_page(c, doc):
    c.saveState()
    w, h = letter
    c.setFillColor(PETROL)
    c.rect(0, h - 6, w * 0.45, 6, stroke=0, fill=1)
    c.setFillColor(LUME)
    c.rect(w * 0.45, h - 6, w * 0.55, 6, stroke=0, fill=1)
    if doc.page > 1:
        c.setFont("Text", 7.5)
        c.setFillColor(MUTED)
        c.drawString(inch, 0.55 * inch, "Sample decision diagnostic · synthetic data")
        c.drawRightString(w - inch, 0.55 * inch, f"R.K. Henderson · richardhenderson.io · {doc.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                        title="Sample Decision Diagnostic: consumer loan approvals", author="R.K. Henderson",
                        subject="Sample deliverable built on synthetic data")
P = lambda t, s="body": Paragraph(t, S[s])  # noqa: E731
story = []

# Cover
story += [Spacer(1, 1.3 * inch), P("DECISION DIAGNOSTIC · SAMPLE DELIVERABLE", "eyebrow"),
          P("Consumer loan approvals:<br/>where the cut-off is costing you", "h1"),
          P(f"Prepared for {d.NAME} (a fictional lender). Every figure in this report is computed from synthetic data in the "
            "lending decision lab at richardhenderson.io, so you can see exactly what a diagnostic delivers before you buy one.", "lede"),
          Spacer(1, 0.3 * inch),
          table([["Decision", "Approve or decline card, auto and personal loan applications"],
                 ["Engagement", "Decision diagnostic · two to three weeks · fixed scope"],
                 ["Prepared by", "R.K. Henderson, decision science for lenders"],
                 ["Date", date.today().strftime("%B %Y")]], [1.4, 5.1], header=False),
          Spacer(1, 2.2 * inch),
          P("Kestrel Valley Credit Union, its members, applications and outcomes are synthetic. Results are illustrative and are not "
            "credit policy, legal or compliance advice.", "small"), PageBreak()]

# Executive summary
story += [P("01 · SUMMARY", "eyebrow"), P("The answer on one page", "h2"),
          P(f"The current approval rule (credit score of at least 640 and debt-to-income no higher than 45%) approves "
            f"<b>{legacy['approval']:.0%}</b> of applicants with a <b>{legacy['bad']:.1%}</b> twelve-month bad rate. A points scorecard built "
            f"on the credit union's own history ranks risk better (AUC {U.auc['Scorecard + reject inference']:.3f} vs "
            f"{U.auc['Legacy policy']:.3f}) and finds a more profitable place to draw the line."),
          kpis([("Annual profit uplift", money(gain_rec), f"at cut-off {int(rec_row['cutoff'])}, before adoption"),
                ("At 70% adoption", money(gain_rec * ADOPTION), "recommendations followed"),
                ("Bad rate", f"{rec['bad']:.1%}", f"from {legacy['bad']:.1%} today"),
                ("Adverse impact ratio", f"{rec['air']:.2f}", "above the 0.80 review line")]),
          Spacer(1, 12),
          callout(f"<b>Recommendation: go, with a volume decision first.</b> The profit-maximizing cut-off approves fewer applicants "
                  f"({rec['approval']:.0%} vs {legacy['approval']:.0%}). If growth matters more than margin this year, a volume-neutral "
                  f"cut-off of {vol_cut} keeps approvals level and still adds about {money(gain_vol)} a year by approving "
                  "different people: lower-risk applicants the old rule declined, instead of higher-risk ones it approved."),
          P("What happens next", "h3"),
          P("1. Leadership chooses margin or volume (one meeting). 2. Build: scorecard into the decision engine with adverse-action reasons "
            "and a policy layer, eight weeks. 3. Prove: a randomized pilot on 20% of applications for one quarter before full rollout."),
          PageBreak()]

# Decision frame and data
story += [P("02 · THE DECISION", "eyebrow"), P("Framed before any modeling", "h2"),
          table([["Element", "Agreed with the credit union"],
                 ["Decision", "Approve or decline each consumer loan application at submission"],
                 ["Owner", "Chief credit officer; underwriters keep override authority with logged reasons"],
                 ["Success measure", "Twelve-month profit per application after expected losses, not approval rate or bad rate alone"],
                 ["Guardrails", "Adverse impact ratio of at least 0.80 by group; reasons for every decline; hard policy rules stay above the score"],
                 ["Baseline", "The current score-and-DTI rule, measured on the same applications"],
                 ["Out of scope", "Mortgage and HELOC underwriting, pricing, and line assignment"]], [1.5, 5.0]),
          Spacer(1, 8), P("03 · THE DATA", "eyebrow"), P("What was used and what it can't tell you", "h2"),
          table([["Source", "Coverage", "Finding"],
                 ["Applications", f"{len(d.bank().applications):,} over 24 months", "Complete; six-month out-of-time test window held back"],
                 ["Outcomes", "Approved loans only", "Declined applicants have no outcome: corrected with reject inference"],
                 ["Member history", f"{len(members):,} members", "Tenure and relationship add signal for existing members"],
                 ["Drift", "Last six months vs first twelve", f"Score distribution PSI {M.psi(U.ri.score(M.uw_frame(d.bank().applications[d.bank().applications.month < 12])), U.ri.score(M.uw_frame(d.bank().applications[d.bank().applications.month >= 18]))):.2f}: watch, no action yet"]],
                [1.4, 1.9, 3.2]),
          PageBreak(), P("03 · THE DATA, CONTINUED", "eyebrow"), P("What the history hides", "h2"),
          P("Because the old rule declined a third of applicants, a model trained only on approved loans has never seen how the riskiest "
            "applicants behave. The chart below shows the consequence and the correction."),
          Table([[chart_cal, chart_air]], colWidths=[3.25 * inch, 3.25 * inch]),
          P("Left: the approved-only model understates risk at low scores; reject inference corrects it. Right: the adverse impact ratio "
            "falls as the cut-off rises, so fairness is set on the same chart as profit.", "small"),
          Spacer(1, 14),
          table([["Score band", "Applicants", "Actual bad rate", "Approved-only model", "With reject inference"]] +
                [[f"{int(r.score)}", f"{int(r.n):,}", f"{r.actual:.1%}", f"{r.kgb:.1%}", f"{r.ri:.1%}"] for r in cal.itertuples()],
                [1.1, 1.1, 1.4, 1.45, 1.45]),
          Spacer(1, 10),
          callout("<b>Why it matters.</b> Setting the cut-off with the approved-only model would approve applicants whose real risk is "
                  "a fifth higher than predicted at the bottom of the range. The reject-inferred scorecard is the one used for every "
                  "number in the rest of this report."),
          PageBreak()]

# Sizing
story += [P("04 · SIZING THE PRIZE", "eyebrow"), P("Where the profit is, and what it depends on", "h2"),
          chart_profit,
          P("Annualized profit on next year's applicants across score cut-offs, against the current rule. One-year view per loan: "
            f"{NIM:.1%} net margin on good loans, {LGD:.0%} loss given default, ${OPEX} processing cost per approval.", "small"),
          Spacer(1, 8),
          table([["Policy", "Approval rate", "Bad rate", "Annual profit", "AIR"],
                 ["Current rule (score 640+, DTI up to 45%)", f"{legacy['approval']:.0%}", f"{legacy['bad']:.1%}", money(legacy['profit'] * annual), f"{legacy['air']:.2f}"],
                 [f"Volume-neutral scorecard (cut-off {vol_cut})", f"{vol['approval']:.0%}", f"{vol['bad']:.1%}", money(vol['profit'] * annual), f"{vol['air']:.2f}"],
                 [f"Profit-maximizing scorecard (cut-off {int(rec_row['cutoff'])})", f"{rec['approval']:.0%}", f"{rec['bad']:.1%}", money(rec['profit'] * annual), f"{rec['air']:.2f}"]],
                [2.9, 0.95, 0.8, 1.05, 0.8]),
          Spacer(1, 10), P("Sensitivity", "h3"),
          table([["If…", "Annual uplift (profit-maximizing)"],
                 ["Underwriters follow 100% of recommendations", money(gain_rec)],
                 ["70% are followed (planning case)", money(gain_rec * 0.7)],
                 ["50% are followed", money(gain_rec * 0.5)],
                 ["Loss given default is 75%, not 60%", money((policy(te['score'] >= rec_row['cutoff'], 0.75)['profit'] - policy((te['credit_score'] >= 640) & (te['dti'] <= 0.45), 0.75)['profit']) * annual)]],
                [4.2, 2.3]),
          P("Adoption moves the answer more than any modeling choice, which is why reasons, overrides and an override log are part of the build.", "body"),
          PageBreak()]

# Recommendations
story += [P("05 · RECOMMENDATIONS", "eyebrow"), P("Three actions, ranked by payback", "h2"),
          table([["#", "Action", "Effort", "Value"],
                 ["1", "Replace the score-and-DTI rule with the reject-inferred scorecard at the cut-off leadership chooses; keep hard policy rules above it", "8 weeks", money(gain_rec * ADOPTION) + " a year"],
                 ["2", "Generate adverse-action reasons from scorecard points and log every override with a reason code", "Included in 1", "Protects adoption and exam readiness"],
                 ["3", "Hold back 5% of applications from the score at random each quarter so future models can be retrained on unbiased outcomes", "1 week", "Keeps the model honest"]],
                [0.3, 3.9, 0.9, 1.4]),
          Spacer(1, 12), P("Build plan", "h3"),
          table([["Weeks", "Work", "Exit test"],
                 ["1–2", "Final scorecard, policy layer, reason codes", "Validation sign-off; fair-lending review"],
                 ["3–6", "Decision-engine integration; underwriter view with reasons", "Shadow mode matches offline results"],
                 ["7–8", "Randomized pilot on 20% of applications", "Pre-registered profit and bad-rate read-out"],
                 ["Ongoing", "Monthly monitoring: drift, calibration, AIR, overrides", "Retrain trigger agreed in advance"]],
                [0.8, 3.3, 2.4]),
          Spacer(1, 12), P("Risks and how they are handled", "h3"),
          P("<b>Fair lending.</b> The approval gap between groups flows through income and credit history. A search for less discriminatory "
            "alternatives runs before launch, and the cut-off is never set below an AIR of 0.80 without compliance sign-off."),
          P("<b>Volume.</b> A higher cut-off shrinks the book. Leadership chooses the trade-off explicitly; the volume-neutral option is ready."),
          P("<b>Drift.</b> Recent applicants are shopping for credit more. Monitoring and a retrain trigger are part of the build, not an afterthought."),
          PageBreak()]

# Appendix
story += [P("APPENDIX · METHOD", "eyebrow"), P("How the numbers were produced", "h2"),
          P("<b>Data.</b> Consumer-loan applications (card, auto, personal) from the lab's synthetic credit union. Models were trained on months "
            "0–17 and every figure in this report is measured on months 18–23, which the models never saw."),
          P("<b>Scorecard.</b> Ten quantile bins per characteristic, weight of evidence, logistic regression, scaled so that 600 points means 50:1 "
            "good-to-bad odds and every 20 points doubles them."),
          P("<b>Reject inference.</b> Fuzzy augmentation: each declined applicant enters twice, as good and bad, weighted by an inflated "
            "approved-only risk estimate."),
          P("<b>Profit.</b> One-year view per loan: net margin on loans that perform, minus loss given default on loans that do not, minus processing cost."),
          P("<b>Fairness.</b> Adverse impact ratio: the approval rate of a synthetic protected-class proxy divided by that of everyone else. "
            "No model sees the proxy."),
          Spacer(1, 18),
          callout("This sample shows the format and depth of a Decision diagnostic. A real engagement uses your data, in your environment, "
                  "and ends with the same three things: the decision framed, the prize sized at realistic adoption, and a clear go or no-go. "
                  "<br/><br/><b>richardhenderson.io</b>")]

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"wrote {OUT}  uplift {money(gain_rec)}  volume-neutral cut-off {vol_cut}")
