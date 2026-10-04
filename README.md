# Decision Lab

Interactive demos for [richardhenderson.io](https://richardhenderson.io), built with Streamlit.
Everything except Market intelligence runs on synthetic data. Market intelligence runs on public Census, HMDA and FHFA data. All rules and rates are illustrative.

Organized the way the website is: products, a case study and method notes.

| Shelf | Page | URL path | What it shows |
| --- | --- | --- | --- |
| Products | Lending decision lab | `/lending-lab?stage=know` | The flagship. Kestrel Valley Credit Union (fictional): 50,000 synthetic members and one shared dataset (`lab/cu/data.py`) behind nine connected stages: **Know** (k-means segments named in business language, RFM), **Acquire** (look-alike prospects by metro), **Underwrite** (WOE scorecard vs gradient boosting, reject inference, cut-off economics, adverse-action reasons), **Fraud** (isolation forest, rules, phone/address link graph), **Price** (risk-based pricing, segment elasticity, adverse selection), **Cross-sell** (next-best product, T-learner uplift, Qini), **Retain** (discrete-time prepayment hazard, Kaplan–Meier, churn), **Collect** (roll-rate Markov chain, treatment effects, capacity allocation) and **Govern** (model inventory, adverse impact ratio, drift, model card). A "follow a member" selector carries one person through every stage |
| | Pricing & demand copilot | `/pricing-copilot?stage=frame` | The flagship. One pricing decision taken end to end through seven stages: **Frame** (decision, owner, measure, guardrails), **Data** (data contract with blocking checks; why raw price data is confounded), **Model** (booking-pace forecast backtested against last year; segment elasticities estimated naively, with controls and by 2SLS on random rate tests), **Decide** (expected-revenue rate inside guardrails, EMSR-b protection, reason codes), **Prove** (switchback design, power with and without CUPED, pre-registered read-out), **Run** (weekly accuracy, lead-time PSI drift, retrain triggers, model card) and **Value** (adoption-adjusted ROI, payback, delivery plan). `?stage=` links straight to a stage |
| | Market intelligence | `/market-intelligence` | A data product on public data. Every CBSA is scored from Census, HMDA and FHFA data and rebuilt monthly. Includes strategy weights, rankings, market briefs, lender benchmarking, month-over-month changes, archetypes, anomalies and grounded Q&A |
| Case studies | Lending decisions | `/lending?view=pipeline` · `?view=product` | A capacity-constrained propensity model with a gains curve and reason codes, and an explainable product recommender |
| Method notes | Experimentation | `/experiments` | Power and duration, peeking with sequential correction, CUPED, frequentist and Bayesian read-outs |
| | Model value | `/model-value` | Converts AUC, capacity, cost and adoption into net value |
| | Human–AI decisions | `/human-ai` | A judge–advisor experiment: Brier scores, weight of advice, reliance profile |
| | Goodhart simulator | `/goodhart` | Shows where the reported metric and the true outcome part ways under pressure |

### Optional secrets (Streamlit → app → Settings → Secrets)

```toml
ANTHROPIC_API_KEY = "sk-ant-..."                # free-text answers in "Ask about this market"
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001" # optional override
```

Without a key, Ask about this market answers a fixed set of questions deterministically from
the market's record. With a key, visitors can type their own question, and answers stay
grounded in the same record.

## Market intelligence data (automated)

`data/markets.csv` is built by **`pipeline/build_markets.py`**, running in GitHub Actions
(workflow **Build market data**):

- **When:** the 6th of every month, whenever the builder changes, or on demand
  (GitHub → Actions → Build market data → Run workflow).
- **What it pulls (all public):** Census CBSA population estimates; FHFA House Price Index
  (metro, divisions rolled up to their CBSA); HMDA through the FFIEC Data Browser API
  (purchase and refinance outcomes, loan types, active lenders, for the latest year and three
  years earlier); and ACS 5-year income and home values if a Census API key is set.
- **What it writes:** `data/markets.csv`, `data/markets_meta.json` and `data/build_log.json`.
  Then it commits them, and Streamlit redeploys on that commit.
- **Safety:** if HMDA coverage comes back thin, the previous `markets.csv` is kept and the run
  is marked failed. The log says why.

**Optional, recommended:** get a free Census API key at api.census.gov/data/key_signup.html,
then add it as a repository secret named `CENSUS_API_KEY` (Settings → Secrets and variables →
Actions). That switches on ACS income and home values, which feed borrower capacity,
valuation strain and affordability pressure.

`lab/market_engine.py` recomputes every score, archetype and risk reading from the table.
`lab/market_sample.py` is a labeled synthetic fallback, used only if `markets.csv` is
missing. `pipeline/databricks_export.py` is an optional alternative that exports the same
table from the Mortgage Market Intelligence notebook.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
pytest -q          # optional: the same smoke tests GitHub runs
```

## Deploy: GitHub → Streamlit Community Cloud

The app runs on Streamlit Community Cloud and deploys straight from this GitHub
repository. Every push to `main` redeploys the live app within about a minute.

### 1. Put the code on GitHub

Create an empty repository on github.com, for example `decision-lab`, without a
README or .gitignore. Then, from this folder:

```bash
git remote add origin https://github.com/<your-username>/decision-lab.git
git push -u origin main
```

(The folder is already a git repository with one commit. If you start from the
zip on a machine without that history, run `git init -b main && git add . &&
git commit -m "Decision Lab"` first.)

The **Smoke test** workflow in `.github/workflows/ci.yml` runs on every push. It
renders every page and fails the run if any of them throws, so a broken
commit shows a red ✕ on GitHub before you notice it on the site.

### 2. Connect Streamlit to GitHub

1. Go to share.streamlit.io and choose **Continue with GitHub**. Authorize Streamlit to read your repositories.
2. **Create app → Yup, I have an app.**
3. Repository `<your-username>/decision-lab`, branch `main`, main file `streamlit_app.py`.
4. **App URL**: choose a subdomain, for example `rkh-decision-lab`. That gives `https://rkh-decision-lab.streamlit.app`.
5. **Advanced settings → Python 3.12**, then **Deploy**. The first build takes two to three minutes.

A public repository is simplest. A private one works too, but Streamlit asks for
extra GitHub permission, and the app itself stays publicly viewable either way.

### 3. Point the website at it

In WordPress: **Appearance → Customize → Decision Lab (Streamlit demos)**. Paste
the app URL and publish.

### Updating

Edit, commit, push. Streamlit picks up the change automatically. If the app
ever looks stale, use **Manage app → Reboot** in the bottom-right corner of the
live app (shown to you only when you are signed in).

## Embedding

The website embeds each page on its own with:

```
https://<your-app>.streamlit.app/<page>?embed=true&solo=1
```

`embed=true` is Streamlit's own embed mode. It removes the toolbar and app chrome. `solo=1` is specific to this app: it hides the page navigation and tightens the top padding, so each demo reads as a single tool inside the site.

Community Cloud apps go to sleep after a period without visitors. The first visitor after a quiet spell waits about 30 seconds while the app wakes. The website handles this with a click-to-load panel and a "waking up" message, so a sleeping app never slows the page itself.

## Structure

```
streamlit_app.py        navigation and page config
lab/theme.py            palette, CSS, Plotly template, stat tiles, formatting
lab/pipeline_model.py   synthetic pipeline + logistic regression + reason codes
lab/products.py         illustrative product catalog and the fit engine
lab/market_engine.py    market scoring, risk, archetypes, anomalies (public data)
lab/market_sample.py    labeled synthetic fallback for the market table
pipeline/build_markets.py   monthly public-data build (GitHub Actions)
pipeline/databricks_export.py  optional notebook export of the same table
data/                   markets.csv, markets_meta.json, build_log.json
views/*.py              one file per page
static/                 self-hosted Bodoni Moda and Schibsted Grotesk (SIL OFL), favicon
.streamlit/config.toml  theme matching the website
tests/                  page smoke tests (pytest + streamlit AppTest)
.github/workflows/      runs the tests on every push
```

## Design notes

- Colors follow the site: porcelain `#FAFAF8`, petrol `#0F4640`, aqua `#7FD0BE`. Charts use three series hues (teal `#008A73`, orange `#D9772B`, indigo `#5B6CB8`). All three pass color-vision separation checks, both adjacent and all-pairs.
- Every chart has a hover layer and a title that names it. Two-series charts are direct-labeled or carry a legend.
- Each page ends with a plain-language "how this works" panel and a data disclaimer.
