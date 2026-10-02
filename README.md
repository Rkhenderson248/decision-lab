# Decision Lab

Five interactive demos for [richardhenderson.io](https://richardhenderson.io), built with Streamlit.
Four run on synthetic data. Market intelligence runs on public Census, HMDA and FHFA data. All rules and rates are illustrative.

| Page | URL path | What it shows |
| --- | --- | --- |
| Pipeline prioritizer | `/pipeline` | Propensity scoring on a synthetic lending pipeline, with a gains curve against "newest first" and random, a ranked worklist and reason codes |
| Product fit | `/product-fit` | Eligibility rules across seven illustrative loan products, ranked by cost over the borrower's horizon or by monthly payment, with every exclusion explained |
| Model value | `/model-value` | AUC, capacity, cost and adoption turned into net value, plus where the next dollar comes from (model quality or adoption) |
| Goodhart simulator | `/goodhart` | A team of 300 under metric pressure, showing where the reported metric and the true outcome part ways |
| Market intelligence | `/market-intelligence` | Every U.S. metro and micro area scored on five blocks plus a two-basis risk composite, from public data. Includes tunable strategy weights, a state tile map, an opportunity-vs-risk quadrant, unsupervised archetypes, anomaly detection, divergence signals and a market brief |

## Market intelligence data

The page reads `data/markets.csv`, one row per CBSA of **public-source metrics only**, and
recomputes every score, archetype and risk reading in `lab/market_engine.py`. The method is a
port of the Mortgage Market Intelligence notebook, with no internal features: no footprint,
territories, lender names or lender position.

To publish real data:

1. Paste `pipeline/databricks_export.py` as the last cell of the Mortgage Market Intelligence notebook and run it.
2. Commit the two files it writes, `markets.csv` and `markets_meta.json`, into `data/`. Or use the table's download button.
3. Push. Streamlit redeploys, and the "Sample data" banner disappears.

Until `data/markets.csv` exists, the page runs on a synthetic table with fictional market names
(`lab/market_sample.py`) and says so in a banner at the top.

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

1. Go to share.streamlit.io and choose **Continue with GitHub**. Authorise Streamlit to read your repositories.
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
lab/products.py         illustrative product catalogue and the fit engine
lab/market_engine.py    market scoring, risk, archetypes, anomalies (public data)
lab/market_sample.py    labelled synthetic fallback for the market table
pipeline/               Databricks export cell for the public market table
data/                   markets.csv + markets_meta.json once exported
views/*.py              one file per page
static/                 self-hosted Bodoni Moda and Schibsted Grotesk (SIL OFL), favicon
.streamlit/config.toml  theme matching the website
tests/                  page smoke tests (pytest + streamlit AppTest)
.github/workflows/      runs the tests on every push
```

## Design notes

- Colours follow the site: porcelain `#FAFAF8`, petrol `#0F4640`, aqua `#7FD0BE`. Charts use three series hues (teal `#008A73`, orange `#D9772B`, indigo `#5B6CB8`). All three pass colour-vision separation checks, both adjacent and all-pairs.
- Every chart has a hover layer and a title that names it. Two-series charts are direct-labelled or carry a legend.
- Each page ends with a plain-language "how this works" panel and a data disclaimer.
