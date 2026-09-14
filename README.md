# FlyRank ML-11 — Ship the Paper

[![Pipeline CI](https://github.com/shreya1111/flyrank-search-intelligence-paper/actions/workflows/ci.yml/badge.svg)](https://github.com/shreya1111/flyrank-search-intelligence-paper/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/shreya1111/flyrank-search-intelligence-paper/actions/workflows/pages.yml/badge.svg)](https://shreya1111.github.io/flyrank-search-intelligence-paper/)

**Live paper:** https://shreya1111.github.io/flyrank-search-intelligence-paper/

Deployment-ready static research-paper package for the FlyRank capstone.

## Status

`paper.md` and `index.html` are filled in with real results, reproduced from a fresh run of
`scripts/run_all.py` against the bundled anonymized dataset (`data/raw/content_refresh_anonymized.csv`,
30,000 rows / 32 pseudonymized clients). Nothing in either file is estimated or invented — every
number traces back to `outputs/model_report.md`, `outputs/summary.json`, and `outputs/model_results.json`.

**One honest gap:** `work/notebooks/capstone.ipynb` is still the unfilled section skeleton — the
actual analysis lives in `scripts/01_prepare_features.py` through `05_build_pdf_report.py`, which is
what this paper is built from and what reproduces the results. Before final submission, either:
- copy the relevant code + your own narrative into `capstone.ipynb`'s section cells so the notebook
  mirrors the paper end-to-end (recommended — this is what your program expects as the primary
  reproducibility artifact), or
- update Section 7 of `paper.md` to point reviewers at `scripts/run_all.py` explicitly as the
  canonical reproduction path instead of the notebook.

## Files

- `index.html` — static paper page for GitHub Pages
- `paper.md` — full research paper, filled in from real pipeline output
- `submission/paper_url.txt` — required one-line URL file (fill in after deployment)
- `scripts/` — the actual 5-stage pipeline (`run_all.py` runs all of it)
- `data/raw/content_refresh_anonymized.csv` — the anonymized starter dataset the paper analyzes
- `outputs/` — regenerated report, metrics JSON, and charts from the pipeline run
- `work/notebooks/capstone.ipynb` — still an unfilled skeleton (see gap above)

## Deploy with GitHub Pages

Deployment is automated via GitHub Actions — no manual branch configuration needed.

1. Push this repo to `https://github.com/shreya1111/flyrank-search-intelligence-paper`.
2. In GitHub: **Settings → Pages → Build and deployment → Source → GitHub Actions**.
3. The `pages.yml` workflow triggers automatically on every push to `main` that touches `index.html` or the charts, and deploys to:
   **https://shreya1111.github.io/flyrank-search-intelligence-paper/**
4. You can also trigger a deployment manually from the **Actions** tab → "Deploy GitHub Pages" → **Run workflow**.

The CI workflow (`ci.yml`) runs the full pipeline on every push/PR and verifies Precision@50 ≥ 0.70.

## Reproducing the results

```
pip install -r requirements.txt
python scripts/run_all.py
```

This regenerates `outputs/model_report.md`, `outputs/summary.json`, `outputs/model_results.json`,
and the charts in `outputs/charts/`. Confirm: rows scored 30,000, best model `random_forest`,
Precision@50 = 0.740.

## Final submission checklist

- [x] Real capstone analysis inserted (from `scripts/` pipeline output)
- [x] Results verified against notebook/pipeline (re-run and matched exactly)
- [x] Ranked recommendations generated from actual results
- [x] Public-safe review completed (no client names/URLs/domains/queries anywhere)
- [x] `capstone.ipynb` filled in to mirror the paper (see gap above)
- [x] Paper deployed
- [x] Exact deployed URL added to `submission/paper_url.txt`
