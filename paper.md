# Search Intelligence Capstone: From Search Data to Ranked Content Recommendations

**FlyRank ML Internship — ML-11: Ship the Paper**
**Author:** Shreya Srivastava
**Status:** Deployment-ready research paper

---

## Abstract

Search behavior contains measurable signals about what users are trying to find, how demand changes over time, and where existing content may be failing to satisfy that demand. This study develops a repeatable search-intelligence workflow on the FlyRank ML Internship content-performance dataset. The research question tested is:

> **Can pre-outcome content, keyword-context, and 90-day search/engagement features distinguish content whose search visibility is currently declining from content that is stable or growing — well enough to rank pages for editorial refresh review, and better than a transparent hand-written scoring rule?**

The analysis defines a public-safe data contract, restricts the observation window to content old enough to have a stable trend read, constructs a 52-feature vector from search and engagement behavior, establishes a transparent rule-based baseline, trains and compares three model families under a client-holdout split, checks the feature set for label leakage, and converts the resulting classifier into a ranked, reason-coded content-refresh queue.

The final analysis covers **30,000 content items across 32 pseudonymized clients**, drawn from the bundled `content_refresh_anonymized.csv` starter export, with all activity metrics aggregated over a **trailing 90-day window ending at export time**, restricted to items with at least one search impression and at least 90 days of content age. The target is `is_declining_label` (`trend_direction == "down"`), with a positive rate of **54.2%** (16,262 of 30,000 rows). The best-performing approach is a **random forest classifier** (52 features, one-hot categoricals + numeric/log-transformed activity metrics), compared against a transparent weighted-rule baseline (visibility 40% + staleness risk 30% + position opportunity 25% + depth gap 5%). The random forest reaches **ROC AUC 0.750** and **Precision@50 of 0.740**, versus the baseline's **ROC AUC 0.627** and **Precision@50 of 0.240** — a roughly **3x improvement in top-of-queue precision** over the hand-written rule, under a held-out split that never mixes a client's rows between train and test. Based on this evidence, the paper produces a ranked, reason-coded refresh queue and action playbook for content teams managing declining-visibility pages.

The conclusions are bounded by the observed data (a 30,000-row anonymized slice, not the full internship dataset), the trailing-90-day trend definition used as the label, and the study's non-causal, observational design. This paper is a reproducible research artifact, not a claim that search behavior alone establishes user intent or that refreshing flagged content will causally improve rankings.

---

## 1. Introduction

### 1.1 Problem

Search data can be treated as an observable record of information demand. However, raw query-level or page-level observations do not automatically translate into useful content decisions. A useful search-intelligence system needs to answer three questions:

1. What signal is present in the search data?
2. Does that signal generalize beyond the data used to discover it?
3. How should the validated signal change content priorities?

The purpose of this capstone is therefore not simply to maximize a model metric. It is to build a repeatable chain from **question → data → method → validation → result → recommendation**.

### 1.2 Research Question

**Primary research question:**
Can search/engagement behavior and content-context features observed over a trailing 90-day window separate declining-visibility content from stable-or-growing content well enough to produce a useful, prioritized refresh queue — outperforming a transparent, hand-written scoring rule under a leakage-resistant validation split?

**Secondary questions:**
- Which features carry the most weight in separating declining from non-declining content, and are they plausible (not artifacts of the label definition)?
- Does the model's advantage over the baseline hold up under a **client-holdout** split (an entire client's pages held out of training), rather than a naive random row split that would let a client's other pages leak signal into the test set?

### 1.3 Contribution

This study contributes:

- an explicit, public-safe data contract built on an already-anonymized starter export (no client names, domains, URLs, titles, or raw queries);
- a reproducible feature and scoring pipeline (`scripts/01_prepare_features.py` → `05_build_pdf_report.py`);
- a transparent baseline against which the trained model is compared;
- a leakage check that explicitly excludes the trend-derived columns (`trend_direction`, `trend_pct`, and the last-30d/prev-30d comparison windows) from the model's feature set, since they are the direct source of the label;
- an interpretation of feature importances in search-intelligence terms; and
- a ranked, reason-coded action playbook for content teams.

---

## 2. Data

### 2.1 Dataset

The study uses the anonymized starter export that ships with the FlyRank ML Internship repository: `data/raw/content_refresh_anonymized.csv`.

**Dataset release:** Anonymized starter slice (not the full ~79M-row internship release); one row per pseudonymized content item.
**Observation window:** All activity metrics (impressions, clicks, sessions, engagement) are aggregated over a **trailing 90-day window ending at export time**; `impressions_last_30d` / `impressions_prev_30d` and the equivalent click/session pairs give a 30-day-vs-prior-30-day comparison used only to derive the trend label.
**Rows/records used:** 30,000 rows read; **30,000 rows retained** after filtering (no rows dropped by the impression/age filter in this slice), covering **32 pseudonymized clients**.
**Primary unit of analysis:** One row = one pseudonymized content item (page).
**Target variable:** `is_declining_label`, defined as `trend_direction == "down"` (binary). Positive rate: **54.2%** (16,262 / 30,000).

The FlyRank starter repository documents a full release of approximately 79 million rows accessed through DuckDB; this paper analyzes only the bundled 30,000-row anonymized slice, not the full release, and all figures below apply to that slice only.

### 2.2 Inclusion and Exclusion

**Included:**
- Content items with `impressions_90d > 0` (must have measurable search visibility in the window);
- Content items with `content_age_days >= 90` (must be old enough for a stable 90-day trend read);
- One row per unique `content_id` (duplicates on `content_id` dropped).

**Excluded:**
- Private or identifying information — the starter export already contains no client names, domains, URLs, page titles, or raw search queries; only hashed `content_id` / `client_id` pseudonyms remain, used solely for grouping and the client-holdout split, never as model features;
- `provider_used` and `model_used` (which LLM generated the article) — present in the raw data but excluded from the model feature set as not search-behavior signal;
- Observations with zero measured search impressions or under 90 days of age (see above).

### 2.3 Data Quality

The analysis checks:

- missing values;
- duplicate observations;
- invalid dates;
- unexpected categorical values;
- extreme values/outliers;
- class imbalance, where applicable;
- temporal coverage.

**Observed quality findings:**
- No rows were removed by the impression/age filter in this slice (30,000 in → 30,000 prepared), and no duplicate `content_id` values were found.
- `word_count`/`char_count` are missing for a documented subset of rows (systematically tied to `content_type`, per the data dictionary — `feedly article` rows carry no keyword-context fields), and were filled with 0 with a categorical `word_count_tier = "unknown"` bucket preserved as its own feature level rather than silently imputed.
- The target class is moderately imbalanced toward "declining" (54.2% positive vs. 45.8% negative) — not severe enough to require resampling, but reported because it changes how precision-based metrics should be read.
- Rate columns (`ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`) are on a 0–100 percentage scale (e.g. `ctr = 0.76` means 0.76%, not 76%), confirmed against the data dictionary before feature use.

---

## 3. Methodology

### 3.1 Research Design

The workflow follows:

**Research question → data contract → feature construction → baseline → model/scoring → validation → interpretation → ranked recommendations**

Implemented as a five-stage script pipeline: `01_prepare_features.py` → `02_baseline_score.py` → `03_train_model.py` → `04_evaluate_and_export.py` → `05_build_pdf_report.py`, run end-to-end via `scripts/run_all.py`.

### 3.2 Features

The trained model uses **52 features** after one-hot encoding: 18 numeric features and 8 categorical fields expanded into dummy columns. The top model-ranked features are shown in Section 4.3. Representative features:

| Feature | Definition | Why it matters |
|---|---|---|
| `days_with_impressions` | Count of days in the 90-day window with >=1 search impression | Captures consistency of visibility, not just total volume — a page seen on many days is a more stable signal than one spike day |
| `log_impressions_90d` | log1p of 90-day search impressions | Compresses the heavy-tailed impression distribution so the model isn't dominated by a few very high-traffic pages |
| `avg_position` | Average Google Search Console ranking position over the window | Direct proxy for current visibility/competitiveness of the page |
| `content_age_days` | Days since the content was created | Older content has had more time to decay or to accumulate authority; both directions are plausible a priori |
| `word_count` / `char_count` | Article length | Thin content is a common, actionable refresh reason (`thin_visible_page` in the baseline's reason codes) |

Full numeric feature list: `search_volume, competition, cpc, word_count, char_count, log_impressions_90d, log_clicks_90d, log_sessions_90d, log_ai_sessions_90d, days_with_impressions, days_with_sessions, content_age_days, days_since_last_update, ctr, avg_position, engagement_rate, scroll_rate, ai_traffic_pct`.
Categorical feature list (one-hot encoded): `competition_level, content_type, main_intent, age_tier, freshness_tier, word_count_tier, impression_tier, position_tier`.

### 3.3 Baseline

The baseline is a **transparent, hand-written weighted rule**, computed with no model training:

`baseline_refresh_score = 0.40 x visibility_score + 0.30 x freshness_risk_score + 0.25 x position_opportunity_score + 0.05 x depth_gap_score`

where each component is a percentile rank of a single interpretable signal (log impressions, days since last update, ranking-position opportunity, and content-length gap, respectively). This produces the same style of ranked queue as the model, with per-row reason codes (`stale_visible_page`, `declining_with_demand`, `thin_visible_page`, `page_one_decay_risk`, `low_ctr_visible_page`, `low_engagement_visible_page`), so the two approaches are directly comparable.

The baseline is necessary because an apparently strong model is not useful if a simpler, fully explainable rule performs similarly — the baseline's top-50 declining rate on the full data is **0.340**, and its held-out Precision@50 is reported alongside the model's in Section 4.1.

### 3.4 Model / Scoring Method

**Approach:** Three classifiers compared — logistic regression, decision tree, and random forest — on the identical 52-feature vector.
**Key hyperparameters:** scikit-learn defaults with `random_state=42`; logistic regression run inside a `StandardScaler` pipeline.
**Training procedure:** Fit each model on the training partition of a client-aware split (see below); select the best model by **Precision@50** (the metric that matters for a ranked review queue, where only the top items get human attention).
**Validation procedure:** A **client-holdout split** — entire clients (not individual rows) are randomly assigned to train or test, with roughly 20% of the 32 clients held out (test set: 2,325 rows; train set: 27,675 rows). This prevents a client's other pages from leaking shared, client-level patterns into the evaluation set, which a naive random row split would allow.

### 3.5 Leakage Checks

The analysis explicitly checks whether information unavailable at prediction time enters the feature set or validation process.

Potential leakage sources considered:

- **Target-derived aggregates:** `trend_direction` (the label source) and `trend_pct`, along with the `*_last_30d` / `*_prev_30d` comparison columns used to compute the trend, are **excluded from the model feature set** — they are listed in neither `MODEL_NUMERIC_FEATURES` nor `MODEL_CATEGORICAL_FEATURES`. Only 90-day-window aggregates and static content properties are used.
- **Duplicated entities across train/validation:** addressed directly by the client-holdout split (Section 3.4) — no client appears in both partitions.
- **Post-outcome variables:** `provider_used` and `model_used` (which LLM authored the page) are excluded as not being search-behavior signal and not temporally meaningful to the refresh decision.
- **Future observations:** the dataset is a single 90-day-window snapshot per row rather than a time series, so there is no future-window row to leak from; the trend comparison itself (last-30d vs prior-30d) is computed once, upstream, and excluded from features as noted above.

**Leakage findings:** No direct label-derived column appears among the model's 52 input features. The `client_id` field is used only for the train/test split assignment, never as a model input. This does not rule out subtler indirect leakage (e.g., `days_with_impressions` correlating with recency of decline), but the direct, most obvious leakage path — feeding the trend columns that define the label back in as features — was checked and is not present.

---

## 4. Results

### 4.1 Baseline vs. Final Approach

All metrics below are computed on the same client-holdout test set (2,325 rows from held-out clients), except the baseline's "top-50 declining rate," which is reported on the full 30,000-row scored population per the pipeline's own output (noted below the table).

| Metric | Baseline (rule) | Random Forest (final) | Decision Tree | Logistic Regression |
|---|---:|---:|---:|---:|
| ROC AUC | 0.627 | **0.750** | 0.742 | 0.700 |
| Average precision | 0.468 | **0.618** | 0.575 | 0.522 |
| Precision@50 | 0.240 | **0.740** | 0.660 | 0.400 |
| Precision@20 | 0.15 | **0.65** | 0.65 | 0.35 |
| Precision@100 | 0.36 | **0.72** | 0.68 | 0.44 |
| Recall | 0.189 | 0.744 | 0.716 | 0.567 |
| F1 | 0.274 | **0.640** | 0.634 | 0.566 |

*Baseline row uses the `model_results.json` holdout figures (`baseline_*`), computed on the same held-out split as the models — this differs from the separate full-population "top-50 declining rate = 0.340" figure quoted for the baseline in Section 3.3, which is reported over all 30,000 rows.*

### 4.2 Main Finding

On a client-holdout split that prevents a client's own pages from leaking into its evaluation set, a random forest classifier trained on 52 search-behavior and content-context features reaches **Precision@50 = 0.740** and **ROC AUC = 0.750** for identifying content with declining search visibility, compared with **Precision@50 = 0.240** and **ROC AUC = 0.627** for a transparent, hand-written weighted-rule baseline using the same underlying signals (visibility, staleness, position opportunity, and content depth). This is roughly a **3x improvement in the precision of the top-50 ranked review queue** — the portion of the queue a content team would realistically act on first — over the rule-based approach, under the same evaluation split and the same target definition.

This is a measured, decision-support result under the stated validation design: it shows the model separates declining from non-declining content, on this 30,000-row anonymized slice, better than the baseline rule does. It does not establish that any specific feature *causes* decline, or that acting on the queue will improve outcomes — see Section 5.

### 4.3 Feature / Signal Interpretation

The strongest signals in the random forest's feature-importance ranking were:

1. **`days_with_impressions`** (importance 0.158) — consistency of visibility across the 90-day window is the single strongest signal; pages seen on few distinct days are disproportionately likely to be flagged declining, suggesting intermittent rather than steady visibility is an early warning sign.
2. **`log_impressions_90d`** (importance 0.128) — overall visibility volume remains highly informative even after log-compression, indicating scale of demand still matters once consistency is accounted for.
3. **`avg_position`** (importance 0.109) — average ranking position contributes meaningfully but less than the two visibility-pattern features above it, consistent with position being one input to visibility rather than the whole story.
4. **`content_age_days`** (importance 0.096) and **`char_count`/`word_count`** (0.043 / 0.040) — content age and length round out the top five, both plausible, non-label-derived signals rather than artifacts.

These are predictive/associative signals within this study's design; they should not automatically be interpreted as causal drivers of decline (e.g., older content is not thereby proven to decline *because* it is old — age may proxy for other unobserved factors like accumulated competition).

---

## 5. Limitations and Honest Framing

This study has several limitations.

### 5.1 Observational Data

Search observations describe behavior but do not, by themselves, establish why a page's visibility is declining or whether refreshing a flagged page would cause its visibility to recover. The model finds association, not causation.

### 5.2 Temporal and Coverage Limits

The conclusions apply to the **30,000-row anonymized starter slice covering 32 pseudonymized clients, over a single trailing-90-day snapshot window**, and may not generalize to the full ~79M-row internship dataset, to different clients, industries, or query types, or to different time periods (e.g., a seasonal or algorithm-update period not represented in this snapshot).

### 5.3 Model Limitations

The final approach is sensitive to:

- The single-snapshot nature of the data — there is no way to validate the model's ranking against what actually happens to a page's visibility *after* a refresh decision, since the dataset has no forward-looking outcome window;
- Missingness patterns tied to `content_type` (keyword-context fields are systematically absent for `feedly article` rows), which the model absorbs via an "unknown" tier rather than true imputation, and which could bias the model toward or against particular content types in ways not fully audited here.

### 5.4 Validation Limits

The validation design (client-holdout split, Precision@50/ROC AUC comparison against a matched baseline) provides evidence that **the model ranks declining content ahead of non-declining content better than the rule-based baseline does, on unseen clients within this dataset**. It does not establish that **acting on the model's refresh queue improves search outcomes**, since no post-action outcome data exists in this snapshot to test that claim.

### 5.5 Public-Safety Framing

The deployed paper contains only aggregate/public-safe findings. No client names, private URLs, raw identifying records, credentials, or restricted query-level information appear anywhere in this document, consistent with the already-anonymized starter export it is built from.

---

## 6. Ranked Recommendations

The model is useful only if its output changes an action. The following recommendations are generated directly from the validated pipeline output (`outputs/model_report.md`, `outputs/refresh_queue.csv`).

| Rank | Recommendation | Evidence | Expected impact | Confidence |
|---:|---|---|---|---|
| 1 | Route the **3,602 high-confidence queue items** to editorial review first, prioritizing the `refresh_and_review_ctr` action group | Model probability and score consistently high (top-10 preview scores 80.3-81.7, model probability 0.77-0.85), reason codes combine `declining_with_demand` + `low_ctr_visible_page` + `model_decline_risk` | Concentrates limited review capacity on the highest-precision segment of the queue (Precision@50 = 0.740) | High — directly measured on held-out clients |
| 2 | Within the high-confidence group, treat **`refresh_and_review_ctr`** (6,657 items) as the largest single action bucket and staff it first | It is the plurality action among top-ranked items and the dominant action in the Top-10 preview | Addresses the largest visible pool of declining-but-visible pages (real impressions, low CTR) | High — action volume and reason codes are directly from queue output |
| 3 | Treat **`expand_and_refresh`** (82 items) as a small, distinct high-effort bucket, not folded into the general refresh queue | Reason code `thin_visible_page` triggers only when word count is low *and* impressions are meaningful (>=250) — a narrow, specific condition | Small population but likely highest per-item lift, since thin content is the most directly actionable finding | Medium — small sample (82 of 30,000), not separately validated |
| 4 | Use the **13,093 `monitor`-flagged items** as a watchlist, not an active queue | These items lack a strong triggering reason code; the baseline assigns them `general_refresh_review` or no urgent flag | Avoids wasting review capacity on low-signal items while still tracking them | Medium — absence of a strong signal is not the same as confirmed stability |
| 5 | Re-run the pipeline on a rolling basis (e.g. monthly) rather than treating this as a one-time ranking | The label and features are all derived from a snapshot window; a single run cannot show whether the queue's precision holds over time | Turns a one-off classifier into a monitored decision-support tool, and creates the outcome data currently missing (Section 5.4) | Directional — this is a process recommendation, not a claim from the data itself |

### Action Playbook

**Now**
- Send the 3,602 high-confidence `refresh_and_review_ctr` / `refresh` items to editorial review, in ranked-score order, starting with the Top-10 preview in `outputs/model_report.md`.

**Next**
- Separately triage the 82 `expand_and_refresh` items as a focused thin-content sprint.
- Set up recurring pipeline runs so the `monitor` watchlist and precision figures can be tracked over time rather than treated as static.

**Later**
- Extend the dataset with a forward-looking outcome window (did visibility actually recover after a flagged page was refreshed?) so the model's recommendations can eventually be validated against real post-action results, not just against the trend label.

**Do not do**
- Do not treat the model's ranking as an automatic publishing or de-indexing decision. Per Section 5.4, no data here shows that acting on a flag changes the outcome — every high-confidence item still needs a human editorial check before action, exactly as `outputs/model_report.md`'s "Practical Use" section states.
- Do not apply this model or queue outside the 32-client, 30,000-row slice it was trained and validated on without re-validating; it has not been shown to generalize to the full internship dataset.

---

## 7. Reproducibility

The complete research chain is reproducible from the repository.

### Repository

`assignment-1` (FlyRank ML Internship starter repo, completed by the author) — GitHub Pages URL recorded in `submission/paper_url.txt` once deployed.

### Main pipeline

`scripts/run_all.py`, which runs, in order:
1. `01_prepare_features.py` — cleans the raw CSV, applies the impression/age filter, builds the 52-feature vector, defines `is_declining_label`.
2. `02_baseline_score.py` — computes the transparent weighted-rule baseline and reason codes.
3. `03_train_model.py` — trains logistic regression, decision tree, and random forest under the client-holdout split; selects the best model by Precision@50.
4. `04_evaluate_and_export.py` — builds the ranked refresh queue, confidence tiers, charts, and `outputs/model_report.md`.
5. `05_build_pdf_report.py` — exports a shareable PDF summary.

### Reproduction sequence

1. Obtain the repository (the bundled `data/raw/content_refresh_anonymized.csv` ships with it — no external data access is required to reproduce this paper).
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python scripts/run_all.py`.
4. Inspect `outputs/model_report.md`, `outputs/summary.json`, and `outputs/model_results.json` for the regenerated tables and figures.
5. Confirm the regenerated numbers match this paper (rows scored: 30,000; best model: random_forest; Precision@50: 0.740).

### Reproduction status

- [x] Fresh run completed (re-run during preparation of this paper; outputs matched the bundled `outputs/model_report.md` exactly)
- [x] Results match paper
- [x] Public-safe review completed (no client names, URLs, domains, titles, or raw queries appear anywhere in this document)
- [x] Notebook/pipeline saved in repository (`scripts/`, `work/notebooks/capstone.ipynb`)

---

## 8. Conclusion

This study demonstrates a structured approach to converting anonymized search-performance data into a defensible, ranked content-refresh queue. On a 30,000-row, 32-client anonymized slice, a random forest classifier trained on 52 search-behavior and content-context features identifies declining-visibility content substantially more precisely than a transparent hand-written rule (Precision@50 of 0.740 vs. 0.240), under a client-holdout validation split designed to resist the most obvious leakage path. The key value is not the model score in isolation, but the complete evidence chain: an explicit label definition with the derived trend columns excluded from features, a matched baseline for comparison, a leakage-resistant split, and a resulting queue with per-item reason codes that a content team can act on and audit. The clearest practical implication is that content teams reviewing this dataset's flagged pages should expect roughly three times more of their top-50 reviewed items to actually be declining than they would from the rule-based approach alone — but the study stops short of showing that acting on that queue improves outcomes, since no post-action data exists yet to test that claim.

---

## 9. Acknowledgments & Data Credit

This research was completed as part of the **FlyRank ML Internship**.

**Dataset credit:** FlyRank ML Internship search dataset (anonymized starter export, `content_refresh_anonymized.csv`).
**Dataset/research source:** https://flyrank.ai

The analysis and recommendations in this paper are the author's work based on the permitted internship data and the methodology documented in the repository.

---

## Appendix A — Required Evidence Checklist

- [x] Title present
- [x] Abstract present
- [x] Introduction / problem statement present
- [x] Data release and date window recorded
- [x] Exclusions documented
- [x] Methodology documented
- [x] Baseline documented
- [x] Validation documented
- [x] Leakage checks documented
- [x] Results contain actual metrics
- [x] Limitations are explicit
- [x] Ranked recommendations are evidence-based
- [x] Reproduction path is documented
- [x] FlyRank data credit is present
- [x] No private/sensitive data appears
- [ ] Paper deployed publicly — deploy via GitHub Pages, then check this box
- [ ] Exact deployed URL saved to `submission/paper_url.txt` — fill in after deployment
