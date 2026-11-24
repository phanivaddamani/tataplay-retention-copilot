# Tata Play Subscriber Retention & Support Copilot

An enterprise-grade, end-to-end Data & AI Copilot designed for subscriber retention, customer support triage, policy grounding, and proactive churn intervention across satellite TV and broadband households.

---

## Overview

Tata Play serves millions of households across India with satellite TV and broadband services. This repository implements an integrated, modular Copilot system addressing three fundamental operational bottlenecks faced by subscriber retention and support desks:
1. **Unreliable Raw Data Exports:** Resolves inconsistent entity naming, payment text pollution, orphan transaction records, duplicate entries, and invalid values into a clean, query-optimized relational database foundation (**Part A**).
2. **Untrustworthy Churn Predictions:** Implements a strictly leak-free machine learning pipeline that avoids target/temporal leakage, cross-validates with stratified splits, optimizes decision thresholds based on real operational business costs rather than default cutoff heuristics, and exposes model interpretability (**Part B**).
3. **Homogeneous Treatment of At-Risk Customers:** Constructs interpretable customer behavioral segments via K-Means clustering, predicts future individual customer spend using regularized regression (Ridge and Lasso), and ranks churn risk by actual Indian Rupee (INR) revenue exposed to prioritize desk outreach (**Part C**).
4. **Manual Support Overhead & Policy Inconsistencies:** Delivers an LLM-powered support copilot featuring few-shot ticket intent classification, resilient schema-validated ticket field extraction with automated self-correction retries, calibrated Retrieval-Augmented Generation (RAG) over corporate policies, and automated tailored retention briefs synthesizing database facts, ML scores, segments, and retrieved policy rules (**Part D**).

### Technology Stack & Architecture
- **Language & Runtime:** Python 3.11.18, SQLite 3
- **Data Engineering & ML:** `pandas 3.1.1`, `numpy 2.4.6`, `scikit-learn 1.9.1`, `joblib 1.6.0`
- **Generative AI & LLM Pipelines:** `langchain-core 1.8.4`, `pydantic 2.14.4`, `chromadb 1.5.9` (with `ONNXMiniLM_L6_V2` embedding model)
- **Networking & Fallbacks:** `requests 2.34.2`, offline deterministic `mock_llm.py` (with live endpoint switch via environment variables)

---

## Reproduce

The entire project is deterministically reproducible from a clean environment without external API keys or paid services.

### 1. Environment Setup
```bash
# Clone repository and enter project root
git clone <repository_url>
cd tataplay-retention-copilot

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Unix / macOS:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### 2. Pipeline Execution (In Order)
Execute each script in sequence, or run `python run_all.py` for end-to-end execution:

```bash
# Step 1: Generate deterministic raw synthetic data and policy documents
python generate_data.py

# Step 2: Part A - Data Foundation (Audit, Clean, SQLite Store, Query Execution)
python src/part_a_data.py

# Step 3: Part B - Leak-Free Churn Modeling, CV Tuning, Cost Threshold Optimization, Scoring
python src/part_b_churn.py

# Step 4: Part C - Customer Segmentation, Spend Regression, and Revenue at Risk Prioritization
python src/part_c_segments.py

# Step 5: Part D Task 1 - LLM Client Demonstrations & Resilient Backoff Testing
python src/test_llm_client.py

# Step 6: Part D Tasks 2 to 5 - Prompts, Extraction, Policy RAG Assistant, and Retention Copilot
python src/part_d_copilot.py

# Step 7: Standalone Copilot Brief Builder (Verifies single subscriber briefs)
python copilot.py
```

---

## Part A — Data Foundation: Audit, Clean, Store, Query

### 1. Raw Data Audit
The raw data exports represent transactions and subscriber attributes as of observation snapshot **2026-03-31**.

| Table | Metric / Column | Value | Notes |
| :--- | :--- | :--- | :--- |
| **subscribers.csv** | Raw Shape | `(2415, 17)` | 2,415 records |
| | Exact Duplicates | `15` | 15 completely identical rows |
| | Distinct `subscriber_id` | `2400` | Canonical distinct subscribers |
| | Blank `age` | `99` (4.10%) | Missing values |
| | Age outside 10–100 | `6` (0.25%) | Values: -1, 0, 150, 999 |
| | Blank `payment_mode` | `116` (4.80%) | Missing values |
| | Raw `city` spellings | `22` | Variants: Madras, Bombay, Hyd, etc. |
| | Raw `auto_renew` spellings | `6` | Variants: Y, yes, Yes, N, no, No |
| | Blank `next_cycle_recharge_inr` | `516` (21.37%) | Known only post-cycle; blank for churners |
| **recharges.csv** | Raw Shape | `(13055, 6)` | 13,055 records |
| | Exact Duplicates | `60` | 60 identical transaction rows |
| | Distinct `recharge_id` | `12995` | Distinct recharge records |
| | Non-numeric `amount_inr` | `366` | String formatting (e.g., "INR 498") |
| | Raw `status` spellings | `6` | Case variants (success, SUCCESS, etc.) |
| | Orphan Recharge Rows | `10` | `subscriber_id` not present in subscribers |

#### Missing Value Audit Table (Raw Subscribers)
| Column | Missing Count | Missing Pct | Raw Data Type |
| :--- | :--- | :--- | :--- |
| `subscriber_id` | 0 | 0.00% | `object` |
| `city` | 0 | 0.00% | `object` |
| `plan_id` | 0 | 0.00% | `object` |
| `tenure_months` | 0 | 0.00% | `int64` |
| `age` | 99 | 4.10% | `float64` |
| `payment_mode` | 116 | 4.80% | `object` |
| `auto_renew` | 0 | 0.00% | `object` |
| `add_on_count` | 0 | 0.00% | `int64` |
| `monthly_recharge_inr` | 0 | 0.00% | `int64` |
| `hours_watched_per_week` | 0 | 0.00% | `float64` |
| `hd_hours_per_week` | 0 | 0.00% | `float64` |
| `sports_hours_per_week` | 0 | 0.00% | `float64` |
| `channels_watched` | 0 | 0.00% | `int64` |
| `support_calls_90d` | 0 | 0.00% | `int64` |
| `closure_request_logged` | 0 | 0.00% | `object` |
| `next_cycle_recharge_inr` | 516 | 21.37% | `float64` |
| `churned` | 0 | 0.00% | `int64` |

#### Missing Value Audit Table (Raw Recharges)
All columns in raw `recharges.csv` (`recharge_id`, `subscriber_id`, `recharge_date`, `amount_inr`, `status`, `channel`) have 0 missing values (0.00%), though text anomalies were present in `amount_inr`.

---

### 2. Cleaning Transformations & Verification
- **Subscribers Deduplication:** 15 exact duplicate rows removed, leaving exactly **2,400** subscribers.
- **City Normalization:** Whitespace trimmed, case folded, and mapped canonical names (e.g., Bangalore -> Bengaluru, Bombay -> Mumbai, Madras -> Chennai, Calcutta -> Kolkata, New Delhi -> Delhi, Hyd -> Hyderabad).
- **Auto-Renew Standardized:** Mapped to binary `"Yes"` / `"No"`.
- **Age Bounds:** Values outside [10, 100] converted to `NaN`.
- **Recharge Cleaning:** 60 exact duplicates dropped; 10 orphan rows with invalid subscriber IDs removed (leaving **12,985** clean recharges).
- **Amount Rebuilding:** Exactly **364** text-formatted amounts (e.g., "INR 498") in valid subscriber rows were rebuilt using the subscriber's canonical `monthly_recharge_inr`. (2 non-numeric amounts belonged to the 10 dropped orphan rows, accounting for 366 total).
- **Status Standardized:** Normalized to lowercase (`success`, `failed`, `pending`).
- **No Imputation in Part A Policy:** Missing values in `age` (104) and `payment_mode` (114) were deliberately left un-imputed in Part A. Imputation must strictly reside inside the Part B preprocessing pipeline after train/validation/test splits to guarantee zero data leakage from evaluation partitions.

#### Clean Subscribers Summary
- **Shape:** `(2400, 17)`
- **City Counts:** Mumbai: 524, Delhi: 436, Chennai: 391, Bengaluru: 386, Hyderabad: 262, Pune: 245, Kolkata: 156.
- **Auto-Renew Counts:** No: 1,355, Yes: 1,045.
- **Remaining Missing:** `age`: 104, `payment_mode`: 114.
- **Churn Rate:** 514 churned out of 2,400 (**21.42%**).

#### Clean Recharges Summary
- **Shape:** `(12985, 6)`
- **Status Breakdown:** `success`: 11,442, `failed`: 1,422, `pending`: 121.
- **Total Successful Recharge Revenue:** **INR 6,457,573**.

---

### 3. SQLite Storage & Analytical SQL Queries

All cleaned tables (`subscribers`, `recharges`, `plans`) were persisted into `data/tataplay.db`. The queries are maintained in `sql/queries.sql`.

#### Q1: Total Revenue and Average Spend by Plan
```sql
SELECT 
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue,
    ROUND(AVG(r.amount_inr), 2) AS avg_recharge,
    COUNT(DISTINCT s.subscriber_id) AS distinct_subscribers
FROM recharges r
INNER JOIN subscribers s ON r.subscriber_id = s.subscriber_id
INNER JOIN plans p ON s.plan_id = p.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue DESC;
```
| Plan Name | Total Revenue (INR) | Average Recharge (INR) | Distinct Subscribers |
| :--- | :--- | :--- | :--- |
| **Premium** | 1,471,199 | 860.35 | 339 |
| **Standard** | 1,323,622 | 492.60 | 559 |
| **Standard HD** | 1,283,471 | 645.61 | 407 |
| **Premium Plus** | 1,262,398 | 1,075.30 | 228 |
| **Basic** | 782,554 | 326.61 | 493 |
| **Basic Lite** | 334,329 | 224.83 | 298 |

*Groupby Confirmation:* Verified in pandas for Premium plan: Total = 1,471,199.0, Avg = 860.35, Distinct Subscribers = 339.

#### Q2: Subscribers Who Never Successfully Recharged
```sql
SELECT 
    s.subscriber_id,
    s.tenure_months,
    s.monthly_recharge_inr
FROM subscribers s
LEFT JOIN recharges r 
    ON s.subscriber_id = r.subscriber_id 
    AND r.status = 'success'
WHERE r.recharge_id IS NULL;
```
- **Count:** Exactly **76** subscribers have never had a successful recharge, of which **68** have `tenure_months = 0` (brand-new accounts who churned or failed before initial payment).
- **SQL Rationale:** An `INNER JOIN` requires matching keys in both tables and would silently drop all 76 subscribers because they have no records in `recharges` where `status = 'success'`. This would cause retention and churn models to completely overlook the highest-friction onboarding dropouts.

#### Q3: Ranking Within City
```sql
WITH subscriber_spend AS (
    SELECT 
        s.subscriber_id,
        s.city,
        SUM(r.amount_inr) AS total_spend
    FROM subscribers s
    INNER JOIN recharges r ON s.subscriber_id = r.subscriber_id
    WHERE r.status = 'success'
    GROUP BY s.subscriber_id, s.city
),
ranked_spend AS (
    SELECT 
        subscriber_id,
        city,
        total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, subscriber_id ASC) AS row_num
    FROM subscriber_spend
)
SELECT * FROM ranked_spend;
```
- **Subscribers with `RANK() = 1` per city:** Bengaluru: 10, Chennai: 10, Delhi: 5, Hyderabad: 5, Kolkata: 2, Mumbai: 6, Pune: 4. (In contrast, `ROW_NUMBER() = 1` is strictly 1 per city).
- **Kolkata Top 4 Rows:**
  1. `TP00074`: INR 6,876 (RANK: 1, DENSE_RANK: 1, ROW_NUMBER: 1)
  2. `TP00917`: INR 6,876 (RANK: 1, DENSE_RANK: 1, ROW_NUMBER: 2)
  3. `TP00143`: INR 6,582 (RANK: 3, DENSE_RANK: 2, ROW_NUMBER: 3)
  4. `TP01355`: INR 6,582 (RANK: 3, DENSE_RANK: 2, ROW_NUMBER: 4)
- **Window Function Selection Rationale:** To award or identify the "top spender per city", one must use `RANK()` because it fairly includes all tied top spenders while correctly skipping subsequent ranks. `ROW_NUMBER()` arbitrarily picks one tied subscriber based on tie-breaking order (silently omitting genuine tied top spenders), while `DENSE_RANK()` does not reflect the correct count of superior spenders when looking beyond the top tier.

#### Q4: Running Total Spend for Subscriber TP00010
```sql
SELECT 
    recharge_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (ORDER BY recharge_date ASC, recharge_id ASC) AS running_total
FROM recharges
WHERE subscriber_id = 'TP00010' AND status = 'success'
ORDER BY recharge_date ASC, recharge_id ASC;
```
| Recharge ID | Recharge Date | Amount (INR) | Running Total (INR) |
| :--- | :--- | :--- | :--- |
| `R000053` | 2025-10-15 | 498 | 498 |
| `R000055` | 2025-12-15 | 498 | 996 |
| `R000056` | 2026-01-15 | 498 | 1,494 |
| `R000057` | 2026-02-15 | 498 | 1,992 |
| `R000058` | 2026-03-13 | 498 | **2,490** |

---

## Part B — Churn Prediction You Can Trust

### 1. Problem Framing
| Dimension | Specification |
| :--- | :--- |
| **Decision Driven** | Proactive outbound outreach, discount offer, or technical support dispatch before billing cycle ends. |
| **Actor & Action** | Retention Desk agents offer loyalty discounts, free add-on packs, or payment setup calls to high-risk subscribers. |
| **Target Column** | `churned` (1 if cancelled in April 2026, 0 if retained). |
| **Target Type** | Binary Category / Indicator. |
| **Error Costs** | **False Negative (FN):** INR 1,200 (lost monthly customer revenue).<br>**False Positive (FP):** INR 300 (unnecessary discount / concession cost). |
| **Class Balance** | 514 churners out of 2,400 (**21.42%**). |

---

### 2. Feature Engineering & Leakage Experiment
Recharge features were computed on clean historical data up to snapshot `2026-03-31`:
- `n_success_6m`: Count of successful recharges in 6-month window.
- `n_failed_6m`: Count of failed recharges in 6-month window.
- `recency_days`: Days between 2026-03-31 and the last successful recharge (missing for 76 subscribers).

**12 Numeric Features:** `tenure_months`, `age`, `add_on_count`, `monthly_recharge_inr`, `hours_watched_per_week`, `hd_hours_per_week`, `sports_hours_per_week`, `channels_watched`, `support_calls_90d`, `n_success_6m`, `n_failed_6m`, `recency_days`.  
**4 Categorical Features:** `city`, `plan_id`, `payment_mode`, `auto_renew`.

#### Data Splitting (3-Way Stratified)
- **Train Set (60%):** 1,440 rows (308 churners, 21.39%) — used for fitting estimators and cross-validation hyperparameter search.
- **Validation Set (20%):** 480 rows (103 churners, 21.46%) — used for baseline comparisons and cost-sensitive threshold tuning.
- **Test Set (20%):** 480 rows (103 churners, 21.46%) — locked holdout used strictly once to evaluate final generalization performance.

#### Leakage Demonstration
Two columns present in the raw data must never be used as predictive features:
1. `closure_request_logged`: Logged during the cancellation process itself.
2. `next_cycle_recharge_inr`: Realized outcome recorded after April 2026 completes.

| Model Pipeline | Validation Accuracy | Validation F1 (Churn Class) |
| :--- | :--- | :--- |
| **With Leakage** (Random Forest) | **0.983** | **0.961** |
| **Without Leakage** (Random Forest) | **0.823** | **0.514** |

*Leakage Explanation:* The leaky model achieves an artificially inflated F1 of 0.961 because it cheats by observing signals generated after the subscriber has already initiated departure. In production, these features do not exist at inference time; deploying a leaky model would cause catastrophic real-world failure as the model relies on signals that are absent when proactive intervention is required.

---

### 3. Leak-Free Preprocessing Pipeline
A single `ColumnTransformer` was constructed and fitted **only on the training split**:
- **Numeric Pipeline:** `SimpleImputer(strategy="median")` followed by `StandardScaler()`.
- **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
- **Transformed Columns:** Exactly **31** features.
- **Learned Imputer Median Age:** `38.0`.
- **Learned Scaler Mean Tenure:** `22.65` months (versus full-data mean `22.46` months).
- *Difference Explanation:* The learned scaler mean is computed exclusively on the 1,440 training rows to prevent data snooping; the slight difference from the full dataset mean reflects standard sampling variation across the stratified split. Code checks confirmed 0 missing values remain post-transformation.

---

### 4. Baseline & Simple Models (Validation Set)
| Model | Training Accuracy | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** ("Stays") | 0.786 | 0.785 | 0.000 | 0.000 | 0.000 |
| **Logistic Regression** | 0.835 | 0.833 | 0.646 | 0.495 | 0.560 |
| **Unrestricted Decision Tree** | 1.000 | 0.767 | 0.457 | 0.466 | 0.459 |
| **Default Random Forest** | 1.000 | 0.823 | 0.627 | 0.437 | 0.514 |

#### Decision Tree Depth Sweep
| Depth | Training Accuracy | Validation Churn F1 |
| :---: | :---: | :---: |
| 1 | 0.815 | 0.434 |
| 2 | 0.832 | 0.413 |
| 3 | 0.839 | 0.408 |
| 4 | 0.844 | 0.422 |
| 5 | 0.863 | 0.529 |
| **6** | **0.883** | **0.545** *(Peak F1)* |
| 7 | 0.908 | 0.488 |
| 8 | 0.924 | 0.489 |
| 9 | 0.944 | 0.468 |
| 10 | 0.957 | 0.440 |

*Overfitting Explanation:* As tree depth expands from 1 to 10, training accuracy climbs monotonically to 95.7% while validation F1 peaks at depth 6 and rapidly degrades to 0.440. A greedy decision tree recursively partitions noisy idiosyncrasies of the training data into pure leaf nodes, resulting in high variance and poor generalization on unseen data.

---

### 5. Hyperparameter Tuning via 5-Fold Cross-Validation
Tuning was performed on training data using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` optimizing for churn-class F1.
- **Logistic Regression Grid:** `C ∈ {0.01, 0.1, 1, 10, 100}` (5 fits).
  - *Best Param:* `C = 100`.
  - *Best CV F1:* **0.507** (std: `0.035`).
- **Random Forest Grid:** `n_estimators ∈ {50, 100, 150}`, `max_depth ∈ {3, 4, 5, 6, 8}`, `max_features ∈ {3, 4, 5}` (45 combinations, 225 total fits).
  - *Best Params:* `max_depth = 8`, `max_features = 5`, `n_estimators = 100`.
  - *Best CV F1:* **0.439** (std: `0.061`).
- *Consistency Analysis:* Logistic Regression displays greater stability across folds (standard deviation of 0.035 vs 0.061 for Random Forest). Smaller cross-validation variance indicates less sensitivity to fold sampling perturbations, ensuring more predictable performance in production.

---

### 6. Cost-Sensitive Threshold Selection & Test Set Evaluation
Using predicted probabilities from `predict_proba` on the validation set, decision thresholds were evaluated across $[0.10, 0.90]$ in steps of 0.05 using the operational cost equation:
$$\text{Cost} = 1200 \times \text{FN} + 300 \times \text{FP}$$

- **Validation Cost Optimal Thresholds:**
  - Random Forest: threshold **0.30** $\rightarrow$ Validation Cost = **INR 53,100**.
  - Logistic Regression: threshold **0.20** $\rightarrow$ Validation Cost = **INR 56,400**.

#### Single Test-Set Evaluation (Thresholds Locked from Validation)
| Evaluation Policy | TN | FP | FN | TP | Precision | Recall | F1 | Total Cost (INR) | % of No-Action Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Action Baseline** | 377 | 0 | 103 | 0 | 0.000 | 0.000 | 0.000 | 123,600 | 100.0% |
| **Offer Everybody** | 0 | 377 | 0 | 103 | 0.215 | 1.000 | 0.353 | 113,100 | 91.5% |
| **Logistic Regression (th=0.20)** | 275 | 102 | 17 | 86 | 0.457 | 0.835 | 0.591 | 51,000 | 41.3% |
| **Selected Random Forest (th=0.30)**| **310**| **67** | **29**| **74** | **0.525** | **0.718** | **0.607** | **54,900** | **44.4%** |

#### Model Selection Justification
Random Forest was selected based on validation cost optimization (INR 53,100 vs INR 56,400 for Logistic Regression) and superior precision (0.525 vs 0.457). While Logistic Regression captures slightly more churners at its aggressive 0.20 threshold, it generates 102 false positives (compared to only 67 for Random Forest), placing excessive operational load on retention agents and eroding concession margins. The selected Random Forest comfortably satisfies all project acceptance criteria:
- Cost (INR 54,900) is **44.4%** of No-Action (well below the 60% ceiling of INR 74,160).
- Test Recall is **0.718** (exceeding the 0.55 requirement).
- Test F1 is **0.607** (exceeding the 0.50 requirement).

---

### 7. Model Interpretability & Artifacts
#### Top 5 Random Forest Features (`feature_importances_`)
1. `num__support_calls_90d` (0.1961): Frequent recent complaints strongly signal dissatisfaction and immediate departure risk.
2. `num__recency_days` (0.0906): Subscribers whose last recharge is distant are drifting into inactivity.
3. `num__tenure_months` (0.0818): Tenured subscribers possess stickiness; early-tenure subscribers churn most readily.
4. `num__n_success_6m` (0.0786): Consistent multi-month payment history reflects embedded routine.
5. `num__hd_hours_per_week` (0.0668): High visual engagement with HD content reinforces retention.

#### Key Logistic Regression Coefficients
- `num__support_calls_90d` (`+1.0747`): Each standard deviation increase in support calls doubles odds of churn.
- `num__tenure_months` (`-0.5112`): Higher tenure strongly reduces log-odds of churning.
- `num__n_failed_6m` (`+0.2111`): Payment friction and failed attempts escalate customer attrition.

#### Persisted Artifacts
- Model saved to `outputs/churn_model.joblib`. Reload verification confirmed identical predictions.
- Test predictions written to `outputs/test_scores.csv` (480 rows).

---

## Part C — Segments, Spend Model and Revenue at Risk

### 1. K-Means Clustering Sweep
Using standardized `tenure_months`, `hours_watched_per_week`, and `monthly_recharge_inr` across all 2,400 clean subscribers:

| K (Clusters) | Inertia | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 3,728.2 | 0.420 |
| 3 | 2,813.0 | 0.346 |
| **4 (Chosen)** | **2,193.9** | **0.340** |
| 5 | 1,781.1 | 0.343 |
| 6 | 1,502.5 | 0.344 |
| 7 | 1,292.8 | 0.364 |
| 8 | 1,191.4 | 0.336 |

*Clustering Trade-offs:* Inertia always decreases monotonically as K increases because adding centroids reduces within-cluster sum of squares, reaching zero when $K=N$. K=2 yields the highest silhouette score by dividing subscribers into coarse halves, but this fails to provide actionable business differentiation. $K=4$ was selected because it captures the elbow inflection and maps directly to the operational archetypes of the subscriber base.

---

### 2. Segment Profiles & Hand Assignment (K=4)
| Segment ID | Name | Size | Mean Tenure (mo) | Mean Weekly Hours | Mean Monthly Recharge (INR) | Churn Rate | Recommended Desk Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | **Budget Linear** | 593 | 23.6 | 7.1 | 301.3 | 22.4% | Value packs, affordable regional add-on promotions. |
| 1 | **Family Streamers** | 543 | 34.2 | 22.2 | 559.7 | 13.4% | Multi-room, kid/infotainment bundle retention. |
| 2 | **Premium** | 480 | 34.2 | 33.6 | **942.3** | 10.6% | Priority engineer concierge, VIP upgrades. |
| 3 | **At-Risk** | 784 | 6.3 | 14.9 | 513.5 | **32.8%** | Immediate onboarding intervention, billing setup call. |

*Saved to `outputs/segments.csv` (2,400 rows).*

#### Hand Assignment Verification
New subscriber: `tenure = 4 months`, `weekly_hours = 15.0`, `monthly_recharge = INR 520`.
- Scaled vector: `[-1.2063, -0.3226, -0.1429]`.
- Distance to Cluster 0 (Budget Linear): `1.7083`
- Distance to Cluster 1 (Family Streamers): `2.0937`
- Distance to Cluster 2 (Premium): `3.1170`
- **Distance to Cluster 3 (At-Risk):** **0.1529**
- *Agreement:* Hand calculation identifies Cluster 3 (At-Risk) as the nearest centroid at distance 0.1529 (nearest alternative is 1.7083), agreeing perfectly with `kmeans.predict()`.

---

### 3. Collinearity & Spend Estimation Models
#### Highly Correlated Viewing Features ($|r| > 0.90$)
1. `hours_watched_per_week` — `hd_hours_per_week`: **0.979**
2. `hours_watched_per_week` — `sports_hours_per_week`: **0.947**
3. `hours_watched_per_week` — `channels_watched`: **0.938**
4. `hd_hours_per_week` — `sports_hours_per_week`: **0.927**
5. `hd_hours_per_week` — `channels_watched`: **0.921**  
*(sports_hours_per_week — channels_watched: 0.891)*

#### Spend Model on Non-Churners (1,508 Train / 378 Test)
Target: `next_cycle_recharge_inr`.

| Model | Tuned Alpha | 5-Fold CV $R^2$ | Test $R^2$ | Test MAE (INR) | Test RMSE (INR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | N/A | 0.9941 | 0.9930 | 20.16 | 24.70 |
| **Ridge Regression** | **0.01** | **0.9941** | **0.9930** | **20.16** | **24.70** |
| **Lasso Regression** | 0.10 | 0.9941 | 0.9930 | 20.11 | 24.65 |

#### Lasso Feature Selection (Standardized Coefficients)
- `monthly_recharge_inr`: `+269.6000`
- `hd_hours_per_week`: `+30.9803`
- `sports_hours_per_week`: `+23.8672`
- `support_calls_90d`: `-0.9309`
- `tenure_months`: **`0.0000`** *(Dropped)*
- `hours_watched_per_week`: **`0.0000`** *(Dropped)*
- `channels_watched`: **`0.0000`** *(Dropped)*

*Lasso Redundancy Explanation:* When features are collinear, their linear combination carries redundant information. Lasso’s L1 penalty shrinks the objective along coordinate axes, zeroing out collinear proxies (`hours_watched_per_week`, `channels_watched`, `tenure_months`) while preserving the most predictive specific components (`monthly_recharge_inr`, `hd_hours_per_week`, `sports_hours_per_week`).

#### Subsample Stability Experiment (5 Random 80% Folds)
Coefficient of `hours_watched_per_week` on standardized data:
- **OLS Fits:** `[-5.731, -1.712, -4.839, -7.736, -4.733]`
  - Population Standard Deviation (ddof=0): **1.95** (Sample std, ddof=1: **2.17**)
- **Ridge ($\alpha=30$) Fits:** `[+13.260, +15.554, +13.581, +13.024, +13.786]`
  - Population Standard Deviation (ddof=0): **0.90** (Sample std, ddof=1: **1.00**)

*Stability Conclusion:* In the presence of strong multicollinearity, OLS coefficient estimates suffer from severe variance and flip wildly depending on the subsample. Ridge regularization conditions the matrix inversion, shrinking coefficient variance by more than half (0.90 vs 1.95), proving that individual OLS coefficients cannot be trusted when features are correlated.

---

### 4. Revenue at Risk Prioritization
For all 480 test subscribers, next-cycle recharge was estimated with the tuned Ridge model and combined with Part B's predicted churn probabilities:
$$\text{Revenue at Risk} = P(\text{Churn}) \times \widehat{\text{Next Recharge}}$$

The top 15 subscribers with the highest revenue at risk were exported to `outputs/retention_priority.csv`:

| Subscriber ID | Churn Probability | Predicted Recharge (INR) | Revenue at Risk (INR) | Segment Name | Actual Churned |
| :---: | :---: | :---: | :---: | :--- | :---: |
| `TP00798` | 0.8176 | 973.55 | 795.97 | Premium | 1 |
| `TP02346` | 0.7270 | 1012.06 | 735.77 | Premium | 1 |
| `TP02228` | 0.5828 | 1194.10 | 695.92 | Premium | 1 |
| `TP00923` | 0.5319 | 1298.76 | 690.81 | Premium | 1 |
| `TP00986` | 0.7046 | 908.66 | 640.24 | At-Risk | 1 |
| `TP00453` | 0.5758 | 1037.07 | 597.14 | At-Risk | 1 |
| `TP00061` | 0.6887 | 866.39 | 596.68 | At-Risk | 1 |
| `TP01620` | 0.6533 | 893.39 | 583.65 | At-Risk | 1 |
| `TP00188` | 0.6572 | 784.09 | 515.30 | Family Streamers | 1 |
| `TP02317` | 0.4429 | 1116.04 | 494.29 | Premium | 1 |
| `TP01548` | 0.3662 | 1320.75 | 483.66 | Premium | 0 |
| `TP01015` | 0.4021 | 1195.13 | 480.56 | Premium | 0 |
| `TP00834` | 0.6513 | 726.82 | 473.38 | At-Risk | 1 |
| `TP00447` | 0.4435 | 1059.25 | 469.78 | At-Risk | 0 |
| `TP02113` | 0.7454 | 621.59 | 463.33 | Family Streamers | 1 |

- **Total Revenue at Risk (Top 15):** **INR 8,716.48**
- **Actual Churners in Top 15:** **12 of 15 (80.0%)** (far exceeding the 21.4% base rate and satisfying the requirement of at least 9).
- *Evaluation Disclaimer:* This priority table is constructed from the held-out test partition whose eventual outcomes are known; it illustrates the prioritization methodology rather than a forward-looking production deployment.

---

## Part D — Support Copilot: Prompts, Validation, Retrieval, Retention Briefs

### 1. Resilient LLM Client (`llm_client.py`)
The client wraps `mock_llm.py` (with live endpoint switching via `LLM_MODE=live` reading `OPENROUTER_API_KEY`). It implements exponential backoff on HTTP 429 (`BASE_DELAY * 2**n` for $n=0,1,2$), immediate fatal errors on 400/401/404, and notes `finish_reason == "length"`.

#### Client Demonstrations
1. **Authentication & Endpoint Validation:** Missing `Authorization` header triggers HTTP 401; an incorrect URL triggers HTTP 404.
2. **Length Truncation:** Request with `max_tokens=3` successfully returns `finish_reason = "length"`.
3. **Statelessness:** Prompting `"What is my name?"` without conversation history returns `"I do not know your name."`, while sending prior conversation turns (`"My name is Asha."`) returns `"Your name is Asha."`.
4. **Rate Limit Recovery:** Running 30 sequential requests triggered exactly 3 HTTP 429 responses (at every 9th call), with all 3 automatically recovered via backoff across **33** total API invocations.

---

### 2. Prompt Engineering & Ticket Classification
Evaluated across all 146 readable tickets from `data/raw/tickets.csv` (excluding four unreadable `...` tickets):

| Prompt Strategy | Output Conformance | Accuracy vs True Label | Recovered 429 Errors |
| :--- | :---: | :---: | :---: |
| **Weak Prompt** | 0 / 146 valid labels (0.0%) | 0.00% | 18 |
| **Strong Zero-Shot** | 146 / 146 valid labels (100.0%) | 84.25% (123 / 146) | 18 |
| **Few-Shot Prompt** | **146 / 146 valid labels (100.0%)** | **97.26% (142 / 146)** | **18** |

*Saved to `outputs/ticket_classification.csv`.*

#### Misclassified Ticket Analysis (Few-Shot Prompt)
Only 4 tickets were misclassified by the few-shot prompt:
1. `T0041` (True: `plan_change`, Pred: `billing`): *"Subscriber ID TP01253. My last recharge went through fine, now I want to move to the Premium plan."*
2. `T0094` (True: `cancellation`, Pred: `signal`): *"I want to cancel my subscription because the picture keeps freezing."*
3. `T0110` (True: `signal`, Pred: `cancellation`): *"Subscriber ID TP00447. I do not want to cancel, but there is no signal since last night."*
4. `T0136` (True: `signal`, Pred: `cancellation`): *"Subscriber ID TP02248. I do not want to cancel, but there is no signal since last night. Please help."*

- **Common Root Cause:** All four errors arise from mixed customer intents where secondary issues or negated phrases (`"do not want to cancel"`, `"last recharge went through fine"`) mislead keyword-weighted scoring.
- **Proposed Fix:** Introduce intent-order and negation heuristics into the system instructions, instructing the model to prioritize actionable primary desires over background history or explicit negations.
- **Chain-of-Thought (CoT) Assessment:** CoT prompting is justified for complex multi-hop reasoning tasks (e.g., policy eligibility arithmetic or SLA penalty calculations) where intermediate reasoning steps prevent logical errors. For single-intent classification on short tickets, CoT is unnecessary and triples latency and token consumption without tangible gain.

---

### 3. Validated Structured Extraction with Automated Retry
Implemented using Pydantic `TicketRecord`:
```python
class TicketRecord(BaseModel):
    subscriber_id: Optional[str] = None
    category: Literal["billing", "signal", "plan_change", "cancellation", "other"]
    urgency: int = Field(ge=1, le=5)
    wants_refund: bool
```

#### Pydantic Validation Demonstrations
- Type Coercion: String `"3"` successfully coerced to integer `3`.
- Out-of-Bounds: `urgency = 9` raises `ValidationError` (`field='urgency', error_type='less_than_equal'`).
- Invalid Literal: `category = 'refund'` raises `ValidationError` (`field='category', error_type='literal_error'`).

#### Batch Extraction Results (150 Tickets)
- **First Attempt Valid:** **118**
- **Valid After Retry:** **28**
- **Manual Review (Corrupted `...` tickets):** **4**
- **First Attempt Failures (32 Total):**
  - Invalid JSON / Chatty Prose: **16** (12 chatty + 4 unreadable)
  - Schema Invalidation - Urgency = 7: **6**
  - Schema Invalidation - Non-standard Category: **10**
- **Recovered 429 Errors:** **23**
- **Valid Record Properties:** 31 requests for refund (`wants_refund == True`), 75 unmentioned subscriber IDs (`subscriber_id is None`).
- *Production Triage Policy:* Unparseable tickets designated as `manual_review` are routed to human triage queues with SLA tagging to guarantee zero unhandled customer complaints.
- *Saved to `outputs/ticket_extractions.jsonl`.*

---

### 4. Policy Retrieval-Augmented Generation (RAG)
Policy documents in `data/policies/` were partitioned using `chunk_text(size=400, overlap=60)`, producing exactly **13** chunks:
- `cancellation_policy.txt`: 2 chunks
- `plan_change_policy.txt`: 3 chunks
- `recharge_and_refund_policy.txt`: 3 chunks
- `retention_offers_policy.txt`: 3 chunks
- `technical_support_policy.txt`: 2 chunks

Embedded via Chroma's `ONNXMiniLM_L6_V2` embedding model using cosine similarity ($\text{Similarity} = 1 - \text{distance}$).

#### Evaluation on 16 Policy Questions (`policy_eval.csv`)
- **Top-3 Retrieval Accuracy (12 Answerable):** **12 of 12 (100.0%)**
- **Top-1 Retrieval Accuracy (12 Answerable):** **12 of 12 (100.0%)**
- **Answerable Top-1 Similarity Range:** `0.334` to `0.637` (mean: `0.514`)
- **Unanswerable Top-1 Similarities:** `0.069`, `-0.047`, `0.295`, `0.304`
- **Keyword Baseline Accuracy:** **9 of 12 (75.0%)**

#### Semantic vs Keyword Retrieval
- *Why Semantic Search Wins:* Questions frequently use customer vernacular (`"top-up"`, `"cheaper pack"`, `"breaking up"`) that diverges lexically from formal policy terms (`"recharge"`, `"downgrade"`, `"pixelation"`). Dense vector embeddings project semantic meanings into proximity, retrieving relevant chunks where keyword intersection fails.
- *Closest Borderline Question:* The unanswerable query *"Is international roaming available on the mobile app?"* yielded top-1 similarity of `0.304` because *"mobile app"* heavily aligns with app buffering clauses in `technical_support_policy.txt`.
- *Rejection Calibration:* Calibrated threshold of **0.32** sits between the highest unanswerable (0.304) and lowest answerable (0.334) similarity scores.

#### Grounding & Chunk Size Comparison
| Metric | 400 Chars / 60 Overlap | 200 Chars / 30 Overlap |
| :--- | :---: | :---: |
| **Total Chunks Produced** | 13 | 25 |
| **Expected Fact Present in Top-3 Chunks** | **12 / 12 (100.0%)** | 11 / 12 (91.7%) |
| **Grounded Answer Contains Expected Fact** | **12 / 12 (100.0%)** | 8 / 12 (66.7%) |
| **Unanswerable Fallback Rate** | **4 / 4 (100.0%)** | 4 / 4 (100.0%) |

*Chunk Size Degradation Explanation:* Shrinking chunks to 200 characters fragments contextual sentences across boundaries (e.g., separating conditional eligibility criteria from their corresponding concession rules), depriving the model of necessary context and dropping grounding accuracy from 100% to 66.7%.

*Saved to `outputs/rag_eval.csv`.*

---

### 5. The Retention Copilot (`copilot.py`)
`copilot.py` generates structured, policy-grounded retention briefs for the top 15 revenue-at-risk subscribers. For each subscriber:
1. Re-scores database facts using `outputs/churn_model.joblib` (asserting 4-decimal agreement with Part B).
2. Assigns Risk Band: **High** ($\ge 0.40$), **Medium** ($[0.20, 0.40)$), **Low** ($< 0.20$).
3. Retrieves top-3 policy chunks and prompts the mock LLM with `RETENTION BRIEF`.
4. Enforces Pydantic schema validation (`RetentionBrief`) and strict consistency checks.

#### Policy Rule Logic & Verified Distribution (Top 15 Subscribers)
| Rule Condition | Policy Entitlement | Observed Count |
| :--- | :--- | :---: |
| **High Risk + Failed Recharges $\ge 3$** | Offer an auto-renew setup call and a 7-day grace extension. | **4** |
| **High Risk + Failed Recharges $< 3$ + Tenure $\ge 12$ mo** | Offer a loyalty discount of 15 percent on the next three recharges. | **4** |
| **High Risk + Failed Recharges $< 3$ + Tenure $< 12$ mo** | Offer a free add-on pack for 2 months. | **6** |
| **Medium / Low Risk Band** | No retention offer applies; keep monitoring the account. | **1** |
| **Total Valid Briefs** | All 15 verified consistent against policy source `retention_offers_policy.txt` | **15** |

*Risk Band Breakdown:* **14 High**, **1 Medium** (TP01548, churn probability = 0.3662).  
*Saved to `outputs/retention_briefs.jsonl`.*

---

## Limitations and Next Steps

1. **Synthetic Data Boundaries:** All customer attributes, tickets, and transaction histories were deterministically generated. Real customer behavior exhibits seasonality, streaming bundle migrations, festival recharge surges, and regional linguistic nuances that require broader domain modeling.
2. **Single Observation Time Window:** Predictions are conditioned on a single 6-month window ending March 31, 2026. Production systems require rolling multi-cohort training and backtesting across rolling quarters to track temporal drift.
3. **Deterministic Mock LLM vs Production GenAI:** Language model evaluations were executed against the supplied deterministic mock client. While the mock replicates latency, rate limits, schema slips, and keyword matching, real hosted LLMs introduce non-deterministic generation, hallucinations, and format variations requiring active guardrails.
4. **Historical Evaluation of Revenue at Risk:** The top 15 retention priority table is constructed from the held-out test partition for validation and demonstration purposes. In operational production, this pipeline runs forward-looking scoring on active subscribers whose churn outcomes have not yet occurred.
5. **Narrow Retrieval Rejection Margin:** The calibrated rejection threshold (0.32) operates with a thin margin ($\approx 0.016$) between the highest unanswerable similarity (0.304) and lowest answerable similarity (0.334). On larger, real-world policy corpora, fine-tuning embedding models with domain contrastive learning or utilizing cross-encoder rerankers is recommended to expand rejection margins.
