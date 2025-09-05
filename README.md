# CEFR Level Prediction from Engineered Features

A multi-page **Streamlit** dashboard that predicts a learner’s **CEFR level (A1–C2)** from **engineered, non-leaky features**, with supporting EDA, model evaluation, and a live **Predict CEFR** page.

---

## Dataset Content

**Source:** Proprietary dataset created by the project author (this repo).
**Unit of analysis:** one row per learner.

The original (clean) dataset includes raw exam outcomes and a human label:

* **Raw exam scores:** `speaking_score`, `reading_score`, `listening_score`, `writing_score`
* **Target (human label):** `overall_cefr` ∈ {A1, A2, B1, B2, C1, C2}

To avoid **target leakage**, the modelling task uses **engineered features** only (no raw totals). The feature matrix used for training and in the Predict page includes:

* **Relative skill patterning (numeric):**
  `speaking_minus_avg`, `reading_minus_avg`, `listening_minus_avg`, `writing_minus_avg`, `strength_weakness_gap`, `productive_dominant` (0=receptive-tilted, 1=productive-tilted)
* **Skill profile (categorical):**
  `strongest_skill`, `weakest_skill`, `second_weakest_skill`, `learning_profile` (Balanced / Uneven Development),
  `speaking_level`, `reading_level`, `listening_level`, `writing_level` (Beginner / Intermediate / Advanced)
* **Encoded target (for training):** `cefr_encoded` with mapping A1→0 … C2→5

**Clean data path:** `data/clean/cleaned_lang_proficiency_results.csv`

---

## Project Terms & Jargon

* **CEFR:** Common European Framework of Reference, ordered levels **A1 < A2 < B1 < B2 < C1 < C2**
* **Target leakage:** Using features that directly encode the label (e.g., raw total/average score)
* **Macro-F1:** Per-class F1 averaged equally (fairness across classes)
* **Weighted-F1:** Per-class F1 weighted by class frequency


---

## Business Requirements


1. **The client has asked us to** analyse the learner dataset to surface patterns driving CEFR outcomes, **quantify class imbalance**, and **detect target leakage**; from these findings, define **engineered, non-leaky features** suitable for modelling.

2. **The client has asked us to** build and operationalise a **fair, interpretable CEFR classifier** using those engineered features, meeting **Accuracy ≥ 0.75** and **Macro-F1 ≥ 0.70**, and expose a Streamlit **Predict CEFR** page with predicted label, class probabilities, and simple learning recommendations.

---

## Hypotheses and how to validate


* **H1 — Raw totals are (too) predictive of CEFR → leakage risk.**
  *Validate with* correlation/PPS vs `overall_cefr`, and demonstrate a redesign: **drop raw scores** in modelling and use engineered features only.

* **H2 — Class imbalance exists (C1/C2 under-represented).**
  *Validate with* class count plots; address via **macro-F1** optimisation, cross-validation, and results commentary by class.

---

## Rationale: mapping requirements → visualisations & ML tasks


* **BR1 → EDA / Feature design**

  * Data overview and schema
  * Class balance (bar plot)
  * Correlation/PPS study to evidence leakage
  * Definition of **engineered features** (relative skill patterns and profiles)
  * Commit to **dropping raw scores** for modelling
* **BR2 → Classification + Evaluation + Inference UI**

  * Train/tune **Logistic Regression, Random Forest, XGBoost** via `GridSearchCV(cv=5, scoring="f1_macro")`
  * Report **test-set** metrics and **cross-validation** summary
  * **Decision rule:** highest Macro-F1 (tie-break on Accuracy; if within ±0.01, prefer simpler model)
  * Deploy the selected model to the **Predict CEFR** page (live inputs → prediction + probabilities + simple recommendations)

---

## ML Business Case — Predict CEFR (Classification)

* **Aim:** Predict CEFR level (A1–C2) from **engineered, non-leaky** features to support fair, explainable placement.
* **Learning method:** Supervised multiclass classification; tuned **LR / RF / XGB** (`f1_macro` with 5-fold CV).
* **Ideal outcome:** Accurate and balanced performance across levels; actionable probabilities for human review.
* **Success criteria:** **Accuracy ≥ 0.75**, **Macro-F1 ≥ 0.70** (test set).
* **Failure conditions:** Materially below thresholds or unstable across folds; heavy reliance on leaky features.
* **Model output & relevance:** CEFR label + class probabilities (top-2 shown), with low-confidence flag and simple learning tips.
* **Heuristics & training data:** Pipeline trained only on engineered features; `OneHotEncoder(handle_unknown="ignore")` for categoricals; class weighting explored.

**Saved artefacts (used by the app):**

* Final pipeline: `models/final_logistic_regression_pipeline.pkl`
* (If needed) Preprocessor: `models/preprocessing_pipeline.pkl`
* Reports: `reports/test_metrics_tuned.csv`, `reports/cv_summary.csv`


---

## Results Summary

**Cross-validation (5-fold, out-of-fold on train):**

| Model               | CV Accuracy | CV Macro-F1 | CV Weighted-F1 |
| ------------------- | ----------: | ----------: | -------------: |
| Logistic Regression |   **0.772** |   **0.710** |      **0.778** |
| Random Forest       |       0.772 |       0.710 |          0.779 |
| XGBoost             |       0.750 |       0.682 |          0.752 |

**Test set (tuned models):**

| Model               |  Accuracy |  Macro-F1 | Weighted-F1 |
| ------------------- | --------: | --------: | ----------: |
| Logistic Regression | **0.771** |     0.697 |   **0.775** |
| Random Forest       |     0.731 |     0.652 |       0.745 |
| XGBoost             |     0.761 | **0.700** |       0.760 |

**Decision rule & selection:** We select by **Macro-F1**, tie-break on Accuracy; when within **±0.01**, we favour **the simpler, more interpretable model**.
**Final model:** **Logistic Regression** (balanced performance, interpretability, stable CV).
**Target assessment:** **Accuracy met** (0.771 ≥ 0.75). **Macro-F1 is borderline** (0.697 ≈ 0.70). We proceed with LR as the **baseline production model**, while tracking improvements (see Limitations).

---

## Limitations (and mitigations)

* **Upper-end performance (C1/C2) weaker** than mid-levels — likely due to **class imbalance**, label granularity, and feature ceiling effects.
  **Mitigations:** collect more C1/C2 data; try **ordinal / hierarchical** classifiers; cost-sensitive tuning; richer advanced-level features; keep **top-2 probs + low-confidence flags** in the UI.

---
## Dashboard Design (Streamlit App UI)

### Page 1 — Quick Project Summary

* Project overview and terms (CEFR, leakage, Macro-F1)
* Dataset description and schema (cleaned, engineered features)
* **Business Requirements (1–2)** and success targets

### Page 2 — Data Explorer (BR1)

* Data inspector (shape, head, dtype summary)
* **Class balance** plot + commentary
* **Leakage check** narrative (why raw totals are excluded)
* Correlation/PPS highlights that informed feature engineering

### Page 3 — Model Evaluation (BR2)

* Load `reports/test_metrics_tuned.csv` and `reports/cv_summary.csv`
* Bar chart + table with **Accuracy / Macro-F1 / Weighted-F1**
* **Decision rule** and **final model** statement
* Short interpretation per plot (meets LO6.2)

### Page 4 — Predict CEFR (BR2)

* Input engineered features via widgets (with clear hints)
* Predict CEFR label + class probabilities; top-2 and **low-confidence flag**
* Simple personalised recommendations based on profile imbalance

### Page 5 — Hypotheses & Validation

* H1 leakage → confirmed; response: engineered features only
* H2 imbalance → confirmed; response: macro-F1 optimisation + CV
* Links to the corresponding notebook sections

### Page 6 — About

* CRISP-DM trace (BU → EDA/FE → Modelling → Deploy)
* Limitations & planned improvements
* Repo structure: `app_pages/` (UI), `src/` (logic), `models/`, `reports/`, `data/`

---

## How to Run Locally

```bash
# 1) Create & activate a venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) From the project root (where app.py lives)
streamlit run app.py
```

---

## Deployment (Heroku)

1. Ensure these files exist: `Procfile`, `setup.sh`, `runtime.txt`, `requirements.txt`.
2. Push to GitHub. In Heroku → **New App** → **Connect to GitHub** → select repo → **Deploy Branch**.
3. Open the public URL. If the build fails, check the logs and verify package versions (esp. `xgboost`, `scikit-learn`, `pandas`, `streamlit`).

---

## Main Data Analysis and Machine Learning Libraries

* **pandas** — data loading & wrangling
* **numpy** — vectorised math for feature engineering
* **scikit-learn** — preprocessing, **LogisticRegression**, **RandomForestClassifier**, metrics, `GridSearchCV`, `Pipeline`
* **xgboost** — `XGBClassifier`
* **seaborn** / **matplotlib** — visualisation
* **streamlit** — multipage dashboard & inference UI
* **joblib** — save/load pipelines & preprocessors
* **pathlib** — robust filesystem paths

---

## Credits

### Content

* **Author & dataset:** All data collection, feature engineering, modelling, and narrative were created by **the project author (Ilyas Hussein)**.
* **Walkthrough inspiration & structure:** Code Institute Walkthrough Projects (dashboard architecture, CRISP-DM flow, deployment guidance).
* **Community examples (dashboard patterns & repo structure):**

  * [Blignaut24 / About-BulldozerPriceGenius\_BPG\_v2](https://github.com/Blignaut24/About-BulldozerPriceGenius-_BPG-_v2)
  * [smtilson / pp5-ml-dashboard](https://github.com/smtilson/pp5-ml-dashboard/tree/main)

### Official docs consulted

* **scikit-learn – Logistic Regression:**
  [https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* **scikit-learn – Random Forest Classifier:**
  [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* **scikit-learn – Pipelines, GridSearchCV, metrics:**
  [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)
  [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  [https://scikit-learn.org/stable/modules/model\_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
* **XGBoost – XGBClassifier:**
  [https://xgboost.readthedocs.io/en/stable/python/python\_api.html#xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
* **Streamlit:**
  [https://docs.streamlit.io/](https://docs.streamlit.io/)

> No external media were used. Any adapted snippets are acknowledged above; remaining code and editorial content are original.

### Other

* **Youtube tutorials** - videos on youtube were also used for inspiration at different stages of the project.
* **Stack overflow** - at certain stages stages of the project stack overflow was used to resolve issues.