# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Template Instructions

Welcome,

This is the Code Institute student template for the bring your own data project option in Predictive Analytics. We have preinstalled all of the tools you need to get started. It's perfectly okay to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Template Instructions section of this README.md file and modify the remaining paragraphs for your own project. Please do read the Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

## How to use this repo

1. Use this template to create your GitHub project repo

1. In your newly created repo click on the green Code button. 

1. Then, from the Codespaces tab, click Create codespace on main.

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Open the jupyter_notebooks directory, and click on the notebook you want to open.

1. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.12.1 as it inherits from the workspace, so it will be Python-3.12.1 as installed by Codespaces. To confirm this, you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked


You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.

---

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


---

## ML Business Case — Predict CEFR (Classification)


**Saved artefacts (used by the app):**


---

## Results Summary

**Cross-validation (5-fold, out-of-fold on train):**

---

## Limitations (and mitigations)

---

## Dashboard Design (Streamlit App UI)

### Page 1 — Quick Project Summary


### Page 2 — Data Explorer (BR1)



### Page 3 — Model Evaluation (BR2)


### Page 4 — Predict CEFR (BR2)



### Page 5 — Hypotheses & Validation



### Page 6 — About

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


