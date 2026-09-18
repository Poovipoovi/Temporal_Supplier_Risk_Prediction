
# Temporal Supplier Risk Prediction and Prioritization

A temporal machine learning and decision-support system for predicting next-month supplier risk and prioritizing suppliers for procurement attention.

<img width="1522" height="907" alt="Executive_Overview" src="https://github.com/user-attachments/assets/12e7028e-9ff9-483b-9024-df3925cde1ff" />

---

## Overview

Manufacturing organizations depend on suppliers for timely delivery, quality, and operational continuity. Supplier delays, quality problems, scrap, rework, and transportation-related issues can affect production schedules, costs, and business continuity.

Traditional supplier evaluation mainly relies on historical or current performance indicators and identifies problems after they become visible.

This project uses current and historical supplier behaviour to predict whether a supplier is likely to be **Low, Medium, or High risk in the following month**.

The predicted risk is then combined with supplier deterioration and business exposure to prioritize suppliers for procurement attention and support intervention planning.

### Project Flow

**Supplier Data → Preprocessing → Temporal Feature Engineering → Risk Prediction → Risk Explanation → Prioritization → Intervention Recommendation → Power BI Decision Support**

---

## Business Problem

The project addresses the following question:

> Can current and historical supplier performance be used to predict next-month supplier risk and identify suppliers that require proactive attention?

The objective is to support a shift from primarily retrospective supplier monitoring toward a more proactive, data-driven approach.

---

## Objectives

- Predict next-month supplier risk as Low, Medium, or High.
- Transform raw operational records into supplier-month observations.
- Capture supplier behaviour over time using temporal features.
- Prevent future information from entering the prediction process.
- Benchmark multiple machine learning algorithms.
- Evaluate models using time-aware validation.
- Identify risk drivers and supporting evidence.
- Combine predicted risk with deterioration and business exposure.
- Prioritize suppliers for procurement attention.
- Generate intervention recommendations.
- Present the results through an integrated Power BI decision-support dashboard.

---

# Dataset & Project Scale

| Metric | Value |
|---|---:|
| Raw operational records | 280,800 |
| Supplier-month modelling observations | 1,450 |
| Monthly periods | 39 |
| Engineered model features | 47 |
| Training period | 30 months |
| Training observations | 1,172 |
| Out-of-Time testing period | 9 months |
| OOT observations | 278 |

The raw operational records were transformed into supplier-month modelling observations before temporal feature engineering and machine learning.

> The complete raw operational dataset is not included in this repository because of its size.

---

# Methodology

## 1. Data Preprocessing

The raw operational data is prepared through:

- Data validation
- Missing-value handling
- Data consistency checks
- Supplier-level aggregation
- Monthly KPI generation
- Preparation of modelling data

The raw records are transformed into supplier-month observations to represent supplier behaviour over time.

---

## 2. Temporal Feature Engineering

The prediction task is structured as:

```text
Month N Supplier Information
            ↓
    Temporal Features
            ↓
    Month N + 1 Risk
````

The engineered feature set includes:

### Operational Features

* Delivery performance
* Late delivery rate
* Average days late
* Quality event indicators
* Scrap and rework indicators
* Transportation-related indicators

### Temporal Features

* Lag values
* Rolling statistics
* Performance trends
* Deterioration indicators
* Acceleration indicators
* Historical risk behaviour

### Business Exposure Features

* Spend exposure
* Supplier criticality
* Business exposure indicators

All predictive features use information available in Month N or earlier.

The Month N+1 target is excluded from the feature set to prevent temporal leakage.

---

## 3. Target Engineering

Supplier risk is generated using a domain-informed composite risk-scoring methodology based on operational performance indicators.

The resulting categories are:

* Low
* Medium
* High

The target is shifted forward to create a genuine temporal prediction task:

```text
Month N Features → Month N + 1 Risk
```

The target is an engineered operational-risk label rather than a naturally observed ground-truth label.

---

# Machine Learning

The project benchmarks the following approaches:

* Logistic Regression
* Random Forest
* Extra Trees
* HistGradientBoosting
* XGBoost
* Persistence Baseline

The persistence baseline provides a reference point by using the supplier's current risk as a prediction of its next-month risk.

---

# Validation Strategy

Because supplier behaviour is time-dependent, a random train-test split was avoided.

The project uses:

* Chronological train/test separation
* Rolling temporal validation
* True Out-of-Time evaluation
* Temporal ordering audit
* Leakage audit
* Training-only preprocessing

### Final Evaluation Setup

```text
30 Months
     ↓
Training
1,172 observations
     ↓
9 Months
     ↓
True Out-of-Time Test
278 observations
```

The final nine months were kept unseen during model development and used for final Out-of-Time evaluation.

---

# Model Performance

## Final Out-of-Time Results

| Model                | Accuracy | Macro-F1 | High-Risk Recall | ROC-AUC |
| -------------------- | -------: | -------: | ---------------: | ------: |
| HistGradientBoosting |   54.32% |   48.93% |           34.78% |  0.7039 |
| Persistence Baseline |   51.44% |   48.28% |           49.28% |  0.5978 |
| XGBoost              |   51.44% |   48.64% |           44.93% | ~0.6985 |
| Logistic Regression  |   44.24% |   44.51% |           79.71% |  0.6711 |
| Random Forest        |   44.24% |   45.14% |           72.46% |  0.6860 |
| Extra Trees          |   42.09% |   40.55% |           85.51% |  0.7013 |

### Key Results

* HistGradientBoosting achieved **54.32% Out-of-Time accuracy** and **0.7039 ROC-AUC**.
* Extra Trees achieved **85.51% recall for high-risk suppliers**.
* The models demonstrate moderate predictive separation on unseen future periods.
* The persistence baseline provides a strong reference because supplier risk can persist across consecutive months.

The system is therefore positioned as an **early-warning decision-support system**, rather than a system that guarantees future supplier outcomes.

---

# Risk Explanation

Prediction alone does not explain why a supplier is considered risky.

The project therefore includes risk-driver and evidence analysis to connect predicted supplier risk with relevant operational indicators.

The analysis considers factors such as:

* Delivery deterioration
* Delay behaviour
* Quality performance
* Scrap and rework
* Transportation indicators
* Historical risk behaviour
* Business exposure

These outputs provide supporting evidence alongside the predicted risk.

---

# Risk Prioritization

Risk prediction and risk prioritization are treated as separate stages.

### Risk Prediction

> Which suppliers are likely to become risky next month?

### Risk Prioritization

> Which suppliers require greater procurement attention?

The prioritization framework combines:

* Predicted risk
* High-risk probability
* Deterioration
* Operational evidence
* Business exposure
* Supplier criticality

This produces a priority score and priority band.

```text
Predicted Risk
      +
Risk Probability
      +
Deterioration
      +
Business Exposure
      ↓
Priority Score
      ↓
Priority Band
      ↓
Intervention Level
```

---

# Supplier Intervention

The system maps supplier risk and priority information to potential procurement interventions.

Examples include:

* Monitoring
* Corrective-action follow-up
* Supplier review
* Increased monitoring
* Escalation for high-priority cases

These are **decision-support recommendations**, not autonomous procurement decisions.

---

# Power BI Dashboard

The project includes an integrated Power BI decision-support dashboard.

## Executive Overview

The dashboard provides a high-level view of:

* Supplier risk distribution
* Predicted high-risk suppliers
* Priority suppliers
* Average priority score
* Monthly risk trends
* Top suppliers by priority

<img width="1522" height="907" alt="Executive_Overview" src="https://github.com/user-attachments/assets/1a885d00-249a-4290-b931-9f3fa926e909" />

---

## Supplier Risk Analysis

Provides supplier-level analysis including:

* Predicted risk
* High-risk probability
* Priority score
* Primary risk domain
* Supporting evidence
* Deterioration indicators
* Recommended intervention

<img width="1573" height="800" alt="Supplier_Risk_Analysis" src="https://github.com/user-attachments/assets/d30b519a-23ca-40d5-b52b-79966f27dc8c" />

---

## Model Performance

Provides:

* Model comparison
* Accuracy
* Macro-F1
* ROC-AUC
* High-risk performance
* Validation information

<img width="1627" height="912" alt="Risk_Trend_Analysis" src="https://github.com/user-attachments/assets/97756526-8402-478c-a233-29e0949824d8" />

---

## Supplier Intervention & Action

Provides:

* Priority bands
* Risk domains
* Intervention levels
* Recommended actions
* Suppliers requiring corrective attention

<img width="1457" height="862" alt="Intervention_Analysis" src="https://github.com/user-attachments/assets/7b6fcbbb-f6a0-4fa8-92d7-1b1868082684" />

---

# Project Outputs

Generated outputs are organized into the following areas:

```text
outputs/
├── dashboard/
├── reports/
├── interventions/
├── explainability/
├── predictions/
└── prioritization/
```

These outputs support:

* Model predictions
* Risk explanations
* Supplier prioritization
* Intervention analysis
* Dashboard preparation
* Analytical reporting

---

# Repository Structure

```text
Supplier_analysis_project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_processing.py
│   ├── data_audit.py
│   ├── data_audit_temporal.py
│   ├── build_prediction_pairs.py
│   ├── modeling_final.py
│   ├── exposure_prioritization.py
│   ├── phase4_decision_support.py
│   ├── phase5_model_explainability.py
│   ├── phase6_supplier_intervention_explainability.py
│   ├── phase6_supplier_intervention_recommendation.py
│   └── phase7_dashboard_preparation.py
│
├── outputs/
│   ├── dashboard/
│   ├── reports/
│   ├── interventions/
│   ├── explainability/
│   ├── predictions/
│   └── prioritization/
│
└── dashboard/
    └── screenshots/
        ├── executive_overview.png
        ├── supplier_risk_analysis.png
        ├── model_performance.png
        └── intervention_action.png
```

---

# Technologies

### Programming & Data Analysis

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Business Intelligence

* Power BI
* DAX
* Power Query
* Excel

### Development

* Python
* Git
* GitHub

---

# Limitations

* The risk target is domain-engineered rather than directly observed.
* Supplier behaviour can change over time.
* The dataset contains a limited number of monthly periods.
* Temporal distribution shift can affect future performance.
* Model predictions should not be interpreted as certainty.
* The dashboard is a project-level decision-support implementation and not a live enterprise procurement system.

---

# Project Status

**Completed**

The project includes:

* Data preprocessing
* Supplier-month aggregation
* Temporal feature engineering
* Target engineering
* Temporal leakage auditing
* Machine learning benchmarking
* Temporal validation
* True Out-of-Time evaluation
* Risk-driver analysis
* Supplier prioritization
* Intervention recommendations
* Power BI decision-support dashboard

---

# Author

**Poovizhi A**

B.E. Computer Science and Engineering
Adhiyamaan College of Engineering

