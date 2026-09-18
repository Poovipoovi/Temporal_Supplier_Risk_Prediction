"""
PHASE 5 - MODEL EXPLAINABILITY
Temporal Supplier Risk Prediction and Prioritization

Purpose
-------
Explain the already-selected Phase 3 model without changing model selection.

Selected model from Phase 3:
    Logistic Regression Standard

Selected high-risk threshold:
    0.40

Training information:
    Month N features -> Month N+1 risk
    Historical training periods: 2022-01 -> 2024-06
    OOT periods: 2024-07 -> 2025-03

Explainability approach
-----------------------
1. Rebuild the exact 47 Phase-3 temporal features.
2. Refit the selected Logistic Regression Standard on historical training data only.
3. Reuse the saved OOT prediction rows from Phase 3.
4. Verify that the refitted model reproduces the saved Phase-3 probabilities.
5. Produce:
      - global coefficient importance
      - feature direction by risk class
      - high-risk contribution summary
      - row-level local explanations
      - supplier-level explanations for the latest supplier snapshot
      - top feature visualisation
      - JSON/text validation reports

IMPORTANT:
    This script does NOT select a new model.
    This script does NOT tune the threshold.
    This script does NOT use actual OOT outcomes for feature selection.
    It is an explainability stage only.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aerospace_prediction_pairs.csv"
)

OOT_PREDICTIONS_FILE = (
    BASE_DIR
    / "outputs"
    / "predictions"
    / "selected_model_predictions.csv"
)

OUTPUT_REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
)

OUTPUT_EXPLAINABILITY_DIR = (
    BASE_DIR
    / "outputs"
    / "explainability"
)

OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

TARGET = "target_risk_N1"
PERIOD_COLUMN = "period_N"
SUPPLIER_COLUMN = "supplier_id"

SELECTED_MODEL = "Logistic Regression Standard"
HIGH_RISK_THRESHOLD = 0.40

CLASS_ORDER = [
    "High",
    "Low",
    "Medium",
]

CLASS_MAP = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
}

BASE_FEATURES = [
    "late_delivery_rate",
    "avg_days_late",
    "short_delivery_rate",
    "route_deviation_flag",
    "transit_delay_rate",
    "quality_event_rate",
    "total_scrap_qty",
    "rework_cost_ratio",
    "supplier_fault_rate",
    "late_rate_trend",
    "quality_rate_trend",
    "avg_days_late_trend",
    "transit_delay_trend",
    "spend_share",
    "critical_component_exposure",
    "is_worsening",
    "has_trend_data",
]

TEMPORAL_FEATURES = (
    [
        f"lag1_{c}"
        for c in [
            "late_delivery_rate",
            "avg_days_late",
            "short_delivery_rate",
            "quality_event_rate",
            "total_scrap_qty",
            "rework_cost_ratio",
            "supplier_fault_rate",
            "spend_share",
            "critical_component_exposure",
        ]
    ]
    + [
        f"roll3_mean_{c}"
        for c in [
            "late_delivery_rate",
            "avg_days_late",
            "quality_event_rate",
            "short_delivery_rate",
            "total_scrap_qty",
            "rework_cost_ratio",
        ]
    ]
    + [
        f"roll3_std_{c}"
        for c in [
            "late_delivery_rate",
            "avg_days_late",
            "quality_event_rate",
            "short_delivery_rate",
            "rework_cost_ratio",
        ]
    ]
    + [
        "late_rate_acceleration",
        "quality_rate_acceleration",
        "delay_acceleration",
        "risk_worsening_streak",
        "current_risk_severity",
        "previous_risk_severity",
        "risk_change",
        "current_high_risk_flag",
        "current_medium_risk_flag",
        "risk_history_available",
    ]
)

FEATURES = BASE_FEATURES + TEMPORAL_FEATURES

EXPECTED_TRAINING_START = "2022-01"
EXPECTED_TRAINING_END = "2024-06"
EXPECTED_OOT_START = "2024-07"
EXPECTED_OOT_END = "2025-03"

EXPECTED_FEATURE_COUNT = 47

FORBIDDEN_EXPLANATION_COLUMNS = {
    TARGET,
    "period_N1",
    "actual_risk_N1",
    "prediction_correct",
    "target_risk_score_N1",
    "target_low_boundary",
    "target_high_boundary",
    "risk_score_N1",
    "risk_category_N1",
}


# ================================================================
# PRINTING
# ================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# FEATURE ENGINEERING
# Exact temporal logic used by Phase 3
# ================================================================

def build_temporal_features(df):
    data = df.copy()

    data[PERIOD_COLUMN] = data[PERIOD_COLUMN].astype(str)

    data["_period_date"] = pd.to_datetime(
        data[PERIOD_COLUMN] + "-01"
    )

    data = (
        data
        .sort_values(
            [SUPPLIER_COLUMN, "_period_date"]
        )
        .reset_index(drop=True)
    )

    grouped = data.groupby(
        SUPPLIER_COLUMN,
        sort=False,
    )

    # N-1 lag features
    lag_sources = [
        "late_delivery_rate",
        "avg_days_late",
        "short_delivery_rate",
        "quality_event_rate",
        "total_scrap_qty",
        "rework_cost_ratio",
        "supplier_fault_rate",
        "spend_share",
        "critical_component_exposure",
    ]

    for column in lag_sources:
        data[f"lag1_{column}"] = (
            grouped[column].shift(1)
        )

    # Rolling means
    rolling_mean_sources = [
        "late_delivery_rate",
        "avg_days_late",
        "quality_event_rate",
        "short_delivery_rate",
        "total_scrap_qty",
        "rework_cost_ratio",
    ]

    for column in rolling_mean_sources:
        data[f"roll3_mean_{column}"] = (
            grouped[column]
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    # Rolling standard deviations
    rolling_std_sources = [
        "late_delivery_rate",
        "avg_days_late",
        "quality_event_rate",
        "short_delivery_rate",
        "rework_cost_ratio",
    ]

    for column in rolling_std_sources:
        data[f"roll3_std_{column}"] = (
            grouped[column]
            .rolling(3, min_periods=2)
            .std(ddof=0)
            .reset_index(
                level=0,
                drop=True,
            )
        )

    # Acceleration
    data["late_rate_acceleration"] = (
        data["late_rate_trend"]
        - grouped["late_rate_trend"].shift(1)
    )

    data["quality_rate_acceleration"] = (
        data["quality_rate_trend"]
        - grouped["quality_rate_trend"].shift(1)
    )

    data["delay_acceleration"] = (
        data["avg_days_late_trend"]
        - grouped["avg_days_late_trend"].shift(1)
    )

    # Worsening streak
    worsening = (
        data["is_worsening"]
        .fillna(0)
        .astype(int)
    )

    streak_values = []
    counters = {}

    for supplier, flag in zip(
        data[SUPPLIER_COLUMN],
        worsening,
    ):
        counters[supplier] = (
            counters.get(supplier, 0) + 1
            if flag
            else 0
        )
        streak_values.append(
            counters[supplier]
        )

    data["risk_worsening_streak"] = streak_values

    # Current risk at Month N
    risk_map = {
        "Low": 0.0,
        "Medium": 0.5,
        "High": 1.0,
    }

    previous_target = (
        grouped[TARGET].shift(1)
    )

    if "period_N1" in data.columns:
        previous_period_n1 = (
            grouped["period_N1"].shift(1)
        )
    else:
        previous_period_n1 = pd.Series(
            index=data.index,
            dtype=object,
        )

    valid_current_risk = (
        previous_period_n1.astype(str)
        .eq(data[PERIOD_COLUMN].astype(str))
    )

    current_risk = (
        previous_target.where(
            valid_current_risk
        )
    )

    data["current_risk_severity"] = (
        current_risk.map(risk_map)
    )

    data["current_high_risk_flag"] = (
        current_risk.eq("High").astype(float)
    )

    data["current_medium_risk_flag"] = (
        current_risk.eq("Medium").astype(float)
    )

    # Previous risk at Month N-1
    previous_two_target = (
        grouped[TARGET].shift(2)
    )

    if "period_N1" in data.columns:
        previous_two_period_n1 = (
            grouped["period_N1"].shift(2)
        )
    else:
        previous_two_period_n1 = pd.Series(
            index=data.index,
            dtype=object,
        )

    previous_period_n = (
        grouped[PERIOD_COLUMN].shift(1)
    )

    valid_previous_risk = (
        previous_two_period_n1.astype(str)
        .eq(previous_period_n.astype(str))
        &
        previous_period_n1.astype(str)
        .eq(data[PERIOD_COLUMN].astype(str))
    )

    previous_risk = (
        previous_two_target.where(
            valid_previous_risk
        )
    )

    data["previous_risk_severity"] = (
        previous_risk.map(risk_map)
    )

    data["risk_change"] = (
        data["current_risk_severity"]
        - data["previous_risk_severity"]
    )

    data["risk_history_available"] = (
        data["current_risk_severity"]
        .notna()
        .astype(float)
    )

    return data.drop(
        columns=["_period_date"]
    )


# ================================================================
# INPUT VALIDATION
# ================================================================

def validate_columns(df, required, label):
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )


def validate_oot_predictions(oot):
    print_header("PHASE 5 INPUT VALIDATION")

    validate_columns(
        oot,
        FEATURES
        + [
            SUPPLIER_COLUMN,
            PERIOD_COLUMN,
            "predicted_risk_N1",
            "probability_high",
            "probability_low",
            "probability_medium",
            "selected_model",
            "high_risk_threshold",
        ],
        "selected_model_predictions.csv",
    )

    if len(oot) == 0:
        raise ValueError(
            "OOT prediction file contains zero rows."
        )

    if oot[FEATURES].isnull().all(axis=None):
        raise ValueError(
            "All explainability features are missing."
        )

    if oot["selected_model"].nunique() != 1:
        raise ValueError(
            "OOT file contains multiple selected models."
        )

    saved_model = str(
        oot["selected_model"].iloc[0]
    )

    if saved_model != SELECTED_MODEL:
        raise ValueError(
            f"Saved selected model is '{saved_model}', "
            f"but Phase 5 expects '{SELECTED_MODEL}'."
        )

    threshold_values = pd.to_numeric(
        oot["high_risk_threshold"],
        errors="coerce",
    ).dropna()

    if len(threshold_values) == 0:
        raise ValueError(
            "No valid high-risk threshold found."
        )

    if not np.isclose(
        threshold_values.iloc[0],
        HIGH_RISK_THRESHOLD,
        atol=1e-8,
    ):
        raise ValueError(
            "Saved threshold does not match the Phase 3 "
            f"selected threshold of {HIGH_RISK_THRESHOLD}."
        )

    forbidden_used = (
        set(FEATURES)
        & FORBIDDEN_EXPLANATION_COLUMNS
    )

    if forbidden_used:
        raise ValueError(
            "Leakage/outcome columns detected inside FEATURES: "
            + str(sorted(forbidden_used))
        )

    print(
        f"OOT rows: {len(oot)}"
    )

    print(
        f"OOT suppliers: "
        f"{oot[SUPPLIER_COLUMN].nunique()}"
    )

    print(
        f"OOT periods: "
        f"{oot[PERIOD_COLUMN].nunique()}"
    )

    print(
        f"Feature count: {len(FEATURES)}"
    )

    print(
        "Selected model: PASSED"
    )

    print(
        "Selected threshold: PASSED"
    )

    print(
        "Explainability feature leakage audit: PASSED"
    )


# ================================================================
# TRAINING DATA PREPARATION
# ================================================================

def prepare_training_data(historical):
    raw = (
        historical[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    medians = (
        raw
        .median(numeric_only=True)
        .fillna(0.0)
    )

    clean = (
        raw
        .fillna(medians)
        .fillna(0.0)
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(clean)

    y_train = (
        historical[TARGET]
        .astype(str)
        .map(CLASS_MAP)
    )

    if y_train.isna().any():
        raise ValueError(
            "Unknown target category found in training data."
        )

    return (
        X_train,
        y_train.astype(int).to_numpy(),
        scaler,
        medians,
    )


# ================================================================
# MODEL
# Exact Phase 3 selected model definition
# ================================================================

def create_selected_model():
    return LogisticRegression(
        max_iter=5000,
        C=1.0,
        class_weight=None,
        random_state=42,
    )


# ================================================================
# EXPLANATION HELPERS
# ================================================================

def class_label(index):
    return CLASS_ORDER[int(index)]


def build_global_coefficient_table(model):
    rows = []

    for class_index, class_name in enumerate(
        CLASS_ORDER
    ):
        coefficients = model.coef_[
            class_index
        ]

        for feature, coefficient in zip(
            FEATURES,
            coefficients,
        ):
            rows.append(
                {
                    "risk_class": class_name,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "absolute_coefficient": float(
                        abs(coefficient)
                    ),
                    "direction": (
                        "increases class score"
                        if coefficient > 0
                        else
                        "decreases class score"
                        if coefficient < 0
                        else
                        "neutral"
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_mean_absolute_importance(model):
    importance = np.mean(
        np.abs(model.coef_),
        axis=0,
    )

    signed_mean = np.mean(
        model.coef_,
        axis=0,
    )

    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_absolute_coefficient": importance,
            "mean_signed_coefficient": signed_mean,
        }
    )

    table["rank"] = (
        table["mean_absolute_coefficient"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    table["overall_direction"] = np.where(
        table["mean_signed_coefficient"] > 0,
        "positive across classes on average",
        np.where(
            table["mean_signed_coefficient"] < 0,
            "negative across classes on average",
            "neutral on average",
        ),
    )

    return (
        table
        .sort_values(
            "mean_absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_high_risk_coefficient_table(model):
    high_coefficients = model.coef_[
        CLASS_MAP["High"]
    ]

    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "high_risk_coefficient": high_coefficients,
        }
    )

    table["absolute_high_risk_coefficient"] = (
        table["high_risk_coefficient"].abs()
    )

    table["direction"] = np.where(
        table["high_risk_coefficient"] > 0,
        "pushes toward High",
        np.where(
            table["high_risk_coefficient"] < 0,
            "pushes away from High",
            "neutral",
        ),
    )

    return (
        table
        .sort_values(
            "absolute_high_risk_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_local_explanations(
    oot,
    X_oot_scaled,
    model,
):
    probabilities = model.predict_proba(
        X_oot_scaled
    )

    base_prediction = np.argmax(
        probabilities,
        axis=1,
    )

    threshold_prediction = base_prediction.copy()

    high_mask = (
        probabilities[
            :,
            CLASS_MAP["High"],
        ]
        >= HIGH_RISK_THRESHOLD
    )

    threshold_prediction[high_mask] = (
        CLASS_MAP["High"]
    )

    rows = []

    for row_position in range(
        len(oot)
    ):
        predicted_class_index = int(
            threshold_prediction[
                row_position
            ]
        )

        coefficients = model.coef_[
            predicted_class_index
        ]

        contributions = (
            X_oot_scaled[
                row_position
            ]
            * coefficients
        )

        order = np.argsort(
            np.abs(contributions)
        )[::-1]

        top_positive = [
            FEATURES[i]
            for i in order
            if contributions[i] > 0
        ][:5]

        top_negative = [
            FEATURES[i]
            for i in order
            if contributions[i] < 0
        ][:5]

        row = {
            "supplier_id": oot.iloc[
                row_position
            ][SUPPLIER_COLUMN],
            "period": oot.iloc[
                row_position
            ][PERIOD_COLUMN],
            "predicted_risk_N1": class_label(
                predicted_class_index
            ),
            "probability_high": float(
                probabilities[
                    row_position,
                    CLASS_MAP["High"],
                ]
            ),
            "probability_low": float(
                probabilities[
                    row_position,
                    CLASS_MAP["Low"],
                ]
            ),
            "probability_medium": float(
                probabilities[
                    row_position,
                    CLASS_MAP["Medium"],
                ]
            ),
            "base_argmax_risk": class_label(
                int(
                    base_prediction[
                        row_position
                    ]
                )
            ),
            "threshold_applied": bool(
                high_mask[row_position]
            ),
            "top_positive_features": " | ".join(
                top_positive
            ),
            "top_negative_features": " | ".join(
                top_negative
            ),
        }

        for rank, feature_index in enumerate(
            order[:10],
            start=1,
        ):
            row[
                f"feature_{rank}"
            ] = FEATURES[feature_index]

            row[
                f"contribution_{rank}"
            ] = float(
                contributions[
                    feature_index
                ]
            )

            row[
                f"scaled_value_{rank}"
            ] = float(
                X_oot_scaled[
                    row_position,
                    feature_index,
                ]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_high_risk_contribution_summary(
    oot,
    X_oot_scaled,
    model,
):
    high_index = CLASS_MAP["High"]

    high_coefficients = model.coef_[
        high_index
    ]

    contributions = (
        X_oot_scaled
        * high_coefficients
    )

    mean_signed = np.mean(
        contributions,
        axis=0,
    )

    mean_absolute = np.mean(
        np.abs(contributions),
        axis=0,
    )

    positive_rate = np.mean(
        contributions > 0,
        axis=0,
    )

    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_high_risk_contribution": mean_signed,
            "mean_absolute_high_risk_contribution": mean_absolute,
            "positive_contribution_rate": positive_rate,
        }
    )

    table["direction"] = np.where(
        table[
            "mean_high_risk_contribution"
        ] > 0,
        "supports High",
        np.where(
            table[
                "mean_high_risk_contribution"
            ] < 0,
            "opposes High",
            "neutral",
        ),
    )

    return (
        table
        .sort_values(
            "mean_absolute_high_risk_contribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ================================================================
# MODEL REPRODUCTION CHECK
# ================================================================

def verify_saved_predictions(
    oot,
    model,
    X_oot_scaled,
):
    print_header(
        "PHASE 5 MODEL REPRODUCTION CHECK"
    )

    probabilities = model.predict_proba(
        X_oot_scaled
    )

    saved_probabilities = (
        oot[
            [
                "probability_high",
                "probability_low",
                "probability_medium",
            ]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy()
    )

    max_probability_difference = float(
        np.max(
            np.abs(
                probabilities
                - saved_probabilities
            )
        )
    )

    print(
        "Maximum probability difference:",
        f"{max_probability_difference:.12f}",
    )

    if max_probability_difference > 1e-7:
        raise ValueError(
            "The Phase 5 refitted model does not reproduce "
            "the saved Phase 3 probabilities within tolerance.\n"
            f"Maximum difference: "
            f"{max_probability_difference:.12f}\n"
            "STOPPING instead of producing potentially incorrect explanations."
        )

    threshold_predictions = np.argmax(
        probabilities,
        axis=1,
    )

    threshold_predictions[
        probabilities[
            :,
            CLASS_MAP["High"],
        ]
        >= HIGH_RISK_THRESHOLD
    ] = CLASS_MAP["High"]

    saved_prediction = (
        oot["predicted_risk_N1"]
        .astype(str)
        .map(CLASS_MAP)
        .to_numpy()
    )

    prediction_match = np.array_equal(
        threshold_predictions,
        saved_prediction,
    )

    print(
        "Saved predicted classes reproduced:",
        prediction_match,
    )

    if not prediction_match:
        raise ValueError(
            "Phase 5 predictions do not match "
            "Phase 3 saved predictions."
        )

    print(
        "✓ Model reproduction check PASSED"
    )

    return probabilities


# ================================================================
# PLOTS
# ================================================================

def save_global_importance_plot(
    importance_table
):
    top = (
        importance_table
        .head(15)
        .sort_values(
            "mean_absolute_coefficient"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["feature"],
        top["mean_absolute_coefficient"],
    )

    plt.xlabel(
        "Mean absolute logistic coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Phase 5 - Global Feature Importance"
    )

    plt.tight_layout()

    path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "global_feature_importance.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_high_risk_contribution_plot(
    contribution_table
):
    top = (
        contribution_table
        .head(15)
        .sort_values(
            "mean_high_risk_contribution"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["feature"],
        top["mean_high_risk_contribution"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Mean contribution to High-risk class score"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Phase 5 - High-Risk Feature Contributions"
    )

    plt.tight_layout()

    path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "high_risk_feature_contributions.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ================================================================
# MAIN
# ================================================================

def run_phase5():
    print_header(
        "PHASE 5: MODEL EXPLAINABILITY"
    )

    print(
        "Selected model:",
        SELECTED_MODEL,
    )

    print(
        "High-risk threshold:",
        HIGH_RISK_THRESHOLD,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Historical input not found:\n{INPUT_FILE}"
        )

    if not OOT_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Phase 3 selected predictions not found:\n"
            f"{OOT_PREDICTIONS_FILE}"
        )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    print_header(
        "LOADING PHASE 3 DATA"
    )

    source = pd.read_csv(
        INPUT_FILE
    )

    oot = pd.read_csv(
        OOT_PREDICTIONS_FILE
    )

    print(
        f"Historical source shape: {source.shape}"
    )

    print(
        f"Saved OOT prediction shape: {oot.shape}"
    )

    validate_oot_predictions(
        oot
    )

    # ------------------------------------------------------------
    # Rebuild exact temporal features
    # ------------------------------------------------------------

    print_header(
        "REBUILDING PHASE 3 TEMPORAL FEATURES"
    )

    engineered = build_temporal_features(
        source
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in engineered.columns
    ]

    if missing_features:
        raise ValueError(
            "Feature reconstruction failed. Missing:\n"
            + "\n".join(
                f" - {feature}"
                for feature in missing_features
            )
        )

    print(
        f"Rebuilt feature count: "
        f"{len(FEATURES)}"
    )

    if len(FEATURES) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} features "
            f"but configuration contains {len(FEATURES)}."
        )

    print(
        "Exact Phase 3 feature schema: PASSED"
    )

    # ------------------------------------------------------------
    # Identify historical training period
    # ------------------------------------------------------------

    periods = sorted(
        engineered[
            PERIOD_COLUMN
        ]
        .astype(str)
        .unique()
    )

    if (
        EXPECTED_TRAINING_START
        not in periods
        or
        EXPECTED_TRAINING_END
        not in periods
    ):
        raise ValueError(
            "Expected Phase 3 historical training boundaries "
            "were not found."
        )

    train_periods = [
        period
        for period in periods
        if period <= EXPECTED_TRAINING_END
    ]

    historical = (
        engineered[
            engineered[
                PERIOD_COLUMN
            ]
            .astype(str)
            .isin(train_periods)
        ]
        .copy()
    )

    print(
        f"Historical training periods: "
        f"{train_periods[0]} -> {train_periods[-1]}"
    )

    print(
        f"Historical training rows: "
        f"{len(historical)}"
    )

    if (
        train_periods[0]
        != EXPECTED_TRAINING_START
        or
        train_periods[-1]
        != EXPECTED_TRAINING_END
    ):
        raise ValueError(
            "Historical training range does not match Phase 3."
        )

    # ------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------

    print_header(
        "TRAINING SELECTED MODEL FOR EXPLANATION"
    )

    (
        X_train,
        y_train,
        scaler,
        medians,
    ) = prepare_training_data(
        historical
    )

    model = create_selected_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        f"Training matrix: {X_train.shape}"
    )

    print(
        "Logistic Regression Standard fitted: PASSED"
    )

    # ------------------------------------------------------------
    # Prepare OOT features
    # ------------------------------------------------------------

    oot_raw = (
        oot[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    oot_clean = (
        oot_raw
        .fillna(medians)
        .fillna(0.0)
    )

    X_oot_scaled = scaler.transform(
        oot_clean
    )

    # ------------------------------------------------------------
    # Reproduce Phase 3 predictions
    # ------------------------------------------------------------

    probabilities = verify_saved_predictions(
        oot,
        model,
        X_oot_scaled,
    )

    # ------------------------------------------------------------
    # Global explainability
    # ------------------------------------------------------------

    print_header(
        "GLOBAL MODEL EXPLAINABILITY"
    )

    coefficient_table = (
        build_global_coefficient_table(
            model
        )
    )

    coefficient_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "logistic_coefficients_by_class.csv"
    )

    coefficient_table.to_csv(
        coefficient_path,
        index=False,
    )

    importance_table = (
        build_mean_absolute_importance(
            model
        )
    )

    importance_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "global_feature_importance.csv"
    )

    importance_table.to_csv(
        importance_path,
        index=False,
    )

    high_coefficient_table = (
        build_high_risk_coefficient_table(
            model
        )
    )

    high_coefficient_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "high_risk_coefficients.csv"
    )

    high_coefficient_table.to_csv(
        high_coefficient_path,
        index=False,
    )

    print(
        "\nTop 15 global features:"
    )

    print(
        importance_table
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nTop 15 High-risk coefficients:"
    )

    print(
        high_coefficient_table
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # OOT contribution analysis
    # ------------------------------------------------------------

    print_header(
        "HIGH-RISK CONTRIBUTION ANALYSIS"
    )

    contribution_table = (
        build_high_risk_contribution_summary(
            oot,
            X_oot_scaled,
            model,
        )
    )

    contribution_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "high_risk_feature_contributions.csv"
    )

    contribution_table.to_csv(
        contribution_path,
        index=False,
    )

    print(
        contribution_table
        .head(15)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Local explanations
    # ------------------------------------------------------------

    print_header(
        "LOCAL PREDICTION EXPLANATIONS"
    )

    local_table = build_local_explanations(
        oot,
        X_oot_scaled,
        model,
    )

    local_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "local_prediction_explanations.csv"
    )

    local_table.to_csv(
        local_path,
        index=False,
    )

    print(
        f"Local explanations created: "
        f"{len(local_table)} rows"
    )

    # ------------------------------------------------------------
    # Latest supplier snapshot
    # ------------------------------------------------------------

    print_header(
        "LATEST SUPPLIER EXPLANATIONS"
    )

    latest_period = (
        oot[PERIOD_COLUMN]
        .astype(str)
        .max()
    )

    latest_mask = (
        oot[PERIOD_COLUMN]
        .astype(str)
        .eq(latest_period)
    )

    latest_local = (
        local_table[
            latest_mask.to_numpy()
        ]
        .copy()
    )

    latest_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "latest_supplier_explanations.csv"
    )

    latest_local.to_csv(
        latest_path,
        index=False,
    )

    print(
        f"Latest OOT period: {latest_period}"
    )

    print(
        f"Latest supplier explanation rows: "
        f"{len(latest_local)}"
    )

    # ------------------------------------------------------------
    # High-risk suppliers
    # ------------------------------------------------------------

    high_risk_latest = (
        latest_local[
            latest_local[
                "predicted_risk_N1"
            ].eq("High")
        ]
        .copy()
    )

    high_latest_path = (
        OUTPUT_EXPLAINABILITY_DIR
        / "latest_high_risk_explanations.csv"
    )

    high_risk_latest.to_csv(
        high_latest_path,
        index=False,
    )

    print(
        f"Latest suppliers predicted High: "
        f"{len(high_risk_latest)}"
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    print_header(
        "CREATING EXPLAINABILITY VISUALS"
    )

    global_plot = save_global_importance_plot(
        importance_table
    )

    high_plot = save_high_risk_contribution_plot(
        contribution_table
    )

    print(
        f"Global importance plot: {global_plot}"
    )

    print(
        f"High-risk contribution plot: {high_plot}"
    )

    # ------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------

    summary = {
        "phase": "Phase 5 - Model Explainability",
        "project": (
            "Temporal Supplier Risk Prediction "
            "and Prioritization"
        ),
        "selected_model": SELECTED_MODEL,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "feature_count": len(FEATURES),
        "training_range": [
            train_periods[0],
            train_periods[-1],
        ],
        "oot_range": [
            str(
                oot[PERIOD_COLUMN]
                .astype(str)
                .min()
            ),
            str(
                oot[PERIOD_COLUMN]
                .astype(str)
                .max()
            ),
        ],
        "training_rows": int(
            len(historical)
        ),
        "oot_rows": int(
            len(oot)
        ),
        "oot_suppliers": int(
            oot[
                SUPPLIER_COLUMN
            ].nunique()
        ),
        "explainability_method": (
            "Multiclass logistic-regression coefficients "
            "and exact per-row standardized-feature "
            "contributions."
        ),
        "model_reproduction_max_probability_difference": float(
            np.max(
                np.abs(
                    probabilities
                    - oot[
                        [
                            "probability_high",
                            "probability_low",
                            "probability_medium",
                        ]
                    ]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .to_numpy()
                )
            )
        ),
        "outcome_variables_used_for_explanation": False,
        "oot_outcomes_used_for_feature_selection": False,
        "threshold_retuned": False,
        "model_selection_changed": False,
        "files": {
            "global_feature_importance": str(
                importance_path
            ),
            "logistic_coefficients_by_class": str(
                coefficient_path
            ),
            "high_risk_coefficients": str(
                high_coefficient_path
            ),
            "high_risk_feature_contributions": str(
                contribution_path
            ),
            "local_prediction_explanations": str(
                local_path
            ),
            "latest_supplier_explanations": str(
                latest_path
            ),
            "latest_high_risk_explanations": str(
                high_latest_path
            ),
            "global_importance_plot": str(
                global_plot
            ),
            "high_risk_contribution_plot": str(
                high_plot
            ),
        },
    }

    summary_path = (
        OUTPUT_REPORT_DIR
        / "phase5_explainability_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------

    validation_lines = [
        "PHASE 5 VALIDATION",
        "",
        "✓ Selected model verified: Logistic Regression Standard",
        "✓ High-risk threshold verified: 0.40",
        f"✓ Feature count verified: {len(FEATURES)}",
        "✓ Exact Phase 3 temporal feature logic rebuilt",
        "✓ Historical training range verified: 2022-01 -> 2024-06",
        "✓ OOT predictions loaded from Phase 3 output",
        "✓ Outcome/future columns excluded from explanation features",
        "✓ Training-only medians used for imputation",
        "✓ Training-only StandardScaler used",
        "✓ Selected model refitted using historical data only",
        "✓ Phase 3 probabilities reproduced within tolerance",
        "✓ Phase 3 predicted classes reproduced",
        "✓ Model selection was not changed",
        "✓ High-risk threshold was not retuned",
        "✓ Global coefficient explanations generated",
        "✓ High-risk contribution explanations generated",
        "✓ Row-level local explanations generated",
        "✓ Latest supplier explanations generated",
        "✓ Explainability plots generated",
        "",
        "PHASE 5 COMPLETE",
    ]

    validation_path = (
        OUTPUT_REPORT_DIR
        / "phase5_validation.txt"
    )

    validation_path.write_text(
        "\n".join(
            validation_lines
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Final console output
    # ------------------------------------------------------------

    print_header(
        "PHASE 5 COMPLETE"
    )

    print(
        "Selected model:",
        SELECTED_MODEL,
    )

    print(
        "Threshold:",
        HIGH_RISK_THRESHOLD,
    )

    print(
        "Global importance:",
        importance_path,
    )

    print(
        "High-risk coefficients:",
        high_coefficient_path,
    )

    print(
        "High-risk contributions:",
        contribution_path,
    )

    print(
        "Local explanations:",
        local_path,
    )

    print(
        "Latest supplier explanations:",
        latest_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Validation:",
        validation_path,
    )

    print(
        "\nNEXT IMPLEMENTATION STAGE:"
    )

    print(
        "PHASE 6 - SUPPLIER ACTION / INTERVENTION RECOMMENDATION"
    )


if __name__ == "__main__":
    run_phase5()
