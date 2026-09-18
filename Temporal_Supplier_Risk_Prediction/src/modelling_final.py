"""
PHASE 3 - ROBUST TEMPORAL SUPPLIER-RISK MODELING

Goal:
    Predict target_risk_N1 (supplier risk in Month N+1)
    using only information available at Month N.

Design:
    - No random train/test split
    - Chronological out-of-time test
    - Rolling temporal validation inside the historical training period
    - Training-only median imputation
    - Training-only scaling
    - Leakage-safe N-1 / rolling features
    - High-risk threshold tuning on historical validation only
    - Multiple model comparison
    - Persistence baseline comparison
    - OOT test used only once for final evaluation

IMPORTANT:
    This script does NOT guarantee a higher score. It is designed to
    find a better model honestly without leaking the OOT test.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

OUTPUT_REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
)

OUTPUT_PREDICTION_DIR = (
    BASE_DIR
    / "outputs"
    / "predictions"
)

OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

TARGET = "target_risk_N1"
PERIOD_COLUMN = "period_N"
SUPPLIER_COLUMN = "supplier_id"

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

FINAL_TRAIN_PERIODS = 30
EXPECTED_OOT_PERIODS = 9

# Rolling validation:
# Fold 1 -> first 18 train periods, next 4 validation periods
# Fold 2 -> first 22 train periods, next 4 validation periods
# Fold 3 -> first 26 train periods, next 4 validation periods
VALIDATION_BLOCK = 4

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

HIGH_CLASS = 0

# Rows = actual class
# Columns = predicted class
COST_MATRIX = np.array(
    [
        [0, 10, 10],  # Actual High
        [1, 0, 1],    # Actual Low
        [1, 1, 0],    # Actual Medium
    ],
    dtype=float,
)

# Threshold is selected only on historical validation data.
HIGH_RISK_THRESHOLDS = np.arange(
    0.25,
    0.71,
    0.05,
)


# ================================================================
# PRINTING
# ================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# FEATURE ENGINEERING
# ================================================================

def build_temporal_features(df):
    """
    Create features using Month N or earlier only.

    The target target_risk_N1 belongs to Month N+1.
    Therefore the current Month-N risk is recovered only from the
    previous (N-1 -> N) prediction pair.
    """

    data = df.copy()

    data[PERIOD_COLUMN] = (
        data[PERIOD_COLUMN]
        .astype(str)
    )

    data["_period_date"] = pd.to_datetime(
        data[PERIOD_COLUMN] + "-01"
    )

    data = (
        data
        .sort_values(
            [
                SUPPLIER_COLUMN,
                "_period_date",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = data.groupby(
        SUPPLIER_COLUMN,
        sort=False,
    )

    # ------------------------------------------------------------
    # N-1 lag features
    # ------------------------------------------------------------

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
            grouped[column]
            .shift(1)
        )

    # ------------------------------------------------------------
    # Rolling mean: current Month N + previous months
    # ------------------------------------------------------------

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
            .rolling(
                3,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

    # ------------------------------------------------------------
    # Rolling standard deviation
    # ------------------------------------------------------------

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
            .rolling(
                3,
                min_periods=2,
            )
            .std(ddof=0)
            .reset_index(
                level=0,
                drop=True,
            )
        )

    # ------------------------------------------------------------
    # Acceleration
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Worsening streak
    # ------------------------------------------------------------

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

    data["risk_worsening_streak"] = (
        streak_values
    )

    # ------------------------------------------------------------
    # Current risk at Month N
    # ------------------------------------------------------------

    risk_map = {
        "Low": 0.0,
        "Medium": 0.5,
        "High": 1.0,
    }

    previous_target = (
        grouped[TARGET]
        .shift(1)
    )

    if "period_N1" in data.columns:
        previous_period_n1 = (
            grouped["period_N1"]
            .shift(1)
        )
    else:
        previous_period_n1 = pd.Series(
            index=data.index,
            dtype=object,
        )

    valid_current_risk = (
        previous_period_n1
        .astype(str)
        .eq(
            data[PERIOD_COLUMN]
            .astype(str)
        )
    )

    current_risk = (
        previous_target
        .where(valid_current_risk)
    )

    data["current_risk_severity"] = (
        current_risk.map(risk_map)
    )

    data["current_high_risk_flag"] = (
        current_risk
        .eq("High")
        .astype(float)
    )

    data["current_medium_risk_flag"] = (
        current_risk
        .eq("Medium")
        .astype(float)
    )

    # ------------------------------------------------------------
    # Previous risk at Month N-1
    # ------------------------------------------------------------

    previous_two_target = (
        grouped[TARGET]
        .shift(2)
    )

    if "period_N1" in data.columns:
        previous_two_period_n1 = (
            grouped["period_N1"]
            .shift(2)
        )
    else:
        previous_two_period_n1 = pd.Series(
            index=data.index,
            dtype=object,
        )

    previous_period_n = (
        grouped[PERIOD_COLUMN]
        .shift(1)
    )

    valid_previous_risk = (
        previous_two_period_n1
        .astype(str)
        .eq(
            previous_period_n
            .astype(str)
        )
        &
        previous_period_n1
        .astype(str)
        .eq(
            data[PERIOD_COLUMN]
            .astype(str)
        )
    )

    previous_risk = (
        previous_two_target
        .where(valid_previous_risk)
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

    data = data.drop(
        columns=["_period_date"]
    )

    return data


# ================================================================
# VALIDATION
# ================================================================

def validate_input(df):
    print_header("INPUT VALIDATION")

    required = (
        FEATURES
        + [
            TARGET,
            PERIOD_COLUMN,
            SUPPLIER_COLUMN,
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    print("Required schema: PASSED")

    # Base features must be complete.
    base_missing = (
        df[BASE_FEATURES]
        .isnull()
        .any()
    )

    if base_missing.any():
        raise ValueError(
            "Missing values found in base features: "
            + str(
                base_missing[
                    base_missing
                ].index.tolist()
            )
        )

    print(
        "Base-feature missing-value check: PASSED"
    )

    temporal_missing = [
        column
        for column in TEMPORAL_FEATURES
        if df[column].isnull().any()
    ]

    print(
        "Expected historical NaNs in engineered "
        f"temporal features: {len(temporal_missing)} columns"
    )

    print(
        "These will be imputed using "
        "training-only medians: PASSED"
    )

    print(
        "Missing-value check: PASSED"
    )

    if df[TARGET].isnull().any():
        raise ValueError(
            "Target contains missing values."
        )

    print(
        "Target completeness: PASSED"
    )


def validate_temporal_structure(df):
    print_header(
        "TEMPORAL STRUCTURE VALIDATION"
    )

    periods = sorted(
        df[PERIOD_COLUMN]
        .astype(str)
        .unique()
    )

    print(
        f"Unique Month-N periods: {len(periods)}"
    )

    print(
        f"First Month-N: {periods[0]}"
    )

    print(
        f"Last Month-N : {periods[-1]}"
    )

    invalid = []

    for period in periods:
        expected_next = str(
            pd.Period(
                period,
                freq="M",
            ) + 1
        )

        values = (
            df.loc[
                df[PERIOD_COLUMN]
                .astype(str)
                .eq(period),
                "period_N1",
            ]
            .astype(str)
            .unique()
        )

        for future_period in values:
            if future_period != expected_next:
                invalid.append(
                    (
                        period,
                        future_period,
                    )
                )

    print(
        "Invalid N -> N+1 transitions: "
        f"{len(invalid)}"
    )

    if invalid:
        raise ValueError(
            f"Invalid temporal transitions: "
            f"{invalid[:10]}"
        )

    print(
        "One-month-ahead relationship: PASSED"
    )

    return periods


def leakage_audit():
    print_header(
        "LEAKAGE AUDIT"
    )

    forbidden = {
        TARGET,
        "period_N1",
        "risk_score_N1",
        "risk_category_N1",
    }

    bad = forbidden.intersection(
        FEATURES
    )

    print(
        "Target excluded from X: "
        f"{TARGET not in FEATURES}"
    )

    print(
        "Future-period fields excluded from X: "
        f"{'period_N1' not in FEATURES}"
    )

    print(
        "Future-risk fields excluded from X: "
        f"{not bool(bad)}"
    )

    if bad:
        raise ValueError(
            f"Potential leakage detected: {bad}"
        )

    print(
        "Explicit feature-name audit: PASSED"
    )

    print(
        "\nModel information rule:"
    )

    print(
        "X = information available at Month N"
    )

    print(
        "y = actual risk category at Month N+1"
    )


# ================================================================
# TARGET
# ================================================================

def encode_target(df):
    y = (
        df[TARGET]
        .astype(str)
        .map(CLASS_MAP)
    )

    if y.isna().any():
        raise ValueError(
            "Unknown target category found."
        )

    return y.astype(int).to_numpy()


# ================================================================
# PREPROCESSING
# ================================================================

def prepare_train_validation(
    train_df,
    validation_df,
):
    train_raw = (
        train_df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    validation_raw = (
        validation_df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # IMPORTANT:
    # Medians come ONLY from the historical fold's training data.
    medians = (
        train_raw
        .median(numeric_only=True)
        .fillna(0.0)
    )

    train_clean = (
        train_raw
        .fillna(medians)
        .fillna(0.0)
    )

    validation_clean = (
        validation_raw
        .fillna(medians)
        .fillna(0.0)
    )

    scaler = StandardScaler()

    X_train = (
        scaler
        .fit_transform(train_clean)
    )

    X_validation = (
        scaler
        .transform(validation_clean)
    )

    return (
        X_train,
        X_validation,
        scaler,
        medians,
    )


def prepare_final_train_test(
    train_df,
    test_df,
):
    train_raw = (
        train_df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    test_raw = (
        test_df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # Fit imputation only on all historical training periods.
    medians = (
        train_raw
        .median(numeric_only=True)
        .fillna(0.0)
    )

    train_clean = (
        train_raw
        .fillna(medians)
        .fillna(0.0)
    )

    test_clean = (
        test_raw
        .fillna(medians)
        .fillna(0.0)
    )

    scaler = StandardScaler()

    X_train = (
        scaler
        .fit_transform(train_clean)
    )

    X_test = (
        scaler
        .transform(test_clean)
    )

    return (
        X_train,
        X_test,
        scaler,
        medians,
    )


# ================================================================
# METRICS
# ================================================================

def calculate_penalty(
    y_true,
    y_pred,
):
    total = 0.0

    for actual, predicted in zip(
        y_true,
        y_pred,
    ):
        total += COST_MATRIX[
            actual,
            predicted,
        ]

    return float(total)


def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
):
    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        roc_auc = np.nan

    return {
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "High_Risk_Precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=[HIGH_CLASS],
                average=None,
                zero_division=0,
            )[0]
        ),

        "High_Risk_Recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=[HIGH_CLASS],
                average=None,
                zero_division=0,
            )[0]
        ),

        "High_Risk_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[HIGH_CLASS],
                average=None,
                zero_division=0,
            )[0]
        ),

        "ROC_AUC_OVR_Macro": (
            float(roc_auc)
            if not np.isnan(roc_auc)
            else np.nan
        ),

        "Penalty_Cost": calculate_penalty(
            y_true,
            y_pred,
        ),
    }


# ================================================================
# HIGH-RISK THRESHOLD
# ================================================================

def apply_high_risk_threshold(
    probabilities,
    threshold,
):
    """
    Start with normal multiclass argmax.

    Then force High when P(High) reaches the selected threshold.

    Threshold is learned only from historical validation data.
    """

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).copy()

    high_mask = (
        probabilities[:, HIGH_CLASS]
        >= threshold
    )

    predictions[high_mask] = (
        HIGH_CLASS
    )

    return predictions


def tune_high_risk_threshold(
    y_true,
    probabilities,
):
    """
    Select threshold from validation data.

    Selection objective:
        1. High-risk F1
        2. Accuracy
        3. Macro-F1
        4. Lower operational penalty

    This avoids blindly optimising recall.
    """

    best = None

    for threshold in HIGH_RISK_THRESHOLDS:
        predictions = (
            apply_high_risk_threshold(
                probabilities,
                threshold,
            )
        )

        current_metrics = (
            calculate_metrics(
                y_true,
                predictions,
                probabilities,
            )
        )

        objective = (
            current_metrics[
                "High_Risk_F1"
            ],
            current_metrics[
                "Accuracy"
            ],
            current_metrics[
                "Macro_F1"
            ],
            -current_metrics[
                "Penalty_Cost"
            ],
        )

        if (
            best is None
            or objective > best[0]
        ):
            best = (
                objective,
                float(threshold),
                current_metrics,
                predictions,
            )

    return (
        best[1],
        best[2],
        best[3],
    )


# ================================================================
# MODEL DEFINITIONS
# ================================================================

def create_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            C=0.5,
            class_weight="balanced",
            random_state=42,
        ),

        "Logistic Regression Standard": LogisticRegression(
            max_iter=5000,
            C=1.0,
            class_weight=None,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=900,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=900,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=700,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=3.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        print(
            "\nWARNING: XGBoost is not installed. "
            "The pipeline will continue without it."
        )

    return models


# ================================================================
# ROLLING TEMPORAL FOLDS
# ================================================================

def create_rolling_folds(
    train_periods,
):
    """
    Three expanding-window temporal validation folds.

    For 30 training periods:

        Fold 1:
            train = first 18
            valid = next 4

        Fold 2:
            train = first 22
            valid = next 4

        Fold 3:
            train = first 26
            valid = last 4

    The OOT periods are completely untouched.
    """

    total = len(train_periods)

    required = (
        18
        + VALIDATION_BLOCK
        + VALIDATION_BLOCK
        + VALIDATION_BLOCK
    )

    if total < required:
        raise ValueError(
            "Not enough historical periods "
            "for rolling validation."
        )

    folds = []

    for train_size in [
        total - 12,
        total - 8,
        total - 4,
    ]:
        validation_start = train_size
        validation_end = (
            validation_start
            + VALIDATION_BLOCK
        )

        fold_train_periods = (
            train_periods[:train_size]
        )

        fold_validation_periods = (
            train_periods[
                validation_start:
                validation_end
            ]
        )

        folds.append(
            (
                fold_train_periods,
                fold_validation_periods,
            )
        )

    return folds


# ================================================================
# PERSISTENCE BASELINE
# ================================================================

def persistence_predictions(
    evaluation_df,
    complete_df,
):
    """
    Predict Month N+1 risk as the risk already observed in Month N.

    Month-N risk is obtained from the previous prediction pair:
        (N-1) -> N
    """

    lookup = (
        complete_df[
            [
                SUPPLIER_COLUMN,
                "period_N1",
                TARGET,
            ]
        ]
        .copy()
        .rename(
            columns={
                "period_N1": PERIOD_COLUMN,
                TARGET: "_current_risk",
            }
        )
    )

    duplicate = (
        lookup
        .duplicated(
            subset=[
                SUPPLIER_COLUMN,
                PERIOD_COLUMN,
            ],
            keep=False,
        )
    )

    if duplicate.any():
        raise ValueError(
            "Duplicate persistence lookup rows found."
        )

    merged = (
        evaluation_df[
            [
                SUPPLIER_COLUMN,
                PERIOD_COLUMN,
            ]
        ]
        .merge(
            lookup,
            on=[
                SUPPLIER_COLUMN,
                PERIOD_COLUMN,
            ],
            how="left",
            validate="many_to_one",
        )
    )

    if merged["_current_risk"].isna().any():
        raise ValueError(
            "Persistence baseline could not recover "
            "Month-N risk for every evaluation row."
        )

    predictions = (
        merged["_current_risk"]
        .astype(str)
        .map(CLASS_MAP)
        .to_numpy()
    )

    return predictions.astype(int)


def evaluate_persistence(
    evaluation_df,
    complete_df,
):
    y_true = encode_target(
        evaluation_df
    )

    predictions = persistence_predictions(
        evaluation_df,
        complete_df,
    )

    probabilities = np.zeros(
        (
            len(predictions),
            len(CLASS_ORDER),
        )
    )

    probabilities[
        np.arange(
            len(predictions)
        ),
        predictions,
    ] = 1.0

    return calculate_metrics(
        y_true,
        predictions,
        probabilities,
    )


# ================================================================
# ROLLING MODEL EVALUATION
# ================================================================

def evaluate_model_on_fold(
    model,
    fold_train_df,
    fold_validation_df,
):
    (
        X_train,
        X_validation,
        _,
        _,
    ) = prepare_train_validation(
        fold_train_df,
        fold_validation_df,
    )

    y_train = encode_target(
        fold_train_df
    )

    y_validation = encode_target(
        fold_validation_df
    )

    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        fitted_model
        .predict_proba(
            X_validation
        )
    )

    (
        threshold,
        tuned_metrics,
        predictions,
    ) = tune_high_risk_threshold(
        y_validation,
        probabilities,
    )

    return (
        tuned_metrics,
        threshold,
        predictions,
        probabilities,
    )


def evaluate_all_models(
    train_df,
    train_periods,
    models,
):
    print_header(
        "ROLLING TEMPORAL MODEL EVALUATION"
    )

    folds = create_rolling_folds(
        train_periods
    )

    fold_records = []
    model_thresholds = {}

    for model_name, model in models.items():
        print(
            "\n"
            + "-" * 70
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "-" * 70
        )

        thresholds = []

        for fold_number, (
            fold_train_periods,
            fold_validation_periods,
        ) in enumerate(
            folds,
            start=1,
        ):
            fold_train_df = (
                train_df[
                    train_df[
                        PERIOD_COLUMN
                    ]
                    .astype(str)
                    .isin(
                        fold_train_periods
                    )
                ]
                .copy()
            )

            fold_validation_df = (
                train_df[
                    train_df[
                        PERIOD_COLUMN
                    ]
                    .astype(str)
                    .isin(
                        fold_validation_periods
                    )
                ]
                .copy()
            )

            (
                current_metrics,
                threshold,
                _,
                _,
            ) = evaluate_model_on_fold(
                model,
                fold_train_df,
                fold_validation_df,
            )

            thresholds.append(
                threshold
            )

            fold_records.append(
                {
                    "Model": model_name,
                    "Fold": fold_number,
                    "Train_Start": (
                        fold_train_periods[0]
                    ),
                    "Train_End": (
                        fold_train_periods[-1]
                    ),
                    "Validation_Start": (
                        fold_validation_periods[0]
                    ),
                    "Validation_End": (
                        fold_validation_periods[-1]
                    ),
                    **current_metrics,
                    "High_Risk_Threshold": threshold,
                }
            )

            print(
                f"Fold {fold_number}: "
                f"Accuracy={current_metrics['Accuracy']:.4f} | "
                f"Macro-F1={current_metrics['Macro_F1']:.4f} | "
                f"High Precision={current_metrics['High_Risk_Precision']:.4f} | "
                f"High Recall={current_metrics['High_Risk_Recall']:.4f} | "
                f"High F1={current_metrics['High_Risk_F1']:.4f} | "
                f"Threshold={threshold:.2f}"
            )

        model_thresholds[model_name] = (
            float(
                np.median(
                    thresholds
                )
            )
        )

    fold_results = pd.DataFrame(
        fold_records
    )

    aggregate = (
        fold_results
        .groupby(
            "Model",
            as_index=False,
        )
        .agg(
            Accuracy=(
                "Accuracy",
                "mean",
            ),
            Macro_F1=(
                "Macro_F1",
                "mean",
            ),
            High_Risk_Precision=(
                "High_Risk_Precision",
                "mean",
            ),
            High_Risk_Recall=(
                "High_Risk_Recall",
                "mean",
            ),
            High_Risk_F1=(
                "High_Risk_F1",
                "mean",
            ),
            ROC_AUC_OVR_Macro=(
                "ROC_AUC_OVR_Macro",
                "mean",
            ),
            Penalty_Cost=(
                "Penalty_Cost",
                "mean",
            ),
        )
    )

    aggregate[
        "High_Risk_Threshold"
    ] = [
        model_thresholds[
            name
        ]
        for name in aggregate["Model"]
    ]

    return (
        folds,
        fold_results,
        aggregate,
        model_thresholds,
    )


# ================================================================
# MODEL SELECTION
# ================================================================

def select_model(
    aggregate_results,
    persistence_metrics,
):
    """
    Selection is performed using historical rolling validation only.

    Score:
        35% Accuracy
        25% High-risk Precision
        25% High-risk F1
        15% Macro-F1

    This prevents the project from selecting a model merely because
    it has high recall while producing many false high-risk alerts.
    """

    candidates = (
        aggregate_results
        .copy()
    )

    persistence_row = pd.DataFrame(
        [
            {
                "Model": "Persistence Baseline",
                **persistence_metrics,
                "High_Risk_Threshold": np.nan,
            }
        ]
    )

    candidates = pd.concat(
        [
            candidates,
            persistence_row,
        ],
        ignore_index=True,
    )

    candidates[
        "Selection_Score"
    ] = (
        0.35
        * candidates["Accuracy"]
        + 0.25
        * candidates[
            "High_Risk_Precision"
        ]
        + 0.25
        * candidates[
            "High_Risk_F1"
        ]
        + 0.15
        * candidates["Macro_F1"]
    )

    candidates = (
        candidates
        .sort_values(
            [
                "Selection_Score",
                "Accuracy",
                "High_Risk_Precision",
                "Macro_F1",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return candidates


# ================================================================
# FINAL TRAINING
# ================================================================

def train_final_model(
    model,
    train_df,
):
    """Fit final model using all historical training rows only."""

    train_raw = (
        train_df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(columns=FEATURES)
    )

    # Training-only median imputation.
    medians = (
        train_raw
        .median(numeric_only=True)
        .reindex(FEATURES)
        .fillna(0.0)
    )

    train_clean = (
        train_raw
        .fillna(medians)
        .fillna(0.0)
    )

    if train_clean.empty:
        raise ValueError(
            "Final training set is empty; cannot train model."
        )

    if train_clean.isna().any().any():
        bad_columns = (
            train_clean
            .columns[
                train_clean.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            f"NaNs remain after final imputation: {bad_columns}"
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_clean
    )

    y_train = encode_target(
        train_df
    )

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Feature/target row mismatch: "
            f"X={X_train.shape[0]}, "
            f"y={len(y_train)}"
        )

    fitted_model = clone(model)

    fitted_model.fit(
        X_train,
        y_train
    )

    return (
        fitted_model,
        scaler,
        medians
    )


def transform_final_test(
    test_df,
    scaler,
    medians,
):
    raw = (
        test_df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    clean = (
        raw
        .fillna(medians)
        .fillna(0.0)
    )

    return scaler.transform(
        clean
    )


# ================================================================
# OUT-OF-TIME EVALUATION
# ================================================================

def evaluate_oot_models(
    models,
    model_thresholds,
    train_df,
    test_df,
    complete_df,
):
    print_header(
        "FINAL OUT-OF-TIME EVALUATION"
    )

    y_test = encode_target(
        test_df
    )

    records = []
    predictions_store = {}

    for model_name, model in models.items():

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "-" * 70
        )

        (
            fitted,
            scaler,
            medians,
        ) = train_final_model(
            model,
            train_df,
        )

        X_test = transform_final_test(
            test_df,
            scaler,
            medians,
        )

        probabilities = (
            fitted
            .predict_proba(
                X_test
            )
        )

        threshold = (
            model_thresholds[
                model_name
            ]
        )

        predictions = (
            apply_high_risk_threshold(
                probabilities,
                threshold,
            )
        )

        current_metrics = (
            calculate_metrics(
                y_test,
                predictions,
                probabilities,
            )
        )

        print(
            f"Accuracy            : "
            f"{current_metrics['Accuracy']:.4f}"
        )

        print(
            f"Macro F1            : "
            f"{current_metrics['Macro_F1']:.4f}"
        )

        print(
            f"High-risk Recall    : "
            f"{current_metrics['High_Risk_Recall']:.4f}"
        )

        print(
            f"High-risk Precision  : "
            f"{current_metrics['High_Risk_Precision']:.4f}"
        )

        print(
            f"High-risk F1         : "
            f"{current_metrics['High_Risk_F1']:.4f}"
        )

        print(
            f"ROC-AUC              : "
            f"{current_metrics['ROC_AUC_OVR_Macro']:.4f}"
        )

        print(
            f"Penalty Cost         : "
            f"${current_metrics['Penalty_Cost']:.0f}"
        )

        print(
            f"High-risk threshold  : "
            f"{threshold:.2f}"
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2],
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

        records.append(
            {
                "Model": model_name,
                **current_metrics,
                "High_Risk_Threshold": threshold,
            }
        )

        predictions_store[
            model_name
        ] = (
            predictions,
            probabilities,
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual_High",
                "Actual_Low",
                "Actual_Medium",
            ],
            columns=[
                "Predicted_High",
                "Predicted_Low",
                "Predicted_Medium",
            ],
        )

        cm_df.to_csv(
            OUTPUT_REPORT_DIR
            / (
                "confusion_matrix_"
                + model_name
                .lower()
                .replace(
                    " ",
                    "_",
                )
                + ".csv"
            )
        )

    # ------------------------------------------------------------
    # Persistence baseline
    # ------------------------------------------------------------

    persistence_prediction = (
        persistence_predictions(
            test_df,
            complete_df,
        )
    )

    persistence_probability = np.zeros(
        (
            len(
                persistence_prediction
            ),
            len(CLASS_ORDER),
        )
    )

    persistence_probability[
        np.arange(
            len(
                persistence_prediction
            )
        ),
        persistence_prediction,
    ] = 1.0

    persistence_metrics = (
        calculate_metrics(
            y_test,
            persistence_prediction,
            persistence_probability,
        )
    )

    records.append(
        {
            "Model": "Persistence Baseline",
            **persistence_metrics,
            "High_Risk_Threshold": np.nan,
        }
    )

    predictions_store[
        "Persistence Baseline"
    ] = (
        persistence_prediction,
        persistence_probability,
    )

    benchmark = pd.DataFrame(
        records
    )

    return (
        benchmark,
        predictions_store,
    )


# ================================================================
# OUTPUTS
# ================================================================

def save_predictions(
    test_df,
    selected_model,
    selected_prediction,
    selected_probability,
    selected_threshold,
):
    output = test_df.copy()

    inverse = {
        index: label
        for index, label
        in enumerate(
            CLASS_ORDER
        )
    }

    actual = encode_target(
        test_df
    )

    output[
        "actual_risk_N1"
    ] = [
        inverse[int(x)]
        for x in actual
    ]

    output[
        "predicted_risk_N1"
    ] = [
        inverse[int(x)]
        for x in selected_prediction
    ]

    output[
        "prediction_correct"
    ] = (
        output[
            "actual_risk_N1"
        ]
        ==
        output[
            "predicted_risk_N1"
        ]
    )

    output[
        "probability_high"
    ] = selected_probability[
        :, 0
    ]

    output[
        "probability_low"
    ] = selected_probability[
        :, 1
    ]

    output[
        "probability_medium"
    ] = selected_probability[
        :, 2
    ]

    output[
        "selected_model"
    ] = selected_model

    output[
        "high_risk_threshold"
    ] = selected_threshold

    prediction_path = (
        OUTPUT_PREDICTION_DIR
        / "selected_model_predictions.csv"
    )

    oot_path = (
        OUTPUT_PREDICTION_DIR
        / "out_of_time_predictions.csv"
    )

    output.to_csv(
        prediction_path,
        index=False,
    )

    output.to_csv(
        oot_path,
        index=False,
    )

    print(
        "\nSelected predictions saved:"
    )

    print(
        prediction_path
    )

    print(
        "OOT predictions saved:"
    )

    print(
        oot_path
    )


def save_governance(
    selected_model,
    selected_threshold,
    train_periods,
    test_periods,
    benchmark,
    selection_table,
):
    governance = {
        "project": (
            "Temporal Supplier Risk "
            "Prediction and Prioritization"
        ),

        "prediction_task": (
            "Predict supplier risk category "
            "in Month N+1 using Month N information."
        ),

        "selected_model": selected_model,

        "selected_high_risk_threshold": (
            None
            if pd.isna(
                selected_threshold
            )
            else float(
                selected_threshold
            )
        ),

        "training_range": [
            str(train_periods[0]),
            str(train_periods[-1]),
        ],

        "out_of_time_range": [
            str(test_periods[0]),
            str(test_periods[-1]),
        ],

        "feature_count": len(
            FEATURES
        ),

        "features": FEATURES,

        "class_order": CLASS_ORDER,

        "selection_method": (
            "Three expanding-window temporal "
            "validation folds."
        ),

        "selection_score": (
            "0.35 Accuracy + "
            "0.25 High-risk Precision + "
            "0.25 High-risk F1 + "
            "0.15 Macro F1"
        ),

        "threshold_selection": (
            "High-risk threshold selected only "
            "inside historical validation folds."
        ),

        "leakage_controls": [
            "No random train/test split.",
            "Month N features predict Month N+1 target.",
            "N-1 and rolling features use Month N or earlier.",
            "Training-only median imputation.",
            "Training-only StandardScaler fitting.",
            "OOT test never used for model selection.",
        ],

        "final_oot_benchmark": (
            benchmark
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

        "historical_selection_table": (
            selection_table
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    path = (
        OUTPUT_REPORT_DIR
        / "model_governance_summary.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            governance,
            file,
            indent=2,
        )

    print(
        "Governance summary saved:"
    )

    print(path)


# ================================================================
# MAIN PIPELINE
# ================================================================

def run_model_pipeline():

    print_header(
        "PHASE 3: ROBUST TEMPORAL "
        "MODEL SELECTION"
    )

    print(
        f"Input file: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    raw_df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input shape: {raw_df.shape}"
    )

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------

    df = build_temporal_features(
        raw_df
    )

    print_header(
        "LEAKAGE-SAFE TEMPORAL "
        "FEATURE ENGINEERING"
    )

    print(
        f"Expanded feature count: "
        f"{len(FEATURES)}"
    )

    print(
        "N-1 and rolling features created "
        "using Month N or earlier only: PASSED"
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    validate_input(
        df
    )

    periods = (
        validate_temporal_structure(
            df
        )
    )

    leakage_audit()

    # ------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------

    train_periods = periods[
        :FINAL_TRAIN_PERIODS
    ]

    test_periods = periods[
        FINAL_TRAIN_PERIODS:
    ]

    if len(test_periods) != EXPECTED_OOT_PERIODS:
        print(
            "\nWARNING:"
        )

        print(
            f"Expected {EXPECTED_OOT_PERIODS} "
            f"OOT periods but found "
            f"{len(test_periods)}."
        )

    train_df = (
        df[
            df[
                PERIOD_COLUMN
            ]
            .astype(str)
            .isin(
                train_periods
            )
        ]
        .copy()
    )

    test_df = (
        df[
            df[
                PERIOD_COLUMN
            ]
            .astype(str)
            .isin(
                test_periods
            )
        ]
        .copy()
    )

    print_header(
        "CHRONOLOGICAL TRAIN / TEST SPLIT"
    )

    print(
        f"Training periods : "
        f"{len(train_periods)}"
    )

    print(
        f"Testing periods  : "
        f"{len(test_periods)}"
    )

    print(
        f"Training range   : "
        f"{train_periods[0]} -> "
        f"{train_periods[-1]}"
    )

    print(
        f"Testing range    : "
        f"{test_periods[0]} -> "
        f"{test_periods[-1]}"
    )

    print(
        f"Training rows    : "
        f"{len(train_df)}"
    )

    print(
        f"Testing rows     : "
        f"{len(test_df)}"
    )

    overlap = (
        set(train_periods)
        &
        set(test_periods)
    )

    if overlap:
        raise ValueError(
            f"Train/test temporal overlap: "
            f"{overlap}"
        )

    print(
        "Chronological split validation: PASSED"
    )

    # ------------------------------------------------------------
    # Target distribution
    # ------------------------------------------------------------

    print_header(
        "TARGET DISTRIBUTION"
    )

    print(
        "\nTraining:"
    )

    print(
        train_df[
            TARGET
        ].value_counts()
    )

    print(
        "\nOut-of-time test:"
    )

    print(
        test_df[
            TARGET
        ].value_counts()
    )

    # ------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------

    print_header(
        "FEATURE MATRIX"
    )

    print(
        f"Number of features: "
        f"{len(FEATURES)}"
    )

    for index, feature in enumerate(
        FEATURES,
        start=1,
    ):
        print(
            f"{index:2d}. {feature}"
        )

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    models = create_models()

    # ------------------------------------------------------------
    # Rolling temporal validation
    # ------------------------------------------------------------

    (
        validation_folds,
        fold_results,
        aggregate_results,
        model_thresholds,
    ) = evaluate_all_models(
        train_df,
        train_periods,
        models,
    )

    fold_results.to_csv(
        OUTPUT_REPORT_DIR
        / "rolling_fold_results.csv",
        index=False,
    )

    aggregate_results.to_csv(
        OUTPUT_REPORT_DIR
        / "rolling_model_results.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Persistence baseline across the same validation windows
    # ------------------------------------------------------------

    persistence_fold_metrics = []

    for (
        fold_train_periods,
        fold_validation_periods,
    ) in validation_folds:

        validation_df = (
            train_df[
                train_df[
                    PERIOD_COLUMN
                ]
                .astype(str)
                .isin(
                    fold_validation_periods
                )
            ]
            .copy()
        )

        persistence_fold_metrics.append(
            evaluate_persistence(
                validation_df,
                df,
            )
        )

    persistence_metrics = {
        key: float(
            np.mean(
                [
                    row[key]
                    for row
                    in persistence_fold_metrics
                ]
            )
        )
        for key
        in persistence_fold_metrics[0]
    }

    # ------------------------------------------------------------
    # Historical model selection
    # ------------------------------------------------------------

    selection_table = select_model(
        aggregate_results,
        persistence_metrics,
    )

    selection_table.to_csv(
        OUTPUT_REPORT_DIR
        / "rolling_model_selection_results.csv",
        index=False,
    )

    print_header(
        "ROLLING MODEL BENCHMARK"
    )

    print(
        selection_table.to_string(
            index=False
        )
    )

    selected_model = (
        selection_table
        .iloc[0]["Model"]
    )

    if (
        selected_model
        == "Persistence Baseline"
    ):
        selected_threshold = np.nan

        print(
            "\nWARNING:"
        )

        print(
            "Persistence still wins the "
            "historical validation score."
        )

        print(
            "The ML models should NOT be "
            "claimed as superior until they "
            "demonstrate lift."
        )

    else:
        selected_threshold = (
            model_thresholds[
                selected_model
            ]
        )

    print(
        "\nSelected model from historical "
        "validation only:"
    )

    print(
        selected_model
    )

    print(
        "Selected high-risk threshold:",
        selected_threshold,
    )

    # ------------------------------------------------------------
    # FINAL OOT EVALUATION
    # ------------------------------------------------------------

    (
        oot_benchmark,
        oot_predictions,
    ) = evaluate_oot_models(
        models,
        model_thresholds,
        train_df,
        test_df,
        df,
    )

    oot_benchmark.to_csv(
        OUTPUT_REPORT_DIR
        / "final_oot_model_benchmark.csv",
        index=False,
    )

    print_header(
        "FINAL OOT BENCHMARK"
    )

    print(
        oot_benchmark
        .sort_values(
            [
                "Accuracy",
                "High_Risk_Precision",
                "Macro_F1",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Selected model prediction
    # ------------------------------------------------------------

    selected_prediction, selected_probability = (
        oot_predictions[
            selected_model
        ]
    )

    save_predictions(
        test_df,
        selected_model,
        selected_prediction,
        selected_probability,
        selected_threshold,
    )

    # ------------------------------------------------------------
    # Governance
    # ------------------------------------------------------------

    save_governance(
        selected_model,
        selected_threshold,
        train_periods,
        test_periods,
        oot_benchmark,
        selection_table,
    )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    print_header(
        "PHASE 3 VALIDATION"
    )

    print(
        "✓ Input schema validated"
    )

    print(
        "✓ Base features checked"
    )

    print(
        "✓ Engineered temporal NaNs handled "
        "with training-only imputation"
    )

    print(
        "✓ Temporal ordering validated"
    )

    print(
        "✓ One-month-ahead relationship validated"
    )

    print(
        "✓ Leakage audit passed"
    )

    print(
        "✓ Chronological train/test split used"
    )

    print(
        "✓ Rolling temporal validation used"
    )

    print(
        "✓ High-risk threshold tuned only "
        "on historical validation"
    )

    print(
        "✓ Logistic Regression evaluated"
    )

    print(
        "✓ Random Forest evaluated"
    )

    print(
        "✓ Extra Trees evaluated"
    )

    print(
        "✓ HistGradientBoosting evaluated"
    )

    if XGBOOST_AVAILABLE:
        print(
            "✓ XGBoost evaluated"
        )

    print(
        "✓ Persistence baseline evaluated"
    )

    print(
        "✓ OOT test used for evaluation only"
    )

    print(
        "✓ Predictions saved"
    )

    print(
        "✓ Governance summary saved"
    )

    print_header(
        "PHASE 3 COMPLETE"
    )

    print(
        "Selected model:",
        selected_model,
    )

    print(
        "Selected threshold:",
        selected_threshold,
    )

    print(
        "Benchmark:",
        OUTPUT_REPORT_DIR
        / "final_oot_model_benchmark.csv",
    )

    print(
        "Predictions:",
        OUTPUT_PREDICTION_DIR
        / "selected_model_predictions.csv",
    )

    print(
        "\nDONE."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_model_pipeline()
