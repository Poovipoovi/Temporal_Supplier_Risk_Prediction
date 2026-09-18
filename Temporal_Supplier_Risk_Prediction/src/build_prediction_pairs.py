# build_prediction_pairs.py
#
# PHASE 2
# Temporal Target Engineering + N -> N+1 Prediction Pair Creation
#
# Input:
#   data/processed/aerospace_supplier_monthly.csv
#
# Output:
#   data/processed/aerospace_prediction_pairs.csv
#
# Method:
#   1. Read monthly supplier-level feature data.
#   2. Establish true chronological month order.
#   3. Calibrate the risk-score normalization ONLY on training-period data.
#   4. Construct a domain-informed composite risk score.
#   5. Convert the score into Low / Medium / High risk categories.
#   6. Create genuine Month-N -> Month-N+1 prediction pairs.
#   7. Perform explicit temporal and leakage validation.
#
# IMPORTANT:
#   The target risk category is engineered from operational KPIs.
#   It is NOT a naturally observed ground-truth label.
#
#   The risk score itself and future-period target variables are NEVER
#   included in the model feature matrix.


import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/aerospace_supplier_monthly.csv"
OUTPUT_FILE = "data/processed/aerospace_prediction_pairs.csv"

# Chronological train calibration window.
# The final 10 months are reserved as the out-of-time evaluation period
# later in the modeling phase.
TEST_MONTHS = 10

# Domain-informed KPI weights.
#
# Delivery reliability = 35%
# Quality performance  = 30%
# Delivery severity    = 20%
# Scrap impact         = 15%
#
# These sum to 1.00.
WEIGHTS = {
    "late_delivery_rate": 0.35,
    "avg_days_late_norm": 0.20,
    "quality_event_rate": 0.30,
    "scrap_qty_norm": 0.15,
}

RANDOM_STATE = 42


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "supplier_id",
    "period",

    # Operational performance
    "late_delivery_rate",
    "avg_days_late",
    "short_delivery_rate",
    "route_deviation_flag",
    "transit_delay_rate",

    # Quality / cost
    "quality_event_rate",
    "total_scrap_qty",
    "rework_cost_ratio",
    "supplier_fault_rate",

    # Temporal trend features
    "late_rate_trend",
    "quality_rate_trend",
    "avg_days_late_trend",
    "transit_delay_trend",

    # Business exposure
    "spend_share",
    "critical_component_exposure",

    # History indicators
    "is_worsening",
    "has_trend_data",
]


# ============================================================
# MODEL FEATURE COLUMNS
# ============================================================

MODEL_FEATURES = [
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_input(df):
    """Validate the Phase-1 monthly dataset."""

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "\nMissing required columns:\n - "
            + "\n - ".join(missing)
        )

    if df["supplier_id"].isna().any():
        raise ValueError("supplier_id contains missing values.")

    if df["period"].isna().any():
        raise ValueError("period contains missing values.")

    print("Input schema validation: PASSED")


def convert_period_to_month(period_series):
    """
    Convert YYYY-MM period strings to pandas Period[M].

    This prevents accidental lexicographical ordering problems.
    """

    converted = pd.PeriodIndex(
        period_series.astype(str),
        freq="M"
    )

    return converted


def safe_min_max_fit(train_series):
    """
    Fit min-max normalization using TRAINING data only.

    Returns the training minimum and maximum.
    """

    values = pd.to_numeric(train_series, errors="coerce")

    min_value = values.min()
    max_value = values.max()

    if pd.isna(min_value) or pd.isna(max_value):
        raise ValueError(
            "Cannot normalize a column containing no valid numeric values."
        )

    return float(min_value), float(max_value)


def safe_min_max_transform(series, min_value, max_value):
    """
    Apply training-fitted min-max scaling.

    Values outside the training range are clipped to [0, 1].
    """

    values = pd.to_numeric(series, errors="coerce")

    if pd.isna(min_value) or pd.isna(max_value):
        raise ValueError("Invalid normalization parameters.")

    if max_value == min_value:
        return pd.Series(
            np.zeros(len(values)),
            index=series.index,
            dtype=float
        )

    scaled = (values - min_value) / (max_value - min_value)

    return scaled.clip(0.0, 1.0)


def calculate_risk_score(df, normalization_parameters):
    """
    Calculate composite operational risk score.

    IMPORTANT:
    Normalization parameters originate from training periods only.
    """

    days_min, days_max = normalization_parameters["avg_days_late"]
    scrap_min, scrap_max = normalization_parameters["total_scrap_qty"]

    df = df.copy()

    df["avg_days_late_norm"] = safe_min_max_transform(
        df["avg_days_late"],
        days_min,
        days_max
    )

    df["scrap_qty_norm"] = safe_min_max_transform(
        df["total_scrap_qty"],
        scrap_min,
        scrap_max
    )

    df["risk_score"] = (
        WEIGHTS["late_delivery_rate"]
        * df["late_delivery_rate"]
        +
        WEIGHTS["avg_days_late_norm"]
        * df["avg_days_late_norm"]
        +
        WEIGHTS["quality_event_rate"]
        * df["quality_event_rate"]
        +
        WEIGHTS["scrap_qty_norm"]
        * df["scrap_qty_norm"]
    )

    return df


def determine_target_boundaries(training_scores):
    """
    Determine Low / Medium / High boundaries using TRAINING scores only.

    We use the 25th and 85th percentiles to create an operationally
    useful three-tier target:

        Low    = bottom 25%
        Medium = middle 60%
        High   = top 15%

    The thresholds are learned only from the historical calibration
    window and are frozen before the future evaluation period.
    """

    scores = pd.Series(training_scores).dropna()

    if len(scores) < 20:
        raise ValueError(
            "Too few training observations to calibrate risk boundaries."
        )

    low_boundary = float(scores.quantile(0.25))
    high_boundary = float(scores.quantile(0.85))

    if high_boundary <= low_boundary:
        raise ValueError(
            "Invalid risk boundaries: High boundary must exceed "
            "Low boundary."
        )

    return low_boundary, high_boundary


def assign_risk_category(score, low_boundary, high_boundary):
    """Assign Low / Medium / High category."""

    if score <= low_boundary:
        return "Low"

    if score <= high_boundary:
        return "Medium"

    return "High"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase_2():

    print("=" * 70)
    print("PHASE 2: TEMPORAL TARGET ENGINEERING AND PREDICTION PAIR CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD MONTHLY DATA
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n\n"
            "Run the Phase-1 monthly feature engineering script first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Input shape: {df.shape}")

    validate_input(df)

    # --------------------------------------------------------
    # 2. VALIDATE PERIOD STRUCTURE
    # --------------------------------------------------------

    df["_period_month"] = convert_period_to_month(df["period"])

    unique_periods = (
        df["_period_month"]
        .drop_duplicates()
        .sort_values()
    )

    if len(unique_periods) < TEST_MONTHS + 2:
        raise ValueError(
            "Insufficient number of time periods for temporal "
            "calibration and out-of-time testing."
        )

    print(f"Unique periods detected: {len(unique_periods)}")

    chronological_periods = [
        str(p) for p in unique_periods
    ]

    print("\nChronological period order:")
    print(chronological_periods)

    # --------------------------------------------------------
    # 3. CHECK FOR DUPLICATE SUPPLIER-MONTH RECORDS
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=["supplier_id", "_period_month"]
    )

    duplicate_count = int(duplicates.sum())

    print("\nSupplier-month uniqueness check:")

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate supplier-month records."
        )

    print("Duplicate supplier-month records: 0")
    print("Supplier-month uniqueness: PASSED")

    # --------------------------------------------------------
    # 4. TEMPORAL TRAIN / TEST CALIBRATION BOUNDARY
    # --------------------------------------------------------

    # The final TEST_MONTHS are held out for future modeling evaluation.
    calibration_periods = unique_periods[:-TEST_MONTHS]
    holdout_periods = unique_periods[-TEST_MONTHS:]

    print("\n" + "=" * 70)
    print("TARGET CALIBRATION WINDOW")
    print("=" * 70)

    print(
        f"\nCalibration periods: {len(calibration_periods)}"
    )

    print(
        f"Calibration range: "
        f"{calibration_periods.iloc[0]} -> {calibration_periods.iloc[-1]}"
    )

    print(
        f"\nFuture holdout periods: {len(holdout_periods)}"
    )

    print(
        f"Holdout range: "
        f"{holdout_periods.iloc[0]} -> {holdout_periods.iloc[-1]}"
    )

    # --------------------------------------------------------
    # 5. FIT NORMALIZATION ONLY ON HISTORICAL DATA
    # --------------------------------------------------------

    calibration_mask = df["_period_month"].isin(
        calibration_periods
    )

    calibration_df = df.loc[calibration_mask].copy()

    if calibration_df.empty:
        raise ValueError("Calibration dataset is empty.")

    print("\n" + "=" * 70)
    print("RISK SCORE CALIBRATION")
    print("=" * 70)

    print(
        f"\nCalibration observations: {len(calibration_df)}"
    )

    normalization_parameters = {}

    for column in [
        "avg_days_late",
        "total_scrap_qty",
    ]:

        min_value, max_value = safe_min_max_fit(
            calibration_df[column]
        )

        normalization_parameters[column] = (
            min_value,
            max_value
        )

        print(
            f"\n{column}:"
            f"\n  Training minimum = {min_value:.6f}"
            f"\n  Training maximum = {max_value:.6f}"
        )

    # --------------------------------------------------------
    # 6. CALCULATE RISK SCORE
    # --------------------------------------------------------

    df = calculate_risk_score(
        df,
        normalization_parameters
    )

    calibration_scores = df.loc[
        calibration_mask,
        "risk_score"
    ]

    # --------------------------------------------------------
    # 7. CALIBRATE RISK CATEGORY BOUNDARIES
    # --------------------------------------------------------

    low_boundary, high_boundary = determine_target_boundaries(
        calibration_scores
    )

    print("\n" + "=" * 70)
    print("CALIBRATED RISK CATEGORY BOUNDARIES")
    print("=" * 70)

    print(
        f"\nLow / Medium boundary  : {low_boundary:.6f}"
    )

    print(
        f"Medium / High boundary : {high_boundary:.6f}"
    )

    # --------------------------------------------------------
    # 8. ASSIGN CURRENT-PERIOD RISK CATEGORY
    # --------------------------------------------------------

    df["risk_category_current"] = df["risk_score"].apply(
        lambda x: assign_risk_category(
            x,
            low_boundary,
            high_boundary
        )
    )

    print("\nCurrent-period risk distribution:")

    current_distribution = (
        df["risk_category_current"]
        .value_counts()
    )

    print(current_distribution)

    print("\nCurrent-period percentages:")

    print(
        (
            df["risk_category_current"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # --------------------------------------------------------
    # 9. CREATE TEMPORAL N -> N+1 PAIRS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING CHRONOLOGICAL N -> N+1 PREDICTION PAIRS")
    print("=" * 70)

    # Sort using actual monthly Period values.
    df = df.sort_values(
        by=["supplier_id", "_period_month"]
    ).reset_index(drop=True)

    # Previous/next period for each supplier.
    df["next_period"] = (
        df.groupby("supplier_id")["_period_month"]
        .shift(-1)
    )

    df["next_risk_category"] = (
        df.groupby("supplier_id")["risk_category_current"]
        .shift(-1)
    )

    df["next_risk_score"] = (
        df.groupby("supplier_id")["risk_score"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # 10. REQUIRE TRUE CONSECUTIVE MONTHS
    # --------------------------------------------------------

    df["months_to_next"] = (
        df["next_period"] - df["_period_month"]
    ).apply(
        lambda x: x.n if pd.notna(x) else np.nan
    )

    valid_next_month = (
        df["months_to_next"] == 1
    )

    skipped_supplier_months = int(
        (
            df["next_period"].notna()
            &
            (~valid_next_month)
        ).sum()
    )

    print(
        f"\nNon-consecutive supplier-month transitions: "
        f"{skipped_supplier_months}"
    )

    if skipped_supplier_months > 0:
        print(
            "WARNING: Some suppliers do not have observations "
            "in consecutive months."
        )

    # Only genuine N -> N+1 monthly observations.
    prediction_pairs = df.loc[
        valid_next_month
    ].copy()

    # --------------------------------------------------------
    # 11. RENAME TARGET COLUMNS
    # --------------------------------------------------------

    prediction_pairs = prediction_pairs.rename(
        columns={
            "next_risk_category": "target_risk_N1",
            "next_risk_score": "target_risk_score_N1",
            "_period_month": "period_N",
            "next_period": "period_N1",
        }
    )

    # --------------------------------------------------------
    # 12. LEAKAGE-SAFE MODEL FEATURE MATRIX
    # --------------------------------------------------------

    feature_cols = MODEL_FEATURES.copy()

    missing_model_features = [
        c for c in feature_cols
        if c not in prediction_pairs.columns
    ]

    if missing_model_features:
        raise ValueError(
            "\nMissing model features:\n - "
            + "\n - ".join(missing_model_features)
        )

    # --------------------------------------------------------
    # 13. EXPLICIT LEAKAGE CHECK
    # --------------------------------------------------------

    forbidden_features = {
        "risk_score",
        "risk_category_current",
        "target_risk_N1",
        "target_risk_score_N1",
        "period_N1",
        "next_period",
        "next_risk_category",
        "next_risk_score",
        "months_to_next",
        "avg_days_late_norm",
        "scrap_qty_norm",
    }

    leakage_features = [
        c for c in feature_cols
        if c in forbidden_features
        or "N1" in c
        or "future" in c.lower()
        or "next" in c.lower()
    ]

    print("\n" + "=" * 70)
    print("LEAKAGE VALIDATION")
    print("=" * 70)

    if leakage_features:
        raise ValueError(
            "\nPotential leakage features detected:\n - "
            + "\n - ".join(leakage_features)
        )

    print("Target accidentally present in X? False")
    print("Future risk score present in X? False")
    print("Future period present in X? False")
    print("Leakage validation: PASSED")

    # --------------------------------------------------------
    # 14. MISSING-VALUE VALIDATION
    # --------------------------------------------------------

    missing_counts = (
        prediction_pairs[feature_cols]
        .isna()
        .sum()
    )

    missing_features = (
        missing_counts[missing_counts > 0]
    )

    print("\n" + "=" * 70)
    print("FEATURE MATRIX VALIDATION")
    print("=" * 70)

    if not missing_features.empty:

        print("\nMissing values detected:")

        print(missing_features)

        raise ValueError(
            "Model feature matrix contains missing values."
        )

    print(
        f"\nNumber of X features: {len(feature_cols)}"
    )

    print("\nX features:")

    for i, feature in enumerate(feature_cols, start=1):
        print(f" {i:2d}. {feature}")

    print("\nMissing values in model features: None")
    print("Feature validation: PASSED")

    # --------------------------------------------------------
    # 15. CONVERT PERIODS BACK TO STRINGS
    # --------------------------------------------------------

    prediction_pairs["period_N"] = (
        prediction_pairs["period_N"]
        .astype(str)
    )

    prediction_pairs["period_N1"] = (
        prediction_pairs["period_N1"]
        .astype(str)
    )

    # --------------------------------------------------------
    # 16. FINAL TARGET VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    target_distribution = (
        prediction_pairs["target_risk_N1"]
        .value_counts()
    )

    print("\nTarget distribution:")
    print(target_distribution)

    print("\nTarget percentages:")

    print(
        (
            prediction_pairs["target_risk_N1"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    required_classes = {"Low", "Medium", "High"}

    actual_classes = set(
        prediction_pairs["target_risk_N1"]
        .dropna()
        .unique()
    )

    missing_classes = required_classes - actual_classes

    if missing_classes:
        raise ValueError(
            "Target is missing required classes: "
            + ", ".join(sorted(missing_classes))
        )

    print("\nAll three target classes are present.")
    print("Target validation: PASSED")

    # --------------------------------------------------------
    # 17. TEMPORAL RELATIONSHIP VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL RELATIONSHIP VALIDATION")
    print("=" * 70)

    period_n = pd.PeriodIndex(
        prediction_pairs["period_N"],
        freq="M"
    )

    period_n1 = pd.PeriodIndex(
        prediction_pairs["period_N1"],
        freq="M"
    )

    month_difference = (
        period_n1 - period_n
    ).map(lambda x: x.n)

    invalid_pairs = (
        month_difference != 1
    )

    print(
        f"\nInvalid N -> N+1 transitions: "
        f"{int(invalid_pairs.sum())}"
    )

    if invalid_pairs.any():
        raise ValueError(
            "Temporal validation failed: "
            "some pairs are not consecutive months."
        )

    print(
        "Every prediction pair represents exactly "
        "one-month-ahead prediction."
    )

    print("Temporal validation: PASSED")

    # --------------------------------------------------------
    # 18. REMOVE INTERNAL COLUMNS
    # --------------------------------------------------------

    # Keep the current-period category as metadata for the persistence baseline.
    # It is NOT included in MODEL_FEATURES and therefore cannot enter X.
    prediction_pairs = prediction_pairs.rename(
        columns={"risk_category_current": "current_risk_N"}
    )

    columns_to_remove = [
        "risk_score",
        "risk_category_current",
        "avg_days_late_norm",
        "scrap_qty_norm",
        "next_period",
        "next_risk_category",
        "next_risk_score",
        "months_to_next",
    ]

    prediction_pairs = prediction_pairs.drop(
        columns=[
            c for c in columns_to_remove
            if c in prediction_pairs.columns
        ]
    )

    # --------------------------------------------------------
    # 19. ADD METADATA COLUMNS
    # --------------------------------------------------------

    prediction_pairs["target_low_boundary"] = low_boundary
    prediction_pairs["target_high_boundary"] = high_boundary

    # --------------------------------------------------------
    # 20. SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    prediction_pairs.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 21. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput saved as:\n{OUTPUT_FILE}"
    )

    print(
        f"\nFinal shape: {prediction_pairs.shape}"
    )

    print("\nFinal target distribution:")

    print(
        prediction_pairs["target_risk_N1"]
        .value_counts()
    )

    print("\nFinal target percentages:")

    print(
        (
            prediction_pairs["target_risk_N1"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nRisk boundaries used:")

    print(
        f"Low / Medium  : {low_boundary:.6f}"
    )

    print(
        f"Medium / High : {high_boundary:.6f}"
    )

    print("\nIMPORTANT:")
    print("- Phase-1 input CSV was NOT modified.")
    print("- Only the Phase-2 output CSV was created.")
    print("- X contains Month-N operational information only.")
    print("- y contains Month-N+1 risk category only.")
    print("- Normalization was fitted using historical calibration periods only.")
    print("- Risk thresholds were calibrated using historical periods only.")
    print("- Non-consecutive supplier-month transitions were excluded.")
    print("- No future-period target information enters X.")

    print("\nDONE.")


if __name__ == "__main__":
    run_phase_2()