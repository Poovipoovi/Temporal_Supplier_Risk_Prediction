
"""
PHASE 4: EXPOSURE-WEIGHTED SUPPLIER RISK PRIORITIZATION

Purpose
-------
Convert Phase-3 one-month-ahead supplier risk predictions into
business-prioritized procurement intelligence.

Methodology
-----------
ML prediction
    -> Expected risk severity
    -> Business exposure
    -> Exposure-weighted priority score
    -> Priority tier
    -> Procurement watchlist
    -> Supplier-level summary

Important
---------
This phase DOES NOT retrain the ML model.
This phase DOES NOT modify Phase-1, Phase-2, or Phase-3 data.
This phase only consumes Phase-3 out-of-time predictions.

Expected Phase-3 input columns include:
    supplier_id
    period_N
    period_N1
    actual_risk_N1
    predicted_risk_N1
    probability_high
    probability_low
    probability_medium
    spend_share
    critical_component_exposure

The script is deliberately defensive about column naming.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "selected_model_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prioritization"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

ALL_PERIODS_FILE = OUTPUT_DIR / "supplier_priority_all_periods.csv"
LATEST_FILE = OUTPUT_DIR / "supplier_priority_latest.csv"
WATCHLIST_FILE = OUTPUT_DIR / "high_risk_watchlist.csv"
SUPPLIER_SUMMARY_FILE = REPORT_DIR / "phase4_supplier_summary.csv"
SUMMARY_JSON_FILE = REPORT_DIR / "phase4_prioritization_summary.json"


# ============================================================================
# CONFIGURATION
# ============================================================================

RISK_SEVERITY = {
    "Low": 0.0,
    "Medium": 0.5,
    "High": 1.0,
}

# Probability threshold used for identifying predictions where the model
# has meaningful probability mass on the High-risk class.
HIGH_PROBABILITY_THRESHOLD = 0.50

# Business exposure weighting.
SPEND_WEIGHT = 0.70
CRITICAL_WEIGHT = 0.30

# Priority thresholds are applied to a 0-100 priority score.
#
# Priority score:
#     expected_risk_severity
#     × business_exposure
#     × 100
#
# These thresholds are deliberately transparent rather than learned from
# the out-of-time test set.
HIGH_PRIORITY_THRESHOLD = 20.0
MEDIUM_PRIORITY_THRESHOLD = 7.5


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_columns(df, required_columns, dataset_name):
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"\n{dataset_name} is missing required columns:\n"
            + "\n".join(f" - {c}" for c in missing)
        )


def safe_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def validate_no_missing(df, columns, dataset_name):
    missing_counts = df[columns].isna().sum()
    bad = missing_counts[missing_counts > 0]

    if len(bad) > 0:
        raise ValueError(
            f"\n{dataset_name} contains missing values:\n{bad.to_string()}"
        )


# ============================================================================
# PHASE 4 INPUT VALIDATION
# ============================================================================

def load_predictions():
    print_header("LOADING PHASE-3 PREDICTIONS")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Phase-3 prediction file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file: {INPUT_FILE}")
    print(f"Input shape: {df.shape}")

    return df


def validate_predictions(df):
    print_header("INPUT VALIDATION")

    required = [
        "supplier_id",
        "period_N",
        "period_N1",
        "probability_high",
        "probability_low",
        "probability_medium",
        "spend_share",
        "critical_component_exposure",
    ]

    # Phase-3 uses actual_risk_N1 / predicted_risk_N1.
    # These are the authoritative columns produced by Phase 3.
    if "actual_risk_N1" not in df.columns:
        raise ValueError(
            "Phase-3 output does not contain 'actual_risk_N1'."
        )

    if "predicted_risk_N1" not in df.columns:
        raise ValueError(
            "Phase-3 output does not contain 'predicted_risk_N1'."
        )

    validate_columns(df, required, "Prediction file")

    print("Prediction schema validation: PASSED")

    numeric_columns = [
        "probability_high",
        "probability_low",
        "probability_medium",
        "spend_share",
        "critical_component_exposure",
    ]

    safe_numeric(df, numeric_columns)

    validate_no_missing(
        df,
        required,
        "Prediction file"
    )

    if len(df) == 0:
        raise ValueError("Prediction file contains zero records.")

    print("Required prediction-field completeness: PASSED")


# ============================================================================
# COLUMN STANDARDIZATION
# ============================================================================

def standardize_prediction_columns(df):
    print_header("PREDICTION COLUMN STANDARDIZATION")

    # Do NOT change the source CSV.
    # Create internal standardized aliases for Phase 4.
    df["actual_risk_category"] = df["actual_risk_N1"]
    df["predicted_risk_category"] = df["predicted_risk_N1"]

    print("Source actual-risk column    : actual_risk_N1")
    print("Source predicted-risk column: predicted_risk_N1")

    print("Internal actual-risk column    : actual_risk_category")
    print("Internal predicted-risk column: predicted_risk_category")

    print("Column standardization: PASSED")

    return df


# ============================================================================
# TEMPORAL VALIDATION
# ============================================================================

def validate_temporal_structure(df):
    print_header("TEMPORAL STRUCTURE VALIDATION")

    print(f"Total prediction pairs: {len(df):,}")
    print(f"Unique suppliers: {df['supplier_id'].nunique():,}")
    print(f"Unique Month-N periods: {df['period_N'].nunique():,}")
    print(f"Unique Month-N+1 periods: {df['period_N1'].nunique():,}")

    invalid = []

    for n, n1 in zip(df["period_N"], df["period_N1"]):
        n_period = pd.Period(str(n), freq="M")
        n1_period = pd.Period(str(n1), freq="M")

        if n1_period != n_period + 1:
            invalid.append((n, n1))

    print(f"Invalid N -> N+1 transitions: {len(invalid)}")

    if invalid:
        print("Examples of invalid transitions:")
        for pair in invalid[:10]:
            print(f"  {pair[0]} -> {pair[1]}")

        raise ValueError(
            "Temporal validation failed: invalid Month-N -> Month-N+1 transitions."
        )

    print("One-month-ahead relationship: PASSED")


# ============================================================================
# RISK CATEGORY VALIDATION
# ============================================================================

def validate_risk_categories(df):
    print_header("RISK CATEGORY VALIDATION")

    valid_categories = {"Low", "Medium", "High"}

    actual_categories = set(
        df["actual_risk_category"].dropna().unique()
    )

    predicted_categories = set(
        df["predicted_risk_category"].dropna().unique()
    )

    print(f"Actual categories: {sorted(actual_categories)}")
    print(f"Predicted categories: {sorted(predicted_categories)}")

    if not actual_categories.issubset(valid_categories):
        raise ValueError(
            f"Invalid actual risk categories: "
            f"{actual_categories - valid_categories}"
        )

    if not predicted_categories.issubset(valid_categories):
        raise ValueError(
            f"Invalid predicted risk categories: "
            f"{predicted_categories - valid_categories}"
        )

    print("Risk-category validation: PASSED")


# ============================================================================
# PROBABILITY VALIDATION
# ============================================================================

def validate_probabilities(df):
    print_header("MODEL PROBABILITY VALIDATION")

    probability_columns = [
        "probability_low",
        "probability_medium",
        "probability_high",
    ]

    for col in probability_columns:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite probability values found in {col}")

        if (df[col] < 0).any() or (df[col] > 1).any():
            raise ValueError(
                f"Probability values outside [0, 1] found in {col}"
            )

    probability_sum = (
        df["probability_low"]
        + df["probability_medium"]
        + df["probability_high"]
    )

    max_error = np.max(np.abs(probability_sum - 1.0))

    print(f"Maximum probability-sum error: {max_error:.10f}")

    if max_error > 1e-6:
        raise ValueError(
            "Probability validation failed: probabilities do not sum to 1."
        )

    print("Probability consistency check: PASSED")


# ============================================================================
# EXPOSURE VALIDATION
# ============================================================================

def validate_exposure_fields(df):
    print_header("EXPOSURE FIELD VALIDATION")

    for col in ["spend_share", "critical_component_exposure"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values found in {col}")

        if (df[col] < 0).any():
            raise ValueError(
                f"Negative values found in {col}"
            )

    print("Probability validation: PASSED")
    print("Spend-share validation: PASSED")
    print("Critical-exposure validation: PASSED")


# ============================================================================
# RISK SEVERITY
# ============================================================================

def calculate_expected_risk(df):
    print_header("RISK SEVERITY CALCULATION")

    print("Risk severity mapping:")
    print("  Low      -> 0.0")
    print("  Medium   -> 0.5")
    print("  High     -> 1.0")

    # Direct predicted category severity.
    df["predicted_risk_severity"] = (
        df["predicted_risk_category"]
        .map(RISK_SEVERITY)
    )

    # Probability-weighted expected severity.
    #
    # This is more informative than simply using the predicted class:
    #
    # Expected Risk =
    #     P(Low)*0
    #   + P(Medium)*0.5
    #   + P(High)*1
    #
    df["expected_risk_severity"] = (
        df["probability_low"] * RISK_SEVERITY["Low"]
        + df["probability_medium"] * RISK_SEVERITY["Medium"]
        + df["probability_high"] * RISK_SEVERITY["High"]
    )

    min_expected = df["expected_risk_severity"].min()
    max_expected = df["expected_risk_severity"].max()

    print(
        f"Expected risk severity range: "
        f"{min_expected:.4f} to {max_expected:.4f}"
    )

    if not np.isfinite(df["expected_risk_severity"]).all():
        raise ValueError("Expected risk contains non-finite values.")

    print("Expected-risk calculation: PASSED")

    return df


# ============================================================================
# BUSINESS EXPOSURE
# ============================================================================

def calculate_business_exposure(df):
    print_header("BUSINESS EXPOSURE CALCULATION")

    # Business exposure combines:
    #
    # 70% financial exposure
    # 30% critical-component exposure
    #
    # Both fields originate from Month N.
    #
    df["business_exposure"] = (
        SPEND_WEIGHT * df["spend_share"]
        + CRITICAL_WEIGHT * df["critical_component_exposure"]
    )

    min_exposure = df["business_exposure"].min()
    max_exposure = df["business_exposure"].max()

    print(
        f"Business exposure = "
        f"{SPEND_WEIGHT:.0%} spend share + "
        f"{CRITICAL_WEIGHT:.0%} critical-component exposure"
    )

    print(
        f"Business exposure range: "
        f"{min_exposure:.6f} to {max_exposure:.6f}"
    )

    if not np.isfinite(df["business_exposure"]).all():
        raise ValueError("Business exposure contains non-finite values.")

    if (df["business_exposure"] < 0).any():
        raise ValueError("Business exposure contains negative values.")

    print("Business-exposure calculation: PASSED")

    return df


# ============================================================================
# PRIORITY SCORE
# ============================================================================

def calculate_priority_score(df):
    print_header("EXPOSURE-WEIGHTED PRIORITIZATION")

    # Expected risk severity represents the likelihood-weighted severity.
    #
    # Business exposure represents how important the supplier is
    # operationally/financially.
    #
    # Their product therefore identifies suppliers where:
    #     risk is high AND exposure is high.
    #
    df["priority_score"] = (
        df["expected_risk_severity"]
        * df["business_exposure"]
        * 100.0
    )

    print(
        "Priority score formula:"
    )
    print(
        "Expected Risk Severity x Business Exposure x 100"
    )

    max_score = df["priority_score"].max()

    print(f"Maximum priority score: {max_score:.6f}")

    if not np.isfinite(df["priority_score"]).all():
        raise ValueError("Priority score contains non-finite values.")

    if (df["priority_score"] < 0).any():
        raise ValueError("Negative priority scores found.")

    print("Priority-score calculation: PASSED")

    return df


# ============================================================================
# PRIORITY TIERS
# ============================================================================

def assign_priority_tiers(df):
    print_header("PRIORITY TIER ASSIGNMENT")

    def tier(score):
        if score >= HIGH_PRIORITY_THRESHOLD:
            return "High Priority"
        elif score >= MEDIUM_PRIORITY_THRESHOLD:
            return "Medium Priority"
        return "Monitor"

    df["priority_tier"] = df["priority_score"].apply(tier)

    print(
        df["priority_tier"]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================================
# WATCHLIST FLAGS
# ============================================================================

def create_watchlist_flags(df):
    print_header("RISK WATCHLIST FLAGS")

    # These two columns are explicitly created BEFORE supplier summary.
    # This fixes the KeyError from the previous implementation.
    df["high_risk_prediction_flag"] = (
        df["predicted_risk_category"] == "High"
    ).astype(int)

    df["high_probability_high_risk_flag"] = (
        df["probability_high"] >= HIGH_PROBABILITY_THRESHOLD
    ).astype(int)

    # A critical-exposure flag identifies records where the supplier has
    # measurable critical-component exposure.
    df["critical_exposure_flag"] = (
        df["critical_component_exposure"] > 0
    ).astype(int)

    # Main procurement watchlist:
    # predicted High-risk suppliers.
    df["watchlist_flag"] = (
        df["high_risk_prediction_flag"] == 1
    ).astype(int)

    high_risk_count = int(
        df["high_risk_prediction_flag"].sum()
    )

    high_probability_count = int(
        df["high_probability_high_risk_flag"].sum()
    )

    critical_count = int(
        df["critical_exposure_flag"].sum()
    )

    watchlist_count = int(
        df["watchlist_flag"].sum()
    )

    print(f"High-risk predictions: {high_risk_count}")
    print(
        "High-probability High-risk predictions: "
        f"{high_probability_count}"
    )
    print(f"Critical-exposure flags: {critical_count}")
    print(f"Watchlist records: {watchlist_count}")

    return df


# ============================================================================
# LATEST VIEW
# ============================================================================

def create_latest_view(df):
    print_header("LATEST SUPPLIER PRIORITIZATION VIEW")

    periods = sorted(
        df["period_N"].astype(str).unique()
    )

    if not periods:
        raise ValueError("No Month-N periods available.")

    latest_period = periods[-1]

    latest = (
        df[df["period_N"].astype(str) == latest_period]
        .copy()
        .sort_values(
            by=["priority_score", "expected_risk_severity"],
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(f"Latest Month-N: {latest_period}")
    print(f"Latest supplier records: {len(latest)}")

    return latest


# ============================================================================
# SUPPLIER SUMMARY
# ============================================================================

def create_supplier_summary(df):
    print_header("SUPPLIER-LEVEL SUMMARY")

    # Defensive check: all fields used below must exist.
    summary_required = [
        "supplier_id",
        "period_N",
        "period_N1",
        "predicted_risk_category",
        "actual_risk_category",
        "expected_risk_severity",
        "business_exposure",
        "priority_score",
        "priority_tier",
        "high_risk_prediction_flag",
        "high_probability_high_risk_flag",
        "critical_exposure_flag",
        "watchlist_flag",
        "probability_high",
        "spend_share",
        "critical_component_exposure",
    ]

    validate_columns(
        df,
        summary_required,
        "Phase-4 working dataset"
    )

    # Latest record per supplier based on Month N.
    working = df.copy()

    working["_period_sort"] = pd.PeriodIndex(
        working["period_N"].astype(str),
        freq="M"
    )

    working = working.sort_values(
        ["supplier_id", "_period_sort"]
    )

    latest_records = (
        working
        .groupby("supplier_id", as_index=False)
        .tail(1)
        .copy()
    )

    # Historical supplier-level statistics.
    supplier_summary = (
        working
        .groupby("supplier_id")
        .agg(
            prediction_periods=("period_N", "nunique"),

            average_expected_risk=(
                "expected_risk_severity",
                "mean"
            ),

            maximum_expected_risk=(
                "expected_risk_severity",
                "max"
            ),

            average_priority_score=(
                "priority_score",
                "mean"
            ),

            maximum_priority_score=(
                "priority_score",
                "max"
            ),

            high_risk_prediction_count=(
                "high_risk_prediction_flag",
                "sum"
            ),

            high_probability_high_risk_count=(
                "high_probability_high_risk_flag",
                "sum"
            ),

            critical_exposure_count=(
                "critical_exposure_flag",
                "sum"
            ),

            watchlist_count=(
                "watchlist_flag",
                "sum"
            ),

            average_probability_high=(
                "probability_high",
                "mean"
            ),

            maximum_probability_high=(
                "probability_high",
                "max"
            ),

            average_spend_share=(
                "spend_share",
                "mean"
            ),

            maximum_spend_share=(
                "spend_share",
                "max"
            ),

            average_critical_exposure=(
                "critical_component_exposure",
                "mean"
            ),

            maximum_critical_exposure=(
                "critical_component_exposure",
                "max"
            ),
        )
        .reset_index()
    )

    # Add latest prediction information.
    latest_information = latest_records[
        [
            "supplier_id",
            "period_N",
            "period_N1",
            "actual_risk_category",
            "predicted_risk_category",
            "expected_risk_severity",
            "business_exposure",
            "priority_score",
            "priority_tier",
            "probability_high",
            "spend_share",
            "critical_component_exposure",
        ]
    ].rename(
        columns={
            "period_N": "latest_period_N",
            "period_N1": "latest_period_N1",
            "actual_risk_category": "latest_actual_risk",
            "predicted_risk_category": "latest_predicted_risk",
            "expected_risk_severity": "latest_expected_risk",
            "business_exposure": "latest_business_exposure",
            "priority_score": "latest_priority_score",
            "priority_tier": "latest_priority_tier",
            "probability_high": "latest_probability_high",
            "spend_share": "latest_spend_share",
            "critical_component_exposure": (
                "latest_critical_component_exposure"
            ),
        }
    )

    supplier_summary = supplier_summary.merge(
        latest_information,
        on="supplier_id",
        how="left"
    )

    # Supplier-level risk status.
    supplier_summary["supplier_watchlist_flag"] = (
        supplier_summary["watchlist_count"] > 0
    ).astype(int)

    supplier_summary["supplier_high_risk_flag"] = (
        supplier_summary["high_risk_prediction_count"] > 0
    ).astype(int)

    # Rank suppliers by latest priority score.
    supplier_summary = supplier_summary.sort_values(
        by=[
            "latest_priority_score",
            "maximum_priority_score",
            "average_priority_score",
        ],
        ascending=False
    ).reset_index(drop=True)

    supplier_summary["supplier_priority_rank"] = (
        np.arange(len(supplier_summary)) + 1
    )

    supplier_summary = supplier_summary.drop(
        columns=["_period_sort"],
        errors="ignore"
    )

    print(
        f"Supplier summary records: "
        f"{len(supplier_summary)}"
    )

    return supplier_summary


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================

def validate_outputs(
    df,
    latest,
    watchlist,
    supplier_summary
):
    print_header("PHASE 4 FINAL VALIDATION")

    if len(df) == 0:
        raise ValueError("All-period output is empty.")

    if len(latest) == 0:
        raise ValueError("Latest supplier view is empty.")

    if len(supplier_summary) == 0:
        raise ValueError("Supplier summary is empty.")

    if not np.isfinite(df["priority_score"]).all():
        raise ValueError("Non-finite priority scores found.")

    if (df["priority_score"] < 0).any():
        raise ValueError("Negative priority scores found.")

    expected_tiers = {
        "Monitor",
        "Medium Priority",
        "High Priority",
    }

    actual_tiers = set(df["priority_tier"].unique())

    if not actual_tiers.issubset(expected_tiers):
        raise ValueError(
            f"Unexpected priority tiers: "
            f"{actual_tiers - expected_tiers}"
        )

    print("✓ Prediction schema")
    print("✓ N -> N+1 temporal relationship")
    print("✓ Risk categories valid")
    print("✓ Probability values valid")
    print("✓ Business exposure valid")
    print("✓ Priority scores finite")
    print("✓ No negative priority scores")
    print("✓ Priority tiers valid")
    print("✓ Watchlist flags created")
    print("✓ Latest supplier view created")
    print("✓ Supplier summary created")

    print("\nPhase 4 validation: PASSED")


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    df,
    latest,
    watchlist,
    supplier_summary
):
    print_header("SAVING PHASE 4 OUTPUTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove internal helper fields from final all-period output.
    final_df = df.drop(
        columns=["_period_sort"],
        errors="ignore"
    )

    final_df.to_csv(
        ALL_PERIODS_FILE,
        index=False
    )

    latest.to_csv(
        LATEST_FILE,
        index=False
    )

    watchlist.to_csv(
        WATCHLIST_FILE,
        index=False
    )

    supplier_summary.to_csv(
        SUPPLIER_SUMMARY_FILE,
        index=False
    )

    print(
        f"All-period prioritization: "
        f"{ALL_PERIODS_FILE}"
    )

    print(
        f"Latest supplier priorities: "
        f"{LATEST_FILE}"
    )

    print(
        f"High-risk watchlist: "
        f"{WATCHLIST_FILE}"
    )

    print(
        f"Supplier summary: "
        f"{SUPPLIER_SUMMARY_FILE}"
    )


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def create_summary_report(
    df,
    latest,
    watchlist,
    supplier_summary
):
    print_header("PHASE 4 SUMMARY")

    latest_period = str(
        latest["period_N"].iloc[0]
    )

    predicted_distribution = (
        df["predicted_risk_category"]
        .value_counts()
        .to_dict()
    )

    priority_distribution = (
        df["priority_tier"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "phase": "Phase 4",
        "methodology": (
            "ML prediction -> expected risk severity -> "
            "business exposure -> priority score -> "
            "procurement watchlist"
        ),

        "input_records": int(len(df)),
        "unique_suppliers": int(df["supplier_id"].nunique()),

        "month_N_periods": int(
            df["period_N"].nunique()
        ),

        "latest_month_N": latest_period,

        "predicted_risk_distribution": {
            str(k): int(v)
            for k, v in predicted_distribution.items()
        },

        "priority_distribution": {
            str(k): int(v)
            for k, v in priority_distribution.items()
        },

        "high_risk_predictions": int(
            df["high_risk_prediction_flag"].sum()
        ),

        "high_probability_high_risk_predictions": int(
            df["high_probability_high_risk_flag"].sum()
        ),

        "critical_exposure_flags": int(
            df["critical_exposure_flag"].sum()
        ),

        "watchlist_records": int(
            df["watchlist_flag"].sum()
        ),

        "priority_score": {
            "formula": (
                "expected_risk_severity * "
                "business_exposure * 100"
            ),
            "business_exposure_formula": (
                "0.70 * spend_share + "
                "0.30 * critical_component_exposure"
            ),
            "high_priority_threshold": HIGH_PRIORITY_THRESHOLD,
            "medium_priority_threshold": MEDIUM_PRIORITY_THRESHOLD,
        },

        "risk_severity_mapping": RISK_SEVERITY,

        "high_probability_threshold": (
            HIGH_PROBABILITY_THRESHOLD
        ),

        "outputs": {
            "all_periods": str(ALL_PERIODS_FILE),
            "latest": str(LATEST_FILE),
            "watchlist": str(WATCHLIST_FILE),
            "supplier_summary": str(SUPPLIER_SUMMARY_FILE),
        },

        "validation": {
            "temporal_validation": True,
            "risk_validation": True,
            "probability_validation": True,
            "exposure_validation": True,
            "priority_validation": True,
        },
    }

    with open(
        SUMMARY_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        f"Summary saved: "
        f"{SUMMARY_JSON_FILE}"
    )

    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_phase_4():

    print(
        "======================================================================"
    )
    print(
        "PHASE 4: EXPOSURE-WEIGHTED SUPPLIER RISK PRIORITIZATION"
    )
    print(
        "======================================================================"
    )

    # ----------------------------------------------------------------------
    # 1. LOAD
    # ----------------------------------------------------------------------
    df = load_predictions()

    # ----------------------------------------------------------------------
    # 2. VALIDATE INPUT
    # ----------------------------------------------------------------------
    validate_predictions(df)

    # ----------------------------------------------------------------------
    # 3. STANDARDIZE RISK COLUMN NAMES INTERNALLY
    # ----------------------------------------------------------------------
    df = standardize_prediction_columns(df)

    # ----------------------------------------------------------------------
    # 4. TEMPORAL VALIDATION
    # ----------------------------------------------------------------------
    validate_temporal_structure(df)

    # ----------------------------------------------------------------------
    # 5. RISK CATEGORY VALIDATION
    # ----------------------------------------------------------------------
    validate_risk_categories(df)

    # ----------------------------------------------------------------------
    # 6. PROBABILITY VALIDATION
    # ----------------------------------------------------------------------
    validate_probabilities(df)

    # ----------------------------------------------------------------------
    # 7. EXPOSURE VALIDATION
    # ----------------------------------------------------------------------
    validate_exposure_fields(df)

    # ----------------------------------------------------------------------
    # 8. EXPECTED RISK
    # ----------------------------------------------------------------------
    df = calculate_expected_risk(df)

    # ----------------------------------------------------------------------
    # 9. BUSINESS EXPOSURE
    # ----------------------------------------------------------------------
    df = calculate_business_exposure(df)

    # ----------------------------------------------------------------------
    # 10. PRIORITY SCORE
    # ----------------------------------------------------------------------
    df = calculate_priority_score(df)

    # ----------------------------------------------------------------------
    # 11. PRIORITY TIER
    # ----------------------------------------------------------------------
    df = assign_priority_tiers(df)

    # ----------------------------------------------------------------------
    # 12. WATCHLIST FLAGS
    # ----------------------------------------------------------------------
    #
    # IMPORTANT:
    # These flags are created BEFORE create_supplier_summary().
    # This fixes the previous KeyError.
    #
    df = create_watchlist_flags(df)

    # ----------------------------------------------------------------------
    # 13. LATEST VIEW
    # ----------------------------------------------------------------------
    latest = create_latest_view(df)

    # ----------------------------------------------------------------------
    # 14. WATCHLIST
    # ----------------------------------------------------------------------
    watchlist = (
        df[df["watchlist_flag"] == 1]
        .copy()
        .sort_values(
            by=[
                "priority_score",
                "probability_high",
                "business_exposure",
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------------------
    # 15. SUPPLIER SUMMARY
    # ----------------------------------------------------------------------
    supplier_summary = create_supplier_summary(df)

    # ----------------------------------------------------------------------
    # 16. SAVE
    # ----------------------------------------------------------------------
    save_outputs(
        df,
        latest,
        watchlist,
        supplier_summary
    )

    # ----------------------------------------------------------------------
    # 17. SUMMARY
    # ----------------------------------------------------------------------
    create_summary_report(
        df,
        latest,
        watchlist,
        supplier_summary
    )

    # ----------------------------------------------------------------------
    # 18. VALIDATION
    # ----------------------------------------------------------------------
    validate_outputs(
        df,
        latest,
        watchlist,
        supplier_summary
    )

    # ----------------------------------------------------------------------
    # 19. FINAL CONSOLE SUMMARY
    # ----------------------------------------------------------------------
    print(
        "\n======================================================================"
    )
    print(
        "PHASE 4 COMPLETE"
    )
    print(
        "======================================================================"
    )

    print(
        f"Prediction records processed: {len(df):,}"
    )

    print(
        f"Unique suppliers: {df['supplier_id'].nunique():,}"
    )

    print(
        f"Latest prediction period: "
        f"{latest['period_N'].iloc[0]}"
    )

    print("\nLatest predicted risk distribution:")

    print(
        latest["predicted_risk_category"]
        .value_counts()
        .to_string()
    )

    print("\nLatest priority distribution:")

    print(
        latest["priority_tier"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 latest supplier priorities:")

    display_columns = [
        "supplier_id",
        "period_N",
        "period_N1",
        "predicted_risk_category",
        "probability_high",
        "spend_share",
        "critical_component_exposure",
        "expected_risk_severity",
        "business_exposure",
        "priority_score",
        "priority_tier",
    ]

    available_display_columns = [
        c for c in display_columns
        if c in latest.columns
    ]

    print(
        latest[
            available_display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nOutputs:")
    print(f"  all_periods      : {ALL_PERIODS_FILE}")
    print(f"  latest           : {LATEST_FILE}")
    print(f"  watchlist        : {WATCHLIST_FILE}")
    print(f"  supplier_summary : {SUPPLIER_SUMMARY_FILE}")
    print(f"  summary          : {SUMMARY_JSON_FILE}")

    print("\nPhase 4 methodology:")
    print(
        "  ML prediction -> expected risk severity -> "
        "business exposure -> priority score -> "
        "procurement watchlist"
    )

    print("\nDONE.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_phase_4()
