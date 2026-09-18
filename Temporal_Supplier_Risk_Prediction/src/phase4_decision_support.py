"""
PHASE 4 - SUPPLIER RISK PRIORITISATION

Purpose
-------
Convert Phase 3 model predictions into an operational supplier
prioritisation table.

IMPORTANT
---------
This script:
1. Uses ONLY information available at Month N for operational scoring.
2. Does NOT use actual_risk_N1 for prioritisation.
3. Does NOT use prediction_correct for prioritisation.
4. Keeps ML probability_high separate from the business priority score.
5. Uses the latest available Month-N observation for each supplier.
6. Produces CSV outputs and validation reports.

Input
-----
outputs/predictions/selected_model_predictions.csv

Outputs
-------
outputs/
    predictions/
        supplier_risk_prioritization.csv
        high_risk_suppliers.csv

    reports/
        phase4_risk_summary.csv
        phase4_priority_distribution.csv
        phase4_validation.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "selected_model_predictions.csv"
)

OUTPUT_PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

OUTPUT_REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

OUTPUT_PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# CONFIGURATION
# ================================================================

SUPPLIER_COLUMN = "supplier_id"
PERIOD_COLUMN = "period"

# Model prediction
PREDICTED_RISK_COLUMN = "predicted_risk_N1"
PROBABILITY_HIGH_COLUMN = "probability_high"

# Current Month-N operational information
CURRENT_RISK_COLUMN = "current_risk_N"

LATE_RATE_COLUMN = "late_delivery_rate"
AVG_DAYS_LATE_COLUMN = "avg_days_late"
SHORT_RATE_COLUMN = "short_delivery_rate"

QUALITY_EVENT_RATE_COLUMN = "quality_event_rate"
REWORK_RATIO_COLUMN = "rework_cost_ratio"
SUPPLIER_FAULT_COLUMN = "supplier_fault_rate"

CRITICAL_EXPOSURE_COLUMN = (
    "critical_component_exposure"
)

SPEND_SHARE_COLUMN = "spend_share"

ROUTE_DEVIATION_COLUMN = (
    "route_deviation_flag"
)

TRANSIT_DELAY_COLUMN = (
    "transit_delay_rate"
)

WORSENING_COLUMN = "is_worsening"

RISK_STREAK_COLUMN = (
    "risk_worsening_streak"
)

TREND_AVAILABLE_COLUMN = (
    "has_trend_data"
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

REQUIRED_COLUMNS = [
    SUPPLIER_COLUMN,
    PERIOD_COLUMN,

    PREDICTED_RISK_COLUMN,
    PROBABILITY_HIGH_COLUMN,

    CURRENT_RISK_COLUMN,

    LATE_RATE_COLUMN,
    AVG_DAYS_LATE_COLUMN,
    SHORT_RATE_COLUMN,

    QUALITY_EVENT_RATE_COLUMN,
    REWORK_RATIO_COLUMN,
    SUPPLIER_FAULT_COLUMN,

    CRITICAL_EXPOSURE_COLUMN,
    SPEND_SHARE_COLUMN,

    ROUTE_DEVIATION_COLUMN,
    TRANSIT_DELAY_COLUMN,

    WORSENING_COLUMN,
    RISK_STREAK_COLUMN,
    TREND_AVAILABLE_COLUMN,
]


# ================================================================
# VALIDATION
# ================================================================

def validate_input(df):
    print("\n" + "=" * 70)
    print("PHASE 4 INPUT VALIDATION")
    print("=" * 70)

    missing = [
        column
        for column in REQUIRED_COLUMNS
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

    print(
        f"Required columns: PASSED "
        f"({len(REQUIRED_COLUMNS)} checked)"
    )

    if df.empty:
        raise ValueError(
            "Input file contains zero rows."
        )

    print(
        f"Input rows: {len(df):,}"
    )

    print(
        f"Input suppliers: "
        f"{df[SUPPLIER_COLUMN].nunique():,}"
    )

    print(
        f"Input periods: "
        f"{df[PERIOD_COLUMN].nunique():,}"
    )

    # ------------------------------------------------------------
    # Forbidden future/outcome information
    # ------------------------------------------------------------

    forbidden_columns = [
        "actual_risk_N1",
        "prediction_correct",
        "target_risk_N1",
        "target_risk_score_N1",
        "target_low_boundary",
        "target_high_boundary",
    ]

    present_forbidden = [
        column
        for column in forbidden_columns
        if column in df.columns
    ]

    print(
        "\nFuture/outcome columns detected in source:"
    )

    if present_forbidden:
        for column in present_forbidden:
            print(f" - {column}")

        print(
            "\nThese columns are intentionally excluded "
            "from the Phase 4 operational score."
        )
    else:
        print(
            "None detected."
        )

    return True


# ================================================================
# NUMERIC CLEANING
# ================================================================

def clean_numeric_columns(df):
    numeric_columns = [
        PROBABILITY_HIGH_COLUMN,
        LATE_RATE_COLUMN,
        AVG_DAYS_LATE_COLUMN,
        SHORT_RATE_COLUMN,
        QUALITY_EVENT_RATE_COLUMN,
        REWORK_RATIO_COLUMN,
        SUPPLIER_FAULT_COLUMN,
        CRITICAL_EXPOSURE_COLUMN,
        SPEND_SHARE_COLUMN,
        ROUTE_DEVIATION_COLUMN,
        TRANSIT_DELAY_COLUMN,
        WORSENING_COLUMN,
        RISK_STREAK_COLUMN,
        TREND_AVAILABLE_COLUMN,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ================================================================
# NORMALISATION HELPERS
# ================================================================

def min_max_score(series):
    """
    Convert a numeric series to 0-100.

    Constant series receive 0 rather than artificially
    assigning everybody 100.
    """

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            0.0,
            index=series.index,
        )

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100.0
    )


def clip_percentage(series):
    """
    Convert a ratio/proportion into 0-100 while
    protecting against malformed values.
    """

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .clip(
            lower=0,
            upper=1,
        )
        * 100
    )


# ================================================================
# CREATE LATEST SUPPLIER SNAPSHOT
# ================================================================

def create_latest_supplier_snapshot(df):
    print("\n" + "=" * 70)
    print("LATEST SUPPLIER SNAPSHOT")
    print("=" * 70)

    working = df.copy()

    working["_period_sort"] = pd.to_datetime(
        working[PERIOD_COLUMN].astype(str),
        errors="coerce",
    )

    if working["_period_sort"].isna().all():
        raise ValueError(
            "Could not parse period column."
        )

    working = working.sort_values(
        [
            SUPPLIER_COLUMN,
            "_period_sort",
        ]
    )

    latest = (
        working
        .groupby(
            SUPPLIER_COLUMN,
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest = latest.drop(
        columns=["_period_sort"],
        errors="ignore",
    )

    latest = latest.reset_index(
        drop=True
    )

    print(
        f"Unique suppliers: "
        f"{latest[SUPPLIER_COLUMN].nunique():,}"
    )

    print(
        f"Latest snapshot rows: "
        f"{len(latest):,}"
    )

    if len(latest) != latest[
        SUPPLIER_COLUMN
    ].nunique():
        raise ValueError(
            "Latest supplier snapshot contains "
            "duplicate suppliers."
        )

    print(
        "One latest observation per supplier: PASSED"
    )

    return latest


# ================================================================
# OPERATIONAL RISK COMPONENTS
# ================================================================

def create_risk_components(df):
    print("\n" + "=" * 70)
    print("OPERATIONAL RISK COMPONENTS")
    print("=" * 70)

    result = df.copy()

    # ------------------------------------------------------------
    # 1. Delivery risk
    # ------------------------------------------------------------
    #
    # Late delivery rate:
    # strongest direct delivery signal.
    #
    # Average days late:
    # severity of delay.
    #
    # Short delivery rate:
    # fulfilment quantity issue.
    #
    # Transit delay:
    # logistics-related warning.
    #
    # Route deviation:
    # binary warning.
    # ------------------------------------------------------------

    late_score = min_max_score(
        result[LATE_RATE_COLUMN]
    )

    avg_delay_score = min_max_score(
        result[AVG_DAYS_LATE_COLUMN]
    )

    short_score = min_max_score(
        result[SHORT_RATE_COLUMN]
    )

    transit_score = min_max_score(
        result[TRANSIT_DELAY_COLUMN]
    )

    route_score = (
        result[ROUTE_DEVIATION_COLUMN]
        .fillna(0)
        .clip(0, 1)
        * 100
    )

    result["delivery_risk_score"] = (
        0.35 * late_score
        + 0.25 * avg_delay_score
        + 0.15 * short_score
        + 0.15 * transit_score
        + 0.10 * route_score
    )

    # ------------------------------------------------------------
    # 2. Quality risk
    # ------------------------------------------------------------

    quality_score = min_max_score(
        result[QUALITY_EVENT_RATE_COLUMN]
    )

    rework_score = min_max_score(
        result[REWORK_RATIO_COLUMN]
    )

    supplier_fault_score = min_max_score(
        result[SUPPLIER_FAULT_COLUMN]
    )

    result["quality_risk_score"] = (
        0.45 * quality_score
        + 0.30 * rework_score
        + 0.25 * supplier_fault_score
    )

    # ------------------------------------------------------------
    # 3. Strategic exposure
    # ------------------------------------------------------------

    critical_score = clip_percentage(
        result[CRITICAL_EXPOSURE_COLUMN]
    )

    spend_score = clip_percentage(
        result[SPEND_SHARE_COLUMN]
    )

    result["strategic_exposure_score"] = (
        0.60 * critical_score
        + 0.40 * spend_score
    )

    # ------------------------------------------------------------
    # 4. Trend / deterioration
    # ------------------------------------------------------------

    worsening_score = (
        result[WORSENING_COLUMN]
        .fillna(0)
        .clip(0, 1)
        * 100
    )

    streak_score = min_max_score(
        result[RISK_STREAK_COLUMN]
    )

    result["deterioration_score"] = (
        0.60 * worsening_score
        + 0.40 * streak_score
    )

    # ------------------------------------------------------------
    # 5. Model risk
    # ------------------------------------------------------------

    result["model_high_risk_score"] = (
        result[PROBABILITY_HIGH_COLUMN]
        .fillna(0)
        .clip(0, 1)
        * 100
    )

    print(
        "Delivery risk component: PASSED"
    )

    print(
        "Quality risk component: PASSED"
    )

    print(
        "Strategic exposure component: PASSED"
    )

    print(
        "Deterioration component: PASSED"
    )

    print(
        "Model high-risk probability component: PASSED"
    )

    return result


# ================================================================
# PRIORITY SCORE
# ================================================================

def create_priority_score(df):
    print("\n" + "=" * 70)
    print("SUPPLIER PRIORITY SCORE")
    print("=" * 70)

    result = df.copy()

    """
    Business priority score.

    Components:

    30% model high-risk probability
    25% delivery risk
    20% quality risk
    15% strategic exposure
    10% deterioration

    This is NOT an ML probability.

    It is a transparent operational prioritisation score.
    """

    result["priority_score"] = (
        0.30
        * result["model_high_risk_score"]

        + 0.25
        * result["delivery_risk_score"]

        + 0.20
        * result["quality_risk_score"]

        + 0.15
        * result["strategic_exposure_score"]

        + 0.10
        * result["deterioration_score"]
    )

    result["priority_score"] = (
        result["priority_score"]
        .clip(0, 100)
        .round(2)
    )

    # ------------------------------------------------------------
    # Priority bands
    # ------------------------------------------------------------

    result["priority_band"] = pd.cut(
        result["priority_score"],
        bins=[
            -np.inf,
            25,
            50,
            75,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
        right=False,
    )

    print(
        "Priority score created: PASSED"
    )

    print(
        "\nPriority distribution:"
    )

    print(
        result[
            "priority_band"
        ]
        .value_counts(
            sort=False
        )
    )

    return result


# ================================================================
# ACTION RECOMMENDATION
# ================================================================

def create_action_recommendation(df):
    result = df.copy()

    def recommendation(row):

        predicted_risk = str(
            row[PREDICTED_RISK_COLUMN]
        ).strip()

        priority = float(
            row["priority_score"]
        )

        delivery = float(
            row["delivery_risk_score"]
        )

        quality = float(
            row["quality_risk_score"]
        )

        strategic = float(
            row["strategic_exposure_score"]
        )

        deterioration = float(
            row["deterioration_score"]
        )

        # --------------------------------------------------------
        # Critical
        # --------------------------------------------------------

        if priority >= 75:

            if delivery >= 65:
                return (
                    "Immediate supplier review - "
                    "focus on delivery performance"
                )

            if quality >= 65:
                return (
                    "Immediate supplier review - "
                    "focus on quality and rework"
                )

            if strategic >= 65:
                return (
                    "Immediate supplier review - "
                    "focus on critical component exposure"
                )

            if deterioration >= 65:
                return (
                    "Immediate supplier review - "
                    "supplier deterioration detected"
                )

            return (
                "Immediate supplier review"
            )

        # --------------------------------------------------------
        # High
        # --------------------------------------------------------

        if priority >= 50:

            if delivery >= 65:
                return (
                    "Corrective action - "
                    "delivery monitoring"
                )

            if quality >= 65:
                return (
                    "Corrective action - "
                    "quality monitoring"
                )

            if strategic >= 65:
                return (
                    "Enhanced monitoring - "
                    "strategic exposure"
                )

            return (
                "Enhanced supplier monitoring"
            )

        # --------------------------------------------------------
        # Moderate
        # --------------------------------------------------------

        if priority >= 25:

            if deterioration >= 60:
                return (
                    "Monitor closely - "
                    "negative trend"
                )

            return (
                "Routine monitoring"
            )

        # --------------------------------------------------------
        # Low
        # --------------------------------------------------------

        return (
            "Routine monitoring"
        )

    result["recommended_action"] = (
        result.apply(
            recommendation,
            axis=1,
        )
    )

    return result


# ================================================================
# ADD RISK FLAGS
# ================================================================

def add_risk_flags(df):
    result = df.copy()

    result["model_high_risk_flag"] = (
        result[PREDICTED_RISK_COLUMN]
        .astype(str)
        .str.strip()
        .eq("High")
        .astype(int)
    )

    result["high_probability_flag"] = (
        result[PROBABILITY_HIGH_COLUMN]
        .fillna(0)
        .ge(0.40)
        .astype(int)
    )

    result["delivery_problem_flag"] = (
        result["delivery_risk_score"]
        .ge(60)
        .astype(int)
    )

    result["quality_problem_flag"] = (
        result["quality_risk_score"]
        .ge(60)
        .astype(int)
    )

    result["strategic_exposure_flag"] = (
        result["strategic_exposure_score"]
        .ge(60)
        .astype(int)
    )

    result["deterioration_flag"] = (
        result["deterioration_score"]
        .ge(60)
        .astype(int)
    )

    result["requires_attention"] = (
        (
            result["priority_score"]
            >= 50
        )
        | (
            result["model_high_risk_flag"]
            == 1
        )
    ).astype(int)

    return result


# ================================================================
# FINAL COLUMN ORDER
# ================================================================

def select_output_columns(df):

    columns = [
        # Identity
        SUPPLIER_COLUMN,
        PERIOD_COLUMN,

        # Model information
        PREDICTED_RISK_COLUMN,
        PROBABILITY_HIGH_COLUMN,

        # Current risk
        CURRENT_RISK_COLUMN,

        # Core operational metrics
        LATE_RATE_COLUMN,
        AVG_DAYS_LATE_COLUMN,
        SHORT_RATE_COLUMN,
        QUALITY_EVENT_RATE_COLUMN,
        REWORK_RATIO_COLUMN,
        SUPPLIER_FAULT_COLUMN,
        CRITICAL_EXPOSURE_COLUMN,
        SPEND_SHARE_COLUMN,
        ROUTE_DEVIATION_COLUMN,
        TRANSIT_DELAY_COLUMN,

        # Trends
        WORSENING_COLUMN,
        RISK_STREAK_COLUMN,
        TREND_AVAILABLE_COLUMN,

        # Component scores
        "model_high_risk_score",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",

        # Final decision score
        "priority_score",
        "priority_band",

        # Flags
        "model_high_risk_flag",
        "high_probability_flag",
        "delivery_problem_flag",
        "quality_problem_flag",
        "strategic_exposure_flag",
        "deterioration_flag",
        "requires_attention",

        # Recommendation
        "recommended_action",
    ]

    existing = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[existing].copy()


# ================================================================
# SAVE OUTPUTS
# ================================================================

def save_outputs(df):
    print("\n" + "=" * 70)
    print("SAVING PHASE 4 OUTPUTS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Complete prioritisation table
    # ------------------------------------------------------------

    prioritization_path = (
        OUTPUT_PREDICTION_DIR
        / "supplier_risk_prioritization.csv"
    )

    df.to_csv(
        prioritization_path,
        index=False,
    )

    print(
        "Supplier prioritisation saved:"
    )

    print(
        prioritization_path
    )

    # ------------------------------------------------------------
    # High-priority suppliers
    # ------------------------------------------------------------

    high_risk = (
        df[
            df["priority_score"]
            >= 50
        ]
        .sort_values(
            "priority_score",
            ascending=False,
        )
        .copy()
    )

    high_risk_path = (
        OUTPUT_PREDICTION_DIR
        / "high_risk_suppliers.csv"
    )

    high_risk.to_csv(
        high_risk_path,
        index=False,
    )

    print(
        "\nHigh-priority suppliers saved:"
    )

    print(
        high_risk_path
    )

    print(
        f"High-priority supplier count: "
        f"{len(high_risk):,}"
    )

    # ------------------------------------------------------------
    # Risk summary
    # ------------------------------------------------------------

    risk_summary = (
        df.groupby(
            "priority_band",
            observed=False,
        )
        .agg(
            supplier_count=(
                SUPPLIER_COLUMN,
                "nunique",
            ),
            average_priority_score=(
                "priority_score",
                "mean",
            ),
            average_probability_high=(
                PROBABILITY_HIGH_COLUMN,
                "mean",
            ),
            average_delivery_risk=(
                "delivery_risk_score",
                "mean",
            ),
            average_quality_risk=(
                "quality_risk_score",
                "mean",
            ),
        )
        .reset_index()
    )

    risk_summary[
        "average_priority_score"
    ] = risk_summary[
        "average_priority_score"
    ].round(2)

    risk_summary[
        "average_probability_high"
    ] = risk_summary[
        "average_probability_high"
    ].round(4)

    risk_summary[
        "average_delivery_risk"
    ] = risk_summary[
        "average_delivery_risk"
    ].round(2)

    risk_summary[
        "average_quality_risk"
    ] = risk_summary[
        "average_quality_risk"
    ].round(2)

    risk_summary_path = (
        OUTPUT_REPORT_DIR
        / "phase4_risk_summary.csv"
    )

    risk_summary.to_csv(
        risk_summary_path,
        index=False,
    )

    print(
        "\nRisk summary saved:"
    )

    print(
        risk_summary_path
    )

    # ------------------------------------------------------------
    # Priority distribution
    # ------------------------------------------------------------

    priority_distribution = (
        df["priority_band"]
        .value_counts(
            sort=False
        )
        .rename_axis(
            "priority_band"
        )
        .reset_index(
            name="supplier_count"
        )
    )

    priority_distribution[
        "percentage"
    ] = (
        priority_distribution[
            "supplier_count"
        ]
        / len(df)
        * 100
    ).round(2)

    distribution_path = (
        OUTPUT_REPORT_DIR
        / "phase4_priority_distribution.csv"
    )

    priority_distribution.to_csv(
        distribution_path,
        index=False,
    )

    print(
        "Priority distribution saved:"
    )

    print(
        distribution_path
    )

    return (
        prioritization_path,
        high_risk_path,
        risk_summary_path,
        distribution_path,
    )


# ================================================================
# VALIDATION
# ================================================================

def validate_phase4_output(df):
    print("\n" + "=" * 70)
    print("PHASE 4 VALIDATION")
    print("=" * 70)

    checks = []

    # ------------------------------------------------------------
    # Supplier uniqueness
    # ------------------------------------------------------------

    supplier_unique = (
        df[SUPPLIER_COLUMN]
        .is_unique
    )

    checks.append(
        (
            "One row per supplier",
            supplier_unique,
        )
    )

    # ------------------------------------------------------------
    # Priority score range
    # ------------------------------------------------------------

    score_valid = (
        df["priority_score"]
        .between(0, 100)
        .all()
    )

    checks.append(
        (
            "Priority score range 0-100",
            score_valid,
        )
    )

    # ------------------------------------------------------------
    # Required score components
    # ------------------------------------------------------------

    component_columns = [
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "model_high_risk_score",
    ]

    for column in component_columns:

        valid = (
            df[column]
            .between(0, 100)
            .all()
        )

        checks.append(
            (
                f"{column} range 0-100",
                valid,
            )
        )

    # ------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------

    probability_valid = (
        df[PROBABILITY_HIGH_COLUMN]
        .between(0, 1)
        .all()
    )

    checks.append(
        (
            "Model high-risk probability range 0-1",
            probability_valid,
        )
    )

    # ------------------------------------------------------------
    # No future/outcome columns in output
    # ------------------------------------------------------------

    forbidden_output_columns = [
        "actual_risk_N1",
        "prediction_correct",
        "target_risk_N1",
        "target_risk_score_N1",
        "target_low_boundary",
        "target_high_boundary",
    ]

    future_columns_present = [
        column
        for column in forbidden_output_columns
        if column in df.columns
    ]

    no_future_columns = (
        len(future_columns_present) == 0
    )

    checks.append(
        (
            "No future/outcome variables used in operational output",
            no_future_columns,
        )
    )

    # ------------------------------------------------------------
    # Check score calculation
    # ------------------------------------------------------------

    expected_score = (
        0.30
        * df["model_high_risk_score"]

        + 0.25
        * df["delivery_risk_score"]

        + 0.20
        * df["quality_risk_score"]

        + 0.15
        * df["strategic_exposure_score"]

        + 0.10
        * df["deterioration_score"]
    )

    score_formula_valid = np.allclose(
        df["priority_score"],
        expected_score.round(2),
        atol=0.01,
    )

    checks.append(
        (
            "Priority score formula validated",
            score_formula_valid,
        )
    )

    # ------------------------------------------------------------
    # Print checks
    # ------------------------------------------------------------

    all_passed = True

    for name, passed in checks:

        if passed:
            print(
                f"✓ {name}"
            )
        else:
            print(
                f"✗ {name}"
            )
            all_passed = False

    if not all_passed:

        raise ValueError(
            "PHASE 4 VALIDATION FAILED."
        )

    print(
        "\nAll Phase 4 validation checks PASSED."
    )

    return True


# ================================================================
# VALIDATION REPORT
# ================================================================

def save_validation_report(df):
    report_path = (
        OUTPUT_REPORT_DIR
        / "phase4_validation.txt"
    )

    lines = []

    lines.append(
        "PHASE 4 VALIDATION REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Input file: {INPUT_FILE}"
    )

    lines.append(
        f"Supplier count: "
        f"{df[SUPPLIER_COLUMN].nunique():,}"
    )

    lines.append(
        f"Latest snapshot period range: "
        f"{df[PERIOD_COLUMN].min()} -> "
        f"{df[PERIOD_COLUMN].max()}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Operational score components:"
    )

    lines.append(
        "30% model high-risk probability"
    )

    lines.append(
        "25% delivery risk"
    )

    lines.append(
        "20% quality risk"
    )

    lines.append(
        "15% strategic exposure"
    )

    lines.append(
        "10% deterioration"
    )

    lines.append(
        ""
    )

    lines.append(
        "Future/outcome variables excluded:"
    )

    lines.append(
        "actual_risk_N1"
    )

    lines.append(
        "prediction_correct"
    )

    lines.append(
        "target_risk_N1"
    )

    lines.append(
        "target_risk_score_N1"
    )

    lines.append(
        "target_low_boundary"
    )

    lines.append(
        "target_high_boundary"
    )

    lines.append(
        ""
    )

    lines.append(
        "PHASE 4 STATUS: PASSED"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        "\nValidation report saved:"
    )

    print(
        report_path
    )


# ================================================================
# MAIN
# ================================================================

def run_phase4():

    print("\n")
    print("=" * 70)
    print("PHASE 4: SUPPLIER RISK PRIORITISATION")
    print("=" * 70)

    print(
        f"\nInput file:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file does not exist:\n"
            f"{INPUT_FILE}\n\n"
            "Run Phase 3 first."
        )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input shape: {df.shape}"
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validate_input(
        df
    )

    # ------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------

    df = clean_numeric_columns(
        df
    )

    # ------------------------------------------------------------
    # Latest supplier snapshot
    # ------------------------------------------------------------

    df = create_latest_supplier_snapshot(
        df
    )

    # ------------------------------------------------------------
    # Risk components
    # ------------------------------------------------------------

    df = create_risk_components(
        df
    )

    # ------------------------------------------------------------
    # Priority score
    # ------------------------------------------------------------

    df = create_priority_score(
        df
    )

    # ------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------

    df = add_risk_flags(
        df
    )

    # ------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------

    df = create_action_recommendation(
        df
    )

    # ------------------------------------------------------------
    # Final columns
    # ------------------------------------------------------------

    df = select_output_columns(
        df
    )

    # ------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------

    df = df.sort_values(
        [
            "priority_score",
            PROBABILITY_HIGH_COLUMN,
        ],
        ascending=[
            False,
            False,
        ],
    )

    df = df.reset_index(
        drop=True
    )

    # ------------------------------------------------------------
    # Validate final output
    # ------------------------------------------------------------

    validate_phase4_output(
        df
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    save_outputs(
        df
    )

    save_validation_report(
        df
    )

    # ------------------------------------------------------------
    # Display top suppliers
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP PRIORITY SUPPLIERS")
    print("=" * 70)

    display_columns = [
        SUPPLIER_COLUMN,
        PERIOD_COLUMN,
        PREDICTED_RISK_COLUMN,
        PROBABILITY_HIGH_COLUMN,
        "priority_score",
        "priority_band",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "recommended_action",
    ]

    print(
        df[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    print(
        f"Suppliers prioritised: "
        f"{len(df):,}"
    )

    print(
        "✓ Latest supplier snapshot created"
    )

    print(
        "✓ Delivery risk calculated"
    )

    print(
        "✓ Quality risk calculated"
    )

    print(
        "✓ Strategic exposure calculated"
    )

    print(
        "✓ Deterioration score calculated"
    )

    print(
        "✓ Business priority score calculated"
    )

    print(
        "✓ Risk bands created"
    )

    print(
        "✓ Recommended actions generated"
    )

    print(
        "✓ Future target information excluded"
    )

    print(
        "✓ Validation passed"
    )

    print(
        "\nNext implementation stage:"
    )

    print(
        "PHASE 5 - MODEL EXPLAINABILITY"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_phase4()