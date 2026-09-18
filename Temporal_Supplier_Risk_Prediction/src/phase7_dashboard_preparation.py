# ================================================================
# PHASE 7: DECISION-SUPPORT DASHBOARD DATA PREPARATION
# ================================================================

from pathlib import Path
import json
import pandas as pd
import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
INTERVENTIONS_DIR = ROOT / "outputs" / "interventions"
EXPLAINABILITY_DIR = ROOT / "outputs" / "explainability"
REPORT_DIR = ROOT / "outputs" / "reports"

DASHBOARD_DIR = ROOT / "outputs" / "dashboard"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# INPUT FILES
# ================================================================

PRIORITY_FILE = (
    PREDICTIONS_DIR /
    "supplier_risk_prioritization.csv"
)

INTERVENTION_FILE = (
    INTERVENTIONS_DIR /
    "supplier_intervention_recommendations_explained.csv"
)

REASON_FILE = (
    INTERVENTIONS_DIR /
    "supplier_risk_reasons.csv"
)

OOT_FILE = (
    PREDICTIONS_DIR /
    "out_of_time_predictions.csv"
)

BENCHMARK_FILE = (
    REPORT_DIR /
    "final_oot_model_benchmark.csv"
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


def require_columns(df, columns, name):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(f" - {c}" for c in missing)
        )


def numeric(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ================================================================
# LOAD INPUTS
# ================================================================

def load_inputs():

    print_header("PHASE 7 INPUT LOADING")

    for path in [
        PRIORITY_FILE,
        INTERVENTION_FILE,
        REASON_FILE,
        OOT_FILE,
        BENCHMARK_FILE,
    ]:
        require_file(path)
        print(f"Found: {path}")

    priority = pd.read_csv(PRIORITY_FILE)
    intervention = pd.read_csv(INTERVENTION_FILE)
    reasons = pd.read_csv(REASON_FILE)
    oot = pd.read_csv(OOT_FILE)
    benchmark = pd.read_csv(BENCHMARK_FILE)

    print()
    print("Priority shape:", priority.shape)
    print("Intervention shape:", intervention.shape)
    print("Reason shape:", reasons.shape)
    print("OOT shape:", oot.shape)
    print("Benchmark shape:", benchmark.shape)

    return (
        priority,
        intervention,
        reasons,
        oot,
        benchmark,
    )


# ================================================================
# VALIDATE PRIORITY DATA
# ================================================================

def validate_priority(priority):

    print_header("PHASE 7 PRIORITY DATA VALIDATION")

    required = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
        "priority_score",
        "priority_band",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "model_high_risk_score",
    ]

    require_columns(
        priority,
        required,
        "supplier_risk_prioritization.csv"
    )

    if priority["supplier_id"].duplicated().any():
        raise ValueError(
            "Priority dataset must contain exactly "
            "one latest row per supplier."
        )

    score_columns = [
        "priority_score",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "model_high_risk_score",
    ]

    for column in score_columns:

        values = numeric(
            priority,
            column
        )

        if values.isna().any():
            raise ValueError(
                f"{column} contains non-numeric values."
            )

        if (values < 0).any() or (values > 100).any():
            raise ValueError(
                f"{column} contains values outside 0-100."
            )

    probability = numeric(
        priority,
        "probability_high"
    )

    if probability.isna().any():
        raise ValueError(
            "probability_high contains invalid values."
        )

    if (probability < 0).any() or (probability > 1).any():
        raise ValueError(
            "probability_high must be between 0 and 1."
        )

    print("Required columns: PASSED")
    print("One row per supplier: PASSED")
    print("Score ranges: PASSED")
    print("Probability range: PASSED")


# ================================================================
# VALIDATE INTERVENTION DATA
# ================================================================

def validate_interventions(intervention):

    print_header("PHASE 7 INTERVENTION VALIDATION")

    required = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
        "priority_score",
        "priority_band",
        "intervention_level",
        "primary_risk_domain",
        "why_risky",
        "risk_evidence",
        "recommended_intervention",
    ]

    require_columns(
        intervention,
        required,
        "supplier_intervention_recommendations_explained.csv"
    )

    if intervention["supplier_id"].duplicated().any():
        raise ValueError(
            "Intervention data contains duplicate suppliers."
        )

    if (
        intervention["why_risky"]
        .fillna("")
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Every supplier must have why_risky."
        )

    if (
        intervention["risk_evidence"]
        .fillna("")
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Every supplier must have risk_evidence."
        )

    if (
        intervention["recommended_intervention"]
        .fillna("")
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            "Every supplier must have a recommendation."
        )

    print("Required columns: PASSED")
    print("One row per supplier: PASSED")
    print("why_risky coverage: PASSED")
    print("risk evidence coverage: PASSED")
    print("Recommendation coverage: PASSED")


# ================================================================
# FUTURE / OUTCOME LEAKAGE AUDIT
# ================================================================

def leakage_audit(df, dataset_name):

    forbidden = {
        "actual_risk_N1",
        "target_risk_N1",
        "target_risk_score_N1",
        "target_low_boundary",
        "target_high_boundary",
        "prediction_correct",
    }

    leaked = sorted(
        forbidden.intersection(df.columns)
    )

    if leaked:
        raise ValueError(
            f"{dataset_name} contains future/outcome columns:\n"
            + "\n".join(
                f" - {c}" for c in leaked
            )
        )


# ================================================================
# CREATE MAIN SUPPLIER DASHBOARD TABLE
# ================================================================

def create_supplier_dashboard(priority, intervention):

    print_header(
        "CREATING SUPPLIER DASHBOARD TABLE"
    )

    operational_columns = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
        "priority_score",
        "priority_band",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "model_high_risk_score",
    ]

    operational = priority[
        operational_columns
    ].copy()

    explanation_columns = [
        "supplier_id",
        "intervention_level",
        "primary_risk_domain",
        "risk_domains",
        "why_risky",
        "risk_evidence",
        "top_local_drivers",
        "positive_local_drivers",
        "opposing_local_drivers",
        "recommended_intervention",
    ]

    available = [
        c for c in explanation_columns
        if c in intervention.columns
    ]

    explanations = intervention[
        available
    ].copy()

    supplier_dashboard = operational.merge(
        explanations,
        on="supplier_id",
        how="left",
        validate="one_to_one",
    )

    if len(supplier_dashboard) != len(priority):
        raise ValueError(
            "Supplier dashboard merge changed row count."
        )

    if supplier_dashboard["supplier_id"].duplicated().any():
        raise ValueError(
            "Supplier dashboard contains duplicate suppliers."
        )

    leakage_audit(
        supplier_dashboard,
        "supplier_dashboard"
    )

    path = (
        DASHBOARD_DIR /
        "supplier_dashboard.csv"
    )

    supplier_dashboard.to_csv(
        path,
        index=False
    )

    print(
        f"Supplier dashboard saved:\n{path}"
    )

    return supplier_dashboard


# ================================================================
# CREATE TOP PRIORITY TABLE
# ================================================================

def create_priority_table(supplier_dashboard):

    print_header(
        "CREATING PRIORITY TABLE"
    )

    columns = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
        "priority_score",
        "priority_band",
        "intervention_level",
        "primary_risk_domain",
        "why_risky",
        "risk_evidence",
        "recommended_intervention",
    ]

    columns = [
        c for c in columns
        if c in supplier_dashboard.columns
    ]

    table = supplier_dashboard[
        columns
    ].copy()

    table = table.sort_values(
        "priority_score",
        ascending=False
    ).reset_index(drop=True)

    path = (
        DASHBOARD_DIR /
        "priority_supplier_table.csv"
    )

    table.to_csv(
        path,
        index=False
    )

    print(
        f"Priority table saved:\n{path}"
    )

    return table


# ================================================================
# CREATE RISK DOMAIN SUMMARY
# ================================================================

def create_domain_summary(supplier_dashboard):

    print_header(
        "CREATING RISK DOMAIN SUMMARY"
    )

    if "primary_risk_domain" not in supplier_dashboard.columns:
        raise ValueError(
            "primary_risk_domain missing."
        )

    summary = (
        supplier_dashboard
        .groupby(
            "primary_risk_domain",
            dropna=False
        )
        .agg(
            supplier_count=(
                "supplier_id",
                "nunique"
            ),
            average_priority_score=(
                "priority_score",
                "mean"
            ),
            average_delivery_risk=(
                "delivery_risk_score",
                "mean"
            ),
            average_quality_risk=(
                "quality_risk_score",
                "mean"
            ),
            average_deterioration=(
                "deterioration_score",
                "mean"
            ),
            average_model_high_risk_score=(
                "model_high_risk_score",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "average_priority_score"
    ] = summary[
        "average_priority_score"
    ].round(2)

    path = (
        DASHBOARD_DIR /
        "risk_domain_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"Risk domain summary saved:\n{path}"
    )

    return summary


# ================================================================
# CREATE PRIORITY BAND SUMMARY
# ================================================================

def create_priority_summary(supplier_dashboard):

    print_header(
        "CREATING PRIORITY BAND SUMMARY"
    )

    summary = (
        supplier_dashboard
        .groupby(
            "priority_band",
            dropna=False
        )
        .agg(
            supplier_count=(
                "supplier_id",
                "nunique"
            ),
            average_priority_score=(
                "priority_score",
                "mean"
            ),
            average_probability_high=(
                "probability_high",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "average_priority_score"
    ] = summary[
        "average_priority_score"
    ].round(2)

    summary[
        "average_probability_high"
    ] = summary[
        "average_probability_high"
    ].round(4)

    path = (
        DASHBOARD_DIR /
        "priority_band_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"Priority band summary saved:\n{path}"
    )

    return summary


# ================================================================
# CREATE INTERVENTION SUMMARY
# ================================================================

def create_intervention_summary(supplier_dashboard):

    print_header(
        "CREATING INTERVENTION SUMMARY"
    )

    summary = (
        supplier_dashboard
        .groupby(
            [
                "intervention_level",
                "primary_risk_domain",
            ],
            dropna=False
        )
        .agg(
            supplier_count=(
                "supplier_id",
                "nunique"
            ),
            average_priority_score=(
                "priority_score",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "average_priority_score"
    ] = summary[
        "average_priority_score"
    ].round(2)

    path = (
        DASHBOARD_DIR /
        "intervention_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"Intervention summary saved:\n{path}"
    )

    return summary


# ================================================================
# CREATE OOT TREND DATA
# ================================================================

def create_oot_trend_table(oot):

    print_header(
        "CREATING OOT TREND TABLE"
    )

    required = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
    ]

    require_columns(
        oot,
        required,
        "out_of_time_predictions.csv"
    )

    # Only operational/prediction fields.
    allowed = [
        "supplier_id",
        "period",
        "predicted_risk_N1",
        "probability_high",
        "priority_score",
        "priority_band",
        "delivery_risk_score",
        "quality_risk_score",
        "strategic_exposure_score",
        "deterioration_score",
        "model_high_risk_score",
    ]

    columns = [
        c for c in allowed
        if c in oot.columns
    ]

    trend = oot[
        columns
    ].copy()

    trend["period"] = trend[
        "period"
    ].astype(str)

    trend = trend.sort_values(
        ["period", "supplier_id"]
    ).reset_index(drop=True)

    leakage_audit(
        trend,
        "dashboard OOT trend table"
    )

    path = (
        DASHBOARD_DIR /
        "supplier_risk_trend.csv"
    )

    trend.to_csv(
        path,
        index=False
    )

    print(
        f"OOT trend data saved:\n{path}"
    )

    return trend


# ================================================================
# CREATE MODEL BENCHMARK TABLE
# ================================================================

def create_model_benchmark(benchmark):

    print_header(
        "CREATING MODEL BENCHMARK TABLE"
    )

    required = [
        "Model",
        "Accuracy",
        "Macro_F1",
        "High_Risk_Precision",
        "High_Risk_Recall",
        "High_Risk_F1",
        "ROC_AUC_OVR_Macro",
        "Penalty_Cost",
        "High_Risk_Threshold",
    ]

    require_columns(
        benchmark,
        required,
        "final_oot_model_benchmark.csv"
    )

    output = benchmark.copy()

    path = (
        DASHBOARD_DIR /
        "model_benchmark.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"Model benchmark saved:\n{path}"
    )

    return output


# ================================================================
# CREATE KPI SUMMARY
# ================================================================

def create_kpi_summary(
    supplier_dashboard,
    benchmark,
):

    print_header(
        "CREATING DASHBOARD KPI SUMMARY"
    )

    total_suppliers = (
        supplier_dashboard[
            "supplier_id"
        ].nunique()
    )

    high_priority = int(
        supplier_dashboard[
            "priority_band"
        ]
        .isin(["High", "Critical"])
        .sum()
    )

    moderate = int(
        (
            supplier_dashboard[
                "priority_band"
            ] == "Moderate"
        ).sum()
    )

    low = int(
        (
            supplier_dashboard[
                "priority_band"
            ] == "Low"
        ).sum()
    )

    immediate = 0

    if "intervention_level" in supplier_dashboard.columns:
        immediate = int(
            (
                supplier_dashboard[
                    "intervention_level"
                ] == "Immediate intervention"
            ).sum()
        )

    corrective = 0

    if "intervention_level" in supplier_dashboard.columns:
        corrective = int(
            (
                supplier_dashboard[
                    "intervention_level"
                ] == "Corrective action"
            ).sum()
        )

    selected_model = "Unknown"
    selected_accuracy = np.nan
    selected_high_precision = np.nan
    selected_high_recall = np.nan
    selected_high_f1 = np.nan
    selected_auc = np.nan

    if "Model" in benchmark.columns:

        preferred = benchmark[
            benchmark["Model"]
            .astype(str)
            .str.lower()
            .eq("logistic regression standard")
        ]

        if not preferred.empty:

            row = preferred.iloc[0]

            selected_model = row["Model"]
            selected_accuracy = row["Accuracy"]
            selected_high_precision = row[
                "High_Risk_Precision"
            ]
            selected_high_recall = row[
                "High_Risk_Recall"
            ]
            selected_high_f1 = row[
                "High_Risk_F1"
            ]
            selected_auc = row[
                "ROC_AUC_OVR_Macro"
            ]

    rows = [
        {
            "kpi": "Total suppliers",
            "value": total_suppliers,
        },
        {
            "kpi": "High/Critical priority suppliers",
            "value": high_priority,
        },
        {
            "kpi": "Moderate priority suppliers",
            "value": moderate,
        },
        {
            "kpi": "Low priority suppliers",
            "value": low,
        },
        {
            "kpi": "Immediate intervention suppliers",
            "value": immediate,
        },
        {
            "kpi": "Corrective action suppliers",
            "value": corrective,
        },
        {
            "kpi": "Selected model",
            "value": selected_model,
        },
        {
            "kpi": "Selected model accuracy",
            "value": selected_accuracy,
        },
        {
            "kpi": "High-risk precision",
            "value": selected_high_precision,
        },
        {
            "kpi": "High-risk recall",
            "value": selected_high_recall,
        },
        {
            "kpi": "High-risk F1",
            "value": selected_high_f1,
        },
        {
            "kpi": "ROC AUC",
            "value": selected_auc,
        },
    ]

    output = pd.DataFrame(rows)

    path = (
        DASHBOARD_DIR /
        "dashboard_kpis.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"KPI summary saved:\n{path}"
    )

    return output


# ================================================================
# CREATE DASHBOARD DATA DICTIONARY
# ================================================================

def create_data_dictionary():

    print_header(
        "CREATING DASHBOARD DATA DICTIONARY"
    )

    dictionary = pd.DataFrame([
        {
            "field": "supplier_id",
            "meaning": "Unique supplier identifier",
            "dashboard_use": "Supplier selection/filter",
        },
        {
            "field": "period",
            "meaning": "Latest supplier observation period",
            "dashboard_use": "Time context",
        },
        {
            "field": "predicted_risk_N1",
            "meaning": "Predicted next-period risk category",
            "dashboard_use": "Risk classification",
        },
        {
            "field": "probability_high",
            "meaning": "Model probability assigned to High risk",
            "dashboard_use": "Model risk signal",
        },
        {
            "field": "priority_score",
            "meaning": "Operational supplier priority score from 0-100",
            "dashboard_use": "Supplier ranking",
        },
        {
            "field": "priority_band",
            "meaning": "Operational priority category",
            "dashboard_use": "Priority segmentation",
        },
        {
            "field": "delivery_risk_score",
            "meaning": "Delivery-related operational risk score",
            "dashboard_use": "Risk driver",
        },
        {
            "field": "quality_risk_score",
            "meaning": "Quality-related operational risk score",
            "dashboard_use": "Risk driver",
        },
        {
            "field": "strategic_exposure_score",
            "meaning": "Strategic exposure component",
            "dashboard_use": "Business exposure",
        },
        {
            "field": "deterioration_score",
            "meaning": "Recent deterioration/trend component",
            "dashboard_use": "Trend risk",
        },
        {
            "field": "model_high_risk_score",
            "meaning": "Model high-risk component converted to 0-100",
            "dashboard_use": "Model signal",
        },
        {
            "field": "primary_risk_domain",
            "meaning": "Main operational reason associated with supplier priority",
            "dashboard_use": "Risk explanation",
        },
        {
            "field": "why_risky",
            "meaning": "Human-readable explanation of supplier risk",
            "dashboard_use": "Supplier drill-through",
        },
        {
            "field": "risk_evidence",
            "meaning": "Evidence supporting the risk explanation",
            "dashboard_use": "Decision support",
        },
        {
            "field": "recommended_intervention",
            "meaning": "Deterministic procurement intervention",
            "dashboard_use": "Action recommendation",
        },
    ])

    path = (
        DASHBOARD_DIR /
        "dashboard_data_dictionary.csv"
    )

    dictionary.to_csv(
        path,
        index=False
    )

    print(
        f"Data dictionary saved:\n{path}"
    )


# ================================================================
# VALIDATE FINAL DASHBOARD LAYER
# ================================================================

def final_validation(
    supplier_dashboard,
    priority_table,
    domain_summary,
    priority_summary,
    intervention_summary,
    trend,
    benchmark,
):

    print_header(
        "PHASE 7 FINAL VALIDATION"
    )

    checks = []

    checks.append(
        (
            "Supplier dashboard one row per supplier",
            not supplier_dashboard[
                "supplier_id"
            ].duplicated().any()
        )
    )

    checks.append(
        (
            "Supplier count = 40",
            supplier_dashboard[
                "supplier_id"
            ].nunique() == 40
        )
    )

    checks.append(
        (
            "Priority score 0-100",
            supplier_dashboard[
                "priority_score"
            ].between(0, 100).all()
        )
    )

    checks.append(
        (
            "Delivery score 0-100",
            supplier_dashboard[
                "delivery_risk_score"
            ].between(0, 100).all()
        )
    )

    checks.append(
        (
            "Quality score 0-100",
            supplier_dashboard[
                "quality_risk_score"
            ].between(0, 100).all()
        )
    )

    checks.append(
        (
            "Deterioration score 0-100",
            supplier_dashboard[
                "deterioration_score"
            ].between(0, 100).all()
        )
    )

    checks.append(
        (
            "Probability 0-1",
            supplier_dashboard[
                "probability_high"
            ].between(0, 1).all()
        )
    )

    checks.append(
        (
            "Why-risky explanations complete",
            supplier_dashboard[
                "why_risky"
            ].notna().all()
        )
    )

    checks.append(
        (
            "Risk evidence complete",
            supplier_dashboard[
                "risk_evidence"
            ].notna().all()
        )
    )

    checks.append(
        (
            "Recommendations complete",
            supplier_dashboard[
                "recommended_intervention"
            ].notna().all()
        )
    )

    for name, passed in checks:

        if not passed:
            raise AssertionError(
                f"FAILED: {name}"
            )

        print(
            f"✓ {name}: PASSED"
        )

    print()
    print(
        "Future/outcome leakage audit: PASSED"
    )

    # ------------------------------------------------------------
    # Save validation report
    # ------------------------------------------------------------

    validation_lines = [
        "PHASE 7 VALIDATION",
        "==================",
        "",
    ]

    for name, passed in checks:
        validation_lines.append(
            f"✓ {name}: PASSED"
        )

    validation_lines.extend([
        "",
        "✓ Future/outcome leakage audit: PASSED",
        "✓ Dashboard data layer validated",
        "✓ Power BI source tables generated",
        "",
        "PHASE 7 DATA PREPARATION COMPLETE",
    ])

    validation_path = (
        REPORT_DIR /
        "phase7_dashboard_validation.txt"
    )

    validation_path.write_text(
        "\n".join(validation_lines),
        encoding="utf-8"
    )

    return validation_path


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "PHASE 7: DECISION-SUPPORT DASHBOARD DATA PREPARATION"
    )

    (
        priority,
        intervention,
        reasons,
        oot,
        benchmark,
    ) = load_inputs()

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    validate_priority(priority)
    validate_interventions(intervention)

    # The reasons file must also contain supplier-level explanations.
    require_columns(
        reasons,
        [
            "supplier_id",
            "why_risky",
            "risk_evidence",
        ],
        "supplier_risk_reasons.csv"
    )

    # ------------------------------------------------------------
    # Leakage checks on dashboard inputs
    # ------------------------------------------------------------

    print_header(
        "DASHBOARD LEAKAGE AUDIT"
    )

    leakage_audit(
        priority,
        "priority input"
    )

    leakage_audit(
        intervention,
        "intervention input"
    )

    leakage_audit(
        reasons,
        "reason input"
    )

    print(
        "✓ Future/outcome variables excluded: PASSED"
    )

    # ------------------------------------------------------------
    # Dashboard tables
    # ------------------------------------------------------------

    supplier_dashboard = create_supplier_dashboard(
        priority,
        intervention
    )

    priority_table = create_priority_table(
        supplier_dashboard
    )

    domain_summary = create_domain_summary(
        supplier_dashboard
    )

    priority_summary = create_priority_summary(
        supplier_dashboard
    )

    intervention_summary = create_intervention_summary(
        supplier_dashboard
    )

    trend = create_oot_trend_table(
        oot
    )

    benchmark_output = create_model_benchmark(
        benchmark
    )

    kpis = create_kpi_summary(
        supplier_dashboard,
        benchmark
    )

    create_data_dictionary()

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    validation_path = final_validation(
        supplier_dashboard,
        priority_table,
        domain_summary,
        priority_summary,
        intervention_summary,
        trend,
        benchmark_output,
    )

    # ------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------

    summary = {
        "phase": "Phase 7 - Decision-Support Dashboard Data Preparation",
        "supplier_count": int(
            supplier_dashboard[
                "supplier_id"
            ].nunique()
        ),
        "dashboard_tables_created": [
            "supplier_dashboard.csv",
            "priority_supplier_table.csv",
            "risk_domain_summary.csv",
            "priority_band_summary.csv",
            "intervention_summary.csv",
            "supplier_risk_trend.csv",
            "model_benchmark.csv",
            "dashboard_kpis.csv",
            "dashboard_data_dictionary.csv",
        ],
        "validation": str(
            validation_path
        ),
        "dashboard_directory": str(
            DASHBOARD_DIR
        ),
    }

    summary_path = (
        REPORT_DIR /
        "phase7_dashboard_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------

    print_header(
        "PHASE 7 COMPLETE"
    )

    print(
        "Dashboard suppliers:",
        supplier_dashboard[
            "supplier_id"
        ].nunique()
    )

    print(
        "Dashboard files created:"
    )

    for file in sorted(
        DASHBOARD_DIR.glob("*.csv")
    ):
        print(
            f" - {file.name}"
        )

    print()
    print(
        "Validation:",
        validation_path
    )

    print(
        "Summary:",
        summary_path
    )

    print()
    print(
        "NEXT IMPLEMENTATION STAGE:"
    )

    print(
        "POWER BI DASHBOARD BUILD"
    )


if __name__ == "__main__":
    main()