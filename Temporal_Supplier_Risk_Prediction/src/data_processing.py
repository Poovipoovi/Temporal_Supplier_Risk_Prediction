import os
import numpy as np
import pandas as pd


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "aerospace_supplier_monthly.csv")


def safe_divide(numerator, denominator):
    return np.where(
        denominator != 0,
        numerator / denominator,
        0.0
    )


def run_data_processing():

    print("=" * 70)
    print("PHASE 1: AEROSPACE SUPPLIER MONTHLY FEATURE ENGINEERING")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD RAW DATA
    # ---------------------------------------------------------

    parts = pd.read_csv(
        os.path.join(RAW_DIR, "parts_master.csv")
    )

    purchase_orders = pd.read_csv(
        os.path.join(RAW_DIR, "purchase_orders.csv")
    )

    quality = pd.read_csv(
        os.path.join(RAW_DIR, "quality_incidents.csv")
    )

    supply_history = pd.read_csv(
        os.path.join(RAW_DIR, "supply_chain_history.csv")
    )

    print(f"Parts master:       {parts.shape}")
    print(f"Purchase orders:    {purchase_orders.shape}")
    print(f"Quality incidents:  {quality.shape}")
    print(f"Supply history:     {supply_history.shape}")

    # ---------------------------------------------------------
    # 2. DATE CONVERSION
    # ---------------------------------------------------------

    purchase_orders["order_date"] = pd.to_datetime(
        purchase_orders["order_date"],
        errors="coerce"
    )

    purchase_orders["promised_date"] = pd.to_datetime(
        purchase_orders["promised_date"],
        errors="coerce"
    )

    purchase_orders["receipt_date"] = pd.to_datetime(
        purchase_orders["receipt_date"],
        errors="coerce"
    )

    quality["incident_date"] = pd.to_datetime(
        quality["incident_date"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 3. BASIC DATA VALIDATION
    # ---------------------------------------------------------

    required_po = [
        "po_id",
        "supplier_id",
        "part_id",
        "order_date",
        "promised_date",
        "receipt_date",
        "ordered_qty",
        "received_qty"
    ]

    required_quality = [
        "supplier_id",
        "incident_date",
        "scrap_qty"
    ]

    missing_po = [
        c for c in required_po
        if c not in purchase_orders.columns
    ]

    missing_quality = [
        c for c in required_quality
        if c not in quality.columns
    ]

    if missing_po:
        raise ValueError(
            f"Purchase-order dataset missing columns: {missing_po}"
        )

    if missing_quality:
        raise ValueError(
            f"Quality dataset missing columns: {missing_quality}"
        )

    # ---------------------------------------------------------
    # 4. PURCHASE-ORDER OPERATIONAL METRICS
    # ---------------------------------------------------------

    purchase_orders["days_late"] = (
        purchase_orders["receipt_date"]
        - purchase_orders["promised_date"]
    ).dt.days

    # Negative lateness is treated as zero delay.
    purchase_orders["days_late_positive"] = (
        purchase_orders["days_late"]
        .clip(lower=0)
    )

    purchase_orders["is_late"] = (
        purchase_orders["days_late"] > 0
    ).astype(int)

    purchase_orders["is_short_delivery"] = (
        purchase_orders["received_qty"]
        < purchase_orders["ordered_qty"]
    ).astype(int)

    purchase_orders["short_qty"] = (
        purchase_orders["ordered_qty"]
        - purchase_orders["received_qty"]
    ).clip(lower=0)

    purchase_orders["period"] = (
        purchase_orders["receipt_date"]
        .dt.to_period("M")
        .astype(str)
    )

    quality["period"] = (
        quality["incident_date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------------
    # 5. PART MASTER INFORMATION
    # ---------------------------------------------------------

    part_columns = [
        "part_id",
        "criticality_class",
        "unit_cost"
    ]

    missing_part_columns = [
        c for c in part_columns
        if c not in parts.columns
    ]

    if missing_part_columns:
        raise ValueError(
            f"Parts dataset missing columns: {missing_part_columns}"
        )

    parts_subset = parts[part_columns].drop_duplicates(
        subset=["part_id"]
    )

    orders = purchase_orders.merge(
        parts_subset,
        on="part_id",
        how="left"
    )

    orders["is_critical"] = (
        orders["criticality_class"]
        .astype(str)
        .str.lower()
        .eq("high")
    ).astype(int)

    orders["po_spend"] = (
        orders["received_qty"]
        * orders["unit_cost"].fillna(0)
    )

    # ---------------------------------------------------------
    # 6. MONTHLY PURCHASE-ORDER AGGREGATION
    # ---------------------------------------------------------

    po_monthly = (
        orders
        .groupby(["supplier_id", "period"])
        .agg(
            n_orders=("po_id", "nunique"),
            late_orders=("is_late", "sum"),
            avg_days_late=("days_late_positive", "mean"),
            total_days_late=("days_late_positive", "sum"),
            short_orders=("is_short_delivery", "sum"),
            total_short_qty=("short_qty", "sum"),
            total_spend=("po_spend", "sum"),
            critical_orders=("is_critical", "sum")
        )
        .reset_index()
    )

    po_monthly["late_delivery_rate"] = safe_divide(
        po_monthly["late_orders"],
        po_monthly["n_orders"]
    )

    po_monthly["short_delivery_rate"] = safe_divide(
        po_monthly["short_orders"],
        po_monthly["n_orders"]
    )

    po_monthly["critical_component_exposure"] = safe_divide(
        po_monthly["critical_orders"],
        po_monthly["n_orders"]
    )

    # ---------------------------------------------------------
    # 7. QUALITY INCIDENT AGGREGATION
    # ---------------------------------------------------------

    quality_monthly = (
        quality
        .groupby(["supplier_id", "period"])
        .agg(
            quality_events=("incident_date", "count"),
            total_scrap_qty=("scrap_qty", "sum")
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 8. MERGE OPERATIONAL + QUALITY DATA
    # ---------------------------------------------------------

    supplier_monthly = po_monthly.merge(
        quality_monthly,
        on=["supplier_id", "period"],
        how="left"
    )

    supplier_monthly["quality_events"] = (
        supplier_monthly["quality_events"]
        .fillna(0)
    )

    supplier_monthly["total_scrap_qty"] = (
        supplier_monthly["total_scrap_qty"]
        .fillna(0)
    )

    supplier_monthly["quality_event_rate"] = safe_divide(
        supplier_monthly["quality_events"],
        supplier_monthly["n_orders"]
    )

    # ---------------------------------------------------------
    # 9. REWORK COST RATIO
    # ---------------------------------------------------------

    # Use scrap quantity × unit cost as a consistent cost proxy.
    # This is calculated from the current month only.

    orders["scrap_cost_proxy"] = (
        orders["received_qty"]
        * orders["unit_cost"].fillna(0)
    )

    cost_monthly = (
        orders
        .groupby(["supplier_id", "period"])
        .agg(
            estimated_quality_cost=("scrap_cost_proxy", "sum")
        )
        .reset_index()
    )

    supplier_monthly = supplier_monthly.merge(
        cost_monthly,
        on=["supplier_id", "period"],
        how="left"
    )

    supplier_monthly["estimated_quality_cost"] = (
        supplier_monthly["estimated_quality_cost"]
        .fillna(0)
    )

    supplier_monthly["rework_cost_ratio"] = safe_divide(
        supplier_monthly["estimated_quality_cost"],
        supplier_monthly["total_spend"]
    )

    # ---------------------------------------------------------
    # 10. SUPPLIER FAULT RATE
    # ---------------------------------------------------------

    if "supplier_fault" in quality.columns:

        fault_monthly = (
            quality
            .assign(
                supplier_fault_flag=pd.to_numeric(
                    quality["supplier_fault"],
                    errors="coerce"
                ).fillna(0)
            )
            .groupby(["supplier_id", "period"])
            .agg(
                supplier_fault_events=("supplier_fault_flag", "sum")
            )
            .reset_index()
        )

        supplier_monthly = supplier_monthly.merge(
            fault_monthly,
            on=["supplier_id", "period"],
            how="left"
        )

        supplier_monthly["supplier_fault_events"] = (
            supplier_monthly["supplier_fault_events"]
            .fillna(0)
        )

        supplier_monthly["supplier_fault_rate"] = safe_divide(
            supplier_monthly["supplier_fault_events"],
            supplier_monthly["quality_events"]
        )

    else:

        print(
            "WARNING: supplier_fault column not available. "
            "Setting supplier_fault_rate to 0."
        )

        supplier_monthly["supplier_fault_rate"] = 0.0

    # ---------------------------------------------------------
    # 11. SPEND SHARE
    # ---------------------------------------------------------

    monthly_total_spend = (
        supplier_monthly
        .groupby("period")["total_spend"]
        .transform("sum")
    )

    supplier_monthly["spend_share"] = safe_divide(
        supplier_monthly["total_spend"],
        monthly_total_spend
    )

    # ---------------------------------------------------------
    # 12. TRANSPORT / TRANSIT FEATURES
    # ---------------------------------------------------------

    # Only create these if the raw dataset actually contains
    # defensible transport variables.

    supplier_monthly["route_deviation_flag"] = 0.0
    supplier_monthly["transit_delay_rate"] = 0.0

    possible_route_columns = [
        "route_deviation_flag",
        "route_deviation",
        "route_deviation_days"
    ]

    route_column = next(
        (
            c for c in possible_route_columns
            if c in orders.columns
        ),
        None
    )

    if route_column is not None:

        orders["route_deviation_indicator"] = (
            pd.to_numeric(
                orders[route_column],
                errors="coerce"
            )
            .fillna(0)
            > 0
        ).astype(int)

        route_monthly = (
            orders
            .groupby(["supplier_id", "period"])
            .agg(
                route_deviation_flag=(
                    "route_deviation_indicator",
                    "mean"
                )
            )
            .reset_index()
        )

        supplier_monthly = supplier_monthly.drop(
            columns=["route_deviation_flag"]
        )

        supplier_monthly = supplier_monthly.merge(
            route_monthly,
            on=["supplier_id", "period"],
            how="left"
        )

    # ---------------------------------------------------------
    # 13. TEMPORAL SORT
    # ---------------------------------------------------------

    supplier_monthly["period_date"] = pd.to_datetime(
        supplier_monthly["period"] + "-01"
    )

    supplier_monthly = supplier_monthly.sort_values(
        ["supplier_id", "period_date"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 14. PERIOD-OVER-PERIOD TRENDS
    # ---------------------------------------------------------

    grouped = supplier_monthly.groupby("supplier_id")

    supplier_monthly["previous_late_rate"] = (
        grouped["late_delivery_rate"].shift(1)
    )

    supplier_monthly["previous_quality_rate"] = (
        grouped["quality_event_rate"].shift(1)
    )

    supplier_monthly["previous_avg_days_late"] = (
        grouped["avg_days_late"].shift(1)
    )

    supplier_monthly["previous_total_spend"] = (
        grouped["total_spend"].shift(1)
    )

    supplier_monthly["late_rate_trend"] = (
        supplier_monthly["late_delivery_rate"]
        - supplier_monthly["previous_late_rate"]
    ).fillna(0)

    supplier_monthly["quality_rate_trend"] = (
        supplier_monthly["quality_event_rate"]
        - supplier_monthly["previous_quality_rate"]
    ).fillna(0)

    supplier_monthly["avg_days_late_trend"] = (
        supplier_monthly["avg_days_late"]
        - supplier_monthly["previous_avg_days_late"]
    ).fillna(0)

    supplier_monthly["transit_delay_trend"] = (
        supplier_monthly["transit_delay_rate"]
        - grouped["transit_delay_rate"].shift(1)
    ).fillna(0)

    supplier_monthly["has_trend_data"] = (
        supplier_monthly["previous_late_rate"]
        .notna()
    ).astype(int)

    supplier_monthly["is_worsening"] = (
        (
            supplier_monthly["late_rate_trend"] > 0
        )
        |
        (
            supplier_monthly["quality_rate_trend"] > 0
        )
        |
        (
            supplier_monthly["avg_days_late_trend"] > 0
        )
    ).astype(int)

    # ---------------------------------------------------------
    # 15. REMOVE TEMPORARY COLUMNS
    # ---------------------------------------------------------

    temporary_columns = [
        "period_date",
        "previous_late_rate",
        "previous_quality_rate",
        "previous_avg_days_late",
        "previous_total_spend"
    ]

    supplier_monthly = supplier_monthly.drop(
        columns=[
            c for c in temporary_columns
            if c in supplier_monthly.columns
        ]
    )

    # ---------------------------------------------------------
    # 16. FINAL CLEANING
    # ---------------------------------------------------------

    numeric_columns = supplier_monthly.select_dtypes(
        include=[np.number]
    ).columns

    supplier_monthly[numeric_columns] = (
        supplier_monthly[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # 17. FINAL VALIDATION
    # ---------------------------------------------------------

    supplier_monthly = supplier_monthly.sort_values(
        ["supplier_id", "period"]
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("FINAL MONTHLY DATASET")
    print("=" * 70)

    print(f"Rows: {len(supplier_monthly):,}")
    print(
        f"Suppliers: "
        f"{supplier_monthly['supplier_id'].nunique():,}"
    )
    print(
        f"Periods: "
        f"{supplier_monthly['period'].nunique():,}"
    )

    print("\nFeature columns:")

    feature_preview = [
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
        "has_trend_data"
    ]

    for feature in feature_preview:
        if feature in supplier_monthly.columns:
            print(f"  ✓ {feature}")

    # ---------------------------------------------------------
    # 18. SAVE
    # ---------------------------------------------------------

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    supplier_monthly.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSUCCESS")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Final shape: {supplier_monthly.shape}")


if __name__ == "__main__":
    run_data_processing()