# src/data_audit_temporal.py
import pandas as pd
import numpy as np

def run_temporal_audit():
    print("=" * 60)
    print("PHASE 1 (PART 2): TEMPORAL STRUCTURE & SUPPLIER COVERAGE")
    print("=" * 60)

    pos = pd.read_csv("data/raw/purchase_orders.csv")
    qis = pd.read_csv("data/raw/quality_incidents.csv")
    parts = pd.read_csv("data/raw/parts_master.csv")

    # Convert date fields
    pos['receipt_date'] = pd.to_datetime(pos['receipt_date'])
    pos['order_date'] = pd.to_datetime(pos['order_date'])
    qis['incident_date'] = pd.to_datetime(qis['incident_date'])

    print(f"Purchase Orders Date Range: {pos['receipt_date'].min().strftime('%Y-%m-%d')} to {pos['receipt_date'].max().strftime('%Y-%m-%d')}")
    print(f"Quality Incidents Date Range: {qis['incident_date'].min().strftime('%Y-%m-%d')} to {qis['incident_date'].max().strftime('%Y-%m-%d')}")

    # Unique Supplier Counts
    po_suppliers = set(pos['supplier_id'].unique())
    qi_suppliers = set(qis['supplier_id'].unique())
    part_suppliers = set(parts['supplier_id_primary'].unique())

    print(f"\nUnique Suppliers in POs: {len(po_suppliers)}")
    print(f"Unique Suppliers in Quality Incidents: {len(qi_suppliers)}")
    print(f"Unique Primary Suppliers in Parts Master: {len(part_suppliers)}")
    print(f"Suppliers in QI but missing from POs: {len(qi_suppliers - po_suppliers)}")

    # Evaluate Monthly vs. Quarterly Aggregation Sparsity
    pos['year_month'] = pos['receipt_date'].dt.to_period('M')
    pos['year_quarter'] = pos['receipt_date'].dt.to_period('Q')

    monthly_counts = pos.groupby(['supplier_id', 'year_month']).size().unstack(fill_value=0)
    quarterly_counts = pos.groupby(['supplier_id', 'year_quarter']).size().unstack(fill_value=0)

    print("\n--- SPARSITY ASSESSMENT ---")
    print(f"Total Periods (Monthly): {monthly_counts.shape[1]}")
    print(f"Active Supplier-Month Pairs (>0 POs): {(monthly_counts > 0).sum().sum()} / {monthly_counts.size} ({((monthly_counts > 0).sum().sum() / monthly_counts.size)*100:.2f}%)")
    
    print(f"Total Periods (Quarterly): {quarterly_counts.shape[1]}")
    print(f"Active Supplier-Quarter Pairs (>0 POs): {(quarterly_counts > 0).sum().sum()} / {quarterly_counts.size} ({((quarterly_counts > 0).sum().sum() / quarterly_counts.size)*100:.2f}%)")

if __name__ == '__main__':
    run_temporal_audit()