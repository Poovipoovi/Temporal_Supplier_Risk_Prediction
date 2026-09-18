# src/data_audit.py
import os
import glob
import pandas as pd
import numpy as np

def audit_raw_data(data_dir="data/raw"):
    print("=" * 60)
    print("PHASE 1: RAW AEROSPACE DATASET AUDIT")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory '{data_dir}' does not exist.")
        return

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{data_dir}'.")
        return

    print(f"Found {len(csv_files)} CSV files in '{data_dir}':")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")
    print("-" * 60)

    for filepath in csv_files:
        fname = os.path.basename(filepath)
        print(f"\n>>> AUDITING FILE: {fname}")
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"[ERROR] Could not read {fname}: {e}")
            continue

        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print("\nColumns and Data Types:")
        for col, dtype in df.dtypes.items():
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            print(f"  - {col} ({dtype}): {missing} missing ({missing_pct:.2f}%)")

        print(f"\nDuplicate Rows: {df.duplicated().sum()}")
        print("\nSample Head (2 rows):")
        print(df.head(2).to_string())
        print("-" * 60)

if __name__ == "__main__":
    audit_raw_data()