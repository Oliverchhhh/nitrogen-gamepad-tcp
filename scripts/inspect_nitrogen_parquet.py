#!/usr/bin/env python3
"""Inspect NitroGen parquet file structure."""

import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python inspect_nitrogen_parquet.py <path_to_actions_processed.parquet>")
    sys.exit(1)

parquet_path = Path(sys.argv[1])
if not parquet_path.exists():
    print(f"File not found: {parquet_path}")
    sys.exit(1)

df = pd.read_parquet(parquet_path)

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  - {col}")

print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nSample values (first row):")
for col in df.columns:
    print(f"  {col}: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
