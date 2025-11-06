"""
Wine Quality PCA starter using ucimlrepo dataset fetch.
This script demonstrates fetching the dataset from the UCI repository via ucimlrepo
and printing metadata / variable info as requested.

Usage (PowerShell):
    # create & activate venv (one-time)
    python -m venv .venv
    .\\.venv\\Scripts\\Activate
    pip install -r requirements.txt

Run:
  python main.py
"""

import sys

try:
    from ucimlrepo import fetch_ucirepo
except Exception as e:
    print("ucimlrepo is not installed or failed to import:", e)
    print("Please run: pip install ucimlrepo")
    sys.exit(1)


def main():
    # fetch dataset (user requested id=186)
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    print("--- Dataset loaded ---")
    print("Features shape:", getattr(X, 'shape', 'N/A'))
    print("Targets shape:", getattr(y, 'shape', 'N/A'))

    # metadata
    print("\n--- Metadata ---")
    try:
        print(wine_quality.metadata)
    except Exception as ex:
        print("Could not print metadata:", ex)

    # variable information
    print("\n--- Variable information ---")
    try:
        print(wine_quality.variables)
    except Exception as ex:
        print("Could not print variables info:", ex)

    # Optionally save to CSV for quick inspection
    try:
        import pandas as pd
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.to_csv('wine_ucirepo_186.csv', index=False)
        print("Saved combined dataset to wine_ucirepo_186.csv")
    except Exception:
        # pandas might not be installed; skip silently
        pass


if __name__ == '__main__':
    main()
