"""
pca_demo.py

Run a small reproducible PCA-based analysis on the Wine Quality dataset (UCI id=186).
- Loads `wine_ucirepo_186.csv` if present; otherwise tries to fetch and save it.
- Performs EDA prints + saves histograms and correlation heatmap.
- Creates binary target (good: quality >= 7).
- Trains a baseline RandomForest on scaled features.
- Fits PCA (explained variance and n_components=0.95), and trains RandomForest on PCA features.
- Saves plots and a short results summary `results_summary.txt`.

Usage:
    python pca_demo.py

This script is defensive: if required packages are missing it prints install instructions.
"""

import os
import sys
from datetime import datetime

# Check imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except Exception as e:
    print("Missing required packages or failed to import:", e)
    print("Please create and activate a venv then install requirements:")
    print(r"python -m venv .venv ; .\\.venv\\Scripts\\Activate ; pip install -r requirements.txt")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_DIR, 'wine_ucirepo_186.csv')
OUT_DIR = os.path.join(PROJECT_DIR, 'pca_outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    if os.path.exists(CSV_PATH):
        print(f"Loading CSV from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        return df

    # Try to fetch using ucimlrepo if available
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception:
        # fallback: instruct user
        print("CSV not found and ucimlrepo not available to fetch. Please run: python main.py or provide the CSV file.")
        sys.exit(1)

    try:
        print("Fetching dataset via ucimlrepo.fetch_ucirepo(id=186)...")
        ds = fetch_ucirepo(id=186)
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        df.to_csv(CSV_PATH, index=False)
        print(f"Saved fetched dataset to {CSV_PATH}")
        return df
    except Exception as e:
        print("Failed to fetch dataset using ucimlrepo:", e)
        print("Either provide 'wine_ucirepo_186.csv' in the project folder or install ucimlrepo and run main.py")
        sys.exit(1)


def quick_eda(df):
    print('\n=== Quick EDA ===')
    print('Shape:', df.shape)
    print('\nHead:')
    print(df.head())
    print('\nInfo:')
    print(df.info())
    print('\nDescribe:')
    print(df.describe())
    if 'quality' in df.columns:
        print('\nQuality value counts:')
        print(df['quality'].value_counts().sort_index())

    # Histograms
    try:
        hist_path = os.path.join(OUT_DIR, 'feature_histograms.png')
        df.hist(figsize=(12, 10))
        plt.suptitle('Feature histograms')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(hist_path)
        plt.close()
        print('Saved histograms to', hist_path)
    except Exception as e:
        print('Failed to save histograms:', e)

    # Correlation heatmap
    try:
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Feature correlation matrix')
        heatmap_path = os.path.join(OUT_DIR, 'correlation_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        print('Saved correlation heatmap to', heatmap_path)
    except Exception as e:
        print('Failed to save correlation heatmap:', e)


def prepare_data(df):
    # Target: binary good wine (quality >= 7)
    if 'quality' not in df.columns:
        raise ValueError("CSV does not contain 'quality' column")

    X = df.drop('quality', axis=1).copy()
    y = (df['quality'] >= 7).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


def baseline_model(X_train_s, X_test_s, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print('\n--- Baseline RandomForest (scaled features) ---')
    print('Accuracy:', acc)
    print(report)

    # Save confusion matrix
    try:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Baseline')
        path = os.path.join(OUT_DIR, 'confusion_baseline.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print('Saved baseline confusion matrix to', path)
    except Exception as e:
        print('Failed to save baseline confusion matrix:', e)

    return clf, acc, report, cm


def pca_and_model(X_train_s, X_test_s, y_train, y_test, original_X_train):
    # PCA for explained variance (full PCA)
    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(np.vstack([X_train_s, X_test_s]))
    evr = pca_full.explained_variance_ratio_

    # Plot cumulative explained variance
    try:
        cum = np.cumsum(evr)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(cum) + 1), cum, marker='o')
        plt.axhline(0.90, color='gray', linestyle='--', label='90%')
        plt.axhline(0.95, color='red', linestyle='--', label='95%')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('PCA cumulative explained variance')
        plt.legend()
        path = os.path.join(OUT_DIR, 'explained_variance.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print('Saved explained variance curve to', path)
    except Exception as e:
        print('Failed to save explained variance plot:', e)

    # Choose n_components=0.95 to keep ~95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    print('\nPCA reduced shape (train):', X_train_p.shape)
    print('PCA n_components chosen:', pca.n_components_)
    print('Explained variance ratio (first components):', pca.explained_variance_ratio_)

    # Train classifier on PCA features
    clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_pca.fit(X_train_p, y_train)
    y_pred_p = clf_pca.predict(X_test_p)

    acc_p = accuracy_score(y_test, y_pred_p)
    report_p = classification_report(y_test, y_pred_p, digits=4)
    cm_p = confusion_matrix(y_test, y_pred_p)

    print('\n--- RandomForest on PCA features ---')
    print('Accuracy:', acc_p)
    print(report_p)

    try:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_p, annot=True, fmt='d', cmap='Greens')
        plt.xlabel('Pred')
        plt.ylabel('True')
        plt.title('Confusion Matrix - PCA features')
        path = os.path.join(OUT_DIR, 'confusion_pca.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print('Saved PCA confusion matrix to', path)
    except Exception as e:
        print('Failed to save PCA confusion matrix:', e)

    # For visualization 2-component PCA scatter
    try:
        pca2 = PCA(n_components=2, random_state=42)
        X2 = pca2.fit_transform(np.vstack([X_train_s, X_test_s]))
        # reconstruct labels for the combined set
        y_combined = np.concatenate([y_train.values, y_test.values])
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=y_combined, palette=['tab:orange', 'tab:blue'], alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA (2 components) scatter - colored by good (1)')
        path = os.path.join(OUT_DIR, 'pca_scatter.png')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print('Saved 2-PC scatter to', path)
    except Exception as e:
        print('Failed to save PCA scatter plot:', e)

    # Return the PCA model and metrics
    return pca, clf_pca, acc_p, report_p, cm_p


def save_summary(baseline_stats, pca_stats):
    baseline_acc, baseline_report = baseline_stats
    pca_acc, pca_report = pca_stats
    summary_path = os.path.join(OUT_DIR, 'results_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Wine Quality PCA - Results summary\n')
        f.write('Run at: ' + datetime.utcnow().isoformat() + ' UTC\n\n')
        f.write('Baseline (RandomForest on scaled features)\n')
        f.write(f'Accuracy: {baseline_acc:.4f}\n')
        f.write('\nClassification report:\n')
        f.write(baseline_report + '\n')

        f.write('\nPCA (n_components=0.95) + RandomForest\n')
        f.write(f'Accuracy: {pca_acc:.4f}\n')
        f.write('\nClassification report:\n')
        f.write(pca_report + '\n')

        f.write('\nNote: Plots saved to folder: ' + OUT_DIR + '\n')
    print('Saved results summary to', summary_path)


def main():
    df = load_data()
    quick_eda(df)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_data(df)

    clf_baseline, baseline_acc, baseline_report, baseline_cm = baseline_model(X_train_s, X_test_s, y_train, y_test)

    pca_model, clf_pca, pca_acc, pca_report, pca_cm = pca_and_model(X_train_s, X_test_s, y_train, y_test, X_train)

    # Save textual summary
    save_summary((baseline_acc, baseline_report), (pca_acc, pca_report))

    print('\nDone. Generated outputs in folder:', OUT_DIR)
    print('Files to inspect: explained_variance.png, pca_scatter.png, confusion_baseline.png, confusion_pca.png, results_summary.txt')


if __name__ == '__main__':
    main()
