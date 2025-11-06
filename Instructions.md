Wine Quality PCA — Implementation Instructions

Goal

Create a short, reproducible PCA-based feature-extraction analysis for the Wine Quality dataset (UCI id=186). The final deliverables should be a runnable notebook or script that: fetches the dataset, performs EDA and preprocessing, applies PCA, trains a simple model (baseline vs PCA), visualizes results (2‑PC scatter and explained variance), and a short results write-up.

Estimated time: 2–3 hours (fast, minimal polish)

Prerequisites

- Python 3.8+ installed
- PowerShell (Windows) or a POSIX shell (WSL/Ubuntu)
- Optional: Jupyter Notebook / JupyterLab for the deliverable

Project layout (what we'll create/use)

c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA/
- main.py              # includes dataset fetch and CSV save (already present)
- wine_ucirepo_186.csv # saved dataset (created by main.py)
- requirements.txt     # dependencies
- README.md            # project & run instructions
- Instructions.md      # this file (step-by-step)
- pca_demo.ipynb       # (optional) notebook demo you may create

Step-by-step implementation

1) Create and activate a virtual environment (PowerShell)

```powershell
Set-Location -Path "c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA"
python -m venv .venv
. .\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If using WSL/Ubuntu, replace with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Fetch the dataset (already implemented in `main.py`)

- Run `python main.py`. This uses `ucimlrepo.fetch_ucirepo(id=186)` to fetch the Wine Quality dataset and will save `wine_ucirepo_186.csv` in the folder.
- Inspect the printed metadata and `wine_ucirepo_186.csv`.

3) Quick EDA (create a notebook or add these steps to `main.py`)

- Load dataset with pandas:
  - df = pd.read_csv('wine_ucirepo_186.csv')
- Inspect:
  - df.shape
  - df.head()
  - df.info()
  - df.describe()
  - df['quality'].value_counts().sort_index()
- Visual checks:
  - histograms for features (df.hist)
  - correlation heatmap: sns.heatmap(df.corr(), annot=True)

Goal: understand distributions, ranges, any obvious preprocessing needs (missing values, scaling)

4) Decide target label and problem type

- For a simple classification assignment map `quality` into a binary target:
  - good = (quality >= 7) -> 1, else 0
- Alternative: 3-class grouping (low/medium/high) or regression (predict raw score). For fast completion choose binary.

5) Preprocessing

- Split features / target:
  - X = df.drop('quality', axis=1)
  - y = (df['quality'] >= 7).astype(int)
- Train / test split:
  - X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
- Scaling:
  - scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
- Missing values: (the Wine dataset has no missing values per metadata) — if present, use SimpleImputer(strategy='median')

6) Baseline model (no PCA)

- Use a simple, robust classifier such as RandomForestClassifier or LogisticRegression:
  - clf = RandomForestClassifier(n_estimators=100, random_state=42)
  - clf.fit(X_train_s, y_train)
  - y_pred = clf.predict(X_test_s)
- Report metrics: accuracy, precision, recall, f1-score (use classification_report)

7) PCA for dimensionality reduction

a) Choose components
- Option A (visualization): n_components=2 for plotting 2D scatter
- Option B (feature reduction): n_components=0.95 to keep 95% variance (or pick k based on explained variance)

b) Fit and transform
- pca = PCA(n_components=0.95, random_state=42)
- X_train_p = pca.fit_transform(X_train_s)
- X_test_p = pca.transform(X_test_s)

c) Inspect explained variance
- pca.explained_variance_ratio_
- Plot cumulative explained variance: plt.plot(np.cumsum(pca_full.explained_variance_ratio_))

8) Model on PCA features

- Train the same classifier on PCA features:
  - clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
  - clf_pca.fit(X_train_p, y_train)
  - y_pred_p = clf_pca.predict(X_test_p)
- Compare metrics (baseline vs PCA): accuracy, precision, recall, confusion matrix

9) Visualization (required plots)

- PCA scatter (2 components): color points by `good` (target). Use alpha for overplot.
- Explained variance curve (cumulative) showing how many components needed for 90/95%.
- Confusion matrices for baseline and PCA models (side-by-side heatmaps).
- Optional: feature importances from RandomForest (map top features before PCA), and loadings (component weights) to say which original features influence PCs.

10) Short write-up / results section

Include a 1–2 paragraph summary with:
- What you did (preprocessing, PCA config, models trained)
- Baseline vs PCA performance numbers (table or short list)
- Plots inserted and short interpretation (e.g., "PC1 vs PC2 shows partial separation between good and not-good wines; 95% variance kept with k=4 components")
- Limitations and next steps (class imbalance, hyperparameter tuning, cross-validation, alternative dimensionality reduction like t-SNE for visualization)

11) Reproducibility / submission checklist

- `requirements.txt` included
- `main.py` (or `pca_demo.ipynb`) runs end-to-end and creates plots / outputs
- CSV `wine_ucirepo_186.csv` included (or fetch is automatic via `main.py`)
- Short `README.md` explains how to run
- Zip or push the project folder as submission

Commands to include in your README for the grader (PowerShell)

```powershell
# Create venv and install
python -m venv .venv
. .\.venv\Scripts\Activate
pip install -r requirements.txt

# Run demo
python main.py   # fetches dataset and saves CSV
python pca_demo.py   # if you create a script-based demo
# or
jupyter notebook pca_demo.ipynb
```

Tips & small notes (to save time)

- Keep the pipeline minimal but reproducible: data fetch -> preprocessing -> PCA -> model -> plots.
- Log random_state for reproducibility.
- Use stratify=y in train_test_split for classification with imbalanced classes.
- If short on time, deliver a notebook with narrative cells and the three required plots + a one-page conclusion.

Optional enhancements (only if you have time)

- Cross-validated hyperparameter search (GridSearchCV) for the classifier.
- Use PCA loadings to interpret PCs (which original features contribute most).
- Evaluate regression approach (predict raw `quality`) and compare with classification.
- Add unit tests for data loading and a small smoke test that the notebook/script runs.

If you want, I can now:
- create `pca_demo.ipynb` in this folder with the full pipeline ready to run, OR
- create a `pca_demo.py` script that performs the full pipeline and saves figures.

Pick one and I will implement it next (notebook recommended for assignments).