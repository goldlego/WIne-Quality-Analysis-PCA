# Wine Quality PCA — Presentation

This README is formatted as a presentation you can use directly for an assignment walk-through. It explains the dataset, shows the pipeline (with Mermaid diagrams), lists key results and visuals, and includes concise interpretation and next steps.

---

## Slide 1 — Project & Goal

Goal: apply Principal Component Analysis (PCA) to the UCI Wine Quality dataset and evaluate if PCA-based dimensionality reduction preserves predictive power for a binary quality classifier (good if quality >= 7).

Dataset: chemical measurements for wines (11 numeric features) and a quality score. We create a binary target: good = (quality >= 7).

Key deliverables:

- EDA and preprocessing
- Baseline classifier on scaled features
- PCA (95% variance) + classifier on PCA features
- Visuals: correlation heatmap, explained variance curve, 2-PC scatter, confusion matrices
- A short results summary

---

## Slide 2 — Pipeline (high level)

```mermaid
flowchart LR
  A[Load CSV or fetch dataset (main.py)]
  A --> B[EDA: histograms & correlation heatmap]
  B --> C[Preprocess: train/test split (80/20), StandardScaler]
  C --> D[Baseline: RandomForest on scaled features]
  C --> E[PCA: fit PCA, choose n_components (0.95)]
  E --> F[RandomForest on PCA features]
  D --> G[Compare metrics & save plots]
  F --> G
  G --> H[results_summary.txt]
```

Notes: all steps use fixed random_state for reproducibility (random_state=42) and stratified splitting.

---

## Slide 3 — Preprocessing details

```mermaid
flowchart TB
  X[Raw features (11 dims)]
  X --> S[StandardScaler: mean=0, std=1]
  S --> PCA1[PCA (fit on training set)]
  PCA1 --> Y[Reduced features (k components; k chosen for 95% variance)]
```

Edge cases handled:

- Missing values: dataset has none; code includes imputer fallback (median) if needed.
- Class imbalance: uses stratify=y in train_test_split.

---

## Slide 4 — Models & metrics

Baseline model:

- Algorithm: RandomForestClassifier (n_estimators=100, random_state=42)
- Trained on standardized full feature set

PCA model:

- PCA with n_components=0.95 (keeps ~95% variance)
- Same RandomForest architecture trained on PCA-transformed features

Metrics reported (in `results_summary.txt`):
- Accuracy, precision, recall, f1-score, and confusion matrices for both models

---

## Slide 5 — Visuals (where to find them)

All generated visuals are saved to `pca_outputs/`. Include these images in slides or present them live from the README:

- `feature_histograms.png` — per-feature distributions
- `correlation_heatmap.png` — Pearson correlation matrix
- `explained_variance.png` — cumulative explained variance vs number of components
- `pca_scatter.png` — PC1 vs PC2 scatter colored by target (good vs not)
- `confusion_baseline.png` — baseline model confusion matrix
- `confusion_pca.png` — PCA-model confusion matrix

You can open `pca_outputs/results_summary.txt` for the numeric results.

---

## Slide 6 — Example interpretation (copy into your presentation)

Key findings (example text you can paste into slides):

"The baseline RandomForest trained on standardized features achieved strong classification performance. Applying PCA to preserve 95% of the variance reduced the feature space to k components (reported in the summary) while maintaining comparable accuracy. The 2‑PC scatter shows partial separation between 'good' and 'not good' wines along PC1 and PC2, indicating that the principal components capture relevant chemical variation related to quality. Confusion matrices indicate which class mistakes remain — possible next steps are class rebalancing, hyperparameter tuning, and PCA loading inspection to interpret PC axes." 

---

## Slide 7 — Quick demo & reproduction

PowerShell quickstart (one-time):

```powershell
Set-Location -Path "c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA"
python -m venv .venv
. .\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the demo and produce outputs:

```powershell
python main.py      # fetch and save CSV (if not present)
python pca_demo.py  # run analysis, saves plots to pca_outputs/
```

Alternatively, open `pca_demo_matlab.m` in MATLAB and run it to reproduce the same analysis and display results in MATLAB.

---

## Slide 8 — Limitations & next steps

- The binary target (quality >= 7) simplifies the task — regression or 3-class grouping are alternatives.
- Generated images are included in the repo for convenience; for a clean repo, consider adding `pca_outputs/` to `.gitignore` and regenerating outputs as part of the demo run.
- Next steps: cross-validated hyperparameter tuning, inspect PCA loadings to interpret components, try other dimensionality reduction (t-SNE/UMAP) for visualization.

---

## References

- UCI Wine Quality dataset (id=186)
- scikit-learn PCA and RandomForestClassifier docs

---

If you want, I can:

- Convert this README into a sequence of actual PowerPoint slides (pptx) with the same content and embedded images, or
- Generate a cleaned `pca_demo.ipynb` notebook with narrative cells for submission.

Tell me which output you prefer and I will prepare it next.
  # Wine Quality PCA

  This repository contains a compact, reproducible PCA-based demo for the UCI Wine Quality dataset (id=186). It provides both a small fetch script and an end-to-end demo that performs EDA, preprocessing, PCA, modeling, and saves plots and a short results summary.

  Top-level files

  - `main.py` — fetches the Wine Quality dataset via `ucimlrepo` (id=186) and saves `wine_ucirepo_186.csv`.
  - `pca_demo.py` — end-to-end script: loads CSV (or fetches it), runs EDA, preprocessing, PCA, trains models, and saves outputs to `pca_outputs/`.
  - `wine_ucirepo_186.csv` — the combined dataset (created by `main.py` when run).
  - `requirements.txt` — Python dependencies.
  - `run_demo.bat` — Windows batch file to automate venv creation, install, fetch, and demo run.
  - `Instructions.md` — step-by-step instructions for the assignment.

  ## Quickstart (Windows)

  Open PowerShell, change to the project folder and either run the bundled batch or run commands manually.

  Run the automated batch (recommended on Windows):

  ```powershell
  Set-Location -Path "c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA"
  .\run_demo.bat
  ```

  Manual (PowerShell) — if you prefer to control each step:

  ```powershell
  Set-Location -Path "c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA"
  python -m venv .venv
  . .\.venv\Scripts\Activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  python main.py      # fetch dataset and save CSV
  python pca_demo.py  # run the PCA demo and save outputs
  ```

  If you use WSL/Ubuntu, replace `python` with `python3` and activate with `source .venv/bin/activate`.

  ## What the demo does

  - Loads `wine_ucirepo_186.csv` if present, otherwise fetches the dataset with `ucimlrepo` and saves it.
  - Runs quick EDA (prints shape, head, describe), saves histograms and a correlation heatmap.
  - Converts `quality` into a binary target by default (`good = quality >= 7`), but `pca_demo.py` can be adapted for regression or multi-class.
  - Splits data (stratified), scales features (StandardScaler), trains a baseline RandomForest on scaled features.
  - Fits PCA (visual 2-component scatter and n_components=0.95), trains the same classifier on PCA features, and compares metrics.
  - Saves plots and a short `results_summary.txt` into `pca_outputs/`.

  ## Outputs

  After running `pca_demo.py` you should find the following under `pca_outputs/`:

  - `feature_histograms.png` — feature distributions
  - `correlation_heatmap.png` — correlation matrix heatmap
  - `explained_variance.png` — cumulative explained variance
  - `pca_scatter.png` — PC1 vs PC2 scatter colored by target
  - `confusion_baseline.png` — confusion matrix for baseline model
  - `confusion_pca.png` — confusion matrix for PCA model
  - `results_summary.txt` — short textual comparison (metrics + classification reports)

  ## Running notes & troubleshooting

  - The demo uses `random_state=42` and `stratify=y` for reproducibility.
  - The bundled batch (`run_demo.bat`) creates and activates a `.venv`, installs packages, then runs `main.py` and `pca_demo.py` in sequence. If you prefer PowerShell-style activation or want the venv to remain active in your shell after the script runs, use the manual commands above.
  - The scripts require internet access the first time to fetch the dataset and to download packages. If your environment blocks outbound HTTP, run `main.py` on a machine with network access and copy `wine_ucirepo_186.csv` into the project.

  ## Next steps (optional enhancements)

  - Convert `pca_demo.py` into a Jupyter notebook (`pca_demo.ipynb`) for a more presentable assignment submission.
  - Replace the binary classification target with regression (predict raw `quality`) to make the task distinct from other PCA assignments.
  - Add PCA loadings and a short interpretation paragraph to the results summary to highlight which chemical features drive PC1/PC2.

  If you want, I can implement any of the optional enhancements and update the repository accordingly.

