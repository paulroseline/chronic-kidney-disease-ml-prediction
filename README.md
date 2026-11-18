# CKD classification: Logistic Regression vs non-linear ML

Python project behind the article *"A Comparative Evaluation of Logistic Regression and Non-Linear
Machine Learning Algorithms for the Prediction and Classification of Chronic Kidney Disease"*.
Everything reported in the article (tables 1-18, figures 1-13) is produced by the scripts in `analysis/`
from the single data file in `data/`.

## Dataset

`data/kidney_disease.csv` is the UCI Chronic Kidney Disease dataset (ID 336) as mirrored on Kaggle.

- Kaggle (copy used): https://www.kaggle.com/datasets/mansoordaku/ckdisease
- UCI original: https://doi.org/10.24432/C5G020 (https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

See `data/DATASET_SOURCE.md` for citation, provenance and the column dictionary.

## Project layout

```
ckd_ml_analysis/
├── data/
│   ├── kidney_disease.csv        # the dataset (400 x 25)
│   └── DATASET_SOURCE.md         # download links, citation, columns
├── analysis/
│   ├── common.py                 # loading, cleaning, encoding, palette, table writer
│   ├── 01_eda.py                 # missingness, summaries, tests, VIF, PCA, t-SNE  (Tables 1-8, Figs 1-5)
│   ├── 02_models.py              # 70/30 split, nested repeated 10-fold CV, bootstrap ORs,
│   │                             #   holdout metrics, SHAP, importance (Tables 9-18, Figs 6-13)
│   ├── 03_postprocess.py         # calibration, DeLong, corrected paired t-tests, PDP, ROC
│   ├── 04_build.py               # assembles CKD_article.md / .docx from the three source parts
│   └── outputs/                  # results of the published run (tables/*.csv|md, figures/*.png,
│                                 #   eda_results.json, model_results.json, logs)
├── build/equations/              # equation images used by 04_build.py
├── CKD_article_src.md, CKD_article_partC.md, CKD_article_partD.md   # article source parts
├── run_all.py                    # runs the four stages in order
└── requirements.txt
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Running

```
python analysis/01_eda.py          # ~1-2 min
python analysis/02_models.py       # ~45 min full run; set CKD_QUICK=1 for a ~17 min smoke run
python analysis/03_postprocess.py  # uses analysis/outputs by default
python analysis/04_build.py        # optional: rebuilds CKD_article.md and CKD_article.docx
```

or simply `python run_all.py`. Outputs go to `analysis/outputs/tables` and `analysis/outputs/figures`.
CSV tables are full precision; the `.md` copies are rounded to 4 d.p.

## Methods summary

- Cleaning: whitespace / `?` tokens to missing; binary markers coded 1 for the adverse level (`common.py`).
- Univariate: Welch t and Mann-Whitney for continuous, chi-square / Fisher for binary; VIF; PCA; t-SNE.
- Models: Lasso LR, Elastic-Net LR, unregularised GLM, k-NN, RBF-SVM, Random Forest, Gradient Boosting.
- Leak-free protocol: stratified 70/30 split; imputation (median / mode), SMOTE-NC and scaling fitted
  inside every fold; GridSearchCV inside a 3x10 repeated stratified outer CV; single final evaluation on the holdout.
- Uncertainty: exact binomial CI on accuracy, DeLong CI on AUC, Nadeau-Bengio corrected paired t-tests
  across folds, 500-resample bootstrap for odds-ratio CIs, imputation sensitivity (median / kNN / iterative).
- Interpretation: coefficients and odds ratios, permutation importance, SHAP, partial dependence, threshold analysis.
- Seed: 42 throughout.

## Reproducibility note

The committed `analysis/outputs/` is the run the article was written from (`full_run.log`, ~45 min).
Re-running with the pinned versions reproduces it; small last-digit differences are possible with
other library versions because of changes in solver defaults.
