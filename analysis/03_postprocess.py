"""Post-processing on saved outputs: calibration metrics, corrected paired t-tests across folds,
PDP figure (manual), ROC with inset. Usage: python analysis/03_postprocess.py [outputs_dir]"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.pipeline import Pipeline

from common import (load_raw, encode, FEATURES, BIN, LABEL, SEED, style, C_BLUE, C_ORANGE, C_AQUA, C_YELLOW,
                    C_MAGENTA, C_GREEN, C_VIOLET, MUTED, INK2, OUT)

style()
OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
TAB, FIG = OUTD / "tables", OUTD / "figures"
MODELS = ["Lasso Logistic Regression (L1)", "Elastic-Net Logistic Regression", "k-Nearest Neighbours (k-NN)",
          "SVM (RBF Kernel)", "Random Forest", "Gradient Boosting (GBM)", "Random Forest + SMOTE-NC"]
SHORT = dict(zip(MODELS, ["Lasso LR", "Elastic-Net LR", "k-NN", "SVM-RBF", "RF", "GBM", "RF + SMOTE-NC"]))
COLORS = dict(zip(MODELS, [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET]))
STYLES = dict(zip(MODELS, ["-", "--", "-.", ":", "-", "--", "-."]))


def save_table(df, name, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index, float_format="%.4f")
    (TAB / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=".4f"), encoding="utf-8")
    print("[table]", name, df.shape)


# ------------------------------------------------------------------ calibration metrics from saved predictions
preds = pd.read_csv(TAB / "holdout_predictions.csv")
y = preds["y_true"].values


def ece(y, p, bins=5):
    qs = np.quantile(p, np.linspace(0, 1, bins + 1)); qs[-1] += 1e-9
    idx = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, bins - 1)
    e, mx = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            gap = abs(p[m].mean() - y[m].mean()); e += m.mean() * gap; mx = max(mx, gap)
    return e, mx


rows = []
for name in MODELS:
    p = preds[name].values
    e, mx = ece(y, p)
    # Spiegelhalter z-test of calibration
    num = np.sum((y - p) * (1 - 2 * p)); den = np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p)))
    z = num / den if den > 0 else np.nan
    rows.append({"Model Architecture": name, "Brier Score": brier_score_loss(y, p),
                 "Expected calibration error (5 quantile bins)": e, "Maximum calibration error": mx,
                 "Spiegelhalter Z": z, "Spiegelhalter p": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
                 "Mean predicted P(CKD)": p.mean(), "Observed CKD proportion": y.mean()})
cal = pd.DataFrame(rows).sort_values("Brier Score")
save_table(cal, "table15_brier_calibration")

# ------------------------------------------------------------------ corrected resampled t-test across CV folds (Nadeau & Bengio 2003)
folds = pd.read_csv(TAB / "cv_fold_level_results.csv")
n_tr = json.load(open(OUTD / "model_results.json"))["n_train"]
rho = (n_tr / 10) / (n_tr - n_tr / 10)  # n_test/n_train per fold = 1/9
rows = []
L = MODELS[0]
pairs = [(L, m) for m in MODELS[1:]] + [("Elastic-Net Logistic Regression", "Random Forest"),
                                         ("Random Forest", "SVM (RBF Kernel)"), ("Random Forest", "Gradient Boosting (GBM)")]
for a, b in pairs:
    for met, lab in [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("brier", "Brier score")]:
        fa = folds[folds.model == a].sort_values(["repeat", "fold"])[met].values
        fb = folds[folds.model == b].sort_values(["repeat", "fold"])[met].values
        d = fa - fb; k = len(d); md = d.mean(); vd = d.var(ddof=1)
        if vd == 0:
            t = np.nan; p = np.nan
        else:
            t = md / np.sqrt(vd * (1 / k + rho)); p = 2 * stats.t.sf(abs(t), k - 1)
        try:
            w = stats.wilcoxon(fa, fb, zero_method="wilcox").pvalue if np.any(d != 0) else np.nan
        except ValueError:
            w = np.nan
        rows.append({"Model 1": a, "Model 2": b, "Metric": lab, "Mean (Model 1)": fa.mean(), "Mean (Model 2)": fb.mean(),
                     "Mean difference": md, "Corrected t": t, "p-value": p, "Wilcoxon p": w,
                     "Folds": k, "Significance (alpha = 0.05)": "Significant" if (np.isfinite(p) and p < 0.05) else "Not significant"})
save_table(pd.DataFrame(rows), "table14b_corrected_paired_tests")

# ------------------------------------------------------------------ ROC with inset
fig, ax = plt.subplots(figsize=(6.4, 5.8))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
for name in MODELS:
    fpr, tpr, _ = roc_curve(y, preds[name]); ax.plot(fpr, tpr, color=COLORS[name], ls=STYLES[name], lw=1.8,
                                                      label=f"{SHORT[name]} (AUC = {roc_auc_score(y, preds[name]):.4f})")
ax.set_xlabel("1 - Specificity (false positive rate)"); ax.set_ylabel("Sensitivity (true positive rate)")
ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower right")
ax.set_title(f"ROC curves on the 30% holdout test set (n = {len(y)})", loc="left")
ins = ax.inset_axes([0.30, 0.30, 0.38, 0.38])
for name in MODELS:
    fpr, tpr, _ = roc_curve(y, preds[name]); ins.plot(fpr, tpr, color=COLORS[name], ls=STYLES[name], lw=1.6)
ins.set_xlim(-0.005, 0.12); ins.set_ylim(0.88, 1.005); ins.set_title("Upper-left corner (zoom)", fontsize=8); ins.tick_params(labelsize=7)
fig.savefig(FIG / "fig09_roc_holdout.png"); plt.close(fig)

# ------------------------------------------------------------------ PDP (manual) for RF with the tuned hyperparameters
hp = pd.read_csv(TAB / "table11_hyperparameters.csv")
bp = json.loads(hp.loc[hp["Model Architecture"] == "Random Forest", "Selected hyperparameters"].iloc[0])
BIN_IDX = [FEATURES.index(b) for b in BIN]


def round_binary(X):
    X = np.asarray(X, dtype=float).copy(); X[:, BIN_IDX] = np.clip(np.round(X[:, BIN_IDX]), 0, 1); return X


df = encode(load_raw()); X = df[FEATURES]; yy = df["class"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, yy, test_size=0.30, stratify=yy, random_state=SEED)
rf = Pipeline([("impute", SimpleImputer(strategy="median")), ("round", FunctionTransformer(round_binary)),
               ("scale", StandardScaler()),
               ("model", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1,
                                                max_features=bp["model__max_features"], min_samples_leaf=bp["model__min_samples_leaf"]))]).fit(X_tr, y_tr)
imp = pd.read_csv(TAB / "table16_feature_importance.csv")
top4 = list(imp["Code"].head(4))
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)
pd_rows = []
for ax, f in zip(axes, top4):
    obs = X_tr[f].dropna().values
    gx = np.unique(np.quantile(obs, np.linspace(0.02, 0.98, 40))) if len(np.unique(obs)) > 10 else np.unique(obs)
    gy = []
    for v in gx:
        Xg = X_tr.copy(); Xg[f] = v
        gy.append(rf.predict_proba(Xg)[:, 1].mean())
    gy = np.array(gy)
    ax.plot(gx, gy, color=C_ORANGE, lw=2, marker="o" if len(gx) <= 10 else None, ms=4)
    ax.plot(obs, np.full(obs.shape, gy.min() - 0.02), "|", color=INK2, ms=6, alpha=0.3)
    ax.set_xlabel(LABEL[f], fontsize=8); ax.grid(axis="x", visible=False)
    pd_rows.append({"Feature": f, "Grid min": gx.min(), "Grid max": gx.max(), "PD min": gy.min(), "PD max": gy.max(),
                    "PD range": gy.max() - gy.min(),
                    "Value at steepest change": gx[np.argmax(np.abs(np.diff(gy))) + 1] if len(gy) > 1 else np.nan})
axes[0].set_ylabel("Partial dependence (P[CKD])")
fig.suptitle("Partial dependence of the tuned Random Forest CKD probability on its four most influential predictors", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig11_pdp.png"); plt.close(fig)
save_table(pd.DataFrame(pd_rows), "table16b_pdp_summary")
print("postprocess done ->", OUTD)
