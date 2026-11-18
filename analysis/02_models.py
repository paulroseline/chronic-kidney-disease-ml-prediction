"""Stage 5 analysis part 2: leak-free model benchmarking.

70/30 stratified split -> all preprocessing (imputation, binary rounding, SMOTE-NC, scaling)
and hyperparameter tuning inside each fold of a repeated stratified 10-fold CV on the training
set (nested CV) -> final tuned models evaluated once on the untouched 30% holdout set.
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_val_score)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
import statsmodels.api as sm
import shap

from common import (load_raw, encode, save_table, style, NUM, ORD, BIN, FEATURES, LABEL, SEED,
                    C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED,
                    COL_NEG, COL_POS, INK2, MUTED, GRID, FIG, TAB, OUT)

warnings.filterwarnings("ignore", category=FutureWarning)
style()
T0 = time.time()
QUICK = bool(os.environ.get("CKD_QUICK"))
N_REP = 1 if QUICK else 3
N_BOOT = 20 if QUICK else 500
N_PERM = 5 if QUICK else 50
BIN_IDX = [FEATURES.index(b) for b in BIN]


def round_binary(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, BIN_IDX] = np.clip(np.round(X[:, BIN_IDX]), 0, 1)
    return X


# ------------------------------------------------------------------ data & split
df = encode(load_raw())
X = df[FEATURES]
y = df["class"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
res = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
       "train_ckd": int(y_tr.sum()), "test_ckd": int(y_te.sum()),
       "train_pct_ckd": 100 * float(y_tr.mean()), "test_pct_ckd": 100 * float(y_te.mean())}
print(res)


def imputer(kind="median"):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "mice":
        return IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False)
    raise ValueError(kind)


def make_pipe(model, smote=False, imp="median"):
    steps = [("impute", imputer(imp)), ("round", FunctionTransformer(round_binary))]
    if smote:
        steps.append(("smote", SMOTENC(categorical_features=BIN_IDX, random_state=SEED, k_neighbors=5)))
    steps += [("scale", StandardScaler()), ("model", model)]
    return Pipeline(steps)


MODELS = {
    "Lasso Logistic Regression (L1)": (
        make_pipe(LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=5000, random_state=SEED)),
        {"model__C": np.logspace(-2, 2, 9)}),
    "Elastic-Net Logistic Regression": (
        make_pipe(LogisticRegression(l1_ratio=0.5, solver="saga", max_iter=20000, random_state=SEED)),
        {"model__C": np.logspace(-2, 2, 5), "model__l1_ratio": [0.2, 0.5, 0.8]}),
    "k-Nearest Neighbours (k-NN)": (
        make_pipe(KNeighborsClassifier()),
        {"model__n_neighbors": [3, 5, 7, 9, 11, 15]}),
    "SVM (RBF Kernel)": (
        make_pipe(SVC(kernel="rbf", probability=True, random_state=SEED)),
        {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (
        make_pipe(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1)),
        {"model__max_features": ["sqrt", 0.5], "model__min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting (GBM)": (
        make_pipe(GradientBoostingClassifier(random_state=SEED)),
        {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}),
    "Random Forest + SMOTE-NC": (
        make_pipe(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1), smote=True),
        {"model__max_features": ["sqrt", 0.5], "model__min_samples_leaf": [1, 3]}),
}
SHORT = {"Lasso Logistic Regression (L1)": "Lasso LR", "Elastic-Net Logistic Regression": "Elastic-Net LR",
         "k-Nearest Neighbours (k-NN)": "k-NN", "SVM (RBF Kernel)": "SVM-RBF", "Random Forest": "RF",
         "Gradient Boosting (GBM)": "GBM", "Random Forest + SMOTE-NC": "RF + SMOTE-NC"}
COLORS = dict(zip(MODELS, [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET]))
STYLES = dict(zip(MODELS, ["-", "--", "-.", ":", "-", "--", "-."]))


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred), "sensitivity": recall_score(y_true, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "precision": precision_score(y_true, pred, zero_division=0), "f1": f1_score(y_true, pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, pred), "roc_auc": roc_auc_score(y_true, p),
            "brier": brier_score_loss(y_true, p), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ------------------------------------------------------------------ DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def _delong(y_true, preds):
    order = np.argsort(-y_true); y_s = y_true[order]; m = int(y_s.sum()); n = len(y_s) - m
    P = preds[:, order]; k = P.shape[0]
    tx = np.array([_midrank(P[r, :m]) for r in range(k)])
    ty = np.array([_midrank(P[r, m:]) for r in range(k)])
    tz = np.array([_midrank(P[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ci(y_true, p, alpha=0.95):
    aucs, cov = _delong(np.asarray(y_true), np.atleast_2d(p))
    se = np.sqrt(cov[0, 0]); z = stats.norm.ppf(0.5 + alpha / 2)
    return aucs[0], max(0, aucs[0] - z * se), min(1, aucs[0] + z * se), se


def delong_test(y_true, p1, p2):
    aucs, cov = _delong(np.asarray(y_true), np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


# ------------------------------------------------------------------ nested repeated CV
outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=N_REP, random_state=SEED)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_rows = []
for name, (pipe, grid) in MODELS.items():
    t = time.time()
    for k, (tr, va) in enumerate(outer.split(X_tr, y_tr)):
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_tr.iloc[tr], y_tr[tr])
        p = gs.predict_proba(X_tr.iloc[va])[:, 1]
        m = metrics(y_tr[va], p)
        m.update({"model": name, "repeat": k // 10, "fold": k % 10, "best_params": json.dumps(gs.best_params_, default=str)})
        fold_rows.append(m)
    print(f"[nested CV] {name}: {time.time() - t:.0f}s")
folds = pd.DataFrame(fold_rows)
folds.to_csv(TAB / "cv_fold_level_results.csv", index=False)

rows = []
for name in MODELS:
    f = folds[folds.model == name]
    r = {"Model Architecture": name, "Mean ROC-AUC": f.roc_auc.mean(), "SD ROC-AUC": f.roc_auc.std(ddof=1)}
    tcrit = stats.t.ppf(0.975, len(f) - 1)
    r["Lower 95% CI"] = r["Mean ROC-AUC"] - tcrit * r["SD ROC-AUC"] / np.sqrt(len(f))
    r["Upper 95% CI"] = r["Mean ROC-AUC"] + tcrit * r["SD ROC-AUC"] / np.sqrt(len(f))
    r["Min ROC-AUC"] = f.roc_auc.min()
    for mname in ["accuracy", "sensitivity", "specificity", "f1", "mcc", "brier"]:
        r[f"Mean {mname}"] = f[mname].mean(); r[f"SD {mname}"] = f[mname].std(ddof=1)
    rows.append(r)
cvtab = pd.DataFrame(rows)
save_table(cvtab, "table12_cv_uncertainty")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
for ax, met, lab in zip(axes, ["roc_auc", "accuracy", "brier"], ["ROC-AUC", "Accuracy", "Brier score"]):
    data = [folds.loc[folds.model == n, met].values for n in MODELS]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops=dict(color="white", linewidth=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED, markeredgecolor="none"))
    for patch, n in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COLORS[n]); patch.set_edgecolor("none")
    ax.set_yticks(range(1, len(MODELS) + 1)); ax.set_yticklabels([SHORT[n] for n in MODELS] if ax is axes[0] else [""] * len(MODELS))
    ax.set_xlabel(lab); ax.grid(axis="y", visible=False); ax.invert_yaxis()
fig.suptitle("Out-of-fold performance across %d x 10 stratified" % N_REP + " cross-validation folds (training set, n = %d)" % len(y_tr),
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig08_cv_boxplots.png"); plt.close(fig)

# ------------------------------------------------------------------ final tuned models on full training set
final = {}; best_rows = []; hold_rows = []; preds = pd.DataFrame({"y_true": y_te})
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for name, (pipe, grid) in MODELS.items():
    gs = GridSearchCV(clone(pipe), grid, cv=cv10, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    final[name] = gs
    p = gs.predict_proba(X_te)[:, 1]
    preds[name] = p
    m = metrics(y_te, p)
    k = int(m["tp"] + m["tn"]); nn = len(y_te)
    ci = stats.binomtest(k, nn).proportion_ci(confidence_level=0.95, method="exact")
    auc, lo, hi, se = delong_ci(y_te, p)
    best_rows.append({"Model Architecture": name, "Hyperparameter grid": json.dumps(grid, default=str),
                      "Selected hyperparameters": json.dumps(gs.best_params_, default=str),
                      "Inner 10-fold CV ROC-AUC": gs.best_score_})
    hold_rows.append({"Model Architecture": name, "Accuracy": m["accuracy"], "Accuracy CI lower": ci.low,
                      "Accuracy CI upper": ci.high, "Sensitivity": m["sensitivity"], "Specificity": m["specificity"],
                      "Precision": m["precision"], "F1-Score": m["f1"], "MCC": m["mcc"], "ROC-AUC": m["roc_auc"],
                      "AUC CI lower (DeLong)": lo, "AUC CI upper (DeLong)": hi, "Brier Score": m["brier"],
                      "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]})
    print(f"[holdout] {name}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} brier={m['brier']:.4f} best={gs.best_params_}")
save_table(pd.DataFrame(best_rows), "table11_hyperparameters")
hold = pd.DataFrame(hold_rows)
save_table(hold, "table13_holdout_metrics")
preds.to_csv(TAB / "holdout_predictions.csv", index=False)

# ROC overlay
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_te, preds[name])
    ax.plot(fpr, tpr, color=COLORS[name], ls=STYLES[name], lw=1.8,
            label=f"{SHORT[name]} (AUC = {roc_auc_score(y_te, preds[name]):.4f})")
ax.set_xlabel("1 - Specificity (false positive rate)"); ax.set_ylabel("Sensitivity (true positive rate)")
ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower right")
ax.set_title(f"ROC curves on the 30% holdout test set (n = {len(y_te)})", loc="left")
fig.savefig(FIG / "fig09_roc_holdout.png"); plt.close(fig)

# Confusion matrices figure
fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.4)); axes = axes.ravel()
for ax, name in zip(axes, MODELS):
    r = hold[hold["Model Architecture"] == name].iloc[0]
    cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("s", ["#ffffff", C_BLUE]), vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK2)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["non-CKD", "CKD"]); ax.set_yticklabels(["non-CKD", "CKD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(SHORT[name], loc="left", fontsize=9); ax.grid(False)
axes[-1].axis("off")
fig.suptitle("Confusion matrices on the holdout test set at the 0.50 decision threshold", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig13_confusion_matrices.png"); plt.close(fig)

# ------------------------------------------------------------------ DeLong paired tests
L = "Lasso Logistic Regression (L1)"
rows = []
for other in MODELS:
    if other == L:
        continue
    a1, a2, z, p = delong_test(y_te, preds[L].values, preds[other].values)
    rows.append({"Model 1": L, "Model 2": other, "AUC 1": a1, "AUC 2": a2, "Z-statistic": z, "p-value": p,
                 "Significance (alpha = 0.05)": "Not significant" if (np.isnan(p) or p >= 0.05) else "Significant"})
for a, b in [("Elastic-Net Logistic Regression", "Random Forest"), ("Random Forest", "Gradient Boosting (GBM)"),
             ("Random Forest", "SVM (RBF Kernel)")]:
    a1, a2, z, p = delong_test(y_te, preds[a].values, preds[b].values)
    rows.append({"Model 1": a, "Model 2": b, "AUC 1": a1, "AUC 2": a2, "Z-statistic": z, "p-value": p,
                 "Significance (alpha = 0.05)": "Not significant" if (np.isnan(p) or p >= 0.05) else "Significant"})
save_table(pd.DataFrame(rows), "table14_delong")

# ------------------------------------------------------------------ calibration
rows = []
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
for name in MODELS:
    p = preds[name].values
    frac, mean_pred = calibration_curve(y_te, p, n_bins=5, strategy="quantile")
    ax.plot(mean_pred, frac, marker="o", ms=5, color=COLORS[name], ls=STYLES[name], lw=1.6, label=SHORT[name])
    # calibration slope / intercept (logistic recalibration)
    eps = 1e-6; lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    try:
        fit = sm.Logit(y_te, sm.add_constant(lo)).fit(disp=0)
        slope, intercept = fit.params[1], fit.params[0]
    except Exception:
        slope, intercept = np.nan, np.nan
    rows.append({"Model Architecture": name, "Brier Score": brier_score_loss(y_te, p),
                 "Calibration slope": slope, "Calibration intercept": intercept})
ax.set_xlabel("Mean predicted probability of CKD (quintile bins)"); ax.set_ylabel("Observed proportion of CKD")
ax.legend(loc="upper left"); ax.set_title("Reliability diagram on the holdout test set", loc="left")
fig.savefig(FIG / "fig10_calibration.png"); plt.close(fig)
cal = pd.DataFrame(rows).sort_values("Brier Score")
save_table(cal, "table15_brier_calibration")

# ------------------------------------------------------------------ Lasso: coefficients, ORs, bootstrap CIs, path
lasso = final[L].best_estimator_
coef = lasso.named_steps["model"].coef_.ravel()
bestC = final[L].best_params_["model__C"]
rng = np.random.default_rng(SEED)
B = N_BOOT; boot = np.zeros((B, len(FEATURES)))
Xtr_np = X_tr.reset_index(drop=True)
for b in range(B):
    idx = rng.integers(0, len(y_tr), len(y_tr))
    while len(np.unique(y_tr[idx])) < 2:
        idx = rng.integers(0, len(y_tr), len(y_tr))
    m = clone(lasso).fit(Xtr_np.iloc[idx], y_tr[idx])
    boot[b] = m.named_steps["model"].coef_.ravel()
lo_, hi_ = np.percentile(boot, [2.5, 97.5], axis=0)
sel_freq = (np.abs(boot) > 1e-8).mean(axis=0)
lt = pd.DataFrame({"Term": [LABEL[f] for f in FEATURES], "Code": FEATURES, "Lasso coefficient (per SD)": coef,
                   "Odds ratio (per SD)": np.exp(coef), "Bootstrap OR 2.5%": np.exp(lo_), "Bootstrap OR 97.5%": np.exp(hi_),
                   "Bootstrap selection frequency": sel_freq})
lt = lt.sort_values("Lasso coefficient (per SD)", key=np.abs, ascending=False)
save_table(lt, "table09_lasso_coefficients_or")
res["lasso_C"] = float(bestC); res["lasso_nonzero"] = int((np.abs(coef) > 1e-8).sum())

# Elastic-net coefficients as well
en = final["Elastic-Net Logistic Regression"].best_estimator_.named_steps["model"].coef_.ravel()
save_table(pd.DataFrame({"Code": FEATURES, "Elastic-Net coefficient (per SD)": en, "OR (per SD)": np.exp(en)})
           .sort_values("Elastic-Net coefficient (per SD)", key=np.abs, ascending=False), "table09b_elasticnet_coefficients")

# Forest plot of retained features
ret = lt[np.abs(lt["Lasso coefficient (per SD)"]) > 1e-8].sort_values("Lasso coefficient (per SD)")
fig, ax = plt.subplots(figsize=(7, 0.32 * len(ret) + 1.4))
yy = np.arange(len(ret))
ax.axvline(1, color=MUTED, lw=1, ls="--")
ax.hlines(yy, ret["Bootstrap OR 2.5%"], ret["Bootstrap OR 97.5%"], color=INK2, lw=1.2)
ax.scatter(ret["Odds ratio (per SD)"], yy, color=[COL_POS if v > 1 else COL_NEG for v in ret["Odds ratio (per SD)"]], s=36, zorder=3)
ax.set_xscale("log"); ax.set_yticks(yy); ax.set_yticklabels(ret["Term"]); ax.set_xlabel("Odds ratio per 1 SD increase (log scale), 95% bootstrap CI")
ax.grid(axis="y", visible=False); ax.set_title("Lasso-retained predictors of CKD: regularised odds ratios", loc="left")
fig.savefig(FIG / "fig07_forest_or.png"); plt.close(fig)

# Regularisation path figure
Cs = np.logspace(-3, 2, 40)
pre = Pipeline(lasso.steps[:-1]).fit(X_tr, y_tr)
Ztr = pre.transform(X_tr)
paths = np.array([LogisticRegression(l1_ratio=1.0, solver="liblinear", C=c, max_iter=5000).fit(Ztr, y_tr).coef_.ravel() for c in Cs])
cvr = pd.DataFrame(final[L].cv_results_)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.errorbar(cvr["param_model__C"].astype(float), cvr["mean_test_score"], yerr=cvr["std_test_score"], color=C_BLUE,
            marker="o", ms=4, capsize=3, lw=1.5)
ax.axvline(bestC, color=C_ORANGE, ls="--", lw=1.2, label=f"Selected C = {bestC:g}")
ax.set_xscale("log"); ax.set_xlabel("Inverse regularisation strength C (log scale)"); ax.set_ylabel("10-fold CV ROC-AUC (mean +/- SD)")
ax.legend(loc="lower right"); ax.set_title("Cross-validated discrimination across the L1 penalty grid", loc="left")
ax = axes[1]
order = np.argsort(-np.abs(paths[-1]))
for j in range(len(FEATURES)):
    ax.plot(Cs, paths[:, j], color=C_BLUE if j in order[:6] else "#c3c2b7", lw=1.6 if j in order[:6] else 0.9)
    if j in order[:6]:
        ax.text(Cs[-1] * 1.1, paths[-1, j], FEATURES[j], fontsize=7.5, va="center", color=INK2)
ax.axvline(bestC, color=C_ORANGE, ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("Inverse regularisation strength C (log scale)"); ax.set_ylabel("Standardised coefficient")
ax.set_title("Coefficient shrinkage profile (six largest labelled)", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig06_lasso_path.png"); plt.close(fig)

# ------------------------------------------------------------------ unregularised GLM (separation diagnostics)
Zdf = pd.DataFrame(Ztr, columns=FEATURES)
glm_rows = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        glm = sm.Logit(y_tr, sm.add_constant(Zdf)).fit(method="newton", maxiter=200, disp=0)
        conv = glm.mle_retvals.get("converged", None)
    except Exception as e:
        glm = sm.Logit(y_tr, sm.add_constant(Zdf)).fit(method="bfgs", maxiter=2000, disp=0)
        conv = f"newton failed ({type(e).__name__}); bfgs used"
    glm_warn = [str(x.message)[:120] for x in w]
for term in glm.params.index:
    glm_rows.append({"Term": term, "Estimate": glm.params[term], "Std. Error": glm.bse[term],
                     "Odds Ratio": np.exp(glm.params[term]), "z": glm.tvalues[term], "p-value": glm.pvalues[term]})
save_table(pd.DataFrame(glm_rows), "table10_unregularised_glm")
res["glm_converged"] = str(conv); res["glm_warnings"] = glm_warn[:5]
res["glm_max_se"] = float(glm.bse.max()); res["glm_min_p"] = float(glm.pvalues.drop("const").min())
res["glm_train_auc"] = float(roc_auc_score(y_tr, glm.predict(sm.add_constant(Zdf))))

# ------------------------------------------------------------------ importance: permutation + SHAP (RF)
rf = final["Random Forest"].best_estimator_
pi = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=N_PERM, random_state=SEED, n_jobs=-1)
rf_pre = Pipeline(rf.steps[:-1])
Ztr_rf = rf_pre.transform(X_tr)
expl = shap.TreeExplainer(rf.named_steps["model"])
sv = expl.shap_values(Ztr_rf)
if isinstance(sv, list):
    sv = sv[1]
elif np.ndim(sv) == 3:
    sv = sv[:, :, 1]
imp = pd.DataFrame({"Variable": [LABEL[f] for f in FEATURES], "Code": FEATURES,
                    "Gini importance (RF)": rf.named_steps["model"].feature_importances_,
                    "Permutation importance (holdout AUC drop)": pi.importances_mean,
                    "Permutation SD": pi.importances_std,
                    "Mean |SHAP| (training set)": np.abs(sv).mean(axis=0)}).sort_values("Mean |SHAP| (training set)", ascending=False)
save_table(imp, "table16_feature_importance")

fig = plt.figure(figsize=(7.5, 6.5))
shap.summary_plot(sv, Ztr_rf, feature_names=FEATURES, show=False, max_display=15,
                  cmap=LinearSegmentedColormap.from_list("bo", [C_BLUE, "#d9d8d2", C_ORANGE]), plot_size=None)
ax = plt.gca(); ax.set_xlabel("SHAP value (impact on Random Forest CKD probability)")
ax.set_title("SHAP summary of the tuned Random Forest (training set, standardised predictors)", loc="left", fontsize=10)
fig.savefig(FIG / "fig12_shap_summary.png"); plt.close(fig)

# PDP for the top 4 SHAP features (RF pipeline, raw feature scale)
top4 = list(imp["Code"].head(4))
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
disp = PartialDependenceDisplay.from_estimator(rf, X_tr, features=top4, kind="average", ax=axes,
                                               line_kw={"color": C_ORANGE, "lw": 2}, grid_resolution=50)
for ax, f in zip(axes, top4):
    ax.set_xlabel(LABEL[f], fontsize=8); ax.set_ylabel("Partial dependence (P[CKD])" if ax is axes[0] else "")
fig.suptitle("Partial dependence of the Random Forest CKD probability on its four most influential predictors", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig11_pdp.png"); plt.close(fig)

# ------------------------------------------------------------------ threshold analysis (RF and Lasso)
rows = []
for name in ["Random Forest", L]:
    p = preds[name].values
    for thr in list(np.arange(0.05, 0.96, 0.05)):
        m = metrics(y_te, p, thr=thr)
        rows.append({"Model": SHORT[name], "Threshold": round(float(thr), 2), "Sensitivity": m["sensitivity"],
                     "Specificity": m["specificity"], "Precision": m["precision"], "F1-Score": m["f1"], "MCC": m["mcc"],
                     "False negatives (n)": m["fn"], "False positives (n)": m["fp"]})
save_table(pd.DataFrame(rows), "table17_threshold_analysis")

# ------------------------------------------------------------------ imputation sensitivity (Lasso & RF, fixed tuned params)
rows = []
for name in [L, "Random Forest"]:
    bp = final[name].best_params_
    for imp_kind in ["median", "knn", "mice"]:
        base = MODELS[name][0]
        pipe = make_pipe(clone(base.named_steps["model"]), smote=False, imp=imp_kind).set_params(**bp)
        sc = cross_val_score(pipe, X_tr, y_tr, cv=outer, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr); ph = pipe.predict_proba(X_te)[:, 1]
        rows.append({"Model": SHORT[name], "Imputation strategy": {"median": "Median / mode", "knn": "k-NN (k = 5)", "mice": "Iterative (MICE-type)"}[imp_kind],
                     "CV ROC-AUC mean": sc.mean(), "CV ROC-AUC SD": sc.std(ddof=1),
                     "Holdout ROC-AUC": roc_auc_score(y_te, ph), "Holdout accuracy": accuracy_score(y_te, ph >= 0.5),
                     "Holdout Brier": brier_score_loss(y_te, ph)})
save_table(pd.DataFrame(rows), "table18_imputation_sensitivity")

res["runtime_s"] = time.time() - T0
with open(OUT / "model_results.json", "w") as fh:
    json.dump(res, fh, indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
print("DONE in %.0fs" % res["runtime_s"])
