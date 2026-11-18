"""Stage 5 analysis part 1: descriptive statistics, missingness, inference, correlation, VIF, PCA, t-SNE."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import Table2x2

from common import (load_raw, encode, save_table, fmt_p, style, NUM, ORD, BIN, FEATURES, LABEL,
                    COL_NEG, COL_POS, SEQ_BLUE, C_BLUE, C_RED, MUTED, INK2, FIG, OUT, SEED)

style()
raw = load_raw()
df = encode(raw)
y = df["class"]
n = len(df)
res = {"n": n, "n_ckd": int(y.sum()), "n_notckd": int((1 - y).sum()),
       "pct_ckd": 100 * y.mean(), "pct_notckd": 100 * (1 - y.mean())}
print(res)

# ---------------- Table 1: missingness ----------------
miss = pd.DataFrame({
    "Variable": [LABEL[c] for c in FEATURES], "Code": FEATURES,
    "Missing (n)": [int(df[c].isna().sum()) for c in FEATURES],
    "Missing (%)": [100 * df[c].isna().mean() for c in FEATURES],
    "Missing CKD (%)": [100 * df.loc[y == 1, c].isna().mean() for c in FEATURES],
    "Missing non-CKD (%)": [100 * df.loc[y == 0, c].isna().mean() for c in FEATURES],
}).sort_values("Missing (%)", ascending=False)
save_table(miss, "table01_missingness")
res["complete_cases"] = int(df[FEATURES].dropna().shape[0])
res["rows_any_missing"] = n - res["complete_cases"]

fig, ax = plt.subplots(figsize=(6.5, 5.2))
m2 = miss.sort_values("Missing (%)")
ax.barh(m2["Variable"], m2["Missing (%)"], color=C_BLUE, height=0.62)
for i, v in enumerate(m2["Missing (%)"]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7.5, color=INK2)
ax.set_xlabel("Observations missing (%)"); ax.set_xlim(0, 45); ax.grid(axis="y", visible=False)
ax.set_title("Proportion of missing values per clinical variable (N = 400)", loc="left")
fig.savefig(FIG / "fig01_missingness.png"); plt.close(fig)

# ---------------- Table 2: numeric summaries ----------------
rows = []
for c in NUM + ORD:
    s = df[c].dropna()
    rows.append({"Variable": LABEL[c], "n": len(s), "Min": s.min(), "Q1": s.quantile(.25), "Median": s.median(),
                 "Mean": s.mean(), "SD": s.std(ddof=1), "Q3": s.quantile(.75), "Max": s.max(),
                 "Skewness": stats.skew(s)})
save_table(pd.DataFrame(rows), "table02_numeric_summary")

# ---------------- Table 3: categorical frequencies by class ----------------
rows = []
for b in BIN:
    s = df[b]
    for grp, name in [(1, "CKD"), (0, "non-CKD")]:
        pass
    a1 = s[y == 1].dropna(); a0 = s[y == 0].dropna()
    rows.append({"Marker": LABEL[b], "CKD n/N": f"{int(a1.sum())}/{len(a1)}", "CKD (%)": 100 * a1.mean(),
                 "non-CKD n/N": f"{int(a0.sum())}/{len(a0)}", "non-CKD (%)": 100 * a0.mean(),
                 "Overall (%)": 100 * s.dropna().mean()})
save_table(pd.DataFrame(rows), "table03_categorical_by_class")

# ---------------- Figure 2: class-stratified densities ----------------
fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
for ax, c in zip(axes.ravel(), ["sc", "bu", "hemo", "pcv", "sg", "al"]):
    for grp, col, lab in [(0, COL_NEG, "non-CKD"), (1, COL_POS, "CKD")]:
        s = df.loc[y == grp, c].dropna()
        if c in ("sg", "al"):
            vals, cnts = np.unique(s, return_counts=True)
            off = -0.15 if grp == 0 else 0.15
            w = 0.3 if c == "al" else 0.0015
            ax.bar(np.arange(len(vals)) + off if c == "al" else vals + off * 0.01, cnts / cnts.sum(),
                   width=w, color=col, label=lab, alpha=0.9)
            if c == "al":
                ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels([f"{v:.0f}" for v in vals])
            ax.set_ylabel("Proportion")
        else:
            kde = stats.gaussian_kde(s)
            xs = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=col, label=lab)
            ax.fill_between(xs, kde(xs), color=col, alpha=0.15)
            ax.set_ylabel("Density")
    ax.set_title(LABEL[c], loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
axes[0, 0].legend(loc="upper right")
fig.suptitle("Distribution of key renal and haematological markers by diagnostic class", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig02_densities.png"); plt.close(fig)

# ---------------- Correlation (Pearson, pairwise complete) ----------------
corr = df[FEATURES].corr(method="pearson")
pairs = []
for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) >= 0.60:
            nn = df[[a, b]].dropna().shape[0]
            pairs.append({"Feature 1": LABEL[a], "Feature 2": LABEL[b], "Pearson r": r, "n pairs": nn})
pairs = pd.DataFrame(pairs).sort_values("Pearson r", key=np.abs, ascending=False)
save_table(pairs, "table04_correlated_pairs")

fig, ax = plt.subplots(figsize=(8.2, 7))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", [C_BLUE, "#f0efec", C_RED])
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES))); ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=90, fontsize=8); ax.set_yticklabels(FEATURES, fontsize=8)
ax.grid(False)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        v = corr.values[i, j]
        if abs(v) >= 0.5 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.7 else INK2)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Pearson correlation (r)")
ax.set_title("Pairwise correlation among the 24 encoded clinical predictors", loc="left")
fig.savefig(FIG / "fig03_corr_heatmap.png"); plt.close(fig)

# ---------------- Table 5: VIF (median/mode-imputed, descriptive) ----------------
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[FEATURES]), columns=FEATURES)
Xc = np.column_stack([np.ones(len(Ximp)), Ximp.values])
vif = pd.DataFrame({"Variable": [LABEL[c] for c in FEATURES], "Code": FEATURES,
                    "VIF": [variance_inflation_factor(Xc, i + 1) for i in range(len(FEATURES))]})
vif["Tolerance"] = 1 / vif["VIF"]
vif = vif.sort_values("VIF", ascending=False)
save_table(vif, "table05_vif")

# ---------------- PCA & t-SNE (descriptive, full data, median-imputed) ----------------
Z = StandardScaler().fit_transform(Ximp)
pca = PCA(n_components=min(10, Z.shape[1]), random_state=SEED).fit(Z)
pcs = pca.transform(Z)
pct = pd.DataFrame({
    "Metric": ["Standard deviation", "Proportion of variance", "Cumulative proportion"]})
for k in range(6):
    pct[f"PC{k+1}"] = [np.sqrt(pca.explained_variance_[k]), pca.explained_variance_ratio_[k],
                       pca.explained_variance_ratio_[:k + 1].sum()]
save_table(pct, "table06_pca_variance")
load = pd.DataFrame(pca.components_[:2].T, index=FEATURES, columns=["PC1", "PC2"])
save_table(load.reset_index().rename(columns={"index": "Code"}), "table06b_pca_loadings")

fig, ax = plt.subplots(figsize=(6.8, 5.4))
for grp, col, lab in [(0, COL_NEG, "non-CKD"), (1, COL_POS, "CKD")]:
    ax.scatter(pcs[y == grp, 0], pcs[y == grp, 1], s=22, color=col, alpha=0.75, label=lab,
               edgecolor=SURFACE if False else "white", linewidth=0.4)
sc_ = 6.0
top = load.apply(np.linalg.norm, axis=1).sort_values(ascending=False).index[:8]
for f in top:
    ax.annotate("", xy=(load.loc[f, "PC1"] * sc_, load.loc[f, "PC2"] * sc_), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(load.loc[f, "PC1"] * sc_ * 1.12, load.loc[f, "PC2"] * sc_ * 1.12, f, fontsize=8, color=INK2)
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% of variance)")
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% of variance)")
ax.legend(loc="upper right"); ax.set_title("PCA biplot of the standardised clinical predictors", loc="left")
fig.savefig(FIG / "fig04_pca_biplot.png"); plt.close(fig)

ts = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(Z)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for grp, col, lab in [(0, COL_NEG, "non-CKD"), (1, COL_POS, "CKD")]:
    ax.scatter(ts[y == grp, 0], ts[y == grp, 1], s=22, color=col, alpha=0.8, label=lab, edgecolor="white", linewidth=0.4)
ax.set_xlabel("t-SNE dimension 1"); ax.set_ylabel("t-SNE dimension 2"); ax.legend()
ax.set_title("t-SNE embedding of the 24 standardised predictors (perplexity = 30)", loc="left")
fig.savefig(FIG / "fig05_tsne.png"); plt.close(fig)

# ---------------- Table 7: Welch t and Mann-Whitney for numeric/ordinal ----------------
rows = []
for c in NUM + ORD:
    a = df.loc[y == 1, c].dropna(); b = df.loc[y == 0, c].dropna()
    t = stats.ttest_ind(a, b, equal_var=False)
    ci = t.confidence_interval(0.95)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"Variable": LABEL[c], "Code": c, "n CKD": len(a), "n non-CKD": len(b),
                 "Mean CKD": a.mean(), "Mean non-CKD": b.mean(), "Difference": a.mean() - b.mean(),
                 "t-statistic": t.statistic, "df": t.df, "p-value": t.pvalue,
                 "CI lower": ci.low, "CI upper": ci.high, "Mann-Whitney U": u.statistic, "MW p-value": u.pvalue,
                 "Significant (alpha=0.05)": "Yes" if t.pvalue < 0.05 else "No"})
welch = pd.DataFrame(rows).sort_values("t-statistic", key=np.abs, ascending=False)
save_table(welch, "table07_welch_mannwhitney")

# ---------------- Table 8: chi-square / Fisher with odds ratios for binary ----------------
rows = []
for b in BIN:
    sub = df[[b]].assign(y=y).dropna()
    ct = pd.crosstab(sub["y"], sub[b]).reindex(index=[1, 0], columns=[1.0, 0.0], fill_value=0)
    tab = ct.values  # rows: CKD, non-CKD ; cols: positive, negative
    chi2, p, dof, exp = stats.chi2_contingency(tab, correction=True)
    fisher = stats.fisher_exact(tab)
    t22 = Table2x2(tab + 0.5 if (tab == 0).any() else tab)
    orr = t22.oddsratio; lo, hi = t22.oddsratio_confint()
    rows.append({"Marker": LABEL[b], "Code": b, "CKD positive": int(tab[0, 0]), "CKD negative": int(tab[0, 1]),
                 "non-CKD positive": int(tab[1, 0]), "non-CKD negative": int(tab[1, 1]),
                 "Chi-square": chi2, "Chi-square p": p, "Fisher p": fisher.pvalue, "Min expected count": exp.min(),
                 "Odds ratio": orr, "OR CI lower": lo, "OR CI upper": hi,
                 "Significant (alpha=0.05)": "Yes" if fisher.pvalue < 0.05 else "No"})
chi = pd.DataFrame(rows).sort_values("Chi-square", ascending=False)
save_table(chi, "table08_chisq_fisher")

res["welch_top"] = welch.head(5)[["Code", "t-statistic", "p-value"]].to_dict("records")
res["chisq_top"] = chi.head(5)[["Code", "Chi-square", "Fisher p", "Odds ratio"]].to_dict("records")
res["max_abs_r"] = pairs.iloc[0].to_dict() if len(pairs) else None
res["vif_top"] = vif.head(5)[["Code", "VIF"]].to_dict("records")
res["pca_cum2"] = float(pca.explained_variance_ratio_[:2].sum())
res["pca_cum6"] = float(pca.explained_variance_ratio_[:6].sum())
with open(OUT / "eda_results.json", "w") as fh:
    json.dump(res, fh, indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
