"""Build the article: CKD_article.md (tables inline) and CKD_article.docx (figures, tables, equations).

Usage (from the 'CKD paper' folder):
    python analysis/04_build.py                    # uses analysis/outputs
    python analysis/04_build.py analysis/outputs_quick
Source parts: CKD_article_src.md (front matter, abstract, intro, methods),
              CKD_article_partC.md (results), CKD_article_partD.md (conclusion, declarations, references).
Markers: {{TABLE:<csv stem>|<caption>}}  {{FIG:<png stem>|<caption>}}
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis" / "outputs"
if not OUT.is_absolute():
    OUT = ROOT / OUT
TAB, FIG = OUT / "tables", OUT / "figures"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)
EQ_DIR = BUILD / "equations"
EQ_DIR.mkdir(exist_ok=True)

PARTS = ["CKD_article_src.md", "CKD_article_partC.md", "CKD_article_partD.md"]
src = "\n\n".join((ROOT / p).read_text(encoding="utf-8") for p in PARTS)
src = re.sub(r"<!--.*?-->", "", src)  # drop build-time comments

# ----------------------------------------------------------------------------- table formatting
GRID_TEXT = {
    "Lasso Logistic Regression (L1)": "C in {0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100}",
    "Elastic-Net Logistic Regression": "C in {0.01, 0.1, 1, 10, 100}; l1 ratio in {0.2, 0.5, 0.8}",
    "k-Nearest Neighbours (k-NN)": "k in {3, 5, 7, 9, 11, 15}",
    "SVM (RBF Kernel)": "C in {0.1, 1, 10, 100}; gamma in {scale, 0.01, 0.1}",
    "Random Forest": "max features in {sqrt(p), 0.5p}; min leaf in {1, 3, 5}",
    "Gradient Boosting (GBM)": "trees in {100, 300}; learning rate in {0.05, 0.1}; depth in {2, 3}",
    "Random Forest + SMOTE-NC": "max features in {sqrt(p), 0.5p}; min leaf in {1, 3}",
}
DROP_COLS = {
    "table12_cv_uncertainty": ["Mean f1", "SD f1", "Mean sensitivity", "SD sensitivity",
                               "Mean specificity", "SD specificity"],
    "table13_holdout_metrics": [],
    "table07_welch_mannwhitney": ["n CKD", "n non-CKD"],
}
RENAME = {
    "table12_cv_uncertainty": {"Mean ROC-AUC": "Mean AUC", "SD ROC-AUC": "SD AUC", "Min ROC-AUC": "Min AUC",
                               "Mean accuracy": "Mean acc.", "SD accuracy": "SD acc.", "Mean mcc": "Mean MCC",
                               "SD mcc": "SD MCC", "Mean brier": "Mean Brier", "SD brier": "SD Brier"},
    "table13_holdout_metrics": {"Accuracy CI lower": "Acc. CI low", "Accuracy CI upper": "Acc. CI high",
                                "AUC CI lower (DeLong)": "AUC CI low", "AUC CI upper (DeLong)": "AUC CI high"},
    "table09_lasso_coefficients_or": {"Lasso coefficient (per SD)": "Coefficient (per SD)",
                                      "Odds ratio (per SD)": "OR (per SD)", "Bootstrap OR 2.5%": "OR 2.5%",
                                      "Bootstrap OR 97.5%": "OR 97.5%",
                                      "Bootstrap selection frequency": "Selection freq."},
    "table16_feature_importance": {"Permutation importance (holdout AUC drop)": "Permutation (AUC drop)",
                                   "Mean |SHAP| (training set)": "Mean |SHAP|"},
    "table07_welch_mannwhitney": {"Significant (alpha=0.05)": "Sig."},
    "table08_chisq_fisher": {"Significant (alpha=0.05)": "Sig.", "Min expected count": "Min expected"},
    "table14_delong": {"Significance (alpha = 0.05)": "Significance"},
}


SUPS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(v):
    """3.61e-76 -> '3.61 × 10⁻⁷⁶' (0 -> '< 1 × 10⁻³⁰⁰' guard)."""
    if v == 0:
        return "< 0.0001"
    mant, exp = f"{v:.2e}".split("e")
    return f"{mant} × 10{str(int(exp)).translate(SUPS)}"


def fmt_num(v, col):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
    if isinstance(v, (float, np.floating)):
        c = col.lower()
        if "p-value" in c or c.endswith(" p") or c == "fisher p" or c == "mw p-value" or "spiegelhalter p" in c:
            return sci(v) if v < 1e-4 else f"{v:.4f}"
        if abs(v) >= 1e5:
            return f"{v:.2e}"
        if float(v).is_integer() and ("(n)" in c or c in {"tp", "fp", "fn", "tn", "missing (n)", "n pairs", "n",
                                                           "ckd positive", "ckd negative", "non-ckd positive",
                                                           "non-ckd negative", "false negatives (n)", "false positives (n)"}):
            return str(int(v))
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.4f}"
    return str(v)


def load_table(stem):
    df = pd.read_csv(TAB / f"{stem}.csv")
    if stem == "table11_hyperparameters":
        df["Hyperparameter grid"] = df["Model Architecture"].map(GRID_TEXT).fillna(df["Hyperparameter grid"])
        df["Selected hyperparameters"] = df["Selected hyperparameters"].apply(
            lambda s: ", ".join(f"{k.replace('model__', '')} = {(f'{v:.4g}' if isinstance(v, float) else v)}" for k, v in json.loads(s).items()) if isinstance(s, str) and s.startswith("{") else s)
    if stem == "table10_unregularised_glm":
        df["Odds Ratio"] = df["Odds Ratio"].astype(float)
    if stem == "table12_cv_uncertainty":
        df["Upper 95% CI"] = df["Upper 95% CI"].clip(upper=1.0)
    for c in DROP_COLS.get(stem, []):
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.rename(columns=RENAME.get(stem, {}))
    if stem == "table10_unregularised_glm":
        df["Odds Ratio"] = df["Odds Ratio"].apply(lambda v: "—" if pd.isna(v) else (f"{v:.2e}" if (v >= 1e4 or v < 1e-3) else f"{v:.4f}"))
    cells = [[fmt_num(v, c) for c, v in zip(df.columns, row)] for row in df.itertuples(index=False)]
    return list(df.columns), cells


def md_table(cols, cells):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join(out)


# ----------------------------------------------------------------------------- Markdown output
def render_md(text):
    def tab(m):
        stem, cap = m.group(1), m.group(2)
        cols, cells = load_table(stem)
        return f"**{cap}**\n\n{md_table(cols, cells)}"

    def fig(m):
        stem, cap = m.group(1), m.group(2)
        return f"![{cap}]({(FIG / (stem + '.png')).relative_to(ROOT).as_posix()})\n\n**{cap}**"

    text = re.sub(r"\{\{TABLE:([^|]+)\|([^}]+)\}\}", tab, text)
    text = re.sub(r"\{\{FIG:([^|]+)\|([^}]+)\}\}", fig, text)
    return text


(ROOT / "CKD_article.md").write_text(render_md(src), encoding="utf-8")

# ----------------------------------------------------------------------------- DOCX output
doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Cm(2.5)
    s.left_margin = s.right_margin = Cm(2.5)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
for name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)]:
    h = doc.styles[name]
    h.font.name = "Times New Roman"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

SUP = re.compile(r"\^([^^]+)\^")
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|\$[^$]+\$)")


def add_runs(par, text):
    """Markdown-ish inline: **bold**, *italic*, ^sup^, $math$ (rendered as italic text)."""
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**"):
            par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("^"):
            par.add_run(tok[1:-1]).font.superscript = True
        elif tok.startswith("$"):
            par.add_run(latex_inline(tok[1:-1])).italic = True
        elif tok.startswith("*"):
            par.add_run(tok[1:-1]).italic = True
        else:
            par.add_run(tok)


GREEK = {r"\beta": "β", r"\alpha": "α", r"\lambda": "λ", r"\gamma": "γ", r"\sigma": "σ", r"\chi": "χ", r"\ell": "ℓ",
         r"\in": "∈", r"\ldots": "…", r"\mid": "|", r"\top": "ᵀ", r"\times": "×"}


def latex_inline(s):
    s = s.replace(r"\bar{X}_1", "X̄₁").replace(r"\bar{X}_2", "X̄₂").replace(r"\bar{d}", "d̄")
    s = re.sub(r"\\hat\{\\sigma\}_d\^2", "σ̂²_d", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\boldsymbol\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\{", "{", s)
    s = re.sub(r"\\\}", "}", s)
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = s.replace("^2", "²").replace("_1", "₁").replace("_2", "₂").replace("_j", "ⱼ").replace("_i", "ᵢ").replace("_0", "₀").replace("_t", "ₜ")
    s = s.replace("{", "").replace("}", "").replace("\\,", " ").replace("\\", "")
    return s


def render_equation(latex, number, path):
    body = re.sub(r"\\tag\{\d+\}", "", latex).strip().rstrip(",.")
    body = body.replace(r"\dfrac", r"\frac").replace(r"\text{", r"\mathrm{")
    body = body.replace(r"\lVert", r"\Vert").replace(r"\rVert", r"\Vert").replace(r"\boldsymbol", r"\mathbf")
    body = body.replace(r"\top", r"\mathsf{T}").replace(r"\,", " ")
    width = 7.0 if len(body) < 120 else 9.5
    fig = plt.figure(figsize=(width, 1.3))
    fig.text(0.46, 0.5, f"${body}$", ha="center", va="center", fontsize=13, family="serif")
    fig.text(0.995, 0.5, f"({number})", ha="right", va="center", fontsize=12, family="serif")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def add_table(cols, cells, caption):
    cap = doc.add_paragraph()
    cap.add_run(caption).bold = True
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    fs = Pt(8) if len(cols) <= 8 else Pt(7)
    for i, c in enumerate(cols):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        r.bold = True
        r.font.size = fs
    for row in cells:
        rc = t.add_row().cells
        for i, v in enumerate(row):
            rc[i].text = ""
            p = rc[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = fs
            if i > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.add_paragraph()


def add_figure(stem, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / f"{stem}.png"), width=Cm(15.5))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.add_run(caption).bold = True
    cap.paragraph_format.space_after = Pt(10)


lines = src.split("\n")
i = 0
eq_n = 0
title_done = False
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln.strip():
        i += 1
        continue
    if ln.startswith("# "):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(ln[2:])
        r.bold = True
        r.font.size = Pt(15)
        title_done = True
    elif ln.startswith("## "):
        doc.add_heading(ln[3:], level=1)
    elif ln.startswith("### "):
        doc.add_heading(ln[4:], level=2)
    elif ln.startswith("#### "):
        doc.add_heading(ln[5:], level=3)
    elif ln.startswith("$$"):
        eq_n += 1
        m = re.search(r"\\tag\{(\d+)\}", ln)
        num = m.group(1) if m else str(eq_n)
        path = EQ_DIR / f"eq{num}.png"
        render_equation(ln.strip("$ "), num, path)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(13))
    elif ln.startswith("{{TABLE:"):
        m = re.match(r"\{\{TABLE:([^|]+)\|([^}]+)\}\}", ln)
        cols, cells = load_table(m.group(1))
        add_table(cols, cells, m.group(2))
    elif ln.startswith("{{FIG:"):
        m = re.match(r"\{\{FIG:([^|]+)\|([^}]+)\}\}", ln)
        add_figure(m.group(1), m.group(2))
    elif re.match(r"^\d+\. ", ln):
        p = doc.add_paragraph(style="List Number")
        add_runs(p, re.sub(r"^\d+\. ", "", ln))
    elif ln.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        add_runs(p, ln[2:])
    else:
        # reference entries get a hanging indent (detected after the References heading)
        p = doc.add_paragraph()
        add_runs(p, ln)
        p.paragraph_format.space_after = Pt(6)
        if in_refs if (in_refs := any(l.strip() == "## References" for l in lines[:i])) else False:
            p.paragraph_format.left_indent = Cm(1)
            p.paragraph_format.first_line_indent = Cm(-1)
        elif not title_done or ln.startswith("[") or ln.startswith("^") or ln.startswith("Corresponding"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    i += 1

doc.save(ROOT / "CKD_article.docx")
print("Wrote", ROOT / "CKD_article.md", "and", ROOT / "CKD_article.docx", "from", OUT)
