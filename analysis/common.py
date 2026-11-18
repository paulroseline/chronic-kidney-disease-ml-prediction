"""Shared data loading, cleaning and encoding for the CKD study."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "kidney_disease.csv"
OUT = BASE / "analysis" / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42

NUM = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
ORD = ["sg", "al", "su"]
BIN = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
FEATURES = NUM + ORD + BIN
BIN_POS = {"rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
           "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"}

LABEL = {
    "age": "Age (years)", "bp": "Blood pressure (mm Hg)", "sg": "Specific gravity",
    "al": "Albumin (0-5)", "su": "Sugar (0-5)", "rbc": "Red blood cells (abnormal)",
    "pc": "Pus cell (abnormal)", "pcc": "Pus cell clumps (present)", "ba": "Bacteria (present)",
    "bgr": "Blood glucose random (mg/dL)", "bu": "Blood urea (mg/dL)",
    "sc": "Serum creatinine (mg/dL)", "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)",
    "hemo": "Haemoglobin (g/dL)", "pcv": "Packed cell volume (%)",
    "wc": "White blood cell count (cells/cumm)", "rc": "Red blood cell count (millions/cmm)",
    "htn": "Hypertension (yes)", "dm": "Diabetes mellitus (yes)", "cad": "Coronary artery disease (yes)",
    "appet": "Appetite (poor)", "pe": "Pedal oedema (yes)", "ane": "Anaemia (yes)",
}

# Validated categorical palette (dataviz reference instance, light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
COL_NEG, COL_POS = C_BLUE, C_ORANGE          # non-CKD = blue, CKD = orange
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 220, "savefig.bbox": "tight",
    })


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df.rename(columns={"__FeatEng_target__": "class"})
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            s = df[c].astype("string").str.strip()
            s = s.where(~s.isin(["?", "nan", "", "None"]), pd.NA)
            df[c] = s
    for c in ["pcv", "wc", "rc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["class"] = (df["class"].str.lower().str.strip() == "ckd").astype(int)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Binary markers -> 1 (clinically adverse level) / 0; ordinal kept numeric."""
    d = df.copy()
    for b in BIN:
        d[b] = d[b].map(lambda v: np.nan if pd.isna(v) else float(v == BIN_POS[b])).astype(float)
    for c in NUM + ORD:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    return d[FEATURES + ["class"]]


def save_table(df: pd.DataFrame, name: str, index=False, floatfmt="%.4f"):
    df.to_csv(TAB / f"{name}.csv", index=index)  # full precision; md is rounded
    with open(TAB / f"{name}.md", "w", encoding="utf-8") as fh:
        fh.write(df.to_markdown(index=index, floatfmt=".4f"))
    print(f"[table] {name}: {df.shape}")


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"
