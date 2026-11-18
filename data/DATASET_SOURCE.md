# Dataset: Chronic Kidney Disease (UCI ML Repository dataset 336)

File in this folder: `kidney_disease.csv` (400 rows, 24 predictors + 1 target column named `__FeatEng_target__`;
250 `ckd` / 150 `notckd`). It is the exact file used for every number, table and figure in the article.

## Where to download it

| Source | Link |
|---|---|
| Kaggle mirror (the copy used here) | https://www.kaggle.com/datasets/mansoordaku/ckdisease |
| Original UCI repository entry (DOI) | https://doi.org/10.24432/C5G020 |
| UCI landing page | https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease |

Citation: Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic kidney disease* [Data set].
UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

## Provenance

Collected at Apollo Hospitals, Tamil Nadu, India, over about two months in 2015. The Kaggle mirror
is a CSV conversion of the original ARFF file. Known defects of the public file (stray tab characters,
a leading space in one `dm` entry, and `\t?` tokens in `pcv`, `wc`, `rc`) are NOT edited in the CSV;
they are corrected at load time by `load_raw()` in `analysis/common.py`, so the raw file is kept as downloaded.

## Columns

age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, class.
Continuous: age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc. Ordinal: sg, al, su.
Binary: rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane (encoded 1 = clinically adverse level).
