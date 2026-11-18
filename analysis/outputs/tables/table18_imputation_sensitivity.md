| Model    | Imputation strategy   |   CV ROC-AUC mean |   CV ROC-AUC SD |   Holdout ROC-AUC |   Holdout accuracy |   Holdout Brier |
|:---------|:----------------------|------------------:|----------------:|------------------:|-------------------:|----------------:|
| Lasso LR | Median / mode         |            0.9996 |          0.0020 |            1.0000 |             0.9917 |          0.0182 |
| Lasso LR | k-NN (k = 5)          |            0.9975 |          0.0070 |            0.9994 |             0.9667 |          0.0207 |
| Lasso LR | Iterative (MICE-type) |            0.9995 |          0.0022 |            0.9988 |             0.9833 |          0.0208 |
| RF       | Median / mode         |            1.0000 |          0.0000 |            1.0000 |             0.9917 |          0.0082 |
| RF       | k-NN (k = 5)          |            0.9991 |          0.0035 |            0.9997 |             0.9917 |          0.0140 |
| RF       | Iterative (MICE-type) |            1.0000 |          0.0000 |            0.9991 |             0.9833 |          0.0146 |