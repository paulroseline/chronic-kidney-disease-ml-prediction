## 4 Conclusion

This study established a leak-free comparative framework for weighing regularised parametric modelling against non-linear machine learning algorithms in the classification of chronic kidney disease from routine clinical and laboratory markers. Analysis of the benchmark Apollo Hospitals CKD dataset (N = 400; 250 CKD, 150 non-CKD) identified the haematological and urinary concentration markers, specifically haemoglobin (t = -24.36, p = 3.61 × 10^-76^), urinary specific gravity (t = -22.21, p = 1.04 × 10^-67^) and packed cell volume (t = -21.15, p = 2.31 × 10^-61^), as the primary quantitative discriminators between CKD and non-CKD patients, with hypertension (χ² = 135.48) and diabetes mellitus (χ² = 121.26) as the dominant categorical markers, neither of which was recorded in any non-CKD patient. Odds-ratio estimation from the Lasso model placed these findings on an inferential footing: each standard-deviation increase in haemoglobin reduces the odds of CKD to 0.18 of their former value, each standard-deviation increase in urinary albumin raises them 2.34-fold, and diabetes (OR = 1.98) and hypertension (OR = 1.78) approximately double them after adjustment. Every direction is consistent with renal physiology and with the SHAP attributions of the best non-linear model.

With respect to the trade-off between interpretability and predictive performance, the findings are unambiguous. The unregularised generalised linear model fails entirely under quasi-complete separation (standard errors of 10^6^ to 10^7^, every p-value equal to 1), whereas Lasso-penalised Logistic Regression resolves the non-identifiability and yields a sparse 11-predictor model with an excellent predictive profile (99.17% holdout accuracy, 95% CI [95.44%, 99.98%]; ROC-AUC 1.0000; MCC 0.9825). Evaluated on an untouched 30% holdout set (n = 120) within a repeated nested cross-validation in which imputation, scaling, resampling and tuning were confined to training folds, the Lasso model matched the holdout accuracy of SVM-RBF, Random Forest, Gradient Boosting and Random Forest with SMOTE-NC exactly, misclassifying the same single patient at the default threshold, and exceeded that of Elastic-Net (98.33%) and k-NN (92.50%). Across the 30 cross-validation folds, its mean ROC-AUC (0.9980, 95% CI [0.9957, 1.0000]) differed from the ensembles' by less than its own fold-to-fold standard deviation; variance-corrected paired tests found no significant difference in discrimination between the Lasso model and any ensemble (all p ≥ 0.41), and the only estimable DeLong contrast on the holdout set was likewise non-significant (Z = 0.80, p = 0.42). The non-linear learners held a measurable advantage only on the probability scale (holdout Brier scores of 0.0060 to 0.0097 against 0.0182, a gap that the fold-wise tests confirmed as significant for SVM-RBF and both Random Forests), because shrinkage pulls the Lasso probabilities away from the extremes. This cost affects the magnitude of the probabilities rather than their ordering, and for classification purposes it is removed by lowering the decision threshold to 0.30 to 0.40, at which the Lasso classifies every holdout patient correctly. SMOTE-NC resampling conferred no benefit at this degree of imbalance, and the conclusions were unchanged under k-nearest-neighbour or chained-equation imputation. Taken together, the evidence indicates that regularised linear models retain top-tier diagnostic efficacy on routine renal markers without sacrificing mathematical transparency or clinical explainability, in agreement with the broader meta-research finding that fairly compared machine learning rarely outperforms logistic regression on tabular clinical data (Christodoulou et al., 2019; Wang et al., 2026).

Several recommendations follow from these findings. Clinicians screening for CKD should give priority to haemoglobin, packed cell volume, urinary specific gravity and albuminuria, together with the recorded history of hypertension and diabetes, since these markers carry most of the discriminative information and are available from a routine blood count and dipstick urinalysis before serum creatinine is interpreted. Clinical decision support systems should adopt a hybrid architecture in which a regularised logistic model, with its auditable odds ratios, provides transparent first-line screening and tree ensembles are reserved for secondary verification where sharper probability estimates are required. Any such deployment should tune the decision threshold on the training data rather than accept the conventional 0.5 cut-off, should report calibration and fold-wise uncertainty as the TRIPOD+AI statement requires (Collins et al., 2024), and should never impute, resample or tune outside the cross-validation loop. Future research should validate the framework externally on multi-centre cohorts with documented provenance, since a single-centre sample of 400 patients from 2015, in which every adverse binary finding is absent among controls and missingness is informative of class, almost certainly overstates the separability to be expected in unselected primary-care populations. It should also extend the binary task to KDIGO staging and to the prediction of progression to kidney failure (Bai et al., 2022), incorporate decision-curve analysis to express the threshold choice in terms of net clinical benefit (Vickers & Elkin, 2006), and pair the SHAP audit used here with local explanation methods in prospective settings, in line with the growing emphasis on explainable CKD models (Arif et al., 2024).

## Declarations

### Ethics approval and consent to participate

Not applicable. The study analysed a fully anonymised, publicly available benchmark dataset.

### Consent for publication

Not applicable.

### Availability of data and materials

The chronic kidney disease dataset analysed in this study is publicly available from the UCI Machine Learning Repository (dataset 336, https://doi.org/10.24432/C5G020) and from its Kaggle mirror at https://www.kaggle.com/datasets/mansoordaku/ckdisease. All source code, the corrected data file, the analysis scripts, the cross-validation and bootstrap outputs, and every table and figure generated during this study are available in the accompanying repository at [repository URL to be inserted by the authors].

### Competing interests

The authors declare that they have no competing interests.

### Funding

The authors received no external funding for this study.

### Authors' contributions

[Author 1 initials]: Conceptualisation, Methodology, Software, Formal analysis, Investigation, Validation, Visualisation, Data curation, Writing - original draft, Writing - review and editing, Project administration, Supervision. [Author 2 initials]: Methodology, Resources, Investigation, Validation, Data curation, Writing - original draft, Writing - review and editing. All authors read and approved the final manuscript.

### Acknowledgements

Not applicable.

### Authors' information

Not applicable.

## References

Albert, A., & Anderson, J. A. (1984). On the existence of maximum likelihood estimates in logistic regression models. *Biometrika*, *71*(1), 1–10. https://doi.org/10.1093/biomet/71.1.1

Almasoud, M., & Ward, T. E. (2019). Detection of chronic kidney disease using machine learning algorithms with least number of predictors. *International Journal of Advanced Computer Science and Applications*, *10*(8). https://doi.org/10.14569/ijacsa.2019.0100813

Arif, M. S., Rehman, A. U., & Asif, D. (2024). Explainable machine learning model for chronic kidney disease prediction. *Algorithms*, *17*(10), 443. https://doi.org/10.3390/a17100443

Bai, Q., Su, C., Tang, W., & Li, Y. (2022). Machine learning to predict end stage kidney disease in chronic kidney disease. *Scientific Reports*, *12*(1), 8377. https://doi.org/10.1038/s41598-022-12316-z

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/a:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953

Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, *21*(1), 6. https://doi.org/10.1186/s12864-019-6413-7

Chittora, P., Chaurasia, S., Chakrabarti, P., Kumawat, G., Chakrabarti, T., Leonowicz, Z., Jasinski, M., Jasinski, L., Gono, R., Jasinska, E., & Bolshev, V. (2021). Prediction of chronic kidney disease: A machine learning perspective. *IEEE Access*, *9*, 17312–17334. https://doi.org/10.1109/access.2021.3053763

Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y., & Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*, *110*, 12–22. https://doi.org/10.1016/j.jclinepi.2019.02.004

Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., Ghassemi, M., Liu, X., Reitsma, J. B., van Smeden, M., Boulesteix, A.-L., Camaradou, J. C., Celi, L. A., Denaxas, S., Denniston, A. K., Glocker, B., Golub, R. M., Harvey, H., Heinze, G., … Logullo, P. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, *385*, e078378. https://doi.org/10.1136/bmj-2023-078378

Debal, D. A., & Sitote, T. M. (2022). Chronic kidney disease prediction using machine learning techniques. *Journal of Big Data*, *9*(1), 109. https://doi.org/10.1186/s40537-022-00657-5

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*, *44*(3), 837–845. https://doi.org/10.2307/2531595

Dritsas, E., & Trigka, M. (2022). Machine learning techniques for chronic kidney disease risk prediction. *Big Data and Cognitive Computing*, *6*(3), 98. https://doi.org/10.3390/bdcc6030098

Dwivedi, R., Dave, D., Naik, H., Singhal, S., Omer, R., Patel, P., Qian, B., Wen, Z., Shah, T., Morgan, G., & Ranjan, R. (2023). Explainable AI (XAI): Core ideas, techniques, and solutions. *ACM Computing Surveys*, *55*(9), 1–33. https://doi.org/10.1145/3561048

Elhoseny, M., Shankar, K., & Uthayakumar, J. (2019). Intelligent diagnostic prediction and classification system for chronic kidney disease. *Scientific Reports*, *9*(1), 9583. https://doi.org/10.1038/s41598-019-46074-2

Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, *80*(1), 27–38. https://doi.org/10.1093/biomet/80.1.27

GBD Chronic Kidney Disease Collaboration. (2020). Global, regional, and national burden of chronic kidney disease, 1990–2017: A systematic analysis for the Global Burden of Disease Study 2017. *The Lancet*, *395*(10225), 709–733. https://doi.org/10.1016/S0140-6736(20)30045-3

Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, *143*(1), 29–36. https://doi.org/10.1148/radiology.143.1.7063747

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley. https://doi.org/10.1002/9781118548387

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, *4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

Khan, B., Naseem, R., Muhammad, F., Abbas, G., & Kim, S. (2020). An empirical evaluation of machine learning techniques for chronic kidney disease prophecy. *IEEE Access*, *8*, 55012–55022. https://doi.org/10.1109/access.2020.2981689

Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. (2024). KDIGO 2024 clinical practice guideline for the evaluation and management of chronic kidney disease. *Kidney International*, *105*(4S), S117–S314. https://doi.org/10.1016/j.kint.2023.10.018

Kovesdy, C. P. (2022). Epidemiology of chronic kidney disease: An update 2022. *Kidney International Supplements*, *12*(1), 7–11. https://doi.org/10.1016/j.kisu.2021.11.003

Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, *18*(17), 1–5. https://www.jmlr.org/papers/v18/16-365.html

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, *52*(3), 239–281. https://doi.org/10.1023/a:1024068626366

Ogunleye, A., & Wang, Q.-G. (2020). XGBoost model for chronic kidney disease diagnosis. *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, *17*(6), 2131–2140. https://doi.org/10.1109/tcbb.2019.2911071

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

Poonia, R. C., Gupta, M. K., Abunadi, I., Albraikan, A. A., Al-Wesabi, F. N., Hamza, M. A., & Tulasi, B. (2022). Intelligent diagnostic prediction and classification models for detection of kidney disease. *Healthcare*, *10*(2), 371. https://doi.org/10.3390/healthcare10020371

Qin, J., Chen, L., Liu, Y., Liu, C., Feng, C., & Chen, B. (2020). A machine learning methodology for diagnosing chronic kidney disease. *IEEE Access*, *8*, 20991–21002. https://doi.org/10.1109/access.2019.2963053

Rácz, A., Bajusz, D., & Héberger, K. (2021). Effect of dataset size and train/test split ratios in QSAR/QSPR multiclass classification. *Molecules*, *26*(4), 1111. https://doi.org/10.3390/molecules26041111

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, *63*(3), 581–592. https://doi.org/10.1093/biomet/63.3.581

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic kidney disease* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, *1*(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x

Sanmarchi, F., Fanconi, C., Golinelli, D., Gori, D., Hernandez-Boussard, T., & Capodici, A. (2023). Predict, diagnose, and treat chronic kidney disease with machine learning: A systematic literature review. *Journal of Nephrology*, *36*(4), 1101–1117. https://doi.org/10.1007/s40620-023-01573-4

Senan, E. M., Al-Adhaileh, M. H., Alsaade, F. W., Aldhyani, T. H. H., Alqarni, A. A., Alsharif, N., Uddin, M. I., Alahmadi, A. H., Jadhav, M. E., & Alzahrani, M. Y. (2021). Diagnosis of chronic kidney disease using effective classification algorithms and recursive feature elimination techniques. *Journal of Healthcare Engineering*, *2021*, 1004767. https://doi.org/10.1155/2021/1004767

Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials. *Statistics in Medicine*, *5*(5), 421–433. https://doi.org/10.1002/sim.4780050506

Steyerberg, E. W. (2019). *Clinical prediction models: A practical approach to development, validation, and updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., Pencina, M. J., & Kattan, M. W. (2010). Assessing the performance of prediction models: A framework for traditional and novel measures. *Epidemiology*, *21*(1), 128–138. https://doi.org/10.1097/ede.0b013e3181c30fb2

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, *58*(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, *45*(3), 1–67. https://doi.org/10.18637/jss.v045.i03

Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine*, *17*(1), 230. https://doi.org/10.1186/s12916-019-1466-7

Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, *180*, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061

Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making*, *26*(6), 565–574. https://doi.org/10.1177/0272989x06295361

Wang, S.-H., Shao, S.-C., Yang, M.-R., Chen, S. C.-C., & Su, E. C.-Y. (2026). Systematic bias in comparative evaluations of machine learning vs logistic regression for clinical prediction models: A meta-research analysis using trauma mortality as an empirical case. *Journal of Clinical Epidemiology*, *200*, 112508. https://doi.org/10.1016/j.jclinepi.2026.112508

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, *67*(2), 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
