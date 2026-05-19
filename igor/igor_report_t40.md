# Chemoinformatics Experiment Report: Threshold = 40.0

> **Goal:** Predict Odor Intensity Class ($Imax \ge 40.0$) using Raw Molecular Descriptors and Cross-Model Selection.
> **Class Balance:** Balanced (162 Weak / 150 Strong)

## 1. Selected Features

Through Random Forest selection on 2499 raw descriptors and Greedy Cross-Model Search with Logistic Regression, the following optimal features were identified:

1. **Di_x**
2. **Eig08_AEA_bo_**
3. **Mor28i**
4. **RDF020u**
5. **SM2_B_v_**
6. **VE3_B_p_**
7. **SM14_AEA_bo_**
8. **Eig06_EA_ri_**

## 2. LOOCV Model Evaluation

| Model | Accuracy | F1-Score | Precision | Recall | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Decision Tree (tuned) | 78.53% | 0.7759 | 0.7785 | 0.7733 | `TN:129, FP:33, FN:34, TP:116` |
| Logistic Regression (L2) | 78.85% | 0.7785 | 0.7838 | 0.7733 | `TN:130, FP:32, FN:34, TP:116` |
| Support Vector Machine (RBF) | 79.49% | 0.7714 | 0.8308 | 0.7200 | `TN:140, FP:22, FN:42, TP:108` |
| K-Nearest Neighbors (K=5) | 77.24% | 0.7577 | 0.7762 | 0.7400 | `TN:130, FP:32, FN:39, TP:111` |
| Random Forest (100 trees) | 76.60% | 0.7474 | 0.7770 | 0.7200 | `TN:131, FP:31, FN:42, TP:108` |

## 3. Physical Insights
A perfectly balanced threshold of 40.0 provides a much harder challenge. These features represent the most robust, non-linear geometric properties determining high odor intensity.
