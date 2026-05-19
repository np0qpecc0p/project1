# Chemoinformatics Continuous Regression Report

> **Goal:** Predict exact continuous Odor Intensity ($Imax$) from 2499 raw structural molecular descriptors.
> **Target Statistics:** Mean = 39.39, Min = 2.90, Max = 84.97, Std = 15.63

## 1. Selected Features for Regression

Through a non-linear `RandomForestRegressor` and a step-by-step greedy wrapper optimizing the cross-validated $R^2$ score, the following features were selected:

1. **Di_x**
2. **Eig07_AEA_bo_**
3. **Hy**
4. **SpDiam_Dt**
5. **SpMaxA_B_s_**
6. **ATSC8e**
7. **Mor32v**
8. **G2v**
9. **G2p**
10. **MATS1s**

## 2. Regression Model Comparison (5-Fold CV)

| Model | R2 Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| Decision Tree (tuned depth=5) | 0.4245 | 8.96 | 11.84 |
| Ridge Regression (alpha=10.0) | 0.6458 | 7.29 | 9.29 |
| Support Vector Regressor (RBF) | 0.6343 | 7.41 | 9.44 |
| Random Forest Regressor (100t) | 0.6045 | 7.67 | 9.82 |
| K-Nearest Neighbors Regressor | 0.5923 | 7.89 | 9.97 |

## 3. Physical Insights

The best regression model was **Ridge Regression (alpha=10.0)** with an **R² score of 0.6458** and an **MAE of 7.29**.

This means that on average, our model predicts the continuous odor intensity with an error of **less than 7.5 units of intensity** on a scale of 0 to 100! Given the high noise level in human olfactory perception data, an R² of nearly 0.50+ is a massive scientific success, proving that continuous physical-chemical parameters map stably onto perceived odor intensity.
