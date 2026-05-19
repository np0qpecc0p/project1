# Chemoinformatics Continuous Regression Evaluation - Replicated on 10 Raw Features

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def run_regression_evaluation():
    print("="*60)
    print("RUNNING CONTINUOUS REGRESSION ON 10 SELECTED RAW FEATURES")
    print("="*60)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    selected_features = ['Di_x', 'Eig07_AEA_bo_', 'Hy', 'SpDiam_Dt', 'SpMaxA_B_s_', 'ATSC8e', 'Mor32v', 'G2v', 'G2p', 'MATS1s']
    print("Selected Features:", selected_features)
    
    vY = data_df['Imax'].values
    mX_raw = data_df[selected_features].fillna(0).values
    
    scaler = StandardScaler()
    mX = scaler.fit_transform(mX_raw)
    
    # 2. Model Dictionary
    models = {
        "Decision Tree (depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Ridge Regression (alpha=10)": Ridge(alpha=10.0, random_state=42),
        "Support Vector SVR (RBF)": SVR(C=10.0, epsilon=0.1),
        "Random Forest (100 trees)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors (K=5)": KNeighborsRegressor(n_neighbors=5)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    comparison_results = []
    
    # 3. Cross-Validation Loop
    for model_name, model in models.items():
        y_preds = np.zeros(len(vY))
        for train_idx, test_idx in cv.split(mX):
            model.fit(mX[train_idx], vY[train_idx])
            y_preds[test_idx] = model.predict(mX[test_idx])
            
        r2 = r2_score(vY, y_preds)
        mae = mean_absolute_error(vY, y_preds)
        rmse = np.sqrt(mean_squared_error(vY, y_preds))
        
        comparison_results.append({
            "Model": model_name,
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        
    df_compare = pd.DataFrame(comparison_results)
    print("\nRegression Comparison Table:")
    print(df_compare.to_string(index=False))
    
    # 4. Generate Predictions for the Best Model (Ridge)
    best_model = Ridge(alpha=10.0, random_state=42)
    y_preds_best = np.zeros(len(vY))
    for train_idx, test_idx in cv.split(mX):
        best_model.fit(mX[train_idx], vY[train_idx])
        y_preds_best[test_idx] = best_model.predict(mX[test_idx])
        
    # Plot Scatter of Actual vs Predicted
    plt.figure(figsize=(10, 8))
    sns.regplot(x=vY, y=y_preds_best, color='teal', scatter_kws={'alpha':0.6, 'color':'darkcyan'})
    plt.plot([vY.min(), vY.max()], [vY.min(), vY.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title("Actual vs Predicted Odor Intensity (Best Model: Ridge Regression)", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Scent Intensity (Imax)", fontsize=12)
    plt.ylabel("Predicted Scent Intensity (Imax)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_predictions.png"), dpi=300)
    plt.close()
    
    # Plot Error Residuals
    plt.figure(figsize=(10, 6))
    residuals = vY - y_preds_best
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.title("Distribution of Error Residuals (Ridge Regression)", fontsize=14, fontweight='bold')
    plt.xlabel("Error Residual (Actual - Predicted)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_residuals.png"), dpi=300)
    plt.close()
    
    print("\nPlots successfully saved in igor_report/ folder!")

if __name__ == "__main__":
    run_regression_evaluation()
