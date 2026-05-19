# Consensus Stability Feature Selection and Full Data Regression Experiment
# Goal: Run 10-Fold Feature Selection, select features chosen > 3 times (consensus features),
# train final models on the entire dataset using only these stable features, and evaluate.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def select_features_subset(X_raw, y, feature_names):
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Phase 1: RF Screener
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    # Phase 2: Greedy Forward Selection (Ridge Validator)
    candidate_pool = sorted_features[:50]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validator = Ridge(alpha=10.0, random_state=42)
    
    selected_features = []
    current_best_r2 = -999.0
    
    for i in range(10):
        best_feat_to_add = None
        best_r2_step = current_best_r2
        
        for candidate in candidate_pool:
            if candidate in selected_features:
                continue
                
            test_subset = selected_features + [candidate]
            test_indices = [feature_names.index(f) for f in test_subset]
            mX_test = X[:, test_indices]
            
            y_preds = np.zeros(len(y))
            for train_idx, test_idx in cv.split(mX_test):
                validator.fit(mX_test[train_idx], y[train_idx])
                y_preds[test_idx] = validator.predict(mX_test[test_idx])
                
            score = r2_score_inner(y, y_preds)
            if score > best_r2_step:
                best_r2_step = score
                best_feat_to_add = candidate
                
        if best_feat_to_add:
            selected_features.append(best_feat_to_add)
            current_best_r2 = best_r2_step
        else:
            break
            
    return selected_features

def r2_score_inner(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))

def main():
    print("="*80)
    print("STARTING 10-FOLD CONSENSUS STABILITY FEATURE SELECTION & EVALUATION")
    print("="*80)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    vY = data_df['Imax'].values
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    mX_raw = data_df[feature_names].fillna(0).values
    
    # 2. Run 10-Fold Feature Selection
    print("\n[Step 1] Running feature selection across 10 folds...")
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_selected_features = {}
    all_chosen_features = []
    
    for fold_idx, (train_idx, _) in enumerate(cv_outer.split(mX_raw)):
        fold_name = f"Fold {fold_idx + 1}"
        X_train, y_train = mX_raw[train_idx], vY[train_idx]
        selected = select_features_subset(X_train, y_train, feature_names)
        fold_selected_features[fold_name] = selected
        all_chosen_features.extend(selected)
        print(f"  -> {fold_name} Selected Features: {selected}")
        
    # 3. Calculate frequencies and filter consensus features (frequency > 3)
    feature_counts = pd.Series(all_chosen_features).value_counts()
    print("\nDescriptor selection counts across 10 folds:")
    for feat, count in feature_counts.items():
        print(f"  - '{feat}': selected in {count} out of 10 folds")
        
    consensus_features = [feat for feat, count in feature_counts.items() if count > 3]
    print(f"\n[Step 2] Filtering Consensus Features (selected > 3 times out of 10 folds):")
    print(f"=> Consensus Features ({len(consensus_features)} features): {consensus_features}")
    
    # 4. Generate 10-Fold Heatmap
    unique_features = sorted(list(set(all_chosen_features)))
    matrix_data = {f"Fold {i+1}": [] for i in range(10)}
    for feat in unique_features:
        for i in range(10):
            fold_name = f"Fold {i+1}"
            if feat in fold_selected_features[fold_name]:
                matrix_data[fold_name].append(1)
            else:
                matrix_data[fold_name].append(0)
                
    df_stability = pd.DataFrame(matrix_data, index=unique_features)
    df_stability["Selection Frequency"] = df_stability.sum(axis=1)
    df_stability = df_stability.sort_values(by="Selection Frequency", ascending=False)
    
    df_plot = df_stability.drop(columns=["Selection Frequency"])
    
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        df_plot, 
        cmap=sns.color_palette(["#f3f4f6", "#10b981"]),  # Green for selected!
        cbar=False, 
        linewidths=1.5, 
        linecolor="#e5e7eb",
        annot=True,
        fmt="d",
        annot_kws={"size": 11, "weight": "bold", "color": "white"}
    )
    plt.title("10-Fold Feature Selection Stability Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Evaluation Fold Split", fontsize=13, labelpad=10)
    plt.ylabel("Dragon Molecular Descriptor", fontsize=13, labelpad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_stability_10folds.png"), dpi=300)
    plt.close()
    print(f"-> 10-fold stability heatmap successfully saved as feature_stability_10folds.png")
    
    # 5. Predict Full Data using ONLY Consensus Features
    print("\n[Step 3] Training and predicting Full Dataset with Consensus Features...")
    
    # Scale entire dataset
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(mX_raw)
    
    consensus_indices = [feature_names.index(f) for f in consensus_features]
    X_consensus = X_full_scaled[:, consensus_indices]
    
    models = {
        "Ridge Regression (alpha=10)": Ridge(alpha=10.0, random_state=42),
        "Support Vector Regressor (SVR)": SVR(C=10.0, epsilon=0.1),
        "K-Nearest Neighbors (KNN, K=5)": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor (Depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    print("\nFull Data Performance using Consensus Features:")
    for model_name, model in models.items():
        # Train on 100% of data
        model.fit(X_consensus, vY)
        y_pred = model.predict(X_consensus)
        
        r2 = r2_score(vY, y_pred)
        mae = mean_absolute_error(vY, y_pred)
        rmse = np.sqrt(mean_squared_error(vY, y_pred))
        
        print(f"  * {model_name} -> R2 Score: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        
        # Save SVR scatter plot
        if "SVR" in model_name:
            plt.figure(figsize=(10, 8))
            sns.regplot(x=vY, y=y_pred, color='teal', scatter_kws={'alpha':0.6, 'color':'mediumseagreen'}, line_kws={'color':'darkgreen'})
            plt.plot([vY.min(), vY.max()], [vY.min(), vY.max()], 'r--', lw=2, label='Perfect Prediction')
            plt.title(f"Full Dataset Prediction using Consensus Features ({model_name})", fontsize=14, fontweight='bold')
            plt.xlabel("Actual Scent Intensity (Imax)", fontsize=12)
            plt.ylabel("Predicted Scent Intensity (Imax)", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "consensus_predictions.png"), dpi=300)
            plt.close()
            
    print("\nConsensus Stability Experiment finished successfully!")

if __name__ == "__main__":
    main()
