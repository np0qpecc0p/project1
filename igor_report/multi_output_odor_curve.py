# Multi-Output Scent Curve Parameters Prediction Pipeline
# Target Variables: ['Imax', 'Ci', 'Di_x'] (Odor Intensity Curve Parameters)
# Feature Pool: 2490+ Dragon molecular descriptors (excluding targets Ci and Di_x)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def select_target_features(X_raw, y, feature_names, target_label):
    print(f"  -> Selecting features for target: '{target_label}'...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Phase 1: RF screening
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx][:30] # top 30
    
    # Phase 2: Greedy Forward Selection (Ridge Validator)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validator = Ridge(alpha=10.0, random_state=42)
    
    selected_features = []
    current_best_r2 = -999.0
    
    for i in range(7): # select up to 7 features per target
        best_feat_to_add = None
        best_r2_step = current_best_r2
        
        for candidate in sorted_features:
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

def run_stability_selection_target(mX_raw, y, feature_names, target_label):
    print(f"\n--- 10-Fold Stability Selection for Target '{target_label}' ---")
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
    all_chosen = []
    
    for fold_idx, (train_idx, _) in enumerate(cv_outer.split(mX_raw)):
        X_train, y_train = mX_raw[train_idx], y[train_idx]
        selected = select_target_features(X_train, y_train, feature_names, target_label)
        all_chosen.extend(selected)
        
    counts = pd.Series(all_chosen).value_counts()
    # Consensus: selected in > 3 out of 10 folds
    consensus = [feat for feat, count in counts.items() if count > 3]
    print(f"Consensus features for '{target_label}' (selected > 3 folds): {consensus}")
    return consensus

def main():
    print("="*80)
    print("MULTI-OUTPUT SCENT CURVE PARAMETERS PREDICTION PIPELINE")
    print("="*80)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    # Curve Target Parameters
    targets = ['Imax', 'Ci', 'Di_x']
    y_data = data_df[targets].values
    
    # Feature names - EXCLUDING Wakayama targets to avoid leakage!
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class'] + targets
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    mX_raw = data_df[feature_names].fillna(0).values
    
    print(f"Number of molecules: {len(data_df)}")
    print(f"Target variables: {targets}")
    print(f"Number of molecular structure features in pool: {mX_raw.shape[1]}")
    
    # 1. Run consensus feature selection for each target separately
    consensus_features_by_target = {}
    all_consensus_features = set()
    
    for idx, target in enumerate(targets):
        consensus = run_stability_selection_target(mX_raw, y_data[:, idx], feature_names, target)
        consensus_features_by_target[target] = consensus
        all_consensus_features.update(consensus)
        
    scent_curve_descriptors = sorted(list(all_consensus_features))
    print("\n" + "="*80)
    print("UNIFIED OLFACTORY SCENT CURVE DESCRIPTORS SET:")
    print(f"Total Unique Descriptors: {len(scent_curve_descriptors)}")
    print(f"Descriptors: {scent_curve_descriptors}")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(mX_raw)
    consensus_indices = [feature_names.index(f) for f in scent_curve_descriptors]
    X_consensus = X_scaled[:, consensus_indices]
    
    # 2. Honest 5-Fold Cross-Validation for the entire curve
    print("\nEvaluating Multi-Output Prediction accuracy using 5-Fold Cross-Validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for out-of-fold predictions
    oof_predictions = np.zeros_like(y_data)
    
    # Train optimized SVR for each target separately (Independent Multi-Output Regression)
    models = {
        'Imax': SVR(C=10.0, epsilon=0.1),
        'Ci': SVR(C=10.0, epsilon=0.1),
        'Di_x': SVR(C=10.0, epsilon=0.1)
    }
    
    for train_idx, test_idx in cv.split(X_consensus):
        for idx, target in enumerate(targets):
            model = models[target]
            model.fit(X_consensus[train_idx], y_data[train_idx, idx])
            oof_predictions[test_idx, idx] = model.predict(X_consensus[test_idx])
            
    # Calculate CV Metrics
    print("\n--- Unbiased Multi-Output Cross-Validation Performance ---")
    for idx, target in enumerate(targets):
        r2 = r2_score(y_data[:, idx], oof_predictions[:, idx])
        mae = mean_absolute_error(y_data[:, idx], oof_predictions[:, idx])
        print(f"Target '{target}' -> CV R2 Score: {r2:.4f} | CV MAE: {mae:.4f}")
        
    # 3. Train final production models on 100% data and generate regression plots
    print("\nTraining final Scent Curve Prediction models on 100% data...")
    final_predictions = np.zeros_like(y_data)
    for idx, target in enumerate(targets):
        model = models[target]
        model.fit(X_consensus, y_data[:, idx])
        final_predictions[:, idx] = model.predict(X_consensus)
        
    # Plot predictions
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    colors = ['#10b981', '#3b82f6', '#f59e0b']
    titles = [
        "Maximum Intensity (Imax)",
        "Inflection Parameter (Ci)",
        "Slope Parameter (Di)"
    ]
    
    for idx, ax in enumerate(axes):
        sns.regplot(
            x=y_data[:, idx], 
            y=final_predictions[:, idx], 
            ax=ax, 
            color=colors[idx],
            scatter_kws={'alpha':0.6},
            line_kws={'color':'red', 'lw':2}
        )
        ax.plot([y_data[:, idx].min(), y_data[:, idx].max()], [y_data[:, idx].min(), y_data[:, idx].max()], 'k--', lw=1.5)
        ax.set_title(titles[idx], fontsize=13, fontweight='bold')
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Scent Curve Parameters Prediction on 100% Data (Consensus Model)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_output_predictions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("-> Scent curve prediction scatter plots successfully saved to multi_output_predictions.png")

if __name__ == "__main__":
    main()
