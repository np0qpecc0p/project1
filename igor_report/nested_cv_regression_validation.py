# Leakage-Free Nested Cross-Validation Feature Selection and Validation Script
# Goal: Run the 2-phase hybrid feature selection only on 4 folds of data, 
# and validate the selected features on the 5th fold using a model that did not participate in the search.

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

def run_nested_cv_experiment():
    print("="*80)
    print("STARTING NESTED 5-FOLD CROSS-VALIDATED FEATURE SELECTION (UNBIASED GENERALIZATION)")
    print("="*80)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw Dragon database...")
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    vY = data_df['Imax'].values
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    
    mX_raw = data_df[feature_names].fillna(0).values
    
    print(f"-> Total compounds: {mX_raw.shape[0]}")
    print(f"-> Total raw descriptors: {len(feature_names)}")
    
    # 2. Setup Outer 5-Fold Cross-Validation
    # This splits the data into 5 outer folds (4 folds for feature selection & training, 1 fold for testing)
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Non-participating models to evaluate on the 5th validation fold
    non_participating_models = {
        "Support Vector Regressor (SVR)": SVR(C=10.0, epsilon=0.1),
        "K-Nearest Neighbors Regressor (KNN)": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor (Depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    # Dictionaries to store results per outer fold
    fold_selected_features = {}
    fold_metrics = {model_name: [] for model_name in non_participating_models}
    
    # Array to store out-of-fold predictions for the best non-participating model (e.g. SVR)
    y_preds_oof = {model_name: np.zeros(len(vY)) for model_name in non_participating_models}
    
    # Loop over the 5 outer folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(mX_raw)):
        print("\n" + "-"*60)
        print(f"OUTER FOLD {fold_idx + 1} / 5")
        print(f"-> Training on 4 folds (N={len(train_idx)}) | Validating on 5th fold (N={len(test_idx)})")
        print("-"*60)
        
        # Split raw features and targets
        X_train_raw, X_test_raw = mX_raw[train_idx], mX_raw[test_idx]
        y_train, y_test = vY[train_idx], vY[test_idx]
        
        # Prevent data leakage: Fit scaler ONLY on the 4 outer train folds
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # ----------------------------------------------------
        # Phase 1: RF Screener on the 4 outer train folds
        # ----------------------------------------------------
        print(f"  [Phase 1] Screening descriptors with Random Forest on training folds...")
        rf_screener = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_screener.fit(X_train, y_train)
        importances = rf_screener.feature_importances_
        
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        candidate_pool = sorted_features[:50]
        print(f"  -> RF screening finished. Selected top 50 candidate descriptors.")
        
        # ----------------------------------------------------
        # Phase 2: Greedy Forward Selection on the 4 outer train folds
        # ----------------------------------------------------
        print(f"  [Phase 2] Running Greedy Forward Search (Ridge Validator) on training folds...")
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
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
                mX_inner_test = X_train[:, test_indices]
                
                y_preds_inner = np.zeros(len(y_train))
                for train_inner_idx, test_inner_idx in cv_inner.split(mX_inner_test):
                    validator.fit(mX_inner_test[train_inner_idx], y_train[train_inner_idx])
                    y_preds_inner[test_inner_idx] = validator.predict(mX_inner_test[test_inner_idx])
                    
                score = r2_score(y_train, y_preds_inner)
                if score > best_r2_step:
                    best_r2_step = score
                    best_feat_to_add = candidate
                    
            if best_feat_to_add:
                selected_features.append(best_feat_to_add)
                current_best_r2 = best_r2_step
                print(f"    Slot {i+1} -> Added '{best_feat_to_add}' | Inner Train CV R2: {current_best_r2:.4f}")
            else:
                break
                
        fold_selected_features[fold_idx] = selected_features
        print(f"  => Winning Features for Fold {fold_idx + 1}: {selected_features}")
        
        # ----------------------------------------------------
        # Phase 3: Evaluate on the 5th Outer Validation Fold
        # ----------------------------------------------------
        print(f"  [Phase 3] Evaluating selected features on the unseen 5th Fold...")
        # Extract indices of selected features
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_train_sub = X_train[:, selected_indices]
        X_test_sub = X_test[:, selected_indices]
        
        for model_name, model in non_participating_models.items():
            # Train model ONLY on training folds using selected features
            model.fit(X_train_sub, y_train)
            # Predict on the unseen 5th validation fold
            y_pred_val = model.predict(X_test_sub)
            
            # Save predictions for out-of-fold visual metrics
            y_preds_oof[model_name][test_idx] = y_pred_val
            
            # Compute metrics on 5th fold
            r2 = r2_score(y_test, y_pred_val)
            mae = mean_absolute_error(y_test, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_val))
            
            fold_metrics[model_name].append({
                "Fold": fold_idx + 1,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            })
            print(f"    * {model_name} -> Validation R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    # 3. Print Summary of Selected Features across Folds
    print("\n" + "="*80)
    print("SUMMARY OF SELECTED FEATURES PER OUT-OF-FOLD SPLIT")
    print("="*80)
    for fold_idx, feats in fold_selected_features.items():
        print(f"Fold {fold_idx + 1} Features: {feats}")
        
    # Check consistency: which descriptors are chosen in multiple folds?
    all_chosen_features = []
    for feats in fold_selected_features.values():
        all_chosen_features.extend(feats)
    feature_counts = pd.Series(all_chosen_features).value_counts()
    print("\nDescriptor Stability / Selection Frequency across folds:")
    for feat, count in feature_counts.items():
        print(f"  - '{feat}': selected in {count} out of 5 folds")

    # 4. Print Unbiased Performance Metic Summary
    print("\n" + "="*80)
    print("FINAL UNBIASED PERFORMANCE (OUTER FOLD GENERALIZATION)")
    print("="*80)
    
    summary_data = []
    for model_name in non_participating_models:
        df_metrics = pd.DataFrame(fold_metrics[model_name])
        mean_r2, std_r2 = df_metrics["R2"].mean(), df_metrics["R2"].std()
        mean_mae, std_mae = df_metrics["MAE"].mean(), df_metrics["MAE"].std()
        mean_rmse, std_rmse = df_metrics["RMSE"].mean(), df_metrics["RMSE"].std()
        
        print(f"\n{model_name}:")
        print(f"  Mean R2 Score:  {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  Mean MAE:       {mean_mae:.2f} ± {std_mae:.2f}")
        print(f"  Mean RMSE:      {mean_rmse:.2f} ± {std_rmse:.2f}")
        
        summary_data.append({
            "Model": model_name,
            "R2 Mean": mean_r2,
            "R2 Std": std_r2,
            "MAE Mean": mean_mae,
            "MAE Std": std_mae
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    # 5. Generate and Save Visualizations
    
    # Plot 1: R2 score distribution across folds
    plt.figure(figsize=(10, 6))
    fold_plot_data = []
    for model_name in non_participating_models:
        for metric in fold_metrics[model_name]:
            fold_plot_data.append({
                "Model": model_name,
                "Fold": f"Fold {metric['Fold']}",
                "R2": metric['R2']
            })
    df_plot = pd.DataFrame(fold_plot_data)
    sns.barplot(data=df_plot, x="Fold", y="R2", hue="Model", palette="muted")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Generalization R2 Score on 5th Unseen Fold (Leakage-Free)", fontsize=14, fontweight='bold')
    plt.ylabel("R2 Score")
    plt.ylim(-0.2, 1.0)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_cv_metrics.png"), dpi=300)
    plt.close()
    
    # Plot 2: Actual vs Out-of-fold Predicted for SVR (the best non-participating model)
    best_model_name = "Support Vector Regressor (SVR)"
    plt.figure(figsize=(10, 8))
    sns.regplot(x=vY, y=y_preds_oof[best_model_name], color='indigo', scatter_kws={'alpha':0.6, 'color':'darkorchid'})
    plt.plot([vY.min(), vY.max()], [vY.min(), vY.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f"Unbiased Actual vs Out-of-Fold Predictions ({best_model_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Actual Scent Intensity (Imax)", fontsize=12)
    plt.ylabel("Out-of-Fold Predicted Scent Intensity (Imax)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_cv_predictions.png"), dpi=300)
    plt.close()
    
    print(f"\nAll nested validation plots successfully saved to the {output_dir}/ folder!")
    print("Nested CV validation process successfully completed!")

if __name__ == "__main__":
    run_nested_cv_experiment()
