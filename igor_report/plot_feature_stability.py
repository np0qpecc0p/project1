# Feature Selection Stability Visualizer across Folds and Full Dataset
# Goal: Re-run selection on 5 folds + Full Data, collect the selected features, 
# and plot a beautiful presence-absence heatmap showing feature stability.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def select_features_subset(X_raw, y, feature_names):
    # Fit StandardScaler
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
    print("="*70)
    print("GENERATING FEATURE STABILITY VISUAL COMPARISON")
    print("="*70)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    vY = data_df['Imax'].values
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    mX_raw = data_df[feature_names].fillna(0).values
    
    # Dictionaries to store selected features per run
    runs = {}
    
    # 1. Run on Full Data
    print("\n[Step 1] Running feature selection on the Full Dataset...")
    runs["Full Data"] = select_features_subset(mX_raw, vY, feature_names)
    print(f"-> Full Data Selected Features: {runs['Full Data']}")
    
    # 2. Run on 5 folds
    print("\n[Step 2] Running feature selection on 5 outer folds (80% training splits)...")
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, _) in enumerate(cv_outer.split(mX_raw)):
        fold_name = f"Fold {fold_idx + 1}"
        X_train, y_train = mX_raw[train_idx], vY[train_idx]
        runs[fold_name] = select_features_subset(X_train, y_train, feature_names)
        print(f"-> {fold_name} Selected Features: {runs[fold_name]}")
        
    # 3. Create Presence-Absence Matrix
    # Get all unique features selected in at least one run
    all_selected_features = []
    for feats in runs.values():
        all_selected_features.extend(feats)
    unique_features = sorted(list(set(all_selected_features)))
    
    matrix_data = {run_name: [] for run_name in runs}
    for feat in unique_features:
        for run_name, feats in runs.items():
            if feat in feats:
                matrix_data[run_name].append(1)
            else:
                matrix_data[run_name].append(0)
                
    df_stability = pd.DataFrame(matrix_data, index=unique_features)
    
    # Sort features by frequency of selection (total sum across columns)
    df_stability["Selection Frequency"] = df_stability.sum(axis=1)
    df_stability = df_stability.sort_values(by="Selection Frequency", ascending=False)
    
    # Drop frequency column for plotting
    df_plot = df_stability.drop(columns=["Selection Frequency"])
    
    # 4. Plot Heatmap
    plt.figure(figsize=(12, 10))
    # We use a custom color palette: Purple/Indigo for selected, white for not selected
    sns.heatmap(
        df_plot, 
        cmap=sns.color_palette(["#f3f4f6", "#6366f1"]), 
        cbar=False, 
        linewidths=1.5, 
        linecolor="#e5e7eb",
        annot=True,
        fmt="d",
        annot_kws={"size": 12, "weight": "bold", "color": "white"}
    )
    
    # Style plot
    plt.title("Feature Selection Stability Analysis (Presence-Absence Matrix)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Evaluation Run / CV Split", fontsize=13, labelpad=10)
    plt.ylabel("Dragon Molecular Descriptor Name", fontsize=13, labelpad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # Save Image
    output_path = os.path.join(output_dir, "feature_stability_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\nStability plot successfully generated and saved to {output_path}!")
    
    # Save the DataFrame as markdown file
    matrix_path = os.path.join(output_dir, "stability_matrix.md")
    with open(matrix_path, "w", encoding="utf-8") as f:
        f.write("# Feature Selection Stability Matrix (All 5 Folds + Full Data)\n\n")
        f.write("Below is the presence-absence matrix representing descriptor selection consistency. ")
        f.write("A value of `1` indicates that the feature was selected during that specific run, and `0` indicates absence.\n\n")
        f.write(df_stability.to_markdown())
    print(f"Stability matrix saved as markdown to {matrix_path}!")
    
    # Print the DataFrame as markdown-compatible text for copy-paste
    print("\n" + "="*50)
    print("STABILITY MATRIX DATA:")
    print("="*50)
    print(df_stability.to_markdown())

if __name__ == "__main__":
    main()

