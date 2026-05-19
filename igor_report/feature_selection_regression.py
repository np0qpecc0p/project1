import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def run_feature_selection_regression():
    print("="*60)
    print("HYBRID FEATURE SELECTION: CONTINUOUS ODOR INTENSITY REGRESSION")
    print("="*60)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw Dragon database...")
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    
    vY = data_df['Imax'].values
    
    print(f"-> Total raw descriptors: {len(feature_names)}")
    print(f"-> Target mean intensity: {np.mean(vY):.2f} (std={np.std(vY):.2f})")
    
    mX_raw = data_df[feature_names].fillna(0).values
    scaler = StandardScaler()
    mX_all = scaler.fit_transform(mX_raw)
    
    # 2. Random Forest Regressor Screening
    print("\n[Step 2] Screening all 2499 descriptors via Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(mX_all, vY)
    importances = rf.feature_importances_
    
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    # Plot top 20
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances[:20], y=sorted_features[:20], palette="viridis")
    plt.title("Top 20 RF Regressor Feature Importances")
    plt.xlabel("RF Importance Score")
    plt.ylabel("Descriptor Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_feature_importances.png"), dpi=300)
    plt.close()
    
    # 3. Greedy Forward Feature Selection (Ridge Regression Validator)
    print("\n[Step 3] Running Greedy Cross-Model Search with Ridge Regression...")
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
            mX_test = mX_all[:, test_indices]
            
            y_preds = np.zeros(len(vY))
            for train_idx, test_idx in cv.split(mX_test):
                validator.fit(mX_test[train_idx], vY[train_idx])
                y_preds[test_idx] = validator.predict(mX_test[test_idx])
                
            score = r2_score(vY, y_preds)
            if score > best_r2_step:
                best_r2_step = score
                best_feat_to_add = candidate
                
        if best_feat_to_add:
            selected_features.append(best_feat_to_add)
            current_best_r2 = best_r2_step
            print(f"  Slot {i+1} -> Added '{best_feat_to_add}' | Validator R2: {current_best_r2:.4f}")
        else:
            print(f"  Slot {i+1} -> No improvement. Stopping early.")
            break
            
    print("\n" + "="*50)
    print("WINNING FEATURES (Regression):")
    print(selected_features)
    print("="*50)

if __name__ == "__main__":
    run_feature_selection_regression()
