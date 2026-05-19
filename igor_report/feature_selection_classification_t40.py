import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run_feature_selection_t40():
    print("="*60)
    print("HYBRID FEATURE SELECTION: BINARY CLASSIFICATION AT THRESHOLD 40")
    print("="*60)
    
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw Dragon database...")
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    
    vImax = data_df['Imax'].values
    vY = np.where(vImax >= 40.0, 1, 0)
    
    print(f"-> Total raw descriptors: {len(feature_names)}")
    print(f"-> Class balance (Thresh 40): {np.sum(vY==0)} Weak / {np.sum(vY==1)} Strong")
    
    mX_raw = data_df[feature_names].fillna(0).values
    scaler = StandardScaler()
    mX_all = scaler.fit_transform(mX_raw)
    
    # 2. Random Forest Screening
    print("\n[Step 2] Screening all 2499 descriptors via Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(mX_all, vY)
    importances = rf.feature_importances_
    
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    # Plot top 20
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances[:20], y=sorted_features[:20], palette="flare")
    plt.title("Top 20 RF Feature Importances (Threshold = 40.0)")
    plt.xlabel("RF Importance Score")
    plt.ylabel("Descriptor Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_feature_importances_t40.png"), dpi=300)
    plt.close()
    
    # 3. Greedy Forward Feature Selection (Logistic Regression Validator)
    print("\n[Step 3] Running Greedy Cross-Model Search with Logistic Regression...")
    candidate_pool = sorted_features[:50]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validator = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
    
    selected_features = []
    current_best_f1 = 0.0
    
    for i in range(10):
        best_feat_to_add = None
        best_f1_step = current_best_f1
        
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
                
            score = f1_score(vY, y_preds)
            if score > best_f1_step:
                best_f1_step = score
                best_feat_to_add = candidate
                
        if best_feat_to_add:
            selected_features.append(best_feat_to_add)
            current_best_f1 = best_f1_step
            print(f"  Slot {i+1} -> Added '{best_feat_to_add}' | Validator F1: {current_best_f1:.4f}")
        else:
            print(f"  Slot {i+1} -> No improvement. Stopping early.")
            break
            
    print("\n" + "="*50)
    print("WINNING FEATURES (Threshold 40.0):")
    print(selected_features)
    print("="*50)

if __name__ == "__main__":
    run_feature_selection_t40()
