import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def run_raw_experiment():
    print("="*60)
    print("STARTING RAW 2800+ FEATURES HYBRID SELECTION (USER'S IDEA)")
    print("="*60)
    
    output_dir = "igor"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Raw Data
    print("\n[Step 1] Loading raw Dragon database (2499+ features)...")
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['Unnamed: 0.1', 'Unnamed: 0', 'CAS', 'Name', 'CID', 'SMILES', 'intensity_class', 'Imax']
    feature_names = [col for col in data_df.columns if col not in exclude_cols]
    
    print(f"-> Total raw features extracted: {len(feature_names)}")
    print(f"-> Total compounds: {data_df.shape[0]}")
    
    # Define Target and Feature Matrix
    vImax = data_df['Imax'].values
    vY = np.where(vImax >= 35.0, 1, 0)
    
    # Handle NaNs in raw features by filling with 0 (standard approach for missing dragon descriptors)
    mX_raw = data_df[feature_names].fillna(0).values
    
    # Scale features (important for SVM and Logistic Regression later)
    scaler = StandardScaler()
    mX_all = scaler.fit_transform(mX_raw)
    
    # 2. Random Forest Feature Selection on all 2400+ features
    print("\n[Step 2] Feeding all features to Random Forest to extract Top Candidates non-linearly...")
    rf_selector = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_selector.fit(mX_all, vY)
    importances = rf_selector.feature_importances_
    
    # Sort and rank
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]
    
    print("\nTop 20 Raw Features (Ranked by Random Forest):")
    for rank, (feat, imp) in enumerate(zip(sorted_features[:20], sorted_importances[:20]), 1):
        print(f"  {rank:2d}. {feat:<20} (Importance: {imp*100:6.3f}%)")
        
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances[:20], y=sorted_features[:20])
    plt.title("Top 20 Raw Feature Importances (Random Forest on 2499 Features)")
    plt.xlabel("Importance Score")
    plt.ylabel("Raw Dragon Descriptor")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_feature_importances.png"), dpi=300)
    plt.close()
    
    # 3. User's Cross-Model Validation Algorithm
    print("\n[Step 3] Executing User's Cross-Model Validation...")
    print("Testing RF-selected features on an independent Logistic Regression model to filter out tree-specific overfitting.")
    
    # Let's take the Top 50 features from RF as our candidate pool
    candidate_pool = sorted_features[:50]
    
    cv_fast = KFold(n_splits=5, shuffle=True, random_state=42)
    validator = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
    
    # We will greedily build a Top 10 list that maximizes the Validator's F1 score
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
            
            y_preds_test = np.zeros(len(vY))
            for train_idx, test_idx in cv_fast.split(mX_test):
                validator.fit(mX_test[train_idx], vY[train_idx])
                y_preds_test[test_idx] = validator.predict(mX_test[test_idx])
                
            score = f1_score(vY, y_preds_test)
            
            if score > best_f1_step:
                best_f1_step = score
                best_feat_to_add = candidate
                
        if best_feat_to_add:
            selected_features.append(best_feat_to_add)
            current_best_f1 = best_f1_step
            print(f"  [Slot {i+1}] Added '{best_feat_to_add}' -> Validator CV F1 improved to {current_best_f1:.4f}")
        else:
            print(f"  [Slot {i+1}] No further feature improved the score. Stopping early.")
            break
            
    print("\n" + "="*60)
    print(f"FINAL SURVIVORS (Top {len(selected_features)} Raw Features Validated by Second Model):")
    print("="*60)
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
        
    # 4. Final LOOCV Evaluation of the Selected Features across all models
    print("\n[Step 4] Running final Leave-One-Out CV on the winning raw features...")
    selected_indices = [feature_names.index(f) for f in selected_features]
    mX_final = mX_all[:, selected_indices]
    
    cv_loocv = KFold(n_splits=mX_final.shape[0])
    
    models = {
        "Decision Tree (tuned)": DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=7, random_state=42),
        "Logistic Regression (L2)": LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, random_state=42),
        "Support Vector Machine (RBF)": SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        "K-Nearest Neighbors (K=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    comparison_results = []
    
    for model_name, model in models.items():
        y_preds = np.zeros(len(vY))
        for train_idx, test_idx in cv_loocv.split(mX_final):
            model.fit(mX_final[train_idx], vY[train_idx])
            y_preds[test_idx] = model.predict(mX_final[test_idx])
            
        acc = accuracy_score(vY, y_preds)
        f1 = f1_score(vY, y_preds)
        prec = precision_score(vY, y_preds)
        rec = recall_score(vY, y_preds)
        tn, fp, fn, tp = confusion_matrix(vY, y_preds).ravel()
        
        comparison_results.append({
            "Model": model_name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Confusion Matrix": f"TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}"
        })
        
    df_compare = pd.DataFrame(comparison_results)
    print("\nModel Comparison Table (LOOCV on Raw 2800 -> Validated Top Features):")
    print(df_compare.to_string(index=False))
    
    # Plot Model Comparison
    plt.figure(figsize=(10, 5))
    df_melted = pd.melt(df_compare, id_vars=["Model"], value_vars=["Accuracy", "F1-Score"],
                        var_name="Metric", value_name="Score")
    sns.barplot(x="Model", y="Score", hue="Metric", data=df_melted)
    plt.title(f"Model Comparison on Top {len(selected_features)} Raw Features (LOOCV)")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_model_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_raw_experiment()
