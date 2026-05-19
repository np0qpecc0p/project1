import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def run_3features_test():
    print("="*60)
    print("TESTING 3 RAW FEATURES UNDER STRATIFIED 5-FOLD CV")
    print("="*60)
    
    # 1. Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    vImax = data_df['Imax'].values
    vY = np.where(vImax >= 35.0, 1, 0)
    
    # Define our 3 Features
    # G2 (mass distribution), Ho_H2 (hydrogen topology), HyWi_B_e_ (Wiener index weighted by electronegativity)
    raw_cols = ['G2', 'Ho_H2', 'HyWi_B_e_']
    
    X_our = data_df[raw_cols].fillna(0).values
    scaler = StandardScaler()
    X_our_scaled = scaler.fit_transform(X_our)
    
    print(f"-> Testing features: {', '.join(raw_cols)}")
    
    # 2. Setup Stratified 5-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define Models
    models = {
        "Decision Tree (tuned)": DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=7, random_state=42),
        "Logistic Regression (L2)": LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 3. Evaluate
    scoring = ['accuracy', 'f1', 'precision', 'recall']
    
    print("\nResults with 3 Raw Features (Average over 5 Folds):")
    print("-" * 65)
    for model_name, model in models.items():
        scores = cross_validate(model, X_our_scaled, vY, cv=cv, scoring=scoring)
        print(f"Model: {model_name}")
        print(f"  Accuracy : {np.mean(scores['test_accuracy'])*100:.2f}%")
        print(f"  F1-Score : {np.mean(scores['test_f1']):.4f}")
        print(f"  Precision: {np.mean(scores['test_precision']):.4f}")
        print(f"  Recall   : {np.mean(scores['test_recall']):.4f}")
        print("-" * 65)

if __name__ == "__main__":
    run_3features_test()
