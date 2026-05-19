import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

def run_tree_on_10_features():
    print("="*60)
    print("RUNNING TUNED DECISION TREE ON 10 REGRESSION RAW FEATURES")
    print("="*60)
    
    output_dir = "igor"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading raw Dragon database...")
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    # Selected 10 Raw Regression Features
    selected_features = ['Di_x', 'Eig07_AEA_bo_', 'Hy', 'SpDiam_Dt', 'SpMaxA_B_s_', 'ATSC8e', 'Mor32v', 'G2v', 'G2p', 'MATS1s']
    print(f"-> Selected features: {selected_features}")
    
    # Prepare Features and Labels
    vImax = data_df['Imax'].values
    vY = np.where(vImax >= 35.0, 1, 0)
    
    mX = data_df[selected_features].fillna(0).values
    
    # 2. Replicate Friend's Model Setup
    print("\n[Step 2] Initializing Decision Tree Classifier with friend's parameters...")
    print("-> criterion = 'entropy', max_leaf_nodes = 7, random_state = 42")
    model = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=7, random_state=42)
    
    # 3. Leave-One-Out Cross-Validation (LOOCV)
    print("\n[Step 3] Running Leave-One-Out Cross-Validation (312 splits)...")
    cv_loocv = KFold(n_splits=mX.shape[0])
    
    y_preds = np.zeros(len(vY))
    for train_idx, test_idx in cv_loocv.split(mX):
        model.fit(mX[train_idx], vY[train_idx])
        y_preds[test_idx] = model.predict(mX[test_idx])
        
    # Calculate LOOCV metrics
    acc = accuracy_score(vY, y_preds)
    f1 = f1_score(vY, y_preds)
    prec = precision_score(vY, y_preds)
    rec = recall_score(vY, y_preds)
    tn, fp, fn, tp = confusion_matrix(vY, y_preds).ravel()
    
    print("\n" + "="*50)
    print("LEAVE-ONE-OUT CV RESULTS (10 REGRESSION FEATURES):")
    print("="*50)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*50)
    
    # 4. Train on Entire Dataset and Plot Tree
    print("\n[Step 4] Training on the full dataset to plot and save tree structure...")
    model.fit(mX, vY)
    
    plt.figure(figsize=(16, 10))
    plot_tree(
        model, 
        feature_names=selected_features, 
        class_names=["Weak (<35.0)", "Strong (>=35.0)"], 
        filled=True, 
        rounded=True, 
        fontsize=9
    )
    plt.title("Decision Tree Classifier (Trained on 10 Raw Regression Features)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    tree_plot_path = os.path.join(output_dir, "decision_tree_10_features.png")
    plt.savefig(tree_plot_path, dpi=300)
    plt.close()
    print(f"-> Decision tree visualization successfully saved to: {tree_plot_path}")
    
    # Save a small text report
    report_path = os.path.join(output_dir, "tree_10_features_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Decision Tree Evaluation on 10 Raw Regression Features\n\n")
        f.write(f"- **Features:** {selected_features}\n")
        f.write("- **Binarized Target Threshold:** >= 35.0\n")
        f.write("- **Model Hyperparameters:** `criterion='entropy'`, `max_leaf_nodes=7`, `random_state=42`\n")
        f.write("- **Validation Method:** Leave-One-Out CV (LOOCV)\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Accuracy:** {acc*100:.2f}%\n")
        f.write(f"- **F1-Score:** {f1*100:.2f}%\n")
        f.write(f"- **Precision:** {prec*100:.2f}%\n")
        f.write(f"- **Recall:** {rec*100:.2f}%\n")
        f.write(f"- **Confusion Matrix:** TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
        f.write("## Visualization\n\n")
        f.write(f"The decision tree structure has been saved as a premium PNG image at `igor/decision_tree_10_features.png`.\n")
        
    print(f"-> Text report saved to: {report_path}")
    print("="*60)
    print("EXPERIMENT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    run_tree_on_10_features()
