# Consensus Stability Classification Experiment
# Goal: Run classification on the 9 consensus features for thresholds 35 and 40.
# Evaluate generalization performance via 5-Fold Cross-Validation and plot results.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def main():
    print("="*80)
    print("STARTING CONSENSUS STABILITY CLASSIFICATION (THRESHOLDS 35 & 40)")
    print("="*80)
    
    output_dir = "2_production_ready"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    vY_raw = data_df['Imax'].values
    
    # 9 Consensus features found from regression stability selection
    consensus_features = ['ATSC8e', 'Hy', 'MATS1e', 'SpDiam_Dt', 'H2m', 'SpMaxA_B_s_', 'Eig05_EA_ri_', 'SdsCH', 'MATS1s']
    
    print(f"Using the 9 Unified Scent Descriptors: {consensus_features}")
    
    # Extract and scale features
    X_raw = data_df[consensus_features].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Classifiers to evaluate
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Classifier (SVC)": SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (Depth=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = [35, 40]
    
    results = []
    
    # We will create a beautiful 2x2 plot
    # Top Row: Bar charts of Macro F1 performance
    # Bottom Row: Confusion Matrices for the best models at each threshold
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Store predictions of the best models for drawing confusion matrices later
    best_models_info = {}
    
    for t_idx, threshold in enumerate(thresholds):
        print(f"\n--- Evaluating Classification for Threshold {threshold} ---")
        # Define binary target class
        y_class = (vY_raw >= threshold).astype(int)
        
        # Calculate class distribution
        pos_pct = np.mean(y_class) * 100
        neg_pct = 100 - pos_pct
        print(f"  Class Balance: {pos_pct:.1f}% Strong Scent (>= {threshold}) | {neg_pct:.1f}% Weak Scent (< {threshold})")
        
        t_results = {}
        best_f1 = -1.0
        best_y_pred = None
        best_model_name = ""
        best_metrics = {}
        
        for name, clf in models.items():
            # Honest 5-Fold Cross-Validation prediction
            y_pred = cross_val_predict(clf, X, y_class, cv=cv)
            
            # Metrics calculation
            f1 = f1_score(y_class, y_pred, average='macro')
            accuracy = accuracy_score(y_class, y_pred)
            precision = precision_score(y_class, y_pred, zero_division=0)
            recall = recall_score(y_class, y_pred, zero_division=0)
            
            # Full data training fit (Resubstitution Score)
            clf.fit(X, y_class)
            y_full_pred = clf.predict(X)
            f1_full = f1_score(y_class, y_full_pred, average='macro')
            
            print(f"  * {name}:")
            print(f"    -> Out-of-Fold CV -> Macro F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            print(f"    -> Full Data Fit  -> Macro F1: {f1_full:.4f}")
            
            results.append({
                "Threshold": threshold,
                "Model": name,
                "CV_F1": f1,
                "CV_Accuracy": accuracy,
                "CV_Precision": precision,
                "CV_Recall": recall,
                "Full_F1": f1_full
            })
            
            t_results[name] = f1
            
            # Track best model by Macro F1
            if f1 > best_f1:
                best_f1 = f1
                best_y_pred = y_pred
                best_model_name = name
                best_metrics = {
                    "Model": best_model_name,
                    "F1": f1,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "y_class": y_class,
                    "y_pred": y_pred
                }
                
        best_models_info[threshold] = best_metrics
        
        # Draw bar chart for current threshold (Top Row)
        ax_bar = axes[0, t_idx]
        df_t = pd.DataFrame(list(t_results.items()), columns=["Model", "Macro F1-Score"])
        sns.barplot(
            x="Macro F1-Score", 
            y="Model", 
            data=df_t, 
            ax=ax_bar, 
            palette="viridis"
        )
        ax_bar.set_title(f"Macro F1 Comparison (Threshold = {threshold})", fontsize=13, fontweight='bold')
        ax_bar.set_xlabel("Honest 5-Fold CV Macro F1-Score", fontsize=11)
        ax_bar.set_ylabel("" if t_idx > 0 else "Classification Model", fontsize=11)
        ax_bar.set_xlim(0, 1.0)
        ax_bar.grid(True, alpha=0.3, axis='x')
        
        # Add labels to the bars
        for container in ax_bar.containers:
            ax_bar.bar_label(container, fmt="%.3f", padding=5, fontweight='bold', fontsize=10)
            
    # Now draw Confusion Matrices for the best models (Bottom Row)
    for t_idx, threshold in enumerate(thresholds):
        info = best_models_info[threshold]
        y_class = info["y_class"]
        y_pred = info["y_pred"]
        
        cm = confusion_matrix(y_class, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Re-arrange for display:
        # We will make a visually stunning annotated heatmap
        cm_display = np.array([[tn, fp], [fn, tp]])
        
        ax_cm = axes[1, t_idx]
        sns.heatmap(
            cm_display, 
            annot=False, 
            cmap="Purples", 
            cbar=False, 
            linewidths=3, 
            linecolor="white",
            ax=ax_cm
        )
        
        # Custom annotations for cell texts to show TN/FP/FN/TP clearly
        labels = [
            [f"True Weak (TN)\n{tn}\n(Correct)", f"False Strong (FP)\n{fp}\n(Type I Error)"],
            [f"False Weak (FN)\n{fn}\n(Type II Error)", f"True Strong (TP)\n{tp}\n(Correct)"]
        ]
        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j + 0.5, i + 0.5, labels[i][j], 
                    ha="center", va="center", 
                    fontsize=12, fontweight="bold",
                    color="white" if cm_display[i][j] > (len(y_class) / 4) else "black"
                )
                
        ax_cm.set_title(f"Best Model Confusion Matrix (T = {threshold})\n{info['Model']}", fontsize=13, fontweight='bold')
        ax_cm.set_xticks([0.5, 1.5])
        ax_cm.set_xticklabels(["Predicted Weak (<{})".format(threshold), "Predicted Strong (>={})".format(threshold)], fontsize=10)
        ax_cm.set_yticks([0.5, 1.5])
        ax_cm.set_yticklabels(["Actual Weak (<{})".format(threshold), "Actual Strong (>={})".format(threshold)], fontsize=10, rotation=90, va="center")
        
        # Draw a metrics text box under the confusion matrix
        metrics_text = (
            "🔥 HONEST GENERALIZATION METRICS:\n"
            "• Macro F1-Score: {:.4f}\n"
            "• Accuracy (Точность): {:.2%}\n"
            "• Precision (Точность предсказания): {:.2%}\n"
            "• Recall (Полнота выявления): {:.2%}"
        ).format(info["F1"], info["Accuracy"], info["Precision"], info["Recall"])
        
        ax_cm.text(
            0.5, -0.22, metrics_text, 
            transform=ax_cm.transAxes, 
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#d1d5db", alpha=0.9)
        )
            
    plt.suptitle("Generalization Classification Performance using the 9 Consensus Descriptors", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35)
    plt.savefig(os.path.join(output_dir, "classification_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n-> Classification performance plots successfully saved to classification_performance.png")
    
    # Print summary table in console
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("CLASSIFICATION PERFORMANCE SUMMARY TABLE")
    print("="*80)
    print(df_res.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
