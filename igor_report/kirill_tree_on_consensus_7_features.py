# Kirill's Decision Tree Classifier - Evaluated on 7 Consensus Features
# Goal: Re-run Kirill's LOOCV grid search on the 7 consensus regression features 
# for both Scent Intensity thresholds (35.0 and 40.0), and compare metrics.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

def run_kirill_grid_search(mX, vY, lfeatures_names, threshold_val):
    print("="*60)
    print(f"RUNNING KIRILL GRID SEARCH FOR THRESHOLD = {threshold_val}")
    print("="*60)
    
    lCriterion = ['gini', 'entropy', 'log_loss']
    lMaxLeaf = list(range(5, 11))
    numComb = len(lCriterion) * len(lMaxLeaf)
    
    dict_results = {
        'criterion': [],
        'max_leaf_nodes': [],
        'f1_score': [0.0] * numComb,
        'accuracy': [0.0] * numComb,
        'precision': [0.0] * numComb,
        'recall': [0.0] * numComb
    }
    
    for criterion in lCriterion:
        for max_leaf_nodes in lMaxLeaf:
            dict_results['criterion'].append(criterion)
            dict_results['max_leaf_nodes'].append(max_leaf_nodes)
            
    modelScores_df = pd.DataFrame(data=dict_results)
    
    # Run Leave-One-Out CV
    loocv = KFold(n_splits=mX.shape[0])
    
    for i in range(numComb):
        paramCriterion = modelScores_df.loc[i, 'criterion']
        maxLeaf = modelScores_df.loc[i, 'max_leaf_nodes']
        
        clf = DecisionTreeClassifier(criterion=paramCriterion, max_leaf_nodes=maxLeaf, random_state=42)
        vYPred = cross_val_predict(clf, mX, vY, cv=loocv)
        
        modelScores_df.loc[i, 'f1_score'] = f1_score(vY, vYPred)
        modelScores_df.loc[i, 'accuracy'] = accuracy_score(vY, vYPred)
        modelScores_df.loc[i, 'precision'] = precision_score(vY, vYPred)
        modelScores_df.loc[i, 'recall'] = recall_score(vY, vYPred)
        
    # Find Best Model
    indArgmax = np.argmax(modelScores_df['f1_score'])
    best_row = modelScores_df.loc[indArgmax]
    
    print("\n--- Top Models sorted by F1 ---")
    print(modelScores_df.sort_values(by='f1_score', ascending=False).head(3))
    
    optimalCriterion = best_row['criterion']
    optimalMaxLeaf = int(best_row['max_leaf_nodes'])
    print(f'\nBest Model: criterion = {optimalCriterion}, max_leaf_nodes = {optimalMaxLeaf}')
    print(f'F1: {best_row["f1_score"]:.4f} | Accuracy: {best_row["accuracy"]:.4f} | Precision: {best_row["precision"]:.4f} | Recall: {best_row["recall"]:.4f}')
    
    # Fit Final model and plot importance
    best_clf = DecisionTreeClassifier(criterion=optimalCriterion, max_leaf_nodes=optimalMaxLeaf, random_state=42)
    best_clf.fit(mX, vY)
    
    vFeatureImportance = best_clf.feature_importances_
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x=lfeatures_names, height=vFeatureImportance, color='mediumseagreen')
    plt.title(f'Feature Importance on 7 Consensus Features (Threshold = {threshold_val})', fontsize=13, fontweight='bold')
    plt.xlabel('Feature Name')
    plt.ylabel('Importance')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'igor_report/consensus_tree_importance_t{int(threshold_val)}.png', dpi=300)
    plt.close()
    
    # Save Heatmap Plot
    heat_df = modelScores_df.pivot(index='max_leaf_nodes', columns='criterion', values='f1_score')
    plt.figure(figsize=(8, 6))
    sns.heatmap(data=heat_df, annot=True, fmt='.3f', cmap='Greens', linewidths=1)
    plt.title(f'Decision Tree F1 Heatmap (Threshold = {threshold_val})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'igor_report/consensus_tree_f1_heatmap_t{int(threshold_val)}.png', dpi=300)
    plt.close()
    
    return best_row

def main():
    output_dir = "igor_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    data_df = pd.read_csv('data/waka_dragon_merged.csv')
    
    # Use our 7 Consensus Regression Descriptors
    consensus_features = ['Di_x', 'ATSC8e', 'SpMaxA_B_s_', 'MATS1e', 'SpDiam_Dt', 'Mor32v', 'SpMaxA_AEA_dm_']
    mX = data_df[consensus_features].fillna(0).values
    
    # Threshold 35.0 classification
    vY_35 = np.where(data_df['Imax'].values >= 35.0, 1, 0)
    print("\n" + "#"*70)
    print("EVALUATING ON THRESHOLD 35.0")
    print("#"*70)
    best_35 = run_kirill_grid_search(mX, vY_35, consensus_features, 35.0)
    
    # Threshold 40.0 classification
    vY_40 = np.where(data_df['Imax'].values >= 40.0, 1, 0)
    print("\n" + "#"*70)
    print("EVALUATING ON THRESHOLD 40.0")
    print("#"*70)
    best_40 = run_kirill_grid_search(mX, vY_40, consensus_features, 40.0)
    
    print("\n" + "="*70)
    print("SUMMARY COMPARISON:")
    print("="*70)
    print(f"Threshold 35.0 Classification (LOOCV) -> Best F1: {best_35['f1_score']:.4f} | Accuracy: {best_35['accuracy']:.4f} | Precision: {best_35['precision']:.4f} | Recall: {best_35['recall']:.4f}")
    print(f"Threshold 40.0 Classification (LOOCV) -> Best F1: {best_40['f1_score']:.4f} | Accuracy: {best_40['accuracy']:.4f} | Precision: {best_40['precision']:.4f} | Recall: {best_40['recall']:.4f}")

if __name__ == "__main__":
    main()
