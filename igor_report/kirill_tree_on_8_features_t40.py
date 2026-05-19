# Kirill's Decision Tree Classifier - Replicated on 8 Raw Features (Threshold = 40.0)

# Import Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters

#===========================Fill This===========================#
# 1. Set the options for the `criterion` parameter (Use all options).
# 2. Set the options for the `max_leaf_nodes` parameter.
lCriterion   = ['gini', 'entropy', 'log_loss'] #<! List
lMaxLeaf     = list(range(5, 11)) #<! List

# create the features matrix, in this trial we use the 8 features from waka_dragon_merged.csv
data_df = pd.read_csv('data/waka_dragon_merged.csv')
features_8 = ['Di_x', 'Eig08_AEA_bo_', 'Mor28i', 'RDF020u', 'SM2_B_v_', 'VE3_B_p_', 'SM14_AEA_bo_', 'Eig06_EA_ri_']
mX = data_df[features_8].fillna(0).to_numpy(dtype=float)

lfeatures_names = features_8
print("Selected features:", lfeatures_names)

# create the Imax label vector
vImax = data_df['Imax'].values
vY = np.where(vImax >= 40.0, 1, 0) # we set the threshold to 40.0

print('The shape of the feature matrix is:', mX.shape)
print('The shape of the label vector is:', vY.shape)
vLabels, vCounts = np.unique(vY, return_counts=True)
print(f'labels: {vLabels}, counts: {vCounts}')

# plot the histofram of labels ()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(vLabels, vCounts, width=0.5, color='coral')
ax.set_xlabel('Class')
ax.set_xticks(vLabels)
ax.set_ylabel('Count')
ax.set_title('Histogram of Intensity classes, threshold = 40.0')
plt.savefig('igor_report/histogram_labels_t40.png')
plt.close()

# i want to check if there are any missing values in the feature matrix
print(f'Number of missing values in the feature matrix: {np.isnan(mX).sum()}')

# i want to plot distribution of each feature in mX matrix for each class (nrows=2, ncols=4 for 8 features)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
for feat_idx, axis in enumerate(ax.flat):
    axis.hist(mX[vY == 0, feat_idx], bins=20, alpha=0.5, label='Class 0')
    axis.hist(mX[vY == 1, feat_idx], bins=20, alpha=0.5, label='Class 1')
    axis.set_title(lfeatures_names[feat_idx])
    axis.legend()
plt.tight_layout()
plt.savefig('igor_report/feature_distributions_t40.png')
plt.close()

numComb = len(lCriterion) * len(lMaxLeaf)

dict_results = {
    'criterion': [],
    'max_leaf_nodes': [],
    'f1_score': [0.0] * numComb,
    'accuracy': [0.0] * numComb,
    'precision': [0.0] * numComb,
    'recall': [0.0] * numComb
}

for i, criterion in enumerate(lCriterion):
    for j, max_leaf_nodes in enumerate(lMaxLeaf):
        dict_results['criterion'].append(criterion)
        dict_results['max_leaf_nodes'].append(max_leaf_nodes)

modelScores_df = pd.DataFrame(data=dict_results)
print("\nInitial modelScores DataFrame:")
print(modelScores_df)

for i in range(numComb):
    paramCriterion = modelScores_df.loc[i, 'criterion']
    maxLeaf = modelScores_df.loc[i, 'max_leaf_nodes']

    print(f'Training model {i+1} from {numComb} with criterion: {paramCriterion} and max_leaf_nodes: {maxLeaf}')

    oDecTreeClf = DecisionTreeClassifier(criterion=paramCriterion, max_leaf_nodes=maxLeaf)
    vYPred = cross_val_predict(oDecTreeClf, mX, vY, cv=KFold(n_splits=mX.shape[0]))

    f1Score = f1_score(vY, vYPred)
    accuracy = accuracy_score(vY, vYPred)
    precision = precision_score(vY, vYPred)
    recall = recall_score(vY, vYPred)

    modelScores_df.loc[i, 'f1_score'] = f1Score
    modelScores_df.loc[i, 'accuracy'] = accuracy
    modelScores_df.loc[i, 'precision'] = precision
    modelScores_df.loc[i, 'recall'] = recall

    print(f'F1 score for model {i+1} is: {f1Score}')
    print(f'Accuracy for model {i+1} is: {accuracy}')
    print(f'Precision for model {i+1} is: {precision}')
    print(f'Recall for model {i+1} is: {recall}')

print("\n--- Model Scores DataFrame ---")
print(modelScores_df)

print("\n--- Top 10 Models sorted by F1 ---")
print(modelScores_df.sort_values(by='f1_score', ascending=False).head(10))

# i want to built  heat map of the modelScores_df
heat_df = modelScores_df.pivot(
    index='max_leaf_nodes', columns='criterion', values='f1_score'
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    data=heat_df, annot=True, fmt='.3f', linewidths=1, robust=True, ax=ax
)
ax.set_title('F1 of the Cross Validation')
plt.savefig('igor_report/f1_heatmap_t40.png')
plt.close()

# i want to built  heat map of the modelScores_df
heat_df = modelScores_df.pivot(
    index='max_leaf_nodes', columns='criterion', values='precision'
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    data=heat_df, annot=True, fmt='.3f', linewidths=1, robust=True, ax=ax
)
ax.set_title('Precision of the Cross Validation')
plt.savefig('igor_report/precision_heatmap_t40.png')
plt.close()

indArgmax = np.argmax(modelScores_df['f1_score'])
print("\n--- Best Model Info ---")
print(modelScores_df.loc[indArgmax])

optimalCriterion = modelScores_df.loc[indArgmax, 'criterion']
optimalMaxLeaf = modelScores_df.loc[indArgmax, 'max_leaf_nodes']
print(f'The optimal hyper-parameters are: criterion = {optimalCriterion} and max_leaf_nodes = {optimalMaxLeaf}')

# we define the model with the best found hyperparameters
oDecTreeClf = DecisionTreeClassifier(criterion=optimalCriterion, max_leaf_nodes=optimalMaxLeaf)
oDecTreeClf = oDecTreeClf.fit(mX, vY)
print(f'The accuracy of the model is: {oDecTreeClf.score(mX, vY):.2%}')

# now, let's plot the confusion matrix
cm = confusion_matrix(vY, vYPred) #order (y_true, y_predict)
cm_df = pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])
print("\n--- Confusion Matrix ---")
print(cm_df)

tn, fp, fn, tp = confusion_matrix(vY, vYPred).ravel()
cm_named_df = pd.DataFrame(
    {'Count': [tn, fp, fn, tp]},
    index=['TN (0→0)', 'FP (0→1)', 'FN (1→0)', 'TP (1→1)']
)
print("\n--- Confusion Matrix Breakdown ---")
print(cm_named_df)

print(f'Recall: {tp / (tp + fn):.2%}') # ability to detect true positives from all positives
print(f'Precision: {tp / (tp + fp):.2%}') # ability to detect true positives from all predicted positives
print(f'Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2%}') # overall accuracy

# importance of fetures
vFeatureImportance = oDecTreeClf.feature_importances_
print("\n--- Feature Importance ---")
for name, imp in zip(lfeatures_names, vFeatureImportance):
    print(f"{name}: {imp:.4f}")

# Plot the Feature Importance
hF, hA = plt.subplots(figsize = (10, 6))
hA.bar(x = lfeatures_names, height = vFeatureImportance, color='coral')
hA.set_title('Features Importance of the Model')
hA.set_xlabel('Feature Name')
hA.set_ylabel('Importance')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('igor_report/feature_importance_t40.png')
plt.close()
