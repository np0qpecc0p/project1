import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Загружаем данные
df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv")
if 'Preferred_Imax' in df.columns:
    y = df['Preferred_Imax'].values
else:
    y = df['Imax'].values

preds = df['MoE_Synergy_Pred'].values
df['Error'] = np.abs(y - preds)

# 1. Топ-10 провалов
worst_df = df.sort_values('Error', ascending=False).head(10)
if 'Preferred_Imax' in df.columns:
    worst_mols = worst_df[['Canonical_Name', 'Preferred_Imax', 'MoE_Synergy_Pred', 'Error']].to_dict(orient='records')
else:
    worst_mols = worst_df[['name', 'Imax', 'MoE_Synergy_Pred', 'Error']].to_dict(orient='records')

# 2. Инлаеры и Аутлаеры по группам
with open(r"C:\Users\igork\Desktop\main\Project\moe_selected_features.json", 'r') as f:
    selected_features = json.load(f)

drop_cols = ['SMILES', 'Imax', 'Preferred_Imax', 'name', 'cid', 'CID', 'Canonical_Name', 'intensity_class', 'class', 'label', 'target', 'cluster', 'MoE_Synergy_Pred', 'Error']
X_raw = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

inlier_masks = {}
total_mols = len(y)

for group, feats in selected_features.items():
    if not feats: continue
    X_group = X_scaled[feats]
    
    base_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    ransac = RANSACRegressor(estimator=base_svr, random_state=42, min_samples=0.8)
    ransac.fit(X_group, y)
    
    inlier_masks[group] = ransac.inlier_mask_

# Считаем пересечения
mask_matrix = np.array(list(inlier_masks.values()))
# Аутлаеры во всех 4 группах
absolute_outliers = np.sum(mask_matrix, axis=0) == 0
# Инлаеры во всех 4 группах
absolute_inliers = np.sum(mask_matrix, axis=0) == 4

outliers_count = int(np.sum(absolute_outliers))
inliers_count = int(np.sum(absolute_inliers))

group_stats = {}
for g, m in inlier_masks.items():
    group_stats[g] = {'inliers': int(np.sum(m)), 'outliers': int(total_mols - np.sum(m))}

# 3. Как проверить гипотезу о "чистых" молекулах
# Посчитаем R2 только на абсолютных инлаерах
from sklearn.metrics import r2_score
y_inliers = y[absolute_inliers]
preds_inliers = preds[absolute_inliers]
if len(y_inliers) > 2:
    r2_inliers = r2_score(y_inliers, preds_inliers)
else:
    r2_inliers = 0.0

results = {
    'worst_failures': worst_mols,
    'group_stats': group_stats,
    'absolute_outliers': outliers_count,
    'absolute_inliers': inliers_count,
    'r2_on_absolute_inliers': r2_inliers,
    'total_mols': total_mols
}

with open(r"C:\Users\igork\Desktop\main\Project\analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print("Анализ завершен")
