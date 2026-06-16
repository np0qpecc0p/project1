import pandas as pd
import numpy as np
import json
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 1. Загружаем данные
df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv")
if 'Preferred_Imax' in df.columns:
    y = df['Preferred_Imax'].values
else:
    y = df['Imax'].values

preds = df['MoE_Synergy_Pred'].values

# Убираем "0.0" (те, что не попали в тест из-за рандома)
valid_mask = preds != 0.0

with open(r"C:\Users\igork\Desktop\main\Project\moe_selected_features.json", 'r') as f:
    selected_features = json.load(f)

drop_cols = ['SMILES', 'Imax', 'Preferred_Imax', 'name', 'cid', 'CID', 'Canonical_Name', 'intensity_class', 'class', 'label', 'target', 'cluster', 'MoE_Synergy_Pred', 'Error']
X_raw = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

inlier_masks = {}
for group, feats in selected_features.items():
    if not feats: continue
    X_group = X_scaled[feats]
    base_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    ransac = RANSACRegressor(estimator=base_svr, random_state=42, min_samples=0.8)
    ransac.fit(X_group, y)
    inlier_masks[group] = ransac.inlier_mask_

mask_matrix = np.array(list(inlier_masks.values()))
# Абсолютные аутлаеры (0 инлаеров из 4)
absolute_outliers_mask = np.sum(mask_matrix, axis=0) == 0

df_outliers = df[absolute_outliers_mask]
df_inliers = df[~absolute_outliers_mask]

# Сохраняем аутлаеров
df_outliers.to_csv(r"C:\Users\igork\Desktop\main\Project\absolute_outliers.csv", index=False)

# Аналитика по аутлаерам
if 'SMILES' in df_outliers.columns:
    sulfur_count = df_outliers['SMILES'].str.contains('S', case=False).sum()
else:
    sulfur_count = 0

print("=== АНАЛИЗ АБСОЛЮТНЫХ АУТЛАЕРОВ (320 шт) ===")
print(f"Средний Imax аутлаеров: {df_outliers['Preferred_Imax'].mean():.2f}")
print(f"Средний Imax инлаеров: {df_inliers['Preferred_Imax'].mean():.2f}")
print(f"Количество молекул с Серой (Sulfur) среди аутлаеров: {sulfur_count} из {len(df_outliers)}")

# R2 на валидных инлаерах
from sklearn.metrics import r2_score
valid_inliers_mask = valid_mask & (~absolute_outliers_mask)
valid_outliers_mask = valid_mask & absolute_outliers_mask

if np.sum(valid_inliers_mask) > 0:
    r2_in = r2_score(y[valid_inliers_mask], preds[valid_inliers_mask])
    print(f"R2 сети на чистых инлаерах (исключая аутлаеров): {r2_in:.3f}")

if np.sum(valid_outliers_mask) > 0:
    r2_out = r2_score(y[valid_outliers_mask], preds[valid_outliers_mask])
    print(f"R2 сети на абсолютных аутлаерах: {r2_out:.3f}")
