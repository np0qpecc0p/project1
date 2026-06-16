import pandas as pd
import numpy as np
import json
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv")
if 'Preferred_Imax' in df.columns:
    y = df['Preferred_Imax'].values
else:
    y = df['Imax'].values

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
absolute_outliers_mask = np.sum(mask_matrix, axis=0) == 0

# Целевая переменная: 1 если Аутлаер, 0 если Инлаер
labels = absolute_outliers_mask.astype(int)

# Используем ВСЕ фичи из X_raw, чтобы посмотреть, можно ли их разделить
X_train, X_test, y_train, y_test = train_test_split(X_raw, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== ТОЧНОСТЬ ОТДЕЛЕНИЯ АУТЛАЕРОВ ТОЛЬКО ПО ФИЧАМ ===")
print(classification_report(y_test, y_pred, target_names=['Clean (0)', 'Outlier (1)']))
