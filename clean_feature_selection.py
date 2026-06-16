import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Загрузка данных чистого ядра (900 молекул)...")
    df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_clean.csv")
    
    if 'Preferred_Imax' in df.columns:
        y = df['Preferred_Imax']
    else:
        y = df['Imax']
        
    drop_cols = ['SMILES', 'Imax', 'Preferred_Imax', 'name', 'cid', 'CID', 'Canonical_Name', 'intensity_class', 'class', 'label', 'target', 'cluster']
    X_raw = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    X_raw = X_raw.loc[:, X_raw.nunique() > 1].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)
    
    print(f"Всего признаков: {X_scaled.shape[1]}")
    
    cols = list(X_scaled.columns)
    groups = {'Steric': [], 'Electrostatic': [], 'Hydrophobic': [], 'H_Bonds': []}
    
    for c in cols:
        c_lower = c.lower()
        if any(sub in c_lower for sub in ['logp', 'alogp', 'mlogp', 'xlogp', 'ghose', 'crippen', 'hy']):
            groups['Hydrophobic'].append(c)
        elif any(sub in c_lower for sub in ['hdon', 'hacc', 'n-o', 'o-h', 'n-h', 'h-05', 'tpsa', 'saacc', 'sadon']):
            groups['H_Bonds'].append(c)
        elif any(sub in c_lower for sub in ['charge', 'pol', 'qindex', 'dipole', 'tpsa', 'mats', 'gats', 'jgi', 'peoe', 'q_']):
            groups['Electrostatic'].append(c)
        elif any(sub in c_lower for sub in ['mw', 'sv', 'se', 'sp', 'si', 'mv', 'me', 'mp', 'mi', 'walk', 'path', 'randic', 'balaban', 'wiener', 'zagreb', 'vol', 'vv', 'vd', 'nat', 'nsk', 'nc', 'nb', 'nr', 'chi', 'kier', 'burt', 'pji', 'geom', 'rdf', 'morse', 'whim', 'getaway']):
            groups['Steric'].append(c)
            
    # Отбор фичей: RANSAC больше не нужен, данные уже чистые!
    def select_top_features_clean(X_group, y, top_n=6, n_seeds=3):
        if X_group.shape[1] == 0: return []
        if X_group.shape[1] <= top_n: return list(X_group.columns)
            
        feature_scores = {col: 0.0 for col in X_group.columns}
        
        for seed in range(n_seeds):
            # Обучаем обычный SVR(rbf), так как у нас больше нет экстремальных аутлаеров
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            svr.fit(X_group, y)
            
            result = permutation_importance(svr, X_group, y, n_repeats=3, random_state=seed, n_jobs=-1)
            for idx, col in enumerate(X_group.columns):
                feature_scores[col] += result.importances_mean[idx]
                
        sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
        return [f[0] for f in sorted_features[:top_n]]

    selected_features_dict = {}
    
    # Берем по 6 лучших фичей для каждой группы (так как дисперсия стала тонкой, нужно больше фичей для описания)
    for group_name, feature_list in groups.items():
        if not feature_list: continue
        print(f"Отбор SVR(rbf) для группы {group_name}...")
        X_group = X_scaled[feature_list]
        top_features = select_top_features_clean(X_group, y, top_n=6, n_seeds=3)
        print(f"Отобрано: {top_features}")
        selected_features_dict[group_name] = top_features
        
    out_path = r"C:\Users\igork\Desktop\main\Project\moe_clean_selected_features.json"
    with open(out_path, 'w') as f:
        json.dump(selected_features_dict, f, indent=4)
        
    print(f"\nНовые фичи для чистого ядра сохранены в {out_path}")

if __name__ == "__main__":
    main()
