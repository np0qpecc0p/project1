import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor
from sklearn.inspection import permutation_importance
import json
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Загрузка данных...")
    df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_dragon_merged.csv")
    
    # Отделяем Imax (в klio_dragon_merged.csv колонка называется Preferred_Imax)
    if 'Preferred_Imax' in df.columns:
        y = df['Preferred_Imax']
    else:
        y = df['Imax']
    # Удаляем все не-фичи и целевые переменные, чтобы не было утечки данных!
    drop_cols = ['SMILES', 'Imax', 'Preferred_Imax', 'name', 'cid', 'CID', 'Canonical_Name', 'intensity_class', 'class', 'label', 'target', 'cluster']
    X_raw = df.drop(columns=drop_cols, errors='ignore')
    
    # Оставляем только числовые фичи (чтобы избежать ValueError со строками вроде CAS-номеров)
    X_raw = X_raw.select_dtypes(include=[np.number])
    
    # Удаляем константные колонки
    X_raw = X_raw.loc[:, X_raw.nunique() > 1]
    
    # Заполняем NaN нулями (на случай если остались)
    X_raw = X_raw.fillna(0)
    
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)
    
    print(f"Всего признаков после очистки: {X_scaled.shape[1]}")
    
    # Группировка фичей по названиям (эвристика для дескрипторов Dragon)
    cols = list(X_scaled.columns)
    
    groups = {
        'Steric': [],
        'Electrostatic': [],
        'Hydrophobic': [],
        'H_Bonds': []
    }
    
    for c in cols:
        c_lower = c.lower()
        # Гидрофобность
        if any(sub in c_lower for sub in ['logp', 'alogp', 'mlogp', 'xlogp', 'ghose', 'crippen', 'hy']):
            groups['Hydrophobic'].append(c)
        # Водородные связи
        elif any(sub in c_lower for sub in ['hdon', 'hacc', 'n-o', 'o-h', 'n-h', 'h-05', 'tpsa', 'saacc', 'sadon']):
            groups['H_Bonds'].append(c)
        # Электростатика / Полярность
        elif any(sub in c_lower for sub in ['charge', 'pol', 'qindex', 'dipole', 'tpsa', 'mats', 'gats', 'jgi', 'peoe', 'q_']):
            groups['Electrostatic'].append(c)
        # Стерика / Топология / Форма
        elif any(sub in c_lower for sub in ['mw', 'sv', 'se', 'sp', 'si', 'mv', 'me', 'mp', 'mi', 'walk', 'path', 'randic', 'balaban', 'wiener', 'zagreb', 'vol', 'vv', 'vd', 'nat', 'nsk', 'nc', 'nb', 'nr', 'chi', 'kier', 'burt', 'pji', 'geom', 'rdf', 'morse', 'whim', 'getaway']):
            groups['Steric'].append(c)
    
    print("Распределение признаков по группам:")
    for g, features in groups.items():
        print(f"  {g}: {len(features)} фичей")
        
    # Функция для отбора топ-N фичей через RANSAC+SVR
    def select_top_features(X_group, y, top_n=4, n_seeds=5):
        if X_group.shape[1] == 0:
            return []
        if X_group.shape[1] <= top_n:
            return list(X_group.columns)
            
        feature_scores = {col: 0.0 for col in X_group.columns}
        
        for seed in range(n_seeds):
            # Жесткий RANSAC (ожидаем ~20% выбросов)
            base_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            ransac = RANSACRegressor(estimator=base_svr, random_state=seed, min_samples=0.8)
            ransac.fit(X_group, y)
            
            # Считаем Permutation Importance на инлаерах
            inlier_mask = ransac.inlier_mask_
            X_in = X_group[inlier_mask]
            y_in = y[inlier_mask]
            
            result = permutation_importance(ransac, X_in, y_in, n_repeats=5, random_state=seed, n_jobs=-1)
            
            # Суммируем важности по сидам
            for idx, col in enumerate(X_group.columns):
                feature_scores[col] += result.importances_mean[idx]
                
        # Сортируем и берем Топ-N
        sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
        return [f[0] for f in sorted_features[:top_n]]

    selected_features_dict = {}
    
    for group_name, feature_list in groups.items():
        if not feature_list:
            print(f"Внимание: Группа {group_name} пуста!")
            selected_features_dict[group_name] = []
            continue
            
        print(f"\nЗапуск RANSAC+SVR(rbf) для группы {group_name} (кандидатов: {len(feature_list)})...")
        X_group = X_scaled[feature_list]
        top_features = select_top_features(X_group, y, top_n=4, n_seeds=5)
        print(f"Отобранные фичи для {group_name}: {top_features}")
        selected_features_dict[group_name] = top_features
        
    # Сохраняем в JSON
    out_path = r"C:\Users\igork\Desktop\main\Project\moe_selected_features.json"
    with open(out_path, 'w') as f:
        json.dump(selected_features_dict, f, indent=4)
        
    print(f"\nОтбор завершен! Фичи сохранены в {out_path}")

if __name__ == "__main__":
    main()
