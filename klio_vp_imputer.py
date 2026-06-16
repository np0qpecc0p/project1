import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold

print("1. Загрузка базы данных экспериментального Vapor Pressure...")
vp_df = pd.read_csv(r'C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\output_vp_full_v4.csv')
vp_df = vp_df[['CID', 'vp_mmhg_pubchem_clean']].dropna()
vp_df.rename(columns={'vp_mmhg_pubchem_clean': 'vp_exp'}, inplace=True)
vp_df = vp_df[vp_df['vp_exp'] > 0]

print("2. Загрузка глобальной матрицы Dragon (raw_dragon_matrix.csv)...")
dragon_df = pd.read_csv(r'C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\raw_dragon_matrix.csv')

print("3. Пересечение (Merge) экспериментального VP и фичей Dragon...")
vp_dragon_df = pd.merge(vp_df, dragon_df, how='inner', on='CID').dropna()
print(f"Количество молекул для обучения XGBoost: {vp_dragon_df.shape[0]}")

# Подготовка данных для обучения
X_train_raw = vp_dragon_df.loc[:, 'MW':]
y_train = np.log10(vp_dragon_df['vp_exp'])

print("4. Удаление фичей с нулевой дисперсией...")
vt = VarianceThreshold(threshold=0)
vt.fit(X_train_raw)
X_train = X_train_raw.loc[:, vt.get_support()]

print("5. Обучение модели XGBoost (параметры из vp_regression.ipynb)...")
best_params = {
    'colsample_bytree': 0.7, 
    'learning_rate': 0.05, 
    'max_depth': 4, 
    'min_child_weight': 3, 
    'n_estimators': 1000, 
    'reg_alpha': 0.1, 
    'reg_lambda': 3, 
    'subsample': 0.8,
    'random_state': 42
}
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

print("6. Загрузка ПОЛНОГО датасета Klio (1272 молекул)...")
klio_clean = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_dragon_merged.csv")
klio_clean['CID_merge'] = pd.to_numeric(klio_clean['CID'], errors='coerce')

# Подтягиваем известные (экспериментальные) значения VP
moodify_vp = pd.read_csv(r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\moodify_inventory_pubchem_bp_vp.csv")
moodify_vp['CID_merge'] = pd.to_numeric(moodify_vp['CID'], errors='coerce')
moodify_vp = moodify_vp.drop_duplicates(subset=['CID_merge'])

klio_full = pd.merge(klio_clean, moodify_vp[['CID_merge', 'vp_mmhg_pubchem']], on='CID_merge', how='left')

missing_mask = klio_full['vp_mmhg_pubchem'].isna()
missing_cids = klio_full.loc[missing_mask, 'CID_merge'].dropna().astype(int)

print(f"Молекул с отсутствующим VP в Klio: {missing_mask.sum()}")

print("7. Извлечение фичей Dragon для недостающих молекул Klio и предсказание VP...")
# Находим фичи для недостающих молекул в dragon_df
missing_dragon = dragon_df[dragon_df['CID'].isin(missing_cids)]

# Предсказываем
if not missing_dragon.empty:
    X_missing = missing_dragon[X_train.columns] # Берем только те фичи, на которых обучались
    log10_vp_pred = model.predict(X_missing)
    vp_pred = 10 ** log10_vp_pred
    
    # Создаем маппинг CID -> предсказанный VP
    pred_map = dict(zip(missing_dragon['CID'], vp_pred))
    
    # Заполняем пропуски
    klio_full['Vapor_Pressure'] = klio_full['vp_mmhg_pubchem']
    klio_full.loc[missing_mask, 'Vapor_Pressure'] = klio_full.loc[missing_mask, 'CID_merge'].map(pred_map)
else:
    klio_full['Vapor_Pressure'] = klio_full['vp_mmhg_pubchem']

# Заполняем финальные пропуски (если молекулы вообще не было в dragon_df) медианой, чтобы сеть не падала с NaN
median_vp = klio_full['Vapor_Pressure'].median()
klio_full['Vapor_Pressure'] = klio_full['Vapor_Pressure'].fillna(median_vp)
klio_full['log10_Vapor_Pressure'] = np.log10(klio_full['Vapor_Pressure'].replace(0, 1e-6))

print(f"Готово! В датасете klio_full_with_imputed_vp.csv теперь {klio_full['Vapor_Pressure'].notna().sum()} значений Vapor Pressure.")

klio_full.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_full_with_imputed_vp.csv", index=False)
