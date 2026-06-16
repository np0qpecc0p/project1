import pandas as pd
import numpy as np
import os
import subprocess

print("Шаг 1: Разделение датасета...")
df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_dragon_merged.csv")

# Ключевые слова для серных соединений
sulfur_keywords = ['sulfide', 'mercaptan', 'thio', 'thiane', 'thiazole', 'thiophene', 'sulfoxide', 'sulfone']

# Находим серные по названию
sulfur_mask = df['Canonical_Name'].str.lower().apply(lambda x: any(k in str(x) for k in sulfur_keywords) if pd.notnull(x) else False)

# Также подтягиваем наши 320 "математических" аутлаеров (аминокислоты и прочая жесть)
outliers_path = r"C:\Users\igork\Desktop\main\Project\absolute_outliers.csv"
if os.path.exists(outliers_path):
    df_out = pd.read_csv(outliers_path)
    outlier_names = df_out['Canonical_Name'].values
    math_outlier_mask = df['Canonical_Name'].isin(outlier_names)
else:
    math_outlier_mask = pd.Series([False]*len(df))

final_outlier_mask = sulfur_mask | math_outlier_mask

df_outliers = df[final_outlier_mask]
df_clean = df[~final_outlier_mask]

df_outliers.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_sulfur_outliers.csv", index=False)
df_clean.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_clean.csv", index=False)

print(f"Отделено аутлаеров (серные + аминокислоты): {len(df_outliers)}")
print(f"Осталось чистых молекул: {len(df_clean)}")

print("\nШаг 2: Запуск нейросети...")
python_exe = r"C:\Users\igork\Desktop\main\Meeseeks_Box\env\Meeseeks_Box\python.exe"
subprocess.run([python_exe, r"C:\Users\igork\Desktop\main\Project\moe_pytorch.py"])

print("\nШаг 3: Сбор датасета с ошибками > 15...")
res_df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv")
if 'Preferred_Imax' in res_df.columns:
    res_df['Real_Imax'] = res_df['Preferred_Imax']
else:
    res_df['Real_Imax'] = res_df['Imax']

res_df = res_df[res_df['MoE_Synergy_Pred'] != 0.0].copy()
res_df['Error'] = np.abs(res_df['Real_Imax'] - res_df['MoE_Synergy_Pred'])

df_errors = res_df[res_df['Error'] > 15].copy()
df_errors.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_errors_gt_15.csv", index=False)
print(f"Найдено молекул с ошибкой > 15: {len(df_errors)}")
