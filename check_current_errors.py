import pandas as pd
import numpy as np

# Считываем результаты последнего прогона (где сеть обучалась на всем с флагами)
df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\moe_synergy_results.csv")
if 'Preferred_Imax' in df.columns:
    df['Real_Imax'] = df['Preferred_Imax']
else:
    df['Real_Imax'] = df['Imax']

# Убираем фейковые 0
df_valid = df[df['MoE_Synergy_Pred'] != 0.0].copy()
df_valid['Error'] = np.abs(df_valid['Real_Imax'] - df_valid['MoE_Synergy_Pred'])

worst_true = df_valid.sort_values('Error', ascending=False).head(15)

print("=== ТОП-15 ПРОВАЛОВ НА СЕТИ С ФЛАГАМИ ===")
for idx, row in worst_true.iterrows():
    print(f"{row['Canonical_Name']} | Реал: {row['Real_Imax']:.1f} | Предсказание: {row['MoE_Synergy_Pred']:.1f} | Ошибка: {row['Error']:.1f}")
