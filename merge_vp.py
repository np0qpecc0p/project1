import pandas as pd

klio = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_clean.csv")
vp_db = pd.read_csv(r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\moodify_inventory_pubchem_bp_vp.csv")

klio['CID_merge'] = pd.to_numeric(klio['CID'], errors='coerce')
vp_db['CID_merge'] = pd.to_numeric(vp_db['CID'], errors='coerce')

# Убираем дубликаты из базы давления
vp_db_unique = vp_db.drop_duplicates(subset=['CID_merge'], keep='first')

# Мержим
klio_merged = pd.merge(klio, vp_db_unique[['CID_merge', 'vp_mmhg_pubchem', 'bp_c_pubchem']], on='CID_merge', how='left')

# Считаем пропуски
vp_valid = klio_merged['vp_mmhg_pubchem'].notnull().sum()
bp_valid = klio_merged['bp_c_pubchem'].notnull().sum()

print(f"Из 900 чистых молекул:")
print(f"Найдено Vapor Pressure (давление паров): {vp_valid} молекул")
print(f"Найдено Boiling Point (точка кипения): {bp_valid} молекул")

# Сохраняем расширенный датасет
klio_merged.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_clean_with_vp.csv", index=False)
