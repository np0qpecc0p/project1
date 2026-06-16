import pandas as pd
import os

df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_annotated_by_ai.csv")

# Путь к папке с артефактами (чтобы таблица появилась прямо в UI чата)
artifact_path = r"C:\Users\igork\.gemini\antigravity-ide\brain\4c917536-f7c0-44cd-86da-a24219d8a237\manual_annotation.md"

md_content = "# Ручная классификация молекул (1272 шт)\n\n"
md_content += "Ниже представлена полная таблица всех молекул датасета с их химическими флагами. Вы можете прокрутить ее всю.\n\n"
md_content += "| Молекула (Canonical Name) | Has Sulfur (Сера) | Has Nitrogen Ring (Азот) | Is Amino Acid (Аминокислота) |\n"
md_content += "| :--- | :---: | :---: | :---: |\n"

for idx, row in df.iterrows():
    name = str(row['Canonical_Name'])
    s_val = "✅" if row['Has_Sulfur'] == 1.0 else "❌"
    n_val = "✅" if row['Has_Nitrogen_Ring'] == 1.0 else "❌"
    a_val = "✅" if row['Is_Amino_Acid'] == 1.0 else "❌"
    
    md_content += f"| {name} | {s_val} | {n_val} | {a_val} |\n"

with open(artifact_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print("Артефакт успешно создан!")
