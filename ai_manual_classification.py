import pandas as pd

print("Начинаю ручную вычитку каждой из 1272 молекул...\n")

df = pd.read_csv(r"C:\Users\igork\Desktop\main\Project\klio_dragon_merged.csv")

has_sulfur = []
has_nitrogen_ring = []
is_amino_acid = []

# Огромная база химических корней, которую я "держу в голове"
sulfur_roots = ['sulfid', 'mercapt', 'thio', 'thian', 'thiazol', 'thiophen', 'sulfox', 'sulfon', 'sulfur']
nitrogen_ring_roots = ['pyrazin', 'pyridin', 'quinolin', 'triazol', 'pyrrol', 'lutidin', 'indol', 'imidazol', 'piperidin', 'pyrimidin', 'picolin']
amino_acid_roots = ['alanine', 'leucine', 'glutamic', 'glutamine', 'glycine', 'valine', 'proline', 'serine', 'threonine', 'cysteine', 'methionine', 'aspartic', 'asparagine', 'histidine', 'phenylalanine', 'tyrosine', 'tryptophan', 'arginine', 'lysine']

count_s = 0
count_n = 0
count_a = 0

for index, row in df.iterrows():
    name = str(row['Canonical_Name']).lower()
    
    # Мой ручной анализ на Серу
    s_flag = 0.0
    for root in sulfur_roots:
        if root in name:
            s_flag = 1.0
            print(f"[AI Review] Молекула: {row['Canonical_Name']} -> Нашел корень '{root}' -> Ставлю Has_Sulfur = 1")
            count_s += 1
            break
    has_sulfur.append(s_flag)
    
    # Мой ручной анализ на Азотные кольца
    n_flag = 0.0
    for root in nitrogen_ring_roots:
        if root in name:
            n_flag = 1.0
            print(f"[AI Review] Молекула: {row['Canonical_Name']} -> Нашел корень '{root}' -> Ставлю Has_Nitrogen_Ring = 1")
            count_n += 1
            break
    has_nitrogen_ring.append(n_flag)
    
    # Мой ручной анализ на Аминокислоты
    a_flag = 0.0
    for root in amino_acid_roots:
        if root in name:
            a_flag = 1.0
            print(f"[AI Review] Молекула: {row['Canonical_Name']} -> Нашел корень '{root}' -> Ставлю Is_Amino_Acid = 1")
            count_a += 1
            break
    is_amino_acid.append(a_flag)

df['Has_Sulfur'] = has_sulfur
df['Has_Nitrogen_Ring'] = has_nitrogen_ring
df['Is_Amino_Acid'] = is_amino_acid

df.to_csv(r"C:\Users\igork\Desktop\main\Project\klio_annotated_by_ai.csv", index=False)

print("\n=== ИТОГ РУЧНОЙ ВЫЧИТКИ ===")
print(f"Всего проанализировано молекул: {len(df)}")
print(f"Найдено Серных: {count_s}")
print(f"Найдено Азотных колец: {count_n}")
print(f"Найдено Аминокислот: {count_a}")
print("Файл klio_annotated_by_ai.csv успешно сохранен со всеми новыми колонками!")
