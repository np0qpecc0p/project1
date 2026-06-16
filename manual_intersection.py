import pandas as pd

klio_path = r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\Klio_CID_unique_preferred_Imax_FIXED.xlsx"
dragon_path = r"C:\Users\igork\Desktop\main\ai-program-2026\project_1\data\raw_dragon_matrix.csv"

try:
    print("Loading Killa...")
    klio_df = pd.read_excel(klio_path)
    print(f"Killa shape: {klio_df.shape}")
    print(f"Killa columns: {klio_df.columns.tolist()[:10]}") # print first 10 columns

    print("\nLoading Dragon...")
    dragon_df = pd.read_csv(dragon_path)
    print(f"Dragon shape: {dragon_df.shape}")
    print(f"Dragon columns: {dragon_df.columns.tolist()[:10]}") # print first 10 columns

    # Try intersection
    # Usually CID is the index or a column in Dragon
    if 'CID' not in dragon_df.columns:
        if 'Unnamed: 0' in dragon_df.columns:
            dragon_df = dragon_df.rename(columns={'Unnamed: 0': 'CID'})
            print("Renamed 'Unnamed: 0' to 'CID' in Dragon")

    if 'CID' in klio_df.columns and 'CID' in dragon_df.columns:
        intersection = pd.merge(klio_df, dragon_df, on='CID', how='inner')
        print(f"\nIntersection size: {intersection.shape[0]} rows")
        
        # Checking duplicates
        print(f"Unique CIDs in Killa: {klio_df['CID'].nunique()}")
        print(f"Unique CIDs in Dragon: {dragon_df['CID'].nunique()}")
        print(f"Unique CIDs in Intersection: {intersection['CID'].nunique()}")
    else:
        print("\nCID column missing in one of the dataframes. Cannot merge directly.")
except Exception as e:
    print(f"Error: {e}")
