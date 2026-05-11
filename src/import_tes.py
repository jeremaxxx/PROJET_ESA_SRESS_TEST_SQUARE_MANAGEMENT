import os
import pandas as pd
import re
import openpyxl

def importing_tes(path):
    # Lister tous les fichiers CSV dans le dossier
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    # Dictionnaire pour stocker les DataFrames
    dfs = {}

    # for csv_file in csv_files:
    #     file_path = os.path.join(path, csv_file)
    #     # Prendre uniqument les nombres de 'csv_file'
    #     year = re.sub(r'\D', '', csv_file)
    #     df = pd.read_csv(file_path)
    #     dfs[year] = df

    remove_cols = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR", "OUT"]
    remove_rows = ["OUT", "TLS", "VA"]
    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        # Prendre uniquement les nombres de 'csv_file'
        year = re.sub(r'\D', '', csv_file)
        df = pd.read_csv(file_path)
        
        # -------------------------------
        # Supprimer les colonnes contenant certaines mentions
        # -------------------------------
        cols_to_drop = [col for col in df.columns if any(m in col for m in remove_cols)]
        df.drop(columns=cols_to_drop, inplace=True)
        
        # -------------------------------
        # Filtrer les lignes de la première colonne "V1"
        # -------------------------------
        if "V1" in df.columns:
            mask = ~df["V1"].astype(str).str.contains("|".join(remove_rows))
            df = df[mask].copy()
        
        dfs[year] = df

        
    path_countries = path + "/ReadMe_ICIO_small.xlsx"
    df_pays = pd.read_excel(
        path_countries,
        sheet_name="Area_Activities",
        header=2,      # L'en-tête ("Code") est bien à la ligne 3 (index 2)
        usecols='C'    # ON NE LIT QUE LA COLONNE C
    )
    df_pays.rename(columns={'Code': 'Pays'})
    
    path_nace = path + "/NACE 38 - 88 detaille vf.xlsx"
    descp_nace = pd.read_excel(path_nace)
    descp_nace.ffill(inplace=True)

    return dfs, df_pays, descp_nace

def removing_countries_acronym(df_clean):
    new_cols = [col[4:] if len(col) > 4 else col for col in df_clean.columns]
    df_clean.columns = new_cols
    
    if "V1" in df_clean.columns:
            df_clean["V1"] = df_clean["V1"].apply(lambda x: x[4:] if isinstance(x, str) and len(x) > 4 else x)
            df_clean.set_index("V1", inplace=True)
    return df_clean

def leaving_countries(dfs):
    df_clean = {}
    
    for year, df in dfs.items():
        df_clean[year] = df.copy()
        
        # Nettoyage des noms de colonnes
        df_clean[year] = removing_countries_acronym(df_clean[year])
        # Nettoyage de la première colonne (toutes les lignes)
    
    df_agr = {}
    for year, df in dfs.items():
        # Somme des colonnes ayant le même nom
        df_cols_summed = df.T.groupby(df.T.index).sum().T       
        # Somme des lignes ayant le même nom (index)
        df_rows_summed = df_cols_summed.groupby(df_cols_summed.index).sum()
        df_agr[year] = df_rows_summed
    
    return df_agr

def filtering_country(dfs, country=""):
    """
    Filtre les DataFrames selon un pays unique.

    Args:
        dfs (dict of pd.DataFrame): dictionnaire d'années -> DataFrames
        country (str): pays à conserver

    Returns:
        dict: dictionnaire de DataFrames filtrés
    """
    if not country:
        raise ValueError("Le paramètre 'country' ne peut pas être vide.")

    dfs_filtered = {}

    for year, df in dfs.items():
        df_copy = df.copy()

        # -------------------------------
        # Filtrer les colonnes contenant le pays
        # -------------------------------
        cols_to_keep = [col for col in df_copy.columns if country in col]
        # Ajouter V1 si elle existe pour les lignes
        if "V1" in df_copy.columns:
            cols_to_keep = ["V1"] + cols_to_keep

        df_copy = df_copy[cols_to_keep]

        # -------------------------------
        # Filtrer les lignes selon V1
        # -------------------------------
        if "V1" in df_copy.columns:
            mask = df_copy["V1"].astype(str).str.contains(country)
            df_copy = df_copy[mask].copy()

        df_final = removing_countries_acronym(df_copy)
        dfs_filtered[year] = df_final
    
    return dfs_filtered

    
    

def changing_nace_framework(df_cleaned, descp_nace):
    # Agrégation des DataFrames de df_sum de NACE 88 vers NACE 17
    df_sum_nace17 = {}
    # Créer un mapping NACE 88 -> NACE 17 à partir de descp_nace
    nace88_to_17 = descp_nace.set_index('Nace 88')['NACE 17'].to_dict()

    for year, df in df_cleaned.items():
        # Remplacer les noms de colonnes par leur NACE 17
        new_cols = [nace88_to_17.get(col, col) for col in df.columns]
        df_nace17 = df.copy()
        df_nace17.columns = new_cols
        # Remplacer les indices par leur NACE 17
        new_idx = [nace88_to_17.get(idx, idx) for idx in df_nace17.index]
        df_nace17.index = new_idx
        # Grouper et sommer par NACE 17 (colonnes et lignes)
        df_nace17 = df_nace17.groupby(df_nace17.index).sum()
        df_nace17 = df_nace17.T.groupby(level=0).sum().T
        df_sum_nace17[year] = df_nace17
    return df_sum_nace17 
