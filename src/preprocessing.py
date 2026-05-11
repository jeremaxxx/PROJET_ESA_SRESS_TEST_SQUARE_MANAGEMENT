"""
Module de prétraitement des données pour les analyses de transition de rating
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

def prepare_corporate_ratings(file_path: str) -> pd.DataFrame:
    """
    Charge les données de notation d'entreprises à partir d'un fichier Excel
    et effectue un pré-traitement de base sur les colonnes de dates.

    Args:
        file_path (str): Le chemin d'accès vers le fichier Excel
                         'data_rating_corporate.xlsx'.

    Returns:
        pd.DataFrame: Un DataFrame pandas avec les colonnes de date
                      ('year_month', 'year') ajoutées et formatées.
                      
    Raises:
        FileNotFoundError: Si le fichier spécifié dans file_path n'est pas trouvé.
        Exception: Pour toute autre erreur lors de la lecture ou du traitement.
    """
    data = pd.read_excel(file_path)
    # data = pd.read_csv(file_path)

    # --- 2. Conversion de la colonne de date ---
    # La colonne 'rating_action_date' est convertie au format datetime.
    # errors='coerce' transforme les dates invalides en NaT (Not a Time).
    data['rating_action_date'] = pd.to_datetime(data['rating_action_date'], errors='coerce')
        
    # Il est recommandé de supprimer les lignes où la date est invalide
    data.dropna(subset=['rating_action_date'], inplace=True)

    # --- 3. Création des colonnes temporelles ---
    # Crée la colonne 'year_month' au format yyyy-mm-01
    data['year_month'] = data['rating_action_date'].dt.to_period('M').dt.to_timestamp()

    # Crée la colonne 'year' au format yyyy
    data['year'] = data['rating_action_date'].dt.year
        
    # La colonne 'year' peut être convertie en entier si aucune valeur NaT ne reste
    data['year'] = data['year'].astype(int)

    print("✅ Données chargées et préparées avec succès.")
    return data

def annualize_and_fill_ratings(df, obligor_col='obligor_name', date_col='rating_action_date'):
    """
    Annualise et comble un historique de notations pour chaque entité.

    Cette fonction effectue deux opérations principales :
    1.  Annualisation : Elle ne conserve que la notation la plus récente de chaque année pour chaque entité.
    2.  Remplissage : Elle comble les années manquantes en reportant la dernière notation connue.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée contenant les notations.
        obligor_col (str): Le nom de la colonne identifiant l'entité (ex: 'obligor_name').
        date_col (str): Le nom de la colonne contenant la date de notation.

    Returns:
        pd.DataFrame: Un nouveau DataFrame traité avec une ligne par entité et par an.
    """ 
    # --- Étape 0 : Préparation ---
    # Travailler sur une copie pour ne pas modifier le DataFrame original
    data = df.copy()
    # S'assurer que la colonne de date est au bon format et en extraire l'année
    data[date_col] = pd.to_datetime(data[date_col])
    data['year'] = data[date_col].dt.year

    # --- Étape 1 : Annualisation ---
    # Trier par date et ne garder que la dernière notation pour chaque entité/année
    data_annualised = (
        data.sort_values(date_col)
        .drop_duplicates(subset=[obligor_col, 'year'], keep='last')
    )

    # --- Étape 2 : Remplissage des années manquantes ---
    # Définir une fonction interne qui sera appliquée à chaque groupe
    def fill_gaps(group):
        # Créer un nouvel index avec toutes les années, de la plus ancienne à la plus récente du groupe
        all_years = range(group['year'].min(), group['year'].max() + 1)
        # Réindexer le groupe sur les années. Les années manquantes auront des NaN.
        return group.set_index('year').reindex(all_years)

    # Appliquer la fonction de remplissage et propager les dernières valeurs valides vers l'avant
    data_filled = (
        data_annualised
        .groupby(obligor_col, group_keys=False)  # Grouper par entité
        .apply(fill_gaps)                         # Appliquer la fonction de réindexation
        .ffill()                                  # Propager les données pour combler les NaN
    )

    # --- Étape 3 : Nettoyage Final ---
    # L'année est dans l'index, la remettre en colonne
    data_filled = data_filled.reset_index().rename(columns={'index': 'year'})

    # S'assurer que le type de la colonne 'year' est un entier
    data_filled['year'] = data_filled['year'].astype(int)
    # Recalculer les colonnes basées sur la date pour les nouvelles lignes créées
    # data_filled['year_month'] = pd.to_datetime(data_filled['year'].astype(str) + '-01-01')
    data_filled['next_rating'] = data_filled['rating'].shift(-1)

    # Filtrer pour retirer les lignes où 'next_rating' est NaN (dernière notation de chaque entreprise)
    data_filled_final = data_filled.dropna(subset=['next_rating'])

    return data_filled_final

def create_quarterly_rating_transitions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données de notation en une série temporelle trimestrielle complète
    pour chaque entreprise et calcule les transitions de notation sur 12 mois.

    Le processus se déroule en plusieurs étapes :
    1.  Conserve la dernière notation connue pour chaque entreprise dans un trimestre donné.
    2.  Crée une plage de temps complète (un "squelette") de tous les trimestres
        pour chaque entreprise, de sa première à sa dernière notation.
    3.  Fusionne les notations réelles sur ce squelette et propage la dernière
        notation connue vers l'avant (forward fill) pour combler les trimestres vides.
    4.  Calcule la notation 12 mois plus tard ('next_rating') pour chaque trimestre afin
        d'analyser les transitions.

    Args:
        data (pd.DataFrame): Un DataFrame contenant au minimum les colonnes
                             'obligor_name', 'rating_action_date', et 'rating'.
                             La colonne 'rating_action_date' doit être au format datetime.

    Returns:
        pd.DataFrame: Un DataFrame final avec une ligne par entreprise par trimestre,
                      contenant la notation du trimestre ('rating') et celle
                      12 mois après ('next_rating').
    """
    # Copie pour éviter de modifier le DataFrame original
    df = data.copy()

    # --- Étape 1: Créer une vue trimestrielle avec la dernière notation de chaque trimestre ---
    df['year_quarter'] = df['rating_action_date'].dt.to_period('Q').dt.start_time
    df_sorted = df.sort_values(by=['obligor_name', 'rating_action_date'])
    df_last_rating_in_quarter = df_sorted.drop_duplicates(subset=['obligor_name', 'year_quarter'], keep='last')

    # --- Étape 2: Générer une plage de trimestres complète pour chaque entreprise ---
    min_max_dates = df_sorted.groupby('obligor_name')['rating_action_date'].agg(['min', 'max'])
    all_quarters_dfs = []
    for company, row in min_max_dates.iterrows():
        full_quarter_range = pd.date_range(
            start=row['min'].to_period('Q').to_timestamp(),
            end=row['max'].to_period('Q').to_timestamp(),
            freq='QS'  # 'QS' pour le début de chaque trimestre (Quarter Start)
        )
        company_quarters_df = pd.DataFrame({'year_quarter': full_quarter_range, 'obligor_name': company})
        all_quarters_dfs.append(company_quarters_df)

    if not all_quarters_dfs:
        print("⚠️ Aucune donnée à traiter.")
        return pd.DataFrame()

    full_time_index = pd.concat(all_quarters_dfs, ignore_index=True)

    # --- Étape 3: Fusionner et propager les notations (Forward Fill) ---
    merged_df = pd.merge(
        full_time_index,
        df_last_rating_in_quarter,
        on=['obligor_name', 'year_quarter'],
        how='left'
    )
    merged_df.sort_values(by=['obligor_name', 'year_quarter'], inplace=True)
    
    # Remplir les valeurs manquantes en propageant la dernière notation valide PAR entreprise
    # C'est l'étape cruciale qui assure qu'un trimestre sans nouvelle notation
    # conserve la notation du trimestre précédent.
    # On propage également les autres colonnes utiles.

    # Si "rating_action_date" est manquant affecter la valeur de "year"quarter"
    merged_df['rating_action_date'] = merged_df['rating_action_date'].fillna(merged_df['year_quarter'])
    merged_df['year_month'] = merged_df['rating_action_date'].dt.to_period('M').dt.to_timestamp()

    cols_to_fill = ['rating', 'rating_agency_name', 'sector', 'country', 'legal_entity_identifier', 'year_month', 'year', 'pays', 'nace'] # Ajoutez les colonnes pertinentes
    cols_to_fill = [col for col in cols_to_fill if col in merged_df.columns] # Garde seulement les colonnes existantes
    merged_df[cols_to_fill] = merged_df.groupby('obligor_name')[cols_to_fill].ffill()

    # --- Étape 4: Calculer la notation future (transition sur 12 mois) ---
    merged_df['transition_date_lookup'] = merged_df['year_quarter'] + pd.DateOffset(months=12)

    # Fusionner le DataFrame avec lui-même pour trouver la notation à la date de transition
    final_df = pd.merge(
        merged_df,
        merged_df[['obligor_name', 'year_quarter', 'rating']],
        left_on=['obligor_name', 'transition_date_lookup'],
        right_on=['obligor_name', 'year_quarter'],
        suffixes=('', '_future'),
        how='left'
    )

    # --- Étape 5: Nettoyage final ---
    final_df.rename(columns={'rating_future': 'next_rating'}, inplace=True)
    
    # Supprimer les colonnes de travail et les lignes où il n'y a pas de transition possible
    final_df.drop(columns=['transition_date_lookup', 'year_quarter_future'], inplace=True, errors='ignore')
    final_df.dropna(subset=['rating', 'next_rating'], inplace=True)

    return final_df

def truncating_data(data, date_begin, date_finish):
    return data[(data["rating_action_date"] >= date_begin) & (data["rating_action_date"] <= date_finish)]

@dataclass
class SectorPortfolio:
    name: str
    sectors: List[str]
    annual_data: pd.DataFrame
    quarterly_data: pd.DataFrame
    countries: Optional[List[str]] = None # Pour garder trace du filtre actuel

    @classmethod
    def create_from_processed_data(cls, name, sectors, annual_data, quarterly_data):
        # Cette méthode reste simple : elle filtre juste par secteur (NACE)
        annual_sector_data = annual_data[annual_data['nace'].isin(sectors)].copy()
        quarterly_sector_data = quarterly_data[quarterly_data['nace'].isin(sectors)].copy()
        
        return cls(name=name, sectors=sectors, 
                   annual_data=annual_sector_data, 
                   quarterly_data=quarterly_sector_data)

    def filter_by_region(self, region_name: str, country_list: List[str]) -> 'SectorPortfolio':
        """
        Crée une copie du portefeuille filtrée sur une zone géographique précise.
        """
        # Filtrage des deux DataFrames sur la liste de pays
        new_annual = self.annual_data[self.annual_data['pays'].isin(country_list)].copy()
        new_quarterly = self.quarterly_data[self.quarterly_data['pays'].isin(country_list)].copy()
        
        # Retourne un nouvel objet SectorPortfolio
        return SectorPortfolio(
            name=f"{self.name} - {region_name}",
            sectors=self.sectors,
            countries=country_list,
            annual_data=new_annual,
            quarterly_data=new_quarterly
        )
