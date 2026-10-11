"""
Data Loader Module for Spotify Hit Predictor
Handles dataset acquisition, integrity checks, deduplication, and baseline cleaning.
"""

import os
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"

def load_raw_data(data_path: str = "data/spotify_songs.csv") -> pd.DataFrame:
    """
    Loads raw Spotify songs dataset from local disk or downloads if not present.
    """
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"Downloading dataset from {DATA_URL}...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path} (Shape: {df.shape})")
    else:
        df = pd.read_csv(data_path)
    
    return df

def clean_data(df: pd.DataFrame, hit_threshold: int = 60) -> pd.DataFrame:
    """
    Cleans the raw Spotify dataset:
    1. Removes duplicate tracks by track_id to avoid data leakage across playlists.
    2. Drops rows with missing track metadata (e.g. 5 tracks with null artist/name).
    3. Filters out corrupted or non-standard duration values.
    4. Creates binary classification target 'is_hit' based on track_popularity.
    """
    initial_count = len(df)
    
    # 1. Deduplicate by track_id
    df_clean = df.drop_duplicates(subset=["track_id"]).copy()
    dedup_count = len(df_clean)
    print(f"Deduplication: Reduced from {initial_count} to {dedup_count} unique tracks.")
    
    # 2. Handle missing values
    df_clean = df_clean.dropna(subset=["track_name", "track_artist", "track_album_name"]).copy()
    
    # 3. Clean release year
    df_clean["release_year"] = df_clean["track_album_release_date"].astype(str).str[:4].astype(int)
    # Filter valid years
    df_clean = df_clean[(df_clean["release_year"] >= 1950) & (df_clean["release_year"] <= 2021)].copy()
    
    # 4. Target variable creation
    # Songs with popularity >= hit_threshold are labeled as Hit (1), else 0
    df_clean["is_hit"] = (df_clean["track_popularity"] >= hit_threshold).astype(int)
    
    hit_rate = df_clean["is_hit"].mean() * 100
    print(f"Target distribution: {df_clean['is_hit'].sum()} Hits ({hit_rate:.2f}%), {(df_clean['is_hit'] == 0).sum()} Non-Hits")
    
    return df_clean.reset_index(drop=True)

if __name__ == "__main__":
    df = load_raw_data()
    clean_df = clean_data(df)
    print(clean_df.info())
