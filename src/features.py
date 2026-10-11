"""
Feature Engineering Module for Spotify Hit Predictor
Transforms raw acoustic attributes into domain-informed audio features,
encodes harmonic variables via cyclical representations, and scales inputs.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RAW_NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies music domain feature engineering:
    - Audio emotion index (Russell's circumplex affect: valence * energy)
    - Grooviness & party index (valence * danceability)
    - Dance to Energy balance ratio
    - Spoken word density (speechiness / danceability)
    - Cyclical transformation of musical Key (circle of fifths representation)
    - Song age relative to modern streaming era (2020 reference)
    - Duration in minutes
    """
    df_feat = df.copy()
    
    # 1. Affective & Emotional indices
    df_feat["mood_index"] = df_feat["valence"] * df_feat["energy"]
    df_feat["party_index"] = df_feat["valence"] * df_feat["danceability"]
    df_feat["dance_energy_ratio"] = df_feat["danceability"] / (df_feat["energy"] + 1e-5)
    df_feat["speech_to_dance"] = df_feat["speechiness"] / (df_feat["danceability"] + 1e-5)
    
    # 2. Duration transformation (minutes)
    df_feat["duration_min"] = df_feat["duration_ms"] / 60000.0
    
    # 3. Temporal era feature
    df_feat["song_age"] = 2020 - df_feat["release_year"]
    
    # 4. Harmonic Key cyclical encoding (12 semitones on musical pitch wheel)
    df_feat["key_sin"] = np.sin(2 * np.pi * df_feat["key"] / 12.0)
    df_feat["key_cos"] = np.cos(2 * np.pi * df_feat["key"] / 12.0)
    
    return df_feat

def get_feature_column_names():
    """
    Returns lists of feature column names by type.
    """
    numeric_features = [
        "danceability", "energy", "loudness", "speechiness", 
        "acousticness", "instrumentalness", "liveness", "valence", 
        "tempo", "duration_min", "mood_index", "party_index", 
        "dance_energy_ratio", "speech_to_dance", "song_age",
        "key_sin", "key_cos"
    ]
    categorical_features = ["playlist_genre", "mode"]
    return numeric_features, categorical_features

def build_preprocessor():
    """
    Builds Scikit-Learn ColumnTransformer for scaling and one-hot encoding.
    """
    numeric_cols, cat_cols = get_feature_column_names()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    return preprocessor
