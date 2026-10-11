"""
Machine Learning and Deep Learning Model Architectures for Spotify Hit Prediction.
Includes Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch Deep MLP.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class SpotifyHitMLP(nn.Module):
    """
    Deep PyTorch Multi-Layer Perceptron architecture for tabular audio classification.
    Incorporates Batch Normalization, Dropout regularization, and LeakyReLU activations.
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.25):
        super(SpotifyHitMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2.0),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

def get_ml_models(scale_pos_weight: float = 3.42):
    """
    Returns configured candidate machine learning models.
    """
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=2,
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=7,
            num_leaves=40,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }
    return models
