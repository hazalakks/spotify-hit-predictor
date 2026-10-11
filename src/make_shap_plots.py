"""Regenerate SHAP plots from saved models (same split/sample as train.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from features import get_feature_column_names
from evaluate import generate_shap_plots

df = pd.read_csv("data/processed_data.csv")
num, cat = get_feature_column_names()
X, y = df[num + cat], df["is_hit"].values
_, X_test, _, _ = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
pre = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/lightgbm.joblib")
Xt = pre.transform(X_test)
names = list(num) + list(pre.named_transformers_["cat"].get_feature_names_out(cat))
idx = np.random.RandomState(42).choice(len(Xt), size=min(1000, len(Xt)), replace=False)
generate_shap_plots(model, Xt[idx], names, None)
