"""
End-to-End Training and Benchmarking Pipeline for Spotify Hit Predictor.
Orchestrates EDA visualization generation, feature engineering, cross-model training,
PyTorch deep learning, hyperparameter threshold tuning, evaluation, and SHAP explainability.
"""

import os
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
torch.manual_seed(42)
np.random.seed(42)
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from data_loader import load_raw_data, clean_data
from features import engineer_features, build_preprocessor, get_feature_column_names
from models import get_ml_models, SpotifyHitMLP
from evaluate import (
    calculate_metrics, find_optimal_threshold,
    plot_target_distribution, plot_genre_popularity,
    plot_correlation_matrix, plot_valence_energy,
    plot_model_curves, plot_confusion_matrices,
    plot_feature_importance, generate_shap_plots
)

def train_pytorch_mlp(X_train, y_train, X_val, y_val, pos_weight_val: float, epochs: int = 35, batch_size: int = 256):
    """
    Trains the PyTorch Multi-Layer Perceptron using BCEWithLogitsLoss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch MLP on device: {device}")
    
    input_dim = X_train.shape[1]
    model = SpotifyHitMLP(input_dim=input_dim).to(device)
    
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float("inf")
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_weights)
    model.eval()
    return model

def predict_pytorch_mlp(model, X_test):
    """
    Computes probabilities for PyTorch model.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def main():
    print("==================================================")
    print("   SPOTIFY HIT PREDICTION ML PIPELINE      ")
    print("==================================================")
    
    # 1. Load and clean raw data
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df, hit_threshold=60)
    
    # 2. Feature Engineering
    print("\n--- Applying Audio Feature Engineering ---")
    feat_df = engineer_features(clean_df)
    
    # Save processed dataset
    processed_path = "data/processed_data.csv"
    feat_df.to_csv(processed_path, index=False)
    print(f"Processed feature dataset saved to {processed_path} (Shape: {feat_df.shape})")
    
    # 3. Generate Exploratory Data Analysis (EDA) Figures
    print("\n--- Generating Exploratory Visualizations ---")
    plot_target_distribution(clean_df, "figures/01_target_distribution.png")
    plot_genre_popularity(clean_df, "figures/02_genre_popularity_distribution.png")
    plot_correlation_matrix(feat_df, "figures/03_audio_features_correlation_matrix.png")
    plot_valence_energy(feat_df, "figures/04_valence_vs_energy_scatter.png")
    
    # 4. Train / Val / Test Split (70% / 15% / 15%)
    numeric_cols, cat_cols = get_feature_column_names()
    all_features = numeric_cols + cat_cols
    
    X = feat_df[all_features]
    y = feat_df["is_hit"].values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
    ) # 0.17647 * 0.85 approx 0.15
    
    print(f"Dataset Partitions -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. Fit ColumnTransformer
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    
    # Save preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    
    # Get transformed feature names for SHAP and Feature Importance
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    transformed_feature_names = list(numeric_cols) + list(cat_feature_names)
    
    # Compute imbalance scale ratio
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / max(1, num_pos)
    print(f"Class Imbalance Ratio (Neg/Pos): {scale_pos_weight:.2f}")
    
    # 6. Train Models
    print("\n--- Training Machine Learning & Deep Learning Models ---")
    ml_models = get_ml_models(scale_pos_weight=scale_pos_weight)
    
    test_probs = {}
    optimal_thresholds = {}
    results_list = []
    
    # Train Scikit-Learn / Boosting Models
    for name, model in ml_models.items():
        print(f"Training: {name}...")
        model.fit(X_train_trans, y_train)
        
        # Validation probabilities for threshold tuning
        val_prob = model.predict_proba(X_val_trans)[:, 1]
        best_thresh, best_val_f1 = find_optimal_threshold(y_val, val_prob)
        optimal_thresholds[name] = best_thresh
        
        # Test evaluation
        test_prob = model.predict_proba(X_test_trans)[:, 1]
        test_probs[name] = test_prob
        
        # Metrics at default 0.5 threshold
        metrics_default = calculate_metrics(y_test, test_prob, threshold=0.5)
        metrics_default["Model"] = name
        metrics_default["Threshold Mode"] = "Default (0.50)"
        results_list.append(metrics_default)
        
        # Metrics at optimal threshold
        metrics_opt = calculate_metrics(y_test, test_prob, threshold=best_thresh)
        metrics_opt["Model"] = name
        metrics_opt["Threshold Mode"] = f"Tuned ({best_thresh:.2f})"
        results_list.append(metrics_opt)
        
        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(model, model_filename)
        
    # Train PyTorch Deep MLP
    print("Training: PyTorch Deep MLP...")
    pytorch_mlp = train_pytorch_mlp(X_train_trans, y_train, X_val_trans, y_val, pos_weight_val=scale_pos_weight, epochs=35)
    
    val_prob_pt = predict_pytorch_mlp(pytorch_mlp, X_val_trans)
    best_thresh_pt, best_val_f1_pt = find_optimal_threshold(y_val, val_prob_pt)
    optimal_thresholds["PyTorch Deep MLP"] = best_thresh_pt
    
    test_prob_pt = predict_pytorch_mlp(pytorch_mlp, X_test_trans)
    test_probs["PyTorch Deep MLP"] = test_prob_pt
    
    metrics_pt_def = calculate_metrics(y_test, test_prob_pt, threshold=0.5)
    metrics_pt_def["Model"] = "PyTorch Deep MLP"
    metrics_pt_def["Threshold Mode"] = "Default (0.50)"
    results_list.append(metrics_pt_def)
    
    metrics_pt_opt = calculate_metrics(y_test, test_prob_pt, threshold=best_thresh_pt)
    metrics_pt_opt["Model"] = "PyTorch Deep MLP"
    metrics_pt_opt["Threshold Mode"] = f"Tuned ({best_thresh_pt:.2f})"
    results_list.append(metrics_pt_opt)
    
    torch.save(pytorch_mlp.state_dict(), "models/pytorch_spotify_mlp.pt")
    
    # 7. Model Evaluation Visualizations
    print("\n--- Generating Evaluation Curves and Confusion Matrices ---")
    plot_model_curves(test_probs, y_test, "figures/05_model_roc_pr_curves.png")
    
    # Generate predictions dict for confusion matrices (using optimal thresholds)
    pred_dict = {
        "Logistic Reg (Tuned)": (test_probs["Logistic Regression (Baseline)"] >= optimal_thresholds["Logistic Regression (Baseline)"]).astype(int),
        "Random Forest (Tuned)": (test_probs["Random Forest"] >= optimal_thresholds["Random Forest"]).astype(int),
        "LightGBM (Tuned)": (test_probs["LightGBM"] >= optimal_thresholds["LightGBM"]).astype(int),
        "XGBoost (Tuned)": (test_probs["XGBoost"] >= optimal_thresholds["XGBoost"]).astype(int),
        "Deep MLP (Tuned)": (test_probs["PyTorch Deep MLP"] >= optimal_thresholds["PyTorch Deep MLP"]).astype(int)
    }
    plot_confusion_matrices(y_test, pred_dict, "figures/06_confusion_matrices.png")
    
    # Feature Importance for LightGBM
    lgbm_model = ml_models["LightGBM"]
    plot_feature_importance(lgbm_model, transformed_feature_names, "figures/09_feature_importance_comparison.png")
    
    # 8. SHAP Explainability Plots
    print("\n--- Generating Explainable AI (SHAP) Visualizations ---")
    # Take a 1,000-sample slice for fast TreeSHAP computation
    shap_sample_idx = np.random.RandomState(42).choice(len(X_test_trans), size=min(1000, len(X_test_trans)), replace=False)
    X_shap_sample = X_test_trans[shap_sample_idx]
    
    generate_shap_plots(
        lgbm_model, 
        X_shap_sample, 
        transformed_feature_names, 
        sample_row_df=X_test.iloc[0],
        beeswarm_path="figures/07_shap_summary_beeswarm.png",
        waterfall_path="figures/08_shap_waterfall_case_study.png"
    )
    
    # 9. Results Summary Table
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("benchmark_results.csv", index=False)
    
    print("\n==================================================")
    print("            FINAL BENCHMARK RESULTS               ")
    print("==================================================")
    cols_to_print = ["Model", "Threshold Mode", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    print(results_df[cols_to_print].to_markdown(index=False))
    print("\nAll models, metrics, and figures generated successfully!")

if __name__ == "__main__":
    main()
