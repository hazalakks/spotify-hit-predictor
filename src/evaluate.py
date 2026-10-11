"""
Evaluation and Visualization Module for Spotify Hit Predictor.
Computes comprehensive classification metrics, optimal threshold tuning,
and generates publication-quality visualization figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import shap

# Configure aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_DARK = "#191414"
ACCENT_BLUE = "#1E88E5"
ACCENT_ORANGE = "#FF7043"
ACCENT_PURPLE = "#8E24AA"

def calculate_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Computes a comprehensive dictionary of binary classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Threshold": threshold
    }

def find_optimal_threshold(y_true, y_prob) -> tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1-Score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return float(best_threshold), float(best_f1)

def plot_target_distribution(df: pd.DataFrame, output_path: str = "figures/01_target_distribution.png"):
    """
    Visualizes track popularity distribution and binary Hit vs Non-Hit proportions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Continuous Popularity Distribution
    sns.histplot(df["track_popularity"], kde=True, ax=axes[0], color="#2E7D32", bins=30, alpha=0.6)
    axes[0].axvline(60, color="#D32F2F", linestyle="--", linewidth=2, label="Hit Threshold (Popularity >= 60)")
    axes[0].set_title("Spotify Track Popularity Distribution (0 - 100)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Track Popularity Score")
    axes[0].set_ylabel("Track Count")
    axes[0].legend(frameon=True)
    
    # Subplot 2: Binary Class Balance
    hit_counts = df["is_hit"].value_counts(normalize=True) * 100
    bars = axes[1].bar(["Non-Hit (<60)", "Hit (>=60)"], hit_counts.values, color=["#90A4AE", SPOTIFY_GREEN], width=0.5)
    axes[1].set_title("Target Class Balance (Hit vs. Non-Hit)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_genre_popularity(df: pd.DataFrame, output_path: str = "figures/02_genre_popularity_distribution.png"):
    """
    Visualizes popularity distribution across genres.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Boxplot of Popularity by Genre
    order = df.groupby("playlist_genre")["track_popularity"].median().sort_values(ascending=False).index
    palette = sns.color_palette("viridis", len(order))
    sns.boxplot(data=df, x="playlist_genre", y="track_popularity", order=order, ax=axes[0], hue="playlist_genre", palette=palette, legend=False)
    axes[0].set_title("Track Popularity Distribution Across Genres", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Genre")
    axes[0].set_ylabel("Popularity (0-100)")
    
    # Subplot 2: Hit Rate by Genre
    genre_hit_rate = df.groupby("playlist_genre")["is_hit"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=genre_hit_rate.index, y=genre_hit_rate.values, ax=axes[1], hue=genre_hit_rate.index, palette="crest", legend=False)
    axes[1].set_title("Commercial Hit Rate (%) by Genre", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Genre")
    axes[1].set_ylabel("Hit Rate (%)")
    for i, v in enumerate(genre_hit_rate.values):
        axes[1].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(genre_hit_rate.values) + 8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "figures/03_audio_features_correlation_matrix.png"):
    """
    Plots correlation heatmap among core acoustic features and popularity.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = [
        "track_popularity", "danceability", "energy", "loudness", 
        "speechiness", "acousticness", "instrumentalness", "liveness", 
        "valence", "tempo"
    ]
    if "duration_min" in df.columns:
        cols.append("duration_min")
    elif "duration_ms" in df.columns:
        cols.append("duration_ms")
    if "song_age" in df.columns:
        cols.append("song_age")
        
    corr = df[cols].corr()
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
    plt.title("Acoustic Features & Popularity Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_valence_energy(df: pd.DataFrame, output_path: str = "figures/04_valence_vs_energy_scatter.png"):
    """
    Visualizes Russell's Circumplex Affect grid: Valence vs Energy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df = df.sample(min(2500, len(df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=sample_df,
        x="valence",
        y="energy",
        hue="is_hit",
        palette={0: "#90A4AE", 1: SPOTIFY_GREEN},
        alpha=0.6,
        s=35,
        ax=ax
    )
    
    # Affective Quadrants
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.7)
    ax.text(0.75, 0.95, "Euphoric / Energetic\n(Party Hits)", fontsize=11, fontweight="bold", color="#1B5E20", ha="center")
    ax.text(0.25, 0.95, "Tense / Aggressive\n(Heavy/Trap)", fontsize=11, fontweight="bold", color="#B71C1C", ha="center")
    ax.text(0.25, 0.05, "Melancholic / Sad\n(Lo-Fi/Ballad)", fontsize=11, fontweight="bold", color="#0D47A1", ha="center")
    ax.text(0.75, 0.05, "Peaceful / Calm\n(Acoustic Chill)", fontsize=11, fontweight="bold", color="#E65100", ha="center")
    
    ax.set_title("Russell's Affect Grid: Valence (Happiness) vs. Energy by Hit Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Valence (Musical Positiveness: 0.0 -> 1.0)")
    ax.set_ylabel("Energy (Intensity & Activity: 0.0 -> 1.0)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Non-Hit (<60)", "Hit (>=60)"], loc="lower center", frameon=True, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_model_curves(model_probs: dict, y_true, output_path: str = "figures/05_model_roc_pr_curves.png"):
    """
    Plots comparative ROC and Precision-Recall Curves for all benchmarked models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = [ACCENT_BLUE, "#E65100", SPOTIFY_GREEN, "#6A1B9A", "#D81B60"]
    
    # 1. ROC Curves
    for (name, prob), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    baseline_pr = y_true.mean()
    for (name, prob), color in zip(model_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, prob)
        pr_auc = average_precision_score(y_true, prob)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2)
    
    axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.5, label=f"No Skill Baseline ({baseline_pr:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall (Coverage of Real Hits)")
    axes[1].set_ylabel("Precision (Accuracy of Hit Calls)")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_confusion_matrices(y_true, predictions_dict: dict, output_path: str = "figures/06_confusion_matrices.png"):
    """
    Plots normalized confusion matrices for candidate models side by side.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
        
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Non-Hit", "Hit"], yticklabels=["Non-Hit", "Hit"])
        ax.set_title(f"{name}\nNormalized Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_importance(model, feature_names: list, output_path: str = "figures/09_feature_importance_comparison.png"):
    """
    Plots top feature importance rankings from tree models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Normalize
        importances = importances / np.sum(importances) * 100
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature", palette="mako", legend=False)
        plt.title("Top 15 Feature Importances (LightGBM)", fontsize=13, fontweight="bold")
        plt.xlabel("Relative Importance (%)")
        plt.ylabel("Engineered Audio Feature")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

def generate_shap_plots(model, X_sample, feature_names: list, sample_row_df, 
                        beeswarm_path: str = "figures/07_shap_summary_beeswarm.png",
                        waterfall_path: str = "figures/08_shap_waterfall_case_study.png"):
    """
    Generates TreeSHAP summary beeswarm and local waterfall case study plots.
    """
    os.makedirs(os.path.dirname(beeswarm_path), exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    shap_values.feature_names = list(feature_names)  # readable labels in waterfall plot
    
    # 1. Summary Beeswarm Plot
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=12)
    plt.title("SHAP Beeswarm Plot: Feature Impact on Hit Prediction", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {beeswarm_path}")
    
    # 2. Local Waterfall Plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("SHAP Waterfall: Case Study on Single Track", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {waterfall_path}")
