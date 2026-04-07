"""
utils.py
--------
Logging setup, chart helpers and model persistence for the
UCLA Neural Network project.
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure root logger to write to console and a log file.

    Args:
        log_dir: Directory for log file (default 'logs').
        level:   Logging level (default INFO).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ucla_nn.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Logging initialised. Log: %s", log_path)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, model_name: str = "MLP") -> plt.Figure:
    """
    Render a labelled heatmap for a 2×2 confusion matrix.

    Args:
        cm:         Confusion matrix array.
        model_name: Title prefix.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Admitted", "Admitted"],
        yticklabels=["Not Admitted", "Admitted"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_loss_curve(loss_values: list, model_name: str = "MLP") -> plt.Figure:
    """
    Plot the training loss curve across iterations.

    Args:
        loss_values: List of loss values from model.loss_curve_.
        model_name:  Title label.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_values, color="steelblue", linewidth=2, label="Training Loss")
    ax.set_title(f"{model_name} — Loss Curve")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_activation_comparison(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Grouped bar chart comparing train vs test accuracy across activation functions.

    Args:
        comparison_df: DataFrame with columns Activation, Train Accuracy, Test Accuracy.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(comparison_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], comparison_df["Train Accuracy"], width, label="Train", color="teal", alpha=0.8)
    ax.bar([i + width/2 for i in x], comparison_df["Test Accuracy"], width, label="Test", color="coral", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison_df["Activation"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Activation Function Comparison")
    ax.legend()
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% target")
    plt.tight_layout()
    return fig


def plot_cv_scores(scores: np.ndarray, model_name: str = "MLP") -> plt.Figure:
    """
    Bar chart of cross-validation accuracy per fold with mean line.

    Args:
        scores:     Array of CV accuracy values.
        model_name: Title label.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    ax.bar(folds, scores, color="steelblue", alpha=0.8)
    ax.axhline(scores.mean(), color="red", linestyle="--", label=f"Mean: {scores.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — 5-Fold Cross-Validation")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame) -> plt.Figure:
    """
    GRE Score vs TOEFL Score scatter coloured by Admit_Chance.

    Args:
        df: Raw or binarised DataFrame with GRE_Score, TOEFL_Score, Admit_Chance.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {0: "#3498db", 1: "#e67e22"}
    labels = {0: "Not Admitted", 1: "Admitted"}
    for cls in [0, 1]:
        subset = df[df["Admit_Chance"] == cls]
        ax.scatter(subset["GRE_Score"], subset["TOEFL_Score"],
                   c=colours[cls], label=labels[cls], alpha=0.6, s=40)
    ax.set_xlabel("GRE Score")
    ax.set_ylabel("TOEFL Score")
    ax.set_title("GRE vs TOEFL Score by Admission Outcome")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, filepath: str) -> None:
    """Serialise a fitted model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath)
    logging.getLogger(__name__).info("Saved: %s", filepath)


def load_model(filepath: str):
    """Load a serialised model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    model = joblib.load(filepath)
    logging.getLogger(__name__).info("Loaded: %s", filepath)
    return model
