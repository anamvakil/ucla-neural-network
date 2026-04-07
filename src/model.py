"""
model.py
--------
Trains, evaluates and tunes sklearn MLPClassifier (Neural Network)
for the UCLA admission prediction task.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold

logger = logging.getLogger(__name__)


def train_mlp(
    X_train,
    y_train,
    hidden_layer_sizes: tuple = (3,),
    activation: str = "relu",
    batch_size: int = 50,
    max_iter: int = 200,
    random_state: int = 123,
) -> MLPClassifier:
    """
    Train a Multi-Layer Perceptron classifier.

    Args:
        X_train:            Scaled training features.
        y_train:            Training labels.
        hidden_layer_sizes: Tuple defining neurons per hidden layer (default (3,)).
        activation:         Activation function — 'relu' or 'tanh' (default 'relu').
        batch_size:         Mini-batch size for SGD (default 50).
        max_iter:           Maximum training iterations (default 200).
        random_state:       Seed for reproducibility.

    Returns:
        Fitted MLPClassifier instance.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info(
        "MLP trained. Layers: %s | Activation: %s | Iterations: %d",
        hidden_layer_sizes, activation, model.n_iter_,
    )
    return model


def evaluate_model(model: MLPClassifier, X_test, y_test, model_name: str = "MLP") -> dict:
    """
    Evaluate a trained MLP on the test set.

    Args:
        model:      Fitted MLPClassifier.
        X_test:     Scaled test features.
        y_test:     True test labels.
        model_name: Label for log output.

    Returns:
        Dictionary with accuracy, confusion_matrix, report and y_pred.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("%s | Test Accuracy: %.4f", model_name, acc)
    logger.info("%s | Confusion Matrix:\n%s", model_name, cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


def evaluate_train_accuracy(model: MLPClassifier, X_train, y_train) -> float:
    """
    Compute training accuracy to check for overfitting/underfitting.

    Args:
        model:   Fitted MLPClassifier.
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Training accuracy as float.
    """
    y_pred_train = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred_train)
    logger.info("Train Accuracy: %.4f", acc)
    return acc


def cross_validate_model(
    model: MLPClassifier,
    X_train,
    y_train,
    n_splits: int = 5,
    model_name: str = "MLP",
) -> dict:
    """
    Run KFold cross-validation on the training set.

    Args:
        model:      MLPClassifier instance (will be cloned internally).
        X_train:    Scaled training features.
        y_train:    Training labels.
        n_splits:   Number of CV folds (default 5).
        model_name: Label for log output.

    Returns:
        Dictionary with scores array, mean accuracy and std deviation.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    logger.info(
        "%s | CV Mean: %.4f | CV Std: %.4f",
        model_name, scores.mean(), scores.std(),
    )
    return {
        "scores": scores,
        "mean_accuracy": scores.mean(),
        "std_deviation": scores.std(),
    }


def compare_activations(
    X_train,
    y_train,
    X_test,
    y_test,
    activations: list = None,
) -> pd.DataFrame:
    """
    Train one MLP per activation function and return a comparison table.

    Args:
        X_train:     Scaled training features.
        y_train:     Training labels.
        X_test:      Scaled test features.
        y_test:      True test labels.
        activations: List of activation strings to compare (default relu, tanh, logistic).

    Returns:
        DataFrame with Activation, Train Accuracy and Test Accuracy columns.
    """
    if activations is None:
        activations = ["relu", "tanh", "logistic"]

    rows = []
    for act in activations:
        m = train_mlp(X_train, y_train, activation=act)
        train_acc = evaluate_train_accuracy(m, X_train, y_train)
        test_acc = evaluate_model(m, X_test, y_test, model_name=act)["accuracy"]
        rows.append({"Activation": act, "Train Accuracy": train_acc, "Test Accuracy": test_acc})
        logger.info("Activation %s | Train: %.4f | Test: %.4f", act, train_acc, test_acc)

    return pd.DataFrame(rows)


def get_loss_curve(model: MLPClassifier) -> list:
    """
    Return the training loss curve from a fitted MLP.

    Args:
        model: Fitted MLPClassifier with loss_curve_ attribute.

    Returns:
        List of loss values per iteration.
    """
    if not hasattr(model, "loss_curve_"):
        raise AttributeError("Model has no loss_curve_. Ensure it was trained with max_iter > 1.")
    return model.loss_curve_
