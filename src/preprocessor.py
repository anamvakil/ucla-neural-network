"""
preprocessor.py
---------------
Handles target binarisation, feature encoding and MinMax scaling
for the UCLA Admission dataset.
"""

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def binarise_target(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Convert continuous Admit_Chance to binary class label.
    Admit_Chance >= threshold → 1 (admitted), else → 0.

    Args:
        df:        Raw DataFrame containing Admit_Chance column.
        threshold: Probability cutoff (default 0.8).

    Returns:
        DataFrame with Admit_Chance replaced by 0/1 integer.
    """
    df = df.copy()
    df["Admit_Chance"] = (df["Admit_Chance"] >= threshold).astype(int)
    admitted = df["Admit_Chance"].sum()
    logger.info(
        "Target binarised at %.2f. Admitted: %d / %d (%.1f%%)",
        threshold, admitted, len(df), admitted / len(df) * 100,
    )
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns that carry no predictive signal.

    Args:
        df: DataFrame after binarisation.

    Returns:
        DataFrame without Serial_No.
    """
    df = df.copy()
    if "Serial_No" in df.columns:
        df.drop("Serial_No", axis=1, inplace=True)
        logger.info("Dropped 'Serial_No'.")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode University_Rating and Research.
    Both are semantically categorical despite being stored as integers.

    Args:
        df: DataFrame after dropping unnecessary columns.

    Returns:
        Fully encoded DataFrame ready for modelling.
    """
    df = df.copy()
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")

    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype=int)
    logger.info("One-hot encoding applied. New shape: %s", df.shape)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X and target vector y.

    Args:
        df: Encoded DataFrame.

    Returns:
        Tuple (X, y).
    """
    X = df.drop("Admit_Chance", axis=1)
    y = df["Admit_Chance"]
    logger.info("X shape: %s | y shape: %s", X.shape, y.shape)
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Apply MinMax scaling, fitting only on training data.

    Args:
        X_train: Training feature matrix.
        X_test:  Test feature matrix.

    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("MinMax scaling applied.")
    return X_train_scaled, X_test_scaled, scaler
