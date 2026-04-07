"""
data_loader.py
--------------
Loads and validates the UCLA Admission dataset.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Serial_No", "GRE_Score", "TOEFL_Score", "University_Rating",
    "SOP", "LOR", "CGPA", "Research", "Admit_Chance",
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Admission CSV dataset.

    Args:
        filepath: Path to Admission.csv.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded. Shape: %s", df.shape)
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info("All required columns present.")
    return df


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return value counts of the binarised Admit_Chance column.

    Args:
        df: DataFrame that already has Admit_Chance binarised (0/1).

    Returns:
        Series with counts per class.
    """
    dist = df["Admit_Chance"].value_counts()
    logger.info("Class distribution:\n%s", dist)
    return dist
