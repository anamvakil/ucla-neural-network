"""
pipeline.py
-----------
End-to-end training pipeline for UCLA Neural Network project.

Run directly:  py src/pipeline.py
"""

import logging
import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessor import (
    binarise_target,
    drop_unnecessary_columns,
    encode_features,
    split_features_target,
    scale_features,
)
from src.model import (
    train_mlp,
    evaluate_model,
    evaluate_train_accuracy,
    cross_validate_model,
    compare_activations,
    get_loss_curve,
)
from src.utils import setup_logging, save_model

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("data", "Admission.csv")
MODEL_DIR = "models"


def run_pipeline(data_path: str = DATA_PATH) -> dict:
    """
    Execute the full UCLA neural network pipeline.

    Args:
        data_path: Path to Admission.csv.

    Returns:
        Dictionary with trained model, evaluation results and metadata.
    """
    setup_logging()
    logger.info("=== UCLA Neural Network Pipeline START ===")

    # 1. Load
    df = load_data(data_path)

    # 2. Binarise target
    df = binarise_target(df, threshold=0.8)

    # 3. Drop Serial_No
    df = drop_unnecessary_columns(df)

    # 4. Encode
    df_enc = encode_features(df)

    # 5. Split features / target
    X, y = split_features_target(df_enc)
    feature_names = X.columns.tolist()

    # 6. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 7. Scale
    X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)

    # 8. Train default MLP (relu, 1 hidden layer, 3 neurons)
    mlp = train_mlp(X_tr_sc, y_train)

    # 9. Evaluate
    train_acc = evaluate_train_accuracy(mlp, X_tr_sc, y_train)
    eval_result = evaluate_model(mlp, X_te_sc, y_test, model_name="MLP (relu)")
    cv_result = cross_validate_model(mlp, X_tr_sc, y_train, model_name="MLP (relu)")
    loss_curve = get_loss_curve(mlp)

    # 10. Compare activation functions
    activation_df = compare_activations(X_tr_sc, y_train, X_te_sc, y_test)

    # 11. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(mlp, os.path.join(MODEL_DIR, "mlp_model.pkl"))
    save_model(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    save_model(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    logger.info("=== Pipeline COMPLETE ===")

    return {
        "model": mlp,
        "scaler": scaler,
        "feature_names": feature_names,
        "train_accuracy": train_acc,
        "eval": eval_result,
        "cv": cv_result,
        "loss_curve": loss_curve,
        "activation_comparison": activation_df,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"\nTrain Accuracy : {results['train_accuracy']:.4f}")
    print(f"Test Accuracy  : {results['eval']['accuracy']:.4f}")
    print(f"CV Mean        : {results['cv']['mean_accuracy']:.4f} ± {results['cv']['std_deviation']:.4f}")
    print("\nActivation Comparison:")
    print(results["activation_comparison"].to_string(index=False))
