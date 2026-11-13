
"""
ISFE: Incremental Slimming-Fattening Ensemble
Credit Card Fraud Detection

Consolidated experimental pipeline.

Models:
1. Baseline Random Forest
2. Tuned Random Forest
3. ISFR Adaptation
4. ISFE Adaptation

Input:
30 original credit-card transaction features.

Output:
- Predictions from all four models
- Final evaluation metrics
- Classification reports
- CSV files containing predictions and results
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
SFR_DIR = DATA_DIR / "sfr"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FEATURES = 30


# ============================================================
# 2. LOAD TRAINED MODELS AND PREPROCESSING OBJECTS
# ============================================================

def load_artifacts():

    required_files = {
        "Baseline Random Forest": MODEL_DIR / "random_forest.joblib",
        "Tuned Random Forest": MODEL_DIR / "tuned_random_forest.joblib",
        "ISFR Model": MODEL_DIR / "isfr_best_model.joblib",
        "ISFE Model 1": MODEL_DIR / "subset_1_isfe.joblib",
        "ISFE Model 2": MODEL_DIR / "subset_2_isfe.joblib",
        "ISFE Model 3": MODEL_DIR / "subset_3_isfe.joblib",
        "Standard Scaler": SFR_DIR / "scaler.joblib",
        "PCA": SFR_DIR / "pca.joblib"
    }

    # Verify all required artifacts exist
    missing_files = [
        name for name, path in required_files.items()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing project artifacts: "
            + ", ".join(missing_files)
        )

    artifacts = {
        "baseline_rf": joblib.load(
            required_files["Baseline Random Forest"]
        ),

        "tuned_rf": joblib.load(
            required_files["Tuned Random Forest"]
        ),

        "isfr": joblib.load(
            required_files["ISFR Model"]
        ),

        "isfe_models": [
            joblib.load(required_files["ISFE Model 1"]),
            joblib.load(required_files["ISFE Model 2"]),
            joblib.load(required_files["ISFE Model 3"])
        ],

        "scaler": joblib.load(
            required_files["Standard Scaler"]
        ),

        "pca": joblib.load(
            required_files["PCA"]
        )
    }

    print("All models and preprocessing objects loaded.")

    return artifacts


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def validate_input(X):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"received {X.shape[1]}."
        )

    if X.isnull().any().any():
        raise ValueError(
            "Input contains missing values."
        )

    try:
        X = X.astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError(
            "All input features must be numeric."
        ) from error

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Input contains infinite or non-finite values."
        )

    return X


# ============================================================
# 4. COMPATIBLE MODEL PREDICTION
# ============================================================

def predict_compatible(model, X):

    """
    Match the input format to the format used during
    model training to avoid feature-name warnings.
    """

    if hasattr(model, "feature_names_in_"):
        return model.predict(X)

    return model.predict(X.to_numpy())


# ============================================================
# 5. FEATURE TRANSFORMATION
# ============================================================

def transform_features(X, scaler, pca):

    """
    Apply the previously fitted StandardScaler and PCA.

    No fitting or refitting is performed on new data.
    """

    # Match the scaler's original input format
    if hasattr(scaler, "feature_names_in_"):
        X_scaled = scaler.transform(X)
    else:
        X_scaled = scaler.transform(X.to_numpy())

    # Apply the previously fitted PCA
    X_pca = pca.transform(X_scaled)

    if X_pca.shape[1] != 27:
        raise ValueError(
            f"Expected 27 PCA features, "
            f"received {X_pca.shape[1]}."
        )

    if not np.isfinite(X_pca).all():
        raise ValueError(
            "PCA transformation produced non-finite values."
        )

    return X_pca


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

def predict_transactions(X, artifacts):

    """
    Generate predictions using all four approaches.
    """

    X = validate_input(X)

    # Original 30-feature representation
    baseline_predictions = predict_compatible(
        artifacts["baseline_rf"],
        X
    )

    tuned_predictions = predict_compatible(
        artifacts["tuned_rf"],
        X
    )

    # Refined 27-feature PCA representation
    X_pca = transform_features(
        X,
        artifacts["scaler"],
        artifacts["pca"]
    )

    # ISFR prediction
    isfr_predictions = artifacts["isfr"].predict(
        X_pca
    )

    # ISFE predictions from three Random Forest models
    member_predictions = np.array([
        model.predict(X_pca)
        for model in artifacts["isfe_models"]
    ])

    # Hard voting: at least two models must predict fraud
    isfe_predictions = (
        np.sum(member_predictions, axis=0) >= 2
    ).astype(int)

    predictions = {
        "Baseline Random Forest": baseline_predictions,
        "Tuned Random Forest": tuned_predictions,
        "ISFR Adaptation": isfr_predictions,
        "ISFE Adaptation": isfe_predictions
    }

    return predictions


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_predictions(y_true, predictions):

    """
    Calculate evaluation metrics for every model.
    """

    results = []

    for model_name, y_pred in predictions.items():

        metrics = {
            "Model": model_name,

            "Accuracy": accuracy_score(
                y_true, y_pred
            ),

            "Macro F1": f1_score(
                y_true,
                y_pred,
                average="macro"
            ),

            "Balanced Accuracy": balanced_accuracy_score(
                y_true,
                y_pred
            ),

            "Fraud Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Fraud Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Fraud F1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        }

        results.append(metrics)

    return pd.DataFrame(results)


# ============================================================
# 8. CLASSIFICATION REPORTS
# ============================================================

def display_classification_reports(y_true, predictions):

    for model_name, y_pred in predictions.items():

        print("\n" + "=" * 65)
        print(model_name)
        print("=" * 65)

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=[
                    "Legitimate",
                    "Fraud"
                ],
                zero_division=0
            )
        )

        print("Confusion Matrix:")

        print(
            confusion_matrix(
                y_true,
                y_pred
            )
        )


# ============================================================
# 9. SAVE EXPERIMENT OUTPUTS
# ============================================================

def save_results(y_true, predictions, results):

    # Save model predictions
    prediction_df = pd.DataFrame(predictions)

    prediction_df.insert(
        0,
        "Actual Class",
        y_true
    )

    prediction_path = (
        REPORT_DIR / "pipeline_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # Save evaluation metrics
    results_path = (
        REPORT_DIR / "pipeline_test_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print("\nOutput files saved:")
    print(prediction_path)
    print(results_path)


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("ISFE CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 65)

    # Load models
    print("\n[1/5] Loading trained models...")

    artifacts = load_artifacts()

    # Load test dataset
    print("\n[2/5] Loading test dataset...")

    X_test_path = DATA_DIR / "X_test.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            "Test dataset files are missing."
        )

    X_test = pd.read_csv(X_test_path)

    y_test = pd.read_csv(
        y_test_path
    ).values.ravel()

    X_test = validate_input(X_test)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature and label counts do not match."
        )

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError(
            "Test labels must contain only 0 and 1."
        )

    print("Test samples:", len(X_test))
    print("Features:", X_test.shape[1])
    print("Legitimate transactions:", np.sum(y_test == 0))
    print("Fraudulent transactions:", np.sum(y_test == 1))

    # Generate predictions
    print("\n[3/5] Generating model predictions...")

    predictions = predict_transactions(
        X_test,
        artifacts
    )

    for model_name, y_pred in predictions.items():

        print(
            f"{model_name}: "
            f"{np.sum(y_pred == 1)} fraud predictions"
        )

    # Evaluate models
    print("\n[4/5] Evaluating models...")

    results = evaluate_predictions(
        y_test,
        predictions
    )

    print("\nFINAL TEST RESULTS\n")

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )

    # Classification reports
    display_classification_reports(
        y_test,
        predictions
    )

    # Save outputs
    print("\n[5/5] Saving experiment outputs...")

    save_results(
        y_test,
        predictions,
        results
    )

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
