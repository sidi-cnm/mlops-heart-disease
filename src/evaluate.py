import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pickle
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    """Save a labelled confusion-matrix figure to *output_path*."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate():
    # ── Load params ────────────────────────────────────────────────────────────
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    eval_params   = params["evaluate"]
    mlflow_params = params["mlflow"]
    aws_params    = params["aws"]

    # ── Configure AWS ──────────────────────────────────────────────────────────
    os.environ["AWS_ACCESS_KEY_ID"]      = aws_params["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"]  = aws_params["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"]     = aws_params["region_name"]

    # ── Load test data ─────────────────────────────────────────────────────────
    data_dir = eval_params["data"]
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    # ── Load model ─────────────────────────────────────────────────────────────
    with open(eval_params["model_path"], "rb") as f:
        model = pickle.load(f)

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)

    # predict_proba is available on most sklearn estimators; fall back gracefully
    has_proba = hasattr(model, "predict_proba")
    y_proba   = model.predict_proba(X_test)[:, 1] if has_proba else None

    # ── Metrics ────────────────────────────────────────────────────────────────
    accuracy    = accuracy_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    roc_auc     = roc_auc_score(y_test, y_proba)   if has_proba else None
    avg_prec    = average_precision_score(y_test, y_proba) if has_proba else None

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["No Disease", "Disease"],
                                   output_dict=True)

    # Derived from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)          # true negative rate
    sensitivity = tp / (tp + fn)          # recall / true positive rate

    # ── Print summary ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  F1 (binary)     : {f1:.4f}")
    print(f"  F1 (macro)      : {f1_macro:.4f}")
    print(f"  F1 (weighted)   : {f1_weighted:.4f}")
    if roc_auc  is not None: print(f"  ROC-AUC         : {roc_auc:.4f}")
    if avg_prec is not None: print(f"  Avg Precision   : {avg_prec:.4f}")
    print(f"  Sensitivity     : {sensitivity:.4f}  (recall for Disease)")
    print(f"  Specificity     : {specificity:.4f}  (recall for No Disease)")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    # ── Confusion matrix figure ────────────────────────────────────────────────
    cm_image_path = os.path.join(data_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_image_path)

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Model Evaluation")

        # Core metrics
        mlflow.log_metric("test_accuracy",     accuracy)
        mlflow.log_metric("test_f1_binary",    f1)
        mlflow.log_metric("test_f1_macro",     f1_macro)
        mlflow.log_metric("test_f1_weighted",  f1_weighted)
        mlflow.log_metric("test_sensitivity",  sensitivity)
        mlflow.log_metric("test_specificity",  specificity)

        if roc_auc  is not None: mlflow.log_metric("test_roc_auc",       roc_auc)
        if avg_prec is not None: mlflow.log_metric("test_avg_precision",  avg_prec)

        # Per-class metrics from classification report
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    safe_label = label.replace(" ", "_")
                    mlflow.log_metric(f"test_{safe_label}_{metric}", value)

        # Confusion matrix values (useful for tracking over runs)
        mlflow.log_metric("test_cm_tn", int(tn))
        mlflow.log_metric("test_cm_fp", int(fp))
        mlflow.log_metric("test_cm_fn", int(fn))
        mlflow.log_metric("test_cm_tp", int(tp))

        # Artifacts
        mlflow.log_image(cm_image_path, "confusion_matrix.png")
        mlflow.log_text(
            classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]),
            "classification_report.txt",
        )

        # Log the model itself so it is versioned with this evaluation run
        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"\nÉvaluation terminée — Accuracy: {accuracy:.4f} | F1: {f1:.4f}"
          + (f" | ROC-AUC: {roc_auc:.4f}" if roc_auc else ""))


if __name__ == "__main__":
    evaluate()