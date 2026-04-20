import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import mlflow
import pickle
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate():
    # Charger les paramètres
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    eval_params = params["evaluate"]
    mlflow_params = params["mlflow"]
    aws_params = params["aws"]

    # Configurer AWS
    os.environ["AWS_ACCESS_KEY_ID"] = aws_params["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_params["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = aws_params["region_name"]

    # Charger les données de test
    X_test = pd.read_csv(f"{eval_params['data']}/X_test.csv")
    y_test = pd.read_csv(f"{eval_params['data']}/y_test.csv").values.ravel()

    # Charger le modèle
    with open(eval_params["model_path"], "rb") as f:
        model = pickle.load(f)

    # Prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Logger dans MLflow
    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Model Evaluation")

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)

        # Logger les métriques par classe
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"test_{label}_{metric}", value)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(report), "classification_report.txt")

        print(f"Evaluation terminée - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    evaluate()