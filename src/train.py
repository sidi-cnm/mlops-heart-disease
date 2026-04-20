import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pickle
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train():
    # Charger les paramètres
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    train_params = params["train"]
    mlflow_params = params["mlflow"]
    aws_params = params["aws"]

    # Configurer AWS
    os.environ["AWS_ACCESS_KEY_ID"] = aws_params["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_params["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = aws_params["region_name"]

    # Charger les données
    X_train = pd.read_csv(f"{train_params['data']}/X_train.csv")
    y_train = pd.read_csv(f"{train_params['data']}/y_train.csv").values.ravel()

    # Configurer MLflow
    mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run():
        # Signature du modèle
        signature = infer_signature(X_train, y_train)

        # Tags MLflow
        mlflow.set_tag("mlflow.runName", "RandomForest Training")
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Hyperparamètres
        param_grid = {
            "n_estimators": [train_params["n_estimators"]],
            "max_depth": [train_params["max_depth"]],
            "min_samples_leaf": [1, 2]
        }

        # GridSearch
        rf = RandomForestClassifier(random_state=train_params["random_state"])
        grid_search = GridSearchCV(rf, param_grid, cv=2, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Logger les paramètres et métriques
        mlflow.log_params(grid_search.best_params_)
        train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
        mlflow.log_metric("train_accuracy", train_accuracy)

        # Logger le modèle dans MLflow Registry
        mlflow.sklearn.log_model(
            best_model,
            mlflow_params["model_name"],
            registered_model_name=mlflow_params["model_name"],
            signature=signature
        )

        # Sauvegarder localement
        os.makedirs(os.path.dirname(train_params["model_path"]), exist_ok=True)
        with open(train_params["model_path"], "wb") as f:
            pickle.dump(best_model, f)

        print(f"Modèle entraîné - Train accuracy: {train_accuracy:.4f}")
        print(f"Best params: {grid_search.best_params_}")

if __name__ == "__main__":
    train()