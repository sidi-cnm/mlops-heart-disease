import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess():
    # Charger les paramètres
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    preprocess_params = params["preprocess"]

    # Charger les données
    df = pd.read_csv(preprocess_params["data"])

    # Séparer features et target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Normaliser
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["random_state"]
    )

    # Sauvegarder
    output_dir = preprocess_params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Preprocessing terminé : {len(X_train)} train, {len(X_test)} test")

if __name__ == "__main__":
    preprocess()