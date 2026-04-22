import pandas as pd
from sklearn.model_selection import train_test_split
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
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Vérifier les valeurs manquantes
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Valeurs manquantes détectées : {missing}. Suppression des lignes concernées.")
        df = df.dropna()
    else:
        print("Aucune valeur manquante détectée")

    # Vérifier les doublons
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Doublons détectés : {duplicates}. Suppression des doublons.")
        df = df.drop_duplicates()
    else:
        print("Aucun doublon détecté")

    # Séparer features et target
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"Distribution du target : {dict(y.value_counts())}")

    # Split train/test (sans normalisation - RandomForest n'en a pas besoin)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
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
    print(f"Fichiers sauvegardés dans {output_dir}/")

if __name__ == "__main__":
    preprocess()