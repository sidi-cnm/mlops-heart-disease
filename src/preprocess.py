import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import boto3
from io import StringIO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv_from_s3(s3_path, bucket):
    """Lire un CSV depuis S3"""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=s3_path)
    return pd.read_csv(obj["Body"])


def save_csv_to_s3(df, s3_key, bucket):
    """Sauvegarder un DataFrame en CSV sur S3"""
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=csv_buffer.getvalue())


def preprocess():
    # Charger les paramètres
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    preprocess_params = params["preprocess"]
    aws_params = params["aws"]

    # Configurer AWS
    os.environ["AWS_ACCESS_KEY_ID"] = aws_params["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_params["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = aws_params["region_name"]

    S3_BUCKET = "mlops-heart-disease-groupe-2"

    # Charger les données depuis S3
    df = read_csv_from_s3(preprocess_params["data"], S3_BUCKET)
    print(f"Dataset chargé depuis S3 : {df.shape[0]} lignes, {df.shape[1]} colonnes")

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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["random_state"]
    )

    # Sauvegarder sur S3
    output_dir = preprocess_params["output_dir"]
    save_csv_to_s3(X_train, f"{output_dir}/X_train.csv", S3_BUCKET)
    save_csv_to_s3(X_test, f"{output_dir}/X_test.csv", S3_BUCKET)
    save_csv_to_s3(y_train.to_frame(), f"{output_dir}/y_train.csv", S3_BUCKET)
    save_csv_to_s3(y_test.to_frame(), f"{output_dir}/y_test.csv", S3_BUCKET)

    # Sauvegarder localement (pour le pipeline DVC)
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Preprocessing terminé : {len(X_train)} train, {len(X_test)} test")
    print(f"Fichiers sauvegardés localement dans {output_dir}/ et sur s3://{S3_BUCKET}/{output_dir}/")


if __name__ == "__main__":
    preprocess()