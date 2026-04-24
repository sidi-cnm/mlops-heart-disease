import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Column taxonomy ────────────────────────────────────────────────────────────
NUMERICAL_COLS   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL       = "target"


def clip_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Clip values outside [Q1 - factor*IQR, Q3 + factor*IQR] for each column.
    Operates only on the provided column subset; leaves everything else untouched.
    """
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)
        print(f"  [{col}] clipped to [{lower:.2f}, {upper:.2f}]")
    return df


def preprocess():
    # ── Load params ────────────────────────────────────────────────────────────
    params_path = os.path.join(BASE_DIR, "params.yaml")
    params = yaml.safe_load(open(params_path))
    pp = params["preprocess"]

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(pp["data"])
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # ── Basic validation ───────────────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        print("Warning: missing values detected — filling with column median/mode")
        for col in NUMERICAL_COLS:
            df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_COLS:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Separate features / target ─────────────────────────────────────────────
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # ── Stratified train/test split (before any fitting) ──────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=pp["test_size"],
        random_state=pp["random_state"],
        stratify=y,          # preserves class balance in both splits
    )
    print(f"Split → train: {len(X_train_raw)}, test: {len(X_test_raw)}")
    print(f"Class balance — train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # ── Outlier clipping (fit on train, apply to both) ─────────────────────────
    print("Clipping outliers (IQR × 1.5) on numerical columns …")
    X_train_clipped = clip_outliers_iqr(X_train_raw, NUMERICAL_COLS)

    # Derive clip bounds from train and apply the same bounds to test
    clip_bounds = {}
    for col in NUMERICAL_COLS:
        q1, q3 = X_train_raw[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        clip_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    X_test_clipped = X_test_raw.copy()
    for col, (lo, hi) in clip_bounds.items():
        X_test_clipped[col] = X_test_clipped[col].clip(lo, hi)

    # ── Scale numerical features only ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train_clipped[NUMERICAL_COLS]),
        columns=NUMERICAL_COLS,
        index=X_train_clipped.index,
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test_clipped[NUMERICAL_COLS]),
        columns=NUMERICAL_COLS,
        index=X_test_clipped.index,
    )

    # ── One-hot encode categorical features ───────────────────────────────────
    # fit on train, reindex test to match (fills missing dummies with 0)
    X_train_cat = pd.get_dummies(X_train_clipped[CATEGORICAL_COLS], drop_first=False)
    X_test_cat  = pd.get_dummies(X_test_clipped[CATEGORICAL_COLS],  drop_first=False)
    X_test_cat  = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    # ── Combine numerical + categorical ───────────────────────────────────────
    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test  = pd.concat([X_test_num,  X_test_cat],  axis=1)

    print(f"Final feature count: {X_train.shape[1]} "
          f"({len(NUMERICAL_COLS)} numerical + {X_train_cat.shape[1]} one-hot)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    output_dir = pp["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv( f"{output_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv( f"{output_dir}/y_test.csv",  index=False)

    # Save scaler so inference can reproduce the same transformation
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"Scaler saved → {output_dir}/scaler.pkl")

    print(f"\nPrétraitement terminé : {len(X_train)} train, {len(X_test)} test")


if __name__ == "__main__":
    preprocess()