from pathlib import Path

import numpy as np
import pandas as pd
import shap
import catboost as cb


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_DATA_DIR = PROJECT_ROOT / "intermediate" / "group_data_eps2bins_3_r025065"
OUTPUT_DIR = PROJECT_ROOT / "output" / "shap_eps2bins_3_r025065_niles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20000


# =========================
# Feature settings
# =========================
FEATURE_ORDER = [
    "CR",
    "F",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
    "Instance variance",
    "Stochastic variance",
]

RAW_FEATURES = [
    "CR",
    "F",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
    "iid",
    "seed",
]

CATEGORICAL_FEATURES = [
    "crossover",
    "mutation_base",
    "mutation_reference",
]

TARGET_COL = "auc"

RENAME_MAP = {
    "iid": "Instance variance",
    "seed": "Stochastic variance",
}

CATEGORY_MAPS = {
    "crossover": {
        "bin": 0,
        "exp": 1,
    },
    "mutation_base": {
        "best": 0,
        "rand": 1,
        "target": 2,
    },
    "mutation_reference": {
        "best": 0,
        "nan": 1,
        "pbest": 2,
        "rand": 3,
    },
}


def prepare_features(df: pd.DataFrame):
    """
    Niels-style feature preparation:
    - categorical encoding
    - no one-hot
    - include iid/seed as variance features
    - drop constant columns
    """
    df = df.copy()

    required_cols = RAW_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[RAW_FEATURES].copy()
    X = X.rename(columns=RENAME_MAP)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
        X[col] = X[col].map(CATEGORY_MAPS[col])

        if X[col].isna().any():
            unknown_values = sorted(
                df[col].astype(str)[X[col].isna()].unique().tolist()
            )
            raise ValueError(
                f"Unknown category in {col}: {unknown_values}. "
                f"Please update CATEGORY_MAPS."
            )

        X[col] = X[col].astype(int)

    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    ordered_cols = [c for c in FEATURE_ORDER if c in X.columns]
    X = X[ordered_cols]

    y = df[TARGET_COL].copy()

    return X, y


def run_shap(df: pd.DataFrame, group_id: int):
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling from {len(df)} -> {SAMPLE_SIZE}")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    X, y = prepare_features(df)

    print(f"\nTraining CatBoost model for Group {group_id}...")
    print("X shape:", X.shape)
    print("Features:", list(X.columns))

    model = cb.CatBoostRegressor(
        iterations=100,
        depth=14,
        random_seed=42,
        verbose=False,
    )

    model.fit(X, y)

    train_r2 = model.score(X, y)
    print(f"Train R2: {train_r2:.4f}")

    print("Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance_values = np.abs(shap_values).mean(axis=0)
    total_importance = importance_values.sum()

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance_values,
    })

    importance["importance_norm"] = (
        importance["importance"] / total_importance
        if total_importance > 0
        else 0
    )

    importance["group_id"] = group_id
    importance["train_r2"] = train_r2
    importance["n_samples"] = len(X)

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    return importance


def discover_group_files(group_dir: Path):
    files = sorted(group_dir.glob("de_final_5_group_*.pkl"))
    pairs = []

    for f in files:
        try:
            group_id = int(f.stem.split("_")[-1])
            pairs.append((group_id, f))
        except ValueError:
            continue

    return pairs


def main():
    if not GROUP_DATA_DIR.exists():
        raise FileNotFoundError(f"Group directory not found: {GROUP_DATA_DIR}")

    group_files = discover_group_files(GROUP_DATA_DIR)

    if not group_files:
        raise FileNotFoundError(f"No group pkl files found in: {GROUP_DATA_DIR}")

    print(f"Found {len(group_files)} group files.")
    print(group_files)

    all_results = []

    for group_id, file in group_files:
        print(f"\nLoading group {group_id}: {file}")

        df = pd.read_pickle(file)

        print("Shape:", df.shape)

        if "group_label" in df.columns:
            labels = df["group_label"].dropna().unique().tolist()
            print(f"Group label(s): {labels}")

        importance = run_shap(df, group_id)

        print(f"\n=== Niels-style SHAP importance Group {group_id} ===")
        print(importance.head(20))

        out_file = OUTPUT_DIR / f"shap_group_{group_id}.csv"
        importance.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        all_results.append(importance)

    all_df = pd.concat(all_results, ignore_index=True)

    all_out = OUTPUT_DIR / "shap_all_groups.csv"
    all_df.to_csv(all_out, index=False)

    print("\nSaved combined SHAP results.")
    print("Combined file:", all_out)


if __name__ == "__main__":
    main()