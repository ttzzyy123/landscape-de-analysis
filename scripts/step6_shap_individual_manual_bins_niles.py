from pathlib import Path

import numpy as np
import pandas as pd
import shap
import catboost as cb


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"
GROUP_MAP_FILE = PROJECT_ROOT / "output" / "manual_binning" / "function_group_mapping_for_step3.csv"

OUTPUT_DIR = PROJECT_ROOT / "output" / "shap_individual_manual_bins_niles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 10000


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


def prepare_features(sub: pd.DataFrame):
    """
    Niels-style individual SHAP feature preparation:
    - CatBoostRegressor
    - categorical encoding, no one-hot
    - include iid/seed as variance features
    - fixed feature order
    """
    sub = sub.copy()

    required_cols = RAW_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = sub[RAW_FEATURES].copy()
    X = X.rename(columns=RENAME_MAP)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
        X[col] = X[col].map(CATEGORY_MAPS[col])

        if X[col].isna().any():
            unknown_values = sorted(
                sub[col].astype(str)[X[col].isna()].unique().tolist()
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

    y = sub[TARGET_COL].copy()

    return X, y


def run_function_shap(sub: pd.DataFrame, fid: int):
    if len(sub) > SAMPLE_SIZE:
        print(f"Sampling from {len(sub)} -> {SAMPLE_SIZE}")
        sub = sub.sample(SAMPLE_SIZE, random_state=42)

    group_id = (
        sub["group_id"].dropna().iloc[0]
        if "group_id" in sub.columns and sub["group_id"].notna().any()
        else np.nan
    )

    group_label = (
        sub["group_label"].dropna().iloc[0]
        if "group_label" in sub.columns and sub["group_label"].notna().any()
        else np.nan
    )

    X, y = prepare_features(sub)

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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = np.abs(shap_values).mean(axis=0)
    total_importance = importance.sum()

    result = pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
    })

    result["importance_norm"] = (
        result["importance"] / total_importance
        if total_importance > 0
        else 0
    )

    result["fid"] = fid
    result["group_id"] = group_id
    result["group_label"] = group_label
    result["train_r2"] = train_r2
    result["n_samples"] = len(X)

    result = result.sort_values(
        by="importance",
        ascending=False,
    )

    return result


def main():
    print("Loading data...")
    df = pd.read_pickle(DATA_FILE)
    group_map = pd.read_csv(GROUP_MAP_FILE)

    print("Data shape:", df.shape)
    print("Group map shape:", group_map.shape)

    required_group_cols = ["Function", "group_id", "group_label"]
    missing = [c for c in required_group_cols if c not in group_map.columns]
    if missing:
        raise ValueError(f"Missing required columns in group mapping: {missing}")

    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    group_map = group_map.rename(columns={"Function": "fid"})

    group_map["fid"] = pd.to_numeric(
        group_map["fid"],
        errors="coerce",
    ).astype("Int64")

    df["fid"] = pd.to_numeric(
        df["fid"],
        errors="coerce",
    ).astype("Int64")

    df = df.merge(
        group_map[["fid", "group_id", "group_label"]],
        on="fid",
        how="left",
    )

    print("Merged data shape:", df.shape)
    print("Missing group labels:", df["group_id"].isna().sum())

    functions = sorted(df["fid"].dropna().unique().tolist())
    print("Total functions:", len(functions))

    all_results = []

    for fid in functions:
        print(f"\n=== Processing function f{fid} ===")

        sub = df[df["fid"] == fid].copy()

        result = run_function_shap(sub, fid)

        out_file = OUTPUT_DIR / f"shap_function_{fid}.csv"
        result.to_csv(out_file, index=False)

        print(f"Saved: {out_file}")
        print(result.head(10))

        all_results.append(result)

    all_df = pd.concat(all_results, ignore_index=True)

    all_out = OUTPUT_DIR / "shap_all_functions.csv"
    all_df.to_csv(all_out, index=False)

    print("\nSaved all function SHAP results.")
    print("Combined file:", all_out)


if __name__ == "__main__":
    main()