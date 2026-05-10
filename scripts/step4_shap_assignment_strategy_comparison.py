from pathlib import Path

import numpy as np
import pandas as pd
import shap
import catboost as cb


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUP_DATA_BASE_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "group_data_assignment_strategy_comparison"
)

OUTPUT_BASE_DIR = (
    PROJECT_ROOT
    / "output"
    / "shap_assignment_strategy_comparison_niles"
)

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20000


# =========================
# Configs to compare
# =========================
CONFIGS = [
    "eps2bins_3_r025065",
    "eps2bins_3_r0307_main",
]

STRATEGIES = [
    "ambiguous_on_tie",
    "majority_with_mean_tiebreak",
    "mean_based",
]


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


def run_shap(df: pd.DataFrame, group_id: int, config_name: str, strategy_name: str):
    if len(df) > SAMPLE_SIZE:
        print(f"Sampling from {len(df)} -> {SAMPLE_SIZE}")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    X, y = prepare_features(df)

    print(f"\nTraining CatBoost model")
    print(f"Config: {config_name}")
    print(f"Strategy: {strategy_name}")
    print(f"Group: {group_id}")
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

    importance["config"] = config_name
    importance["strategy"] = strategy_name
    importance["group_id"] = group_id
    importance["train_r2"] = train_r2
    importance["n_samples"] = len(X)

    if "group_label" in df.columns:
        labels = df["group_label"].dropna().unique().tolist()
        importance["group_label"] = (
            labels[0]
            if len(labels) == 1
            else "|".join(map(str, labels))
        )

    if "fid" in df.columns:
        fids = sorted(df["fid"].dropna().unique().tolist())
        importance["functions"] = ",".join([f"f{int(fid)}" for fid in fids])
        importance["n_functions"] = len(fids)

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


def run_one_strategy(config_name: str, strategy_name: str):
    group_data_dir = GROUP_DATA_BASE_DIR / config_name / strategy_name
    output_dir = OUTPUT_BASE_DIR / config_name / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not group_data_dir.exists():
        raise FileNotFoundError(f"Group directory not found: {group_data_dir}")

    group_files = discover_group_files(group_data_dir)

    if not group_files:
        raise FileNotFoundError(f"No group pkl files found in: {group_data_dir}")

    print("\n" + "=" * 90)
    print(f"Running SHAP for config: {config_name}")
    print(f"Running SHAP for strategy: {strategy_name}")
    print(f"Input group data: {group_data_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 90)

    all_results = []

    for group_id, file in group_files:
        print(f"\nLoading group {group_id}: {file}")

        df = pd.read_pickle(file)

        print("Shape:", df.shape)

        if "group_label" in df.columns:
            labels = df["group_label"].dropna().unique().tolist()
            print(f"Group label(s): {labels}")

        if "fid" in df.columns:
            fids = sorted(df["fid"].dropna().unique().tolist())
            print("Functions:", [f"f{int(fid)}" for fid in fids])

        importance = run_shap(
            df=df,
            group_id=group_id,
            config_name=config_name,
            strategy_name=strategy_name,
        )

        print(f"\n=== SHAP importance ===")
        print(f"Config: {config_name}")
        print(f"Strategy: {strategy_name}")
        print(f"Group: {group_id}")
        print(importance.head(20))

        out_file = output_dir / f"shap_group_{group_id}.csv"
        importance.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        all_results.append(importance)

    all_df = pd.concat(all_results, ignore_index=True)

    all_out = output_dir / "shap_all_groups.csv"
    all_df.to_csv(all_out, index=False)

    print("\nSaved combined SHAP result for one strategy.")
    print("Combined file:", all_out)

    return all_df


def main():
    all_strategy_results = []

    for config_name in CONFIGS:
        for strategy_name in STRATEGIES:
            result = run_one_strategy(
                config_name=config_name,
                strategy_name=strategy_name,
            )
            all_strategy_results.append(result)

    combined_all = pd.concat(all_strategy_results, ignore_index=True)

    combined_all_file = OUTPUT_BASE_DIR / "shap_all_configs_all_strategies.csv"
    combined_all.to_csv(combined_all_file, index=False)

    print("\n" + "=" * 90)
    print("DONE: all 6 SHAP runs finished.")
    print("=" * 90)

    print("\nCombined all SHAP results saved to:")
    print(combined_all_file)

    print("\nAll outputs saved under:")
    print(OUTPUT_BASE_DIR)


if __name__ == "__main__":
    main()