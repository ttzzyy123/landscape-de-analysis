from pathlib import Path
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def prepare_features(df: pd.DataFrame):
    """
    准备 SHAP 输入特征：
    - 数值列直接保留
    - 类别列使用 categorical encoding，不使用 one-hot
    - 加入 iid 和 seed，分别表示 Instance variance 和 Stochastic variance
    - 删除常量列
    - 目标变量使用 auc
    """
    df = df.copy()

    numeric_features = [
        "CR",
        "F",
        "lambda_",
        "lpsr",
        "mutation_n_comps",
        "use_archive",
    ]

    categorical_features = [
        "adaptation_method",
        "crossover",
        "mutation_base",
        "mutation_reference",
    ]

    variance_features = [
        "iid",
        "seed",
    ]

    required_cols = numeric_features + categorical_features + variance_features + ["auc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[numeric_features + categorical_features + variance_features].copy()

    # 和 Niels 的图保持一致
    X = X.rename(
        columns={
            "iid": "Instance variance",
            "seed": "Stochastic variance",
        }
    )

    # categorical encoding，不做 one-hot
    for col in categorical_features:
        X[col] = pd.Categorical(X[col]).codes

    # 删除常量列，例如 adaptation_method 如果全是 nan/none
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    y = df["auc"].copy()

    return X, y


def run_shap(df: pd.DataFrame, group_id: int):
    max_samples = 20000
    if len(df) > max_samples:
        print(f"Sampling from {len(df)} -> {max_samples}")
        df = df.sample(n=max_samples, random_state=42)

    X, y = prepare_features(df)

    print(f"\nTraining model for Group {group_id}...")
    print(f"X shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    print("Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": abs(shap_values).mean(axis=0),
    }).sort_values(by="importance", ascending=False)

    return importance


def discover_group_files(group_dir: Path):
    files = sorted(group_dir.glob("de_final_5_group_*.pkl"))
    pairs = []

    for f in files:
        stem = f.stem
        try:
            group_id = int(stem.split("_")[-1])
            pairs.append((group_id, f))
        except ValueError:
            continue

    return pairs


def main():
    project_root = Path(__file__).resolve().parent.parent

    group_dir = project_root / "intermediate" / "group_data_eps2bins_3_r025065"
    output_dir = project_root / "output" / "shap_eps2bins_3_r025065_encoding"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not group_dir.exists():
        raise FileNotFoundError(f"Group directory not found: {group_dir}")

    group_files = discover_group_files(group_dir)
    if not group_files:
        raise FileNotFoundError(f"No group pkl files found in: {group_dir}")

    print(f"Found {len(group_files)} group files.")
    print(group_files)

    all_results = []

    for g, file in group_files:
        print(f"\nLoading group {g}: {file}")
        df = pd.read_pickle(file)

        print(f"Shape: {df.shape}")
        if "group_label" in df.columns:
            labels = df["group_label"].dropna().unique().tolist()
            print(f"Group label(s): {labels}")

        importance = run_shap(df, g)

        print(f"\n=== SHAP importance encoding version (Group {g}) ===")
        print(importance.head(20))

        out_file = output_dir / f"shap_group_{g}.csv"
        importance.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        temp = importance.copy()
        temp["group_id"] = g
        all_results.append(temp)

    if all_results:
        df_all = pd.concat(all_results, axis=0, ignore_index=True)
        all_out = output_dir / "shap_all_groups.csv"
        df_all.to_csv(all_out, index=False)
        print(f"\nSaved combined SHAP results to: {all_out}")


if __name__ == "__main__":
    main()