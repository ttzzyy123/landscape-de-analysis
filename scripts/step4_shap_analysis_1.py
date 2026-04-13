from pathlib import Path
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def run_shap(df, group_id):
    # ===== sampling =====
    max_samples = 20000
    if len(df) > max_samples:
        print(f"Sampling from {len(df)} → {max_samples}")
        df = df.sample(n=max_samples, random_state=42)

    # ===== 特征列（参数）=====
    features = [
        "CR", "F", "lambda_", "lpsr",
        "mutation_n_comps", "use_archive"
    ]

    # ===== categorical 编码 =====
    df = df.copy()
    df["adaptation_method"] = df["adaptation_method"].astype("category").cat.codes
    df["crossover"] = df["crossover"].astype("category").cat.codes
    df["mutation_base"] = df["mutation_base"].astype("category").cat.codes
    df["mutation_reference"] = df["mutation_reference"].astype("category").cat.codes

    features += [
        "adaptation_method",
        "crossover",
        "mutation_base",
        "mutation_reference"
    ]

    X = df[features]
    y = df["auc"]

    print(f"\nTraining model for Group {group_id}...")

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    print("Running SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        "feature": features,
        "importance": abs(shap_values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    return importance


def main():
    # 自动定位项目根目录：scripts 的上一层
    project_root = Path(__file__).resolve().parent.parent

    group_dir = project_root / "intermediate" / "group_data"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    for g in [0, 1, 2]:
        file = group_dir / f"de_final_5_group_{g}.pkl"

        print(f"\nLoading group {g}: {file}")
        df = pd.read_pickle(file)

        print(f"Shape: {df.shape}")

        importance = run_shap(df, g)

        print(f"\n=== SHAP importance (Group {g}) ===")
        print(importance)

        out_file = output_dir / f"shap_group_{g}.csv"
        importance.to_csv(out_file, index=False)

        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()