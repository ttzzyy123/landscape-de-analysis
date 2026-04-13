from pathlib import Path
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def run_shap(df, label):
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
    df["mutation_base"] = df["mutation_base"].astype("category").cat.codes
    df["mutation_reference"] = df["mutation_reference"].astype("category").cat.codes

    # 注意：
    # 这里不再把 crossover 放进特征，
    # 因为现在 bin 和 exp 已经分开了，
    # 在各自子集中 crossover 是常量，没有分析意义。
    features += [
        "adaptation_method",
        "mutation_base",
        "mutation_reference"
    ]

    X = df[features]
    y = df["auc"]

    print(f"\nTraining model for {label}...")
    print(f"X shape: {X.shape}")

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

    modde_dir = project_root / "intermediate" / "modde_split"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # 读取 k=3 的 function -> group 映射
    group_map_file = project_root / "output" / "grouping_sensitivity" / "assignments_kmeans_k3.csv"
    group_map = pd.read_csv(group_map_file)[["Function", "group"]].drop_duplicates()

    # 两个数据集：bin 和 exp
    dataset_files = {
        "bin": modde_dir / "de_final_5_bin.pkl",
        "exp": modde_dir / "de_final_5_exp.pkl",
    }

    for mode, file in dataset_files.items():
        print("\n" + "=" * 60)
        print(f"Processing dataset: {mode}")
        print("=" * 60)

        print(f"Loading: {file}")
        df = pd.read_pickle(file)
        print(f"Original shape: {df.shape}")

        # merge group
        df = df.merge(group_map, left_on="fid", right_on="Function", how="left")

        if df["group"].isna().any():
            missing = df["group"].isna().sum()
            raise ValueError(f"{mode}: {missing} rows have no group assignment.")

        df["group"] = df["group"].astype(int)

        print("After merge group shape:", df.shape)
        print("Group counts:")
        print(df["group"].value_counts().sort_index())

        for g in [0, 1, 2]:
            sub = df[df["group"] == g].copy()

            print(f"\nLoading {mode} group {g}")
            print(f"Shape: {sub.shape}")

            if len(sub) == 0:
                print(f"Skip {mode} group {g}: empty dataset.")
                continue

            importance = run_shap(sub, f"{mode} group {g}")

            print(f"\n=== SHAP importance ({mode} group {g}) ===")
            print(importance)

            out_file = output_dir / f"shap_{mode}_group_{g}.csv"
            importance.to_csv(out_file, index=False)

            print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()