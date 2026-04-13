import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import shap

# ========= 路径 =========
DATA_FILE = "data/de_final_5_processed.pkl"
GROUP_MAP_FILE = "output/grouping_sensitivity/assignments_kmeans_k3.csv"

OUTPUT_DIR = "output/shap_individual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 10000  # 控制计算量


def main():
    print("Loading data...")
    df = pd.read_pickle(DATA_FILE)
    group_map = pd.read_csv(GROUP_MAP_FILE)

    print("Data shape:", df.shape)
    print("Group map shape:", group_map.shape)

    # 映射 group
    df = df.merge(group_map, left_on="fid", right_on="Function")

    # 找特征列（超参数）
    feature_cols = [
        "CR", "F", "lambda_", "lpsr",
        "mutation_base", "mutation_n_comps",
        "mutation_reference", "crossover",
        "use_archive", "adaptation_method"
    ]

    target_col = "auc"

    functions = sorted(df["fid"].unique())
    print("Total functions:", len(functions))

    all_results = []

    for fid in functions:
        print(f"\n=== Processing function {fid} ===")

        sub = df[df["fid"] == fid].copy()

        if len(sub) > SAMPLE_SIZE:
            sub = sub.sample(SAMPLE_SIZE, random_state=42)

        X = sub[feature_cols].copy()
        y = sub[target_col]

        # 自动 one-hot encoding
        X = pd.get_dummies(X)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        importance = np.abs(shap_values).mean(axis=0)

        result = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        })

        result["fid"] = fid

        # 保存单个 function
        out_file = os.path.join(OUTPUT_DIR, f"shap_function_{fid}.csv")
        result.to_csv(out_file, index=False)

        all_results.append(result)

    # 合并所有 function
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(os.path.join(OUTPUT_DIR, "shap_all_functions.csv"), index=False)

    print("\nSaved all function SHAP results.")


if __name__ == "__main__":
    main()