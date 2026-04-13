import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import shap

# ========= 路径 =========
DATA_FILE = "data/de_final_5_processed.pkl"
GROUP_MAP_FILE = "output/manual_binning/function_group_mapping_for_step3.csv"

OUTPUT_DIR = "output/shap_individual_manual_bins"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = 10000  # 控制计算量


def prepare_features(sub: pd.DataFrame):
    """
    和 Step4 保持一致：
    - 数值列直接保留
    - 类别列做 one-hot
    """
    numeric_features = [
        "CR",
        "F",
        "lambda_",
        "lpsr",
        "mutation_n_comps",
        "use_archive",
    ]

    categorical_features = [
        "mutation_base",
        "mutation_reference",
        "crossover",
        "adaptation_method",
    ]

    target_col = "auc"

    required_cols = numeric_features + categorical_features + [target_col]
    missing = [c for c in required_cols if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X_num = sub[numeric_features].copy()
    X_cat = pd.get_dummies(
        sub[categorical_features].copy(),
        columns=categorical_features,
        drop_first=False
    )

    X = pd.concat([X_num, X_cat], axis=1)
    y = sub[target_col].copy()

    return X, y


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

    # 保险起见，只保留 d=5
    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    # merge manual bins group
    group_map = group_map.rename(columns={"Function": "fid"})
    group_map["fid"] = pd.to_numeric(group_map["fid"], errors="coerce").astype("Int64")
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype("Int64")

    df = df.merge(
        group_map[["fid", "group_id", "group_label"]],
        on="fid",
        how="left"
    )

    print("Merged data shape:", df.shape)
    print("Missing group labels:", df["group_id"].isna().sum())

    functions = sorted(df["fid"].dropna().unique().tolist())
    print("Total functions:", len(functions))

    all_results = []

    for fid in functions:
        print(f"\n=== Processing function {fid} ===")

        sub = df[df["fid"] == fid].copy()

        if len(sub) > SAMPLE_SIZE:
            print(f"Sampling from {len(sub)} -> {SAMPLE_SIZE}")
            sub = sub.sample(SAMPLE_SIZE, random_state=42)

        group_id = sub["group_id"].dropna().iloc[0] if "group_id" in sub.columns and sub["group_id"].notna().any() else np.nan
        group_label = sub["group_label"].dropna().iloc[0] if "group_label" in sub.columns and sub["group_label"].notna().any() else np.nan

        X, y = prepare_features(sub)

        print("X shape:", X.shape)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        importance = np.abs(shap_values).mean(axis=0)

        result = pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        })

        result["fid"] = fid
        result["group_id"] = group_id
        result["group_label"] = group_label

        # 保存单个 function
        out_file = os.path.join(OUTPUT_DIR, f"shap_function_{fid}.csv")
        result.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        all_results.append(result)

    # 合并所有 function
    all_df = pd.concat(all_results, ignore_index=True)
    all_out = os.path.join(OUTPUT_DIR, "shap_all_functions.csv")
    all_df.to_csv(all_out, index=False)

    print("\nSaved all function SHAP results.")
    print("Combined file:", all_out)


if __name__ == "__main__":
    main()