import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score


# =========================
# 1. 路径配置
# =========================
FEATURE_FILE = "intermediate/dim5_selected_features.csv"
OUTPUT_DIR = "/data/s3795888/ioh_project/my_landscape_experiments/output/manual_grouping_sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 特征列配置
# =========================
FEATURE_1 = "eps_ratio"
FEATURE_2 = "adj_r2"

# =========================
# 3. Manual bin configurations
# 这里是你要比较的“人工阈值方案”
# =========================
MANUAL_CONFIGS = {
    # 你当前最终采用的主方案
    "manual_2x3_main": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },

    # 更保守一点的 r2 分法
    "manual_2x3_alt1": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.6, np.inf],
    },

    # 之前试过的更细 eps 分法
    "manual_3x3_alt2": {
        "eps_ratio": [-np.inf, 2.5, 5.0, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.6, np.inf],
    },

    # 更粗的 2x2 方案
    "manual_2x2_alt3": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.5, np.inf],
    },
}


def load_feature_data() -> pd.DataFrame:
    """
    读取 step1 输出的 Function-Instance level feature 表
    期望列:
    Function | Instance | eps_ratio | adj_r2
    """
    print("Loading feature data...")
    df = pd.read_csv(FEATURE_FILE)

    required_cols = ["Function", "Instance", FEATURE_1, FEATURE_2]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Function"] = pd.to_numeric(df["Function"], errors="coerce").astype("Int64")
    df["Instance"] = pd.to_numeric(df["Instance"], errors="coerce").astype("Int64")
    df[FEATURE_1] = pd.to_numeric(df[FEATURE_1], errors="coerce")
    df[FEATURE_2] = pd.to_numeric(df[FEATURE_2], errors="coerce")

    df = df.dropna(subset=["Function", "Instance", FEATURE_1, FEATURE_2]).copy()
    df["Function"] = df["Function"].astype(int)
    df["Instance"] = df["Instance"].astype(int)

    print("Loaded shape:", df.shape)
    print(df.head())

    # 顺手保存 function-level mean，后面画图/写文档也方便
    func_mean = (
        df.groupby("Function", as_index=False)[[FEATURE_1, FEATURE_2]]
        .mean()
        .copy()
    )
    func_mean.to_csv(os.path.join(OUTPUT_DIR, "function_level_features.csv"), index=False)
    print("Saved:", os.path.join(OUTPUT_DIR, "function_level_features.csv"))

    return df


def assign_bin_label(values: pd.Series, edges, prefix: str):
    labels = [f"{prefix}_bin_{i}" for i in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)


def majority_vote(series: pd.Series):
    vc = series.dropna().astype(str).value_counts()
    if len(vc) == 0:
        return np.nan
    max_count = vc.max()
    winners = sorted(vc[vc == max_count].index.tolist())
    return winners[0]


def summarize_group_sizes(assign_df: pd.DataFrame, label_col: str) -> dict:
    counts = assign_df[label_col].value_counts().sort_index()
    return {
        "n_functions": int(len(assign_df)),
        "n_groups": int(counts.shape[0]),
        "min_group_size": int(counts.min()),
        "max_group_size": int(counts.max()),
        "mean_group_size": float(counts.mean()),
        "std_group_size": float(counts.std(ddof=0)),
        "singleton_groups": int((counts == 1).sum()),
        "small_groups_le_2": int((counts <= 2).sum()),
        "small_groups_le_3": int((counts <= 3).sum()),
    }


def run_manual_config(df: pd.DataFrame, config_name: str, bin_config: dict):
    """
    对一套 manual bins:
    - 在 instance level 上 assign bins
    - 对每个 function 做 majority vote
    - 生成 Function -> group 映射
    """
    work = df.copy()

    eps_edges = bin_config[FEATURE_1]
    r2_edges = bin_config[FEATURE_2]

    work["bin_feat1"] = assign_bin_label(work[FEATURE_1], eps_edges, "eps")
    work["bin_feat2"] = assign_bin_label(work[FEATURE_2], r2_edges, "r2")

    rows = []
    for fid, sub in work.groupby("Function"):
        maj_bin_1 = majority_vote(sub["bin_feat1"])
        maj_bin_2 = majority_vote(sub["bin_feat2"])
        group_label = f"{maj_bin_1}__{maj_bin_2}"

        row = {
            "Function": fid,
            "majority_bin_feat1": maj_bin_1,
            "majority_bin_feat2": maj_bin_2,
            "group_label": group_label,
            "config": config_name,
        }

        # 记录比例，方便判断干净程度
        feat1_dist = sub["bin_feat1"].astype(str).value_counts(normalize=True, dropna=False).to_dict()
        feat2_dist = sub["bin_feat2"].astype(str).value_counts(normalize=True, dropna=False).to_dict()

        for k, v in feat1_dist.items():
            row[f"feat1_prop__{k}"] = v
        for k, v in feat2_dist.items():
            row[f"feat2_prop__{k}"] = v

        rows.append(row)

    assign_df = pd.DataFrame(rows).sort_values("Function").reset_index(drop=True)

    # group label -> group id
    unique_groups = sorted(assign_df["group_label"].dropna().unique())
    label_map = {lab: i for i, lab in enumerate(unique_groups)}
    assign_df["group"] = assign_df["group_label"].map(label_map)

    return assign_df


def build_group_profiles(assign_df: pd.DataFrame, df: pd.DataFrame):
    """
    输出每个 group 的 function-level 均值 profile
    """
    func_mean = (
        df.groupby("Function", as_index=False)[[FEATURE_1, FEATURE_2]]
        .mean()
        .copy()
    )

    merged = assign_df[["Function", "group", "group_label"]].merge(
        func_mean,
        on="Function",
        how="left"
    )

    profile = (
        merged.groupby(["group", "group_label"], as_index=False)[[FEATURE_1, FEATURE_2]]
        .mean()
        .sort_values("group")
    )

    return profile


def run_manual_grouping_experiments(df: pd.DataFrame):
    print("\n=== Running manual bin grouping experiments ===")

    all_assignments = []
    summary_rows = []

    for config_name, bin_config in MANUAL_CONFIGS.items():
        print(f"\nRunning config: {config_name}")
        print(json.dumps(bin_config, indent=2, default=str))

        assign_df = run_manual_config(df, config_name, bin_config)

        # 保存 assignment
        assign_out = os.path.join(OUTPUT_DIR, f"assignments_{config_name}.csv")
        assign_df.to_csv(assign_out, index=False)

        # 保存 group size
        size_df = (
            assign_df["group"]
            .value_counts()
            .sort_index()
            .rename_axis("group")
            .reset_index(name="n_functions")
        )
        size_out = os.path.join(OUTPUT_DIR, f"group_sizes_{config_name}.csv")
        size_df.to_csv(size_out, index=False)

        # 保存 group profile
        profile_df = build_group_profiles(assign_df, df)
        profile_out = os.path.join(OUTPUT_DIR, f"group_profile_{config_name}.csv")
        profile_df.to_csv(profile_out, index=False)

        # 汇总
        size_summary = summarize_group_sizes(assign_df, "group")

        row = {
            "method": "manual_bins",
            "config": config_name,
            "eps_edges": json.dumps(bin_config[FEATURE_1]),
            "adj_r2_edges": json.dumps(bin_config[FEATURE_2]),
            "n_eps_bins": len(bin_config[FEATURE_1]) - 1,
            "n_r2_bins": len(bin_config[FEATURE_2]) - 1,
            **size_summary,
        }
        summary_rows.append(row)

        all_assignments.append(assign_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_out = os.path.join(OUTPUT_DIR, "summary_manual_bins.csv")
    summary_df.to_csv(summary_out, index=False)
    print("\nSaved:", summary_out)

    return all_assignments, summary_df


def compute_pairwise_ari(all_assignments):
    """
    比较不同 manual bin config 之间的一致性
    """
    print("\n=== Computing pairwise ARI ===")

    rows = []

    for i in range(len(all_assignments)):
        for j in range(i + 1, len(all_assignments)):
            a = all_assignments[i].copy()
            b = all_assignments[j].copy()

            config_a = a["config"].iloc[0]
            config_b = b["config"].iloc[0]

            merged = a[["Function", "group"]].merge(
                b[["Function", "group"]],
                on="Function",
                suffixes=("_a", "_b"),
                how="inner"
            )

            ari = adjusted_rand_score(merged["group_a"], merged["group_b"])

            rows.append({
                "config_a": config_a,
                "config_b": config_b,
                "n_common_functions": int(len(merged)),
                "ARI": float(ari)
            })

    ari_df = pd.DataFrame(rows).sort_values(["config_a", "config_b"])
    out_path = os.path.join(OUTPUT_DIR, "pairwise_ari_manual_bins.csv")
    ari_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return ari_df


def build_overall_summary(summary_df):
    """
    标记哪种 manual config 更合理
    """
    overall = summary_df.copy()

    def flag_row(row):
        if row["singleton_groups"] > 0:
            return "has_singletons"
        if row["small_groups_le_2"] >= max(1, row["n_groups"] // 2):
            return "too_many_tiny_groups"
        if row["max_group_size"] >= 0.6 * row["n_functions"]:
            return "one_group_too_large"
        return "reasonable"

    overall["grouping_flag"] = overall.apply(flag_row, axis=1)

    out_path = os.path.join(OUTPUT_DIR, "summary_overall_manual_grouping_sensitivity.csv")
    overall.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return overall


def write_readme(overall_df, ari_df):
    lines = []
    lines.append("Manual bin grouping sensitivity experiment summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Feature file: {FEATURE_FILE}")
    lines.append(f"Features used: {FEATURE_1}, {FEATURE_2}")
    lines.append("")

    lines.append("Overall grouping summary:")
    lines.append(overall_df.to_string(index=False))
    lines.append("")

    if len(ari_df) > 0:
        lines.append("Top pairwise ARI (higher = more similar grouping):")
        lines.append(ari_df.sort_values("ARI", ascending=False).head(10).to_string(index=False))
        lines.append("")

    out_path = os.path.join(OUTPUT_DIR, "README_manual_grouping_sensitivity.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:", out_path)


def main():
    df = load_feature_data()

    all_assignments, summary_df = run_manual_grouping_experiments(df)
    ari_df = compute_pairwise_ari(all_assignments)
    overall_df = build_overall_summary(summary_df)
    write_readme(overall_df, ari_df)

    print("\nDone.")
    print("Main outputs:")
    print(os.path.join(OUTPUT_DIR, "summary_manual_bins.csv"))
    print(os.path.join(OUTPUT_DIR, "summary_overall_manual_grouping_sensitivity.csv"))
    print(os.path.join(OUTPUT_DIR, "pairwise_ari_manual_bins.csv"))


if __name__ == "__main__":
    main()