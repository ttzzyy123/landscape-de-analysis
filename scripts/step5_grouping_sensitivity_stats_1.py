import os
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

# =========================
# 1. 路径配置
# =========================
FEATURE_FILE = "data/features_summary_dim_5_sobol.csv"
OUTPUT_DIR = "/data/s3795888/ioh_project/my_landscape_experiments/output/grouping_sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 固定实验设置
# =========================
TARGET_SAMPLES = 5000   # 1000d for d=5
TARGET_FEATURES = ["ic.eps.ratio", "ela_meta.lin_simple.adj_r2"]

# KMeans 对照设置
K_LIST = [2, 3, 4, 5, 6]

# Quantile bin 对照设置
# 2x2, 3x3, 4x4 ...
BIN_LIST = [2, 3, 4]

RANDOM_STATE = 42
N_INIT = 20


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名首尾空格和可能的 BOM 字符"""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_and_prepare_function_level_features() -> pd.DataFrame:
    """
    从 Zenodo 的 long-format feature file 生成 function-level 特征表：
    Function | ic.eps.ratio | ela_meta.lin_simple.adj_r2
    """
    print("Loading feature file...")
    df = pd.read_csv(FEATURE_FILE, sep=";")
    df = clean_columns(df)

    print("Detected columns:")
    print(df.columns.tolist())

    required_cols = ["Function", "Instance", "# samples", "Feature name", "Value"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Raw shape:", df.shape)

    # 只保留指定 budget
    df = df[df["# samples"] == TARGET_SAMPLES].copy()
    print(f"After filtering # samples == {TARGET_SAMPLES}: {df.shape}")

    # 只保留目标 feature
    df = df[df["Feature name"].isin(TARGET_FEATURES)].copy()
    print(f"After filtering target features {TARGET_FEATURES}: {df.shape}")

    # 检查两个特征是否都存在
    existing_features = sorted(df["Feature name"].unique().tolist())
    print("Existing target features:", existing_features)

    missing = [f for f in TARGET_FEATURES if f not in existing_features]
    if missing:
        raise ValueError(f"Missing target features in filtered data: {missing}")

    # 转成宽表：Function, Instance, feature1, feature2
    wide = df.pivot_table(
        index=["Function", "Instance"],
        columns="Feature name",
        values="Value",
        aggfunc="mean"
    ).reset_index()

    wide.columns.name = None
    print("Wide shape (Function-Instance level):", wide.shape)

    # 去除缺失
    wide = wide.dropna(subset=TARGET_FEATURES).copy()
    print("After dropping NA in target features:", wide.shape)

    # 聚合到 Function level
    func_df = (
        wide.groupby("Function", as_index=False)[TARGET_FEATURES]
        .mean()
        .copy()
    )

    print("Function-level feature shape:", func_df.shape)
    print(func_df.head())

    out_path = os.path.join(OUTPUT_DIR, "function_level_features.csv")
    func_df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return func_df


def summarize_group_sizes(assign_df: pd.DataFrame, label_col: str) -> dict:
    """
    统计组大小分布
    """
    counts = assign_df[label_col].value_counts().sort_index()
    summary = {
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
    return summary


def run_kmeans_experiments(func_df: pd.DataFrame):
    """
    对不同 k 做 KMeans，对比组大小和聚类质量
    """
    print("\n=== Running KMeans experiments ===")

    X = func_df[TARGET_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_assignments = []
    all_summaries = []

    for k in K_LIST:
        print(f"Running KMeans k={k} ...")
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )
        labels = model.fit_predict(X_scaled)

        assign_df = func_df[["Function"]].copy()
        assign_df["group"] = labels
        assign_df["method"] = "kmeans"
        assign_df["config"] = f"kmeans_k{k}"

        # 组大小统计
        size_summary = summarize_group_sizes(assign_df, "group")

        # 聚类质量指标
        # silhouette 至少需要 2 个 cluster 且每簇非空（这里 KMeans 满足）
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)

        row = {
            "method": "kmeans",
            "config": f"kmeans_k{k}",
            "k_or_bins": k,
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
            **size_summary,
        }
        all_summaries.append(row)

        # 保存 assignment
        assign_out = os.path.join(OUTPUT_DIR, f"assignments_kmeans_k{k}.csv")
        assign_df.to_csv(assign_out, index=False)

        # 保存每组均值
        profile_df = assign_df.merge(func_df, on="Function", how="left")
        group_profile = (
            profile_df.groupby("group", as_index=False)[TARGET_FEATURES]
            .mean()
            .sort_values("group")
        )
        profile_out = os.path.join(OUTPUT_DIR, f"group_profile_kmeans_k{k}.csv")
        group_profile.to_csv(profile_out, index=False)

        # 保存组大小
        size_df = (
            assign_df["group"]
            .value_counts()
            .sort_index()
            .rename_axis("group")
            .reset_index(name="n_functions")
        )
        size_out = os.path.join(OUTPUT_DIR, f"group_sizes_kmeans_k{k}.csv")
        size_df.to_csv(size_out, index=False)

        all_assignments.append(assign_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_out = os.path.join(OUTPUT_DIR, "summary_kmeans.csv")
    summary_df.to_csv(summary_out, index=False)
    print("Saved:", summary_out)

    return all_assignments, summary_df


def run_quantile_bin_experiments(func_df: pd.DataFrame):
    """
    用二维 quantile bins 对比不同 bin size
    """
    print("\n=== Running quantile-bin experiments ===")

    all_assignments = []
    all_summaries = []

    base = func_df.copy()

    f1, f2 = TARGET_FEATURES

    for b in BIN_LIST:
        print(f"Running quantile bins {b}x{b} ...")
        df = base.copy()

        # qcut 可能因为边界重复导致 bins 数减少，所以 duplicates='drop'
        df["bin1"] = pd.qcut(df[f1], q=b, labels=False, duplicates="drop")
        df["bin2"] = pd.qcut(df[f2], q=b, labels=False, duplicates="drop")

        df = df.dropna(subset=["bin1", "bin2"]).copy()
        df["bin1"] = df["bin1"].astype(int)
        df["bin2"] = df["bin2"].astype(int)

        df["group_label"] = "g_" + df["bin1"].astype(str) + "_" + df["bin2"].astype(str)

        # 转连续 group id
        unique_labels = sorted(df["group_label"].unique())
        label_map = {lab: i for i, lab in enumerate(unique_labels)}
        df["group"] = df["group_label"].map(label_map)

        assign_df = df[["Function", "group"]].copy()
        assign_df["method"] = "quantile_bin"
        assign_df["config"] = f"qbin_{b}x{b}"

        size_summary = summarize_group_sizes(assign_df, "group")

        # bins 没有 silhouette 这种“优化目标”，这里只统计组数量和大小
        row = {
            "method": "quantile_bin",
            "config": f"qbin_{b}x{b}",
            "k_or_bins": b,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            **size_summary,
        }
        all_summaries.append(row)

        assign_out = os.path.join(OUTPUT_DIR, f"assignments_qbin_{b}x{b}.csv")
        assign_df.to_csv(assign_out, index=False)

        group_profile = (
            df.groupby(["group", "group_label"], as_index=False)[TARGET_FEATURES]
            .mean()
            .sort_values("group")
        )
        profile_out = os.path.join(OUTPUT_DIR, f"group_profile_qbin_{b}x{b}.csv")
        group_profile.to_csv(profile_out, index=False)

        size_df = (
            assign_df["group"]
            .value_counts()
            .sort_index()
            .rename_axis("group")
            .reset_index(name="n_functions")
        )
        size_out = os.path.join(OUTPUT_DIR, f"group_sizes_qbin_{b}x{b}.csv")
        size_df.to_csv(size_out, index=False)

        all_assignments.append(assign_df)

    summary_df = pd.DataFrame(all_summaries)
    summary_out = os.path.join(OUTPUT_DIR, "summary_quantile_bins.csv")
    summary_df.to_csv(summary_out, index=False)
    print("Saved:", summary_out)

    return all_assignments, summary_df


def compute_pairwise_ari(all_assignments):
    """
    比较不同 grouping config 之间的一致性
    用 Adjusted Rand Index (ARI)
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
    out_path = os.path.join(OUTPUT_DIR, "pairwise_ari.csv")
    ari_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    return ari_df


def build_overall_summary(kmeans_summary, qbin_summary):
    """
    汇总成一个总表，便于你 thesis 里直接看
    """
    overall = pd.concat([kmeans_summary, qbin_summary], ignore_index=True)

    # 添加一个简单建议标记：
    # 组太小（<=2）的组太多，通常不适合后续解释
    def flag_row(row):
        if row["small_groups_le_2"] >= max(1, row["n_groups"] // 2):
            return "too_many_tiny_groups"
        if row["max_group_size"] >= 0.6 * row["n_functions"]:
            return "one_group_too_large"
        return "reasonable"

    overall["grouping_flag"] = overall.apply(flag_row, axis=1)

    out_path = os.path.join(OUTPUT_DIR, "summary_overall_grouping_sensitivity.csv")
    overall.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return overall


def write_readme(overall_df, ari_df):
    """
    输出一份简单文字总结
    """
    lines = []
    lines.append("Grouping sensitivity experiment summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Feature file: {FEATURE_FILE}")
    lines.append(f"Target samples: {TARGET_SAMPLES}")
    lines.append(f"Target features: {TARGET_FEATURES}")
    lines.append("")

    lines.append("Overall grouping summary:")
    lines.append(overall_df.to_string(index=False))
    lines.append("")

    if len(ari_df) > 0:
        lines.append("Top pairwise ARI (higher = more similar grouping):")
        lines.append(ari_df.sort_values("ARI", ascending=False).head(10).to_string(index=False))
        lines.append("")

    out_path = os.path.join(OUTPUT_DIR, "README_grouping_sensitivity.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Saved:", out_path)


def main():
    func_df = load_and_prepare_function_level_features()

    kmeans_assignments, kmeans_summary = run_kmeans_experiments(func_df)
    qbin_assignments, qbin_summary = run_quantile_bin_experiments(func_df)

    all_assignments = kmeans_assignments + qbin_assignments
    ari_df = compute_pairwise_ari(all_assignments)

    overall_df = build_overall_summary(kmeans_summary, qbin_summary)
    write_readme(overall_df, ari_df)

    print("\nDone.")
    print("Main outputs:")
    print(os.path.join(OUTPUT_DIR, "summary_kmeans.csv"))
    print(os.path.join(OUTPUT_DIR, "summary_quantile_bins.csv"))
    print(os.path.join(OUTPUT_DIR, "summary_overall_grouping_sensitivity.csv"))
    print(os.path.join(OUTPUT_DIR, "pairwise_ari.csv"))


if __name__ == "__main__":
    main()