from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# 路径配置
# =========================
GROUP_SHAP_DIR = "output/manual_bins_shap"
INDIVIDUAL_SHAP_FILE = "output/shap_individual_manual_bins/shap_all_functions.csv"
OUTPUT_DIR = "output/group_vs_individual_consistency"

TOP_K = 5


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def discover_group_shap_files(group_dir: Path):
    files = sorted(group_dir.glob("shap_group_*.csv"))
    pairs = []
    for f in files:
        try:
            group_id = int(f.stem.split("_")[-1])
            pairs.append((group_id, f))
        except ValueError:
            continue
    return pairs


def load_group_level_shap(group_dir: Path) -> pd.DataFrame:
    pairs = discover_group_shap_files(group_dir)
    if not pairs:
        raise FileNotFoundError(f"No shap_group_*.csv found in {group_dir}")

    dfs = []
    for group_id, file in pairs:
        df = pd.read_csv(file)
        required = ["feature", "importance"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file} missing columns: {missing}")

        df = df.copy()
        df["group_id"] = group_id
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def load_individual_shap(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required = ["feature", "importance", "fid", "group_id", "group_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} missing columns: {missing}")
    return df


def compute_groupwise_individual_mean(ind_df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 group 内的所有 individual functions，
    计算 feature importance 的平均值、标准差、函数数
    """
    out = (
        ind_df.groupby(["group_id", "group_label", "feature"], as_index=False)
        .agg(
            individual_mean_importance=("importance", "mean"),
            individual_std_importance=("importance", "std"),
            n_functions=("fid", "nunique"),
        )
        .copy()
    )
    out["individual_std_importance"] = out["individual_std_importance"].fillna(0.0)
    return out


def rank_within_group(df: pd.DataFrame, value_col: str, rank_col: str) -> pd.DataFrame:
    df = df.copy()
    df[rank_col] = (
        df.groupby("group_id")[value_col]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return df


def compute_spearman_from_ranks(sub: pd.DataFrame) -> float:
    """
    用 rank 之间的 Pearson 相关作为 Spearman。
    """
    if len(sub) < 2:
        return np.nan

    r1 = sub["group_rank"].astype(float)
    r2 = sub["individual_rank"].astype(float)

    if r1.nunique() < 2 or r2.nunique() < 2:
        return np.nan

    return float(r1.corr(r2, method="pearson"))


def topk_overlap(sub: pd.DataFrame, k: int = 5):
    g_top = set(
        sub.sort_values("group_importance", ascending=False)
        .head(k)["feature"]
        .tolist()
    )
    i_top = set(
        sub.sort_values("individual_mean_importance", ascending=False)
        .head(k)["feature"]
        .tolist()
    )
    overlap = len(g_top & i_top)
    union = len(g_top | i_top) if len(g_top | i_top) > 0 else 0
    jaccard = overlap / union if union > 0 else np.nan
    return overlap, jaccard, sorted(g_top), sorted(i_top)


def main():
    project_root = Path(__file__).resolve().parent.parent
    group_dir = project_root / GROUP_SHAP_DIR
    ind_file = project_root / INDIVIDUAL_SHAP_FILE
    output_dir = project_root / OUTPUT_DIR
    ensure_dir(output_dir)

    print("Loading group-level SHAP...")
    group_df = load_group_level_shap(group_dir)
    print("Group-level shape:", group_df.shape)

    print("Loading individual-level SHAP...")
    ind_df = load_individual_shap(ind_file)
    print("Individual-level shape:", ind_df.shape)

    print("Computing group-wise individual mean SHAP...")
    ind_mean_df = compute_groupwise_individual_mean(ind_df)
    print("Group-wise individual mean shape:", ind_mean_df.shape)

    # group_label 从 individual mean 中取
    group_labels = (
        ind_mean_df[["group_id", "group_label"]]
        .drop_duplicates()
        .sort_values("group_id")
    )

    # 合并 group-level 与 group内 individual 平均
    merged = group_df.merge(
        ind_mean_df,
        on=["group_id", "feature"],
        how="left"
    )

    merged = merged.merge(group_labels, on="group_id", how="left", suffixes=("", "_from_map"))

    # 清理 group_label
    if "group_label_from_map" in merged.columns:
        merged["group_label"] = merged["group_label"].fillna(merged["group_label_from_map"])
        merged = merged.drop(columns=["group_label_from_map"])

    merged = merged.rename(columns={"importance": "group_importance"})

    # 组内排名
    merged = rank_within_group(merged, "group_importance", "group_rank")
    merged = rank_within_group(merged, "individual_mean_importance", "individual_rank")

    comparison_file = output_dir / "group_vs_individual_feature_comparison.csv"
    merged.to_csv(comparison_file, index=False)
    print("Saved:", comparison_file)

    # 每个 group 的一致性统计
    summary_rows = []

    for gid, sub in merged.groupby("group_id"):
        group_label = sub["group_label"].dropna().iloc[0] if sub["group_label"].notna().any() else "unknown"

        spearman = compute_spearman_from_ranks(sub)
        overlap_k, jaccard_k, group_topk, ind_topk = topk_overlap(sub, k=TOP_K)

        row = {
            "group_id": gid,
            "group_label": group_label,
            "n_features": int(len(sub)),
            "n_functions_in_group": int(sub["n_functions"].dropna().max()) if sub["n_functions"].notna().any() else np.nan,
            "spearman_rank_corr": spearman,
            f"top_{TOP_K}_overlap_count": overlap_k,
            f"top_{TOP_K}_jaccard": jaccard_k,
            "group_top_features": " | ".join(group_topk),
            "individual_mean_top_features": " | ".join(ind_topk),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("group_id")
    summary_file = output_dir / "group_vs_individual_consistency_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print("Saved:", summary_file)

    # 再输出一个“全局平均 individual importance”表，方便画总图
    global_individual = (
        ind_df.groupby("feature", as_index=False)
        .agg(
            global_individual_mean_importance=("importance", "mean"),
            global_individual_std_importance=("importance", "std"),
            n_functions=("fid", "nunique"),
        )
        .sort_values("global_individual_mean_importance", ascending=False)
    )
    global_individual["global_individual_std_importance"] = global_individual["global_individual_std_importance"].fillna(0.0)

    global_file = output_dir / "global_individual_importance_summary.csv"
    global_individual.to_csv(global_file, index=False)
    print("Saved:", global_file)

    # 输出简单文字总结
    readme_lines = []
    readme_lines.append("Group vs Individual SHAP Consistency Summary")
    readme_lines.append("=" * 60)
    readme_lines.append("")
    readme_lines.append(f"Top-k used for overlap: {TOP_K}")
    readme_lines.append("")

    readme_lines.append("Per-group consistency:")
    readme_lines.append(summary_df.to_string(index=False))
    readme_lines.append("")
    readme_lines.append("Global individual-level mean importance:")
    readme_lines.append(global_individual.to_string(index=False))

    readme_file = output_dir / "README_group_vs_individual.txt"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
    print("Saved:", readme_file)

    print("\nDone.")
    print("Main outputs:")
    print(comparison_file)
    print(summary_file)
    print(global_file)


if __name__ == "__main__":
    main()