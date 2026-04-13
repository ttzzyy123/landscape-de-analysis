import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Config
# =========================================================

FEATURE_1 = "eps_ratio"
FEATURE_2 = "adj_r2"

DEFAULT_SMALL_BIN_WIDTHS = {
    FEATURE_1: 0.05,
    FEATURE_2: 0.05,
}

DEFAULT_MANUAL_BINS = {
    "eps_ratio": [-np.inf, 3.0,  np.inf],
    "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
}

DEFAULT_DROP_TAILS = 0.0  # 例如 0.05 表示去掉每个 function 内部的 5% 和 95% 尾部


# =========================================================
# Utils
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def check_required_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}\n当前列名: {df.columns.tolist()}")


def load_feature_data(csv_path: str) -> pd.DataFrame:
    """
    兼容两种列名：
    1. 原始长列名:
       - ic.eps.ratio
       - ela_meta.lin_simple.adj_r2
    2. 简化列名:
       - eps_ratio
       - adj_r2
    最终统一重命名为:
       - eps_ratio
       - adj_r2
    """
    df = pd.read_csv(csv_path)

    # 自动兼容列名
    rename_map = {}
    if "ic.eps.ratio" in df.columns:
        rename_map["ic.eps.ratio"] = "eps_ratio"
    if "ela_meta.lin_simple.adj_r2" in df.columns:
        rename_map["ela_meta.lin_simple.adj_r2"] = "adj_r2"

    df = df.rename(columns=rename_map)

    required = ["Function", "Instance", "eps_ratio", "adj_r2"]
    check_required_columns(df, required)

    df = df[required].copy()

    df["Function"] = pd.to_numeric(df["Function"], errors="coerce").astype("Int64")
    df["Instance"] = pd.to_numeric(df["Instance"], errors="coerce").astype("Int64")
    df["eps_ratio"] = pd.to_numeric(df["eps_ratio"], errors="coerce")
    df["adj_r2"] = pd.to_numeric(df["adj_r2"], errors="coerce")

    df = df.dropna(subset=["Function", "Instance", "eps_ratio", "adj_r2"]).copy()
    df["Function"] = df["Function"].astype(int)
    df["Instance"] = df["Instance"].astype(int)

    return df

def trim_tails_within_function(df: pd.DataFrame, features, q: float) -> pd.DataFrame:
    """
    对每个 function 内部的 feature 分布做尾部裁剪。
    q=0 时不裁剪。
    """
    if q <= 0:
        return df.copy()

    kept_parts = []
    for fid, sub in df.groupby("Function"):
        mask = pd.Series(True, index=sub.index)
        for feat in features:
            lower = sub[feat].quantile(q)
            upper = sub[feat].quantile(1 - q)
            mask &= sub[feat].between(lower, upper, inclusive="both")
        kept_parts.append(sub.loc[mask])

    out = pd.concat(kept_parts, axis=0).sort_index()
    return out


def plot_overall_histograms(df: pd.DataFrame, output_dir: str):
    for feat in [FEATURE_1, FEATURE_2]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[feat].dropna(), bins=30)
        plt.xlabel(feat)
        plt.ylabel("Count")
        plt.title(f"Overall distribution of {feat}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"overall_hist_{feat}.png"), dpi=200)
        plt.close()


def plot_stacked_histograms(df: pd.DataFrame, output_dir: str, bins_per_feature=None):
    """
    导师提到的 stacked histogram:
    每个 function 一种颜色
    """
    if bins_per_feature is None:
        bins_per_feature = {}

    functions = sorted(df["Function"].unique())

    for feat in [FEATURE_1, FEATURE_2]:
        vals_per_func = [df.loc[df["Function"] == fid, feat].dropna().values for fid in functions]

        # 自动 bins
        if feat in bins_per_feature and bins_per_feature[feat] is not None:
            bins = bins_per_feature[feat]
        else:
            all_vals = df[feat].dropna().values
            bins = np.linspace(all_vals.min(), all_vals.max(), 25)

        plt.figure(figsize=(10, 6))
        plt.hist(vals_per_func, bins=bins, stacked=True, label=[f"f{fid}" for fid in functions])
        plt.xlabel(feat)
        plt.ylabel("Count")
        plt.title(f"Stacked histogram of {feat} by function")
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            ncol=1,
            borderaxespad=0
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"stacked_hist_{feat}.png"), dpi=220)
        plt.close()


def build_small_bins(series: pd.Series, bin_width: float):
    min_val = float(series.min())
    max_val = float(series.max())

    if min_val == max_val:
        return np.array([min_val - 1e-9, max_val + 1e-9])

    start = np.floor(min_val / bin_width) * bin_width
    end = np.ceil(max_val / bin_width) * bin_width + bin_width
    bins = np.arange(start, end + 1e-12, bin_width)
    return bins


def function_coverage_in_small_bins(df: pd.DataFrame, feature: str, bin_width: float) -> pd.DataFrame:
    """
    小 bin 统计:
    对每个小区间，统计有多少个 function 的 feature 值落入其中
    """
    bins = build_small_bins(df[feature], bin_width)
    temp = df[["Function", feature]].copy()
    temp["small_bin"] = pd.cut(temp[feature], bins=bins, include_lowest=True, right=False)

    coverage = (
        temp.groupby("small_bin", observed=False)["Function"]
        .nunique()
        .reset_index(name="n_functions")
    )

    # 显式转成 float，避免 categorical 类型后面无法做减法
    coverage["bin_left"] = coverage["small_bin"].apply(
        lambda x: float(x.left) if pd.notna(x) else np.nan
    )
    coverage["bin_right"] = coverage["small_bin"].apply(
        lambda x: float(x.right) if pd.notna(x) else np.nan
    )

    coverage["bin_left"] = pd.to_numeric(coverage["bin_left"], errors="coerce")
    coverage["bin_right"] = pd.to_numeric(coverage["bin_right"], errors="coerce")

    coverage["feature"] = feature
    coverage = coverage[["feature", "small_bin", "bin_left", "bin_right", "n_functions"]]
    return coverage

def plot_function_coverage(coverage_df: pd.DataFrame, feature: str, output_dir: str):
    coverage_df = coverage_df.copy()

    coverage_df["bin_left"] = pd.to_numeric(coverage_df["bin_left"], errors="coerce")
    coverage_df["bin_right"] = pd.to_numeric(coverage_df["bin_right"], errors="coerce")
    coverage_df["n_functions"] = pd.to_numeric(coverage_df["n_functions"], errors="coerce")

    coverage_df = coverage_df.dropna(subset=["bin_left", "bin_right", "n_functions"]).copy()

    plt.figure(figsize=(10, 5))
    x = coverage_df["bin_left"].values
    y = coverage_df["n_functions"].values

    if len(coverage_df) > 0:
        width = float((coverage_df["bin_right"] - coverage_df["bin_left"]).iloc[0])
    else:
        width = 0.1

    plt.bar(x, y, width=width, align="edge")
    plt.xlabel(feature)
    plt.ylabel("Number of functions covered")
    plt.title(f"Function coverage in small bins: {feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"small_bin_function_coverage_{feature}.png"), dpi=220)
    plt.close()


def assign_bin_label(values: pd.Series, edges, prefix: str):
    """
    把一组数值分到人工定义的大 bins 中。
    返回每个样本的标签，例如:
    eps_bin_0, eps_bin_1, eps_bin_2
    """
    labels = [f"{prefix}_bin_{i}" for i in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)


def majority_vote(series: pd.Series):
    """
    多数表决；如果并列，取排序后第一个，保证稳定。
    """
    cnt = Counter(series.dropna().astype(str))
    if not cnt:
        return np.nan
    max_count = max(cnt.values())
    winners = sorted([k for k, v in cnt.items() if v == max_count])
    return winners[0]


def compute_majority_assignments(df: pd.DataFrame, manual_bins: dict) -> pd.DataFrame:
    """
    对每个 function:
    - 分别对两个 feature 做 majority vote
    - 再组合成 2D group
    """
    out = df.copy()

    out["bin_feat1"] = assign_bin_label(out[FEATURE_1], manual_bins[FEATURE_1], "eps")
    out["bin_feat2"] = assign_bin_label(out[FEATURE_2], manual_bins[FEATURE_2], "r2")

    # 如果有样本掉在 bins 外面，pd.cut 会是 NaN
    # 这里先保留，后面看函数层面多数投票结果
    summary_rows = []
    for fid, sub in out.groupby("Function"):
        maj_bin_1 = majority_vote(sub["bin_feat1"])
        maj_bin_2 = majority_vote(sub["bin_feat2"])

        group_label = f"{maj_bin_1}__{maj_bin_2}"

        row = {
            "Function": fid,
            "majority_bin_feat1": maj_bin_1,
            "majority_bin_feat2": maj_bin_2,
            "group_label": group_label,
            "n_instances": len(sub),
        }

        # 记录比例，方便你判断这个函数分组是否“干净”
        feat1_dist = sub["bin_feat1"].astype(str).value_counts(normalize=True, dropna=False).to_dict()
        feat2_dist = sub["bin_feat2"].astype(str).value_counts(normalize=True, dropna=False).to_dict()

        for k, v in feat1_dist.items():
            row[f"feat1_prop__{k}"] = v
        for k, v in feat2_dist.items():
            row[f"feat2_prop__{k}"] = v

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("Function").reset_index(drop=True)

    # 给 group 一个整数 id，方便 Step3 merge 使用
    unique_groups = sorted(summary["group_label"].dropna().unique())
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    summary["group_id"] = summary["group_label"].map(group_to_id)

    return out, summary


def plot_2d_scatter_with_groups(df: pd.DataFrame, function_assignments: pd.DataFrame, output_dir: str):
    """
    画二维 feature space 散点图。
    这里先用 function 的均值位置做图，颜色表示最终分到的 group。
    """
    func_means = (
        df.groupby("Function")[[FEATURE_1, FEATURE_2]]
        .mean()
        .reset_index()
        .merge(function_assignments[["Function", "group_id", "group_label"]], on="Function", how="left")
    )

    plt.figure(figsize=(8, 6))
    for gid, sub in func_means.groupby("group_id"):
        plt.scatter(sub[FEATURE_1], sub[FEATURE_2], label=f"group {gid}", s=50)

        for _, row in sub.iterrows():
            plt.text(row[FEATURE_1], row[FEATURE_2], f"f{int(row['Function'])}", fontsize=8)

    plt.xlabel(FEATURE_1)
    plt.ylabel(FEATURE_2)
    plt.title("2D feature space with manual-bin groups")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "manual_bin_2d_scatter.png"), dpi=220)
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="intermediate/dim5_selected_features.csv",
        help="Step1 输出的 feature CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/manual_binning",
        help="Step2 输出目录"
    )
    parser.add_argument(
        "--drop_tails",
        type=float,
        default=DEFAULT_DROP_TAILS,
        help="每个 function 内部裁掉两端 quantile，比如 0.05"
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("Loading feature data...")
    df = load_feature_data(args.input_csv)
    print(f"Loaded shape: {df.shape}")
    print(df.head())

    # 可选：裁尾
    df_used = trim_tails_within_function(
        df,
        features=[FEATURE_1, FEATURE_2],
        q=args.drop_tails
    )
    print(f"After tail trimming: {df_used.shape}")

    # 保存当前实际用于 step2 的数据
    df_used.to_csv(os.path.join(args.output_dir, "step2_feature_data_used.csv"), index=False)

    # 1) 总体分布图
    print("Plotting overall histograms...")
    plot_overall_histograms(df_used, args.output_dir)

    # 2) stacked histogram
    print("Plotting stacked histograms...")
    plot_stacked_histograms(df_used, args.output_dir)

    # 3) 小 bin 覆盖统计
    print("Computing small-bin function coverage...")
    coverage_all = []
    for feat in [FEATURE_1, FEATURE_2]:
        bw = DEFAULT_SMALL_BIN_WIDTHS[feat]
        cov = function_coverage_in_small_bins(df_used, feat, bw)
        coverage_all.append(cov)
        plot_function_coverage(cov, feat, args.output_dir)

    coverage_all = pd.concat(coverage_all, axis=0, ignore_index=True)
    coverage_all.to_csv(os.path.join(args.output_dir, "small_bin_function_coverage.csv"), index=False)

    # 4) 人工 bins + majority vote
    print("Assigning functions to manual bins...")
    instance_level_with_bins, function_assignments = compute_majority_assignments(
        df_used,
        manual_bins=DEFAULT_MANUAL_BINS
    )

    instance_level_with_bins.to_csv(
        os.path.join(args.output_dir, "instance_level_manual_bins.csv"),
        index=False
    )

    function_assignments.to_csv(
        os.path.join(args.output_dir, "function_to_manual_groups.csv"),
        index=False
    )

    # 单独导出一个简洁版 mapping，方便后面 Step3 直接 merge
    function_assignments[["Function", "group_id", "group_label"]].to_csv(
        os.path.join(args.output_dir, "function_group_mapping_for_step3.csv"),
        index=False
    )

    # 5) 2D scatter
    print("Plotting 2D scatter...")
    plot_2d_scatter_with_groups(df_used, function_assignments, args.output_dir)

    # 6) 控制台摘要
    print("\n===== Group summary =====")
    print(function_assignments[["Function", "majority_bin_feat1", "majority_bin_feat2", "group_id", "group_label"]])

    print("\n===== Group sizes =====")
    print(function_assignments["group_label"].value_counts(dropna=False))

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()