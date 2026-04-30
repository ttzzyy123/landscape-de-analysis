import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_1 = "eps_ratio"
FEATURE_2 = "adj_r2"

DEFAULT_SMALL_BIN_WIDTHS = {
    FEATURE_1: 0.05,
    FEATURE_2: 0.05,
}

BIN_CONFIGS = {
    # =====================================================
    # Current main candidate: 2 x 3
    # =====================================================
    "eps2bins_3_r0307_main": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },

    # nearby alternatives around main
    "eps2bins_3_r025065": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.25, 0.65, np.inf],
    },
    "eps2bins_3_r0206": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.6, np.inf],
    },
    "eps2bins_3_r04075": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.4, 0.75, np.inf],
    },

    # eps boundary sensitivity
    "eps2bins_35_r0307": {
        "eps_ratio": [-np.inf, 3.5, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },
    "eps2bins_25_r0307": {
        "eps_ratio": [-np.inf, 2.5, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },

    # =====================================================
    # Coarser 2 x 2 baselines
    # =====================================================
    "eps2bins_3_r05": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.5, np.inf],
    },
    "eps2bins_3_r04": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.4, np.inf],
    },
    "eps2bins_3_r06": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.6, np.inf],
    },

    # =====================================================
    # eps_ratio as 3 bins, motivated by visual distribution
    # =====================================================
    "eps3bins_3_55_r0307": {
        "eps_ratio": [-np.inf, 3.0, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },
    "eps3bins_25_55_r0307": {
        "eps_ratio": [-np.inf, 2.5, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },
    "eps3bins_3_6_r0307": {
        "eps_ratio": [-np.inf, 3.0, 6.0, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },

    # eps 3 bins, adj 2 bins
    "eps3bins_3_55_r05": {
        "eps_ratio": [-np.inf, 3.0, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.5, np.inf],
    },
    "eps3bins_25_55_r05": {
        "eps_ratio": [-np.inf, 2.5, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.5, np.inf],
    },

    # eps 3 bins, adj 3 bins
    "eps3bins_3_55_r025065": {
        "eps_ratio": [-np.inf, 3.0, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.25, 0.65, np.inf],
    },
    "eps3bins_25_55_r0206": {
        "eps_ratio": [-np.inf, 2.5, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.6, np.inf],
    },

    # =====================================================
    # adj_r2 as 4 bins, to test finer adj partition
    # =====================================================
    "eps2bins_3_r4bins_0205075": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.5, 0.75, np.inf],
    },
    "eps2bins_3_r4bins_0306075": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.6, 0.75, np.inf],
    },

    # =====================================================
    # Very fine control: likely over-partitioning
    # =====================================================
    "eps3bins_3_55_r4bins_0205075": {
        "eps_ratio": [-np.inf, 3.0, 5.5, np.inf],
        "adj_r2": [-np.inf, 0.2, 0.5, 0.75, np.inf],
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}\n当前列名: {df.columns.tolist()}")


def load_feature_data(csv_path):
    df = pd.read_csv(csv_path)

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


def trim_tails_within_function(df, features, q):
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

    return pd.concat(kept_parts, axis=0).sort_index()


def assign_bin_label(values, edges, prefix):
    labels = [f"{prefix}_bin_{i}" for i in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)


def majority_vote(series):
    cnt = Counter(series.dropna().astype(str))
    if not cnt:
        return np.nan
    max_count = max(cnt.values())
    winners = sorted([k for k, v in cnt.items() if v == max_count])
    return winners[0]


def compute_majority_assignments(df, manual_bins):
    out = df.copy()

    out["bin_feat1"] = assign_bin_label(out[FEATURE_1], manual_bins[FEATURE_1], "eps")
    out["bin_feat2"] = assign_bin_label(out[FEATURE_2], manual_bins[FEATURE_2], "r2")

    rows = []
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

        feat1_dist = sub["bin_feat1"].astype(str).value_counts(normalize=True, dropna=False).to_dict()
        feat2_dist = sub["bin_feat2"].astype(str).value_counts(normalize=True, dropna=False).to_dict()

        for k, v in feat1_dist.items():
            row[f"feat1_prop__{k}"] = v
        for k, v in feat2_dist.items():
            row[f"feat2_prop__{k}"] = v

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("Function").reset_index(drop=True)

    unique_groups = sorted(summary["group_label"].dropna().unique())
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    summary["group_id"] = summary["group_label"].map(group_to_id)

    return out, summary


def summarize_grouping(function_assignments, config_name, manual_bins):
    counts = function_assignments["group_label"].value_counts()

    summary = {
        "config": config_name,
        "eps_edges": str(manual_bins["eps_ratio"]),
        "adj_r2_edges": str(manual_bins["adj_r2"]),
        "n_eps_bins": len(manual_bins["eps_ratio"]) - 1,
        "n_r2_bins": len(manual_bins["adj_r2"]) - 1,
        "n_functions": int(function_assignments["Function"].nunique()),
        "n_groups": int(counts.shape[0]),
        "min_group_size": int(counts.min()),
        "max_group_size": int(counts.max()),
        "mean_group_size": float(counts.mean()),
        "std_group_size": float(counts.std(ddof=0)),
        "singleton_groups": int((counts == 1).sum()),
        "small_groups_le_2": int((counts <= 2).sum()),
        "small_groups_le_3": int((counts <= 3).sum()),
    }

    if summary["singleton_groups"] > 0:
        summary["flag"] = "has_singletons"
    elif summary["max_group_size"] >= 0.5 * summary["n_functions"]:
        summary["flag"] = "one_group_too_large"
    elif summary["small_groups_le_2"] >= max(1, summary["n_groups"] // 2):
        summary["flag"] = "too_many_small_groups"
    else:
        summary["flag"] = "reasonable"

    return summary


def plot_2d_scatter_with_groups(df, function_assignments, output_dir, config_name, manual_bins):
    func_means = (
        df.groupby("Function")[[FEATURE_1, FEATURE_2]]
        .mean()
        .reset_index()
        .merge(function_assignments[["Function", "group_id", "group_label"]], on="Function", how="left")
    )

    plt.figure(figsize=(8, 6))

    for gid, sub in func_means.groupby("group_id"):
        plt.scatter(sub[FEATURE_1], sub[FEATURE_2], label=f"G{gid}", s=60)

        for _, row in sub.iterrows():
            plt.text(
                row[FEATURE_1] + 0.04,
                row[FEATURE_2] + 0.008,
                f"f{int(row['Function'])}",
                fontsize=8
            )

    for e in manual_bins["eps_ratio"]:
        if np.isfinite(e):
            plt.axvline(e, color="black", linestyle="--", linewidth=1.2)

    for e in manual_bins["adj_r2"]:
        if np.isfinite(e):
            plt.axhline(e, color="black", linestyle="--", linewidth=1.2)

    plt.xlabel(FEATURE_1)
    plt.ylabel(FEATURE_2)
    plt.title(f"Manual bin grouping: {config_name}")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{config_name}.png"), dpi=300)
    plt.close()


def plot_group_sizes(function_assignments, output_dir, config_name):
    counts = (
        function_assignments["group_label"]
        .value_counts()
        .rename_axis("group_label")
        .reset_index(name="n_functions")
    )

    plt.figure(figsize=(9, 4))
    plt.bar(range(len(counts)), counts["n_functions"])
    plt.xticks(range(len(counts)), counts["group_label"], rotation=45, ha="right")
    plt.ylabel("Number of functions")
    plt.title(f"Group sizes: {config_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"group_sizes_{config_name}.png"), dpi=300)
    plt.close()


def run_one_config(df_used, config_name, manual_bins, base_output_dir):
    config_dir = os.path.join(base_output_dir, config_name)
    ensure_dir(config_dir)

    instance_level_with_bins, function_assignments = compute_majority_assignments(
        df_used,
        manual_bins=manual_bins
    )

    df_used.to_csv(os.path.join(config_dir, "step2_feature_data_used.csv"), index=False)

    instance_level_with_bins.to_csv(
        os.path.join(config_dir, "instance_level_manual_bins.csv"),
        index=False
    )

    function_assignments.to_csv(
        os.path.join(config_dir, "function_to_manual_groups.csv"),
        index=False
    )

    function_assignments[["Function", "group_id", "group_label"]].to_csv(
        os.path.join(config_dir, "function_group_mapping_for_step3.csv"),
        index=False
    )

    plot_2d_scatter_with_groups(df_used, function_assignments, config_dir, config_name, manual_bins)
    plot_group_sizes(function_assignments, config_dir, config_name)

    summary = summarize_grouping(function_assignments, config_name, manual_bins)

    group_sizes = (
        function_assignments["group_label"]
        .value_counts()
        .rename_axis("group_label")
        .reset_index(name="n_functions")
    )
    group_sizes.to_csv(os.path.join(config_dir, "group_sizes.csv"), index=False)

    print(f"\n===== {config_name} =====")
    print(group_sizes)
    print(summary)

    return summary


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
        default="output/manual_binning_experiments",
        help="多方案 Step2 输出目录"
    )
    parser.add_argument(
        "--drop_tails",
        type=float,
        default=0.0,
        help="每个 function 内部裁掉两端 quantile，比如 0.05"
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("Loading feature data...")
    df = load_feature_data(args.input_csv)
    print(f"Loaded shape: {df.shape}")
    print(df.head())

    df_used = trim_tails_within_function(
        df,
        features=[FEATURE_1, FEATURE_2],
        q=args.drop_tails
    )
    print(f"After tail trimming: {df_used.shape}")

    summaries = []

    for config_name, manual_bins in BIN_CONFIGS.items():
        summary = run_one_config(
            df_used=df_used,
            config_name=config_name,
            manual_bins=manual_bins,
            base_output_dir=args.output_dir
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_file = os.path.join(args.output_dir, "summary_all_bin_configs.csv")
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Overall summary =====")
    print(summary_df)
    print(f"\nSaved summary to: {summary_file}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()