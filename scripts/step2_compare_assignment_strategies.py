import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_1 = "eps_ratio"
FEATURE_2 = "adj_r2"

BIN_CONFIGS = {
    "eps2bins_3_r025065": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.25, 0.65, np.inf],
    },
    "eps2bins_3_r0307_main": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.3, 0.7, np.inf],
    },
}

ASSIGNMENT_STRATEGIES = [
    "ambiguous_on_tie",
    "majority_with_mean_tiebreak",
    "mean_based",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nCurrent columns: {df.columns.tolist()}")


def load_feature_data(csv_path):
    df = pd.read_csv(csv_path)

    rename_map = {}
    if "ic.eps.ratio" in df.columns:
        rename_map["ic.eps.ratio"] = "eps_ratio"
    if "ela_meta.lin_simple.adj_r2" in df.columns:
        rename_map["ela_meta.lin_simple.adj_r2"] = "adj_r2"

    df = df.rename(columns=rename_map)

    required = ["Function", "Instance", FEATURE_1, FEATURE_2]
    check_required_columns(df, required)

    df = df[required].copy()
    df["Function"] = pd.to_numeric(df["Function"], errors="coerce").astype("Int64")
    df["Instance"] = pd.to_numeric(df["Instance"], errors="coerce").astype("Int64")
    df[FEATURE_1] = pd.to_numeric(df[FEATURE_1], errors="coerce")
    df[FEATURE_2] = pd.to_numeric(df[FEATURE_2], errors="coerce")

    df = df.dropna(subset=["Function", "Instance", FEATURE_1, FEATURE_2]).copy()
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


def assign_single_value_to_bin(value, edges, prefix):
    labels = [f"{prefix}_bin_{i}" for i in range(len(edges) - 1)]
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return labels[i]
    return np.nan


def majority_vote_with_status(series):
    cnt = Counter(series.dropna().astype(str))
    if not cnt:
        return np.nan, "empty", {}

    max_count = max(cnt.values())
    winners = sorted([k for k, v in cnt.items() if v == max_count])

    if len(winners) > 1:
        return winners, "tie", dict(cnt)

    return winners[0], "clear", dict(cnt)


def compute_assignments(df, manual_bins, strategy):
    out = df.copy()

    out["bin_feat1"] = assign_bin_label(out[FEATURE_1], manual_bins[FEATURE_1], "eps")
    out["bin_feat2"] = assign_bin_label(out[FEATURE_2], manual_bins[FEATURE_2], "r2")

    rows = []

    for fid, sub in out.groupby("Function"):
        mean_feat1 = sub[FEATURE_1].mean()
        mean_feat2 = sub[FEATURE_2].mean()

        mean_bin_1 = assign_single_value_to_bin(mean_feat1, manual_bins[FEATURE_1], "eps")
        mean_bin_2 = assign_single_value_to_bin(mean_feat2, manual_bins[FEATURE_2], "r2")

        maj_1, status_1, cnt_1 = majority_vote_with_status(sub["bin_feat1"])
        maj_2, status_2, cnt_2 = majority_vote_with_status(sub["bin_feat2"])

        is_ambiguous = False

        if strategy == "ambiguous_on_tie":
            if status_1 == "tie":
                final_bin_1 = "ambiguous"
                is_ambiguous = True
            else:
                final_bin_1 = maj_1

            if status_2 == "tie":
                final_bin_2 = "ambiguous"
                is_ambiguous = True
            else:
                final_bin_2 = maj_2

        elif strategy == "majority_with_mean_tiebreak":
            if status_1 == "tie":
                final_bin_1 = mean_bin_1
            else:
                final_bin_1 = maj_1

            if status_2 == "tie":
                final_bin_2 = mean_bin_2
            else:
                final_bin_2 = maj_2

        elif strategy == "mean_based":
            final_bin_1 = mean_bin_1
            final_bin_2 = mean_bin_2

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        group_label = f"{final_bin_1}__{final_bin_2}"

        row = {
            "Function": fid,
            "mean_eps_ratio": mean_feat1,
            "mean_adj_r2": mean_feat2,
            "mean_bin_feat1": mean_bin_1,
            "mean_bin_feat2": mean_bin_2,
            "majority_bin_feat1": maj_1 if not isinstance(maj_1, list) else "|".join(maj_1),
            "majority_bin_feat2": maj_2 if not isinstance(maj_2, list) else "|".join(maj_2),
            "majority_status_feat1": status_1,
            "majority_status_feat2": status_2,
            "final_bin_feat1": final_bin_1,
            "final_bin_feat2": final_bin_2,
            "group_label": group_label,
            "is_ambiguous": is_ambiguous,
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


def plot_2d_scatter_with_groups(df, function_assignments, output_dir, config_name, strategy, manual_bins):
    func_means = (
        df.groupby("Function")[[FEATURE_1, FEATURE_2]]
        .mean()
        .reset_index()
        .merge(
            function_assignments[["Function", "group_id", "group_label", "is_ambiguous"]],
            on="Function",
            how="left"
        )
    )

    plt.figure(figsize=(8, 6))

    for gid, sub in func_means.groupby("group_id"):
        label = f"G{gid}"
        plt.scatter(sub[FEATURE_1], sub[FEATURE_2], label=label, s=60)

        for _, row in sub.iterrows():
            marker_text = f"f{int(row['Function'])}"
            if bool(row["is_ambiguous"]):
                marker_text += "*"

            plt.text(
                row[FEATURE_1] + 0.04,
                row[FEATURE_2] + 0.008,
                marker_text,
                fontsize=8
            )

    for e in manual_bins[FEATURE_1]:
        if np.isfinite(e):
            plt.axvline(e, color="black", linestyle="--", linewidth=1.2)

    for e in manual_bins[FEATURE_2]:
        if np.isfinite(e):
            plt.axhline(e, color="black", linestyle="--", linewidth=1.2)

    plt.xlabel(FEATURE_1)
    plt.ylabel(FEATURE_2)
    plt.title(f"{config_name}\nAssignment: {strategy}")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"scatter_{config_name}_{strategy}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_group_sizes(function_assignments, output_dir, config_name, strategy):
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
    plt.title(f"Group sizes: {config_name}\nAssignment: {strategy}")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"group_sizes_{config_name}_{strategy}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def summarize(function_assignments, config_name, strategy, manual_bins):
    counts = function_assignments["group_label"].value_counts()

    return {
        "config": config_name,
        "strategy": strategy,
        "eps_edges": str(manual_bins[FEATURE_1]),
        "adj_r2_edges": str(manual_bins[FEATURE_2]),
        "n_functions": int(function_assignments["Function"].nunique()),
        "n_groups": int(counts.shape[0]),
        "min_group_size": int(counts.min()),
        "max_group_size": int(counts.max()),
        "mean_group_size": float(counts.mean()),
        "std_group_size": float(counts.std(ddof=0)),
        "singleton_groups": int((counts == 1).sum()),
        "small_groups_le_2": int((counts <= 2).sum()),
        "ambiguous_functions": int(function_assignments["is_ambiguous"].sum()),
        "tie_feat1_functions": int((function_assignments["majority_status_feat1"] == "tie").sum()),
        "tie_feat2_functions": int((function_assignments["majority_status_feat2"] == "tie").sum()),
    }


def run_one(df_used, config_name, manual_bins, strategy, base_output_dir):
    out_dir = os.path.join(base_output_dir, config_name, strategy)
    ensure_dir(out_dir)

    instance_level_with_bins, function_assignments = compute_assignments(
        df=df_used,
        manual_bins=manual_bins,
        strategy=strategy
    )

    instance_level_with_bins.to_csv(
        os.path.join(out_dir, "instance_level_manual_bins.csv"),
        index=False
    )

    function_assignments.to_csv(
        os.path.join(out_dir, "function_to_groups.csv"),
        index=False
    )

    function_assignments[["Function", "group_id", "group_label"]].to_csv(
        os.path.join(out_dir, "function_group_mapping_for_step3.csv"),
        index=False
    )

    group_sizes = (
        function_assignments["group_label"]
        .value_counts()
        .rename_axis("group_label")
        .reset_index(name="n_functions")
    )
    group_sizes.to_csv(os.path.join(out_dir, "group_sizes.csv"), index=False)

    plot_2d_scatter_with_groups(
        df=df_used,
        function_assignments=function_assignments,
        output_dir=out_dir,
        config_name=config_name,
        strategy=strategy,
        manual_bins=manual_bins
    )

    plot_group_sizes(
        function_assignments=function_assignments,
        output_dir=out_dir,
        config_name=config_name,
        strategy=strategy
    )

    print(f"\n===== {config_name} | {strategy} =====")
    print(function_assignments[
        [
            "Function",
            "mean_bin_feat1",
            "mean_bin_feat2",
            "majority_bin_feat1",
            "majority_bin_feat2",
            "majority_status_feat1",
            "majority_status_feat2",
            "final_bin_feat1",
            "final_bin_feat2",
            "group_id",
            "group_label",
            "is_ambiguous",
        ]
    ])

    print("\nGroup sizes:")
    print(group_sizes)

    return summarize(function_assignments, config_name, strategy, manual_bins)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="intermediate/dim5_selected_features.csv",
        help="Step1 output feature CSV"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/manual_binning_assignment_strategies",
        help="Output directory"
    )

    parser.add_argument(
        "--drop_tails",
        type=float,
        default=0.0,
        help="Trim quantile tails within each function, e.g. 0.05"
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("Loading feature data...")
    df = load_feature_data(args.input_csv)
    print(f"Loaded shape: {df.shape}")

    df_used = trim_tails_within_function(
        df,
        features=[FEATURE_1, FEATURE_2],
        q=args.drop_tails
    )
    print(f"After tail trimming: {df_used.shape}")

    df_used.to_csv(os.path.join(args.output_dir, "step2_feature_data_used.csv"), index=False)

    all_summaries = []

    for config_name, manual_bins in BIN_CONFIGS.items():
        for strategy in ASSIGNMENT_STRATEGIES:
            summary = run_one(
                df_used=df_used,
                config_name=config_name,
                manual_bins=manual_bins,
                strategy=strategy,
                base_output_dir=args.output_dir
            )
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_file = os.path.join(args.output_dir, "summary_all_assignment_strategies.csv")
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Overall summary =====")
    print(summary_df)
    print(f"\nSaved summary to: {summary_file}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()