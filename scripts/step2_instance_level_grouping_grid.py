import os
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_1 = "eps_ratio"
FEATURE_2 = "adj_r2"


# ============================================================
# Manual bin configs for instance-level grouping
# ============================================================

BIN_CONFIGS = {
    # Existing configs
    "eps2bins_3_r025065_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.25, 0.65, np.inf],
    },
    "eps2bins_3_r0307_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.30, 0.70, np.inf],
    },

    # Slightly different r2 boundaries
    "eps2bins_3_r020060_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.20, 0.60, np.inf],
    },
    "eps2bins_3_r035075_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.35, 0.75, np.inf],
    },

    # More sensitive eps boundaries
    "eps2bins_3_r0307_eps25_instance": {
        "eps_ratio": [-np.inf, 2.5, np.inf],
        "adj_r2": [-np.inf, 0.30, 0.70, np.inf],
    },
    "eps2bins_3_r0307_eps35_instance": {
        "eps_ratio": [-np.inf, 3.5, np.inf],
        "adj_r2": [-np.inf, 0.30, 0.70, np.inf],
    },

    # 3 eps bins x 3 r2 bins
    "eps3bins_3_r0307_instance": {
        "eps_ratio": [-np.inf, 2.0, 4.0, np.inf],
        "adj_r2": [-np.inf, 0.30, 0.70, np.inf],
    },

    # 2 eps bins x 4 r2 bins
    "eps2bins_4_r020406_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.20, 0.40, 0.60, np.inf],
    },
    "eps2bins_4_r025050075_instance": {
        "eps_ratio": [-np.inf, 3.0, np.inf],
        "adj_r2": [-np.inf, 0.25, 0.50, 0.75, np.inf],
    },
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\nCurrent columns: {df.columns.tolist()}"
        )


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

    df["Function"] = pd.to_numeric(df["Function"], errors="coerce")
    df["Instance"] = pd.to_numeric(df["Instance"], errors="coerce")
    df[FEATURE_1] = pd.to_numeric(df[FEATURE_1], errors="coerce")
    df[FEATURE_2] = pd.to_numeric(df[FEATURE_2], errors="coerce")

    df = df.dropna(subset=["Function", "Instance", FEATURE_1, FEATURE_2]).copy()

    df["Function"] = df["Function"].astype(int)
    df["Instance"] = df["Instance"].astype(int)

    df["function_instance"] = (
        "f" + df["Function"].astype(str) + "_i" + df["Instance"].astype(str)
    )

    return df


def assign_bin_label(values, edges, prefix):
    labels = [f"{prefix}_bin_{i}" for i in range(len(edges) - 1)]
    return pd.cut(
        values,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False
    )


def assign_instance_groups(df, manual_bins):
    out = df.copy()

    out["bin_feat1"] = assign_bin_label(
        out[FEATURE_1],
        manual_bins[FEATURE_1],
        "eps"
    )

    out["bin_feat2"] = assign_bin_label(
        out[FEATURE_2],
        manual_bins[FEATURE_2],
        "r2"
    )

    out["group_label"] = out["bin_feat1"].astype(str) + "__" + out["bin_feat2"].astype(str)

    unique_groups = sorted(out["group_label"].dropna().unique())
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    out["group_id"] = out["group_label"].map(group_to_id)

    return out


def compute_function_split_summary(instance_assignments):
    """
    Summarise how instances of each function are split across groups.
    This is important for studying instance variance / boundary functions.
    """
    rows = []

    for fid, sub in instance_assignments.groupby("Function"):
        group_counts = sub["group_label"].value_counts().to_dict()
        group_ids = sorted(sub["group_id"].dropna().unique().astype(int).tolist())

        main_group_label = sub["group_label"].value_counts().idxmax()
        main_group_id = int(
            sub.loc[sub["group_label"] == main_group_label, "group_id"].iloc[0]
        )

        rows.append({
            "Function": fid,
            "n_instances": int(sub["Instance"].nunique()),
            "n_groups_covered": int(sub["group_label"].nunique()),
            "main_group_id": main_group_id,
            "main_group_label": main_group_label,
            "group_ids_covered": "|".join([str(x) for x in group_ids]),
            "group_label_counts": str(group_counts),
            "eps_ratio_mean": float(sub[FEATURE_1].mean()),
            "eps_ratio_std": float(sub[FEATURE_1].std(ddof=0)),
            "adj_r2_mean": float(sub[FEATURE_2].mean()),
            "adj_r2_std": float(sub[FEATURE_2].std(ddof=0)),
            "is_instance_split": int(sub["group_label"].nunique() > 1),
        })

    return pd.DataFrame(rows).sort_values("Function").reset_index(drop=True)


def compute_group_summary(instance_assignments):
    rows = []

    for group_label, sub in instance_assignments.groupby("group_label"):
        rows.append({
            "group_id": int(sub["group_id"].iloc[0]),
            "group_label": group_label,
            "n_function_instances": int(len(sub)),
            "n_functions": int(sub["Function"].nunique()),
            "functions": ",".join([f"f{x}" for x in sorted(sub["Function"].unique())]),
            "function_instances": ",".join(sorted(sub["function_instance"].tolist())),
            "mean_eps_ratio": float(sub[FEATURE_1].mean()),
            "mean_adj_r2": float(sub[FEATURE_2].mean()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("group_id")
        .reset_index(drop=True)
    )


def evaluate_config(instance_assignments, function_split_summary):
    group_counts = instance_assignments["group_label"].value_counts()

    n_total_instances = len(instance_assignments)
    n_groups = group_counts.shape[0]

    return {
        "n_function_instances": int(n_total_instances),
        "n_functions": int(instance_assignments["Function"].nunique()),
        "n_groups": int(n_groups),
        "min_group_size_instances": int(group_counts.min()),
        "max_group_size_instances": int(group_counts.max()),
        "mean_group_size_instances": float(group_counts.mean()),
        "std_group_size_instances": float(group_counts.std(ddof=0)),
        "singleton_groups_instances": int((group_counts == 1).sum()),
        "small_groups_le_2_instances": int((group_counts <= 2).sum()),
        "split_functions": int(function_split_summary["is_instance_split"].sum()),
        "non_split_functions": int((function_split_summary["is_instance_split"] == 0).sum()),
        "mean_groups_per_function": float(function_split_summary["n_groups_covered"].mean()),
        "max_groups_per_function": int(function_split_summary["n_groups_covered"].max()),
    }


def plot_instance_scatter(instance_assignments, output_dir, config_name, manual_bins):
    plt.figure(figsize=(9, 6))

    for gid, sub in instance_assignments.groupby("group_id"):
        plt.scatter(
            sub[FEATURE_1],
            sub[FEATURE_2],
            s=45,
            alpha=0.75,
            label=f"G{int(gid)}"
        )

        for _, row in sub.iterrows():
            plt.text(
                row[FEATURE_1] + 0.03,
                row[FEATURE_2] + 0.006,
                row["function_instance"],
                fontsize=6
            )

    for e in manual_bins[FEATURE_1]:
        if np.isfinite(e):
            plt.axvline(e, linestyle="--", linewidth=1.1)

    for e in manual_bins[FEATURE_2]:
        if np.isfinite(e):
            plt.axhline(e, linestyle="--", linewidth=1.1)

    plt.xlabel(FEATURE_1)
    plt.ylabel(FEATURE_2)
    plt.title(f"Instance-level grouping: {config_name}")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"scatter_instance_level_{config_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_group_sizes(group_summary, output_dir, config_name):
    plt.figure(figsize=(10, 4))

    labels = group_summary["group_label"].tolist()
    values = group_summary["n_function_instances"].tolist()

    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Number of function-instances")
    plt.title(f"Instance-level group sizes: {config_name}")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"group_sizes_instance_level_{config_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_function_split_bar(function_split_summary, output_dir, config_name):
    plt.figure(figsize=(10, 4))

    labels = ["f" + str(x) for x in function_split_summary["Function"].tolist()]
    values = function_split_summary["n_groups_covered"].tolist()

    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Number of groups covered by instances")
    plt.title(f"Function instance split summary: {config_name}")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"function_instance_split_{config_name}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()


def run_one_config(df, config_name, manual_bins, base_output_dir):
    out_dir = os.path.join(base_output_dir, config_name)
    ensure_dir(out_dir)

    instance_assignments = assign_instance_groups(df, manual_bins)

    function_split_summary = compute_function_split_summary(instance_assignments)
    group_summary = compute_group_summary(instance_assignments)

    instance_assignments.to_csv(
        os.path.join(out_dir, "instance_group_assignments.csv"),
        index=False
    )

    function_split_summary.to_csv(
        os.path.join(out_dir, "function_instance_split_summary.csv"),
        index=False
    )

    group_summary.to_csv(
        os.path.join(out_dir, "group_summary.csv"),
        index=False
    )

    # This is the key file for Step3 merge
    instance_assignments[
        [
            "Function",
            "Instance",
            "function_instance",
            "group_id",
            "group_label",
            "bin_feat1",
            "bin_feat2",
        ]
    ].to_csv(
        os.path.join(out_dir, "instance_group_mapping_for_step3.csv"),
        index=False
    )

    plot_instance_scatter(instance_assignments, out_dir, config_name, manual_bins)
    plot_group_sizes(group_summary, out_dir, config_name)
    plot_function_split_bar(function_split_summary, out_dir, config_name)

    summary = evaluate_config(instance_assignments, function_split_summary)
    summary["config"] = config_name
    summary["eps_edges"] = str(manual_bins[FEATURE_1])
    summary["adj_r2_edges"] = str(manual_bins[FEATURE_2])

    print(f"\n===== {config_name} =====")
    print("\nGroup summary:")
    print(group_summary[["group_id", "group_label", "n_function_instances", "n_functions", "functions"]])

    print("\nFunction split summary:")
    print(function_split_summary[
        [
            "Function",
            "n_instances",
            "n_groups_covered",
            "main_group_id",
            "main_group_label",
            "group_ids_covered",
            "is_instance_split",
            "eps_ratio_std",
            "adj_r2_std",
        ]
    ])

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default="intermediate/dim5_selected_features_by_instance.csv",
        help="Step1 output CSV with Function, Instance, eps_ratio, adj_r2"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/step2_instance_level_grouping_grid",
        help="Output directory"
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("Loading feature data...")
    df = load_feature_data(args.input_csv)

    print("\n=== Loaded feature data ===")
    print(df.shape)
    print(df.head())

    print("\n=== Number of instances per function ===")
    print(df.groupby("Function")["Instance"].nunique())

    all_summaries = []

    for config_name, manual_bins in BIN_CONFIGS.items():
        summary = run_one_config(
            df=df,
            config_name=config_name,
            manual_bins=manual_bins,
            base_output_dir=args.output_dir
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    # A simple ranking heuristic:
    # prefer fewer singleton groups, fewer tiny groups,
    # but allow some function splits because that is what we want to investigate.
    summary_df["score_for_screening"] = (
        summary_df["singleton_groups_instances"] * 3
        + summary_df["small_groups_le_2_instances"] * 2
        + summary_df["std_group_size_instances"]
        - summary_df["split_functions"] * 0.2
    )

    summary_df = summary_df.sort_values("score_for_screening").reset_index(drop=True)

    summary_file = os.path.join(args.output_dir, "summary_all_instance_level_configs.csv")
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Overall instance-level config summary =====")
    print(summary_df[
        [
            "config",
            "n_groups",
            "min_group_size_instances",
            "max_group_size_instances",
            "singleton_groups_instances",
            "small_groups_le_2_instances",
            "split_functions",
            "mean_groups_per_function",
            "score_for_screening",
        ]
    ])

    print(f"\nSaved summary to: {summary_file}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()