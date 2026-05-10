from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path("/data/s3795888/ioh_project/my_landscape_experiments")

PKL_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"

MAPPING_BASE_DIR = PROJECT_ROOT / "output" / "manual_binning_assignment_strategies"

OUTPUT_BASE_DIR = PROJECT_ROOT / "intermediate" / "group_data_assignment_strategy_comparison"

CONFIGS = [
    "eps2bins_3_r025065",
    "eps2bins_3_r0307_main",
]

STRATEGIES = [
    "ambiguous_on_tie",
    "majority_with_mean_tiebreak",
    "mean_based",
]


def load_experiment_data():
    print(f"Loading experiment data: {PKL_FILE}")

    if not PKL_FILE.exists():
        raise FileNotFoundError(f"Experiment data file not found: {PKL_FILE}")

    df = pd.read_pickle(PKL_FILE)

    print("\n=== Original experiment data shape ===")
    print(df.shape)

    print("\n=== Original columns ===")
    print(df.columns.tolist())

    if "fid" not in df.columns:
        raise ValueError("Experiment data must contain column: fid")

    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    print("\n=== After filtering dim=5 ===")
    print(df.shape)

    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fid"]).copy()
    df["fid"] = df["fid"].astype(int)

    return df


def prepare_group_mapping(group_file: Path):
    print(f"\nLoading group mapping: {group_file}")

    if not group_file.exists():
        raise FileNotFoundError(f"Group mapping file not found: {group_file}")

    df_group = pd.read_csv(group_file)

    required_group_cols = ["Function", "group_id", "group_label"]
    missing = [c for c in required_group_cols if c not in df_group.columns]
    if missing:
        raise ValueError(f"Missing required columns in group mapping: {missing}")

    df_group = df_group.rename(columns={"Function": "fid"})

    df_group["fid"] = pd.to_numeric(df_group["fid"], errors="coerce").astype("Int64")
    df_group["group_id"] = pd.to_numeric(df_group["group_id"], errors="coerce").astype("Int64")

    df_group = df_group.dropna(subset=["fid", "group_id"]).copy()

    df_group["fid"] = df_group["fid"].astype(int)
    df_group["group_id"] = df_group["group_id"].astype(int)

    return df_group


def split_one_strategy(df: pd.DataFrame, config_name: str, strategy_name: str):
    print("\n" + "=" * 80)
    print(f"Processing config: {config_name}")
    print(f"Processing strategy: {strategy_name}")
    print("=" * 80)

    group_file = (
        MAPPING_BASE_DIR
        / config_name
        / strategy_name
        / "function_group_mapping_for_step3.csv"
    )

    output_dir = (
        OUTPUT_BASE_DIR
        / config_name
        / strategy_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df_group = prepare_group_mapping(group_file)

    print("\n=== Group mapping ===")
    print(df_group[["fid", "group_id", "group_label"]].sort_values("fid"))

    mapping_summary = (
        df_group.groupby(["group_id", "group_label"], dropna=False)
        .agg(n_functions=("fid", "nunique"))
        .reset_index()
        .sort_values("group_id")
    )

    print("\n=== Function count per group mapping ===")
    print(mapping_summary)

    mapping_summary.to_csv(output_dir / "function_count_per_group_mapping.csv", index=False)

    df_merged = df.merge(
        df_group[["fid", "group_id", "group_label"]],
        on="fid",
        how="left"
    )

    print("\n=== After merging group labels ===")
    print(df_merged.shape)

    missing_groups = df_merged["group_id"].isna().sum()
    print("\n=== Missing group labels ===")
    print(missing_groups)

    if missing_groups > 0:
        missing_fids = (
            df_merged[df_merged["group_id"].isna()][["fid"]]
            .drop_duplicates()
            .sort_values("fid")
        )
        print("\nRows with missing group labels:")
        print(missing_fids)

    print("\n=== Group distribution in experiment data ===")
    print(df_merged["group_id"].value_counts(dropna=False).sort_index())

    print("\n=== Group label distribution in experiment data ===")
    print(df_merged["group_label"].value_counts(dropna=False))

    merged_file = output_dir / "de_final_5_with_manual_groups.pkl"
    df_merged.to_pickle(merged_file)
    print(f"\nSaved merged dataset to: {merged_file}")

    valid_groups = sorted(df_group["group_id"].dropna().unique())

    for g in valid_groups:
        df_g = df_merged[df_merged["group_id"] == g].copy()

        if len(df_g) > 0 and df_g["group_label"].notna().any():
            group_label = df_g["group_label"].dropna().iloc[0]
        else:
            group_label = "unknown"

        print(f"\n=== Group {g} ({group_label}) ===")
        print(f"Shape: {df_g.shape}")
        print("Unique fid:", sorted(df_g["fid"].dropna().unique().tolist()))

        out_file = output_dir / f"de_final_5_group_{g}.pkl"
        df_g.to_pickle(out_file)
        print(f"Saved: {out_file}")

    agg_dict = {
        "n_rows": ("fid", "size"),
        "n_functions": ("fid", "nunique"),
    }

    if "iid" in df_merged.columns:
        agg_dict["n_instances"] = ("iid", "nunique")
    if "auc" in df_merged.columns:
        agg_dict["mean_auc"] = ("auc", "mean")
    if "aucLarge" in df_merged.columns:
        agg_dict["mean_aucLarge"] = ("aucLarge", "mean")

    summary = (
        df_merged.groupby(["group_id", "group_label"], dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .sort_values("group_id")
    )

    summary["config"] = config_name
    summary["strategy"] = strategy_name

    summary_file = output_dir / "group_data_summary_manual_bins.csv"
    summary.to_csv(summary_file, index=False)

    print("\n=== Group summary ===")
    print(summary)

    print(f"\nSaved summary to: {summary_file}")
    print(f"All group data saved to: {output_dir}")

    return summary


def main():
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_experiment_data()

    all_summaries = []

    for config_name in CONFIGS:
        for strategy_name in STRATEGIES:
            summary = split_one_strategy(
                df=df,
                config_name=config_name,
                strategy_name=strategy_name,
            )
            all_summaries.append(summary)

    combined_summary = pd.concat(all_summaries, ignore_index=True)

    combined_summary_file = OUTPUT_BASE_DIR / "summary_all_6_group_data_splits.csv"
    combined_summary.to_csv(combined_summary_file, index=False)

    print("\n" + "=" * 80)
    print("DONE: all six group data splits finished.")
    print("=" * 80)

    print(f"\nCombined summary saved to:")
    print(combined_summary_file)

    print(f"\nAll output directories saved under:")
    print(OUTPUT_BASE_DIR)


if __name__ == "__main__":
    main()