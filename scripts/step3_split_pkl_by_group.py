from pathlib import Path
import pandas as pd


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")

    group_file = project_root / "output" / "manual_binning" / "function_group_mapping_for_step3.csv"
    pkl_file = project_root / "data" / "de_final_5_processed.pkl"
    output_dir = project_root / "intermediate" / "group_data_manual_bins"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading group mapping: {group_file}")
    df_group = pd.read_csv(group_file)

    print(f"Loading experiment data: {pkl_file}")
    df = pd.read_pickle(pkl_file)

    print("\n=== Original experiment data shape ===")
    print(df.shape)

    print("\n=== Original columns ===")
    print(df.columns.tolist())

    required_group_cols = ["Function", "group_id", "group_label"]
    missing = [c for c in required_group_cols if c not in df_group.columns]
    if missing:
        raise ValueError(f"Missing required columns in group mapping: {missing}")

    # 如果有 dim 列，就保险起见只保留 d=5
    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    print("\n=== After filtering dim=5 ===")
    print(df.shape)

    # group 文件里的 Function 对应 pkl 里的 fid
    df_group = df_group.rename(columns={"Function": "fid"})

    # 类型统一，避免 merge 出问题
    df_group["fid"] = pd.to_numeric(df_group["fid"], errors="coerce").astype("Int64")
    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype("Int64")

    # 合并 group 标签
    df_merged = df.merge(
        df_group[["fid", "group_id", "group_label"]],
        on="fid",
        how="left"
    )

    print("\n=== After merging group labels ===")
    print(df_merged.shape)

    print("\n=== Missing group labels ===")
    print(df_merged["group_id"].isna().sum())

    print("\n=== Group distribution in experiment data ===")
    print(df_merged["group_id"].value_counts(dropna=False).sort_index())

    print("\n=== Group label distribution in experiment data ===")
    print(df_merged["group_label"].value_counts(dropna=False))

    # 保存整份带 group 的数据
    merged_file = output_dir / "de_final_5_with_manual_groups.pkl"
    df_merged.to_pickle(merged_file)
    print(f"\nSaved merged dataset to: {merged_file}")

    # 每个 group 单独保存
    valid_groups = sorted(df_group["group_id"].dropna().unique())

    for g in valid_groups:
        df_g = df_merged[df_merged["group_id"] == g].copy()
        group_label = df_g["group_label"].dropna().iloc[0] if len(df_g) > 0 else "unknown"

        print(f"\n=== Group {g} ({group_label}) ===")
        print(f"Shape: {df_g.shape}")
        print("Unique fid:", sorted(df_g["fid"].dropna().unique().tolist()))

        out_file = output_dir / f"de_final_5_group_{g}.pkl"
        df_g.to_pickle(out_file)
        print(f"Saved: {out_file}")

    # summary
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

    summary_file = output_dir / "group_data_summary_manual_bins.csv"
    summary.to_csv(summary_file, index=False)

    print("\n=== Group summary ===")
    print(summary)

    print(f"\nSaved summary to: {summary_file}")


if __name__ == "__main__":
    main()