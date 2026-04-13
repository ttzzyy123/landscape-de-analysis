from pathlib import Path
import pandas as pd


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")

    group_file = project_root / "output" / "manual_binning" / "function_group_mapping_for_step3.csv"

    print(f"Loading groups: {group_file}")
    df_group = pd.read_csv(group_file)

    required_cols = ["Function", "group_id", "group_label"]
    missing = [c for c in required_cols if c not in df_group.columns]
    if missing:
        raise ValueError(f"Missing required columns in group mapping: {missing}")

    print("\n=== Group ID distribution ===")
    print(df_group["group_id"].value_counts().sort_index())

    print("\n=== Group label distribution ===")
    print(df_group["group_label"].value_counts())

    # 每个 group 的 function 列表
    group_dict = {}
    label_dict = {}

    for g in sorted(df_group["group_id"].dropna().unique()):
        sub = df_group[df_group["group_id"] == g].copy()
        funcs = sorted(sub["Function"].tolist())
        label = sub["group_label"].iloc[0]

        group_dict[g] = funcs
        label_dict[g] = label

    print("\n=== Functions per group ===")
    for g, funcs in group_dict.items():
        print(f"\nGroup {g} ({label_dict[g]}):")
        print(funcs)

    # 保存文本 mapping
    output_file = project_root / "intermediate" / "group_to_functions_manual_bins.txt"
    with open(output_file, "w") as f:
        for g, funcs in group_dict.items():
            f.write(f"Group {g} ({label_dict[g]}): {funcs}\n")

    # 再保存一个更正式的 csv
    rows = []
    for g, funcs in group_dict.items():
        for fid in funcs:
            rows.append({
                "group_id": g,
                "group_label": label_dict[g],
                "Function": fid
            })

    output_csv = project_root / "intermediate" / "group_to_functions_manual_bins.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"\nSaved text mapping to: {output_file}")
    print(f"Saved csv mapping to: {output_csv}")


if __name__ == "__main__":
    main()