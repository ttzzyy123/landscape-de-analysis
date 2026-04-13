from pathlib import Path
import pandas as pd


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")
    data_path = project_root / "data" / "features_summary_dim_5_sobol.csv"
    output_dir = project_root / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full dataset (this may take some time)...")
    df = pd.read_csv(data_path, sep=";")

    # 清理列名
    df.columns = [c.strip() for c in df.columns]

    # 清理字符串列
    for col in ["Feature name", "# samples", "Function", "Instance"]:
        df[col] = df[col].astype(str).str.strip()

    # 数值列
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    print("\n=== Original shape ===")
    print(df.shape)

    # 只保留 1000d = 5000
    df = df[df["# samples"] == "5000"]

    print("\n=== After filtering budget=5000 ===")
    print(df.shape)

    # 只保留两个目标 feature
    target_features = [
        "ic.eps.ratio",
        "ela_meta.lin_simple.adj_r2"
    ]
    df = df[df["Feature name"].isin(target_features)]

    print("\n=== After filtering target features ===")
    print(df.shape)

    # 长表转宽表
    print("\nPivoting to wide format...")
    df_wide = df.pivot_table(
        index=["Function", "Instance"],
        columns="Feature name",
        values="Value"
    ).reset_index()

    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={
        "ic.eps.ratio": "eps_ratio",
        "ela_meta.lin_simple.adj_r2": "adj_r2"
    })

    print("\n=== Final dataset shape ===")
    print(df_wide.shape)

    print("\n=== Preview ===")
    print(df_wide.head())

    output_file = output_dir / "dim5_selected_features.csv"
    df_wide.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()