from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")

    input_file = project_root / "intermediate" / "dim5_selected_features.csv"
    output_dir = project_root / "intermediate"

    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)

    # ===== 1️⃣ 按 function 聚合 =====
    df_func = df.groupby("Function").agg({
        "adj_r2": "mean",
        "eps_ratio": "mean"
    }).reset_index()

    print("\n=== Function-level dataset ===")
    print(df_func.shape)
    print(df_func.head())

    # ===== 2️⃣ clustering =====
    X = df_func[["adj_r2", "eps_ratio"]].values

    k = 3
    print(f"\nRunning KMeans with k={k}...")

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_func["group"] = kmeans.fit_predict(X)

    print("\n=== Group distribution ===")
    print(df_func["group"].value_counts())

    print("\n=== Result preview ===")
    print(df_func)

    # ===== 保存 =====
    output_file = output_dir / f"dim5_function_groups_k{k}.csv"
    df_func.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()