from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")

    input_file = project_root / "intermediate" / "dim5_selected_features.csv"
    output_dir = project_root / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)

    print("\n=== Input shape ===")
    print(df.shape)

    # ===== 只用 feature 做 clustering =====
    X = df[["adj_r2", "eps_ratio"]].values

    # ===== 设置 group 数量（先用3个）=====
    k = 3

    print(f"\nRunning KMeans with k={k} ...")

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["group"] = kmeans.fit_predict(X)

    print("\n=== Group distribution ===")
    print(df["group"].value_counts())

    print("\n=== Preview ===")
    print(df.head())

    # 保存
    output_file = output_dir / f"dim5_groups_k{k}.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()