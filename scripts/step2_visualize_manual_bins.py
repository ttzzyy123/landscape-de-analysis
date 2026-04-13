from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    project_root = Path("/data/s3795888/ioh_project/my_landscape_experiments")

    feature_file = project_root / "output/manual_binning/step2_feature_data_used.csv"
    group_file = project_root / "output/manual_binning/function_group_mapping_for_step3.csv"

    output_dir = project_root / "output/manual_binning/plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_feat = pd.read_csv(feature_file)
    df_group = pd.read_csv(group_file)

    # ===== merge =====
    df = df_feat.merge(df_group, on="Function")

    # =========================
    # 图1：Feature distribution
    # =========================
    print("Plotting feature distributions...")

    for col in ["eps_ratio", "adj_r2"]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=20, kde=True)

        # 加 bin 边界（你现在用的是 2x3 main）
        if col == "eps_ratio":
            edges = [3.0]
        else:
            edges = [0.3, 0.7]

        for e in edges:
            plt.axvline(e, color="red", linestyle="--")

        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=300)
        plt.close()

    # =========================
    # 图2：2D scatter（核心图）
    # =========================
    print("Plotting 2D scatter...")

    # 每个 function 取 mean（更干净）
    df_mean = (
        df.groupby(["Function", "group_id", "group_label"], as_index=False)
        [["eps_ratio", "adj_r2"]]
        .mean()
    )

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df_mean,
        x="eps_ratio",
        y="adj_r2",
        hue="group_label",
        palette="tab10",
        s=80
    )

    # bin lines
    plt.axvline(3.0, color="black", linestyle="--")
    plt.axhline(0.3, color="black", linestyle="--")
    plt.axhline(0.7, color="black", linestyle="--")

    plt.title("Function grouping in feature space")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_dir / "scatter_groups.png", dpi=300)
    plt.close()

    # =========================
    # 图3：Group size
    # =========================
    print("Plotting group sizes...")

    group_counts = df_group["group_label"].value_counts().reset_index()
    group_counts.columns = ["group_label", "count"]

    plt.figure(figsize=(8, 4))
    sns.barplot(data=group_counts, x="group_label", y="count")

    plt.xticks(rotation=45, ha="right")
    plt.title("Number of functions per group")
    plt.tight_layout()

    plt.savefig(output_dir / "group_sizes.png", dpi=300)
    plt.close()

    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()