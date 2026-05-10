from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import catboost as cb


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_NAME = "eps2bins_4_r025050075_instance"

GROUP_DATA_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "group_data_instance_level_comparison"
    / CONFIG_NAME
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "step7_instance_level_single_group_beeswarm"
    / CONFIG_NAME
)

GROUP_SINGLE_OUT_DIR = OUTPUT_DIR / "01_group_single_with_function_instances"
GROUP_SINGLE_OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_SAMPLE_SIZE = 20000


# =========================
# Niels-style feature order
# =========================
FEATURE_ORDER = [
    "CR",
    "F",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
    "Instance variance",
    "Stochastic variance",
]

RAW_FEATURES = [
    "CR",
    "F",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
    "iid",
    "seed",
]

CATEGORICAL_FEATURES = [
    "crossover",
    "mutation_base",
    "mutation_reference",
]

CATEGORY_MAPS = {
    "crossover": {
        "bin": 0,
        "exp": 1,
    },
    "mutation_base": {
        "best": 0,
        "rand": 1,
        "target": 2,
    },
    "mutation_reference": {
        "best": 0,
        "nan": 1,
        "pbest": 2,
        "rand": 3,
    },
}

RENAME_MAP = {
    "iid": "Instance variance",
    "seed": "Stochastic variance",
}

TARGET_COL = "auc"


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    required_cols = RAW_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[RAW_FEATURES].copy()
    X = X.rename(columns=RENAME_MAP)

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
        X[col] = X[col].map(CATEGORY_MAPS[col])

        if X[col].isna().any():
            unknown_values = sorted(
                df[col].astype(str)[X[col].isna()].unique().tolist()
            )
            raise ValueError(
                f"Unknown category in {col}: {unknown_values}. "
                f"Please update CATEGORY_MAPS."
            )

        X[col] = X[col].astype(int)

    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    ordered_cols = [c for c in FEATURE_ORDER if c in X.columns]
    X = X[ordered_cols]

    y = df[TARGET_COL].copy()

    return X, y


def compute_shap(df: pd.DataFrame, sample_size: int, random_state: int = 42):
    if len(df) > sample_size:
        print(f"Sampling from {len(df)} -> {sample_size}")
        df = df.sample(sample_size, random_state=random_state)

    X, y = prepare_features(df)

    model = cb.CatBoostRegressor(
        iterations=100,
        depth=14,
        random_seed=42,
        verbose=False,
    )

    model.fit(X, y)
    train_r2 = model.score(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return X, shap_values, train_r2


def shap_dot_on_axis(ax, shap_values, X, title=None, xlabel=None):
    plt.sca(ax)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(X.columns),
        max_display=len(X.columns),
        sort=False,
        show=False,
        color_bar=False,
        plot_size=None,
        cmap=plt.get_cmap("viridis"),
    )

    if title:
        ax.set_title(title, fontsize=12)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    else:
        ax.set_xlabel("")

    ax.tick_params(axis="both", labelsize=9)


def add_shared_colorbar(fig, cax):
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", rotation=90)


def discover_group_files(group_dir: Path):
    files = sorted(group_dir.glob("de_final_5_group_*.pkl"))
    pairs = []

    for f in files:
        try:
            group_id = int(f.stem.split("_")[-1])
            pairs.append((group_id, f))
        except ValueError:
            continue

    return pairs


def get_group_info(df: pd.DataFrame, group_id: int):
    if "group_label" in df.columns and df["group_label"].notna().any():
        group_label = df["group_label"].dropna().iloc[0]
    else:
        group_label = f"group_{group_id}"

    if "function_instance" in df.columns:
        function_instances = sorted(df["function_instance"].dropna().unique().tolist())
    else:
        pairs = (
            df[["fid", "iid"]]
            .drop_duplicates()
            .sort_values(["fid", "iid"])
        )
        function_instances = [
            f"f{int(row.fid)}_i{int(row.iid)}"
            for row in pairs.itertuples(index=False)
        ]

    if "fid" in df.columns:
        fids = sorted(df["fid"].dropna().unique().tolist())
        function_text = ", ".join([f"f{int(fid)}" for fid in fids])
    else:
        function_text = ""

    instance_text = ", ".join(function_instances)

    return group_label, function_text, instance_text, function_instances


def wrap_text_by_items(items, items_per_line=6):
    lines = []
    for i in range(0, len(items), items_per_line):
        lines.append(", ".join(items[i:i + items_per_line]))
    return "\n".join(lines)


def plot_single_group(group_id: int, df_group: pd.DataFrame):
    group_label, function_text, instance_text, function_instances = get_group_info(
        df_group,
        group_id,
    )

    print(f"\nSingle group plot: G{group_id} | {group_label}")
    print(f"Functions: {function_text}")
    print(f"Function-instances: {instance_text}")

    X, shap_values, train_r2 = compute_shap(
        df_group,
        sample_size=GROUP_SAMPLE_SIZE,
        random_state=42,
    )

    wrapped_instances = wrap_text_by_items(function_instances, items_per_line=6)

    # Height increases slightly for groups with many function-instances
    extra_height = max(0, len(function_instances) // 6) * 0.25
    fig = plt.figure(figsize=(10.5, 7.5 + extra_height))

    ax = fig.add_axes([0.16, 0.28, 0.66, 0.62])
    cax = fig.add_axes([0.86, 0.28, 0.025, 0.62])

    title = f"Group {group_id}: {group_label}"
    xlabel = f"Hyper-parameter contributions on Group {group_id} in $d=5$"

    shap_dot_on_axis(
        ax=ax,
        shap_values=shap_values,
        X=X,
        title=title,
        xlabel=xlabel,
    )

    add_shared_colorbar(fig, cax)

    fig.text(
        0.16,
        0.17,
        f"Functions in Group {group_id}: {function_text}",
        fontsize=10,
        ha="left",
        va="top",
    )

    fig.text(
        0.16,
        0.125,
        f"Function-instances in Group {group_id}:\n{wrapped_instances}",
        fontsize=8.5,
        ha="left",
        va="top",
    )

    fig.text(
        0.16,
        0.045,
        f"Train $R^2$ = {train_r2:.4f} | "
        f"n function-instances = {len(function_instances)}",
        fontsize=10,
        ha="left",
        va="top",
    )

    out_file = GROUP_SINGLE_OUT_DIR / f"summary_group_{group_id}_instance_level_d5.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def main():
    print("Using instance-level group data:")
    print(GROUP_DATA_DIR)

    print("Output directory:")
    print(OUTPUT_DIR)

    group_files = discover_group_files(GROUP_DATA_DIR)

    if not group_files:
        raise FileNotFoundError(f"No group files found in {GROUP_DATA_DIR}")

    print(f"\nFound {len(group_files)} group files:")
    print(group_files)

    for group_id, file in group_files:
        print(f"\nLoading group {group_id}: {file}")
        df_group = pd.read_pickle(file)
        print("Shape:", df_group.shape)

        plot_single_group(
            group_id=group_id,
            df_group=df_group,
        )

    print("\nDone.")
    print("Single group instance-level plots:", GROUP_SINGLE_OUT_DIR)


if __name__ == "__main__":
    main()