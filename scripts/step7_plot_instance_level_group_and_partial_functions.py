from pathlib import Path
import math

import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import catboost as cb


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"

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
    / "step7_instance_level_group_and_partial_function_plots"
    / CONFIG_NAME
)

GROUP_COMBINED_OUT_DIR = OUTPUT_DIR / "01_all_groups_combined"
GROUP_WITH_PARTIAL_FUNCTIONS_OUT_DIR = OUTPUT_DIR / "02_group_with_partial_functions"

GROUP_COMBINED_OUT_DIR.mkdir(parents=True, exist_ok=True)
GROUP_WITH_PARTIAL_FUNCTIONS_OUT_DIR.mkdir(parents=True, exist_ok=True)


FUNCTION_SAMPLE_SIZE = 10000
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


def shap_dot_on_axis(ax, shap_values, X, show_ylabels=True, title=None, xlabel=None):
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
        ax.set_title(title, fontsize=10)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    else:
        ax.set_xlabel("")

    if not show_ylabels:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    ax.tick_params(axis="both", labelsize=7)


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


def get_group_label(df: pd.DataFrame, group_id: int):
    if "group_label" in df.columns and df["group_label"].notna().any():
        return df["group_label"].dropna().iloc[0]
    return f"group_{group_id}"


def get_functions_text(df: pd.DataFrame):
    fids = sorted(df["fid"].dropna().unique().tolist())
    return ", ".join([f"f{int(fid)}" for fid in fids])


def get_function_instance_text(df: pd.DataFrame):
    pairs = (
        df[["fid", "iid"]]
        .drop_duplicates()
        .sort_values(["fid", "iid"])
    )

    return ", ".join(
        [f"f{int(row.fid)}_i{int(row.iid)}" for row in pairs.itertuples(index=False)]
    )


def get_group_function_instance_map(df_group: pd.DataFrame):
    """
    返回每个 function 在当前 group 中实际出现的 instances。
    例如:
    {
        14: [4],
        22: [1,2,3,4,5]
    }
    """
    mapping = {}

    pairs = (
        df_group[["fid", "iid"]]
        .drop_duplicates()
        .sort_values(["fid", "iid"])
    )

    for fid, sub in pairs.groupby("fid"):
        mapping[int(fid)] = sorted(sub["iid"].astype(int).tolist())

    return mapping


def load_full_data():
    df = pd.read_pickle(DATA_FILE)

    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
    df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)

    return df


def plot_all_groups_combined(group_records):
    print("\nCombined all-group plot")

    n_groups = len(group_records)

    fig = plt.figure(figsize=(4.1 * n_groups + 1.0, 5.6))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=n_groups + 1,
        width_ratios=[1] * n_groups + [0.06],
        wspace=0.08,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_groups)]
    cax = fig.add_subplot(gs[0, n_groups])

    for idx, record in enumerate(group_records):
        group_id = record["group_id"]
        df_group = record["df"]

        group_label = get_group_label(df_group, group_id)
        function_text = get_functions_text(df_group)

        print(f"Combined plot group {group_id}: {function_text}")

        X, shap_values, train_r2 = compute_shap(
            df_group,
            sample_size=GROUP_SAMPLE_SIZE,
            random_state=42,
        )

        title = f"G{group_id}\n{group_label}\n{function_text}"
        xlabel = f"SHAP on G{group_id}"

        shap_dot_on_axis(
            ax=axes[idx],
            shap_values=shap_values,
            X=X,
            show_ylabels=(idx == 0),
            title=title,
            xlabel=xlabel,
        )

    add_shared_colorbar(fig, cax)

    fig.suptitle(
        f"Instance-level group SHAP beeswarm plots in $d=5$\n{CONFIG_NAME}",
        fontsize=15,
        y=1.04,
    )

    out_file = GROUP_COMBINED_OUT_DIR / "all_groups_beeswarm_one_row_instance_level_d5.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_file}")


def plot_group_with_partial_functions(
    group_id: int,
    df_group: pd.DataFrame,
    df_all: pd.DataFrame,
):
    group_label = get_group_label(df_group, group_id)
    function_text = get_functions_text(df_group)
    instance_text = get_function_instance_text(df_group)
    function_instance_map = get_group_function_instance_map(df_group)

    fids = sorted(function_instance_map.keys())

    print(f"\nGroup + partial functions plot: G{group_id}")
    print(f"Group label: {group_label}")
    print(f"Functions: {function_text}")
    print(f"Function-instances: {instance_text}")

    n_funcs = len(fids)
    funcs_per_row = 4
    n_func_rows = math.ceil(n_funcs / funcs_per_row)

    fig_width = 7.0 + funcs_per_row * 3.7 + 0.8
    fig_height = max(6.4, n_func_rows * 3.5 + 1.5)

    fig = plt.figure(figsize=(fig_width, fig_height))

    outer = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.35, funcs_per_row, 0.08],
        wspace=0.18,
    )

    ax_group = fig.add_subplot(outer[0, 0])
    cax = fig.add_subplot(outer[0, 2])

    Xg, shap_g, r2_g = compute_shap(
        df_group,
        sample_size=GROUP_SAMPLE_SIZE,
        random_state=42,
    )

    shap_dot_on_axis(
        ax=ax_group,
        shap_values=shap_g,
        X=Xg,
        show_ylabels=True,
        title=f"Group {group_id}: {group_label}",
        xlabel=f"SHAP on Group {group_id}",
    )

    ax_group.text(
        0.0,
        -0.24,
        f"Functions: {function_text}\n"
        f"Train $R^2$ = {r2_g:.4f}\n"
        f"n function-instances = {len(df_group[['fid', 'iid']].drop_duplicates())}",
        transform=ax_group.transAxes,
        fontsize=8.5,
        ha="left",
        va="top",
    )

    right_gs = outer[0, 1].subgridspec(
        nrows=n_func_rows,
        ncols=funcs_per_row,
        hspace=0.58,
        wspace=0.25,
    )

    for i, fid in enumerate(fids):
        row = i // funcs_per_row
        col = i % funcs_per_row

        ax = fig.add_subplot(right_gs[row, col])

        iids = function_instance_map[fid]

        df_f_partial = df_all[
            (df_all["fid"] == fid)
            & (df_all["iid"].isin(iids))
        ].copy()

        instance_label = ",".join([f"i{int(x)}" for x in iids])

        print(f"Plot f{fid} with instances: {instance_label}")
        print(f"Shape: {df_f_partial.shape}")

        Xf, shap_f, r2_f = compute_shap(
            df_f_partial,
            sample_size=FUNCTION_SAMPLE_SIZE,
            random_state=42,
        )

        shap_dot_on_axis(
            ax=ax,
            shap_values=shap_f,
            X=Xf,
            show_ylabels=(col == 0),
            title=f"$f_{{{int(fid)}}}$ ({instance_label})",
            xlabel=f"SHAP on $f_{{{int(fid)}}}$",
        )

        ax.text(
            0.0,
            -0.30,
            f"Instances: {instance_label}\nTrain $R^2$={r2_f:.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            ha="left",
            va="top",
        )

    total_slots = n_func_rows * funcs_per_row
    for empty_i in range(n_funcs, total_slots):
        row = empty_i // funcs_per_row
        col = empty_i % funcs_per_row
        ax_empty = fig.add_subplot(right_gs[row, col])
        ax_empty.axis("off")

    add_shared_colorbar(fig, cax)

    fig.suptitle(
        f"Group {group_id} and function-specific instance subsets in $d=5$",
        fontsize=15,
        y=0.98,
    )

    out_file = (
        GROUP_WITH_PARTIAL_FUNCTIONS_OUT_DIR
        / f"group_{group_id}_with_partial_function_instances_d5.png"
    )

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

    group_records = []

    for group_id, file in group_files:
        df_group = pd.read_pickle(file)

        group_records.append(
            {
                "group_id": group_id,
                "file": file,
                "df": df_group,
            }
        )

    # Part 1:
    # 不再输出单个 group beeswarm，因为你已经成功生成过。
    # 这里只输出所有 group 拼接图。
    plot_all_groups_combined(group_records)

    # Part 2:
    # group + function plot，但每个 function 只使用当前 group 中实际出现的 instances。
    df_all = load_full_data()

    for record in group_records:
        plot_group_with_partial_functions(
            group_id=record["group_id"],
            df_group=record["df"],
            df_all=df_all,
        )

    print("\nDone.")
    print("Combined group plot:", GROUP_COMBINED_OUT_DIR)
    print("Group with partial function-instance plots:", GROUP_WITH_PARTIAL_FUNCTIONS_OUT_DIR)


if __name__ == "__main__":
    main()