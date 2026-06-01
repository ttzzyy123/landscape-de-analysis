# scripts/new_function_task_H_individual_shap_for_real_generated_function.py

from pathlib import Path
import json
import os

import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import catboost as cb


# ============================================================
# New Function Task H
# Batch individual SHAP analysis for Quentin affine BBOB functions
#
# Input:
# intermediate/new_function_task/modde_affine_function_results/
#   affine_function_f3_f6_alpha_0p9_modde_results_processed.pkl
#   affine_function_f1_f23_alpha_0p9_modde_results_processed.pkl
#   affine_function_f9_f12_alpha_0p9_modde_results_processed.pkl
#   affine_function_f10_f6_alpha_0p9_modde_results_processed.pkl
#   affine_function_f15_f8_alpha_0p9_modde_results_processed.pkl
#
# Output:
# output/new_function_task/task_H_individual_shap_affine_functions/
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIM = int(os.environ.get("AFFINE_FUNCTION_DIM", "5"))
ALPHA = float(os.environ.get("AFFINE_ALPHA", "0.9"))
ALPHA_TAG = str(ALPHA).replace(".", "p")

TARGET_PAIR_STR = os.environ.get(
    "AFFINE_TARGET_PAIRS",
    "3_6,1_23,9_12,10_6,15_8",
)

TARGET_PAIRS = []
for item in TARGET_PAIR_STR.split(","):
    item = item.strip()
    if not item:
        continue
    a, b = item.split("_")
    TARGET_PAIRS.append((int(a), int(b)))

INPUT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "modde_affine_function_results"
)

ASSIGNMENT_DIR_0307 = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "affine_function_group_assignment"
    / "0307_main"
)

ASSIGNMENT_DIR_INSTANCE = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "affine_function_group_assignment"
    / "eps2bins_4_r025050075_instance"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_affine_functions"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_SAMPLE_SIZE = int(os.environ.get("SHAP_SAMPLE_SIZE", "10000"))


# ============================================================
# Niels-style feature order and preprocessing
# ============================================================

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


# ============================================================
# Path helpers
# ============================================================

def make_function_tag(fid1, fid2):
    return f"f{fid1}_f{fid2}"


def make_function_label(fid1, fid2):
    return f"f{fid1}+f{fid2}"


def get_data_file(function_tag):
    return INPUT_DIR / f"affine_function_{function_tag}_alpha_{ALPHA_TAG}_modde_results_processed.pkl"


def get_assignment_summary_file_0307():
    return ASSIGNMENT_DIR_0307 / "affine_summary_assignment_0307_main.csv"


def get_assignment_summary_file_instance():
    return ASSIGNMENT_DIR_INSTANCE / "affine_summary_assignment_eps2bins_4_r025050075_instance.csv"


# ============================================================
# Assignment info
# ============================================================

def load_assignment_summary(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_assignment_info(function_tag, fid1, fid2):
    info = {
        "function_source": "quentin_affine_bbob",
        "function_tag": function_tag,
        "function_label": make_function_label(fid1, fid2),
        "fid1": fid1,
        "fid2": fid2,
        "alpha": ALPHA,
        "assigned_group_0307": "UNKNOWN",
        "assigned_group_instance": "UNKNOWN",
        "eps_ratio": None,
        "adj_r2": None,
        "group_counts_0307": None,
        "group_counts_instance": None,
    }

    df_0307 = load_assignment_summary(get_assignment_summary_file_0307())
    if df_0307 is not None:
        sub = df_0307[df_0307["function_tag"].astype(str) == function_tag]
        if len(sub) > 0:
            row = sub.iloc[0]
            info["assigned_group_0307"] = row.get("majority_group", "UNKNOWN")
            info["group_counts_0307"] = row.get("group_counts", None)
            info["eps_ratio"] = float(row.get("mean_eps_ratio"))
            info["adj_r2"] = float(row.get("mean_adj_r2"))
            info["eps_ratio_std"] = float(row.get("std_eps_ratio"))
            info["adj_r2_std"] = float(row.get("std_adj_r2"))

    df_instance = load_assignment_summary(get_assignment_summary_file_instance())
    if df_instance is not None:
        sub = df_instance[df_instance["function_tag"].astype(str) == function_tag]
        if len(sub) > 0:
            row = sub.iloc[0]
            info["assigned_group_instance"] = row.get("majority_group", "UNKNOWN")
            info["group_counts_instance"] = row.get("group_counts", None)

            if info["eps_ratio"] is None:
                info["eps_ratio"] = float(row.get("mean_eps_ratio"))
                info["adj_r2"] = float(row.get("mean_adj_r2"))
                info["eps_ratio_std"] = float(row.get("std_eps_ratio"))
                info["adj_r2_std"] = float(row.get("std_adj_r2"))

    return info


# ============================================================
# Data + SHAP
# ============================================================

def load_function_data(data_file):
    if not data_file.exists():
        raise FileNotFoundError(
            f"Missing modDE processed result file:\n{data_file}"
        )

    df = pd.read_pickle(data_file)

    if "dim" in df.columns:
        df = df[df["dim"] == DIM].copy()

    if len(df) == 0:
        raise RuntimeError("Loaded dataframe is empty after dim filtering.")

    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    required_cols = RAW_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[RAW_FEATURES].copy()
    X = X.rename(columns=RENAME_MAP)

    for col in CATEGORICAL_FEATURES:
        original_col = col
        X[col] = X[col].astype(str)
        X[col] = X[col].map(CATEGORY_MAPS[col])

        if X[col].isna().any():
            unknown_values = sorted(
                df[original_col].astype(str)[X[col].isna()].unique().tolist()
            )
            raise ValueError(
                f"Unknown category in {col}: {unknown_values}. "
                f"Please update CATEGORY_MAPS."
            )

        X[col] = X[col].astype(int)

    X = X[FEATURE_ORDER]
    y = df[TARGET_COL].copy()

    return X, y


def compute_shap(df: pd.DataFrame, sample_size: int, random_state: int = 42):
    df_used = df.copy()

    if len(df_used) > sample_size:
        print(f"Sampling from {len(df_used)} -> {sample_size}")
        df_used = df_used.sample(sample_size, random_state=random_state)

    X, y = prepare_features(df_used)

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

    return X, y, shap_values, train_r2


# ============================================================
# Plot helpers
# ============================================================

def shap_dot_on_axis(ax, shap_values, X, title=None, xlabel=None):
    plt.sca(ax)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(X.columns),
        max_display=len(FEATURE_ORDER),
        sort=False,
        show=False,
        color_bar=False,
        plot_size=None,
        cmap=plt.get_cmap("viridis"),
    )

    if title:
        ax.set_title(title, fontsize=14)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    else:
        ax.set_xlabel("")

    ax.tick_params(axis="both", labelsize=10)


def add_shared_colorbar(fig, cax):
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", rotation=90, fontsize=12)
    cbar.ax.tick_params(labelsize=11)


# ============================================================
# Save outputs
# ============================================================

def save_importance_table(
    X,
    shap_values,
    train_r2,
    assignment_info,
    function_tag,
    function_label,
):
    mean_abs = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)
    importance = mean_abs.sort_values(ascending=False).reset_index()
    importance.columns = ["feature", "mean_abs_shap"]

    total = importance["mean_abs_shap"].sum()
    if total > 0:
        importance["importance_norm"] = importance["mean_abs_shap"] / total
    else:
        importance["importance_norm"] = 0.0

    importance["function_source"] = "quentin_affine_bbob"
    importance["function_tag"] = function_tag
    importance["function_label"] = function_label
    importance["fid1"] = assignment_info.get("fid1")
    importance["fid2"] = assignment_info.get("fid2")
    importance["alpha"] = assignment_info.get("alpha")
    importance["assigned_group_0307"] = assignment_info.get("assigned_group_0307", "UNKNOWN")
    importance["assigned_group_instance"] = assignment_info.get("assigned_group_instance", "UNKNOWN")
    importance["eps_ratio"] = assignment_info.get("eps_ratio")
    importance["adj_r2"] = assignment_info.get("adj_r2")
    importance["train_r2"] = train_r2
    importance["n_samples_used"] = len(X)

    out_file = OUTPUT_DIR / f"affine_function_{function_tag}_alpha_{ALPHA_TAG}_individual_shap_importance.csv"
    importance.to_csv(out_file, index=False)

    print(f"Saved importance table: {out_file}")
    print(importance.to_string(index=False))

    return out_file, importance


def plot_individual_shap(
    X,
    shap_values,
    train_r2,
    n_total_rows,
    assignment_info,
    function_tag,
    function_label,
):
    assigned_group_0307 = assignment_info.get("assigned_group_0307", "UNKNOWN")
    assigned_group_instance = assignment_info.get("assigned_group_instance", "UNKNOWN")
    eps_ratio = assignment_info.get("eps_ratio")
    adj_r2 = assignment_info.get("adj_r2")

    fig = plt.figure(figsize=(8.8, 6.9))
    ax = fig.add_axes([0.16, 0.18, 0.68, 0.70])
    cax = fig.add_axes([0.88, 0.18, 0.025, 0.70])

    title = (
        f"Affine function {function_label}\n"
        f"0307: {assigned_group_0307} | instance fine: {assigned_group_instance}"
    )

    xlabel = f"Hyper-parameter contributions on affine {function_label} in $d=5$"

    shap_dot_on_axis(
        ax=ax,
        shap_values=shap_values,
        X=X,
        title=title,
        xlabel=xlabel,
    )

    add_shared_colorbar(fig, cax)

    bottom_text = (
        f"Affine function: {function_label} | alpha = {ALPHA} | "
        f"Train $R^2$ = {train_r2:.4f} | Rows = {n_total_rows}"
    )

    if eps_ratio is not None and adj_r2 is not None:
        bottom_text += f"\neps_ratio = {float(eps_ratio):.4f}, adj_r2 = {float(adj_r2):.4f}"

    group_counts_0307 = assignment_info.get("group_counts_0307")
    group_counts_instance = assignment_info.get("group_counts_instance")
    if group_counts_0307 is not None or group_counts_instance is not None:
        bottom_text += (
            f"\n0307 counts: {group_counts_0307} | "
            f"instance counts: {group_counts_instance}"
        )

    fig.text(
        0.16,
        0.045,
        bottom_text,
        fontsize=9.5,
        ha="left",
    )

    png_file = OUTPUT_DIR / f"affine_function_{function_tag}_alpha_{ALPHA_TAG}_individual_shap_beeswarm_d5.png"
    pdf_file = OUTPUT_DIR / f"affine_function_{function_tag}_alpha_{ALPHA_TAG}_individual_shap_beeswarm_d5.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {png_file}")
    print(f"Saved PDF:  {pdf_file}")

    return png_file, pdf_file


def process_one_function(fid1, fid2):
    function_tag = make_function_tag(fid1, fid2)
    function_label = make_function_label(fid1, fid2)
    data_file = get_data_file(function_tag)

    print("\n" + "=" * 80)
    print(f"NEW FUNCTION TASK H: INDIVIDUAL SHAP FOR AFFINE {function_label}")
    print("=" * 80)
    print(f"Input data: {data_file}")
    print(f"Output dir: {OUTPUT_DIR}")

    assignment_info = load_assignment_info(function_tag, fid1, fid2)

    print("Assignment info:")
    print(json.dumps(assignment_info, indent=2))

    df = load_function_data(data_file)
    print(f"Loaded dataframe shape: {df.shape}")

    X, y, shap_values, train_r2 = compute_shap(
        df,
        sample_size=FUNCTION_SAMPLE_SIZE,
        random_state=42,
    )

    print(f"Train R2: {train_r2:.4f}")

    importance_file, importance = save_importance_table(
        X=X,
        shap_values=shap_values,
        train_r2=train_r2,
        assignment_info=assignment_info,
        function_tag=function_tag,
        function_label=function_label,
    )

    plot_file, pdf_file = plot_individual_shap(
        X=X,
        shap_values=shap_values,
        train_r2=train_r2,
        n_total_rows=len(df),
        assignment_info=assignment_info,
        function_tag=function_tag,
        function_label=function_label,
    )

    summary = {
        "task": "new_function_task_H_individual_shap_for_quentin_affine_functions",
        "function_source": "quentin_affine_bbob",
        "function_tag": function_tag,
        "function_label": function_label,
        "fid1": fid1,
        "fid2": fid2,
        "alpha": ALPHA,
        "assigned_group_0307": assignment_info.get("assigned_group_0307", "UNKNOWN"),
        "assigned_group_instance": assignment_info.get("assigned_group_instance", "UNKNOWN"),
        "group_counts_0307": assignment_info.get("group_counts_0307"),
        "group_counts_instance": assignment_info.get("group_counts_instance"),
        "eps_ratio": assignment_info.get("eps_ratio"),
        "adj_r2": assignment_info.get("adj_r2"),
        "eps_ratio_std": assignment_info.get("eps_ratio_std"),
        "adj_r2_std": assignment_info.get("adj_r2_std"),
        "input_data": str(data_file),
        "n_total_rows": len(df),
        "n_samples_used_for_shap": len(X),
        "train_r2": train_r2,
        "importance_file": str(importance_file),
        "plot_file": str(plot_file),
        "pdf_file": str(pdf_file),
        "next_step": "Compare this individual affine-function SHAP importance with the assigned group-level SHAP importance.",
    }

    summary_file = OUTPUT_DIR / f"affine_function_{function_tag}_alpha_{ALPHA_TAG}_individual_shap_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_file}")

    return summary


def main():
    print("=" * 80)
    print("NEW FUNCTION TASK H: BATCH INDIVIDUAL SHAP FOR QUENTIN AFFINE FUNCTIONS")
    print("=" * 80)
    print(f"TARGET_PAIRS: {TARGET_PAIRS}")
    print(f"ALPHA: {ALPHA}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"SHAP sample size: {FUNCTION_SAMPLE_SIZE}")

    summaries = []

    for fid1, fid2 in TARGET_PAIRS:
        summary = process_one_function(fid1, fid2)
        summaries.append(summary)

    batch_name = "_".join([make_function_tag(a, b) for a, b in TARGET_PAIRS])
    batch_summary_file = (
        OUTPUT_DIR
        / f"affine_functions_{batch_name}_alpha_{ALPHA_TAG}_individual_shap_batch_summary.json"
    )

    batch_summary_file.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("TASK H BATCH DONE")
    print("=" * 80)
    print(f"Batch summary saved to: {batch_summary_file}")

    for s in summaries:
        print(f"{s['function_label']} -> {s['plot_file']}")


if __name__ == "__main__":
    main()