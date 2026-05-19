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
# Batch individual SHAP analysis for selected real generated functions
#
# Input:
# intermediate/new_function_task/modde_real_function_results/
#   real_generated_function_20260517_125137_n3_modde_results_processed.pkl
#   real_generated_function_20260517_125137_n7_modde_results_processed.pkl
#   real_generated_function_20260517_125137_n5_modde_results_processed.pkl
#
# Output:
# output/new_function_task/task_H_individual_shap_real_generated_function/
#   one importance CSV + beeswarm PNG/PDF + summary JSON per function
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_RUN_ID = os.environ.get("REAL_FUNCTION_BASE_RUN_ID", "20260517_125137")

TARGET_FUNCTION_TAGS = os.environ.get("REAL_FUNCTION_TAGS", "n3,n7,n5").split(",")
TARGET_FUNCTION_TAGS = [x.strip() for x in TARGET_FUNCTION_TAGS if x.strip()]

DIM = 5

INPUT_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "modde_real_function_results"

ASSIGNMENT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "real_function_group_assignment"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_real_generated_function"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_SAMPLE_SIZE = 10000


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


def full_id_from_tag(function_tag):
    return f"{BASE_RUN_ID}_{function_tag}"


def get_data_file(full_id):
    return INPUT_DIR / f"real_generated_function_{full_id}_modde_results_processed.pkl"


def get_assignment_file(full_id):
    return ASSIGNMENT_DIR / f"real_generated_function_{full_id}_group_assignment.json"


def load_assignment_info(full_id, function_tag):
    assignment_file = get_assignment_file(full_id)

    if assignment_file.exists():
        with assignment_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback based on current Task G result.
    fallback_group = {
        "n3": "G2",
        "n5": "G1",
        "n7": "G1",
    }.get(function_tag, "UNKNOWN")

    return {
        "assigned_group": fallback_group,
        "eps_ratio": None,
        "adj_r2": None,
        "new_function_label": f"new_{function_tag}",
        "function_tag": function_tag,
        "full_id": full_id,
    }


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


def save_importance_table(
    X,
    shap_values,
    train_r2,
    assignment_info,
    full_id,
    function_tag,
    function_label,
):
    mean_abs = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)
    importance = mean_abs.sort_values(ascending=False).reset_index()
    importance.columns = ["feature", "mean_abs_shap"]
    importance["importance_norm"] = (
        importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    )

    importance["base_run_id"] = BASE_RUN_ID
    importance["full_id"] = full_id
    importance["function_tag"] = function_tag
    importance["function_label"] = function_label
    importance["assigned_group"] = assignment_info.get("assigned_group", "UNKNOWN")
    importance["train_r2"] = train_r2
    importance["n_samples_used"] = len(X)

    out_file = OUTPUT_DIR / f"real_generated_function_{full_id}_individual_shap_importance.csv"
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
    full_id,
    function_tag,
    function_label,
):
    assigned_group = assignment_info.get("assigned_group", "UNKNOWN")
    eps_ratio = assignment_info.get("eps_ratio")
    adj_r2 = assignment_info.get("adj_r2")

    fig = plt.figure(figsize=(8.8, 6.9))
    ax = fig.add_axes([0.16, 0.18, 0.68, 0.70])
    cax = fig.add_axes([0.88, 0.18, 0.025, 0.70])

    title = f"{function_label}: real generated function ({assigned_group})"
    xlabel = f"Hyper-parameter contributions on {function_label} in $d=5$"

    shap_dot_on_axis(
        ax=ax,
        shap_values=shap_values,
        X=X,
        title=title,
        xlabel=xlabel,
    )

    add_shared_colorbar(fig, cax)

    bottom_text = (
        f"Real generated function: {function_label} | Assigned group: {assigned_group} | "
        f"Train $R^2$ = {train_r2:.4f} | Rows = {n_total_rows}"
    )

    if eps_ratio is not None and adj_r2 is not None:
        bottom_text += f"\neps_ratio = {float(eps_ratio):.4f}, adj_r2 = {float(adj_r2):.4f}"

    fig.text(
        0.16,
        0.065,
        bottom_text,
        fontsize=10,
        ha="left",
    )

    png_file = OUTPUT_DIR / f"real_generated_function_{full_id}_individual_shap_beeswarm_d5.png"
    pdf_file = OUTPUT_DIR / f"real_generated_function_{full_id}_individual_shap_beeswarm_d5.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {png_file}")
    print(f"Saved PDF:  {pdf_file}")

    return png_file, pdf_file


def process_one_function(function_tag):
    full_id = full_id_from_tag(function_tag)
    data_file = get_data_file(full_id)

    function_label = f"new_{function_tag}"

    print("\n" + "=" * 80)
    print(f"NEW FUNCTION TASK H: INDIVIDUAL SHAP FOR {function_label}")
    print("=" * 80)
    print(f"Input data: {data_file}")
    print(f"Output dir: {OUTPUT_DIR}")

    assignment_info = load_assignment_info(full_id, function_tag)

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
        full_id=full_id,
        function_tag=function_tag,
        function_label=function_label,
    )

    plot_file, pdf_file = plot_individual_shap(
        X=X,
        shap_values=shap_values,
        train_r2=train_r2,
        n_total_rows=len(df),
        assignment_info=assignment_info,
        full_id=full_id,
        function_tag=function_tag,
        function_label=function_label,
    )

    summary = {
        "task": "new_function_task_H_individual_shap_for_real_generated_function_batch",
        "base_run_id": BASE_RUN_ID,
        "full_id": full_id,
        "function_tag": function_tag,
        "function_label": function_label,
        "assigned_group": assignment_info.get("assigned_group", "UNKNOWN"),
        "eps_ratio": assignment_info.get("eps_ratio"),
        "adj_r2": assignment_info.get("adj_r2"),
        "input_data": str(data_file),
        "n_total_rows": len(df),
        "n_samples_used_for_shap": len(X),
        "train_r2": train_r2,
        "importance_file": str(importance_file),
        "plot_file": str(plot_file),
        "pdf_file": str(pdf_file),
        "next_step": "Compare this individual SHAP importance with the assigned group-level SHAP importance.",
    }

    summary_file = OUTPUT_DIR / f"real_generated_function_{full_id}_individual_shap_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_file}")

    return summary


def main():
    print("=" * 80)
    print("NEW FUNCTION TASK H: BATCH INDIVIDUAL SHAP FOR REAL GENERATED FUNCTIONS")
    print("=" * 80)
    print(f"BASE_RUN_ID: {BASE_RUN_ID}")
    print(f"TARGET_FUNCTION_TAGS: {TARGET_FUNCTION_TAGS}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    summaries = []

    for function_tag in TARGET_FUNCTION_TAGS:
        summary = process_one_function(function_tag)
        summaries.append(summary)

    batch_summary_file = (
        OUTPUT_DIR
        / f"real_generated_function_{BASE_RUN_ID}_{'_'.join(TARGET_FUNCTION_TAGS)}_individual_shap_batch_summary.json"
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