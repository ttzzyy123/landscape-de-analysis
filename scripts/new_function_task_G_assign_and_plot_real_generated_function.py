# scripts/new_function_task_G_assign_and_plot_real_generated_function.py

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# New Function Task G
# Batch assign and plot real generated functions against BBOB groups
#
# Goal:
# 1. Load original BBOB dim=5 feature data
# 2. Extract the two thesis features:
#      - ic.eps.ratio
#      - ela_meta.lin_simple.adj_r2
# 3. Aggregate BBOB instances to one point per BBOB function
# 4. Load Task F batch ELA summary for generated functions n1/n3
# 5. Assign each generated function to existing eps2bins_3_r025065 groups
# 6. Plot:
#      - one combined plot: BBOB + all generated functions
#      - one individual plot per generated function
# ============================================================


PROJECT_ROOT = Path("/data/s3795888/ioh_project/my_landscape_experiments")

BBOB_FEATURE_FILE = PROJECT_ROOT / "data" / "features_summary_dim_5_sobol.csv"

REAL_FEATURE_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "real_function_features"
)

BATCH_FEATURE_SUMMARY_CSV = (
    REAL_FEATURE_DIR
    / "real_generated_functions_batch_ela_features_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_G_assign_and_plot_real_generated_function"
)

INTERMEDIATE_OUT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "real_function_group_assignment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Scheme settings
# ============================================================

SCHEME_NAME = "eps2bins_3_r025065"
ASSIGNMENT_STRATEGY = "ambiguous_on_tie"

EPS_SPLIT = 3.0
ADJ_SPLITS = [0.25, 0.65]

TARGET_FEATURES = [
    "ic.eps.ratio",
    "ela_meta.lin_simple.adj_r2",
]

FUNCTION_TO_GROUP = {
    1: "G3",
    2: "G7",
    3: "G2",
    4: "G2",
    5: "G3",
    6: "G7",
    7: "G0",
    8: "G6",
    9: "G5",
    10: "G7",
    11: "G4",
    12: "G6",
    13: "G3",
    14: "G2",
    15: "G2",
    16: "G1",
    17: "G2",
    18: "G2",
    19: "G1",
    20: "G7",
    21: "G1",
    22: "G1",
    23: "G1",
    24: "G1",
}

GROUP_ORDER = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]

GROUP_COLORS = {
    "G0": "tab:blue",
    "G1": "tab:orange",
    "G2": "tab:green",
    "G3": "tab:red",
    "G4": "tab:purple",
    "G5": "tab:brown",
    "G6": "tab:pink",
    "G7": "tab:gray",
}

NEW_MARKERS = {
    "n1": "*",
    "n2": "P",
    "n3": "X",
    "single": "*",
}

SPECIAL_BBOB_LABELS = {
    7: "f7*",
    11: "f11*",
}


# ============================================================
# Utilities
# ============================================================


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def load_bbob_feature_points(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. LOAD BBOB FEATURE DATA")
    write_line(lines, "=" * 80)

    if not BBOB_FEATURE_FILE.exists():
        raise FileNotFoundError(f"BBOB feature file not found: {BBOB_FEATURE_FILE}")

    write_line(lines, f"Loading: {BBOB_FEATURE_FILE}")
    df = pd.read_csv(BBOB_FEATURE_FILE, sep=";")
    df.columns = [c.strip() for c in df.columns]

    required_columns = ["Feature name", "# samples", "Function", "Instance", "Value"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in BBOB feature file: {missing}")

    for col in ["Feature name", "# samples", "Function", "Instance"]:
        df[col] = df[col].astype(str).str.strip()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    write_line(lines, f"Original shape: {df.shape}")

    df = df[df["# samples"] == "5000"]
    write_line(lines, f"After filtering # samples == 5000: {df.shape}")

    df = df[df["Feature name"].isin(TARGET_FEATURES)]
    write_line(lines, f"After filtering target features: {df.shape}")

    df_wide = df.pivot_table(
        index=["Function", "Instance"],
        columns="Feature name",
        values="Value",
    ).reset_index()

    df_wide.columns.name = None

    df_wide = df_wide.rename(
        columns={
            "ic.eps.ratio": "eps_ratio",
            "ela_meta.lin_simple.adj_r2": "adj_r2",
        }
    )

    df_wide["Function"] = pd.to_numeric(df_wide["Function"], errors="coerce").astype(int)
    df_wide["Instance"] = pd.to_numeric(df_wide["Instance"], errors="coerce").astype(int)

    write_line(lines, f"Instance-level wide shape: {df_wide.shape}")

    df_func = (
        df_wide.groupby("Function", as_index=False)
        .agg(
            eps_ratio=("eps_ratio", "mean"),
            adj_r2=("adj_r2", "mean"),
            eps_ratio_std=("eps_ratio", "std"),
            adj_r2_std=("adj_r2", "std"),
            n_instances=("Instance", "nunique"),
        )
    )

    df_func["label"] = df_func["Function"].apply(
        lambda f: SPECIAL_BBOB_LABELS.get(int(f), f"f{int(f)}")
    )
    df_func["group"] = df_func["Function"].map(FUNCTION_TO_GROUP)

    if df_func["group"].isna().any():
        missing_f = df_func[df_func["group"].isna()]["Function"].tolist()
        raise RuntimeError(f"Some BBOB functions have no group assignment: {missing_f}")

    write_line(lines, "BBOB function-level points:")
    write_line(lines, df_func[["Function", "eps_ratio", "adj_r2", "group"]].to_string(index=False))

    return df_wide, df_func


def load_generated_feature_points(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. LOAD GENERATED FUNCTION FEATURE SUMMARY")
    write_line(lines, "=" * 80)

    if not BATCH_FEATURE_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Batch feature summary CSV not found: {BATCH_FEATURE_SUMMARY_CSV}"
        )

    write_line(lines, f"Loading: {BATCH_FEATURE_SUMMARY_CSV}")
    df = pd.read_csv(BATCH_FEATURE_SUMMARY_CSV)

    required_columns = [
        "status",
        "run_id",
        "function_tag",
        "full_id",
        "function_file",
        "summary_file",
        "ic.eps.ratio_mean",
        "ela_meta.lin_simple.adj_r2_mean",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in batch summary CSV: {missing}")

    df = df[df["status"] == "success"].copy()

    if df.empty:
        raise RuntimeError("No successful generated functions found in batch summary CSV.")

    df["eps_ratio"] = pd.to_numeric(df["ic.eps.ratio_mean"], errors="coerce")
    df["adj_r2"] = pd.to_numeric(df["ela_meta.lin_simple.adj_r2_mean"], errors="coerce")
    df["eps_ratio_std"] = pd.to_numeric(df.get("ic.eps.ratio_std", np.nan), errors="coerce")
    df["adj_r2_std"] = pd.to_numeric(df.get("ela_meta.lin_simple.adj_r2_std", np.nan), errors="coerce")

    df["label"] = df["function_tag"].astype(str)
    df["new_function_label"] = "new_" + df["function_tag"].astype(str)

    df = df.sort_values(["run_id", "function_tag"]).reset_index(drop=True)

    write_line(lines, f"Loaded {len(df)} successful generated function(s):")
    write_line(
        lines,
        df[
            [
                "full_id",
                "function_tag",
                "eps_ratio",
                "adj_r2",
                "function_file",
            ]
        ].to_string(index=False),
    )

    return df


def assign_point_to_group(eps, adj):
    if eps < EPS_SPLIT and adj < ADJ_SPLITS[0]:
        return "G1", "eps_ratio < 3.0 and adj_r2 < 0.25"

    if eps < EPS_SPLIT and ADJ_SPLITS[0] <= adj < ADJ_SPLITS[1]:
        return "G2", "eps_ratio < 3.0 and 0.25 <= adj_r2 < 0.65"

    if eps < EPS_SPLIT and adj >= ADJ_SPLITS[1]:
        return "G3", "eps_ratio < 3.0 and adj_r2 >= 0.65"

    if eps >= EPS_SPLIT and adj < ADJ_SPLITS[0]:
        return "G5", "eps_ratio >= 3.0 and adj_r2 < 0.25"

    if eps >= EPS_SPLIT and ADJ_SPLITS[0] <= adj < ADJ_SPLITS[1]:
        return "right_middle_bin_needs_manual_rule", "eps_ratio >= 3.0 and 0.25 <= adj_r2 < 0.65"

    return "G7", "eps_ratio >= 3.0 and adj_r2 >= 0.65"


def assign_generated_functions_to_groups(df_new, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. ASSIGN GENERATED FUNCTIONS TO EXISTING BIN/GROUP")
    write_line(lines, "=" * 80)

    assignments = []

    for _, row in df_new.iterrows():
        assigned_group, bin_description = assign_point_to_group(
            float(row["eps_ratio"]),
            float(row["adj_r2"]),
        )

        assignment = {
            "scheme": SCHEME_NAME,
            "assignment_strategy": ASSIGNMENT_STRATEGY,
            "new_function_label": row["new_function_label"],
            "function_tag": row["function_tag"],
            "run_id": row["run_id"],
            "full_id": row["full_id"],
            "function_file": row["function_file"],
            "summary_file": row["summary_file"],
            "eps_ratio": float(row["eps_ratio"]),
            "adj_r2": float(row["adj_r2"]),
            "eps_ratio_std": float(row["eps_ratio_std"]) if not pd.isna(row["eps_ratio_std"]) else None,
            "adj_r2_std": float(row["adj_r2_std"]) if not pd.isna(row["adj_r2_std"]) else None,
            "assigned_group": assigned_group,
            "bin_description": bin_description,
            "note": (
                "The generated function is assigned according to the same visible "
                "eps2bins_3_r025065 boundaries used for the BBOB landscape grouping."
            ),
        }

        assignments.append(assignment)

        write_line(lines, "")
        write_line(lines, f"Assignment for {row['full_id']}:")
        write_line(lines, json.dumps(assignment, indent=2))

    df_assign = pd.DataFrame(assignments)

    return df_assign, assignments


def save_assignment_outputs(df_instance, df_func, df_new, df_assign, assignments, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "4. SAVE ASSIGNMENT TABLES")
    write_line(lines, "=" * 80)

    bbob_instance_file = INTERMEDIATE_OUT_DIR / "bbob_dim5_selected_features_by_instance.csv"
    bbob_function_file = INTERMEDIATE_OUT_DIR / "bbob_dim5_selected_features_by_function_mean.csv"

    batch_assignment_csv = INTERMEDIATE_OUT_DIR / "real_generated_functions_group_assignment_batch.csv"
    batch_assignment_json = INTERMEDIATE_OUT_DIR / "real_generated_functions_group_assignment_batch.json"
    batch_points_csv = INTERMEDIATE_OUT_DIR / "real_generated_functions_feature_points_batch.csv"

    df_instance.to_csv(bbob_instance_file, index=False)
    df_func.to_csv(bbob_function_file, index=False)

    df_new_with_group = df_new.merge(
        df_assign[["full_id", "assigned_group", "bin_description"]],
        on="full_id",
        how="left",
    )
    df_new_with_group.to_csv(batch_points_csv, index=False)

    df_assign.to_csv(batch_assignment_csv, index=False)
    batch_assignment_json.write_text(json.dumps(assignments, indent=2), encoding="utf-8")

    write_line(lines, f"[OK] BBOB instance table saved to: {bbob_instance_file}")
    write_line(lines, f"[OK] BBOB function table saved to: {bbob_function_file}")
    write_line(lines, f"[OK] Batch generated points saved to: {batch_points_csv}")
    write_line(lines, f"[OK] Batch assignment CSV saved to: {batch_assignment_csv}")
    write_line(lines, f"[OK] Batch assignment JSON saved to: {batch_assignment_json}")

    for assignment in assignments:
        full_id = assignment["full_id"]
        single_assignment_file = (
            INTERMEDIATE_OUT_DIR
            / f"real_generated_function_{full_id}_group_assignment.json"
        )
        single_point_file = (
            INTERMEDIATE_OUT_DIR
            / f"real_generated_function_{full_id}_feature_point.csv"
        )

        single_assignment_file.write_text(
            json.dumps(assignment, indent=2),
            encoding="utf-8",
        )

        pd.DataFrame([assignment]).to_csv(single_point_file, index=False)

        write_line(lines, f"[OK] Single assignment saved to: {single_assignment_file}")
        write_line(lines, f"[OK] Single point saved to: {single_point_file}")

    return {
        "bbob_instance_file": str(bbob_instance_file),
        "bbob_function_file": str(bbob_function_file),
        "batch_points_csv": str(batch_points_csv),
        "batch_assignment_csv": str(batch_assignment_csv),
        "batch_assignment_json": str(batch_assignment_json),
    }


def add_bbob_points(ax, df_func):
    for group in GROUP_ORDER:
        sub = df_func[df_func["group"] == group]

        ax.scatter(
            sub["eps_ratio"],
            sub["adj_r2"],
            s=90,
            color=GROUP_COLORS[group],
            label=group,
            alpha=0.95,
        )

        for _, row in sub.iterrows():
            ax.text(
                row["eps_ratio"] + 0.04,
                row["adj_r2"] + 0.008,
                row["label"],
                fontsize=11,
            )


def add_bin_boundaries(ax):
    ax.axvline(EPS_SPLIT, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(ADJ_SPLITS[0], color="black", linestyle="--", linewidth=1.5)
    ax.axhline(ADJ_SPLITS[1], color="black", linestyle="--", linewidth=1.5)


def finalize_axes(ax, df_func, df_new, title):
    ax.set_xlabel("eps_ratio", fontsize=14)
    ax.set_ylabel("adj_r2", fontsize=14)
    ax.set_title(title, fontsize=16)

    xmin = min(0.0, df_func["eps_ratio"].min(), df_new["eps_ratio"].min()) - 0.1
    xmax = max(7.7, df_func["eps_ratio"].max(), df_new["eps_ratio"].max()) + 0.25
    ymin = min(-0.05, df_func["adj_r2"].min(), df_new["adj_r2"].min()) - 0.02
    ymax = max(1.05, df_func["adj_r2"].max(), df_new["adj_r2"].max()) + 0.02

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.tick_params(axis="both", labelsize=12)
    ax.legend(
        title="Group / Generated",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=11,
        title_fontsize=12,
    )
    ax.grid(False)


def plot_combined_generated_functions(df_func, df_new, df_assign, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. PLOT COMBINED GENERATED FUNCTIONS")
    write_line(lines, "=" * 80)

    df_plot = df_new.merge(
        df_assign[["full_id", "assigned_group"]],
        on="full_id",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.6))

    add_bbob_points(ax, df_func)

    for _, row in df_plot.iterrows():
        marker = NEW_MARKERS.get(str(row["function_tag"]), "D")

        ax.scatter(
            [row["eps_ratio"]],
            [row["adj_r2"]],
            s=280,
            color="black",
            marker=marker,
            label=f"{row['new_function_label']} ({row['assigned_group']})",
            edgecolors="white",
            linewidths=1.2,
            zorder=10,
        )

        ax.text(
            row["eps_ratio"] + 0.06,
            row["adj_r2"] + 0.018,
            row["new_function_label"],
            fontsize=12,
            fontweight="bold",
            color="black",
            zorder=11,
        )

    add_bin_boundaries(ax)

    title = (
        f"{SCHEME_NAME}\n"
        f"Assignment: {ASSIGNMENT_STRATEGY}\n"
        f"with real generated functions"
    )

    finalize_axes(ax, df_func, df_plot, title)
    fig.tight_layout()

    plot_file = OUTPUT_DIR / "real_generated_functions_batch_with_bbob_groups_plot.png"
    pdf_file = OUTPUT_DIR / "real_generated_functions_batch_with_bbob_groups_plot.pdf"

    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    write_line(lines, f"[OK] Combined plot saved to: {plot_file}")
    write_line(lines, f"[OK] Combined PDF saved to: {pdf_file}")

    return plot_file, pdf_file


def plot_individual_generated_functions(df_func, df_new, df_assign, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. PLOT INDIVIDUAL GENERATED FUNCTIONS")
    write_line(lines, "=" * 80)

    df_plot = df_new.merge(
        df_assign[["full_id", "assigned_group"]],
        on="full_id",
        how="left",
    )

    outputs = []

    for _, row in df_plot.iterrows():
        fig, ax = plt.subplots(figsize=(10.5, 7.5))

        add_bbob_points(ax, df_func)

        marker = NEW_MARKERS.get(str(row["function_tag"]), "*")

        ax.scatter(
            [row["eps_ratio"]],
            [row["adj_r2"]],
            s=280,
            color="black",
            marker=marker,
            label=f"{row['new_function_label']} ({row['assigned_group']})",
            edgecolors="white",
            linewidths=1.2,
            zorder=10,
        )

        ax.text(
            row["eps_ratio"] + 0.06,
            row["adj_r2"] + 0.018,
            row["new_function_label"],
            fontsize=12,
            fontweight="bold",
            color="black",
            zorder=11,
        )

        add_bin_boundaries(ax)

        single_df = pd.DataFrame([row])

        title = (
            f"{SCHEME_NAME}\n"
            f"Assignment: {ASSIGNMENT_STRATEGY}\n"
            f"with {row['new_function_label']}"
        )

        finalize_axes(ax, df_func, single_df, title)
        fig.tight_layout()

        full_id = row["full_id"]
        plot_file = OUTPUT_DIR / f"real_generated_function_{full_id}_with_bbob_groups_plot.png"
        pdf_file = OUTPUT_DIR / f"real_generated_function_{full_id}_with_bbob_groups_plot.pdf"

        fig.savefig(plot_file, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_file, bbox_inches="tight")
        plt.close(fig)

        write_line(lines, f"[OK] Individual plot saved to: {plot_file}")
        write_line(lines, f"[OK] Individual PDF saved to: {pdf_file}")

        outputs.append(
            {
                "full_id": full_id,
                "plot_file": str(plot_file),
                "pdf_file": str(pdf_file),
            }
        )

    return outputs


def main():
    lines = []

    summary_file = OUTPUT_DIR / "task_G_assign_and_plot_real_generated_function_summary.txt"

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK G: BATCH ASSIGN AND PLOT REAL GENERATED FUNCTIONS")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root: {PROJECT_ROOT}")
    write_line(lines, f"BBOB feature file: {BBOB_FEATURE_FILE}")
    write_line(lines, f"Real feature dir: {REAL_FEATURE_DIR}")
    write_line(lines, f"Batch feature CSV: {BATCH_FEATURE_SUMMARY_CSV}")
    write_line(lines, f"Output dir: {OUTPUT_DIR}")
    write_line(lines, f"Scheme: {SCHEME_NAME}")
    write_line(lines, f"Assignment strategy: {ASSIGNMENT_STRATEGY}")

    try:
        df_instance, df_func = load_bbob_feature_points(lines)
        df_new = load_generated_feature_points(lines)

        df_assign, assignments = assign_generated_functions_to_groups(df_new, lines)

        saved_files = save_assignment_outputs(
            df_instance=df_instance,
            df_func=df_func,
            df_new=df_new,
            df_assign=df_assign,
            assignments=assignments,
            lines=lines,
        )

        combined_plot, combined_pdf = plot_combined_generated_functions(
            df_func=df_func,
            df_new=df_new,
            df_assign=df_assign,
            lines=lines,
        )

        individual_outputs = plot_individual_generated_functions(
            df_func=df_func,
            df_new=df_new,
            df_assign=df_assign,
            lines=lines,
        )

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK G CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, "[SUCCESS] Generated functions assigned and plotted.")

        for assignment in assignments:
            write_line(
                lines,
                (
                    f"{assignment['new_function_label']} "
                    f"({assignment['full_id']}) -> {assignment['assigned_group']} "
                    f"| eps_ratio={assignment['eps_ratio']} "
                    f"| adj_r2={assignment['adj_r2']}"
                ),
            )

        write_line(lines, "")
        write_line(lines, f"Combined plot: {combined_plot}")
        write_line(lines, f"Combined PDF:  {combined_pdf}")
        write_line(lines, "")
        write_line(lines, "Saved assignment files:")
        for k, v in saved_files.items():
            write_line(lines, f"  {k}: {v}")

        write_line(lines, "")
        write_line(lines, "Individual plots:")
        for item in individual_outputs:
            write_line(lines, f"  {item['full_id']}: {item['plot_file']}")

        write_line(lines, "")
        write_line(lines, "Next step: select representative generated function(s) for modDE/H SHAP analysis.")

    except Exception as e:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK G FAILED")
        write_line(lines, "=" * 80)
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise

    finally:
        summary_file.write_text("\n".join(lines), encoding="utf-8")
        print("\nSummary written to:")
        print(summary_file)


if __name__ == "__main__":
    main()