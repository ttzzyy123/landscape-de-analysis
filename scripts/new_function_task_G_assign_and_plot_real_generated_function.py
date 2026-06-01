# scripts/new_function_task_G_assign_and_plot_real_generated_function.py

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Task G for Quentin affine BBOB functions
# Assign and plot affine-combined functions into ELA grouping schemes
# ============================================================

PROJECT_ROOT = Path("/data/s3795888/ioh_project/my_landscape_experiments")

BBOB_FEATURE_FILE = PROJECT_ROOT / "data" / "features_summary_dim_5_sobol.csv"

AFFINE_FEATURE_CSV = (
    PROJECT_ROOT
    / "data"
    / "BBOB_features_generalisation_alpha_tony.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_G_assign_and_plot_affine_functions"
)

INTERMEDIATE_OUT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "affine_function_group_assignment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Feature names
# ============================================================

TARGET_FEATURES = [
    "ic.eps.ratio",
    "ela_meta.lin_simple.adj_r2",
]

EPS_COL = "eps_ratio"
R2_COL = "adj_r2"

AFFINE_EPS_CANDIDATES = [
    "ic.eps_ratio",
    "ic.eps.ratio",
    "eps_ratio",
]

AFFINE_R2_CANDIDATES = [
    "ela_meta.lin_simple.adj_r2",
    "adj_r2",
]


# ============================================================
# Grouping schemes
# ============================================================

SCHEMES = {
    "0307_main": {
        "scheme_note": "Main function-level scheme used in the thesis: eps split 3.0, adj_r2 splits 0.3 and 0.7.",
        "eps_bins": [-np.inf, 3.0, np.inf],
        "r2_bins": [-np.inf, 0.3, 0.7, np.inf],
    },
    "eps2bins_4_r025050075_instance": {
        "scheme_note": "Instance-level fine-grained scheme: eps split 3.0, adj_r2 splits 0.25, 0.50 and 0.75.",
        "eps_bins": [-np.inf, 3.0, np.inf],
        "r2_bins": [-np.inf, 0.25, 0.50, 0.75, np.inf],
    },
}

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

SPECIAL_BBOB_LABELS = {
    7: "f7*",
    11: "f11*",
}


# ============================================================
# Utilities
# ============================================================

def write_line(lines, text=""):
    print(text)
    lines.append(str(text))


def read_csv_auto(path):
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) <= 1:
            df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_existing_column(df, candidates, file_name):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find any of columns {candidates} in {file_name}. "
        f"Available columns: {list(df.columns)}"
    )


def make_group_label(eps_bin, r2_bin, n_r2_bins):
    group_id = int(eps_bin) * int(n_r2_bins) + int(r2_bin)
    return f"G{group_id}"


def assign_grid_groups(df, eps_bins, r2_bins):
    out = df.copy()

    out["eps_bin"] = pd.cut(
        out[EPS_COL],
        bins=eps_bins,
        labels=False,
        include_lowest=True,
    )

    out["r2_bin"] = pd.cut(
        out[R2_COL],
        bins=r2_bins,
        labels=False,
        include_lowest=True,
    )

    if out["eps_bin"].isna().any() or out["r2_bin"].isna().any():
        bad = out[out["eps_bin"].isna() | out["r2_bin"].isna()]
        raise RuntimeError(
            "Some points could not be assigned to bins:\n"
            + bad[[EPS_COL, R2_COL]].to_string(index=False)
        )

    n_r2_bins = len(r2_bins) - 1

    out["group_id"] = (
        out["eps_bin"].astype(int) * n_r2_bins
        + out["r2_bin"].astype(int)
    )

    out["group"] = "G" + out["group_id"].astype(int).astype(str)

    return out


def group_sort_key(group_name):
    return int(str(group_name).replace("G", ""))


# ============================================================
# Load BBOB features
# ============================================================

def load_bbob_instance_feature_points(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. LOAD BBOB INSTANCE-LEVEL FEATURE DATA")
    write_line(lines, "=" * 80)

    if not BBOB_FEATURE_FILE.exists():
        raise FileNotFoundError(f"BBOB feature file not found: {BBOB_FEATURE_FILE}")

    write_line(lines, f"Loading: {BBOB_FEATURE_FILE}")
    df = read_csv_auto(BBOB_FEATURE_FILE)

    required_columns = ["Feature name", "# samples", "Function", "Instance", "Value"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in BBOB feature file: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    for col in ["Feature name", "# samples", "Function", "Instance"]:
        df[col] = df[col].astype(str).str.strip()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    write_line(lines, f"Original shape: {df.shape}")

    df = df[df["# samples"] == "5000"].copy()
    write_line(lines, f"After filtering # samples == 5000: {df.shape}")

    df = df[df["Feature name"].isin(TARGET_FEATURES)].copy()
    write_line(lines, f"After filtering target features: {df.shape}")

    df_wide = (
        df.pivot_table(
            index=["Function", "Instance"],
            columns="Feature name",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
    )

    df_wide.columns.name = None

    df_wide = df_wide.rename(
        columns={
            "ic.eps.ratio": EPS_COL,
            "ela_meta.lin_simple.adj_r2": R2_COL,
        }
    )

    df_wide["Function"] = pd.to_numeric(df_wide["Function"], errors="coerce").astype(int)
    df_wide["Instance"] = pd.to_numeric(df_wide["Instance"], errors="coerce").astype(int)

    df_wide["label"] = (
        "f"
        + df_wide["Function"].astype(str)
        + "_i"
        + df_wide["Instance"].astype(str)
    )

    df_wide = df_wide[["Function", "Instance", EPS_COL, R2_COL, "label"]].copy()

    write_line(lines, f"Instance-level wide shape: {df_wide.shape}")
    write_line(lines, "BBOB instance-level feature preview:")
    write_line(lines, df_wide.head(10).to_string(index=False))

    return df_wide


# ============================================================
# Load affine features from Quentin
# ============================================================

def load_affine_feature_points(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. LOAD QUENTIN AFFINE FUNCTION FEATURE DATA")
    write_line(lines, "=" * 80)

    if not AFFINE_FEATURE_CSV.exists():
        raise FileNotFoundError(f"Affine feature CSV not found: {AFFINE_FEATURE_CSV}")

    write_line(lines, f"Loading: {AFFINE_FEATURE_CSV}")
    df = read_csv_auto(AFFINE_FEATURE_CSV)

    eps_source_col = find_first_existing_column(
        df,
        AFFINE_EPS_CANDIDATES,
        "affine feature CSV",
    )

    r2_source_col = find_first_existing_column(
        df,
        AFFINE_R2_CANDIDATES,
        "affine feature CSV",
    )

    rename_map = {
        eps_source_col: EPS_COL,
        r2_source_col: R2_COL,
    }

    df = df.rename(columns=rename_map)

    required_base = ["fid1", "fid2", "alpha", EPS_COL, R2_COL]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in affine feature CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "rid" not in df.columns:
        df["rid"] = df.groupby(["fid1", "fid2", "alpha"]).cumcount() + 1

    df["fid1"] = pd.to_numeric(df["fid1"], errors="coerce").astype(int)
    df["fid2"] = pd.to_numeric(df["fid2"], errors="coerce").astype(int)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["rid"] = pd.to_numeric(df["rid"], errors="coerce").astype(int)
    df[EPS_COL] = pd.to_numeric(df[EPS_COL], errors="coerce")
    df[R2_COL] = pd.to_numeric(df[R2_COL], errors="coerce")

    df = df.dropna(subset=[EPS_COL, R2_COL]).copy()

    df["function_tag"] = (
        "f"
        + df["fid1"].astype(str)
        + "_f"
        + df["fid2"].astype(str)
    )

    df["function_name"] = (
        "affine_"
        + df["function_tag"]
        + "_alpha_"
        + df["alpha"].astype(str)
    )

    df["run_label"] = (
        df["function_tag"]
        + "_r"
        + df["rid"].astype(str)
    )

    write_line(lines, f"Affine feature shape: {df.shape}")
    write_line(lines, "Affine function counts:")
    write_line(
        lines,
        df.groupby(["fid1", "fid2", "alpha"]).size().reset_index(name="n_runs").to_string(index=False),
    )

    write_line(lines, "Affine feature preview:")
    write_line(
        lines,
        df[["fid1", "fid2", "alpha", "rid", EPS_COL, R2_COL, "function_tag"]]
        .head(10)
        .to_string(index=False),
    )

    return df


# ============================================================
# Summaries
# ============================================================

def summarize_affine_assignment(df_affine_assigned):
    def group_counts_json(x):
        counts = x.value_counts()
        counts_dict = {str(k): int(v) for k, v in counts.items()}
        return json.dumps(counts_dict, sort_keys=True)

    summary = (
        df_affine_assigned
        .groupby(["fid1", "fid2", "alpha", "function_tag", "function_name"], as_index=False)
        .agg(
            mean_eps_ratio=(EPS_COL, "mean"),
            std_eps_ratio=(EPS_COL, "std"),
            mean_adj_r2=(R2_COL, "mean"),
            std_adj_r2=(R2_COL, "std"),
            n_runs=("rid", "size"),
            majority_group=("group", lambda x: str(x.value_counts().idxmax())),
            group_counts=("group", group_counts_json),
        )
    )

    summary["n_runs"] = summary["n_runs"].astype(int)
    summary = summary.sort_values(["fid1", "fid2"]).reset_index(drop=True)
    return summary

# ============================================================
# Plot helpers
# ============================================================

def add_boundaries(ax, eps_bins, r2_bins):
    for x in eps_bins[1:-1]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.4, alpha=0.8)

    for y in r2_bins[1:-1]:
        ax.axhline(y, color="black", linestyle="--", linewidth=1.4, alpha=0.8)


def add_bbob_instance_points(ax, df_bbob_assigned):
    groups = sorted(df_bbob_assigned["group"].unique(), key=group_sort_key)

    for group in groups:
        sub = df_bbob_assigned[df_bbob_assigned["group"] == group]
        color = GROUP_COLORS.get(group, None)

        ax.scatter(
            sub[EPS_COL],
            sub[R2_COL],
            s=65,
            color=color,
            label=group,
            alpha=0.75,
        )

        for _, row in sub.iterrows():
            ax.text(
                row[EPS_COL] + 0.025,
                row[R2_COL] + 0.006,
                row["label"],
                fontsize=7,
                alpha=0.85,
            )


def add_affine_points(ax, df_affine_assigned, affine_summary):
    ax.scatter(
        df_affine_assigned[EPS_COL],
        df_affine_assigned[R2_COL],
        s=90,
        marker="x",
        color="black",
        linewidths=2.0,
        label="Affine 20 runs",
        zorder=8,
    )

    ax.scatter(
        affine_summary["mean_eps_ratio"],
        affine_summary["mean_adj_r2"],
        s=260,
        marker="*",
        color="red",
        edgecolors="black",
        linewidths=0.9,
        label="Affine mean",
        zorder=10,
    )

    for _, row in affine_summary.iterrows():
        label = (
            f"f{int(row['fid1'])}+f{int(row['fid2'])}\n"
            f"{row['majority_group']}"
        )

        ax.text(
            row["mean_eps_ratio"] + 0.06,
            row["mean_adj_r2"] + 0.016,
            label,
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=11,
        )


def finalize_axes(ax, title, df_bbob, df_affine):
    ax.set_xlabel("eps_ratio", fontsize=14)
    ax.set_ylabel("adj_r2", fontsize=14)
    ax.set_title(title, fontsize=16)

    xmin = min(0.0, df_bbob[EPS_COL].min(), df_affine[EPS_COL].min()) - 0.1
    xmax = max(7.8, df_bbob[EPS_COL].max(), df_affine[EPS_COL].max()) + 0.35

    ymin = min(-0.05, df_bbob[R2_COL].min(), df_affine[R2_COL].min()) - 0.02
    ymax = max(1.05, df_bbob[R2_COL].max(), df_affine[R2_COL].max()) + 0.03

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.tick_params(axis="both", labelsize=12)
    ax.legend(
        title="Group / Affine",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=10,
        title_fontsize=12,
    )

    ax.grid(False)


def plot_scheme(scheme_name, cfg, df_bbob_assigned, df_affine_assigned, affine_summary, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, f"PLOT SCHEME: {scheme_name}")
    write_line(lines, "=" * 80)

    fig, ax = plt.subplots(figsize=(14, 9))

    add_bbob_instance_points(ax, df_bbob_assigned)
    add_affine_points(ax, df_affine_assigned, affine_summary)
    add_boundaries(ax, cfg["eps_bins"], cfg["r2_bins"])

    title = (
        f"Affine BBOB functions assigned to ELA grouping\n"
        f"{scheme_name}"
    )

    finalize_axes(ax, title, df_bbob_assigned, df_affine_assigned)
    fig.tight_layout()

    png_file = OUTPUT_DIR / f"affine_functions_assignment_{scheme_name}.png"
    pdf_file = OUTPUT_DIR / f"affine_functions_assignment_{scheme_name}.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, bbox_inches="tight")
    plt.close(fig)

    write_line(lines, f"[OK] Plot saved to: {png_file}")
    write_line(lines, f"[OK] PDF saved to:  {pdf_file}")

    return png_file, pdf_file


def plot_individual_affine_functions(
    scheme_name,
    cfg,
    df_bbob_assigned,
    df_affine_assigned,
    affine_summary,
    lines,
):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, f"PLOT INDIVIDUAL AFFINE FUNCTIONS: {scheme_name}")
    write_line(lines, "=" * 80)

    outputs = []

    for _, summary_row in affine_summary.iterrows():
        function_tag = summary_row["function_tag"]
        sub_affine = df_affine_assigned[df_affine_assigned["function_tag"] == function_tag].copy()
        sub_summary = pd.DataFrame([summary_row])

        fig, ax = plt.subplots(figsize=(14, 9))

        add_bbob_instance_points(ax, df_bbob_assigned)
        add_affine_points(ax, sub_affine, sub_summary)
        add_boundaries(ax, cfg["eps_bins"], cfg["r2_bins"])

        title = (
            f"Affine function {function_tag} assigned to ELA grouping\n"
            f"{scheme_name}"
        )

        finalize_axes(ax, title, df_bbob_assigned, sub_affine)
        fig.tight_layout()

        png_file = OUTPUT_DIR / f"affine_function_{function_tag}_assignment_{scheme_name}.png"
        pdf_file = OUTPUT_DIR / f"affine_function_{function_tag}_assignment_{scheme_name}.pdf"

        fig.savefig(png_file, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_file, bbox_inches="tight")
        plt.close(fig)

        write_line(lines, f"[OK] Individual plot saved to: {png_file}")
        write_line(lines, f"[OK] Individual PDF saved to:  {pdf_file}")

        outputs.append(
            {
                "scheme": scheme_name,
                "function_tag": function_tag,
                "plot_file": str(png_file),
                "pdf_file": str(pdf_file),
            }
        )

    return outputs


# ============================================================
# Save outputs
# ============================================================

def save_scheme_outputs(
    scheme_name,
    cfg,
    df_bbob_assigned,
    df_affine_assigned,
    affine_summary,
    lines,
):
    scheme_dir = INTERMEDIATE_OUT_DIR / scheme_name
    scheme_dir.mkdir(parents=True, exist_ok=True)

    bbob_file = scheme_dir / f"bbob_instance_assignment_{scheme_name}.csv"
    affine_runs_file = scheme_dir / f"affine_runs_assignment_{scheme_name}.csv"
    affine_summary_file = scheme_dir / f"affine_summary_assignment_{scheme_name}.csv"
    scheme_json_file = scheme_dir / f"scheme_config_{scheme_name}.json"

    df_bbob_assigned.to_csv(bbob_file, index=False)
    df_affine_assigned.to_csv(affine_runs_file, index=False)
    affine_summary.to_csv(affine_summary_file, index=False)

    scheme_json = {
        "scheme_name": scheme_name,
        "scheme_note": cfg["scheme_note"],
        "eps_bins": [None if np.isneginf(x) or np.isposinf(x) else float(x) for x in cfg["eps_bins"]],
        "r2_bins": [None if np.isneginf(x) or np.isposinf(x) else float(x) for x in cfg["r2_bins"]],
        "bbob_file": str(bbob_file),
        "affine_runs_file": str(affine_runs_file),
        "affine_summary_file": str(affine_summary_file),
    }

    scheme_json_file.write_text(json.dumps(scheme_json, indent=2), encoding="utf-8")

    write_line(lines, f"[OK] BBOB assignment saved to: {bbob_file}")
    write_line(lines, f"[OK] Affine runs assignment saved to: {affine_runs_file}")
    write_line(lines, f"[OK] Affine summary saved to: {affine_summary_file}")
    write_line(lines, f"[OK] Scheme config saved to: {scheme_json_file}")

    return {
        "bbob_file": str(bbob_file),
        "affine_runs_file": str(affine_runs_file),
        "affine_summary_file": str(affine_summary_file),
        "scheme_json_file": str(scheme_json_file),
    }


# ============================================================
# Main
# ============================================================

def main():
    lines = []

    summary_file = OUTPUT_DIR / "task_G_assign_and_plot_affine_functions_summary.txt"

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK G: ASSIGN AND PLOT QUENTIN AFFINE FUNCTIONS")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root: {PROJECT_ROOT}")
    write_line(lines, f"BBOB feature file: {BBOB_FEATURE_FILE}")
    write_line(lines, f"Affine feature CSV: {AFFINE_FEATURE_CSV}")
    write_line(lines, f"Output dir: {OUTPUT_DIR}")
    write_line(lines, f"Intermediate output dir: {INTERMEDIATE_OUT_DIR}")

    try:
        df_bbob = load_bbob_instance_feature_points(lines)
        df_affine = load_affine_feature_points(lines)

        all_saved_files = {}
        all_plots = []
        all_individual_plots = []

        for scheme_name, cfg in SCHEMES.items():
            write_line(lines, "\n" + "#" * 80)
            write_line(lines, f"PROCESSING SCHEME: {scheme_name}")
            write_line(lines, "#" * 80)
            write_line(lines, cfg["scheme_note"])

            df_bbob_assigned = assign_grid_groups(
                df_bbob,
                eps_bins=cfg["eps_bins"],
                r2_bins=cfg["r2_bins"],
            )

            df_affine_assigned = assign_grid_groups(
                df_affine,
                eps_bins=cfg["eps_bins"],
                r2_bins=cfg["r2_bins"],
            )

            affine_summary = summarize_affine_assignment(df_affine_assigned)

            write_line(lines, "\nAffine assignment summary:")
            write_line(
                lines,
                affine_summary[
                    [
                        "fid1",
                        "fid2",
                        "alpha",
                        "mean_eps_ratio",
                        "std_eps_ratio",
                        "mean_adj_r2",
                        "std_adj_r2",
                        "n_runs",
                        "majority_group",
                        "group_counts",
                    ]
                ].to_string(index=False),
            )

            saved_files = save_scheme_outputs(
                scheme_name=scheme_name,
                cfg=cfg,
                df_bbob_assigned=df_bbob_assigned,
                df_affine_assigned=df_affine_assigned,
                affine_summary=affine_summary,
                lines=lines,
            )

            combined_png, combined_pdf = plot_scheme(
                scheme_name=scheme_name,
                cfg=cfg,
                df_bbob_assigned=df_bbob_assigned,
                df_affine_assigned=df_affine_assigned,
                affine_summary=affine_summary,
                lines=lines,
            )

            individual_plots = plot_individual_affine_functions(
                scheme_name=scheme_name,
                cfg=cfg,
                df_bbob_assigned=df_bbob_assigned,
                df_affine_assigned=df_affine_assigned,
                affine_summary=affine_summary,
                lines=lines,
            )

            all_saved_files[scheme_name] = saved_files
            all_plots.append(
                {
                    "scheme": scheme_name,
                    "png": str(combined_png),
                    "pdf": str(combined_pdf),
                }
            )
            all_individual_plots.extend(individual_plots)

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK G CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, "[SUCCESS] Quentin affine functions assigned and plotted.")

        write_line(lines, "\nCombined plots:")
        for item in all_plots:
            write_line(lines, f"  {item['scheme']}: {item['png']}")

        write_line(lines, "\nIntermediate files:")
        for scheme_name, files in all_saved_files.items():
            write_line(lines, f"  Scheme: {scheme_name}")
            for k, v in files.items():
                write_line(lines, f"    {k}: {v}")

        write_line(lines, "\nIndividual plots:")
        for item in all_individual_plots:
            write_line(lines, f"  {item['scheme']} | {item['function_tag']}: {item['plot_file']}")

        write_line(lines, "")
        write_line(lines, "Next step: run H0 modDE performance experiments on these affine functions, then run H SHAP.")

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