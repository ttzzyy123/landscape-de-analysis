from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Step 9: Hall of Fame under the 0307 main grouping scheme
#
# Reproduces the idea of Table 8 in:
# "Explainable Benchmarking for Iterative Optimization Heuristics"
#
# This script does not rerun experiments. It only reads existing MODDE
# result tables and exports best configurations per function.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "0307_main_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HALL_OF_FAME_CSV = OUTPUT_DIR / "hall_of_fame_0307.csv"
HALL_OF_FAME_MD = OUTPUT_DIR / "hall_of_fame_0307.md"
README_FILE = OUTPUT_DIR / "README_hall_of_fame_0307.txt"


# ============================================================
# 0307 main scheme
# ============================================================

SCHEME_NAME = "eps2bins_3_r0307_main_tiebreak"
EPS_SPLIT = 3.0
R2_SPLITS = [0.3, 0.7]


DE_COLUMNS = [
    "CR",
    "F",
    "adaptation_method",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
]

PERFORMANCE_COL = "auc"
LARGE_PERFORMANCE_COL = "aucLarge"


def log(msg: str):
    print(msg, flush=True)


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read CSV {path}: {exc}")
        return None


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Could not read JSON {path}: {exc}")
        return None


def safe_read_pickle(path: Path):
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        warnings.warn(f"Could not read pickle {path}: {exc}")
        return None


def group_id_to_label(group_id):
    if pd.isna(group_id):
        return None
    return f"G{int(group_id)}"


def assign_0307_group_from_features(eps_ratio, adj_r2):
    """Assign one point to the 0307 grid used by the main thesis scheme."""
    if eps_ratio is None or adj_r2 is None:
        return None
    if pd.isna(eps_ratio) or pd.isna(adj_r2):
        return None

    eps_bin = 0 if float(eps_ratio) < EPS_SPLIT else 1
    if float(adj_r2) < R2_SPLITS[0]:
        r2_bin = 0
    elif float(adj_r2) < R2_SPLITS[1]:
        r2_bin = 1
    else:
        r2_bin = 2

    group_id = eps_bin * 3 + r2_bin
    return f"G{group_id}"


def normalize_assignment_group(value):
    if value is None or pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"G?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"G{int(match.group(1))}"
    return text


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


# ============================================================
# 0307 BBOB mapping
# ============================================================

def load_bbob_0307_mapping():
    candidates = [
        PROJECT_ROOT
        / "output"
        / "manual_binning_experiments_tiebreak"
        / "eps2bins_3_r0307_main"
        / "function_group_mapping_for_step3.csv",
        PROJECT_ROOT / "intermediate" / "group_to_functions_manual_bins.csv",
    ]

    mapping_file = find_existing_file(candidates)
    if mapping_file is None:
        warnings.warn("Could not find a 0307 BBOB function-to-group mapping file.")
        return pd.DataFrame(columns=["fid", "assigned_group_0307", "group_label_0307"])

    df = pd.read_csv(mapping_file)
    rename = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc == "function":
            rename[col] = "fid"
        elif lc in {"group", "assigned_group"}:
            rename[col] = "group_id"

    df = df.rename(columns=rename)
    if "fid" not in df.columns:
        warnings.warn(f"Mapping file lacks Function/fid column: {mapping_file}")
        return pd.DataFrame(columns=["fid", "assigned_group_0307", "group_label_0307"])

    df["fid"] = pd.to_numeric(df["fid"], errors="coerce")
    df = df.dropna(subset=["fid"]).copy()
    df["fid"] = df["fid"].astype(int)

    if "group_id" in df.columns:
        df["assigned_group_0307"] = df["group_id"].apply(group_id_to_label)
    elif "group_label" in df.columns:
        # If only labels exist, map sorted labels to stable ids as in step2.
        labels = sorted(df["group_label"].dropna().astype(str).unique().tolist())
        label_to_id = {label: i for i, label in enumerate(labels)}
        df["assigned_group_0307"] = df["group_label"].map(label_to_id).apply(group_id_to_label)
    else:
        df["assigned_group_0307"] = None

    if "group_label" not in df.columns:
        df["group_label"] = None

    out = df[["fid", "assigned_group_0307", "group_label"]].copy()
    out = out.rename(columns={"group_label": "group_label_0307"})
    out["mapping_source"] = str(mapping_file.relative_to(PROJECT_ROOT))
    return out


# ============================================================
# Hall-of-fame aggregation
# ============================================================

def standardize_modde_columns(df: pd.DataFrame):
    df = df.copy()

    rename = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc in {"function", "function_id", "bbob_function"}:
            rename[col] = "fid"
        elif lc in {"instance", "instance_id"}:
            rename[col] = "iid"
        elif lc in {"adaptation", "adaptation"}:
            rename[col] = "adaptation_method"
        elif lc in {"base", "mutationbase"}:
            rename[col] = "mutation_base"
        elif lc in {"diffs", "mutation_n_components"}:
            rename[col] = "mutation_n_comps"
        elif lc in {"ref", "mutation_ref"}:
            rename[col] = "mutation_reference"
        elif lc in {"archive"}:
            rename[col] = "use_archive"
    df = df.rename(columns=rename)

    for col in ["mutation_reference", "adaptation_method"]:
        if col in df.columns:
            df[col] = df[col].replace(np.nan, "nan").astype(str)

    for col in ["crossover", "mutation_base"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "use_archive" in df.columns:
        df["use_archive"] = df["use_archive"].astype(str).map(
            {
                "True": True,
                "False": False,
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        ).fillna(df["use_archive"])

    return df


def summarize_best_config(
    df: pd.DataFrame,
    group_cols,
    metadata: dict,
    source_file: Path,
):
    df = standardize_modde_columns(df)

    if PERFORMANCE_COL not in df.columns:
        warnings.warn(f"Skipping {source_file}: missing {PERFORMANCE_COL}")
        return []

    missing_de = [c for c in DE_COLUMNS if c not in df.columns]
    if missing_de:
        warnings.warn(f"{source_file} missing DE columns {missing_de}; output will contain blanks.")

    available_de_cols = [c for c in DE_COLUMNS if c in df.columns]
    agg_cols = list(dict.fromkeys(group_cols + available_de_cols))

    work = df.dropna(subset=[PERFORMANCE_COL]).copy()
    if work.empty:
        warnings.warn(f"Skipping {source_file}: no valid {PERFORMANCE_COL} values.")
        return []

    grouped = (
        work.groupby(agg_cols, dropna=False)
        .agg(
            actual_auc_mean=(PERFORMANCE_COL, "mean"),
            actual_auc_std=(PERFORMANCE_COL, "std"),
            actual_auc_median=(PERFORMANCE_COL, "median"),
            actual_auc_min=(PERFORMANCE_COL, "min"),
            actual_auc_max=(PERFORMANCE_COL, "max"),
            n_rows=(PERFORMANCE_COL, "size"),
            n_seeds=("seed", "nunique") if "seed" in work.columns else (PERFORMANCE_COL, "size"),
        )
        .reset_index()
    )

    if LARGE_PERFORMANCE_COL in work.columns:
        large = (
            work.groupby(agg_cols, dropna=False)[LARGE_PERFORMANCE_COL]
            .mean()
            .reset_index(name="actual_aucLarge_mean")
        )
        grouped = grouped.merge(large, on=agg_cols, how="left")

    # Higher AUC is better in the existing scripts.
    best = (
        grouped.sort_values(
            ["actual_auc_mean", "actual_auc_median", "n_rows"],
            ascending=[False, False, False],
        )
        .head(1)
        .copy()
    )

    records = []
    for _, row in best.iterrows():
        record = dict(metadata)
        record["source_file"] = str(source_file.relative_to(PROJECT_ROOT))
        record["ranking_metric"] = "mean_auc_desc"

        for col in group_cols + DE_COLUMNS:
            record[col] = row[col] if col in row.index else None

        for col in [
            "actual_auc_mean",
            "actual_auc_std",
            "actual_auc_median",
            "actual_auc_min",
            "actual_auc_max",
            "actual_aucLarge_mean",
            "n_rows",
            "n_seeds",
        ]:
            record[col] = row[col] if col in row.index else None

        records.append(record)

    return records


# ============================================================
# BBOB original functions
# ============================================================

def build_bbob_hall_of_fame():
    data_file = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"
    if not data_file.exists():
        warnings.warn(f"Missing BBOB data file: {data_file}")
        return []

    df = safe_read_pickle(data_file)
    if df is None or df.empty:
        return []

    df = standardize_modde_columns(df)
    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    mapping = load_bbob_0307_mapping()
    if not mapping.empty and "fid" in df.columns:
        df = df.merge(mapping, on="fid", how="left")
    else:
        df["assigned_group_0307"] = None
        df["group_label_0307"] = None

    records = []
    if "fid" not in df.columns:
        warnings.warn("BBOB data lacks fid column; cannot produce per-function Hall of Fame.")
        return records

    for fid, sub in df.groupby("fid"):
        metadata = {
            "function_type": "bbob_original",
            "function_id": f"f{int(fid)}",
            "function_name": f"f{int(fid)}",
            "fid": int(fid),
            "assigned_group_0307": normalize_assignment_group(
                sub["assigned_group_0307"].dropna().iloc[0]
                if sub["assigned_group_0307"].notna().any()
                else None
            ),
            "group_label_0307": (
                sub["group_label_0307"].dropna().iloc[0]
                if "group_label_0307" in sub.columns and sub["group_label_0307"].notna().any()
                else None
            ),
            "scheme": SCHEME_NAME,
        }
        records.extend(
            summarize_best_config(
                sub,
                group_cols=["fid"],
                metadata=metadata,
                source_file=data_file,
            )
        )

    metadata = {
        "function_type": "bbob_original",
        "function_id": "All",
        "function_name": "All BBOB functions",
        "fid": "All",
        "assigned_group_0307": "All",
        "group_label_0307": "All",
        "scheme": SCHEME_NAME,
    }
    records.extend(
        summarize_best_config(
            df,
            group_cols=[],
            metadata=metadata,
            source_file=data_file,
        )
    )

    return records


# ============================================================
# Affine functions
# ============================================================

def load_affine_assignment_summary():
    path = (
        PROJECT_ROOT
        / "intermediate"
        / "new_function_task"
        / "affine_function_group_assignment"
        / "0307_main"
        / "affine_summary_assignment_0307_main.csv"
    )
    if path.exists():
        df = pd.read_csv(path)
        df["function_tag"] = "f" + df["fid1"].astype(str) + "_f" + df["fid2"].astype(str)
        return df
    return pd.DataFrame()


def parse_affine_tag_from_name(path: Path):
    match = re.search(r"affine_function_(f\d+_f\d+)_alpha_([^_]+)", path.stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def build_affine_hall_of_fame():
    modde_dir = PROJECT_ROOT / "intermediate" / "new_function_task" / "modde_affine_function_results"
    if not modde_dir.exists():
        return []

    assignment = load_affine_assignment_summary()
    records = []

    files = sorted(modde_dir.glob("*_modde_results_processed.pkl"))
    for pkl_file in files:
        function_tag, alpha_tag = parse_affine_tag_from_name(pkl_file)
        if function_tag is None:
            continue

        df = safe_read_pickle(pkl_file)
        if df is None or df.empty:
            continue

        info = {}
        if not assignment.empty:
            sub = assignment[assignment["function_tag"].astype(str) == function_tag]
            if not sub.empty:
                info = sub.iloc[0].to_dict()

        assigned_group = normalize_assignment_group(info.get("majority_group"))
        if assigned_group is None:
            assigned_group = assign_0307_group_from_features(
                info.get("mean_eps_ratio"),
                info.get("mean_adj_r2"),
            )

        metadata = {
            "function_type": "affine",
            "function_id": function_tag,
            "function_name": function_tag.replace("_", "+"),
            "fid": df["fid"].dropna().iloc[0] if "fid" in df.columns and df["fid"].notna().any() else None,
            "fid1": info.get("fid1"),
            "fid2": info.get("fid2"),
            "alpha": info.get("alpha"),
            "assigned_group_0307": assigned_group,
            "group_label_0307": None,
            "eps_ratio": info.get("mean_eps_ratio"),
            "adj_r2": info.get("mean_adj_r2"),
            "eps_ratio_std": info.get("std_eps_ratio"),
            "adj_r2_std": info.get("std_adj_r2"),
            "scheme": SCHEME_NAME,
        }

        records.extend(
            summarize_best_config(
                df,
                group_cols=[],
                metadata=metadata,
                source_file=pkl_file,
            )
        )

    if records:
        df_all = pd.DataFrame(records)
        affine_rows = df_all[df_all["function_type"] == "affine"]
        if not affine_rows.empty:
            # Best configuration across the already selected per-function winners.
            best = affine_rows.sort_values("actual_auc_mean", ascending=False).head(1).copy()
            best["function_id"] = "All"
            best["function_name"] = "All affine functions"
            best["assigned_group_0307"] = "All"
            best["source_file"] = "aggregate_of_affine_hall_of_fame_rows"
            records.extend(best.to_dict("records"))

    return records


# ============================================================
# Real generated LLaMEA functions
# ============================================================

def load_real_feature_summary():
    candidates = [
        PROJECT_ROOT
        / "intermediate"
        / "new_function_task"
        / "real_function_features"
        / "real_generated_functions_batch_ela_features_summary.csv",
    ]

    path = find_existing_file(candidates)
    if path is None:
        return pd.DataFrame()

    return pd.read_csv(path)


def parse_real_id_from_name(path: Path):
    stem = path.stem
    stem = re.sub(r"_modde_results_processed$", "", stem)
    stem = re.sub(r"^real_generated_function_", "", stem)
    return stem


def build_real_generated_hall_of_fame():
    modde_dir = PROJECT_ROOT / "intermediate" / "new_function_task" / "modde_real_function_results"
    if not modde_dir.exists():
        return []

    features = load_real_feature_summary()
    records = []

    files = sorted(modde_dir.glob("*_modde_results_processed.pkl"))
    for pkl_file in files:
        full_id = parse_real_id_from_name(pkl_file)
        df = safe_read_pickle(pkl_file)
        if df is None or df.empty:
            continue

        info = {}
        if not features.empty and "full_id" in features.columns:
            sub = features[features["full_id"].astype(str) == full_id]
            if not sub.empty:
                info = sub.iloc[0].to_dict()

        eps_ratio = info.get("eps_ratio")
        adj_r2 = info.get("adj_r2")
        assigned_group = assign_0307_group_from_features(eps_ratio, adj_r2)

        metadata = {
            "function_type": "llamea_generated",
            "function_id": full_id,
            "function_name": f"real_generated_function_{full_id}",
            "fid": df["fid"].dropna().iloc[0] if "fid" in df.columns and df["fid"].notna().any() else None,
            "run_id": info.get("run_id"),
            "function_tag": info.get("function_tag"),
            "assigned_group_0307": assigned_group,
            "group_label_0307": None,
            "eps_ratio": eps_ratio,
            "adj_r2": adj_r2,
            "eps_ratio_std": info.get("eps_ratio_std"),
            "adj_r2_std": info.get("adj_r2_std"),
            "scheme": SCHEME_NAME,
        }

        records.extend(
            summarize_best_config(
                df,
                group_cols=[],
                metadata=metadata,
                source_file=pkl_file,
            )
        )

    if records:
        df_all = pd.DataFrame(records)
        real_rows = df_all[df_all["function_type"] == "llamea_generated"]
        if not real_rows.empty:
            best = real_rows.sort_values("actual_auc_mean", ascending=False).head(1).copy()
            best["function_id"] = "All"
            best["function_name"] = "All LLaMEA generated functions"
            best["assigned_group_0307"] = "All"
            best["source_file"] = "aggregate_of_llamea_hall_of_fame_rows"
            records.extend(best.to_dict("records"))

    return records


def order_columns(df: pd.DataFrame):
    preferred = [
        "function_type",
        "function_id",
        "function_name",
        "fid",
        "fid1",
        "fid2",
        "alpha",
        "run_id",
        "function_tag",
        "scheme",
        "assigned_group_0307",
        "group_label_0307",
        "eps_ratio",
        "adj_r2",
        "eps_ratio_std",
        "adj_r2_std",
        "CR",
        "F",
        "adaptation_method",
        "crossover",
        "lambda_",
        "lpsr",
        "mutation_base",
        "mutation_n_comps",
        "mutation_reference",
        "use_archive",
        "actual_auc_mean",
        "actual_auc_std",
        "actual_auc_median",
        "actual_auc_min",
        "actual_auc_max",
        "actual_aucLarge_mean",
        "n_rows",
        "n_seeds",
        "ranking_metric",
        "source_file",
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def write_markdown_summary(df: pd.DataFrame):
    if df.empty:
        HALL_OF_FAME_MD.write_text("No Hall of Fame rows were generated.\n", encoding="utf-8")
        return

    display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "CR",
        "F",
        "adaptation_method",
        "crossover",
        "lambda_",
        "lpsr",
        "mutation_base",
        "mutation_n_comps",
        "mutation_reference",
        "use_archive",
        "actual_auc_mean",
        "n_rows",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    lines = [
        "# Hall of Fame under 0307 main scheme",
        "",
        f"Scheme: `{SCHEME_NAME}`",
        "",
        "Rows are best configurations by mean `auc` within each function.",
        "",
    ]

    for function_type, sub in df.groupby("function_type", dropna=False):
        lines.append(f"## {function_type}")
        lines.append("")
        rows = sub[display_cols].copy()
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for _, row in rows.iterrows():
            values = []
            for col in display_cols:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.6g}"
                elif pd.isna(value):
                    value = ""
                values.append(str(value).replace("|", "/"))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    HALL_OF_FAME_MD.write_text("\n".join(lines), encoding="utf-8")


def write_readme(df: pd.DataFrame):
    counts = df["function_type"].value_counts(dropna=False).to_dict() if not df.empty else {}
    lines = [
        "Step 9 Hall of Fame under the 0307 main grouping scheme",
        "",
        "Purpose:",
        "- Reproduce the Hall-of-Fame idea from Table 8 of the explainable benchmarking paper.",
        "- Use only existing results; this script does not rerun MODDE or SHAP.",
        "- Align all outputs to eps2bins_3_r0307_main_tiebreak.",
        "",
        "Outputs:",
        f"- {HALL_OF_FAME_CSV.relative_to(PROJECT_ROOT)}",
        f"- {HALL_OF_FAME_MD.relative_to(PROJECT_ROOT)}",
        "",
        "Selection rule:",
        "- For each function, group rows by DE configuration.",
        "- Average auc over repeated seeds / rows.",
        "- Pick the configuration with the highest mean auc.",
        "",
        "Row counts by function_type:",
        json.dumps(counts, indent=2),
        "",
        "Note:",
        "- LLaMEA generated functions may have older assignment files from eps2bins_3_r025065.",
        "- For 0307 alignment, this script recomputes assigned_group_0307 from eps_ratio and adj_r2.",
    ]
    README_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    log("=" * 80)
    log("STEP 9: HALL OF FAME UNDER 0307 MAIN SCHEME")
    log("=" * 80)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Output dir:   {OUTPUT_DIR}")

    records = []

    log("\nCollecting BBOB original functions...")
    records.extend(build_bbob_hall_of_fame())

    log("\nCollecting affine functions...")
    records.extend(build_affine_hall_of_fame())

    log("\nCollecting LLaMEA generated functions...")
    records.extend(build_real_generated_hall_of_fame())

    df = pd.DataFrame(records)
    if not df.empty:
        df = order_columns(df)
        df = df.sort_values(
            ["function_type", "function_id"],
            key=lambda s: s.astype(str),
        ).reset_index(drop=True)

    df.to_csv(HALL_OF_FAME_CSV, index=False)
    write_markdown_summary(df)
    write_readme(df)

    log("\nDone.")
    log(f"Saved CSV: {HALL_OF_FAME_CSV}")
    log(f"Saved MD:  {HALL_OF_FAME_MD}")
    log(f"Rows:      {len(df)}")


if __name__ == "__main__":
    main()
