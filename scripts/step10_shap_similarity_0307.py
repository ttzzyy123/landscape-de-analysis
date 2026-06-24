from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Step 10: SHAP similarity analysis under 0307 main scheme
#
# Scheme:
#   eps2bins_3_r0307_main_tiebreak
#
# This script uses:
#   - group-level SHAP from:
#       output/shap_eps2bins_3_r0307_main_tiebreak_niles/
#
#   - BBOB individual SHAP from:
#       output/shap_individual_manual_bins_niles/
#
#   - affine individual SHAP from:
#       output/new_function_task/task_H_individual_shap_affine_functions/
#
#   - LLaMEA individual SHAP from:
#       output/new_function_task/task_H_individual_shap_real_generated_function/
#
# Important:
#   This script does NOT use:
#     - ambiguous_on_tie
#     - mean_based
#     - eps2bins_3_r025065
#     - shap_individual_manual_bins_encoding
#
# Wasserstein note:
#   Current inputs are summary-level SHAP importance CSV files, not raw SHAP
#   value distributions. Therefore:
#
#   - wasserstein_importance_distribution compares the distribution of
#     normalized importance magnitudes across features.
#
#   - mean_wasserstein / median_wasserstein / max_wasserstein are computed
#     from per-feature absolute differences. With one summary importance value
#     per feature, the one-dimensional Wasserstein distance per feature reduces
#     to abs(group_importance - individual_importance).
#
# ============================================================


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_DIR = PROJECT_ROOT / "output" / "0307_main_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

HALL_OF_FAME_CSV = COMPARISON_DIR / "hall_of_fame_0307.csv"

GROUP_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "shap_eps2bins_3_r0307_main_tiebreak_niles"
)

BBOB_INDIVIDUAL_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "shap_individual_manual_bins_niles"
)

AFFINE_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_affine_functions"
)

LLAMEA_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_real_generated_function"
)

OUT_CSV = COMPARISON_DIR / "shap_similarity_0307.csv"
OUT_MD = COMPARISON_DIR / "shap_similarity_0307.md"
OUT_TYPE_SUMMARY_CSV = COMPARISON_DIR / "shap_similarity_0307_by_function_type.csv"
OUT_GROUP_SUMMARY_CSV = COMPARISON_DIR / "shap_similarity_0307_by_group.csv"
README_FILE = COMPARISON_DIR / "README_shap_similarity_0307.txt"


# =========================
# Basic utilities
# =========================
def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def parse_group_number(group_value):
    if pd.isna(group_value):
        return None

    s = str(group_value).strip()

    if s == "" or s.lower() == "nan" or s.lower() == "all":
        return None

    if s.startswith("G"):
        s = s[1:]

    try:
        return int(s)
    except ValueError:
        return None


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    total = float(np.sum(np.abs(v)))
    if total <= 0:
        return np.zeros_like(v)

    return v / total


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return np.nan

    return float(np.dot(a, b) / denom)


def spearman_rank_corr(a, b):
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))

    if len(a) < 2:
        return np.nan

    ar = a.rank(method="average")
    br = b.rank(method="average")

    if ar.std(ddof=0) == 0 or br.std(ddof=0) == 0:
        return np.nan

    return float(ar.corr(br, method="pearson"))


def wasserstein_1d_equal_weight(values_a, values_b):
    """
    1D Wasserstein distance between two equal-weight empirical distributions.

    Here it is used as a summary-level fallback:
    compare the distribution of normalized importance magnitudes across features.

    Since both vectors are aligned to the same union of features, they have the
    same length. For equal-sized empirical samples in 1D, W1 is the mean absolute
    difference between sorted samples.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    if len(a) == 0 or len(b) == 0:
        return np.nan

    a = np.sort(a)
    b = np.sort(b)

    if len(a) != len(b):
        # Fallback quantile interpolation for unequal lengths.
        qs = np.linspace(0.0, 1.0, max(len(a), len(b)))
        qa = np.quantile(a, qs)
        qb = np.quantile(b, qs)
        return float(np.mean(np.abs(qa - qb)))

    return float(np.mean(np.abs(a - b)))


def top_k_features(vec_dict, k=3):
    items = sorted(
        vec_dict.items(),
        key=lambda kv: (-safe_float(kv[1]), str(kv[0])),
    )
    return [f for f, _ in items[:k]]


def best_config_summary(row):
    config_cols = [
        "CR",
        "F",
        "crossover",
        "lambda_",
        "lpsr",
        "mutation_base",
        "mutation_n_comps",
        "mutation_reference",
        "use_archive",
    ]

    parts = []
    for c in config_cols:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        parts.append(f"{c}={v}")

    return "; ".join(parts)


def feature_name_cleanup(x):
    """
    Keep feature names consistent.

    We intentionally use the non-encoding/niles files, so categorical parameters
    should usually already appear as aggregate features such as:
        mutation_base
        mutation_reference
        crossover

    This function still strips whitespace.
    """
    return str(x).strip()


def detect_value_column(df):
    """
    Detect SHAP importance column.

    Priority:
      - importance_norm
      - mean_abs_shap
      - importance
      - global_individual_mean_importance
    """
    candidates = [
        "importance_norm",
        "mean_abs_shap",
        "importance",
        "global_individual_mean_importance",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = [
        c for c in df.columns
        if c != "feature" and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Cannot detect SHAP importance value column. "
        f"Available columns: {df.columns.tolist()}"
    )


def read_importance_vector(path):
    """
    Read a summary-level SHAP importance CSV into:
        dict(feature -> normalized importance)

    Expected formats include:
      - feature, importance, importance_norm
      - feature, mean_abs_shap, importance_norm
      - feature, global_individual_mean_importance
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"Missing 'feature' column in {path}")

    value_col = detect_value_column(df)

    out = {}
    for _, r in df.iterrows():
        f = feature_name_cleanup(r["feature"])
        v = safe_float(r[value_col])

        if not np.isfinite(v):
            continue

        out[f] = out.get(f, 0.0) + float(v)

    features = list(out.keys())
    vals = normalize_vector([out[f] for f in features])
    out = {f: float(v) for f, v in zip(features, vals)}

    return out, path, value_col


# =========================
# Loaders
# =========================
def load_group_vector(group_id):
    gnum = parse_group_number(group_id)
    if gnum is None:
        raise ValueError(f"Invalid group id: {group_id}")

    path = GROUP_SHAP_DIR / f"shap_group_{gnum}.csv"
    return read_importance_vector(path)


def load_bbob_vector(function_id):
    """
    Hall of Fame uses function_id like:
        f1, f2, ..., f24

    Individual file path:
        output/shap_individual_manual_bins_niles/shap_function_1.csv
    """
    fid_str = str(function_id).strip()

    if fid_str.startswith("f"):
        fid_str = fid_str[1:]

    try:
        fid = int(fid_str)
    except ValueError:
        raise ValueError(f"Cannot parse BBOB function id: {function_id}")

    path = BBOB_INDIVIDUAL_SHAP_DIR / f"shap_function_{fid}.csv"
    return read_importance_vector(path)


def load_affine_vector(function_id):
    path = (
        AFFINE_SHAP_DIR
        / f"affine_function_{function_id}_alpha_0p9_individual_shap_importance.csv"
    )
    return read_importance_vector(path)


def load_llamea_vector(function_id):
    path = (
        LLAMEA_SHAP_DIR
        / f"real_generated_function_{function_id}_individual_shap_importance.csv"
    )
    return read_importance_vector(path)


# =========================
# Metrics
# =========================
def align_vectors(group_vec, individual_vec):
    features = sorted(set(group_vec.keys()) | set(individual_vec.keys()))

    g = np.array([group_vec.get(f, 0.0) for f in features], dtype=float)
    i = np.array([individual_vec.get(f, 0.0) for f in features], dtype=float)

    g = normalize_vector(g)
    i = normalize_vector(i)

    return features, g, i


def compute_similarity_metrics(group_vec, individual_vec):
    features, g, i = align_vectors(group_vec, individual_vec)

    cos = cosine_similarity(g, i)
    spear = spearman_rank_corr(g, i)

    diff = np.abs(g - i)

    l1 = float(np.sum(diff))
    l2 = float(np.sqrt(np.sum((g - i) ** 2)))

    group_aligned = {f: float(v) for f, v in zip(features, g)}
    individual_aligned = {f: float(v) for f, v in zip(features, i)}

    group_top1 = top_k_features(group_aligned, k=1)[0] if features else ""
    individual_top1 = top_k_features(individual_aligned, k=1)[0] if features else ""

    group_top3 = top_k_features(group_aligned, k=3)
    individual_top3 = top_k_features(individual_aligned, k=3)

    top3_overlap_set = set(group_top3) & set(individual_top3)
    top3_overlap_count = len(top3_overlap_set)
    top3_overlap_ratio = top3_overlap_count / 3.0

    if len(features) > 0:
        max_idx = int(np.argmax(diff))
        most_different_feature = features[max_idx]
        most_different_abs_diff = float(diff[max_idx])
    else:
        most_different_feature = ""
        most_different_abs_diff = np.nan

    # Wasserstein-style summary metrics.
    #
    # Current files contain one normalized importance value per feature.
    # Thus, per-feature 1D Wasserstein degenerates to abs difference between
    # the two point masses in that feature dimension.
    per_feature_wasserstein = diff.copy()

    wasserstein_importance_distribution = wasserstein_1d_equal_weight(g, i)

    mean_wasserstein = float(np.mean(per_feature_wasserstein)) if len(diff) else np.nan
    median_wasserstein = float(np.median(per_feature_wasserstein)) if len(diff) else np.nan
    max_wasserstein = float(np.max(per_feature_wasserstein)) if len(diff) else np.nan

    if len(features) > 0:
        w_idx = int(np.argmax(per_feature_wasserstein))
        most_different_feature_wasserstein = features[w_idx]
    else:
        most_different_feature_wasserstein = ""

    return {
        "cosine_similarity": cos,
        "spearman_rank_corr": spear,
        "l1_distance": l1,
        "l2_distance": l2,
        "top1_match": bool(group_top1 == individual_top1),
        "top3_overlap_count": top3_overlap_count,
        "top3_overlap_ratio": top3_overlap_ratio,
        "individual_top1_feature": individual_top1,
        "group_top1_feature": group_top1,
        "individual_top3_features": ",".join(individual_top3),
        "group_top3_features": ",".join(group_top3),
        "top3_overlap_features": ",".join(sorted(top3_overlap_set)),
        "most_different_feature": most_different_feature,
        "most_different_abs_diff": most_different_abs_diff,
        "wasserstein_importance_distribution": wasserstein_importance_distribution,
        "mean_wasserstein": mean_wasserstein,
        "median_wasserstein": median_wasserstein,
        "max_wasserstein": max_wasserstein,
        "most_different_feature_wasserstein": most_different_feature_wasserstein,
        "wasserstein_level": "summary_importance_vector",
        "wasserstein_note": (
            "Computed from normalized summary importance vectors. "
            "Raw SHAP-value distributions were not available in the input CSV files."
        ),
        "n_common_features": len(set(group_vec.keys()) & set(individual_vec.keys())),
        "n_union_features": len(features),
    }


# =========================
# Reporting helpers
# =========================
def df_to_markdown_no_tabulate(df):
    cols = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]

            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))

        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def make_function_type_summary(df):
    ok = df[df["status"].eq("ok")].copy()

    if ok.empty:
        return pd.DataFrame()

    rows = []
    for function_type, sub in ok.groupby("function_type", sort=False):
        rows.append(
            {
                "function_type": function_type,
                "n_functions_ok": len(sub),
                "mean_cosine_similarity": sub["cosine_similarity"].mean(),
                "median_cosine_similarity": sub["cosine_similarity"].median(),
                "mean_spearman_rank_corr": sub["spearman_rank_corr"].mean(),
                "median_spearman_rank_corr": sub["spearman_rank_corr"].median(),
                "mean_top3_overlap_ratio": sub["top3_overlap_ratio"].mean(),
                "median_top3_overlap_ratio": sub["top3_overlap_ratio"].median(),
                "mean_l1_distance": sub["l1_distance"].mean(),
                "median_l1_distance": sub["l1_distance"].median(),
                "mean_wasserstein": sub["mean_wasserstein"].mean(),
                "median_wasserstein": sub["median_wasserstein"].median(),
                "mean_wasserstein_importance_distribution": sub["wasserstein_importance_distribution"].mean(),
                "median_wasserstein_importance_distribution": sub["wasserstein_importance_distribution"].median(),
            }
        )

    return pd.DataFrame(rows)


def make_group_summary(df):
    ok = df[df["status"].eq("ok")].copy()

    if ok.empty:
        return pd.DataFrame()

    rows = []
    for group_id, sub in ok.groupby("assigned_group_0307", sort=True):
        rows.append(
            {
                "assigned_group_0307": group_id,
                "n_functions_ok": len(sub),
                "function_types": ",".join(sorted(sub["function_type"].unique())),
                "mean_cosine_similarity": sub["cosine_similarity"].mean(),
                "median_cosine_similarity": sub["cosine_similarity"].median(),
                "mean_spearman_rank_corr": sub["spearman_rank_corr"].mean(),
                "median_spearman_rank_corr": sub["spearman_rank_corr"].median(),
                "mean_top3_overlap_ratio": sub["top3_overlap_ratio"].mean(),
                "median_top3_overlap_ratio": sub["top3_overlap_ratio"].median(),
                "mean_l1_distance": sub["l1_distance"].mean(),
                "median_l1_distance": sub["l1_distance"].median(),
                "mean_wasserstein": sub["mean_wasserstein"].mean(),
                "median_wasserstein": sub["median_wasserstein"].median(),
                "mean_wasserstein_importance_distribution": sub["wasserstein_importance_distribution"].mean(),
                "median_wasserstein_importance_distribution": sub["wasserstein_importance_distribution"].median(),
            }
        )

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("STEP 10: SHAP SIMILARITY UNDER 0307 MAIN TIEBREAK SCHEME")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Hall of Fame: {HALL_OF_FAME_CSV}")
    print(f"Group SHAP dir: {GROUP_SHAP_DIR}")
    print(f"BBOB individual SHAP dir: {BBOB_INDIVIDUAL_SHAP_DIR}")
    print(f"Affine SHAP dir: {AFFINE_SHAP_DIR}")
    print(f"LLaMEA SHAP dir: {LLAMEA_SHAP_DIR}")
    print(f"Output dir: {COMPARISON_DIR}")

    if not HALL_OF_FAME_CSV.exists():
        raise FileNotFoundError(f"Missing Hall of Fame file: {HALL_OF_FAME_CSV}")

    hof = pd.read_csv(HALL_OF_FAME_CSV)

    # Remove aggregate rows.
    hof = hof[~hof["function_id"].astype(str).eq("All")].copy()

    results = []

    for _, row in hof.iterrows():
        function_type = str(row.get("function_type", ""))
        function_id = str(row.get("function_id", ""))
        assigned_group = row.get("assigned_group_0307", np.nan)
        group_label = row.get("group_label_0307", "")

        base_result = {
            "function_type": function_type,
            "function_id": function_id,
            "assigned_group_0307": assigned_group,
            "group_label_0307": group_label,
            "actual_auc_mean": safe_float(row.get("actual_auc_mean", np.nan)),
            "actual_auc_std": safe_float(row.get("actual_auc_std", np.nan)),
            "hof_n_rows_for_best_config": safe_float(row.get("n_rows", np.nan)),
            "hof_n_seeds_for_best_config": safe_float(row.get("n_seeds", np.nan)),
            "best_config_summary": best_config_summary(row),
            "source_hall_of_fame": str(HALL_OF_FAME_CSV.relative_to(PROJECT_ROOT)),
        }

        try:
            if pd.isna(assigned_group) or str(assigned_group).strip() == "":
                raise ValueError("Missing assigned_group_0307")

            group_vec, group_path, group_value_col = load_group_vector(assigned_group)

            if function_type == "bbob_original":
                individual_vec, individual_path, individual_value_col = load_bbob_vector(function_id)

            elif function_type == "affine":
                individual_vec, individual_path, individual_value_col = load_affine_vector(function_id)

            elif function_type == "llamea_generated":
                individual_vec, individual_path, individual_value_col = load_llamea_vector(function_id)

            else:
                raise ValueError(f"Unsupported function_type: {function_type}")

            metrics = compute_similarity_metrics(group_vec, individual_vec)

            result = {
                **base_result,
                **metrics,
                "source_individual_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                "individual_value_column": individual_value_col,
                "group_value_column": group_value_col,
                "status": "ok",
                "note": "",
            }

        except Exception as e:
            result = {
                **base_result,
                "cosine_similarity": np.nan,
                "spearman_rank_corr": np.nan,
                "l1_distance": np.nan,
                "l2_distance": np.nan,
                "top1_match": np.nan,
                "top3_overlap_count": np.nan,
                "top3_overlap_ratio": np.nan,
                "individual_top1_feature": "",
                "group_top1_feature": "",
                "individual_top3_features": "",
                "group_top3_features": "",
                "top3_overlap_features": "",
                "most_different_feature": "",
                "most_different_abs_diff": np.nan,
                "wasserstein_importance_distribution": np.nan,
                "mean_wasserstein": np.nan,
                "median_wasserstein": np.nan,
                "max_wasserstein": np.nan,
                "most_different_feature_wasserstein": "",
                "wasserstein_level": "",
                "wasserstein_note": "",
                "n_common_features": np.nan,
                "n_union_features": np.nan,
                "source_individual_shap": "",
                "source_group_shap": "",
                "individual_value_column": "",
                "group_value_column": "",
                "status": "error",
                "note": str(e),
            }

        results.append(result)

    out = pd.DataFrame(results)

    function_type_order = {
        "bbob_original": 0,
        "affine": 1,
        "llamea_generated": 2,
    }

    out["_type_order"] = out["function_type"].map(function_type_order).fillna(99)
    out = out.sort_values(["_type_order", "function_type", "function_id"]).drop(columns=["_type_order"])

    out.to_csv(OUT_CSV, index=False)

    type_summary = make_function_type_summary(out)
    group_summary = make_group_summary(out)

    type_summary.to_csv(OUT_TYPE_SUMMARY_CSV, index=False)
    group_summary.to_csv(OUT_GROUP_SUMMARY_CSV, index=False)

    display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
        "cosine_similarity",
        "spearman_rank_corr",
        "l1_distance",
        "l2_distance",
        "top3_overlap_ratio",
        "wasserstein_importance_distribution",
        "mean_wasserstein",
        "max_wasserstein",
        "individual_top3_features",
        "group_top3_features",
        "most_different_feature",
        "most_different_abs_diff",
        "most_different_feature_wasserstein",
        "status",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# SHAP similarity under 0307 main tiebreak scheme\n\n")
        f.write("This analysis compares function-level SHAP importance profiles against the assigned group-level SHAP reference.\n\n")
        f.write("Scheme: `eps2bins_3_r0307_main_tiebreak`\n\n")
        f.write("Important: ambiguous-on-tie and mean-based variants are not used.\n\n")
        f.write("BBOB individual SHAP source: `output/shap_individual_manual_bins_niles/`.\n\n")
        f.write("Wasserstein note: current files contain summary-level importance values rather than raw SHAP-value distributions. The Wasserstein columns are therefore computed from normalized summary importance vectors.\n\n")

        f.write("## Function-type summary\n\n")
        if type_summary.empty:
            f.write("No successful rows.\n\n")
        else:
            f.write(df_to_markdown_no_tabulate(type_summary))
            f.write("\n\n")

        f.write("## Group summary\n\n")
        if group_summary.empty:
            f.write("No successful rows.\n\n")
        else:
            f.write(df_to_markdown_no_tabulate(group_summary))
            f.write("\n\n")

        f.write("## Per-function similarity\n\n")
        f.write(df_to_markdown_no_tabulate(out[[c for c in display_cols if c in out.columns]]))
        f.write("\n\n")

        f.write("## Error rows\n\n")
        errors = out[~out["status"].eq("ok")].copy()
        if errors.empty:
            f.write("No error rows.\n")
        else:
            f.write(
                df_to_markdown_no_tabulate(
                    errors[
                        [
                            "function_type",
                            "function_id",
                            "assigned_group_0307",
                            "status",
                            "note",
                        ]
                    ]
                )
            )
            f.write("\n")

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("Step 10 SHAP similarity analysis under the 0307 main grouping scheme\n")
        f.write("\n")
        f.write("Scheme used:\n")
        f.write("- eps2bins_3_r0307_main_tiebreak\n")
        f.write("\n")
        f.write("This script intentionally uses:\n")
        f.write(f"- {HALL_OF_FAME_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {GROUP_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {BBOB_INDIVIDUAL_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {AFFINE_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {LLAMEA_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("It does not use ambiguous_on_tie, mean_based, eps2bins_3_r025065, or encoding SHAP outputs.\n")
        f.write("\n")
        f.write("Method:\n")
        f.write("- Read Hall of Fame rows and remove aggregate All rows.\n")
        f.write("- For each function, read assigned_group_0307.\n")
        f.write("- Load the corresponding group-level SHAP reference shap_group_k.csv.\n")
        f.write("- Load the individual SHAP importance file for BBOB, affine, or LLaMEA functions.\n")
        f.write("- Convert both into normalized feature-importance vectors.\n")
        f.write("- Compute cosine similarity, Spearman rank correlation, L1/L2 distance, top-3 overlap, largest per-feature difference, and Wasserstein-style summary metrics.\n")
        f.write("\n")
        f.write("Wasserstein note:\n")
        f.write("- True per-feature Wasserstein over raw SHAP distributions requires raw SHAP values.\n")
        f.write("- Current inputs are summary-level importance CSV files.\n")
        f.write("- Therefore mean_wasserstein / median_wasserstein / max_wasserstein are computed from per-feature absolute differences.\n")
        f.write("- wasserstein_importance_distribution compares the distribution of normalized importance magnitudes across features.\n")
        f.write("\n")
        f.write("Outputs:\n")
        f.write(f"- {OUT_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_TYPE_SUMMARY_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_GROUP_SUMMARY_CSV.relative_to(PROJECT_ROOT)}\n")

    print("\nDone.")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved MD:  {OUT_MD}")
    print(f"Saved type summary:  {OUT_TYPE_SUMMARY_CSV}")
    print(f"Saved group summary: {OUT_GROUP_SUMMARY_CSV}")
    print(f"Saved README: {README_FILE}")

    print("\nStatus counts:")
    print(out["status"].value_counts(dropna=False).to_string())

    ok = out[out["status"].eq("ok")].copy()

    print("\nSuccessful rows:")
    if ok.empty:
        print("No successful rows.")
    else:
        show_cols = [
            "function_type",
            "function_id",
            "assigned_group_0307",
            "cosine_similarity",
            "spearman_rank_corr",
            "top3_overlap_ratio",
            "wasserstein_importance_distribution",
            "mean_wasserstein",
            "most_different_feature",
            "actual_auc_mean",
        ]
        print(ok[show_cols].to_string(index=False))

    errors = out[~out["status"].eq("ok")].copy()
    if not errors.empty:
        print("\nError rows:")
        print(
            errors[
                [
                    "function_type",
                    "function_id",
                    "assigned_group_0307",
                    "status",
                    "note",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()