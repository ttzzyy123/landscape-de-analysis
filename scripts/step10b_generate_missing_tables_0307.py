from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Step 10B: Generate missing comparison tables for 0307 main scheme
#
# This script complements:
#   scripts/step10_shap_similarity_0307.py
#
# It generates:
#   Table 10: predicted group configuration vs actual best configuration
#   Table 11: predicted group performance vs actual generated-function performance
#   Table 12: per-feature SHAP difference table
#   Table 13: KL / JS divergence table
#
# Scheme:
#   eps2bins_3_r0307_main_tiebreak
#
# Important:
#   This script does NOT use ambiguous_on_tie or mean_based outputs.
#   This script does NOT use encoding SHAP outputs.
#
# Main idea:
#   - Group-level prediction is estimated from BBOB original functions
#     within each assigned 0307 group.
#   - Generated functions are affine + LLaMEA functions.
#   - Per-feature differences and divergence metrics compare each
#     individual function SHAP vector against its assigned group SHAP vector.
# ============================================================


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_DIR = PROJECT_ROOT / "output" / "0307_main_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

HALL_OF_FAME_CSV = COMPARISON_DIR / "hall_of_fame_0307.csv"
SHAP_SIMILARITY_CSV = COMPARISON_DIR / "shap_similarity_0307.csv"

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

OUT_CONFIG_CSV = COMPARISON_DIR / "table10_predicted_group_config_vs_actual_0307.csv"
OUT_CONFIG_MD = COMPARISON_DIR / "table10_predicted_group_config_vs_actual_0307.md"

OUT_PERFORMANCE_CSV = COMPARISON_DIR / "table11_predicted_group_performance_vs_actual_0307.csv"
OUT_PERFORMANCE_MD = COMPARISON_DIR / "table11_predicted_group_performance_vs_actual_0307.md"

OUT_FEATURE_DIFF_CSV = COMPARISON_DIR / "table12_per_feature_shap_difference_0307.csv"
OUT_FEATURE_DIFF_MD = COMPARISON_DIR / "table12_per_feature_shap_difference_generated_0307.md"

OUT_DIVERGENCE_CSV = COMPARISON_DIR / "table13_kl_js_divergence_0307.csv"
OUT_DIVERGENCE_MD = COMPARISON_DIR / "table13_kl_js_divergence_generated_0307.md"

README_FILE = COMPARISON_DIR / "README_step10b_missing_tables_0307.txt"


# =========================
# Columns
# =========================
CONFIG_COLS = [
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

GENERATED_TYPES = ["affine", "llamea_generated"]


# =========================
# Basic helpers
# =========================
def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    total = float(np.sum(np.abs(v)))
    if total <= 0:
        return np.zeros_like(v)

    return v / total


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


def feature_name_cleanup(x):
    return str(x).strip()


def detect_value_column(df):
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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"Missing 'feature' column in {path}")

    value_col = detect_value_column(df)

    out = {}
    for _, r in df.iterrows():
        feature = feature_name_cleanup(r["feature"])
        value = safe_float(r[value_col])

        if not np.isfinite(value):
            continue

        out[feature] = out.get(feature, 0.0) + float(value)

    features = list(out.keys())
    vals = normalize_vector([out[f] for f in features])
    out = {f: float(v) for f, v in zip(features, vals)}

    return out, path, value_col


def load_group_vector(group_id):
    gnum = parse_group_number(group_id)

    if gnum is None:
        raise ValueError(f"Invalid group id: {group_id}")

    path = GROUP_SHAP_DIR / f"shap_group_{gnum}.csv"
    return read_importance_vector(path)


def load_bbob_vector(function_id):
    fid_str = str(function_id).strip()

    if fid_str.startswith("f"):
        fid_str = fid_str[1:]

    fid = int(fid_str)
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


def load_individual_vector(function_type, function_id):
    if function_type == "bbob_original":
        return load_bbob_vector(function_id)

    if function_type == "affine":
        return load_affine_vector(function_id)

    if function_type == "llamea_generated":
        return load_llamea_vector(function_id)

    raise ValueError(f"Unsupported function_type: {function_type}")


def align_vectors(group_vec, individual_vec):
    features = sorted(set(group_vec.keys()) | set(individual_vec.keys()))

    g = np.array([group_vec.get(f, 0.0) for f in features], dtype=float)
    i = np.array([individual_vec.get(f, 0.0) for f in features], dtype=float)

    g = normalize_vector(g)
    i = normalize_vector(i)

    return features, g, i


def kl_divergence(p, q, eps=1e-12):
    """
    KL(P || Q) with epsilon smoothing.
    Natural log is used.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence with natural log.
    Symmetric and more stable than raw KL.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


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


def canonical_config_value(x):
    """
    Normalize config values for comparison.
    """
    if pd.isna(x):
        return "NA"

    if isinstance(x, float):
        if abs(x - round(x)) < 1e-10:
            return str(int(round(x)))
        return f"{x:.10g}"

    return str(x).strip()


def config_signature(row, config_cols=CONFIG_COLS):
    parts = []

    for c in config_cols:
        if c not in row.index:
            continue
        parts.append(f"{c}={canonical_config_value(row[c])}")

    return "|".join(parts)


def compare_config_values(pred, actual):
    """
    Return True if two config values match.
    Numeric values use small tolerance.
    """
    if pd.isna(pred) and pd.isna(actual):
        return True

    if pd.isna(pred) or pd.isna(actual):
        return False

    pred_float = safe_float(pred)
    actual_float = safe_float(actual)

    if np.isfinite(pred_float) and np.isfinite(actual_float):
        return bool(abs(pred_float - actual_float) <= 1e-9)

    return str(pred).strip() == str(actual).strip()


# =========================
# Table 10
# Predicted group configuration vs actual best configuration
# =========================
def build_group_predicted_config_table(hof):
    """
    Build one predicted configuration per landscape group from BBOB original
    Hall-of-Fame rows.

    Rule:
      1. Use only bbob_original rows, excluding All.
      2. For each assigned group, count the best-config signatures.
      3. Choose the most frequent config.
      4. If tied, choose the config with the highest mean actual_auc_mean.
    """
    bbob = hof[
        (hof["function_type"].eq("bbob_original"))
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for group_id, sub in bbob.groupby("assigned_group_0307", sort=True):
        if pd.isna(group_id) or str(group_id).strip() == "":
            continue

        sub = sub.copy()
        sub["config_signature"] = sub.apply(config_signature, axis=1)

        agg_rows = []
        for sig, ss in sub.groupby("config_signature", sort=False):
            rep = ss.iloc[0]

            item = {
                "assigned_group_0307": group_id,
                "predicted_config_signature": sig,
                "support_n_bbob_functions": len(ss),
                "support_bbob_functions": ",".join(ss["function_id"].astype(str).tolist()),
                "support_mean_auc": ss["actual_auc_mean"].mean(),
                "support_median_auc": ss["actual_auc_mean"].median(),
            }

            for c in CONFIG_COLS:
                item[f"predicted_{c}"] = rep[c] if c in rep.index else np.nan

            agg_rows.append(item)

        agg = pd.DataFrame(agg_rows)

        agg = agg.sort_values(
            ["support_n_bbob_functions", "support_mean_auc"],
            ascending=[False, False],
        )

        rows.append(agg.iloc[0].to_dict())

    return pd.DataFrame(rows)


def build_table10_config_prediction(hof, group_pred_config):
    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for _, r in generated.iterrows():
        group_id = r["assigned_group_0307"]
        pred = group_pred_config[
            group_pred_config["assigned_group_0307"].astype(str).eq(str(group_id))
        ]

        item = {
            "function_type": r["function_type"],
            "function_id": r["function_id"],
            "assigned_group_0307": group_id,
            "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
            "actual_config_signature": config_signature(r),
        }

        if pred.empty:
            item["status"] = "error"
            item["note"] = f"No predicted group config found for {group_id}"
            rows.append(item)
            continue

        pred_row = pred.iloc[0]

        item["predicted_config_signature"] = pred_row["predicted_config_signature"]
        item["support_n_bbob_functions"] = pred_row["support_n_bbob_functions"]
        item["support_bbob_functions"] = pred_row["support_bbob_functions"]
        item["support_mean_auc"] = pred_row["support_mean_auc"]
        item["support_median_auc"] = pred_row["support_median_auc"]

        match_count = 0
        available_count = 0
        mismatch_cols = []

        for c in CONFIG_COLS:
            pred_col = f"predicted_{c}"
            actual_col = f"actual_{c}"

            item[pred_col] = pred_row.get(pred_col, np.nan)
            item[actual_col] = r.get(c, np.nan)

            if c in r.index:
                available_count += 1

                if compare_config_values(item[pred_col], item[actual_col]):
                    match_count += 1
                else:
                    mismatch_cols.append(c)

        item["config_match_count"] = match_count
        item["config_available_count"] = available_count
        item["config_match_ratio"] = (
            match_count / available_count if available_count > 0 else np.nan
        )
        item["mismatch_config_fields"] = ",".join(mismatch_cols)
        item["status"] = "ok"
        item["note"] = (
            "Predicted config is the most frequent BBOB Hall-of-Fame config "
            "within the assigned 0307 group; ties are broken by support mean AUC."
        )

        rows.append(item)

    return pd.DataFrame(rows)


# =========================
# Table 11
# Predicted group performance vs actual generated performance
# =========================
def build_group_performance_reference(hof):
    bbob = hof[
        (hof["function_type"].eq("bbob_original"))
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for group_id, sub in bbob.groupby("assigned_group_0307", sort=True):
        auc = pd.to_numeric(sub["actual_auc_mean"], errors="coerce").dropna()

        if auc.empty:
            continue

        rows.append(
            {
                "assigned_group_0307": group_id,
                "group_n_bbob_functions": len(auc),
                "group_bbob_functions": ",".join(sub["function_id"].astype(str).tolist()),
                "group_mean_auc": float(auc.mean()),
                "group_median_auc": float(auc.median()),
                "group_std_auc": float(auc.std(ddof=0)),
                "group_min_auc": float(auc.min()),
                "group_max_auc": float(auc.max()),
            }
        )

    return pd.DataFrame(rows)


def build_table11_performance_prediction(hof, group_perf):
    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for _, r in generated.iterrows():
        group_id = r["assigned_group_0307"]
        ref = group_perf[
            group_perf["assigned_group_0307"].astype(str).eq(str(group_id))
        ]

        item = {
            "function_type": r["function_type"],
            "function_id": r["function_id"],
            "assigned_group_0307": group_id,
            "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
            "actual_auc_std": safe_float(r.get("actual_auc_std", np.nan)),
        }

        if ref.empty:
            item["status"] = "error"
            item["note"] = f"No BBOB performance reference found for {group_id}"
            rows.append(item)
            continue

        ref_row = ref.iloc[0]
        actual = item["actual_auc_mean"]

        for c in [
            "group_n_bbob_functions",
            "group_bbob_functions",
            "group_mean_auc",
            "group_median_auc",
            "group_std_auc",
            "group_min_auc",
            "group_max_auc",
        ]:
            item[c] = ref_row[c]

        item["actual_minus_group_mean_auc"] = actual - ref_row["group_mean_auc"]
        item["actual_minus_group_median_auc"] = actual - ref_row["group_median_auc"]
        item["abs_error_from_group_mean_auc"] = abs(actual - ref_row["group_mean_auc"])
        item["abs_error_from_group_median_auc"] = abs(actual - ref_row["group_median_auc"])

        item["inside_group_auc_range"] = bool(
            actual >= ref_row["group_min_auc"] and actual <= ref_row["group_max_auc"]
        )

        if ref_row["group_std_auc"] > 0:
            item["z_score_vs_group_mean_auc"] = (
                actual - ref_row["group_mean_auc"]
            ) / ref_row["group_std_auc"]
        else:
            item["z_score_vs_group_mean_auc"] = np.nan

        item["status"] = "ok"
        item["note"] = (
            "Group performance prediction is based on BBOB original Hall-of-Fame "
            "AUC values within the assigned 0307 group."
        )

        rows.append(item)

    return pd.DataFrame(rows)


# =========================
# Table 12
# Per-feature SHAP difference
# =========================
def build_table12_per_feature_difference(hof):
    base = hof[~hof["function_id"].astype(str).eq("All")].copy()

    rows = []

    for _, r in base.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        try:
            group_vec, group_path, group_value_col = load_group_vector(group_id)
            individual_vec, individual_path, individual_value_col = load_individual_vector(
                function_type,
                function_id,
            )

            features, g, i = align_vectors(group_vec, individual_vec)

            for feature, gv, iv in zip(features, g, i):
                rows.append(
                    {
                        "function_type": function_type,
                        "function_id": function_id,
                        "assigned_group_0307": group_id,
                        "feature": feature,
                        "group_importance_norm": float(gv),
                        "individual_importance_norm": float(iv),
                        "signed_difference_individual_minus_group": float(iv - gv),
                        "abs_difference": float(abs(iv - gv)),
                        "per_feature_wasserstein": float(abs(iv - gv)),
                        "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
                        "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                        "source_individual_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                        "group_value_column": group_value_col,
                        "individual_value_column": individual_value_col,
                        "status": "ok",
                        "note": (
                            "Per-feature Wasserstein is computed from one summary "
                            "importance value per feature, so it equals abs_difference."
                        ),
                    }
                )

        except Exception as e:
            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "assigned_group_0307": group_id,
                    "feature": "",
                    "group_importance_norm": np.nan,
                    "individual_importance_norm": np.nan,
                    "signed_difference_individual_minus_group": np.nan,
                    "abs_difference": np.nan,
                    "per_feature_wasserstein": np.nan,
                    "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
                    "source_group_shap": "",
                    "source_individual_shap": "",
                    "group_value_column": "",
                    "individual_value_column": "",
                    "status": "error",
                    "note": str(e),
                }
            )

    return pd.DataFrame(rows)


# =========================
# Table 13
# KL / JS divergence
# =========================
def build_table13_divergence(hof):
    base = hof[~hof["function_id"].astype(str).eq("All")].copy()

    rows = []

    for _, r in base.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        try:
            group_vec, group_path, group_value_col = load_group_vector(group_id)
            individual_vec, individual_path, individual_value_col = load_individual_vector(
                function_type,
                function_id,
            )

            features, g, i = align_vectors(group_vec, individual_vec)

            kl_individual_to_group = kl_divergence(i, g)
            kl_group_to_individual = kl_divergence(g, i)
            js = js_divergence(i, g)
            sqrt_js = float(np.sqrt(js))

            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "assigned_group_0307": group_id,
                    "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
                    "kl_individual_to_group": kl_individual_to_group,
                    "kl_group_to_individual": kl_group_to_individual,
                    "js_divergence": js,
                    "sqrt_js_divergence": sqrt_js,
                    "n_features": len(features),
                    "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                    "source_individual_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                    "group_value_column": group_value_col,
                    "individual_value_column": individual_value_col,
                    "status": "ok",
                    "note": (
                        "KL and JS are computed from normalized summary SHAP "
                        "importance vectors with epsilon smoothing."
                    ),
                }
            )

        except Exception as e:
            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "assigned_group_0307": group_id,
                    "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
                    "kl_individual_to_group": np.nan,
                    "kl_group_to_individual": np.nan,
                    "js_divergence": np.nan,
                    "sqrt_js_divergence": np.nan,
                    "n_features": np.nan,
                    "source_group_shap": "",
                    "source_individual_shap": "",
                    "group_value_column": "",
                    "individual_value_column": "",
                    "status": "error",
                    "note": str(e),
                }
            )

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("STEP 10B: GENERATE MISSING TABLES UNDER 0307 MAIN TIEBREAK SCHEME")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Hall of Fame: {HALL_OF_FAME_CSV}")
    print(f"SHAP similarity: {SHAP_SIMILARITY_CSV}")
    print(f"Group SHAP dir: {GROUP_SHAP_DIR}")
    print(f"BBOB individual SHAP dir: {BBOB_INDIVIDUAL_SHAP_DIR}")
    print(f"Affine SHAP dir: {AFFINE_SHAP_DIR}")
    print(f"LLaMEA SHAP dir: {LLAMEA_SHAP_DIR}")
    print(f"Output dir: {COMPARISON_DIR}")

    if not HALL_OF_FAME_CSV.exists():
        raise FileNotFoundError(f"Missing Hall of Fame file: {HALL_OF_FAME_CSV}")

    hof = pd.read_csv(HALL_OF_FAME_CSV)

    # --------------------------------------------------------
    # Table 10
    # --------------------------------------------------------
    print("\nBuilding Table 10: predicted group config vs actual best config...")

    group_pred_config = build_group_predicted_config_table(hof)
    table10 = build_table10_config_prediction(hof, group_pred_config)

    table10.to_csv(OUT_CONFIG_CSV, index=False)

    table10_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
        "config_match_ratio",
        "config_match_count",
        "config_available_count",
        "mismatch_config_fields",
        "support_n_bbob_functions",
        "support_bbob_functions",
        "support_mean_auc",
        "actual_config_signature",
        "predicted_config_signature",
        "status",
    ]

    with open(OUT_CONFIG_MD, "w", encoding="utf-8") as f:
        f.write("# Table 10. Predicted group configuration vs actual best configuration\n\n")
        f.write("Predicted group configuration is defined as the most frequent BBOB Hall-of-Fame configuration within the assigned 0307 group. Ties are broken by mean AUC.\n\n")
        f.write(df_to_markdown_no_tabulate(table10[[c for c in table10_display_cols if c in table10.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 11
    # --------------------------------------------------------
    print("Building Table 11: predicted group performance vs actual performance...")

    group_perf = build_group_performance_reference(hof)
    table11 = build_table11_performance_prediction(hof, group_perf)

    table11.to_csv(OUT_PERFORMANCE_CSV, index=False)

    table11_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
        "group_mean_auc",
        "group_median_auc",
        "group_min_auc",
        "group_max_auc",
        "actual_minus_group_mean_auc",
        "abs_error_from_group_mean_auc",
        "inside_group_auc_range",
        "z_score_vs_group_mean_auc",
        "group_bbob_functions",
        "status",
    ]

    with open(OUT_PERFORMANCE_MD, "w", encoding="utf-8") as f:
        f.write("# Table 11. Predicted group performance vs actual generated-function performance\n\n")
        f.write("Group performance prediction is estimated from BBOB Hall-of-Fame AUC values within the assigned 0307 group.\n\n")
        f.write(df_to_markdown_no_tabulate(table11[[c for c in table11_display_cols if c in table11.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 12
    # --------------------------------------------------------
    print("Building Table 12: per-feature SHAP difference table...")

    table12 = build_table12_per_feature_difference(hof)
    table12.to_csv(OUT_FEATURE_DIFF_CSV, index=False)

    table12_generated = table12[
        table12["function_type"].isin(GENERATED_TYPES)
        & table12["status"].eq("ok")
    ].copy()

    table12_generated = table12_generated.sort_values(
        ["function_type", "function_id", "abs_difference"],
        ascending=[True, True, False],
    )

    table12_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "feature",
        "group_importance_norm",
        "individual_importance_norm",
        "signed_difference_individual_minus_group",
        "abs_difference",
        "per_feature_wasserstein",
        "actual_auc_mean",
    ]

    with open(OUT_FEATURE_DIFF_MD, "w", encoding="utf-8") as f:
        f.write("# Table 12. Per-feature SHAP difference for generated functions\n\n")
        f.write("This markdown view only shows affine and LLaMEA generated functions. The CSV contains all BBOB, affine, and LLaMEA functions.\n\n")
        f.write(df_to_markdown_no_tabulate(table12_generated[[c for c in table12_display_cols if c in table12_generated.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 13
    # --------------------------------------------------------
    print("Building Table 13: KL / JS divergence table...")

    table13 = build_table13_divergence(hof)
    table13.to_csv(OUT_DIVERGENCE_CSV, index=False)

    table13_generated = table13[
        table13["function_type"].isin(GENERATED_TYPES)
        & table13["status"].eq("ok")
    ].copy()

    table13_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
        "kl_individual_to_group",
        "kl_group_to_individual",
        "js_divergence",
        "sqrt_js_divergence",
        "n_features",
        "status",
    ]

    with open(OUT_DIVERGENCE_MD, "w", encoding="utf-8") as f:
        f.write("# Table 13. KL and JS divergence for generated functions\n\n")
        f.write("KL and JS divergence are computed from normalized summary SHAP importance vectors with epsilon smoothing.\n\n")
        f.write(df_to_markdown_no_tabulate(table13_generated[[c for c in table13_display_cols if c in table13_generated.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("Step 10B missing table generation under 0307 main tiebreak scheme\n")
        f.write("\n")
        f.write("Generated tables:\n")
        f.write(f"- Table 10: {OUT_CONFIG_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 11: {OUT_PERFORMANCE_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 12: {OUT_FEATURE_DIFF_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 13: {OUT_DIVERGENCE_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("Markdown previews:\n")
        f.write(f"- {OUT_CONFIG_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_PERFORMANCE_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_FEATURE_DIFF_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {OUT_DIVERGENCE_MD.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("Scheme:\n")
        f.write("- eps2bins_3_r0307_main_tiebreak\n")
        f.write("\n")
        f.write("Inputs:\n")
        f.write(f"- {HALL_OF_FAME_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {GROUP_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {BBOB_INDIVIDUAL_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {AFFINE_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- {LLAMEA_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("Definitions:\n")
        f.write("- Predicted group configuration: the most frequent BBOB Hall-of-Fame configuration within the assigned group; ties are broken by mean AUC.\n")
        f.write("- Predicted group performance: BBOB Hall-of-Fame AUC distribution within the assigned group.\n")
        f.write("- Per-feature Wasserstein: since each feature only has one summary importance value, this equals abs(group_importance_norm - individual_importance_norm).\n")
        f.write("- KL / JS divergence: computed from normalized summary SHAP importance vectors with epsilon smoothing.\n")
        f.write("\n")
        f.write("Important limitations:\n")
        f.write("- These are summary-level comparisons, not raw SHAP-value distribution comparisons.\n")
        f.write("- Configuration prediction is based on BBOB Hall-of-Fame rows only.\n")
        f.write("- Generated functions are compared against their assigned 0307 group references.\n")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDone.")
    print(f"Saved Table 10 CSV: {OUT_CONFIG_CSV}")
    print(f"Saved Table 11 CSV: {OUT_PERFORMANCE_CSV}")
    print(f"Saved Table 12 CSV: {OUT_FEATURE_DIFF_CSV}")
    print(f"Saved Table 13 CSV: {OUT_DIVERGENCE_CSV}")
    print(f"Saved README: {README_FILE}")

    print("\nTable 10 generated rows:")
    print(
        table10[
            [
                "function_type",
                "function_id",
                "assigned_group_0307",
                "config_match_ratio",
                "mismatch_config_fields",
                "actual_auc_mean",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nTable 11 generated rows:")
    print(
        table11[
            [
                "function_type",
                "function_id",
                "assigned_group_0307",
                "actual_auc_mean",
                "group_mean_auc",
                "actual_minus_group_mean_auc",
                "inside_group_auc_range",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nTable 12 status counts:")
    print(table12["status"].value_counts(dropna=False).to_string())

    print("\nTable 13 generated rows:")
    print(
        table13[
            table13["function_type"].isin(GENERATED_TYPES)
        ][
            [
                "function_type",
                "function_id",
                "assigned_group_0307",
                "kl_individual_to_group",
                "kl_group_to_individual",
                "js_divergence",
                "sqrt_js_divergence",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()