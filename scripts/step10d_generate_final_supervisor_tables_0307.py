from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Step 10D: Generate final supervisor-requested tables
# for 0307 main tiebreak grouping scheme
#
# Generates the final 5 tables:
#
# Table 1:
#   BBOB group-level SHAP distance table
#   Per group, per hyperparameter:
#   average distance between group SHAP values and BBOB function SHAP values.
#
# Table 2:
#   Generated functions predicted vs actual Hall-of-Fame table
#   For affine and LLaMEA functions:
#   group Hall-of-Fame winner as predicted config
#   vs function Hall-of-Fame winner as actual config.
#
# Table 3:
#   Generated functions per-function per-hyperparameter SHAP shape-distance table
#   Compares the shape of group SHAP distribution vs function SHAP distribution.
#
# Table 4:
#   Generated functions per-function per-hyperparameter SHAP value-distance table
#   Compares SHAP value magnitude and direction between assigned group and function.
#
# Table 5:
#   Generated functions SHAP similarity summary table
#   Summarises generated-function similarity to assigned group.
#
# Scheme:
#   eps2bins_3_r0307_main_tiebreak
#
# Important:
#   This script prefers raw SHAP-value distributions when available.
#   If raw SHAP values cannot be found, it falls back to summary-level
#   SHAP importance values. In fallback mode, distribution-shape distance
#   is only an approximation and should be described as such.
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


# =========================
# Output files
# =========================
OUT_TABLE1_CSV = COMPARISON_DIR / "table01_bbob_group_hyperparameter_shap_distance_0307.csv"
OUT_TABLE1_MD = COMPARISON_DIR / "table01_bbob_group_hyperparameter_shap_distance_0307.md"

OUT_TABLE2_CSV = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_0307.csv"
OUT_TABLE2_MD = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_0307.md"

OUT_TABLE3_CSV = COMPARISON_DIR / "table03_generated_shap_shape_distance_0307.csv"
OUT_TABLE3_MD = COMPARISON_DIR / "table03_generated_shap_shape_distance_0307.md"

OUT_TABLE4_CSV = COMPARISON_DIR / "table04_generated_shap_value_distance_0307.csv"
OUT_TABLE4_MD = COMPARISON_DIR / "table04_generated_shap_value_distance_0307.md"

OUT_TABLE5_CSV = COMPARISON_DIR / "table05_generated_shap_similarity_summary_0307.csv"
OUT_TABLE5_MD = COMPARISON_DIR / "table05_generated_shap_similarity_summary_0307.md"

README_FILE = COMPARISON_DIR / "README_step10d_final_supervisor_tables_0307.txt"


# =========================
# Constants
# =========================
GENERATED_TYPES = ["affine", "llamea_generated"]

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

# The final SHAP features / hyperparameters we expect.
# The script will also include extra features if they appear in the files.
EXPECTED_SHAP_FEATURES = [
    "CR",
    "F",
    "crossover",
    "lambda_",
    "lpsr",
    "mutation_base",
    "mutation_n_comps",
    "mutation_reference",
    "use_archive",
    "instance",
    "seed",
]


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


def clean_bool_like(x):
    if pd.isna(x):
        return x

    if isinstance(x, bool):
        return x

    s = str(x).strip()

    if s.lower() in ["true", "1", "yes"]:
        return True
    if s.lower() in ["false", "0", "no"]:
        return False

    return x


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

    if s == "" or s.lower() in ["nan", "none", "all"]:
        return None

    if s.startswith("G"):
        s = s[1:]

    try:
        return int(s)
    except ValueError:
        return None


def canonical_group_label(group_value):
    gnum = parse_group_number(group_value)
    if gnum is None:
        return str(group_value)
    return f"G{gnum}"


def feature_name_cleanup(x):
    s = str(x).strip()

    aliases = {
        "archive": "use_archive",
        "Archive": "use_archive",
        "mutation_n_component": "mutation_n_comps",
        "mutation_n_components": "mutation_n_comps",
        "mutation_reference_nan": "mutation_reference",
        "instance variance": "instance",
        "Instance variance": "instance",
        "stochastic variance": "seed",
        "Stochastic variance": "seed",
        "random seed": "seed",
        "seed variance": "seed",
    }

    return aliases.get(s, s)


def short_function_label(row):
    function_type = str(row.get("function_type", ""))
    function_id = str(row.get("function_id", ""))

    if function_type == "affine":
        return function_id

    if function_type == "llamea_generated":
        return function_id

    if function_type == "bbob_original":
        return function_id

    return function_id


def detect_importance_value_column(df):
    candidates = [
        "importance_norm",
        "mean_abs_shap_norm",
        "mean_abs_shap",
        "importance",
        "global_individual_mean_importance",
        "individual_importance_norm",
        "group_importance_norm",
        "abs_mean_shap",
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


def df_to_markdown_no_tabulate(df, max_rows=None):
    out_df = df.copy()

    if max_rows is not None and len(out_df) > max_rows:
        out_df = out_df.head(max_rows).copy()

    cols = list(out_df.columns)
    lines = []

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in out_df.iterrows():
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


# =========================
# SHAP summary importance loading
# =========================
def read_importance_vector(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"Missing 'feature' column in {path}")

    value_col = detect_importance_value_column(df)

    raw = {}
    for _, r in df.iterrows():
        feature = feature_name_cleanup(r["feature"])
        value = safe_float(r[value_col])

        if not np.isfinite(value):
            continue

        raw[feature] = raw.get(feature, 0.0) + float(value)

    features = list(raw.keys())
    vals = normalize_vector([raw[f] for f in features])
    out = {f: float(v) for f, v in zip(features, vals)}

    return out, path, value_col


def load_group_importance_vector(group_id):
    gnum = parse_group_number(group_id)

    if gnum is None:
        raise ValueError(f"Invalid group id: {group_id}")

    path = GROUP_SHAP_DIR / f"shap_group_{gnum}.csv"
    return read_importance_vector(path)


def load_bbob_importance_vector(function_id):
    fid_str = str(function_id).strip()

    if fid_str.startswith("f"):
        fid_str = fid_str[1:]

    fid = int(fid_str)
    path = BBOB_INDIVIDUAL_SHAP_DIR / f"shap_function_{fid}.csv"
    return read_importance_vector(path)


def load_affine_importance_vector(function_id):
    path = (
        AFFINE_SHAP_DIR
        / f"affine_function_{function_id}_alpha_0p9_individual_shap_importance.csv"
    )
    return read_importance_vector(path)


def load_llamea_importance_vector(function_id):
    path = (
        LLAMEA_SHAP_DIR
        / f"real_generated_function_{function_id}_individual_shap_importance.csv"
    )
    return read_importance_vector(path)


def load_individual_importance_vector(function_type, function_id):
    if function_type == "bbob_original":
        return load_bbob_importance_vector(function_id)

    if function_type == "affine":
        return load_affine_importance_vector(function_id)

    if function_type == "llamea_generated":
        return load_llamea_importance_vector(function_id)

    raise ValueError(f"Unsupported function_type: {function_type}")


def align_importance_vectors(group_vec, individual_vec):
    features = sorted(set(group_vec.keys()) | set(individual_vec.keys()))

    g = np.array([group_vec.get(f, 0.0) for f in features], dtype=float)
    i = np.array([individual_vec.get(f, 0.0) for f in features], dtype=float)

    g = normalize_vector(g)
    i = normalize_vector(i)

    return features, g, i


# =========================
# SHAP raw distribution loading
# =========================
def candidate_raw_shap_files_for_group(group_id):
    gnum = parse_group_number(group_id)

    if gnum is None:
        return []

    patterns = [
        f"*group_{gnum}*raw*.csv",
        f"*group_{gnum}*shap_values*.csv",
        f"*group_{gnum}*values*.csv",
        f"*group_{gnum}*long*.csv",
    ]

    files = []
    for p in patterns:
        files.extend(sorted(GROUP_SHAP_DIR.glob(p)))

    # Remove summary importance files from raw candidates.
    files = [
        f for f in files
        if "importance" not in f.name.lower()
        and f.name != f"shap_group_{gnum}.csv"
    ]

    return list(dict.fromkeys(files))


def candidate_raw_shap_files_for_individual(function_type, function_id):
    function_id = str(function_id)

    if function_type == "bbob_original":
        fid = function_id[1:] if function_id.startswith("f") else function_id
        base_dir = BBOB_INDIVIDUAL_SHAP_DIR
        patterns = [
            f"*function_{fid}*raw*.csv",
            f"*function_{fid}*shap_values*.csv",
            f"*function_{fid}*values*.csv",
            f"*function_{fid}*long*.csv",
        ]

    elif function_type == "affine":
        base_dir = AFFINE_SHAP_DIR
        patterns = [
            f"*{function_id}*raw*.csv",
            f"*{function_id}*shap_values*.csv",
            f"*{function_id}*values*.csv",
            f"*{function_id}*long*.csv",
        ]

    elif function_type == "llamea_generated":
        base_dir = LLAMEA_SHAP_DIR
        patterns = [
            f"*{function_id}*raw*.csv",
            f"*{function_id}*shap_values*.csv",
            f"*{function_id}*values*.csv",
            f"*{function_id}*long*.csv",
        ]

    else:
        return []

    files = []
    for p in patterns:
        files.extend(sorted(base_dir.glob(p)))

    files = [
        f for f in files
        if "importance" not in f.name.lower()
    ]

    return list(dict.fromkeys(files))


def read_raw_shap_distribution_from_csv(path):
    """
    Reads raw SHAP distributions from either:
    1. long format:
       feature, shap_value
       feature, value
       variable, shap_value
    2. wide format:
       one column per feature / hyperparameter

    Returns:
       dict: feature -> np.array(values)
    """
    path = Path(path)
    df = pd.read_csv(path)

    lower_cols = {c.lower(): c for c in df.columns}

    feature_col = None
    value_col = None

    for c in ["feature", "variable", "hyperparameter", "parameter"]:
        if c in lower_cols:
            feature_col = lower_cols[c]
            break

    for c in ["shap_value", "shap", "value", "shap_values"]:
        if c in lower_cols:
            value_col = lower_cols[c]
            break

    # Long format.
    if feature_col is not None and value_col is not None:
        out = {}
        for feature, sub in df.groupby(feature_col):
            vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                out[feature_name_cleanup(feature)] = vals
        return out

    # Wide format.
    ignored_cols = {
        "row_id",
        "sample_id",
        "config_id",
        "function_id",
        "function_type",
        "assigned_group_0307",
        "target",
        "prediction",
        "auc",
        "aoc",
        "aocc",
        "actual_auc_mean",
    }

    out = {}
    for c in df.columns:
        if c in ignored_cols or c.lower() in ignored_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                out[feature_name_cleanup(c)] = vals

    if not out:
        raise ValueError(
            f"Could not parse raw SHAP distribution file: {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    return out


def load_raw_group_distribution(group_id):
    files = candidate_raw_shap_files_for_group(group_id)

    last_error = None
    for f in files:
        try:
            dist = read_raw_shap_distribution_from_csv(f)
            if dist:
                return dist, f, "raw"
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"No usable raw group SHAP distribution found for group {group_id}. "
        f"Candidates: {[str(x) for x in files]}. Last error: {last_error}"
    )


def load_raw_individual_distribution(function_type, function_id):
    files = candidate_raw_shap_files_for_individual(function_type, function_id)

    last_error = None
    for f in files:
        try:
            dist = read_raw_shap_distribution_from_csv(f)
            if dist:
                return dist, f, "raw"
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"No usable raw individual SHAP distribution found for "
        f"{function_type} {function_id}. "
        f"Candidates: {[str(x) for x in files]}. Last error: {last_error}"
    )


def summary_importance_as_distribution(vec):
    """
    Fallback distribution representation:
    one pseudo-value per feature.

    This is NOT a true SHAP-value distribution.
    It only allows the pipeline to complete when raw SHAP values
    are unavailable.
    """
    return {k: np.array([float(v)], dtype=float) for k, v in vec.items()}


def load_best_available_group_distribution(group_id):
    try:
        return load_raw_group_distribution(group_id)
    except Exception:
        vec, path, value_col = load_group_importance_vector(group_id)
        return summary_importance_as_distribution(vec), path, f"summary:{value_col}"


def load_best_available_individual_distribution(function_type, function_id):
    try:
        return load_raw_individual_distribution(function_type, function_id)
    except Exception:
        vec, path, value_col = load_individual_importance_vector(function_type, function_id)
        return summary_importance_as_distribution(vec), path, f"summary:{value_col}"


# =========================
# Statistics helpers
# =========================
def wasserstein_1d(x, y):
    """
    Simple 1D Wasserstein distance implementation without scipy.
    For empirical samples with equal sample weights.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    x = np.sort(x)
    y = np.sort(y)

    # Quantile interpolation to common grid.
    n = max(len(x), len(y))
    if n == 1:
        return float(abs(x[0] - y[0]))

    q = np.linspace(0.0, 1.0, n)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)

    return float(np.mean(np.abs(xq - yq)))


def ks_statistic_1d(x, y):
    """
    Two-sample KS statistic without scipy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    values = np.sort(np.unique(np.concatenate([x, y])))

    if len(values) == 0:
        return np.nan

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    cdf_x = np.searchsorted(x_sorted, values, side="right") / len(x_sorted)
    cdf_y = np.searchsorted(y_sorted, values, side="right") / len(y_sorted)

    return float(np.max(np.abs(cdf_x - cdf_y)))


def skewness(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    std = np.std(x)
    if std <= 0:
        return 0.0

    return float(np.mean(((x - np.mean(x)) / std) ** 3))


def kurtosis_excess(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    std = np.std(x)
    if std <= 0:
        return 0.0

    return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3.0)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom <= 0:
        return np.nan

    return float(np.dot(a, b) / denom)


def rankdata_simple(x):
    """
    Simple average-rank implementation.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1

        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def spearman_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return np.nan

    ra = rankdata_simple(a)
    rb = rankdata_simple(b)

    if np.std(ra) <= 0 or np.std(rb) <= 0:
        return np.nan

    return float(np.corrcoef(ra, rb)[0, 1])


def top_k_overlap(a, b, k=3):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) == 0:
        return np.nan

    k = min(k, len(a))

    top_a = set(np.argsort(-np.abs(a))[:k])
    top_b = set(np.argsort(-np.abs(b))[:k])

    return float(len(top_a & top_b) / k)


# =========================
# Config helpers
# =========================
def canonical_config_value(x):
    if pd.isna(x):
        return "NA"

    x = clean_bool_like(x)

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
    if pd.isna(pred) and pd.isna(actual):
        return True

    if pd.isna(pred) or pd.isna(actual):
        return False

    pred = clean_bool_like(pred)
    actual = clean_bool_like(actual)

    pred_float = safe_float(pred)
    actual_float = safe_float(actual)

    if np.isfinite(pred_float) and np.isfinite(actual_float):
        return bool(abs(pred_float - actual_float) <= 1e-9)

    return str(pred).strip() == str(actual).strip()


def build_group_hof_winner_table(hof):
    """
    Predicted group Hall-of-Fame winner.

    Rule:
      - Use BBOB original functions only.
      - Exclude All rows.
      - Within each assigned group, select the BBOB function Hall-of-Fame
        row with the highest actual_auc_mean.
      - This row is the group's predicted best configuration.
    """
    bbob = hof[
        (hof["function_type"].eq("bbob_original"))
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    bbob["actual_auc_mean"] = pd.to_numeric(bbob["actual_auc_mean"], errors="coerce")

    rows = []

    for group_id, sub in bbob.groupby("assigned_group_0307", sort=True):
        sub = sub.dropna(subset=["actual_auc_mean"]).copy()

        if sub.empty:
            continue

        winner = sub.sort_values("actual_auc_mean", ascending=False).iloc[0]

        item = {
            "assigned_group_0307": group_id,
            "predicted_source_function_type": winner["function_type"],
            "predicted_source_function_id": winner["function_id"],
            "predicted_source_auc_mean": safe_float(winner["actual_auc_mean"]),
            "group_n_bbob_functions": len(sub),
            "group_bbob_functions": ",".join(sub["function_id"].astype(str).tolist()),
            "predicted_config_signature": config_signature(winner),
        }

        for c in CONFIG_COLS:
            item[c] = winner.get(c, np.nan)

        rows.append(item)

    return pd.DataFrame(rows)


# =========================
# Table 1
# BBOB group-level SHAP distance table
# =========================
def build_table1_bbob_group_shap_distance(hof):
    bbob = hof[
        (hof["function_type"].eq("bbob_original"))
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    per_function_rows = []

    for _, r in bbob.iterrows():
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        try:
            group_vec, group_path, group_value_col = load_group_importance_vector(group_id)
            individual_vec, individual_path, individual_value_col = load_individual_importance_vector(
                "bbob_original",
                function_id,
            )

            features, g, i = align_importance_vectors(group_vec, individual_vec)

            for feature, gv, iv in zip(features, g, i):
                per_function_rows.append(
                    {
                        "assigned_group_0307": canonical_group_label(group_id),
                        "function_id": function_id,
                        "hyperparameter": feature,
                        "group_shap_importance_norm": float(gv),
                        "function_shap_importance_norm": float(iv),
                        "signed_difference_function_minus_group": float(iv - gv),
                        "distance": float(abs(iv - gv)),
                        "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                        "source_function_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                        "group_value_column": group_value_col,
                        "function_value_column": individual_value_col,
                        "status": "ok",
                        "note": "",
                    }
                )

        except Exception as e:
            per_function_rows.append(
                {
                    "assigned_group_0307": canonical_group_label(group_id),
                    "function_id": function_id,
                    "hyperparameter": "",
                    "group_shap_importance_norm": np.nan,
                    "function_shap_importance_norm": np.nan,
                    "signed_difference_function_minus_group": np.nan,
                    "distance": np.nan,
                    "source_group_shap": "",
                    "source_function_shap": "",
                    "group_value_column": "",
                    "function_value_column": "",
                    "status": "error",
                    "note": str(e),
                }
            )

    detail = pd.DataFrame(per_function_rows)

    ok = detail[detail["status"].eq("ok")].copy()

    table1 = (
        ok.groupby(["assigned_group_0307", "hyperparameter"], as_index=False)
        .agg(
            n_functions=("function_id", "nunique"),
            mean_distance=("distance", "mean"),
            std_distance=("distance", "std"),
            min_distance=("distance", "min"),
            max_distance=("distance", "max"),
            mean_signed_difference=("signed_difference_function_minus_group", "mean"),
            group_mean_shap_importance_norm=("group_shap_importance_norm", "mean"),
            function_mean_shap_importance_norm=("function_shap_importance_norm", "mean"),
            functions=("function_id", lambda x: ",".join(sorted(x.astype(str).unique()))),
        )
    )

    table1["std_distance"] = table1["std_distance"].fillna(0.0)

    table1 = table1.sort_values(
        ["assigned_group_0307", "mean_distance", "hyperparameter"],
        ascending=[True, False, True],
    )

    return table1, detail


# =========================
# Table 2
# Generated predicted vs actual Hall-of-Fame table
# =========================
def build_table2_generated_predicted_vs_actual_hof(hof):
    group_winners = build_group_hof_winner_table(hof)

    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    generated["actual_auc_mean"] = pd.to_numeric(
        generated["actual_auc_mean"],
        errors="coerce",
    )

    rows = []

    for _, r in generated.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        pred_match = group_winners[
            group_winners["assigned_group_0307"].astype(str).eq(str(group_id))
        ]

        if pred_match.empty:
            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "assigned_group_0307": canonical_group_label(group_id),
                    "config_source": "group HoF predicted",
                    "auc_mean": np.nan,
                    "delta_auc_actual_minus_predicted": np.nan,
                    "config_match_count": np.nan,
                    "config_available_count": np.nan,
                    "config_match_ratio": np.nan,
                    "mismatch_config_fields": "",
                    "status": "error",
                    "note": f"No group Hall-of-Fame winner found for {group_id}",
                }
            )
            continue

        pred = pred_match.iloc[0]

        # Compare config fields once.
        match_count = 0
        available_count = 0
        mismatch_cols = []

        for c in CONFIG_COLS:
            if c in r.index:
                available_count += 1
                if compare_config_values(pred.get(c, np.nan), r.get(c, np.nan)):
                    match_count += 1
                else:
                    mismatch_cols.append(c)

        match_ratio = match_count / available_count if available_count > 0 else np.nan
        actual_auc = safe_float(r.get("actual_auc_mean", np.nan))
        predicted_auc = safe_float(pred.get("predicted_source_auc_mean", np.nan))
        delta_auc = actual_auc - predicted_auc

        common = {
            "function_type": function_type,
            "function_id": function_id,
            "function_label": short_function_label(r),
            "assigned_group_0307": canonical_group_label(group_id),
            "predicted_source_bbob_function": pred.get("predicted_source_function_id", ""),
            "group_n_bbob_functions": pred.get("group_n_bbob_functions", np.nan),
            "group_bbob_functions": pred.get("group_bbob_functions", ""),
            "delta_auc_actual_minus_predicted": delta_auc,
            "config_match_count": match_count,
            "config_available_count": available_count,
            "config_match_ratio": match_ratio,
            "mismatch_config_fields": ",".join(mismatch_cols),
            "status": "ok",
        }

        pred_row = {
            **common,
            "config_source": "group HoF predicted",
            "auc_mean": predicted_auc,
            "config_signature": pred.get("predicted_config_signature", ""),
            "note": (
                "Predicted configuration is the Hall-of-Fame winner among "
                "BBOB functions in the assigned group."
            ),
        }

        actual_row = {
            **common,
            "config_source": "function HoF actual",
            "auc_mean": actual_auc,
            "config_signature": config_signature(r),
            "note": "Actual configuration is the function-level Hall-of-Fame winner.",
        }

        for c in CONFIG_COLS:
            pred_row[c] = pred.get(c, np.nan)
            actual_row[c] = r.get(c, np.nan)

        rows.append(pred_row)
        rows.append(actual_row)

    table2 = pd.DataFrame(rows)

    sort_cols = ["function_type", "function_id", "config_source"]
    table2 = table2.sort_values(sort_cols)

    return table2


# =========================
# Table 3
# Generated SHAP shape-distance table
# =========================
def build_table3_generated_shape_distance(hof):
    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for _, r in generated.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        try:
            group_dist, group_path, group_mode = load_best_available_group_distribution(group_id)
            individual_dist, individual_path, individual_mode = load_best_available_individual_distribution(
                function_type,
                function_id,
            )

            features = sorted(set(group_dist.keys()) | set(individual_dist.keys()))

            for feature in features:
                gv = np.asarray(group_dist.get(feature, np.array([0.0])), dtype=float)
                iv = np.asarray(individual_dist.get(feature, np.array([0.0])), dtype=float)

                rows.append(
                    {
                        "function_type": function_type,
                        "function_id": function_id,
                        "function_label": short_function_label(r),
                        "assigned_group_0307": canonical_group_label(group_id),
                        "hyperparameter": feature,
                        "shape_distance_metric": "wasserstein_1d_and_ks_statistic",
                        "wasserstein_distance": wasserstein_1d(gv, iv),
                        "ks_statistic": ks_statistic_1d(gv, iv),
                        "group_n_values": len(gv),
                        "function_n_values": len(iv),
                        "group_std_shap": float(np.std(gv)) if len(gv) > 0 else np.nan,
                        "function_std_shap": float(np.std(iv)) if len(iv) > 0 else np.nan,
                        "group_skewness": skewness(gv),
                        "function_skewness": skewness(iv),
                        "group_kurtosis_excess": kurtosis_excess(gv),
                        "function_kurtosis_excess": kurtosis_excess(iv),
                        "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                        "source_function_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                        "group_distribution_mode": group_mode,
                        "function_distribution_mode": individual_mode,
                        "status": "ok",
                        "note": (
                            "Raw SHAP distributions are used when available. "
                            "If mode starts with summary:, the result is a summary-level fallback."
                        ),
                    }
                )

        except Exception as e:
            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "function_label": short_function_label(r),
                    "assigned_group_0307": canonical_group_label(group_id),
                    "hyperparameter": "",
                    "shape_distance_metric": "wasserstein_1d_and_ks_statistic",
                    "wasserstein_distance": np.nan,
                    "ks_statistic": np.nan,
                    "group_n_values": np.nan,
                    "function_n_values": np.nan,
                    "group_std_shap": np.nan,
                    "function_std_shap": np.nan,
                    "group_skewness": np.nan,
                    "function_skewness": np.nan,
                    "group_kurtosis_excess": np.nan,
                    "function_kurtosis_excess": np.nan,
                    "source_group_shap": "",
                    "source_function_shap": "",
                    "group_distribution_mode": "",
                    "function_distribution_mode": "",
                    "status": "error",
                    "note": str(e),
                }
            )

    table3 = pd.DataFrame(rows)

    table3 = table3.sort_values(
        ["function_type", "function_id", "wasserstein_distance", "hyperparameter"],
        ascending=[True, True, False, True],
    )

    return table3


# =========================
# Table 4
# Generated SHAP value-distance table
# =========================
def build_table4_generated_value_distance(hof):
    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    for _, r in generated.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        try:
            group_dist, group_path, group_mode = load_best_available_group_distribution(group_id)
            individual_dist, individual_path, individual_mode = load_best_available_individual_distribution(
                function_type,
                function_id,
            )

            # Also load normalized importance for importance-distance columns.
            group_imp, group_imp_path, group_imp_col = load_group_importance_vector(group_id)
            ind_imp, ind_imp_path, ind_imp_col = load_individual_importance_vector(
                function_type,
                function_id,
            )

            features = sorted(
                set(group_dist.keys())
                | set(individual_dist.keys())
                | set(group_imp.keys())
                | set(ind_imp.keys())
            )

            for feature in features:
                gv = np.asarray(group_dist.get(feature, np.array([0.0])), dtype=float)
                iv = np.asarray(individual_dist.get(feature, np.array([0.0])), dtype=float)

                group_mean = float(np.mean(gv)) if len(gv) > 0 else np.nan
                function_mean = float(np.mean(iv)) if len(iv) > 0 else np.nan

                group_mean_abs = float(np.mean(np.abs(gv))) if len(gv) > 0 else np.nan
                function_mean_abs = float(np.mean(np.abs(iv))) if len(iv) > 0 else np.nan

                signed_diff = function_mean - group_mean
                abs_diff = abs(signed_diff)

                group_imp_norm = float(group_imp.get(feature, 0.0))
                function_imp_norm = float(ind_imp.get(feature, 0.0))
                imp_signed_diff = function_imp_norm - group_imp_norm
                imp_abs_diff = abs(imp_signed_diff)

                rows.append(
                    {
                        "function_type": function_type,
                        "function_id": function_id,
                        "function_label": short_function_label(r),
                        "assigned_group_0307": canonical_group_label(group_id),
                        "hyperparameter": feature,
                        "group_mean_shap": group_mean,
                        "function_mean_shap": function_mean,
                        "signed_difference_function_minus_group": signed_diff,
                        "absolute_difference": abs_diff,
                        "group_mean_abs_shap": group_mean_abs,
                        "function_mean_abs_shap": function_mean_abs,
                        "signed_difference_abs_shap_function_minus_group": function_mean_abs - group_mean_abs,
                        "absolute_difference_mean_abs_shap": abs(function_mean_abs - group_mean_abs),
                        "group_importance_norm": group_imp_norm,
                        "function_importance_norm": function_imp_norm,
                        "importance_signed_difference_function_minus_group": imp_signed_diff,
                        "importance_absolute_difference": imp_abs_diff,
                        "source_group_shap": str(Path(group_path).relative_to(PROJECT_ROOT)),
                        "source_function_shap": str(Path(individual_path).relative_to(PROJECT_ROOT)),
                        "group_distribution_mode": group_mode,
                        "function_distribution_mode": individual_mode,
                        "group_importance_column": group_imp_col,
                        "function_importance_column": ind_imp_col,
                        "status": "ok",
                        "note": (
                            "Mean SHAP columns use raw distributions when available. "
                            "Importance columns use normalized summary SHAP importance."
                        ),
                    }
                )

        except Exception as e:
            rows.append(
                {
                    "function_type": function_type,
                    "function_id": function_id,
                    "function_label": short_function_label(r),
                    "assigned_group_0307": canonical_group_label(group_id),
                    "hyperparameter": "",
                    "group_mean_shap": np.nan,
                    "function_mean_shap": np.nan,
                    "signed_difference_function_minus_group": np.nan,
                    "absolute_difference": np.nan,
                    "group_mean_abs_shap": np.nan,
                    "function_mean_abs_shap": np.nan,
                    "signed_difference_abs_shap_function_minus_group": np.nan,
                    "absolute_difference_mean_abs_shap": np.nan,
                    "group_importance_norm": np.nan,
                    "function_importance_norm": np.nan,
                    "importance_signed_difference_function_minus_group": np.nan,
                    "importance_absolute_difference": np.nan,
                    "source_group_shap": "",
                    "source_function_shap": "",
                    "group_distribution_mode": "",
                    "function_distribution_mode": "",
                    "group_importance_column": "",
                    "function_importance_column": "",
                    "status": "error",
                    "note": str(e),
                }
            )

    table4 = pd.DataFrame(rows)

    table4 = table4.sort_values(
        ["function_type", "function_id", "importance_absolute_difference", "hyperparameter"],
        ascending=[True, True, False, True],
    )

    return table4


# =========================
# Table 5
# Generated SHAP similarity summary table
# =========================
def build_table5_generated_summary(hof, table3, table4):
    generated = hof[
        hof["function_type"].isin(GENERATED_TYPES)
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    rows = []

    # Existing Step 10 similarity file is optional.
    similarity = None
    if SHAP_SIMILARITY_CSV.exists():
        try:
            similarity = pd.read_csv(SHAP_SIMILARITY_CSV)
        except Exception:
            similarity = None

    for _, r in generated.iterrows():
        function_type = str(r["function_type"])
        function_id = str(r["function_id"])
        group_id = r["assigned_group_0307"]

        item = {
            "function_type": function_type,
            "function_id": function_id,
            "function_label": short_function_label(r),
            "assigned_group_0307": canonical_group_label(group_id),
            "actual_auc_mean": safe_float(r.get("actual_auc_mean", np.nan)),
        }

        # Compute cosine, Spearman, top-3 directly from normalized importance.
        try:
            group_vec, _, _ = load_group_importance_vector(group_id)
            individual_vec, _, _ = load_individual_importance_vector(function_type, function_id)

            features, g, i = align_importance_vectors(group_vec, individual_vec)

            item["cosine_similarity"] = cosine_similarity(g, i)
            item["spearman_similarity"] = spearman_corr(g, i)
            item["top3_overlap"] = top_k_overlap(g, i, k=3)

        except Exception as e:
            item["cosine_similarity"] = np.nan
            item["spearman_similarity"] = np.nan
            item["top3_overlap"] = np.nan
            item["note_similarity"] = str(e)

        # If Step 10 similarity has values, keep them as reference columns.
        if similarity is not None:
            sim_sub = similarity[
                similarity["function_type"].astype(str).eq(function_type)
                & similarity["function_id"].astype(str).eq(function_id)
            ]

            if not sim_sub.empty:
                sim_row = sim_sub.iloc[0]
                for old_col, new_col in [
                    ("cosine", "step10_cosine"),
                    ("spearman", "step10_spearman"),
                    ("top3_overlap", "step10_top3_overlap"),
                    ("wasserstein_distribution", "step10_wasserstein_distribution"),
                    ("mean_wasserstein", "step10_mean_wasserstein"),
                    ("most_different_feature", "step10_most_different_feature"),
                ]:
                    if old_col in sim_row.index:
                        item[new_col] = sim_row[old_col]

        # Table 4 value distance summary.
        t4 = table4[
            table4["function_type"].astype(str).eq(function_type)
            & table4["function_id"].astype(str).eq(function_id)
            & table4["status"].eq("ok")
        ].copy()

        if not t4.empty:
            item["mean_value_distance"] = float(t4["importance_absolute_difference"].mean())
            item["max_value_distance"] = float(t4["importance_absolute_difference"].max())

            idx = t4["importance_absolute_difference"].idxmax()
            item["most_different_hyperparameter_value"] = t4.loc[idx, "hyperparameter"]
        else:
            item["mean_value_distance"] = np.nan
            item["max_value_distance"] = np.nan
            item["most_different_hyperparameter_value"] = ""

        # Table 3 shape distance summary.
        t3 = table3[
            table3["function_type"].astype(str).eq(function_type)
            & table3["function_id"].astype(str).eq(function_id)
            & table3["status"].eq("ok")
        ].copy()

        if not t3.empty:
            item["mean_shape_distance_wasserstein"] = float(t3["wasserstein_distance"].mean())
            item["max_shape_distance_wasserstein"] = float(t3["wasserstein_distance"].max())
            item["mean_shape_distance_ks"] = float(t3["ks_statistic"].mean())
            item["max_shape_distance_ks"] = float(t3["ks_statistic"].max())

            idx = t3["wasserstein_distance"].idxmax()
            item["most_different_hyperparameter_shape"] = t3.loc[idx, "hyperparameter"]

            item["distribution_mode"] = (
                f"group={t3['group_distribution_mode'].iloc[0]}; "
                f"function={t3['function_distribution_mode'].iloc[0]}"
            )
        else:
            item["mean_shape_distance_wasserstein"] = np.nan
            item["max_shape_distance_wasserstein"] = np.nan
            item["mean_shape_distance_ks"] = np.nan
            item["max_shape_distance_ks"] = np.nan
            item["most_different_hyperparameter_shape"] = ""
            item["distribution_mode"] = ""

        item["status"] = "ok"
        item["note"] = (
            "Summary combines importance-vector similarity, Table 3 shape distance, "
            "and Table 4 value distance."
        )

        rows.append(item)

    table5 = pd.DataFrame(rows)

    table5 = table5.sort_values(["function_type", "function_id"])

    return table5


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("STEP 10D: GENERATE FINAL SUPERVISOR TABLES UNDER 0307 MAIN SCHEME")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Comparison dir: {COMPARISON_DIR}")
    print(f"Hall of Fame CSV: {HALL_OF_FAME_CSV}")
    print(f"SHAP similarity CSV: {SHAP_SIMILARITY_CSV}")
    print(f"Group SHAP dir: {GROUP_SHAP_DIR}")
    print(f"BBOB individual SHAP dir: {BBOB_INDIVIDUAL_SHAP_DIR}")
    print(f"Affine SHAP dir: {AFFINE_SHAP_DIR}")
    print(f"LLaMEA SHAP dir: {LLAMEA_SHAP_DIR}")

    if not HALL_OF_FAME_CSV.exists():
        raise FileNotFoundError(f"Missing Hall-of-Fame CSV: {HALL_OF_FAME_CSV}")

    hof = pd.read_csv(HALL_OF_FAME_CSV)

    required_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
    ]

    missing = [c for c in required_cols if c not in hof.columns]
    if missing:
        raise ValueError(f"Hall-of-Fame file is missing required columns: {missing}")

    # --------------------------------------------------------
    # Table 1
    # --------------------------------------------------------
    print("\nBuilding Table 1: BBOB group-level SHAP distance table...")

    table1, table1_detail = build_table1_bbob_group_shap_distance(hof)

    table1.to_csv(OUT_TABLE1_CSV, index=False)

    with open(OUT_TABLE1_MD, "w", encoding="utf-8") as f:
        f.write("# Table 1. BBOB group-level SHAP distance by hyperparameter\n\n")
        f.write(
            "For each BBOB landscape group and each hyperparameter, this table reports "
            "the average absolute distance between the group-level normalized SHAP importance "
            "and the individual BBOB function-level normalized SHAP importance.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table1))
        f.write("\n")

    # --------------------------------------------------------
    # Table 2
    # --------------------------------------------------------
    print("Building Table 2: generated predicted vs actual Hall-of-Fame table...")

    table2 = build_table2_generated_predicted_vs_actual_hof(hof)

    table2.to_csv(OUT_TABLE2_CSV, index=False)

    table2_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "config_source",
        "auc_mean",
        "delta_auc_actual_minus_predicted",
        "CR",
        "F",
        "crossover",
        "lambda_",
        "lpsr",
        "mutation_base",
        "mutation_n_comps",
        "mutation_reference",
        "use_archive",
        "config_match_ratio",
        "mismatch_config_fields",
        "predicted_source_bbob_function",
        "status",
    ]

    with open(OUT_TABLE2_MD, "w", encoding="utf-8") as f:
        f.write("# Table 2. Generated functions: predicted group Hall-of-Fame vs actual function Hall-of-Fame\n\n")
        f.write(
            "Each generated function is shown with two rows: the predicted configuration from "
            "the assigned group's BBOB Hall-of-Fame winner, and the actual function-level "
            "Hall-of-Fame winner.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table2[[c for c in table2_display_cols if c in table2.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 3
    # --------------------------------------------------------
    print("Building Table 3: generated SHAP shape-distance table...")

    table3 = build_table3_generated_shape_distance(hof)

    table3.to_csv(OUT_TABLE3_CSV, index=False)

    table3_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "hyperparameter",
        "shape_distance_metric",
        "wasserstein_distance",
        "ks_statistic",
        "group_n_values",
        "function_n_values",
        "group_std_shap",
        "function_std_shap",
        "group_distribution_mode",
        "function_distribution_mode",
        "status",
    ]

    with open(OUT_TABLE3_MD, "w", encoding="utf-8") as f:
        f.write("# Table 3. Generated functions: SHAP distribution shape distance\n\n")
        f.write(
            "For each generated function and hyperparameter, this table compares the shape "
            "of the assigned group SHAP distribution with the individual function SHAP distribution. "
            "Raw SHAP values are used when available; otherwise the script falls back to "
            "summary-level importance values.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table3[[c for c in table3_display_cols if c in table3.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 4
    # --------------------------------------------------------
    print("Building Table 4: generated SHAP value-distance table...")

    table4 = build_table4_generated_value_distance(hof)

    table4.to_csv(OUT_TABLE4_CSV, index=False)

    table4_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "hyperparameter",
        "group_mean_shap",
        "function_mean_shap",
        "signed_difference_function_minus_group",
        "absolute_difference",
        "group_mean_abs_shap",
        "function_mean_abs_shap",
        "group_importance_norm",
        "function_importance_norm",
        "importance_signed_difference_function_minus_group",
        "importance_absolute_difference",
        "group_distribution_mode",
        "function_distribution_mode",
        "status",
    ]

    with open(OUT_TABLE4_MD, "w", encoding="utf-8") as f:
        f.write("# Table 4. Generated functions: SHAP value distance\n\n")
        f.write(
            "For each generated function and hyperparameter, this table compares the value "
            "and magnitude of SHAP contributions between the assigned group and the individual function.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table4[[c for c in table4_display_cols if c in table4.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 5
    # --------------------------------------------------------
    print("Building Table 5: generated SHAP similarity summary table...")

    table5 = build_table5_generated_summary(hof, table3, table4)

    table5.to_csv(OUT_TABLE5_CSV, index=False)

    table5_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "actual_auc_mean",
        "cosine_similarity",
        "spearman_similarity",
        "top3_overlap",
        "mean_value_distance",
        "mean_shape_distance_wasserstein",
        "mean_shape_distance_ks",
        "most_different_hyperparameter_value",
        "most_different_hyperparameter_shape",
        "distribution_mode",
        "status",
    ]

    with open(OUT_TABLE5_MD, "w", encoding="utf-8") as f:
        f.write("# Table 5. Generated functions: SHAP similarity summary\n\n")
        f.write(
            "This table summarises, for each generated function, how similar its SHAP profile "
            "is to the assigned group reference. Value distance is derived from Table 4 and "
            "shape distance is derived from Table 3.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table5[[c for c in table5_display_cols if c in table5.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("Step 10D final supervisor-requested tables under 0307 main tiebreak scheme\n")
        f.write("\n")
        f.write("Generated files:\n")
        f.write(f"- Table 1 CSV: {OUT_TABLE1_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 1 MD:  {OUT_TABLE1_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 2 CSV: {OUT_TABLE2_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 2 MD:  {OUT_TABLE2_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 3 CSV: {OUT_TABLE3_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 3 MD:  {OUT_TABLE3_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 4 CSV: {OUT_TABLE4_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 4 MD:  {OUT_TABLE4_MD.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 5 CSV: {OUT_TABLE5_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Table 5 MD:  {OUT_TABLE5_MD.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("Inputs:\n")
        f.write(f"- Hall of Fame: {HALL_OF_FAME_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- SHAP similarity: {SHAP_SIMILARITY_CSV.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Group SHAP dir: {GROUP_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- BBOB individual SHAP dir: {BBOB_INDIVIDUAL_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- Affine SHAP dir: {AFFINE_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write(f"- LLaMEA SHAP dir: {LLAMEA_SHAP_DIR.relative_to(PROJECT_ROOT)}\n")
        f.write("\n")
        f.write("Definitions:\n")
        f.write("- Table 1 distance = abs(group normalized SHAP importance - BBOB function normalized SHAP importance).\n")
        f.write("- Table 2 predicted configuration = the BBOB Hall-of-Fame winner inside the assigned group.\n")
        f.write("- Table 2 actual configuration = the generated function's own Hall-of-Fame winner.\n")
        f.write("- Table 3 shape distance = Wasserstein distance and KS statistic between group and function SHAP distributions.\n")
        f.write("- Table 4 value distance = differences in mean SHAP, mean absolute SHAP, and normalized SHAP importance.\n")
        f.write("- Table 5 summary combines cosine, Spearman, top-3 overlap, mean value distance, and mean shape distance.\n")
        f.write("\n")
        f.write("Important limitation:\n")
        f.write("- If raw SHAP-value files are unavailable, Tables 3 and 4 fall back to summary-level SHAP importance values.\n")
        f.write("- In fallback mode, shape-distance results should be interpreted as summary-level approximations, not full raw SHAP distribution comparisons.\n")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDone.")
    print(f"Saved Table 1 CSV: {OUT_TABLE1_CSV}")
    print(f"Saved Table 2 CSV: {OUT_TABLE2_CSV}")
    print(f"Saved Table 3 CSV: {OUT_TABLE3_CSV}")
    print(f"Saved Table 4 CSV: {OUT_TABLE4_CSV}")
    print(f"Saved Table 5 CSV: {OUT_TABLE5_CSV}")
    print(f"Saved README:     {README_FILE}")

    print("\nTable 1 preview:")
    print(table1.head(20).to_string(index=False))

    print("\nTable 2 preview:")
    print(
        table2[
            [
                "function_type",
                "function_id",
                "assigned_group_0307",
                "config_source",
                "auc_mean",
                "delta_auc_actual_minus_predicted",
                "config_match_ratio",
                "mismatch_config_fields",
                "status",
            ]
        ].to_string(index=False)
    )

    print("\nTable 3 status counts:")
    print(table3["status"].value_counts(dropna=False).to_string())

    print("\nTable 3 distribution modes:")
    if "group_distribution_mode" in table3.columns and "function_distribution_mode" in table3.columns:
        print(
            table3[["group_distribution_mode", "function_distribution_mode"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    print("\nTable 4 status counts:")
    print(table4["status"].value_counts(dropna=False).to_string())

    print("\nTable 5 preview:")
    print(
        table5[
            [
                "function_type",
                "function_id",
                "assigned_group_0307",
                "actual_auc_mean",
                "cosine_similarity",
                "spearman_similarity",
                "top3_overlap",
                "mean_value_distance",
                "mean_shape_distance_wasserstein",
                "most_different_hyperparameter_value",
                "most_different_hyperparameter_shape",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()