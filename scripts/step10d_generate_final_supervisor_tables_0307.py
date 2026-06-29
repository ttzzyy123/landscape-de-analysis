from pathlib import Path
import os
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
#   assigned BBOB group mean reference as predicted config
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

INPUT_COMPARISON_DIR = PROJECT_ROOT / "output" / "0307_main_comparison"

COMPARISON_DIR = Path(
    os.getenv(
        "STEP10D_OUTPUT_DIR",
        str(INPUT_COMPARISON_DIR),
    )
)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

HALL_OF_FAME_CSV = Path(
    os.getenv(
        "STEP10D_HALL_OF_FAME_CSV",
        str(INPUT_COMPARISON_DIR / "hall_of_fame_0307.csv"),
    )
)

SHAP_SIMILARITY_CSV = Path(
    os.getenv(
        "STEP10D_SHAP_SIMILARITY_CSV",
        str(INPUT_COMPARISON_DIR / "shap_similarity_0307.csv"),
    )
)

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

def default_raw_shap_base():
    env_value = os.getenv("RAW_SHAP_LOCAL_BASE")
    if env_value:
        return Path(env_value)

    candidates = [
        Path("/local/s3795888/my_landscape_experiments_raw_shap"),
    ]

    # Local Windows backup layout:
    #   server_full_backup_20260624/
    #     landscape-de-analysis-main/landscape-de-analysis-main/
    #     my_landscape_experiments_raw_shap/
    for parent in PROJECT_ROOT.parents:
        candidates.append(parent / "my_landscape_experiments_raw_shap")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


LOCAL_RAW_BASE = default_raw_shap_base()

PROJECT_NEW_FUNCTION_SHAP_BASE = PROJECT_ROOT / "output" / "new_function_task"
AFFINE_SHAP_DIR = PROJECT_NEW_FUNCTION_SHAP_BASE / "task_H_individual_shap_affine_functions"
LLAMEA_SHAP_DIR = PROJECT_NEW_FUNCTION_SHAP_BASE / "task_H_individual_shap_real_generated_function"
AFFINE_SHAP_FALLBACK_DIR = LOCAL_RAW_BASE / "task_H_individual_shap_affine_functions"
LLAMEA_SHAP_FALLBACK_DIR = LOCAL_RAW_BASE / "task_H_individual_shap_real_generated_function"

HYPERPARAMETERS = [
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

# =========================
# Output files
# =========================
OUT_TABLE1_CSV = COMPARISON_DIR / "table01_bbob_group_hyperparameter_shap_distance_0307.csv"
OUT_TABLE1_MD = COMPARISON_DIR / "table01_bbob_group_hyperparameter_shap_distance_0307.md"

# Additional Table 1 outputs requested for Overleaf:
# one row per BBOB function, with all hyperparameters shown as
# group importance / function importance (group - function).
OUT_TABLE1_DETAIL_CSV = COMPARISON_DIR / "table01_bbob_function_hyperparameter_shap_distance_detail_0307.csv"
OUT_TABLE1_WIDE_CSV = COMPARISON_DIR / "table01_bbob_function_hyperparameter_shap_distance_wide_0307.csv"
OUT_TABLE1_WIDE_MD = COMPARISON_DIR / "table01_bbob_function_hyperparameter_shap_distance_wide_0307.md"
OUT_TABLE1_WIDE_TEX = COMPARISON_DIR / "table01_bbob_function_hyperparameter_shap_distance_wide_0307.tex"

OUT_TABLE2_CSV = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_0307.csv"
OUT_TABLE2_WIDE_CSV = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_wide_0307.csv"
OUT_TABLE2_WIDE_MD = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_wide_0307.md"
OUT_TABLE2_WIDE_TEX = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_wide_0307.tex"
OUT_TABLE2_MD = COMPARISON_DIR / "table02_generated_predicted_vs_actual_hof_0307.md"

OUT_TABLE3_CSV = COMPARISON_DIR / "table03_generated_shap_shape_distance_0307.csv"
OUT_TABLE3_WIDE_CSV = COMPARISON_DIR / "table03_generated_shap_shape_distance_wide_0307.csv"
OUT_TABLE3_WIDE_MD = COMPARISON_DIR / "table03_generated_shap_shape_distance_wide_0307.md"
OUT_TABLE3_WIDE_TEX = COMPARISON_DIR / "table03_generated_shap_shape_distance_wide_0307.tex"
OUT_TABLE3_WASSERSTEIN_WIDE_CSV = COMPARISON_DIR / "table03_generated_shap_shape_wasserstein_wide_0307.csv"
OUT_TABLE3_WASSERSTEIN_WIDE_MD = COMPARISON_DIR / "table03_generated_shap_shape_wasserstein_wide_0307.md"
OUT_TABLE3_WASSERSTEIN_WIDE_TEX = COMPARISON_DIR / "table03_generated_shap_shape_wasserstein_wide_0307.tex"
OUT_TABLE3_KL_WIDE_CSV = COMPARISON_DIR / "table03_generated_shap_shape_kl_wide_0307.csv"
OUT_TABLE3_KL_WIDE_MD = COMPARISON_DIR / "table03_generated_shap_shape_kl_wide_0307.md"
OUT_TABLE3_KL_WIDE_TEX = COMPARISON_DIR / "table03_generated_shap_shape_kl_wide_0307.tex"
OUT_TABLE3_MD = COMPARISON_DIR / "table03_generated_shap_shape_distance_0307.md"

OUT_TABLE4_CSV = COMPARISON_DIR / "table04_generated_shap_value_distance_0307.csv"
OUT_TABLE4_WIDE_CSV = COMPARISON_DIR / "table04_generated_shap_value_distance_wide_0307.csv"
OUT_TABLE4_WIDE_MD = COMPARISON_DIR / "table04_generated_shap_value_distance_wide_0307.md"
OUT_TABLE4_WIDE_TEX = COMPARISON_DIR / "table04_generated_shap_value_distance_wide_0307.tex"
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


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
    
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
        base_dirs = [AFFINE_SHAP_DIR, AFFINE_SHAP_FALLBACK_DIR]
        patterns = [
            f"*{function_id}*raw*.csv",
            f"*{function_id}*shap_values*.csv",
            f"*{function_id}*values*.csv",
            f"*{function_id}*long*.csv",
        ]

    elif function_type == "llamea_generated":
        base_dirs = [LLAMEA_SHAP_DIR, LLAMEA_SHAP_FALLBACK_DIR]
        patterns = [
            f"*{function_id}*raw*.csv",
            f"*{function_id}*shap_values*.csv",
            f"*{function_id}*values*.csv",
            f"*{function_id}*long*.csv",
        ]

    else:
        return []

    if function_type == "bbob_original":
        base_dirs = [base_dir]

    files = []
    for base in base_dirs:
        if not base.exists():
            continue
        for p in patterns:
            files.extend(sorted(base.glob(p)))

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


def read_raw_shap_records_from_csv(path):
    """
    Reads raw SHAP records with feature values for strict A/B matching.

    Required long-format columns:
        feature, feature_value, shap_value

    Returns:
        dict: feature -> DataFrame[feature_value, shap_value]
    """
    path = Path(path)
    df = pd.read_csv(path)
    lower_cols = {str(c).lower(): c for c in df.columns}

    feature_col = None
    value_col = None
    shap_col = None

    for c in ["feature", "variable", "hyperparameter", "parameter"]:
        if c in lower_cols:
            feature_col = lower_cols[c]
            break

    for c in ["feature_value", "x_value", "parameter_value", "value_x"]:
        if c in lower_cols:
            value_col = lower_cols[c]
            break

    for c in ["shap_value", "shap", "shap_values"]:
        if c in lower_cols:
            shap_col = lower_cols[c]
            break

    if feature_col is None or value_col is None or shap_col is None:
        raise ValueError(
            f"File does not contain strict raw-long SHAP records: {path}. "
            f"Need feature, feature_value, shap_value. Columns: {df.columns.tolist()}"
        )

    out = {}
    for feature, sub in df.groupby(feature_col):
        tmp = pd.DataFrame(
            {
                "feature_value": pd.to_numeric(sub[value_col], errors="coerce"),
                "shap_value": pd.to_numeric(sub[shap_col], errors="coerce"),
            }
        ).dropna()
        if not tmp.empty:
            out[feature_name_cleanup(feature)] = tmp.reset_index(drop=True)

    if not out:
        raise ValueError(f"No usable raw-long SHAP records in {path}")

    return out


def load_raw_group_records(group_id):
    files = candidate_raw_shap_files_for_group(group_id)

    last_error = None
    for f in files:
        try:
            records = read_raw_shap_records_from_csv(f)
            if records:
                return records, f, "raw_long"
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"No usable raw-long group SHAP records found for group {group_id}. "
        f"Candidates: {[str(x) for x in files]}. Last error: {last_error}"
    )


def load_raw_individual_records(function_type, function_id):
    files = candidate_raw_shap_files_for_individual(function_type, function_id)

    last_error = None
    for f in files:
        try:
            records = read_raw_shap_records_from_csv(f)
            if records:
                return records, f, "raw_long"
        except Exception as e:
            last_error = e

    raise FileNotFoundError(
        f"No usable raw-long individual SHAP records found for "
        f"{function_type} {function_id}. "
        f"Candidates: {[str(x) for x in files]}. Last error: {last_error}"
    )


def nearest_feature_value_mean_abs_distance(group_records, function_records):
    """
    Strict A/B point matching distance.

    For each generated-function point A=(feature_value, shap_value), find the
    group point B with the nearest feature_value and average |A_shap - B_shap|.
    Exact feature-value matches are therefore used whenever they exist.
    """
    if group_records is None or function_records is None:
        return np.nan, 0, np.nan

    g = group_records[["feature_value", "shap_value"]].dropna().copy()
    f = function_records[["feature_value", "shap_value"]].dropna().copy()

    if g.empty or f.empty:
        return np.nan, 0, np.nan

    g = g.sort_values("feature_value").reset_index(drop=True)
    gx = g["feature_value"].to_numpy(dtype=float)
    gy = g["shap_value"].to_numpy(dtype=float)
    fx = f["feature_value"].to_numpy(dtype=float)
    fy = f["shap_value"].to_numpy(dtype=float)

    pos = np.searchsorted(gx, fx, side="left")
    left = np.clip(pos - 1, 0, len(gx) - 1)
    right = np.clip(pos, 0, len(gx) - 1)

    left_gap = np.abs(fx - gx[left])
    right_gap = np.abs(fx - gx[right])
    use_right = right_gap < left_gap
    nearest = np.where(use_right, right, left)

    shap_gap = np.abs(fy - gy[nearest])
    feature_value_gap = np.abs(fx - gx[nearest])

    return (
        float(np.mean(shap_gap)),
        int(len(shap_gap)),
        float(np.mean(feature_value_gap)),
    )


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


def kl_divergence_1d(x, y, bins=40, epsilon=1e-12):
    """
    Histogram-based Kullback-Leibler divergence KL(group || function).

    The two empirical distributions are binned on a shared range and smoothed
    with epsilon so that zero-probability bins do not make the divergence
    infinite. Smaller values mean more similar distribution shapes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    all_values = np.concatenate([x, y])
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan

    if abs(hi - lo) <= 1e-15:
        return 0.0

    hist_x, bin_edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    hist_y, _ = np.histogram(y, bins=bin_edges, density=False)

    p = hist_x.astype(float) + epsilon
    q = hist_y.astype(float) + epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)

    return float(np.sum(p * np.log(p / q)))


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
    Predicted group Hall-of-Fame reference.

    Rule:
      - Prefer Step 9's bbob_group_mean rows when present.
      - These rows average numeric BBOB HoF values inside each assigned group.
      - Categorical settings are represented by the group mode.
      - If older Hall-of-Fame files do not contain group-mean rows, compute
        the same reference directly from BBOB original HoF rows.
    """
    group_mean = hof[
        (hof["function_type"].eq("bbob_group_mean"))
        & (~hof["function_id"].astype(str).eq("All"))
    ].copy()

    if not group_mean.empty:
        rows = []
        group_mean["actual_auc_mean"] = pd.to_numeric(
            group_mean["actual_auc_mean"],
            errors="coerce",
        )

        for _, r in group_mean.iterrows():
            group_id = r["assigned_group_0307"]
            item = {
                "assigned_group_0307": group_id,
                "predicted_source_function_type": r["function_type"],
                "predicted_source_function_id": r.get("function_id", group_id),
                "predicted_source_auc_mean": safe_float(r["actual_auc_mean"]),
                "group_n_bbob_functions": r.get("group_n_bbob_functions", np.nan),
                "group_bbob_functions": r.get("group_bbob_functions", ""),
                "predicted_config_signature": config_signature(r),
                "prediction_reference_method": "bbob_group_mean_row",
            }

            for c in CONFIG_COLS:
                item[c] = r.get(c, np.nan)

            rows.append(item)

        return pd.DataFrame(rows)

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

        auc = sub["actual_auc_mean"].dropna()
        if auc.empty:
            continue

        item = {
            "assigned_group_0307": group_id,
            "predicted_source_function_type": "bbob_group_mean",
            "predicted_source_function_id": str(group_id),
            "predicted_source_auc_mean": float(auc.mean()),
            "group_n_bbob_functions": len(sub),
            "group_bbob_functions": ",".join(sub["function_id"].astype(str).tolist()),
            "prediction_reference_method": "computed_from_bbob_hof_rows",
        }

        for c in ["CR", "F", "lambda_", "mutation_n_comps", "lpsr", "use_archive"]:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                item[c] = float(vals.mean()) if not vals.empty else np.nan

        for c in ["crossover", "mutation_base", "mutation_reference"]:
            if c in sub.columns:
                vals = sub[c].dropna().astype(str)
                vals = vals[~vals.str.lower().isin(["", "nan", "none"])]
                item[c] = vals.mode().iloc[0] if not vals.empty else np.nan

        item["predicted_config_signature"] = config_signature(pd.Series(item))

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
                        "source_group_shap": display_path(group_path),
                        "source_function_shap": display_path(individual_path),
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
                    "config_source": "group mean predicted",
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
            "prediction_reference_method": pred.get("prediction_reference_method", ""),
            "delta_auc_actual_minus_predicted": delta_auc,
            "config_match_count": match_count,
            "config_available_count": available_count,
            "config_match_ratio": match_ratio,
            "mismatch_config_fields": ",".join(mismatch_cols),
            "status": "ok",
        }

        pred_row = {
            **common,
            "config_source": "group mean predicted",
            "auc_mean": predicted_auc,
            "config_signature": pred.get("predicted_config_signature", ""),
            "note": (
                "Predicted reference is the assigned group's mean BBOB "
                "Hall-of-Fame behavior; numeric hyperparameters are means "
                "and categorical hyperparameters are modes."
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
                        "kl_divergence": kl_divergence_1d(gv, iv),
                        "ks_statistic": ks_statistic_1d(gv, iv),
                        "group_n_values": len(gv),
                        "function_n_values": len(iv),
                        "group_std_shap": float(np.std(gv)) if len(gv) > 0 else np.nan,
                        "function_std_shap": float(np.std(iv)) if len(iv) > 0 else np.nan,
                        "group_skewness": skewness(gv),
                        "function_skewness": skewness(iv),
                        "group_kurtosis_excess": kurtosis_excess(gv),
                        "function_kurtosis_excess": kurtosis_excess(iv),
                        "source_group_shap": display_path(group_path),
                        "source_function_shap": display_path(individual_path),
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
                    "kl_divergence": np.nan,
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
            try:
                group_records, group_records_path, group_records_mode = load_raw_group_records(group_id)
            except Exception:
                group_records, group_records_path, group_records_mode = {}, "", ""

            try:
                individual_records, individual_records_path, individual_records_mode = load_raw_individual_records(
                    function_type,
                    function_id,
                )
            except Exception:
                individual_records, individual_records_path, individual_records_mode = {}, "", ""

            # Also load normalized importance for importance-distance columns.
            group_imp, group_imp_path, group_imp_col = load_group_importance_vector(group_id)
            ind_imp, ind_imp_path, ind_imp_col = load_individual_importance_vector(
                function_type,
                function_id,
            )

            feature_candidates = set(individual_dist.keys()) | set(ind_imp.keys())
            preferred_features = [hp for hp in HYPERPARAMETERS if hp in feature_candidates]
            remaining_features = sorted(f for f in feature_candidates if f not in preferred_features)
            features = preferred_features + remaining_features

            for feature in features:
                gv = np.asarray(group_dist.get(feature, np.array([0.0])), dtype=float)
                iv = np.asarray(individual_dist.get(feature, np.array([0.0])), dtype=float)

                group_mean = float(np.mean(gv)) if len(gv) > 0 else np.nan
                function_mean = float(np.mean(iv)) if len(iv) > 0 else np.nan

                group_mean_abs = float(np.mean(np.abs(gv))) if len(gv) > 0 else np.nan
                function_mean_abs = float(np.mean(np.abs(iv))) if len(iv) > 0 else np.nan
                value_wasserstein = wasserstein_1d(gv, iv)
                matched_distance, matched_n_pairs, matched_mean_feature_gap = (
                    nearest_feature_value_mean_abs_distance(
                        group_records.get(feature),
                        individual_records.get(feature),
                    )
                )

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
                        "wasserstein_distance": value_wasserstein,
                        "matched_feature_value_mean_abs_distance": matched_distance,
                        "matched_n_pairs": matched_n_pairs,
                        "matched_mean_feature_value_gap": matched_mean_feature_gap,
                        "group_importance_norm": group_imp_norm,
                        "function_importance_norm": function_imp_norm,
                        "importance_signed_difference_function_minus_group": imp_signed_diff,
                        "importance_absolute_difference": imp_abs_diff,
                        "source_group_shap": display_path(group_path),
                        "source_function_shap": display_path(individual_path),
                        "source_group_raw_long": display_path(group_records_path) if group_records_path else "",
                        "source_function_raw_long": display_path(individual_records_path) if individual_records_path else "",
                        "group_distribution_mode": group_mode,
                        "function_distribution_mode": individual_mode,
                        "group_record_mode": group_records_mode,
                        "function_record_mode": individual_records_mode,
                        "group_importance_column": group_imp_col,
                        "function_importance_column": ind_imp_col,
                        "status": "ok",
                        "note": (
                            "matched_feature_value_mean_abs_distance averages |function SHAP - nearest group SHAP| "
                            "after matching by the same or nearest feature_value. Wasserstein is retained "
                            "as an unpaired SHAP-distribution distance."
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
                    "wasserstein_distance": np.nan,
                    "matched_feature_value_mean_abs_distance": np.nan,
                    "matched_n_pairs": np.nan,
                    "matched_mean_feature_value_gap": np.nan,
                    "group_importance_norm": np.nan,
                    "function_importance_norm": np.nan,
                    "importance_signed_difference_function_minus_group": np.nan,
                    "importance_absolute_difference": np.nan,
                    "source_group_shap": "",
                    "source_function_shap": "",
                    "source_group_raw_long": "",
                    "source_function_raw_long": "",
                    "group_distribution_mode": "",
                    "function_distribution_mode": "",
                    "group_record_mode": "",
                    "function_record_mode": "",
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
            if "matched_feature_value_mean_abs_distance" in t4.columns:
                value_distance_col = "matched_feature_value_mean_abs_distance"
            else:
                value_distance_col = "importance_absolute_difference"

            item["mean_value_distance"] = float(t4[value_distance_col].mean())
            item["max_value_distance"] = float(t4[value_distance_col].max())

            idx = t4[value_distance_col].idxmax()
            item["most_different_hyperparameter_value"] = t4.loc[idx, "hyperparameter"]
            item["value_distance_metric"] = value_distance_col
        else:
            item["mean_value_distance"] = np.nan
            item["max_value_distance"] = np.nan
            item["most_different_hyperparameter_value"] = ""
            item["value_distance_metric"] = ""

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
# Additional Table 1 wide-format helpers
# =========================
def format_group_function_delta(row):
    """
    Format one cell as:
    group importance / individual function importance (group - function)
    """
    g = row["group_importance_norm"]
    f = row["function_importance_norm"]

    if "signed_difference_group_minus_function" in row.index:
        d = row["signed_difference_group_minus_function"]
    elif "signed_difference" in row.index:
        d = row["signed_difference"]
    else:
        d = g - f

    if pd.isna(g) or pd.isna(f) or pd.isna(d):
        return ""

    return f"{g:.3f}/{f:.3f} ({d:+.3f})"


def natural_function_sort_key(x):
    s = str(x)

    if s.startswith("f"):
        s2 = s[1:]
        if s2.isdigit():
            return int(s2)

    if s.isdigit():
        return int(s)

    return 10**9


def build_table1_bbob_function_wide(table1_detail: pd.DataFrame) -> pd.DataFrame:
    """Build wide per-BBOB-function Table 1.

    Each hyperparameter cell is:
        group importance / function importance (group - function)

    This version is intentionally tolerant to column-name differences between
    earlier Step10D drafts.
    """
    import re

    detail = table1_detail.copy()
    if detail.empty:
        return pd.DataFrame()

    def pick_column(candidates, label, required=True):
        for col in candidates:
            if col in detail.columns:
                return col
        if required:
            raise KeyError(
                f"Could not find {label} column. "
                f"Tried {candidates}. Available columns: {list(detail.columns)}"
            )
        return None

    function_col = pick_column(
        [
            "function_id",
            "bbob_function_id",
            "fid",
            "function",
            "bbob_function",
            "function_name",
        ],
        "BBOB function id",
    )

    group_col = pick_column(
        [
            "assigned_group_0307",
            "bbob_group",
            "group",
            "group_id",
            "assigned_group",
            "group_label",
        ],
        "assigned group",
    )

    hyperparameter_col = pick_column(
        [
            "hyperparameter",
            "parameter",
            "feature",
            "feature_name",
            "hp",
        ],
        "hyperparameter",
    )

    group_value_col = pick_column(
        [
            "group_importance_norm",
            "group_shap_importance_norm",
            "group_importance",
            "group_imp",
            "group_importance_mean",
            "group_mean_importance",
            "group_mean_abs_shap_norm",
            "group_mean_abs_shap",
            "group_value",
            "Group imp.",
            "Group imp",
        ],
        "group SHAP importance",
    )

    function_value_col = pick_column(
        [
            "function_importance_norm",
            "function_shap_importance_norm",
            "function_importance",
            "bbob_importance_norm",
            "bbob_importance",
            "function_imp",
            "bbob_imp",
            "function_mean_abs_shap_norm",
            "function_mean_abs_shap",
            "individual_importance_norm",
            "individual_importance",
            "BBOB imp.",
            "BBOB imp",
        ],
        "individual BBOB SHAP importance",
    )

    def function_sort_value(value):
        text = str(value)
        nums = re.findall(r"\d+", text)
        if nums:
            return int(nums[-1])
        return 10**9

    def function_label(value):
        n = function_sort_value(value)
        if n != 10**9:
            return f"function{n}"
        return str(value)

    def group_label(value):
        text = str(value)
        if text.startswith("G"):
            return text
        nums = re.findall(r"\d+", text)
        if nums:
            return f"G{int(nums[-1])}"
        return text

    def fmt_cell(row):
        g = row["_group_value"]
        f = row["_function_value"]
        if pd.isna(g) or pd.isna(f):
            return "--"
        d = g - f
        return f"{g:.3f}/{f:.3f} ({d:+.3f})"

    detail["_function_sort"] = detail[function_col].map(function_sort_value)
    detail["function"] = detail[function_col].map(function_label)
    detail["group"] = detail[group_col].map(group_label)
    detail["hyperparameter"] = detail[hyperparameter_col].astype(str)

    detail["_group_value"] = pd.to_numeric(detail[group_value_col], errors="coerce")
    detail["_function_value"] = pd.to_numeric(detail[function_value_col], errors="coerce")

    compact = (
        detail.groupby(
            ["_function_sort", "function", "group", "hyperparameter"],
            as_index=False,
            dropna=False,
        )
        .agg(
            _group_value=("_group_value", "mean"),
            _function_value=("_function_value", "mean"),
        )
    )
    compact["_cell"] = compact.apply(fmt_cell, axis=1)

    preferred_hyperparameters = [
        hp for hp in globals().get("HYPERPARAMETERS", [])
        if hp in set(compact["hyperparameter"])
    ]
    remaining_hyperparameters = sorted(
        hp for hp in compact["hyperparameter"].unique()
        if hp not in preferred_hyperparameters
    )
    hyperparameter_order = preferred_hyperparameters + remaining_hyperparameters

    wide = (
        compact.pivot_table(
            index=["_function_sort", "function", "group"],
            columns="hyperparameter",
            values="_cell",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("_function_sort")
    )

    for hp in hyperparameter_order:
        if hp not in wide.columns:
            wide[hp] = "--"

    wide = wide[["function", "group"] + hyperparameter_order]
    wide = wide.fillna("--")

    return wide

def latex_escape_text(x):
    s = str(x)

    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }

    return "".join(replacements.get(ch, ch) for ch in s)


def df_to_latex_table1_wide(df):
    cols = list(df.columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-function BBOB SHAP-importance comparison against the assigned 0307 group reference. "
        r"Each hyperparameter cell reports group importance/function importance (group minus function).}"
    )
    lines.append(r"\label{tab:bbob_function_group_shap_comparison}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{ll" + "c" * (len(cols) - 2) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape_text(c) for c in cols) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append("" if pd.isna(v) else latex_escape_text(v))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

# =========================

def normalize_hof_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def format_predicted_actual_delta(predicted, actual):
    pred = normalize_hof_value(predicted)
    act = normalize_hof_value(actual)

    if pred == "" and act == "":
        return "--"

    pred_num = pd.to_numeric(pred, errors="coerce")
    act_num = pd.to_numeric(act, errors="coerce")

    if pd.notna(pred_num) and pd.notna(act_num):
        delta = float(pred_num) - float(act_num)
        return f"{pred}/{act} ({delta:+.3f})"

    if pred == act:
        return f"{pred}/{act} (match)"

    return f"{pred}/{act} (diff)"


def latex_escape_cell(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_first_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def find_hof_value_column(df, hp, side):
    """Find predicted/actual HoF value column for one hyperparameter.

    side must be "predicted" or "actual".
    This is tolerant to several Step10D / hall_of_fame naming styles.
    """
    hp_candidates = [hp, hp.replace("lambda_", "lambda"), hp.replace("_", "")]
    if side == "predicted":
        templates = [
            "predicted_{hp}",
            "predicted_hof_{hp}",
            "predicted_group_{hp}",
            "group_hof_{hp}",
            "reference_{hp}",
            "pred_{hp}",
        ]
    else:
        templates = [
            "actual_{hp}",
            "actual_hof_{hp}",
            "best_{hp}",
            "function_hof_{hp}",
            "generated_hof_{hp}",
            "act_{hp}",
        ]

    for h in hp_candidates:
        for template in templates:
            col = template.format(hp=h)
            if col in df.columns:
                return col

    # Case-insensitive fallback.
    lowered = {c.lower(): c for c in df.columns}
    for h in hp_candidates:
        h_low = h.lower()
        for template in templates:
            col_low = template.format(hp=h_low).lower()
            if col_low in lowered:
                return lowered[col_low]

    return None


def build_table2_generated_predicted_actual_wide(table2: pd.DataFrame) -> pd.DataFrame:
    """Build wide generated-function HoF table from the original Table 2 output.

    Original Table 2 stores each generated function as two rows:
        - function HoF actual
        - group HoF predicted

    The wide table keeps:
        Type, Function, Group

    Then each hyperparameter cell reports:
        predicted / actual (predicted - actual)

    For non-numeric hyperparameters, the parenthesized value is match/diff.
    """
    df = table2.copy()
    if df.empty:
        return pd.DataFrame()

    required = [
        "function_type",
        "function_id",
        "function_label",
        "assigned_group_0307",
        "config_source",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Table 2 wide conversion is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    hyperparameters = [
        hp for hp in globals().get("HYPERPARAMETERS", [])
        if hp in df.columns
    ]
    if not hyperparameters:
        hyperparameters = [
            c for c in df.columns
            if c not in {
                "function_type",
                "function_id",
                "function_label",
                "assigned_group_0307",
                "predicted_source_bbob_function",
                "group_n_bbob_functions",
                "group_bbob_functions",
                "prediction_reference_method",
                "delta_auc_actual_minus_predicted",
                "config_match_count",
                "config_available_count",
                "config_match_ratio",
                "mismatch_config_fields",
                "status",
                "config_source",
                "auc_mean",
                "config_signature",
                "note",
            }
        ]

    numeric_delta_hps = {"CR", "F", "lambda_", "mutation_n_comps"}

    def normalize_value(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, bool):
            return str(value)
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text.lower() == "nan":
            return "NA"
        return text

    def format_cell(predicted, actual, hp):
        pred = normalize_value(predicted)
        act = normalize_value(actual)

        if hp in numeric_delta_hps:
            pred_num = pd.to_numeric(pred, errors="coerce")
            act_num = pd.to_numeric(act, errors="coerce")
            if pd.notna(pred_num) and pd.notna(act_num):
                delta = float(pred_num) - float(act_num)
                return f"{pred}/{act} ({delta:+.3f})"

        if pred == act:
            return f"{pred}/{act} (match)"
        return f"{pred}/{act} (diff)"

    key_cols = [
        "function_type",
        "function_id",
        "function_label",
        "assigned_group_0307",
    ]

    rows = []
    grouped = df.groupby(key_cols, dropna=False, sort=False)
    for key, group_df in grouped:
        actual_rows = group_df[
            group_df["config_source"].astype(str).str.contains("actual", case=False, na=False)
        ]
        predicted_rows = group_df[
            group_df["config_source"].astype(str).str.contains("predicted", case=False, na=False)
        ]

        if actual_rows.empty or predicted_rows.empty:
            continue

        actual = actual_rows.iloc[0]
        predicted = predicted_rows.iloc[0]

        row = {
            "Type": actual["function_type"],
            "Function": actual["function_label"],
            "Group": actual["assigned_group_0307"],
        }

        for hp in hyperparameters:
            row[hp] = format_cell(predicted[hp], actual[hp], hp)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "Could not build wide Table 2 because no actual/predicted row pairs were found. "
            f"config_source values: {sorted(df['config_source'].astype(str).unique())}"
        )

    out = out.sort_values(["Type", "Function", "Group"]).reset_index(drop=True)
    return out

def df_to_latex_table2_wide(df: pd.DataFrame) -> str:
    column_spec = "lll" + ("l" * max(0, len(df.columns) - 3))
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{Generated functions: predicted BBOB group-mean reference compared with actual function-level Hall-of-Fame configuration under the 0307 grouping. Each hyperparameter cell reports predicted/actual (predicted-actual).}")
    lines.append(r"\label{tab:generated-predicted-actual-hof-wide-0307}")
    lines.append(r"\begin{tabular}{" + column_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape_cell(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape_cell(row[c]) for c in df.columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)



def _pick_col_from_df(df: pd.DataFrame, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find {label}. Tried {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _format_short_type(value):
    text = str(value)
    low = text.lower()
    if "llamea" in low:
        return "LLaMEA"
    if "affine" in low:
        return "affine"
    return text


def _format_numeric_cell(group_value, function_value):
    g = pd.to_numeric(group_value, errors="coerce")
    f = pd.to_numeric(function_value, errors="coerce")

    if pd.isna(g) or pd.isna(f):
        return "--"

    d = float(g) - float(f)
    return f"{float(g):.3f}/{float(f):.3f} ({d:+.3f})"


def _latex_escape_table_value(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_generated_group_function_metric_wide(
    detail: pd.DataFrame,
    group_metric_candidates,
    function_metric_candidates,
    metric_label: str,
) -> pd.DataFrame:
    """Build Type/Function/Group wide table for generated-function SHAP metrics.

    Each hyperparameter cell is:
        group metric / function metric (group - function)
    """
    df = detail.copy()
    if df.empty:
        return pd.DataFrame()

    type_col = _pick_col_from_df(
        df,
        ["function_type", "type", "Type", "source_type", "generated_type"],
        "function type",
    )
    function_col = _pick_col_from_df(
        df,
        ["function_label", "function_name", "function_id", "function", "Function", "generated_function"],
        "function label",
    )
    group_col = _pick_col_from_df(
        df,
        ["assigned_group_0307", "group_label_0307", "group", "Group", "predicted_group"],
        "assigned group",
    )
    hp_col = _pick_col_from_df(
        df,
        ["hyperparameter", "parameter", "feature", "feature_name", "hp"],
        "hyperparameter",
    )

    group_metric_col = _pick_col_from_df(
        df,
        group_metric_candidates,
        f"group {metric_label} metric",
    )
    function_metric_col = _pick_col_from_df(
        df,
        function_metric_candidates,
        f"function {metric_label} metric",
    )

    preferred_hps = [
        hp for hp in globals().get("HYPERPARAMETERS", [])
        if hp in set(df[hp_col].astype(str))
    ]
    remaining_hps = sorted(
        hp for hp in df[hp_col].astype(str).unique()
        if hp not in preferred_hps
    )
    hp_order = preferred_hps + remaining_hps

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Type": _format_short_type(r[type_col]),
            "Function": str(r[function_col]),
            "Group": str(r[group_col]),
            "hyperparameter": str(r[hp_col]),
            "_cell": _format_numeric_cell(r[group_metric_col], r[function_metric_col]),
        })

    compact = pd.DataFrame(rows)

    wide = (
        compact.pivot_table(
            index=["Type", "Function", "Group"],
            columns="hyperparameter",
            values="_cell",
            aggfunc="first",
        )
        .reset_index()
    )

    for hp in hp_order:
        if hp not in wide.columns:
            wide[hp] = "--"

    wide = wide[["Type", "Function", "Group"] + hp_order]
    wide = wide.fillna("--")

    type_order = {"affine": 0, "LLaMEA": 1}
    wide["_type_order"] = wide["Type"].map(type_order).fillna(9)
    wide = wide.sort_values(["_type_order", "Function", "Group"]).drop(columns=["_type_order"])
    wide = wide.reset_index(drop=True)

    return wide


def build_generated_distance_metric_wide(
    detail: pd.DataFrame,
    distance_col: str,
    decimals: int = 3,
) -> pd.DataFrame:
    """Build Type/Function/Group wide table where each cell is one distance.

    Smaller values indicate more similar group and function distributions.
    """
    df = detail.copy()
    if df.empty:
        return pd.DataFrame()

    type_col = _pick_col_from_df(
        df,
        ["function_type", "type", "Type", "source_type", "generated_type"],
        "function type",
    )
    function_col = _pick_col_from_df(
        df,
        ["function_label", "function_name", "function_id", "function", "Function", "generated_function"],
        "function label",
    )
    group_col = _pick_col_from_df(
        df,
        ["assigned_group_0307", "group_label_0307", "group", "Group", "predicted_group"],
        "assigned group",
    )
    hp_col = _pick_col_from_df(
        df,
        ["hyperparameter", "parameter", "feature", "feature_name", "hp"],
        "hyperparameter",
    )

    if distance_col not in df.columns:
        raise KeyError(f"Missing distance column {distance_col}. Available columns: {list(df.columns)}")

    preferred_hps = [
        hp for hp in globals().get("HYPERPARAMETERS", [])
        if hp in set(df[hp_col].astype(str))
    ]
    remaining_hps = sorted(
        hp for hp in df[hp_col].astype(str).unique()
        if hp not in preferred_hps
    )
    hp_order = preferred_hps + remaining_hps

    def fmt_distance(x):
        x = safe_float(x)
        if not np.isfinite(x):
            return "--"
        return f"{x:.{decimals}f}"

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Type": _format_short_type(r[type_col]),
            "Function": str(r[function_col]),
            "Group": str(r[group_col]),
            "hyperparameter": str(r[hp_col]),
            "_cell": fmt_distance(r[distance_col]),
        })

    compact = pd.DataFrame(rows)
    wide = (
        compact.pivot_table(
            index=["Type", "Function", "Group"],
            columns="hyperparameter",
            values="_cell",
            aggfunc="first",
        )
        .reset_index()
    )

    for hp in hp_order:
        if hp not in wide.columns:
            wide[hp] = "--"

    wide = wide[["Type", "Function", "Group"] + hp_order]
    wide = wide.fillna("--")

    type_order = {"affine": 0, "LLaMEA": 1}
    wide["_type_order"] = wide["Type"].map(type_order).fillna(9)
    wide = wide.sort_values(["_type_order", "Function", "Group"]).drop(columns=["_type_order"])
    wide = wide.reset_index(drop=True)

    return wide


def build_table3_generated_shap_shape_wide(table3: pd.DataFrame) -> pd.DataFrame:
    """Wide Table 3: group vs generated-function raw SHAP distribution shape.

    Default shape descriptor: skewness.
    If skewness columns are unavailable, this falls back to other shape-like columns.
    """
    return build_generated_group_function_metric_wide(
        table3,
        group_metric_candidates=[
            "group_skewness",
            "group_shap_skewness",
            "group_raw_skewness",
            "group_distribution_skewness",
            "group_shape_skewness",
            "group_shape_value",
        ],
        function_metric_candidates=[
            "function_skewness",
            "function_shap_skewness",
            "function_raw_skewness",
            "function_distribution_skewness",
            "function_shape_skewness",
            "function_shape_value",
        ],
        metric_label="shape",
    )


def build_table3_wasserstein_wide(table3: pd.DataFrame) -> pd.DataFrame:
    return build_generated_distance_metric_wide(
        table3,
        distance_col="wasserstein_distance",
        decimals=3,
    )


def build_table3_kl_wide(table3: pd.DataFrame) -> pd.DataFrame:
    return build_generated_distance_metric_wide(
        table3,
        distance_col="kl_divergence",
        decimals=3,
    )


def build_table4_generated_shap_value_wide(table4: pd.DataFrame) -> pd.DataFrame:
    """Wide Table 4: matched feature-value SHAP distance.

    Each cell averages |function SHAP - nearest group SHAP| after matching
    each generated-function point to a group point with the same or nearest
    feature value. Smaller means more similar local SHAP behavior.
    """
    if "matched_feature_value_mean_abs_distance" in table4.columns:
        return build_generated_distance_metric_wide(
            table4,
            distance_col="matched_feature_value_mean_abs_distance",
            decimals=3,
        )

    if "wasserstein_distance" in table4.columns:
        return build_generated_distance_metric_wide(
            table4,
            distance_col="wasserstein_distance",
            decimals=3,
        )

    return build_generated_group_function_metric_wide(
        table4,
        group_metric_candidates=[
            "group_mean_abs_shap",
            "group_importance_norm",
            "group_shap_importance_norm",
        ],
        function_metric_candidates=[
            "function_mean_abs_shap",
            "function_importance_norm",
            "function_shap_importance_norm",
        ],
        metric_label="mean absolute SHAP value",
    )

def df_to_compact_wide_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    """Compact Overleaf table for wide SHAP comparison tables.

    Requires:
        \\usepackage{booktabs}
        \\usepackage{adjustbox}
    """
    column_spec = "lll" + ("l" * max(0, len(df.columns) - 3))
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{1.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.88}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{adjustbox}{width=\textwidth,center}")
    lines.append(r"\begin{tabular}{" + column_spec + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(_latex_escape_table_value(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape_table_value(row[c]) for c in df.columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


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

    # Additional requested Table 1 detail and wide-format outputs.
    table1_detail.to_csv(OUT_TABLE1_DETAIL_CSV, index=False)

    table1_wide = build_table1_bbob_function_wide(table1_detail)
    table1_wide.to_csv(OUT_TABLE1_WIDE_CSV, index=False)

    with open(OUT_TABLE1_WIDE_MD, "w", encoding="utf-8") as f:
        f.write("# Table 1 wide. BBOB per-function SHAP comparison against assigned group\n\n")
        f.write(
            "Each hyperparameter cell reports `group importance/function importance "
            "(group minus function)`.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table1_wide))
        f.write("\n")

    with open(OUT_TABLE1_WIDE_TEX, "w", encoding="utf-8") as f:
        f.write(df_to_latex_table1_wide(table1_wide))
        f.write("\n")

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

    table2_wide = build_table2_generated_predicted_actual_wide(table2)
    table2_wide.to_csv(OUT_TABLE2_WIDE_CSV, index=False)
    OUT_TABLE2_WIDE_MD.write_text(table2_wide.to_markdown(index=False), encoding="utf-8")
    OUT_TABLE2_WIDE_TEX.write_text(df_to_latex_table2_wide(table2_wide), encoding="utf-8")

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
        "prediction_reference_method",
        "status",
    ]

    with open(OUT_TABLE2_MD, "w", encoding="utf-8") as f:
        f.write("# Table 2. Generated functions: predicted BBOB group mean vs actual function Hall-of-Fame\n\n")
        f.write(
            "Each generated function is shown with two rows: the predicted configuration from "
            "the assigned group's BBOB Hall-of-Fame mean reference, and the actual "
            "function-level Hall-of-Fame winner. Numeric hyperparameters in the predicted "
            "row are group means; categorical hyperparameters are group modes.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table2[[c for c in table2_display_cols if c in table2.columns]]))
        f.write("\n")

    # --------------------------------------------------------
    # Table 3
    # --------------------------------------------------------
    print("Building Table 3: generated SHAP shape-distance table...")

    table3 = build_table3_generated_shape_distance(hof)

    table3.to_csv(OUT_TABLE3_CSV, index=False)

    table3_wide = build_table3_generated_shap_shape_wide(table3)
    table3_wide.to_csv(OUT_TABLE3_WIDE_CSV, index=False)
    OUT_TABLE3_WIDE_MD.write_text(table3_wide.to_markdown(index=False), encoding="utf-8")
    OUT_TABLE3_WIDE_TEX.write_text(
        df_to_compact_wide_latex(
            table3_wide,
            caption="Generated functions: per-function and per-hyperparameter comparison of raw SHAP distribution shape between the assigned BBOB group and the generated function. Each cell reports group/function (group-function), using skewness as the distribution-shape descriptor.",
            label="tab:generated-shap-shape-wide-0307",
        ),
        encoding="utf-8",
    )

    table3_wasserstein_wide = build_table3_wasserstein_wide(table3)
    table3_wasserstein_wide.to_csv(OUT_TABLE3_WASSERSTEIN_WIDE_CSV, index=False)
    OUT_TABLE3_WASSERSTEIN_WIDE_MD.write_text(
        table3_wasserstein_wide.to_markdown(index=False),
        encoding="utf-8",
    )
    OUT_TABLE3_WASSERSTEIN_WIDE_TEX.write_text(
        df_to_compact_wide_latex(
            table3_wasserstein_wide,
            caption="Generated functions: Wasserstein distance between the assigned BBOB group SHAP distribution and the generated-function SHAP distribution. Each cell reports one distance value; smaller values indicate more similar distribution shape.",
            label="tab:generated-shap-shape-wasserstein-0307",
        ),
        encoding="utf-8",
    )

    table3_kl_wide = build_table3_kl_wide(table3)
    table3_kl_wide.to_csv(OUT_TABLE3_KL_WIDE_CSV, index=False)
    OUT_TABLE3_KL_WIDE_MD.write_text(
        table3_kl_wide.to_markdown(index=False),
        encoding="utf-8",
    )
    OUT_TABLE3_KL_WIDE_TEX.write_text(
        df_to_compact_wide_latex(
            table3_kl_wide,
            caption="Generated functions: Kullback-Leibler divergence between the assigned BBOB group SHAP distribution and the generated-function SHAP distribution. Each cell reports one divergence value; smaller values indicate more similar distribution shape.",
            label="tab:generated-shap-shape-kl-0307",
        ),
        encoding="utf-8",
    )

    table3_display_cols = [
        "function_type",
        "function_id",
        "assigned_group_0307",
        "hyperparameter",
        "shape_distance_metric",
        "wasserstein_distance",
        "kl_divergence",
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

    table4_wide = build_table4_generated_shap_value_wide(table4)
    table4_wide.to_csv(OUT_TABLE4_WIDE_CSV, index=False)
    OUT_TABLE4_WIDE_MD.write_text(table4_wide.to_markdown(index=False), encoding="utf-8")
    OUT_TABLE4_WIDE_TEX.write_text(
        df_to_compact_wide_latex(
            table4_wide,
            caption="Generated functions: matched SHAP-value distance between the assigned BBOB group and the generated function. Each generated-function point is paired with the same or nearest group feature value; each cell reports the mean absolute SHAP-value distance, where smaller values indicate more similar local SHAP behavior.",
            label="tab:generated-shap-value-wide-0307",
        ),
        encoding="utf-8",
    )

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
        "wasserstein_distance",
        "matched_feature_value_mean_abs_distance",
        "matched_n_pairs",
        "matched_mean_feature_value_gap",
        "group_importance_norm",
        "function_importance_norm",
        "importance_signed_difference_function_minus_group",
        "importance_absolute_difference",
        "group_record_mode",
        "function_record_mode",
        "group_distribution_mode",
        "function_distribution_mode",
        "status",
    ]

    with open(OUT_TABLE4_MD, "w", encoding="utf-8") as f:
        f.write("# Table 4. Generated functions: SHAP value distance\n\n")
        f.write(
            "For each generated function and hyperparameter, this table compares the assigned "
            "group SHAP values with the individual function SHAP values. The wide table reports "
            "`matched_feature_value_mean_abs_distance`: for each generated-function point, the script "
            "finds the same or nearest group feature value and averages the absolute SHAP-value "
            "difference. Wasserstein distance is retained in the detailed CSV as an unpaired "
            "distribution-distance reference.\n\n"
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
        f.write(f"- Table 1 CSV: {display_path(OUT_TABLE1_CSV)}\n")
        f.write(f"- Table 1 MD:  {display_path(OUT_TABLE1_MD)}\n")
        f.write(f"- Table 1 detail CSV: {display_path(OUT_TABLE1_DETAIL_CSV)}\n")
        f.write(f"- Table 1 wide CSV:   {display_path(OUT_TABLE1_WIDE_CSV)}\n")
        f.write(f"- Table 1 wide MD:    {display_path(OUT_TABLE1_WIDE_MD)}\n")
        f.write(f"- Table 1 wide TeX:   {display_path(OUT_TABLE1_WIDE_TEX)}\n")
        f.write(f"- Table 2 CSV: {display_path(OUT_TABLE2_CSV)}\n")
        f.write(f"- Table 2 MD:  {display_path(OUT_TABLE2_MD)}\n")
        f.write(f"- Table 3 CSV: {display_path(OUT_TABLE3_CSV)}\n")
        f.write(f"- Table 3 MD:  {display_path(OUT_TABLE3_MD)}\n")
        f.write(f"- Table 3 Wasserstein wide CSV: {display_path(OUT_TABLE3_WASSERSTEIN_WIDE_CSV)}\n")
        f.write(f"- Table 3 Wasserstein wide TeX: {display_path(OUT_TABLE3_WASSERSTEIN_WIDE_TEX)}\n")
        f.write(f"- Table 3 KL wide CSV: {display_path(OUT_TABLE3_KL_WIDE_CSV)}\n")
        f.write(f"- Table 3 KL wide TeX: {display_path(OUT_TABLE3_KL_WIDE_TEX)}\n")
        f.write(f"- Table 4 CSV: {display_path(OUT_TABLE4_CSV)}\n")
        f.write(f"- Table 4 MD:  {display_path(OUT_TABLE4_MD)}\n")
        f.write(f"- Table 5 CSV: {display_path(OUT_TABLE5_CSV)}\n")
        f.write(f"- Table 5 MD:  {display_path(OUT_TABLE5_MD)}\n")
        f.write("\n")
        f.write("Inputs:\n")
        f.write(f"- Hall of Fame: {display_path(HALL_OF_FAME_CSV)}\n")
        f.write(f"- SHAP similarity: {display_path(SHAP_SIMILARITY_CSV)}\n")
        f.write(f"- Group SHAP dir: {display_path(GROUP_SHAP_DIR)}\n")
        f.write(f"- BBOB individual SHAP dir: {display_path(BBOB_INDIVIDUAL_SHAP_DIR)}\n")
        f.write(f"- Affine SHAP dir: {display_path(AFFINE_SHAP_DIR)}\n")
        f.write(f"- LLaMEA SHAP dir: {display_path(LLAMEA_SHAP_DIR)}\n")
    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDone.")
    print(f"Saved Table 1 CSV: {OUT_TABLE1_CSV}")
    print(f"Saved Table 1 detail CSV: {OUT_TABLE1_DETAIL_CSV}")
    print(f"Saved Table 1 wide CSV:   {OUT_TABLE1_WIDE_CSV}")
    print(f"Saved Table 1 wide MD:    {OUT_TABLE1_WIDE_MD}")
    print(f"Saved Table 1 wide TeX:   {OUT_TABLE1_WIDE_TEX}")
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
