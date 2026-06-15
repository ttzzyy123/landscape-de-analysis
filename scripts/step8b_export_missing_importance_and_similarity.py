from pathlib import Path
import os
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool


# ============================================================
# Step 8B
# Export missing SHAP importance CSVs for:
# 4.6 partial-function instance-level SHAP similarity
# 4.7 generated / affine unseen-function SHAP similarity
#
# Then this script also computes the similarity tables directly.
# ============================================================


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_COL = os.getenv("STEP8B_TARGET", "auc")
SAMPLE_SIZE = int(os.getenv("STEP8B_SAMPLE_SIZE", "10000"))
MODEL_ITERATIONS = int(os.getenv("STEP8B_ITERATIONS", "250"))
RANDOM_STATE = int(os.getenv("STEP8B_RANDOM_STATE", "42"))

INSTANCE_SCHEME = "eps2bins_4_r025050075_instance"

OUTPUT_DIR = PROJECT_ROOT / "output" / "step8b_export_missing_importance_and_similarity"
OUT_46 = OUTPUT_DIR / "04_6_partial_function_effects"
OUT_47 = OUTPUT_DIR / "04_7_unseen_generalisation"

for d in [OUTPUT_DIR, OUT_46, OUT_47]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# Main project paths
# =========================
DATA_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"

# Existing full-function SHAP vectors from old pipeline, if available
FUNCTION_SHAP_DIR = PROJECT_ROOT / "output" / "shap_individual_manual_bins_niles"

# Existing function-level group SHAP vectors from old pipeline, if available
GROUP_SHAP_DIR = PROJECT_ROOT / "output" / "shap_eps2bins_3_r0307_main_tiebreak_niles"

# Instance-level output folder used by existing plots
INSTANCE_OUTPUT_BASE = (
    PROJECT_ROOT
    / "output"
    / "step7_instance_level_group_and_partial_function_plots"
    / INSTANCE_SCHEME
)

INSTANCE_IMPORTANCE_OUT = INSTANCE_OUTPUT_BASE / "03_importance_csv_for_similarity"
INSTANCE_IMPORTANCE_OUT.mkdir(parents=True, exist_ok=True)

# New function task dirs
REAL_MODDE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "modde_real_function_results"
AFFINE_MODDE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "modde_affine_function_results"

REAL_SHAP_OUT = PROJECT_ROOT / "output" / "new_function_task" / "task_H_individual_shap_real_generated_function"
AFFINE_SHAP_OUT = PROJECT_ROOT / "output" / "new_function_task" / "task_H_individual_shap_affine_functions"

REAL_ASSIGNMENT_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "real_function_group_assignment"
AFFINE_ASSIGNMENT_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "affine_function_group_assignment"

for d in [REAL_SHAP_OUT, AFFINE_SHAP_OUT]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# Utility functions
# =========================
def log(msg: str):
    print(msg, flush=True)


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_pickle(path: Path):
    try:
        return pd.read_pickle(path)
    except Exception as e:
        warnings.warn(f"Could not read pickle {path}: {e}")
        return None


def normalize_importance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total = df["importance"].abs().sum()
    if total > 0:
        df["importance_norm"] = df["importance"].abs() / total
    else:
        df["importance_norm"] = 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def vector_from_importance_csv(path: Path):
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    feature_col = None
    for c in ["feature", "Feature", "parameter", "param", "name"]:
        if c in df.columns:
            feature_col = c
            break

    imp_col = None
    for c in ["importance_norm", "importance", "mean_abs_shap", "shap_importance"]:
        if c in df.columns:
            imp_col = c
            break

    if feature_col is None or imp_col is None:
        return None

    return dict(zip(df[feature_col].astype(str), df[imp_col].astype(float)))


def cosine_dict(a: dict, b: dict):
    if not a or not b:
        return np.nan

    keys = sorted(set(a) | set(b))
    x = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    y = np.array([b.get(k, 0.0) for k in keys], dtype=float)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx == 0 or ny == 0:
        return np.nan

    return float(np.dot(x, y) / (nx * ny))


def extract_group_id(text):
    text = str(text)
    patterns = [
        r"group[_\- ]?(\d+)",
        r"G(\d+)",
        r"group_id[_\- ]?(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_fid(text):
    text = str(text)
    patterns = [
        r"(?:^|[_\-])f(\d+)(?:[_\-\.]|$)",
        r"function[_\-]?(\d+)",
        r"fid[_\-]?(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def clean_function_name_from_path(path: Path):
    stem = path.stem
    stem = re.sub(r"_modde_results_processed$", "", stem)
    stem = re.sub(r"_modde_results$", "", stem)
    stem = re.sub(r"_processed$", "", stem)
    stem = re.sub(r"_individual_shap_importance$", "", stem)
    return stem


def standardize_fid_iid_cols(df: pd.DataFrame):
    df = df.copy()

    rename = {}
    for c in df.columns:
        lc = str(c).lower()

        if lc in ["function", "function_id", "fid", "bbob_function"]:
            rename[c] = "fid"
        elif lc in ["instance", "instance_id", "iid"]:
            rename[c] = "iid"
        elif lc in ["group", "assigned_group"]:
            rename[c] = "group_id"

    df = df.rename(columns=rename)

    # remove duplicated columns after renaming
    df = df.loc[:, ~df.columns.duplicated()]

    if "fid" in df.columns:
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce")
        df = df.dropna(subset=["fid"])
        df["fid"] = df["fid"].astype(int)

    if "iid" in df.columns:
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce")
        df = df.dropna(subset=["iid"])
        df["iid"] = df["iid"].astype(int)

    if "group_id" in df.columns:
        df["group_id"] = df["group_id"].apply(
            lambda x: extract_group_id(x) if isinstance(x, str) else x
        )
        df["group_id"] = pd.to_numeric(df["group_id"], errors="coerce")
        df = df.dropna(subset=["group_id"])
        df["group_id"] = df["group_id"].astype(int)

    return df

def find_instance_assignment_file():
    candidates = []

    for root in [PROJECT_ROOT / "intermediate", PROJECT_ROOT / "output"]:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            name = str(p).lower()
            if INSTANCE_SCHEME.lower() in name or "instance_level" in name:
                df = safe_read_csv(p)
                if df is None or df.empty:
                    continue

                df2 = standardize_fid_iid_cols(df.copy())
                cols = set(df2.columns)
                if {"fid", "iid", "group_id"}.issubset(cols):
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not find instance-level assignment CSV with columns fid, iid, group_id.\n"
            "Search location: intermediate/ and output/.\n"
            f"Expected scheme keyword: {INSTANCE_SCHEME}"
        )

    # Prefer files that contain the exact scheme name
    candidates = sorted(
        candidates,
        key=lambda p: (INSTANCE_SCHEME.lower() not in str(p).lower(), len(str(p))),
    )

    return candidates[0]


def choose_feature_columns(df: pd.DataFrame):
    exclude = {
        "fid",
        "iid",
        "dim",
        "seed",
        "auc",
        "aucLarge",
        "auc_list",
        "aucLarge_list",
        "group_id",
        "group_label",
        "function",
        "function_name",
        "name",
    }

    preferred = [
        "CR",
        "F",
        "lambda_",
        "lpsr",
        "mutation_n_comps",
        "use_archive",
        "adaptation_method",
        "crossover",
        "mutation_base",
        "mutation_reference",
        "Instance variance",
        "Stochastic variance",
        "instance_variance",
        "stochastic_variance",
    ]

    cols = [c for c in preferred if c in df.columns and c != TARGET_COL]

    # Add other usable columns, but avoid lists / object blobs.
    for c in df.columns:
        if c in cols or c in exclude or c == TARGET_COL:
            continue
        if df[c].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]) or df[c].dtype == object:
            cols.append(c)

    return cols


def shap_importance_from_df(df: pd.DataFrame, out_csv: Path, title: str = ""):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[TARGET_COL]).copy()
    if df.empty:
        raise ValueError("Empty dataframe after dropping missing target values.")

    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    feature_cols = choose_feature_columns(df)
    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(float).copy()

    # One-hot encode categorical columns for robust CatBoost/SHAP compatibility.
    X = pd.get_dummies(X, dummy_na=True)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = CatBoostRegressor(
        iterations=MODEL_ITERATIONS,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X, y)

    pool = Pool(X, y)
    shap_values = model.get_feature_importance(pool, type="ShapValues")

    # Last column is expected value.
    vals = np.abs(shap_values[:, :-1]).mean(axis=0)

    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": vals,
        }
    )

    imp = normalize_importance(imp)
    imp.to_csv(out_csv, index=False)

    return imp


def load_existing_function_vectors():
    vectors = {}

    if not FUNCTION_SHAP_DIR.exists():
        return vectors

    for path in FUNCTION_SHAP_DIR.rglob("*.csv"):
        fid = extract_fid(path.name)
        if fid is None:
            continue
        v = vector_from_importance_csv(path)
        if v:
            vectors[fid] = v

    return vectors


def load_existing_group_vectors():
    vectors = {}

    if not GROUP_SHAP_DIR.exists():
        return vectors

    # individual group files
    for path in GROUP_SHAP_DIR.rglob("*.csv"):
        gid = extract_group_id(path.name)
        v = vector_from_importance_csv(path)
        if gid is not None and v:
            vectors[gid] = v

    # all-groups file
    for path in GROUP_SHAP_DIR.rglob("*all*group*.csv"):
        df = safe_read_csv(path)
        if df is None or df.empty:
            continue

        feature_col = "feature" if "feature" in df.columns else None
        imp_col = "importance_norm" if "importance_norm" in df.columns else "importance" if "importance" in df.columns else None
        group_col = "group_id" if "group_id" in df.columns else None

        if feature_col and imp_col and group_col:
            for gid, sub in df.groupby(group_col):
                vectors[int(gid)] = dict(zip(sub[feature_col].astype(str), sub[imp_col].astype(float)))

    return vectors


# ============================================================
# 4.6 Partial-function export and similarity
# ============================================================
def run_46_partial_function_exports():
    log("\n" + "=" * 80)
    log("4.6 Exporting instance-level group and partial-function SHAP importance CSVs")
    log("=" * 80)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Main processed data not found: {DATA_FILE}")

    assignment_file = find_instance_assignment_file()
    log(f"Using instance assignment file:\n{assignment_file}")

    assign_df = pd.read_csv(assignment_file)
    assign_df = standardize_fid_iid_cols(assign_df)

    assign_df = assign_df[["fid", "iid", "group_id"]].drop_duplicates()

    data = pd.read_pickle(DATA_FILE)
    data = standardize_fid_iid_cols(data)

    if "fid" not in data.columns or "iid" not in data.columns:
        raise ValueError("Main data must contain fid and iid columns.")

    merged = data.merge(assign_df, on=["fid", "iid"], how="inner")

    log(f"Main data shape:       {data.shape}")
    log(f"Assignment shape:      {assign_df.shape}")
    log(f"Merged instance shape: {merged.shape}")

    # -------------------------
    # Group-level instance vectors
    # -------------------------
    instance_group_vectors = {}

    group_dir = INSTANCE_IMPORTANCE_OUT / "instance_group_importance"
    group_dir.mkdir(parents=True, exist_ok=True)

    for gid in sorted(merged["group_id"].unique()):
        out_csv = group_dir / f"instance_group_{gid}_importance.csv"

        if out_csv.exists():
            v = vector_from_importance_csv(out_csv)
            if v:
                instance_group_vectors[int(gid)] = v
                continue

        sub = merged[merged["group_id"] == gid]
        log(f"Training instance group G{gid}, rows={len(sub)}")

        try:
            imp = shap_importance_from_df(sub, out_csv, title=f"Instance group {gid}")
            instance_group_vectors[int(gid)] = dict(zip(imp["feature"], imp["importance_norm"]))
        except Exception as e:
            warnings.warn(f"Failed instance group {gid}: {e}")

    # -------------------------
    # Full-function vectors
    # -------------------------
    full_function_vectors = load_existing_function_vectors()

    full_dir = INSTANCE_IMPORTANCE_OUT / "full_function_importance"
    full_dir.mkdir(parents=True, exist_ok=True)

    missing_fids = sorted(set(assign_df["fid"].unique()) - set(full_function_vectors.keys()))

    for fid in missing_fids:
        out_csv = full_dir / f"full_function_f{fid}_importance.csv"

        if out_csv.exists():
            v = vector_from_importance_csv(out_csv)
            if v:
                full_function_vectors[int(fid)] = v
                continue

        sub = data[data["fid"] == fid]
        if sub.empty:
            continue

        log(f"Training full function f{fid}, rows={len(sub)}")

        try:
            imp = shap_importance_from_df(sub, out_csv, title=f"Full function f{fid}")
            full_function_vectors[int(fid)] = dict(zip(imp["feature"], imp["importance_norm"]))
        except Exception as e:
            warnings.warn(f"Failed full function f{fid}: {e}")

    # -------------------------
    # Partial function vectors
    # -------------------------
    partial_dir = INSTANCE_IMPORTANCE_OUT / "partial_function_importance"
    partial_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    pairs = (
        merged[["group_id", "fid"]]
        .drop_duplicates()
        .sort_values(["group_id", "fid"])
        .itertuples(index=False)
    )

    for gid, fid in pairs:
        gid = int(gid)
        fid = int(fid)

        out_csv = partial_dir / f"partial_group_{gid}_f{fid}_importance.csv"

        if out_csv.exists():
            partial_vec = vector_from_importance_csv(out_csv)
        else:
            sub = merged[(merged["group_id"] == gid) & (merged["fid"] == fid)]
            if sub.empty:
                continue

            log(f"Training partial G{gid}-f{fid}, rows={len(sub)}")

            try:
                imp = shap_importance_from_df(sub, out_csv, title=f"Partial G{gid} f{fid}")
                partial_vec = dict(zip(imp["feature"], imp["importance_norm"]))
            except Exception as e:
                warnings.warn(f"Failed partial G{gid}-f{fid}: {e}")
                continue

        group_vec = instance_group_vectors.get(gid)
        full_vec = full_function_vectors.get(fid)

        if not group_vec or not partial_vec or not full_vec:
            continue

        full_sim = cosine_dict(group_vec, full_vec)
        partial_sim = cosine_dict(group_vec, partial_vec)

        rows.append(
            {
                "group_id": gid,
                "fid": fid,
                "full_function_similarity": full_sim,
                "partial_function_similarity": partial_sim,
                "improvement": partial_sim - full_sim,
            }
        )

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(OUT_46 / "partial_similarity_improvement.csv", index=False)

    if not sim_df.empty:
        summary = (
            sim_df.groupby("group_id")[["full_function_similarity", "partial_function_similarity", "improvement"]]
            .agg(["count", "mean", "std", "min", "max"])
        )
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(OUT_46 / "partial_similarity_summary_by_group.csv", index=False)

        # Plot full vs partial
        plot_df = sim_df.sort_values("improvement", ascending=False).copy()
        labels = [f"G{g}-f{f}" for g, f in zip(plot_df["group_id"], plot_df["fid"])]
        x = np.arange(len(plot_df))
        width = 0.38

        plt.figure(figsize=(max(10, len(plot_df) * 0.35), 5))
        plt.bar(x - width / 2, plot_df["full_function_similarity"], width, label="Full function")
        plt.bar(x + width / 2, plot_df["partial_function_similarity"], width, label="Partial function")
        plt.xticks(x, labels, rotation=75, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine similarity")
        plt.title("Full vs partial-function SHAP similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_46 / "full_vs_partial_similarity_barplot.png", dpi=300)
        plt.close()

        plt.figure(figsize=(max(10, len(plot_df) * 0.35), 5))
        plt.bar(labels, plot_df["improvement"])
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Partial - full similarity")
        plt.title("Similarity improvement from partial-function grouping")
        plt.tight_layout()
        plt.savefig(OUT_46 / "partial_similarity_improvement_barplot.png", dpi=300)
        plt.close()

    log(f"4.6 outputs saved to: {OUT_46}")
    return sim_df


# ============================================================
# 4.7 Generated / affine importance export and similarity
# ============================================================
def collect_modde_pkls():
    files = []

    if REAL_MODDE_DIR.exists():
        for p in REAL_MODDE_DIR.rglob("*.pkl"):
            files.append(("real", p))

    if AFFINE_MODDE_DIR.exists():
        for p in AFFINE_MODDE_DIR.rglob("*.pkl"):
            files.append(("affine", p))

    return sorted(files, key=lambda x: str(x[1]))


def parse_assignment_files():
    assignments = {}

    # ==========================================================
    # REAL GENERATED FUNCTIONS
    # ==========================================================
    real_csv = (
        PROJECT_ROOT
        / "intermediate"
        / "new_function_task"
        / "real_function_group_assignment"
        / "real_generated_functions_group_assignment_batch.csv"
    )

    if real_csv.exists():
        df = pd.read_csv(real_csv)

        for _, r in df.iterrows():
            gid = extract_group_id(r["assigned_group"])

            if gid is None:
                continue

            # full id
            full_id = str(r["full_id"])
            assignments[full_id] = gid

            # tag (n1 n2 n3 ...)
            tag = str(r["function_tag"])
            assignments[tag] = gid

    # ==========================================================
    # AFFINE FUNCTIONS
    # ==========================================================
    affine_csv = (
        PROJECT_ROOT
        / "intermediate"
        / "new_function_task"
        / "affine_function_group_assignment"
        / "0307_main"
        / "affine_summary_assignment_0307_main.csv"
    )

    if affine_csv.exists():
        df = pd.read_csv(affine_csv)

        for _, r in df.iterrows():
            gid = extract_group_id(r["majority_group"])

            if gid is None:
                continue

            name = str(r["function_name"])

            assignments[name] = gid

            # normalize for filename matching
            normalized = (
                name.replace("affine_", "affine_function_")
                .replace(".", "p")
            )

            assignments[normalized] = gid

    return assignments


def fuzzy_group_lookup(name: str, assignments: dict):

    clean = (
        name.lower()
        .replace("-", "_")
        .replace(".", "p")
    )

    for k, gid in assignments.items():

        kk = (
            str(k).lower()
            .replace("-", "_")
            .replace(".", "p")
        )

        if clean == kk:
            return gid

        if clean in kk or kk in clean:
            return gid

    return None


def run_47_generated_affine_exports():
    log("\n" + "=" * 80)
    log("4.7 Exporting generated / affine SHAP importance CSVs")
    log("=" * 80)

    group_vectors = load_existing_group_vectors()
    if not group_vectors:
        warnings.warn(
            "No function-level group vectors found. "
            "Generated/affine CSVs will still be exported, but similarity to assigned group may not be computed."
        )

    assignments = parse_assignment_files()
    log(f"Loaded assignments: {len(assignments)}")

    pkls = collect_modde_pkls()
    log(f"Found modDE pkl files: {len(pkls)}")

    rows = []

    for kind, pkl_path in pkls:
        name = clean_function_name_from_path(pkl_path)
        out_base = REAL_SHAP_OUT if kind == "real" else AFFINE_SHAP_OUT
        out_base.mkdir(parents=True, exist_ok=True)

        out_csv = out_base / f"{name}_individual_shap_importance.csv"

        if out_csv.exists():
            vec = vector_from_importance_csv(out_csv)
        else:
            df = safe_read_pickle(pkl_path)
            if df is None or df.empty:
                continue

            log(f"Training {kind} function {name}, rows={len(df)}")
            try:
                imp = shap_importance_from_df(df, out_csv, title=name)
                vec = dict(zip(imp["feature"], imp["importance_norm"]))
            except Exception as e:
                warnings.warn(f"Failed {name}: {e}")
                continue

        gid = fuzzy_group_lookup(name, assignments)
        sim = np.nan
        agreement = "NA"

        if gid is not None and gid in group_vectors and vec:
            sim = cosine_dict(group_vectors[gid], vec)
            if sim >= 0.85:
                agreement = "high"
            elif sim >= 0.70:
                agreement = "moderate"
            else:
                agreement = "low"

        rows.append(
            {
                "function": name,
                "kind": kind,
                "assigned_group": gid,
                "cosine_similarity": sim,
                "agreement": agreement,
                "importance_csv": str(out_csv),
            }
        )

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(OUT_47 / "generated_affine_similarity.csv", index=False)

    plot_df = sim_df.dropna(subset=["cosine_similarity"]).copy()
    if not plot_df.empty:
        labels = [f"{n}\nG{int(g)}" for n, g in zip(plot_df["function"], plot_df["assigned_group"])]

        plt.figure(figsize=(max(8, len(plot_df) * 1.0), 4.5))
        plt.bar(labels, plot_df["cosine_similarity"])
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine similarity")
        plt.title("Similarity between unseen functions and assigned landscape groups")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_47 / "generated_affine_similarity_barplot.png", dpi=300)
        plt.close()

    log(f"4.7 outputs saved to: {OUT_47}")
    return sim_df


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 80)
    log("STEP 8B: EXPORT MISSING SHAP IMPORTANCE CSVs")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Target:       {TARGET_COL}")
    log(f"Sample size:  {SAMPLE_SIZE}")
    log(f"Iterations:   {MODEL_ITERATIONS}")
    log("=" * 80)

    try:
        run_46_partial_function_exports()
    except Exception as e:
        warnings.warn(f"4.6 failed: {e}")

    try:
        run_47_generated_affine_exports()
    except Exception as e:
        warnings.warn(f"4.7 failed: {e}")

    readme = OUTPUT_DIR / "README_step8b_outputs.txt"
    readme.write_text(
        "Step 8B outputs\n"
        "===============\n\n"
        "4.6 Partial-function outputs:\n"
        f"- {INSTANCE_IMPORTANCE_OUT / 'instance_group_importance'}\n"
        f"- {INSTANCE_IMPORTANCE_OUT / 'full_function_importance'}\n"
        f"- {INSTANCE_IMPORTANCE_OUT / 'partial_function_importance'}\n"
        f"- {OUT_46 / 'partial_similarity_improvement.csv'}\n"
        f"- {OUT_46 / 'partial_similarity_summary_by_group.csv'}\n\n"
        "4.7 Generated / affine outputs:\n"
        f"- {REAL_SHAP_OUT}\n"
        f"- {AFFINE_SHAP_OUT}\n"
        f"- {OUT_47 / 'generated_affine_similarity.csv'}\n\n"
        "After this script, rerun:\n"
        "python scripts/step8_quantitative_similarity_analysis.py\n",
        encoding="utf-8",
    )

    log("\nDone.")
    log(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()