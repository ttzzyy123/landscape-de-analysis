from pathlib import Path
import re
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Step 8: Quantitative similarity analysis
#
# Goals:
# 4.5 group-level vs function-level SHAP consistency
# 4.6 full-function vs partial-instance SHAP similarity
# 4.7 unseen/generated/affine function vs assigned-group similarity
# ============================================================


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "step8_quantitative_similarity_analysis"

OUT_45 = OUTPUT_DIR / "04_5_group_function_consistency"
OUT_46 = OUTPUT_DIR / "04_6_partial_function_effects"
OUT_47 = OUTPUT_DIR / "04_7_unseen_generalisation"

for d in [OUTPUT_DIR, OUT_45, OUT_46, OUT_47]:
    d.mkdir(parents=True, exist_ok=True)


# Existing expected paths
GROUP_MAPPING_FILE = PROJECT_ROOT / "intermediate" / "group_to_functions_manual_bins.txt"

GROUP_SHAP_DIRS = [
    PROJECT_ROOT / "output" / "shap_eps2bins_3_r0307_main_tiebreak_niles",
    PROJECT_ROOT / "output" / "manual_bins_shap",
    PROJECT_ROOT / "output" / "manual_bins_shap_beeswarm_niles",
]

FUNCTION_SHAP_DIRS = [
    PROJECT_ROOT / "output" / "shap_individual_manual_bins_niles",
    PROJECT_ROOT / "output" / "individual_shap",
    PROJECT_ROOT / "output" / "manual_bins_shap_beeswarm_niles",
]

PARTIAL_SHAP_DIRS = [
    PROJECT_ROOT / "output" / "step7_instance_level_group_and_partial_function_plots",
    PROJECT_ROOT / "output" / "instance_level_shap",
]

REAL_GENERATED_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_real_generated_function"
)

AFFINE_SHAP_DIR = (
    PROJECT_ROOT
    / "output"
    / "new_function_task"
    / "task_H_individual_shap_affine_functions"
)

REAL_ASSIGNMENT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "real_function_group_assignment"
)

AFFINE_ASSIGNMENT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "affine_function_group_assignment"
)


# =========================
# Utilities
# =========================
def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_importance_column(df: pd.DataFrame):
    candidates = [
        "importance_norm",
        "importance_normalized",
        "norm_importance",
        "mean_abs_shap_norm",
        "mean_abs_shap",
        "importance",
        "shap_importance",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"fid", "iid", "group_id", "Group", "group"}]
    if numeric_cols:
        return numeric_cols[0]

    return None


def find_feature_column(df: pd.DataFrame):
    candidates = ["feature", "Feature", "parameter", "param", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_vector_dict(v: dict):
    clean = {}
    for k, val in v.items():
        if pd.isna(val):
            continue
        try:
            clean[str(k)] = float(val)
        except Exception:
            continue

    total = sum(abs(x) for x in clean.values())
    if total <= 0:
        return clean
    return {k: abs(v) / total for k, v in clean.items()}


def cosine_similarity_dict(a: dict, b: dict):
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return np.nan

    x = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    y = np.array([b.get(k, 0.0) for k in keys], dtype=float)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx == 0 or ny == 0:
        return np.nan

    return float(np.dot(x, y) / (nx * ny))


def extract_first_int(text: str):
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


def extract_fid_from_text(text: str):
    text = str(text)
    patterns = [
        r"(?:^|[_\-])f(\d+)(?:[_\-.]|$)",
        r"function[_\-]?(\d+)",
        r"fid[_\-]?(\d+)",
        r"shap_function_(\d+)",
        r"summary_f(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_group_id_from_text(text: str):
    text = str(text)
    patterns = [
        r"group[_\- ]?(\d+)",
        r"G(\d+)",
        r"shap_group_(\d+)",
        r"summary_group_(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def load_importance_vector_from_csv(path: Path):
    df = safe_read_csv(path)
    if df is None or df.empty:
        return None

    feature_col = find_feature_column(df)
    imp_col = find_importance_column(df)

    if feature_col is None or imp_col is None:
        return None

    tmp = df[[feature_col, imp_col]].dropna()
    if tmp.empty:
        return None

    v = dict(zip(tmp[feature_col].astype(str), tmp[imp_col].astype(float)))
    return normalize_vector_dict(v)


def parse_group_mapping(path: Path):
    """
    Supports formats like:
    Group 0 (...): [7, 16, 19]
    G0: f7, f16, f19
    """
    group_map = {}

    if not path.exists():
        warnings.warn(f"Group mapping file not found: {path}")
        return group_map

    text = path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        gid = extract_group_id_from_text(line)
        if gid is None:
            continue

        # Prefer fXX format if present
        fids = [int(x) for x in re.findall(r"\bf(\d+)\b", line, flags=re.IGNORECASE)]

        # Otherwise parse list after colon
        if not fids:
            right = line.split(":", 1)[-1]
            fids = [int(x) for x in re.findall(r"\d+", right)]

        if fids:
            group_map[gid] = sorted(set(fids))

    return group_map


def collect_csvs(paths):
    csvs = []
    for p in paths:
        if p.exists():
            csvs.extend(sorted(p.rglob("*.csv")))
    return csvs


# =========================
# Load group SHAP vectors
# =========================
def load_group_vectors():
    vectors = {}

    # 1. Prefer all-groups files
    all_group_patterns = [
        "shap_all_groups.csv",
        "*all*group*.csv",
        "*group*importance*.csv",
    ]

    for base in GROUP_SHAP_DIRS:
        if not base.exists():
            continue

        for pat in all_group_patterns:
            for path in base.rglob(pat):
                df = safe_read_csv(path)
                if df is None or df.empty:
                    continue

                feature_col = find_feature_column(df)
                imp_col = find_importance_column(df)
                group_col = None
                for c in ["group_id", "group", "Group", "group_label"]:
                    if c in df.columns:
                        group_col = c
                        break

                if feature_col and imp_col and group_col:
                    for gid_raw, sub in df.groupby(group_col):
                        gid = extract_group_id_from_text(str(gid_raw))
                        if gid is None:
                            try:
                                gid = int(gid_raw)
                            except Exception:
                                continue
                        v = dict(zip(sub[feature_col].astype(str), sub[imp_col].astype(float)))
                        vectors[gid] = normalize_vector_dict(v)

    # 2. Individual group files
    group_file_patterns = [
        "shap_group_*.csv",
        "group_*importance*.csv",
        "group_*.csv",
    ]

    for base in GROUP_SHAP_DIRS:
        if not base.exists():
            continue

        for pat in group_file_patterns:
            for path in base.rglob(pat):
                gid = extract_group_id_from_text(path.name)
                if gid is None:
                    continue

                v = load_importance_vector_from_csv(path)
                if v:
                    vectors[gid] = v

    return vectors


# =========================
# Load function SHAP vectors
# =========================
def load_function_vectors():
    vectors = {}

    # 1. Prefer all-functions files
    all_function_patterns = [
        "shap_all_functions.csv",
        "*all*function*.csv",
        "*function*importance*.csv",
    ]

    for base in FUNCTION_SHAP_DIRS:
        if not base.exists():
            continue

        for pat in all_function_patterns:
            for path in base.rglob(pat):
                df = safe_read_csv(path)
                if df is None or df.empty:
                    continue

                feature_col = find_feature_column(df)
                imp_col = find_importance_column(df)
                fid_col = None
                for c in ["fid", "function", "Function", "function_id"]:
                    if c in df.columns:
                        fid_col = c
                        break

                if feature_col and imp_col and fid_col:
                    for fid_raw, sub in df.groupby(fid_col):
                        fid = extract_fid_from_text(str(fid_raw))
                        if fid is None:
                            try:
                                fid = int(fid_raw)
                            except Exception:
                                continue
                        v = dict(zip(sub[feature_col].astype(str), sub[imp_col].astype(float)))
                        vectors[fid] = normalize_vector_dict(v)

    # 2. Individual function files
    function_file_patterns = [
        "shap_function_*.csv",
        "function_*importance*.csv",
        "f*_importance*.csv",
        "summary_f*.csv",
    ]

    for base in FUNCTION_SHAP_DIRS:
        if not base.exists():
            continue

        for pat in function_file_patterns:
            for path in base.rglob(pat):
                fid = extract_fid_from_text(path.name)
                if fid is None:
                    continue

                v = load_importance_vector_from_csv(path)
                if v:
                    vectors[fid] = v

    return vectors


# =========================
# 4.5 Group vs Function
# =========================
def run_45_group_function_consistency():
    print("\n" + "=" * 80)
    print("4.5 Group-level vs function-level consistency")
    print("=" * 80)

    group_map = parse_group_mapping(GROUP_MAPPING_FILE)
    group_vectors = load_group_vectors()
    function_vectors = load_function_vectors()

    print(f"Loaded group mapping: {len(group_map)} groups")
    print(f"Loaded group vectors: {len(group_vectors)} groups")
    print(f"Loaded function vectors: {len(function_vectors)} functions")

    rows = []

    for gid, fids in sorted(group_map.items()):
        gv = group_vectors.get(gid)
        if gv is None:
            continue

        for fid in fids:
            fv = function_vectors.get(fid)
            if fv is None:
                continue

            sim = cosine_similarity_dict(gv, fv)
            rows.append(
                {
                    "group_id": gid,
                    "fid": fid,
                    "cosine_similarity": sim,
                    "n_group_features": len(gv),
                    "n_function_features": len(fv),
                }
            )

    sim_df = pd.DataFrame(rows)
    sim_path = OUT_45 / "group_function_cosine_similarity.csv"
    sim_df.to_csv(sim_path, index=False)

    if sim_df.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            sim_df.groupby("group_id")["cosine_similarity"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": "n_functions",
                    "mean": "mean_similarity",
                    "std": "std_similarity",
                    "min": "min_similarity",
                    "max": "max_similarity",
                }
            )
        )

        def interp(x):
            if pd.isna(x):
                return "NA"
            if x >= 0.85:
                return "strong"
            if x >= 0.70:
                return "moderate"
            return "weak"

        summary["interpretation"] = summary["mean_similarity"].apply(interp)

    summary_path = OUT_45 / "group_similarity_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot
    if not summary.empty:
        plt.figure(figsize=(8, 4.5))
        x = np.arange(len(summary))
        y = summary["mean_similarity"].values
        err = summary["std_similarity"].fillna(0).values

        plt.bar(x, y, yerr=err, capsize=4)
        plt.xticks(x, [f"G{g}" for g in summary["group_id"]])
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine similarity")
        plt.xlabel("Landscape group")
        plt.title("Group-level vs function-level SHAP similarity")
        plt.tight_layout()
        plt.savefig(OUT_45 / "group_similarity_barplot.png", dpi=300)
        plt.close()

    return {
        "group_vectors": group_vectors,
        "function_vectors": function_vectors,
        "group_map": group_map,
        "sim_df": sim_df,
        "summary": summary,
    }


# =========================
# Load partial vectors if CSV exists
# =========================
def load_partial_vectors():
    """
    Tries to load partial-function / instance-level SHAP importance CSV files.

    Expected filename examples:
    group_0_f24_partial_importance.csv
    partial_group_0_function_24.csv
    f24_group0_partial_shap_importance.csv
    """
    vectors = {}

    csvs = collect_csvs(PARTIAL_SHAP_DIRS)
    csvs = [p for p in csvs if "partial" in p.name.lower() or "instance" in p.name.lower()]

    for path in csvs:
        gid = extract_group_id_from_text(path.name)
        fid = extract_fid_from_text(path.name)

        # also try parent folder names
        if gid is None:
            gid = extract_group_id_from_text(str(path.parent))
        if fid is None:
            fid = extract_fid_from_text(str(path.parent))

        if gid is None or fid is None:
            continue

        v = load_importance_vector_from_csv(path)
        if v:
            vectors[(gid, fid)] = v

    return vectors


# =========================
# 4.6 Partial-function effects
# =========================
def run_46_partial_effects(group_vectors, function_vectors):
    print("\n" + "=" * 80)
    print("4.6 Partial-function SHAP effects")
    print("=" * 80)

    partial_vectors = load_partial_vectors()
    print(f"Loaded partial vectors: {len(partial_vectors)}")

    rows = []

    for (gid, fid), pv in sorted(partial_vectors.items()):
        gv = group_vectors.get(gid)
        fv = function_vectors.get(fid)

        if gv is None or fv is None:
            continue

        full_sim = cosine_similarity_dict(gv, fv)
        partial_sim = cosine_similarity_dict(gv, pv)
        rows.append(
            {
                "group_id": gid,
                "fid": fid,
                "full_function_similarity": full_sim,
                "partial_function_similarity": partial_sim,
                "improvement": partial_sim - full_sim,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_46 / "partial_similarity_improvement.csv", index=False)

    if not df.empty:
        plt.figure(figsize=(8, 4.5))
        labels = [f"G{g}-f{f}" for g, f in zip(df["group_id"], df["fid"])]
        x = np.arange(len(df))
        width = 0.35

        plt.bar(x - width / 2, df["full_function_similarity"], width, label="Full function")
        plt.bar(x + width / 2, df["partial_function_similarity"], width, label="Partial function")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine similarity")
        plt.title("Full vs partial-function SHAP similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_46 / "full_vs_partial_similarity_barplot.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, df["improvement"])
        plt.axhline(0, linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Partial - full similarity")
        plt.title("Similarity improvement from partial-function grouping")
        plt.tight_layout()
        plt.savefig(OUT_46 / "partial_similarity_improvement_barplot.png", dpi=300)
        plt.close()
    else:
        readme = OUT_46 / "README_missing_partial_csv.txt"
        readme.write_text(
            "No partial-function importance CSV files were found.\n"
            "The partial-function plots may exist, but cosine similarity requires "
            "the underlying SHAP importance vectors.\n"
            "Expected files should contain columns like: feature, importance or importance_norm.\n",
            encoding="utf-8",
        )

    return df


# =========================
# Generated / affine helpers
# =========================
def load_generated_vectors():
    vectors = {}

    dirs = [REAL_GENERATED_SHAP_DIR, AFFINE_SHAP_DIR]

    patterns = [
        "*_individual_shap_importance.csv",
        "*importance*.csv",
        "*.csv",
    ]

    for base in dirs:
        if not base.exists():
            continue

        for pat in patterns:
            for path in base.rglob(pat):
                v = load_importance_vector_from_csv(path)
                if not v:
                    continue

                name = path.stem
                name = re.sub(r"_individual_shap_importance$", "", name)
                name = re.sub(r"_importance$", "", name)
                vectors[name] = v

    return vectors


def collect_assignment_files():
    files = []
    for base in [REAL_ASSIGNMENT_DIR, AFFINE_ASSIGNMENT_DIR]:
        if base.exists():
            files.extend(sorted(base.rglob("*.csv")))
            files.extend(sorted(base.rglob("*.json")))
    return files


def parse_assignments():
    """
    Returns dict:
    function_name -> group_id
    """
    assignments = {}

    for path in collect_assignment_files():
        if path.suffix.lower() == ".csv":
            df = safe_read_csv(path)
            if df is None or df.empty:
                continue

            name_col = None
            for c in ["function", "function_name", "name", "tag", "id"]:
                if c in df.columns:
                    name_col = c
                    break

            group_col = None
            for c in df.columns:
                cl = c.lower()
                if "group" in cl and ("id" in cl or "assigned" in cl or cl == "group"):
                    group_col = c
                    break

            if name_col and group_col:
                for _, row in df.iterrows():
                    name = str(row[name_col])
                    gid = extract_group_id_from_text(str(row[group_col]))
                    if gid is None:
                        try:
                            gid = int(row[group_col])
                        except Exception:
                            continue
                    assignments[name] = gid

        elif path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue

            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # direct mapping
                for k, v in data.items():
                    if isinstance(v, dict):
                        vv = v.copy()
                        vv["function"] = k
                        items.append(vv)
                    else:
                        gid = extract_group_id_from_text(str(v))
                        if gid is not None:
                            assignments[str(k)] = gid

            for item in items:
                if not isinstance(item, dict):
                    continue

                name = None
                for c in ["function", "function_name", "name", "tag", "id"]:
                    if c in item:
                        name = str(item[c])
                        break

                gid = None
                for c, v in item.items():
                    cl = str(c).lower()
                    if "group" in cl:
                        gid = extract_group_id_from_text(str(v))
                        if gid is None:
                            try:
                                gid = int(v)
                            except Exception:
                                pass
                        if gid is not None:
                            break

                if name and gid is not None:
                    assignments[name] = gid

    return assignments


def fuzzy_match_assignment(name, assignments):
    if name in assignments:
        return assignments[name]

    clean = name.lower().replace("-", "_")
    for k, gid in assignments.items():
        kk = str(k).lower().replace("-", "_")
        if clean in kk or kk in clean:
            return gid

    return None


# =========================
# 4.7 Unseen/generalisation
# =========================
def run_47_unseen_generalisation(group_vectors):
    print("\n" + "=" * 80)
    print("4.7 Unseen/generated/affine function generalisation")
    print("=" * 80)

    gen_vectors = load_generated_vectors()
    assignments = parse_assignments()

    print(f"Loaded generated/affine vectors: {len(gen_vectors)}")
    print(f"Loaded assignments: {len(assignments)}")

    rows = []

    for name, v in sorted(gen_vectors.items()):
        gid = fuzzy_match_assignment(name, assignments)
        if gid is None:
            continue

        gv = group_vectors.get(gid)
        if gv is None:
            continue

        sim = cosine_similarity_dict(gv, v)
        rows.append(
            {
                "function": name,
                "assigned_group": gid,
                "cosine_similarity": sim,
                "agreement": (
                    "high"
                    if sim >= 0.85
                    else "moderate"
                    if sim >= 0.70
                    else "low"
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_47 / "generated_function_similarity.csv", index=False)

    if not df.empty:
        plt.figure(figsize=(9, 4.5))
        labels = [f"{n}\nG{g}" for n, g in zip(df["function"], df["assigned_group"])]
        plt.bar(labels, df["cosine_similarity"])
        plt.ylim(0, 1.05)
        plt.ylabel("Cosine similarity")
        plt.xlabel("Generated / affine function")
        plt.title("Similarity between unseen functions and assigned landscape groups")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_47 / "generated_function_similarity_barplot.png", dpi=300)
        plt.close()
    else:
        readme = OUT_47 / "README_missing_generated_similarity_inputs.txt"
        readme.write_text(
            "No generated/affine similarity rows were produced.\n"
            "Possible reasons:\n"
            "1. No *_individual_shap_importance.csv files were found.\n"
            "2. No group assignment files were found.\n"
            "3. Function names in SHAP outputs and assignment files did not match.\n",
            encoding="utf-8",
        )

    return df


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("STEP 8: QUANTITATIVE SIMILARITY ANALYSIS")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print("=" * 80)

    res45 = run_45_group_function_consistency()

    run_46_partial_effects(
        group_vectors=res45["group_vectors"],
        function_vectors=res45["function_vectors"],
    )

    run_47_unseen_generalisation(
        group_vectors=res45["group_vectors"],
    )

    # Write global README
    readme = OUTPUT_DIR / "README_step8_outputs.txt"
    readme.write_text(
        "Step 8 quantitative similarity analysis outputs\n"
        "================================================\n\n"
        "4.5 Group-level vs function-level consistency:\n"
        f"- {OUT_45 / 'group_function_cosine_similarity.csv'}\n"
        f"- {OUT_45 / 'group_similarity_summary.csv'}\n"
        f"- {OUT_45 / 'group_similarity_barplot.png'}\n\n"
        "4.6 Partial-function effects:\n"
        f"- {OUT_46 / 'partial_similarity_improvement.csv'}\n"
        f"- {OUT_46 / 'full_vs_partial_similarity_barplot.png'}\n"
        f"- {OUT_46 / 'partial_similarity_improvement_barplot.png'}\n\n"
        "4.7 Unseen/generalisation:\n"
        f"- {OUT_47 / 'generated_function_similarity.csv'}\n"
        f"- {OUT_47 / 'generated_function_similarity_barplot.png'}\n\n"
        "If some files are empty or README_missing_*.txt appears, the required SHAP importance CSVs were not found.\n",
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()