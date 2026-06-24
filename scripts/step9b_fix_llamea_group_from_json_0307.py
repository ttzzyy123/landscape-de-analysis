from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOF_PATH = PROJECT_ROOT / "output" / "0307_main_comparison" / "hall_of_fame_0307.csv"
FEATURE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task" / "real_function_features"

OUT_CSV = PROJECT_ROOT / "output" / "0307_main_comparison" / "hall_of_fame_0307_fixed.csv"
OUT_MD = PROJECT_ROOT / "output" / "0307_main_comparison" / "hall_of_fame_0307_fixed.md"

def clean_id(x):
    x = str(x)
    x = x.replace("real_generated_function_", "")
    x = x.replace("_ela_features_summary", "")
    x = x.replace(".json", "")
    x = x.replace(".py", "")
    return x

def assign_0307_group(eps_ratio, adj_r2):
    if pd.isna(eps_ratio) or pd.isna(adj_r2):
        return np.nan, np.nan

    eps_ratio = float(eps_ratio)
    adj_r2 = float(adj_r2)

    eps_bin = 0 if eps_ratio < 3.0 else 1

    if adj_r2 < 0.3:
        r2_bin = 0
    elif adj_r2 < 0.7:
        r2_bin = 1
    else:
        r2_bin = 2

    group_id = eps_bin * 3 + r2_bin
    return f"G{group_id}", f"eps_bin_{eps_bin}__r2_bin_{r2_bin}"

def recursive_find_key(obj, candidates):
    """
    Search JSON recursively for one of candidate keys.
    Returns first numeric value found.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in candidates:
                if isinstance(v, (int, float)):
                    return v
                if isinstance(v, dict):
                    for inner_key in ["mean", "value", "avg"]:
                        if inner_key in v and isinstance(v[inner_key], (int, float)):
                            return v[inner_key]
            found = recursive_find_key(v, candidates)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, candidates)
            if found is not None:
                return found

    return None

def load_feature_jsons():
    rows = []

    for p in sorted(FEATURE_DIR.glob("real_generated_function_*_ela_features_summary.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_id = clean_id(p.name)

        eps_ratio = recursive_find_key(
            data,
            {
                "ic.eps.ratio",
                "ic.eps.ratio_mean",
                "eps_ratio",
                "eps_ratio_mean",
            },
        )

        adj_r2 = recursive_find_key(
            data,
            {
                "ela_meta.lin_simple.adj_r2",
                "ela_meta.lin_simple.adj_r2_mean",
                "adj_r2",
                "adj_r2_mean",
            },
        )

        eps_std = recursive_find_key(
            data,
            {
                "ic.eps.ratio_std",
                "eps_ratio_std",
            },
        )

        r2_std = recursive_find_key(
            data,
            {
                "ela_meta.lin_simple.adj_r2_std",
                "adj_r2_std",
            },
        )

        rows.append(
            {
                "merge_id": full_id,
                "feature_file": str(p.relative_to(PROJECT_ROOT)),
                "eps_ratio_json": eps_ratio,
                "adj_r2_json": adj_r2,
                "eps_ratio_std_json": eps_std,
                "adj_r2_std_json": r2_std,
            }
        )

    return pd.DataFrame(rows)

hof = pd.read_csv(HOF_PATH)
features = load_feature_jsons()

print("Loaded feature JSON rows:", len(features))
print(features[["merge_id", "eps_ratio_json", "adj_r2_json"]].to_string(index=False))

hof["merge_id"] = hof["function_id"].map(clean_id)

hof = hof.merge(features, on="merge_id", how="left")

mask = hof["function_type"].eq("llamea_generated") & ~hof["function_id"].eq("All")

hof.loc[mask, "eps_ratio"] = hof.loc[mask, "eps_ratio_json"]
hof.loc[mask, "adj_r2"] = hof.loc[mask, "adj_r2_json"]
hof.loc[mask, "eps_ratio_std"] = hof.loc[mask, "eps_ratio_std_json"]
hof.loc[mask, "adj_r2_std"] = hof.loc[mask, "adj_r2_std_json"]
hof.loc[mask, "source_file"] = hof.loc[mask, "source_file"].astype(str)

for idx in hof.index[mask]:
    group, label = assign_0307_group(hof.loc[idx, "eps_ratio"], hof.loc[idx, "adj_r2"])
    hof.loc[idx, "assigned_group_0307"] = group
    hof.loc[idx, "group_label_0307"] = label

# Keep All row unchanged. It is aggregate, not landscape-assigned.
drop_cols = [
    "merge_id",
    "feature_file",
    "eps_ratio_json",
    "adj_r2_json",
    "eps_ratio_std_json",
    "adj_r2_std_json",
]
hof = hof.drop(columns=[c for c in drop_cols if c in hof.columns])

hof.to_csv(OUT_CSV, index=False)

simple_cols = [
    "function_type",
    "function_id",
    "assigned_group_0307",
    "eps_ratio",
    "adj_r2",
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

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("# Fixed Hall of Fame under 0307 main scheme\n\n")
    f.write("This file fixes LLaMEA generated function group assignments by reading per-function ELA feature JSON files.\n\n")
    for function_type, sub in hof.groupby("function_type", sort=False):
        f.write(f"## {function_type}\n\n")
        available_cols = [c for c in simple_cols if c in sub.columns]
        f.write(df_to_markdown_no_tabulate(sub[available_cols]))
        f.write("\n\n")
        
        
print("\nSaved:")
print(OUT_CSV)
print(OUT_MD)

print("\nFixed LLaMEA rows:")
print(
    hof[hof["function_type"].eq("llamea_generated")]
    [
        [
            "function_type",
            "function_id",
            "assigned_group_0307",
            "group_label_0307",
            "eps_ratio",
            "adj_r2",
            "actual_auc_mean",
            "n_rows",
        ]
    ].to_string(index=False)
)