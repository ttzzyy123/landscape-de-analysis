from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Step 10C: Extract Hall-of-Fame tables for 0307 main scheme
#
# Generates:
#   Table 4: Full Hall-of-Fame configurations
#   Table 5: Generated-functions Hall-of-Fame configurations
#
# Source:
#   output/0307_main_comparison/hall_of_fame_0307.csv
#
# Scheme:
#   eps2bins_3_r0307_main_tiebreak
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISON_DIR = PROJECT_ROOT / "output" / "0307_main_comparison"

HOF_CSV = COMPARISON_DIR / "hall_of_fame_0307.csv"

OUT_TABLE4_CSV = COMPARISON_DIR / "table04_full_hall_of_fame_configurations_0307.csv"
OUT_TABLE4_MD = COMPARISON_DIR / "table04_full_hall_of_fame_configurations_0307.md"

OUT_TABLE5_CSV = COMPARISON_DIR / "table05_generated_hall_of_fame_configurations_0307.csv"
OUT_TABLE5_MD = COMPARISON_DIR / "table05_generated_hall_of_fame_configurations_0307.md"


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
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))

        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def short_label(row):
    function_type = row.get("function_type", "")
    function_id = str(row.get("function_id", ""))

    if function_type == "affine":
        return function_id.replace("_", " + ")

    if function_type == "llamea_generated":
        return "LLaMEA generated"

    if function_type == "bbob_original":
        return function_id

    return function_id


def main():
    print("=" * 80)
    print("STEP 10C: EXTRACT HALL-OF-FAME TABLES UNDER 0307 MAIN SCHEME")
    print("=" * 80)

    if not HOF_CSV.exists():
        raise FileNotFoundError(f"Missing Hall-of-Fame CSV: {HOF_CSV}")

    df = pd.read_csv(HOF_CSV)

    # Remove aggregate All rows for per-function tables.
    df = df[~df["function_id"].astype(str).eq("All")].copy()

    wanted_cols = [
        "function_type",
        "function_id",
        "label",
        "assigned_group_0307",
        "actual_auc_mean",
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

    df["label"] = df.apply(short_label, axis=1)

    # Keep only columns that actually exist.
    existing_cols = [c for c in wanted_cols if c in df.columns]

    table4 = df[existing_cols].copy()

    # Sort for readability.
    type_order = {
        "bbob_original": 0,
        "affine": 1,
        "llamea_generated": 2,
    }

    table4["_type_order"] = table4["function_type"].map(type_order).fillna(99)
    table4 = table4.sort_values(["_type_order", "function_type", "function_id"])
    table4 = table4.drop(columns=["_type_order"])

    # Table 5: generated functions only.
    table5 = table4[
        table4["function_type"].isin(["affine", "llamea_generated"])
    ].copy()

    # Save CSV.
    table4.to_csv(OUT_TABLE4_CSV, index=False)
    table5.to_csv(OUT_TABLE5_CSV, index=False)

    # Markdown output.
    with open(OUT_TABLE4_MD, "w", encoding="utf-8") as f:
        f.write("# Table 4. Hall-of-Fame configurations under the 0307 grouping scheme\n\n")
        f.write(
            "This table reports the best observed MODDE configuration for each function "
            "under the 0307 main tiebreak grouping scheme. The full version is intended "
            "for the appendix.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table4))
        f.write("\n")

    with open(OUT_TABLE5_MD, "w", encoding="utf-8") as f:
        f.write("# Table 5. Best observed MODDE configurations for generated functions\n\n")
        f.write(
            "This table is a reduced Hall-of-Fame table containing only affine and "
            "LLaMEA-generated functions. It is suitable for the main text.\n\n"
        )
        f.write(df_to_markdown_no_tabulate(table5))
        f.write("\n")

    print("\nDone.")
    print(f"Saved Table 4 CSV: {OUT_TABLE4_CSV}")
    print(f"Saved Table 4 MD:  {OUT_TABLE4_MD}")
    print(f"Saved Table 5 CSV: {OUT_TABLE5_CSV}")
    print(f"Saved Table 5 MD:  {OUT_TABLE5_MD}")

    print("\nTable 5 preview:")
    print(table5.to_string(index=False))


if __name__ == "__main__":
    main()