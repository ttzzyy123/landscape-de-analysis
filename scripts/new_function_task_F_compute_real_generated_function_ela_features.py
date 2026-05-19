# scripts/new_function_task_F_compute_real_generated_function_ela_features.py

from pathlib import Path
import json
import traceback
import importlib.util

import numpy as np
import pandas as pd

from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import (
    calculate_ela_meta,
    calculate_information_content,
)


# ============================================================
# New Function Task F
# Compute ELA features for real LLM-generated functions
#
# Updated batch version:
# 1. Load multiple real_generated_function_*.py files from Task E
# 2. Support n1/n2/n3 generated functions
# 3. Skip failed/missing functions automatically
# 4. Instantiate each generated function with dim=5
# 5. Sample the domain [-5, 5]^5 multiple times
# 6. Compute the two ELA features used in the thesis:
#      - ic.eps.ratio
#      - ela_meta.lin_simple.adj_r2
# 7. Save per-repeat values, per-function summaries, and master summary
#
# Output:
#   intermediate/new_function_task/real_function_features/
#      real_generated_function_<RUN_ID>_<TAG>_ela_features_repeats.csv
#      real_generated_function_<RUN_ID>_<TAG>_ela_features_summary.json
#      real_generated_functions_batch_ela_features_summary.csv
#      real_generated_functions_batch_ela_features_summary.json
#
#   output/new_function_task/
#      task_F_compute_real_generated_function_ela_features_summary_<RUN_ID>_<TAG>.txt
#      task_F_compute_real_generated_function_ela_features_master_summary.txt
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task"
REAL_FUNCTION_DIR = INTERMEDIATE_DIR / "real_generated_functions"
REAL_METADATA_DIR = INTERMEDIATE_DIR / "real_function_metadata"
FEATURE_DIR = INTERMEDIATE_DIR / "real_function_features"
OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIM = 5
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0
N_REPEATS = 5
SAMPLE_SIZE_MULTIPLIER = 250
SAMPLE_SIZE = SAMPLE_SIZE_MULTIPLIER * DIM
BASE_SEED = 20260517

TARGET_FEATURES = [
    "ic.eps.ratio",
    "ela_meta.lin_simple.adj_r2",
]

# Set this if you want to compute one specific batch only.
# Example:
#   TARGET_RUN_ID = "20260517_114949"
#
# If None, the script automatically processes all valid *_n*.py files.
TARGET_RUN_ID = "20260519_225148"

# Only process n-style generated functions by default.
PROCESS_N_TAGGED_FUNCTIONS_ONLY = True


# ============================================================
# Utilities
# ============================================================


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def parse_function_identity(function_file):
    """
    Example:
      real_generated_function_20260517_114949_n1.py

    Returns:
      run_id = 20260517_114949
      function_tag = n1
      full_id = 20260517_114949_n1
    """

    prefix = "real_generated_function_"
    stem = function_file.stem

    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected function filename: {function_file.name}")

    full_id = stem.replace(prefix, "")

    parts = full_id.split("_")

    if len(parts) >= 3 and parts[-1].startswith("n"):
        function_tag = parts[-1]
        run_id = "_".join(parts[:-1])
    else:
        function_tag = "single"
        run_id = full_id

    return run_id, function_tag, full_id


def expected_metadata_file(function_file):
    return REAL_METADATA_DIR / f"{function_file.stem}_metadata.json"


def find_target_function_files(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. FIND TARGET REAL GENERATED FUNCTIONS")
    write_line(lines, "=" * 80)

    if TARGET_RUN_ID:
        if PROCESS_N_TAGGED_FUNCTIONS_ONLY:
            pattern = f"real_generated_function_{TARGET_RUN_ID}_n*.py"
        else:
            pattern = f"real_generated_function_{TARGET_RUN_ID}*.py"
    else:
        if PROCESS_N_TAGGED_FUNCTIONS_ONLY:
            pattern = "real_generated_function_*_n*.py"
        else:
            pattern = "real_generated_function_*.py"

    files = sorted(REAL_FUNCTION_DIR.glob(pattern))

    write_line(lines, f"Search pattern: {pattern}")
    write_line(lines, f"Found {len(files)} candidate function file(s).")

    valid_files = []

    for function_file in files:
        metadata_file = expected_metadata_file(function_file)

        if not function_file.exists():
            write_line(lines, f"[SKIP] Function file missing: {function_file}")
            continue

        if not metadata_file.exists():
            write_line(lines, f"[SKIP] Metadata file missing for: {function_file.name}")
            continue

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            write_line(lines, f"[SKIP] Could not read metadata: {metadata_file}")
            continue

        error_value = str(metadata.get("error", "") or "").strip()
        function_tag = metadata.get("function_tag", "")

        if error_value:
            write_line(lines, f"[WARN] Metadata has non-empty error for {function_file.name}: {error_value[:200]}")

        run_id, parsed_tag, full_id = parse_function_identity(function_file)

        write_line(lines, f"[OK] Candidate: {function_file.name}")
        write_line(lines, f"     run_id: {run_id}")
        write_line(lines, f"     tag:    {function_tag or parsed_tag}")
        write_line(lines, f"     id:     {full_id}")

        valid_files.append(function_file)

    if not valid_files:
        raise FileNotFoundError(
            f"No valid generated function files found in {REAL_FUNCTION_DIR} with pattern {pattern}"
        )

    write_line(lines, f"\nUsing {len(valid_files)} valid function file(s):")
    for p in valid_files:
        write_line(lines, f" - {p}")

    return valid_files


def load_generated_module(function_file, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. LOAD GENERATED FUNCTION MODULE")
    write_line(lines, "=" * 80)
    write_line(lines, f"Function file: {function_file}")

    spec = importlib.util.spec_from_file_location(
        function_file.stem,
        function_file,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for: {function_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    write_line(lines, "[OK] Module loaded successfully.")

    return module


def find_generated_class(module, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. FIND GENERATED CLASS")
    write_line(lines, "=" * 80)

    candidates = []

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and not name.startswith("_"):
            candidates.append((name, obj))

    if not candidates:
        raise RuntimeError("No class found in generated function module.")

    write_line(lines, "Detected classes:")
    for name, _ in candidates:
        write_line(lines, f" - {name}")

    # Prefer the expected LLaMEA class name.
    for name, cls in candidates:
        if name == "landscape":
            write_line(lines, f"Selected class: {name}")
            return name, cls

    class_name, cls = candidates[0]
    write_line(lines, f"Selected class: {class_name}")

    return class_name, cls


def instantiate_generated_function(cls, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "4. INSTANTIATE GENERATED FUNCTION")
    write_line(lines, "=" * 80)

    instance = cls(dim=DIM)

    if not hasattr(instance, "f"):
        raise RuntimeError("Generated class does not expose .f")

    write_line(lines, f"[OK] Instantiated with dim={DIM}")
    write_line(lines, "[OK] Method .f exists")

    return instance


def evaluate_function(instance, X):
    y = np.array([float(instance.f(x)) for x in X], dtype=float)

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite objective values detected during sampling.")

    return y


def get_feature_value(feature_dict, possible_keys):
    for key in possible_keys:
        if key in feature_dict:
            value = feature_dict[key]
            if value is not None:
                return float(value)
    return float("nan")


def compute_features_for_repeat(instance, repeat_id, seed, lines):
    write_line(lines, f"\nRepeat {repeat_id}: seed={seed}")

    X = create_initial_sample(
        dim=DIM,
        n=SAMPLE_SIZE,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        sample_type="lhs",
        seed=seed,
    )

    if isinstance(X, pd.DataFrame):
        X_eval = X.to_numpy(dtype=float)
    else:
        X_eval = np.asarray(X, dtype=float)

    y = evaluate_function(instance, X_eval)

    meta_features = calculate_ela_meta(X_eval, y)
    ic_features = calculate_information_content(X_eval, y)

    # Important:
    # pflacco uses "ic.eps.ratio", not "ic.eps_ratio".
    ic_eps_ratio = get_feature_value(
        ic_features,
        [
            "ic.eps.ratio",
            "ic.eps_ratio",
            "eps.ratio",
            "eps_ratio",
        ],
    )

    adj_r2 = get_feature_value(
        meta_features,
        [
            "ela_meta.lin_simple.adj_r2",
            "lin_simple.adj_r2",
        ],
    )

    row = {
        "repeat": repeat_id,
        "seed": seed,
        "dim": DIM,
        "sample_size": SAMPLE_SIZE,
        "lower_bound": LOWER_BOUND,
        "upper_bound": UPPER_BOUND,
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "ic.eps.ratio": ic_eps_ratio,
        "ela_meta.lin_simple.adj_r2": adj_r2,
    }

    write_line(lines, f"  ic.eps.ratio = {row['ic.eps.ratio']}")
    write_line(lines, f"  ela_meta.lin_simple.adj_r2 = {row['ela_meta.lin_simple.adj_r2']}")

    if np.isnan(row["ic.eps.ratio"]):
        write_line(lines, f"  [WARN] ic.eps.ratio is NaN. Available IC keys: {list(ic_features.keys())}")

    if np.isnan(row["ela_meta.lin_simple.adj_r2"]):
        write_line(lines, f"  [WARN] adj_r2 is NaN. Available meta keys: {list(meta_features.keys())}")

    return row


def compute_ela_features(instance, full_id, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. COMPUTE ELA FEATURES")
    write_line(lines, "=" * 80)

    rows = []

    # Use a function-specific offset so n1 and n3 do not use exactly identical samples.
    stable_offset = sum(ord(c) for c in full_id)

    for repeat_id in range(1, N_REPEATS + 1):
        seed = BASE_SEED + stable_offset + repeat_id
        row = compute_features_for_repeat(instance, repeat_id, seed, lines)
        rows.append(row)

    df = pd.DataFrame(rows)

    return df


def save_function_outputs(df, function_file, run_id, function_tag, full_id, class_name, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. SAVE FEATURE OUTPUTS")
    write_line(lines, "=" * 80)

    repeats_file = FEATURE_DIR / f"real_generated_function_{full_id}_ela_features_repeats.csv"
    summary_file = FEATURE_DIR / f"real_generated_function_{full_id}_ela_features_summary.json"
    text_summary_file = OUTPUT_DIR / f"task_F_compute_real_generated_function_ela_features_summary_{full_id}.txt"

    df.to_csv(repeats_file, index=False)

    feature_means = {
        feature: float(df[feature].mean(skipna=True))
        for feature in TARGET_FEATURES
    }

    feature_stds = {
        feature: float(df[feature].std(ddof=1, skipna=True))
        for feature in TARGET_FEATURES
    }

    summary = {
        "task": "new_function_task_F_compute_real_generated_function_ela_features",
        "run_id": run_id,
        "function_tag": function_tag,
        "full_id": full_id,
        "function_file": str(function_file),
        "metadata_file": str(expected_metadata_file(function_file)),
        "class_name": class_name,
        "dim": DIM,
        "n_repeats": N_REPEATS,
        "sample_size": SAMPLE_SIZE,
        "sample_type": "lhs",
        "domain": [LOWER_BOUND, UPPER_BOUND],
        "target_features": TARGET_FEATURES,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "repeats_file": str(repeats_file),
        "summary_file": str(summary_file),
        "text_summary_file": str(text_summary_file),
        "next_step": (
            "Use these mean feature values to assign this generated function "
            "to the existing BBOB landscape group."
        ),
    }

    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_line(lines, f"[OK] Repeats CSV saved to: {repeats_file}")
    write_line(lines, f"[OK] Summary JSON saved to: {summary_file}")

    write_line(lines, "\nFeature means:")
    for k, v in feature_means.items():
        write_line(lines, f"  {k}: {v}")

    write_line(lines, "\nFeature stds:")
    for k, v in feature_stds.items():
        write_line(lines, f"  {k}: {v}")

    return repeats_file, summary_file, text_summary_file, summary


def process_one_function(function_file):
    lines = []
    text_summary_file = None

    run_id, function_tag, full_id = parse_function_identity(function_file)

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK F: COMPUTE REAL GENERATED FUNCTION ELA FEATURES")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root:        {PROJECT_ROOT}")
    write_line(lines, f"Function file:       {function_file}")
    write_line(lines, f"Run ID:              {run_id}")
    write_line(lines, f"Function tag:        {function_tag}")
    write_line(lines, f"Full ID:             {full_id}")
    write_line(lines, f"Feature output dir:  {FEATURE_DIR}")
    write_line(lines, f"Target features:     {TARGET_FEATURES}")
    write_line(lines, f"DIM:                 {DIM}")
    write_line(lines, f"N_REPEATS:           {N_REPEATS}")
    write_line(lines, f"SAMPLE_SIZE:         {SAMPLE_SIZE}")

    try:
        module = load_generated_module(function_file, lines)
        class_name, cls = find_generated_class(module, lines)
        instance = instantiate_generated_function(cls, lines)
        df = compute_ela_features(instance, full_id, lines)

        _, _, text_summary_file, summary = save_function_outputs(
            df=df,
            function_file=function_file,
            run_id=run_id,
            function_tag=function_tag,
            full_id=full_id,
            class_name=class_name,
            lines=lines,
        )

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK F CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, "[SUCCESS] ELA features computed for this real generated function.")
        write_line(lines, "Next step: assign this function to the existing BBOB landscape group.")

        status = {
            "status": "success",
            "run_id": run_id,
            "function_tag": function_tag,
            "full_id": full_id,
            "function_file": str(function_file),
            "feature_means": summary["feature_means"],
            "feature_stds": summary["feature_stds"],
            "repeats_file": summary["repeats_file"],
            "summary_file": summary["summary_file"],
            "text_summary_file": summary["text_summary_file"],
        }

    except Exception as e:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK F FAILED")
        write_line(lines, "=" * 80)
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())

        fallback = OUTPUT_DIR / f"task_F_compute_real_generated_function_ela_features_summary_{full_id}_failed.txt"
        text_summary_file = fallback

        status = {
            "status": "failed",
            "run_id": run_id,
            "function_tag": function_tag,
            "full_id": full_id,
            "function_file": str(function_file),
            "error": str(e),
            "text_summary_file": str(text_summary_file),
        }

    finally:
        if text_summary_file is not None:
            text_summary_file.write_text("\n".join(lines), encoding="utf-8")
            print("\nSummary written to:")
            print(text_summary_file)

    return status


def save_master_outputs(results, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "7. SAVE MASTER BATCH OUTPUTS")
    write_line(lines, "=" * 80)

    master_json_file = FEATURE_DIR / "real_generated_functions_batch_ela_features_summary.json"
    master_csv_file = FEATURE_DIR / "real_generated_functions_batch_ela_features_summary.csv"
    master_text_file = OUTPUT_DIR / "task_F_compute_real_generated_function_ela_features_master_summary.txt"

    master_json_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    rows = []
    for r in results:
        row = {
            "status": r.get("status"),
            "run_id": r.get("run_id"),
            "function_tag": r.get("function_tag"),
            "full_id": r.get("full_id"),
            "function_file": r.get("function_file"),
            "summary_file": r.get("summary_file"),
            "repeats_file": r.get("repeats_file"),
            "error": r.get("error", ""),
        }

        means = r.get("feature_means", {}) or {}
        stds = r.get("feature_stds", {}) or {}

        for feature in TARGET_FEATURES:
            row[f"{feature}_mean"] = means.get(feature, np.nan)
            row[f"{feature}_std"] = stds.get(feature, np.nan)

        rows.append(row)

    pd.DataFrame(rows).to_csv(master_csv_file, index=False)

    write_line(lines, f"[OK] Master JSON saved to: {master_json_file}")
    write_line(lines, f"[OK] Master CSV saved to:  {master_csv_file}")
    write_line(lines, f"[OK] Master TXT saved to:  {master_text_file}")

    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")

    write_line(lines, "")
    write_line(lines, f"Success: {success_count}")
    write_line(lines, f"Failed:  {failed_count}")

    for r in results:
        write_line(lines, "")
        write_line(lines, f"- {r.get('full_id')}: {r.get('status')}")

        if r.get("status") == "success":
            means = r.get("feature_means", {})
            write_line(lines, f"  ic.eps.ratio: {means.get('ic.eps.ratio')}")
            write_line(lines, f"  ela_meta.lin_simple.adj_r2: {means.get('ela_meta.lin_simple.adj_r2')}")
        else:
            write_line(lines, f"  error: {r.get('error')}")

    master_text_file.write_text("\n".join(lines), encoding="utf-8")

    return master_json_file, master_csv_file, master_text_file


def main():
    master_lines = []

    write_line(master_lines, "=" * 80)
    write_line(master_lines, "NEW FUNCTION TASK F: BATCH COMPUTE REAL GENERATED FUNCTION ELA FEATURES")
    write_line(master_lines, "=" * 80)
    write_line(master_lines, f"Project root:       {PROJECT_ROOT}")
    write_line(master_lines, f"Real function dir:  {REAL_FUNCTION_DIR}")
    write_line(master_lines, f"Metadata dir:       {REAL_METADATA_DIR}")
    write_line(master_lines, f"Feature output dir: {FEATURE_DIR}")
    write_line(master_lines, f"Output dir:         {OUTPUT_DIR}")
    write_line(master_lines, f"TARGET_RUN_ID:      {TARGET_RUN_ID}")
    write_line(master_lines, f"Target features:    {TARGET_FEATURES}")
    write_line(master_lines, f"DIM:                {DIM}")
    write_line(master_lines, f"N_REPEATS:          {N_REPEATS}")
    write_line(master_lines, f"SAMPLE_SIZE:        {SAMPLE_SIZE}")

    try:
        function_files = find_target_function_files(master_lines)

        results = []

        for function_file in function_files:
            print("\n" + "=" * 80)
            print(f"PROCESSING FUNCTION: {function_file.name}")
            print("=" * 80)

            result = process_one_function(function_file)
            results.append(result)

        save_master_outputs(results, master_lines)

        write_line(master_lines, "\n" + "=" * 80)
        write_line(master_lines, "TASK F BATCH CONCLUSION")
        write_line(master_lines, "=" * 80)
        write_line(master_lines, "[DONE] Batch ELA feature computation completed.")

    except Exception as e:
        write_line(master_lines, "\n" + "=" * 80)
        write_line(master_lines, "TASK F BATCH FAILED")
        write_line(master_lines, "=" * 80)
        write_line(master_lines, str(e))
        write_line(master_lines, traceback.format_exc())

        failed_master = OUTPUT_DIR / "task_F_compute_real_generated_function_ela_features_master_summary_failed.txt"
        failed_master.write_text("\n".join(master_lines), encoding="utf-8")
        print("\nMaster summary written to:")
        print(failed_master)
        raise


if __name__ == "__main__":
    main()