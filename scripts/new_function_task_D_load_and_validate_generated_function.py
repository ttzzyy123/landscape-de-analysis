# scripts/new_function_task_D_load_and_validate_generated_function.py

from pathlib import Path
import sys
import json
import traceback
import importlib.util

import numpy as np


# ============================================================
# New Function Task D
# Independently load and validate generated_function_001.py
#
# Goal:
# 1. Load generated_function_001.py from Task C
# 2. Instantiate the generated landscape class with dim=5
# 3. Evaluate f(x) on fixed and random points
# 4. Generate sample data for later ELA feature computation
# 5. Save validation outputs and sampled dataset
#
# Note:
# This task does NOT call LLaMEA or any LLM.
# It treats the saved generated function as a standalone benchmark function.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task"
FUNCTION_DIR = INTERMEDIATE_DIR / "extracted_functions"
METADATA_DIR = INTERMEDIATE_DIR / "function_metadata"
SAMPLES_DIR = INTERMEDIATE_DIR / "function_samples"

OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"

FUNCTION_FILE = FUNCTION_DIR / "generated_function_001.py"
TASK_C_METADATA_FILE = METADATA_DIR / "generated_function_001_metadata.json"

TASK_D_METADATA_FILE = METADATA_DIR / "generated_function_001_task_D_validation.json"
SAMPLES_FILE = SAMPLES_DIR / "generated_function_001_samples_dim5_n1000.csv"
SUMMARY_FILE = OUTPUT_DIR / "task_D_load_and_validate_generated_function_summary.txt"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


DIM = 5
N_SAMPLES = 1000
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0
RANDOM_SEED = 42


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_generated_module(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. LOAD GENERATED FUNCTION MODULE")
    write_line(lines, "=" * 80)

    if not FUNCTION_FILE.exists():
        raise FileNotFoundError(f"Generated function file not found: {FUNCTION_FILE}")

    write_line(lines, f"Function file found: {FUNCTION_FILE}")

    spec = importlib.util.spec_from_file_location(
        "generated_function_001",
        FUNCTION_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for: {FUNCTION_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    write_line(lines, "[OK] Module loaded successfully.")

    return module


def find_generated_class(module, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. FIND GENERATED LANDSCAPE CLASS")
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

    class_name, cls = candidates[0]

    write_line(lines, f"Selected class: {class_name}")

    return class_name, cls


def instantiate_function(cls, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. INSTANTIATE GENERATED FUNCTION")
    write_line(lines, "=" * 80)

    instance = cls(dim=DIM)

    if not hasattr(instance, "f"):
        raise RuntimeError("Generated class does not have method .f")

    write_line(lines, f"[OK] Instantiated with dim={DIM}")
    write_line(lines, "[OK] Method .f exists")

    lower = getattr(instance, "lower_bound", LOWER_BOUND)
    upper = getattr(instance, "upper_bound", UPPER_BOUND)

    write_line(lines, f"Detected lower bound: {lower}")
    write_line(lines, f"Detected upper bound: {upper}")

    return instance, float(lower), float(upper)


def validate_function_values(instance, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "4. VALIDATE FUNCTION VALUES")
    write_line(lines, "=" * 80)

    test_points = {
        "zero": np.zeros(DIM),
        "one": np.ones(DIM),
        "lower_bound": np.full(DIM, LOWER_BOUND),
        "upper_bound": np.full(DIM, UPPER_BOUND),
        "mixed": np.array([-5.0, -2.5, 0.0, 2.5, 5.0]),
        "random_seed_42": np.random.default_rng(42).uniform(LOWER_BOUND, UPPER_BOUND, size=DIM),
    }

    values = {}

    for name, x in test_points.items():
        y = instance.f(x)
        y = float(y)

        if not np.isfinite(y):
            raise RuntimeError(f"Non-finite value for test point {name}: {y}")

        values[name] = {
            "x": x.tolist(),
            "y": y,
        }

        write_line(lines, f"[OK] f({name}) = {y}")

    x_det = np.random.default_rng(123).uniform(LOWER_BOUND, UPPER_BOUND, size=DIM)
    y1 = float(instance.f(x_det))
    y2 = float(instance.f(x_det))

    write_line(lines, f"Determinism check: y1={y1}, y2={y2}")

    if not np.isclose(y1, y2):
        raise RuntimeError("Function is not deterministic.")

    write_line(lines, "[OK] Determinism check passed.")

    return values, {
        "x": x_det.tolist(),
        "y1": y1,
        "y2": y2,
        "passed": True,
    }


def sample_function(instance, lower, upper, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. SAMPLE GENERATED FUNCTION")
    write_line(lines, "=" * 80)

    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.uniform(lower, upper, size=(N_SAMPLES, DIM))
    y = np.array([float(instance.f(x)) for x in X])

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite values detected in sampled function values.")

    data = np.column_stack([X, y])

    header = ",".join([f"x{i + 1}" for i in range(DIM)] + ["y"])
    np.savetxt(
        SAMPLES_FILE,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    write_line(lines, f"[OK] Sampled {N_SAMPLES} points in dimension {DIM}.")
    write_line(lines, f"[OK] Saved samples to: {SAMPLES_FILE}")
    write_line(lines, f"y min:  {float(np.min(y))}")
    write_line(lines, f"y max:  {float(np.max(y))}")
    write_line(lines, f"y mean: {float(np.mean(y))}")
    write_line(lines, f"y std:  {float(np.std(y))}")

    return {
        "n_samples": N_SAMPLES,
        "dim": DIM,
        "lower_bound": lower,
        "upper_bound": upper,
        "random_seed": RANDOM_SEED,
        "samples_file": str(SAMPLES_FILE),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }


def save_task_d_metadata(class_name, test_values, determinism, sampling_summary, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. SAVE TASK D METADATA")
    write_line(lines, "=" * 80)

    task_c_metadata = load_json(TASK_C_METADATA_FILE)

    metadata = {
        "task": "new_function_task_D_load_and_validate_generated_function",
        "source_function_file": str(FUNCTION_FILE),
        "task_c_metadata_file": str(TASK_C_METADATA_FILE),
        "class_name": class_name,
        "dimension": DIM,
        "validation": {
            "module_loaded": True,
            "class_found": True,
            "instantiated": True,
            "has_f_method": True,
            "test_values": test_values,
            "determinism": determinism,
        },
        "sampling_summary": sampling_summary,
        "task_c_metadata_available": task_c_metadata is not None,
        "task_c_metadata": task_c_metadata,
        "next_step": (
            "Use the saved sample file to compute ELA features: "
            "ic.eps.ratio and ela_meta.lin_simple.adj_r2."
        ),
    }

    TASK_D_METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_line(lines, f"[OK] Task D metadata saved to: {TASK_D_METADATA_FILE}")

    return metadata


def main():
    lines = []

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK D: LOAD AND VALIDATE GENERATED FUNCTION")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root:   {PROJECT_ROOT}")
    write_line(lines, f"Function file:  {FUNCTION_FILE}")
    write_line(lines, f"Samples file:   {SAMPLES_FILE}")
    write_line(lines, f"Metadata file:  {TASK_D_METADATA_FILE}")
    write_line(lines, f"Summary file:   {SUMMARY_FILE}")

    try:
        module = load_generated_module(lines)
        class_name, cls = find_generated_class(module, lines)
        instance, lower, upper = instantiate_function(cls, lines)
        test_values, determinism = validate_function_values(instance, lines)
        sampling_summary = sample_function(instance, lower, upper, lines)
        save_task_d_metadata(class_name, test_values, determinism, sampling_summary, lines)

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK D CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, "[SUCCESS] Generated function can be loaded independently.")
        write_line(lines, "[SUCCESS] Function values are finite and deterministic.")
        write_line(lines, "[SUCCESS] Sample dataset for later ELA computation has been saved.")
        write_line(lines, "Next step: Task E should compute ic.eps.ratio and ela_meta.lin_simple.adj_r2.")

    except Exception as e:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK D FAILED")
        write_line(lines, "=" * 80)
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise

    finally:
        SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
        print("\nSummary written to:")
        print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
