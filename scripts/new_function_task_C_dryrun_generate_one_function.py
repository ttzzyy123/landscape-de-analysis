# scripts/new_function_task_C_dryrun_generate_one_function.py

from pathlib import Path
import sys
import os
import json
import traceback
from unittest.mock import MagicMock

import numpy as np


# ============================================================
# New Function Task C
# Dry-run generate one function with mock LLM
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "external"
    / "LLaMEA-paper-ela"
    / "XAI-liacs-LLaMEA-6d8b3c1"
)

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task"
FUNCTION_DIR = INTERMEDIATE_DIR / "extracted_functions"
METADATA_DIR = INTERMEDIATE_DIR / "function_metadata"
OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"

FUNCTION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_FILE = FUNCTION_DIR / "generated_function_001.py"
METADATA_FILE = METADATA_DIR / "generated_function_001_metadata.json"
JSONL_FILE = METADATA_DIR / "generated_function_001_solution.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "task_C_dryrun_generate_one_function_summary.txt"


# ============================================================
# Mock LLM response
# ============================================================

MOCK_LLM_RESPONSE = r'''
# Description: Dry-run generated continuous optimization landscape with a smooth quadratic basin and mild multimodal sinusoidal ripples.

# Code:
```python
import numpy as np

class GeneratedDryRunLandscape:
    def __init__(self, dim=5):
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def f(self, x):
        x = np.asarray(x, dtype=float)

        if x.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.shape[0]}")

        shifted = x - 1.25

        quadratic_basin = np.sum(shifted ** 2)

        multimodal_ripples = 0.15 * np.sum(np.sin(3.0 * x) ** 2)

        weak_interaction = (
            0.03 * np.sum(np.sin(x[:-1] * x[1:]))
            if self.dim > 1
            else 0.0
        )

        return float(quadratic_basin + multimodal_ripples + weak_interaction)
```
'''


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def add_external_package_to_path(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. PYTHON PATH SETUP")
    write_line(lines, "=" * 80)

    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"Extracted package directory not found: {EXTRACTED_DIR}")

    sys.path.insert(0, str(EXTRACTED_DIR))
    write_line(lines, f"Added to sys.path: {EXTRACTED_DIR}")

    os.chdir(EXTRACTED_DIR)
    write_line(lines, f"Changed working directory to: {Path.cwd()}")


def import_llamea_components(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. IMPORT LLAMEA COMPONENTS")
    write_line(lines, "=" * 80)

    try:
        from llamea import LLaMEA, Ollama_LLM
        from ELA import ELAproblem

        write_line(lines, "[OK] Imported LLaMEA, Ollama_LLM, ELAproblem")
        return LLaMEA, Ollama_LLM, ELAproblem

    except Exception as e:
        write_line(lines, "[ERROR] Failed to import LLaMEA components")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise


def validate_solution_object(solution, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "4. VALIDATE GENERATED SOLUTION OBJECT")
    write_line(lines, "=" * 80)

    if solution is None:
        raise RuntimeError("LLaMEA returned None as best solution.")

    write_line(lines, f"Solution type: {type(solution)}")
    write_line(lines, f"Solution name: {getattr(solution, 'name', None)}")
    write_line(lines, f"Description:   {getattr(solution, 'description', None)}")
    write_line(lines, f"Fitness:       {getattr(solution, 'fitness', None)}")
    write_line(lines, f"Feedback:      {str(getattr(solution, 'feedback', ''))[:500]}")
    write_line(lines, f"Error:         {str(getattr(solution, 'error', ''))[:500]}")

    code = getattr(solution, "code", "")
    name = getattr(solution, "name", "")

    if not code.strip():
        raise RuntimeError("Generated solution has empty code.")

    if not name.strip():
        raise RuntimeError("Generated solution has empty name.")

    write_line(lines, "\nCode preview:")
    write_line(lines, code[:1500])

    return code, name


def validate_generated_function_code(code, class_name, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. VALIDATE GENERATED FUNCTION CODE")
    write_line(lines, "=" * 80)

    namespace = {}

    try:
        exec(code, namespace)
        write_line(lines, "[OK] exec(code) succeeded.")
    except Exception as e:
        write_line(lines, "[ERROR] exec(code) failed.")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise

    if class_name not in namespace:
        raise RuntimeError(f"Class name {class_name} not found after exec(code).")

    cls = namespace[class_name]

    try:
        instance = cls(dim=5)
        write_line(lines, "[OK] Class instantiated with dim=5.")
    except Exception as e:
        write_line(lines, "[ERROR] Failed to instantiate class with dim=5.")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise

    if not hasattr(instance, "f"):
        raise RuntimeError("Generated class does not expose .f")

    test_points = [
        np.zeros(5),
        np.ones(5),
        np.array([-5.0, -2.5, 0.0, 2.5, 5.0]),
        np.random.default_rng(42).uniform(-5, 5, size=5),
    ]

    values = []

    for idx, x in enumerate(test_points, start=1):
        y = instance.f(x)
        values.append(float(y))
        write_line(lines, f"[OK] f(test_point_{idx}) = {y}")

    if not all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite function values detected: {values}")

    x = np.random.default_rng(123).uniform(-5, 5, size=5)
    y1 = instance.f(x)
    y2 = instance.f(x)

    write_line(lines, f"Determinism check y1={y1}, y2={y2}")

    if not np.isclose(y1, y2):
        raise RuntimeError("Generated function is not deterministic for the same input.")

    write_line(lines, "[OK] Generated function is callable and deterministic.")

    return {
        "test_values": values,
        "determinism_y1": float(y1),
        "determinism_y2": float(y2),
    }


def make_json_serializable(obj):
    """Convert numpy / nested objects into JSON-serializable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def save_generated_function(solution, code, class_name, validation_result, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. SAVE GENERATED FUNCTION AND METADATA")
    write_line(lines, "=" * 80)

    header = '''"""
Generated function extracted from New Function Task C dry-run.

This file was produced by:
scripts/new_function_task_C_dryrun_generate_one_function.py

This is a mock-LLM dry-run function, not a real LLM-generated final function.
"""
'''

    FUNCTION_FILE.write_text(header + "\n\n" + code + "\n", encoding="utf-8")

    metadata = {
        "task": "new_function_task_C_dryrun_generate_one_function",
        "source": "mock_llm_dryrun",
        "package_dir": str(EXTRACTED_DIR),
        "function_file": str(FUNCTION_FILE),
        "class_name": class_name,
        "description": getattr(solution, "description", ""),
        "fitness": getattr(solution, "fitness", None),
        "feedback": getattr(solution, "feedback", ""),
        "error": getattr(solution, "error", ""),
        "dimension_tested": 5,
        "bounds_assumed": [-5.0, 5.0],
        "validation_result": validation_result,
        "note": (
            "This function was generated using a mocked LLM response to validate "
            "the LLaMEA/ELAproblem generation pipeline."
        ),
    }

    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    solution_dict = (
        solution.to_dict()
        if hasattr(solution, "to_dict")
        else {
            "code": code,
            "name": class_name,
            "description": getattr(solution, "description", ""),
            "fitness": getattr(solution, "fitness", None),
            "feedback": getattr(solution, "feedback", ""),
            "error": getattr(solution, "error", ""),
        }
    )

    solution_dict = make_json_serializable(solution_dict)

    with JSONL_FILE.open("w", encoding="utf-8") as f:
        f.write(json.dumps(solution_dict) + "\n")

    write_line(lines, f"[OK] Function saved to: {FUNCTION_FILE}")
    write_line(lines, f"[OK] Metadata saved to: {METADATA_FILE}")
    write_line(lines, f"[OK] Solution jsonl saved to: {JSONL_FILE}")


def run_dryrun_generation(lines):
    add_external_package_to_path(lines)
    LLaMEA, Ollama_LLM, ELAproblem = import_llamea_components(lines)

    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. RUN LLAMEA DRY-RUN WITH MOCK LLM")
    write_line(lines, "=" * 80)

    problem = ELAproblem(
        logger=None,
        name="ELA_DryRun_Separable",
        features=["Separable"],
        dims=[5],
        eval_timeout=60,
    )

    task_prompt = problem.get_prompt()

    write_line(lines, "Created ELAproblem:")
    write_line(lines, f"  features = {problem.features}")
    write_line(lines, f"  dims     = {problem.dims}")
    write_line(lines, f"  prompt length = {len(task_prompt)}")

    llm = Ollama_LLM("mock-model")
    llm.query = MagicMock(return_value=MOCK_LLM_RESPONSE)

    optimizer = LLaMEA(
        f=problem.evaluate_function,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt="You are an expert in continuous black-box optimization benchmark generation.",
        task_prompt=task_prompt,
        experiment_name="new-function-task-C-dryrun",
        elitism=True,
        budget=2,
        log=False,
        minimization=False,
        max_workers=1,
    )

    write_line(lines, "Initialized LLaMEA:")
    write_line(lines, "  n_parents   = 1")
    write_line(lines, "  n_offspring = 1")
    write_line(lines, "  budget      = 2")
    write_line(lines, "  log         = False")
    write_line(lines, "  llm.query   = MagicMock")

    best_solution = optimizer.run()
    write_line(lines, "[OK] optimizer.run() completed.")

    code, class_name = validate_solution_object(best_solution, lines)
    validation_result = validate_generated_function_code(code, class_name, lines)
    save_generated_function(best_solution, code, class_name, validation_result, lines)

    return best_solution


def main():
    lines = []

    write_line(lines, "=" * 80)
    write_line(lines, "NEW FUNCTION TASK C: DRY-RUN GENERATE ONE FUNCTION")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root:   {PROJECT_ROOT}")
    write_line(lines, f"Extracted dir:  {EXTRACTED_DIR}")
    write_line(lines, f"Function file:  {FUNCTION_FILE}")
    write_line(lines, f"Metadata file:  {METADATA_FILE}")
    write_line(lines, f"Summary file:   {SUMMARY_FILE}")

    try:
        run_dryrun_generation(lines)

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK C CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, "[SUCCESS] Dry-run generation pipeline works.")
        write_line(lines, "A mock generated function was parsed, evaluated, validated, and saved.")
        write_line(lines, "Next step: Task D can test the saved generated_function_001.py independently.")

    except Exception as e:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK C FAILED")
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
