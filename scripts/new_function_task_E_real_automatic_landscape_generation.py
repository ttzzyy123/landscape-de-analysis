# scripts/new_function_task_E_real_automatic_landscape_generation.py

from pathlib import Path
import sys
import os
import json
import traceback
from datetime import datetime

import numpy as np


# ============================================================
# New Function Task E
# Real automatic landscape generation with LLaMEA + real LLM
#
# Updated version:
# 1. Generate multiple real continuous optimization functions in one run
# 2. Save them as n1, n2, ... under the same unified directories
# 3. Do not overwrite previous generated functions because every run has a new RUN_ID
# 4. Use paired/combined high-level targets.
# 5. Enable the repository's built-in ELA-space fitness sharing / niching:
#      niching="sharing", distance_metric=ela_distance,
#      niche_radius=0.5, adaptive_niche_radius=True.
#    This is the key diversity mechanism from the LLaMEA paper.
#
# Manual override:
#   LLAMEA_FEATURES=Multimodality,GlobalLocal python scripts/new_function_task_E_real_automatic_landscape_generation.py
#
# Sharing override:
#   LLAMEA_ENABLE_SHARING=false python scripts/new_function_task_E_real_automatic_landscape_generation.py
#
# API key safety:
# - This script can load API keys from:
#     secrets/local_api_keys.env
# - This file should NOT be committed to git.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "external"
    / "LLaMEA-paper-ela"
    / "XAI-liacs-LLaMEA-6d8b3c1"
)

SECRETS_DIR = PROJECT_ROOT / "secrets"
LOCAL_ENV_FILE = SECRETS_DIR / "local_api_keys.env"

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate" / "new_function_task"
FUNCTION_DIR = INTERMEDIATE_DIR / "real_generated_functions"
METADATA_DIR = INTERMEDIATE_DIR / "real_function_metadata"
SAMPLES_DIR = INTERMEDIATE_DIR / "real_function_samples"
OUTPUT_DIR = PROJECT_ROOT / "output" / "new_function_task"

for d in [FUNCTION_DIR, METADATA_DIR, SAMPLES_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Generate n1 ... n8 by default for paired-target exploration.
N_FUNCTIONS = int(os.environ.get("N_REAL_FUNCTIONS", "8"))

DIM = 5
N_SAMPLES = 1000
LOWER_BOUND = -5.0
UPPER_BOUND = 5.0
RANDOM_SEED = 42

# Paper-style default settings for population-level generation.
# The LLaMEA paper uses a parent population μ=8 and offspring population λ=16,
# with a comma strategy / no elitism to encourage exploration.
LLAMEA_BUDGET = int(os.environ.get("LLAMEA_BUDGET", "55"))
N_PARENTS = int(os.environ.get("LLAMEA_N_PARENTS", "8"))
N_OFFSPRING = int(os.environ.get("LLAMEA_N_OFFSPRING", "16"))
MAX_WORKERS = int(os.environ.get("LLAMEA_MAX_WORKERS", "4"))
EVAL_TIMEOUT = int(os.environ.get("LLAMEA_EVAL_TIMEOUT", "60"))

# Enable the repository's built-in ELA-space fitness sharing / niching.
ENABLE_FITNESS_SHARING = os.environ.get("LLAMEA_ENABLE_SHARING", "true").lower() in {
    "1", "true", "yes", "y"
}
NICHE_RADIUS = float(os.environ.get("LLAMEA_NICHE_RADIUS", "0.5"))
ADAPTIVE_NICHE_RADIUS = os.environ.get("LLAMEA_ADAPTIVE_NICHE_RADIUS", "true").lower() in {
    "1", "true", "yes", "y"
}
ELITISM = os.environ.get("LLAMEA_ELITISM", "false").lower() in {
    "1", "true", "yes", "y"
}

# ============================================================
# Feature target strategy
#
# Valid ELAproblem keys:
#   Basins
#   Separable
#   GlobalLocal
#   Multimodality
#   Structure
#   Homogeneous
#   NOT Homogeneous
#   NOT Basins
#
# Default strategy:
#   Use paired/combined targets instead of single targets.
#   This follows the LLaMEA paper idea that paired-property generation
#   can produce more diverse landscapes than single-property prompting.
#
# Manual override:
#   If LLAMEA_FEATURES is set, all generated functions use that target.
#
# Example:
#   LLAMEA_FEATURES=Multimodality,GlobalLocal N_REAL_FUNCTIONS=3 python scripts/new_function_task_E_real_automatic_landscape_generation.py
# ============================================================

DEFAULT_FEATURE_TARGETS_BY_TAG = {
    # Multimodal landscapes with explicit global/local contrast.
    "n1": ["Multimodality", "GlobalLocal"],

    # Non-homogeneous search space plus multimodality.
    "n2": ["NOT Homogeneous", "Multimodality"],

    # Separable and homogeneous landscapes, similar to the paper's pre-experiment setting.
    "n3": ["Separable", "Homogeneous"],

    # Basin-related structure with separability.
    "n4": ["Basins", "Separable"],

    # Global/local contrast with non-homogeneous basins.
    "n5": ["GlobalLocal", "NOT Basins"],

    # Explicit structural landscape with multimodality.
    "n6": ["Structure", "Multimodality"],

    # Non-homogeneous search space with global/local contrast.
    "n7": ["NOT Homogeneous", "GlobalLocal"],

    # Homogeneous basin-size/search-space style target.
    "n8": ["Homogeneous", "Basins"],
}
MANUAL_FEATURE_TARGETS_RAW = os.environ.get("LLAMEA_FEATURES", "").strip()

if MANUAL_FEATURE_TARGETS_RAW:
    MANUAL_FEATURE_TARGETS = [
        x.strip()
        for x in MANUAL_FEATURE_TARGETS_RAW.split(",")
        if x.strip()
    ]
else:
    MANUAL_FEATURE_TARGETS = None


def get_feature_targets_for_tag(function_tag):
    """
    Return the LLaMEA/ELAproblem target features for this generated function.
    """

    if MANUAL_FEATURE_TARGETS is not None:
        return MANUAL_FEATURE_TARGETS

    return DEFAULT_FEATURE_TARGETS_BY_TAG.get(function_tag, ["Multimodality", "GlobalLocal"])


# ============================================================
# Utilities
# ============================================================


def build_output_paths(function_tag):
    """
    Build all output paths for one generated function.
    function_tag should be n1, n2, n3, ...
    """

    base_name = f"real_generated_function_{RUN_ID}_{function_tag}"

    return {
        "base_name": base_name,
        "function_file": FUNCTION_DIR / f"{base_name}.py",
        "metadata_file": METADATA_DIR / f"{base_name}_metadata.json",
        "jsonl_file": METADATA_DIR / f"{base_name}_solution.jsonl",
        "samples_file": SAMPLES_DIR / f"{base_name}_samples_dim5_n1000.csv",
        "summary_file": OUTPUT_DIR / f"task_E_real_generation_summary_{RUN_ID}_{function_tag}.txt",
    }


def write_line(lines, text=""):
    print(text)
    lines.append(text)


def load_local_env_file(lines=None):
    """Load local API keys/config from secrets/local_api_keys.env without overriding existing env vars."""

    if lines is not None:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "0. LOAD LOCAL ENV FILE")
        write_line(lines, "=" * 80)

    if not LOCAL_ENV_FILE.exists():
        if lines is not None:
            write_line(lines, f"[INFO] Local env file not found: {LOCAL_ENV_FILE}")
            write_line(lines, "[INFO] Falling back to existing environment variables.")
        return

    loaded_keys = []

    with LOCAL_ENV_FILE.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue

            if key not in os.environ:
                os.environ[key] = value
                loaded_keys.append(key)

    if lines is not None:
        safe_loaded_keys = [
            k for k in loaded_keys
            if "KEY" not in k.upper() and "TOKEN" not in k.upper()
        ]
        hidden_loaded_keys = [
            k for k in loaded_keys
            if k not in safe_loaded_keys
        ]

        write_line(lines, f"[OK] Loaded local env file: {LOCAL_ENV_FILE}")

        if safe_loaded_keys:
            write_line(lines, f"Loaded non-secret keys: {safe_loaded_keys}")

        if hidden_loaded_keys:
            write_line(lines, f"Loaded secret keys: {[k + '=***' for k in hidden_loaded_keys]}")


def make_json_serializable(obj):
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


def add_external_package_to_path(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "1. PYTHON PATH SETUP")
    write_line(lines, "=" * 80)

    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"Extracted package directory not found: {EXTRACTED_DIR}")

    extracted_str = str(EXTRACTED_DIR)
    if extracted_str not in sys.path:
        sys.path.insert(0, extracted_str)

    write_line(lines, f"Added to sys.path: {EXTRACTED_DIR}")

    os.chdir(EXTRACTED_DIR)
    write_line(lines, f"Changed working directory to: {Path.cwd()}")


def import_llamea_components(lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "2. IMPORT LLAMEA COMPONENTS")
    write_line(lines, "=" * 80)

    try:
        import llamea
        from llamea import LLaMEA
        from ELA import ELAproblem, ela_distance

        available = dir(llamea)
        write_line(lines, "[OK] Imported LLaMEA, ELAproblem, and ela_distance")
        write_line(lines, f"Available llamea exports: {available}")

        return llamea, LLaMEA, ELAproblem, ela_distance

    except Exception as e:
        write_line(lines, "[ERROR] Failed to import LLaMEA components")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise


def create_real_llm(llamea_module, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "3. CREATE REAL LLM BACKEND")
    write_line(lines, "=" * 80)

    backend = os.environ.get("REAL_LLM_BACKEND", "gemini").strip().lower()
    write_line(lines, f"REAL_LLM_BACKEND = {backend}")

    if backend == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        write_line(lines, f"OLLAMA_MODEL = {model}")

        if not hasattr(llamea_module, "Ollama_LLM"):
            raise RuntimeError("llamea module does not expose Ollama_LLM")

        llm = llamea_module.Ollama_LLM(model)
        write_line(lines, "[OK] Created Ollama_LLM")
        return llm, {"backend": backend, "model": model}

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to secrets/local_api_keys.env "
                "or export it in the shell."
            )

        write_line(lines, f"OPENAI_MODEL = {model}")
        write_line(lines, "OPENAI_API_KEY = ***")

        if hasattr(llamea_module, "OpenAI_LLM"):
            try:
                llm = llamea_module.OpenAI_LLM(model=model, api_key=api_key)
            except TypeError:
                try:
                    llm = llamea_module.OpenAI_LLM(api_key=api_key, model=model)
                except TypeError:
                    llm = llamea_module.OpenAI_LLM(model)

            write_line(lines, "[OK] Created OpenAI_LLM")
            return llm, {"backend": backend, "model": model}

        raise RuntimeError("llamea module does not expose OpenAI_LLM. Try REAL_LLM_BACKEND=gemini.")

    if backend == "gemini":
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = gemini_key or google_key
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if not api_key:
            raise RuntimeError(
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. Add it to "
                "secrets/local_api_keys.env or export it in the shell."
            )

        os.environ.setdefault("GEMINI_API_KEY", api_key)
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        write_line(lines, f"GEMINI_MODEL = {model}")
        write_line(lines, "GEMINI_API_KEY / GOOGLE_API_KEY = ***")

        if hasattr(llamea_module, "Gemini_LLM"):
            try:
                llm = llamea_module.Gemini_LLM(model=model, api_key=api_key)
            except TypeError:
                try:
                    llm = llamea_module.Gemini_LLM(api_key=api_key, model=model)
                except TypeError:
                    llm = llamea_module.Gemini_LLM(model)

            write_line(lines, "[OK] Created Gemini_LLM")
            return llm, {"backend": backend, "model": model}

        raise RuntimeError("llamea module does not expose Gemini_LLM. Try REAL_LLM_BACKEND=openai.")

    raise ValueError(f"Unsupported REAL_LLM_BACKEND: {backend}")


def validate_solution_object(solution, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "5. VALIDATE GENERATED SOLUTION OBJECT")
    write_line(lines, "=" * 80)

    if solution is None:
        raise RuntimeError("LLaMEA returned None as best solution.")

    write_line(lines, f"Solution type: {type(solution)}")
    write_line(lines, f"Solution name: {getattr(solution, 'name', None)}")
    write_line(lines, f"Description:   {getattr(solution, 'description', None)}")
    write_line(lines, f"Fitness:       {getattr(solution, 'fitness', None)}")
    write_line(lines, f"Feedback:      {str(getattr(solution, 'feedback', ''))[:1000]}")
    write_line(lines, f"Error:         {str(getattr(solution, 'error', ''))[:1000]}")

    code = getattr(solution, "code", "")
    name = getattr(solution, "name", "")

    if not code.strip():
        raise RuntimeError("Generated solution has empty code.")

    if not name.strip():
        raise RuntimeError("Generated solution has empty name.")

    write_line(lines, "\nCode preview:")
    write_line(lines, code[:2000])

    return code, name


def validate_generated_function_code(code, class_name, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "6. VALIDATE GENERATED FUNCTION CODE")
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
        detected = [(k, v) for k, v in namespace.items() if isinstance(v, type)]
        if not detected:
            raise RuntimeError(f"Class name {class_name} not found and no class detected after exec(code).")

        write_line(lines, f"[WARN] Class name {class_name} not found. Using detected class: {detected[0][0]}")
        class_name, cls = detected[0]
    else:
        cls = namespace[class_name]

    try:
        instance = cls(dim=DIM)
        write_line(lines, f"[OK] Class instantiated with dim={DIM}.")
    except Exception as e:
        write_line(lines, f"[ERROR] Failed to instantiate class with dim={DIM}.")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())
        raise

    if not hasattr(instance, "f"):
        raise RuntimeError("Generated class does not expose .f")

    test_points = [
        np.zeros(DIM),
        np.ones(DIM),
        np.array([-5.0, -2.5, 0.0, 2.5, 5.0]),
        np.random.default_rng(42).uniform(LOWER_BOUND, UPPER_BOUND, size=DIM),
    ]

    values = []

    for idx, x in enumerate(test_points, start=1):
        y = instance.f(x)
        y = float(y)
        values.append(y)
        write_line(lines, f"[OK] f(test_point_{idx}) = {y}")

    if not all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite function values detected: {values}")

    x = np.random.default_rng(123).uniform(LOWER_BOUND, UPPER_BOUND, size=DIM)
    y1 = float(instance.f(x))
    y2 = float(instance.f(x))

    write_line(lines, f"Determinism check y1={y1}, y2={y2}")

    if not np.isclose(y1, y2):
        raise RuntimeError("Generated function is not deterministic for the same input.")

    write_line(lines, "[OK] Generated function is callable and deterministic.")

    return class_name, instance, {
        "test_values": values,
        "determinism_y1": y1,
        "determinism_y2": y2,
    }


def sample_function(instance, paths, lines):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "7. SAMPLE REAL GENERATED FUNCTION")
    write_line(lines, "=" * 80)

    lower = float(getattr(instance, "lower_bound", LOWER_BOUND))
    upper = float(getattr(instance, "upper_bound", UPPER_BOUND))

    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.uniform(lower, upper, size=(N_SAMPLES, DIM))
    y = np.array([float(instance.f(x)) for x in X])

    if not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite values detected in sampled function values.")

    data = np.column_stack([X, y])
    header = ",".join([f"x{i + 1}" for i in range(DIM)] + ["y"])

    np.savetxt(
        paths["samples_file"],
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    summary = {
        "samples_file": str(paths["samples_file"]),
        "n_samples": N_SAMPLES,
        "dim": DIM,
        "lower_bound": lower,
        "upper_bound": upper,
        "random_seed": RANDOM_SEED,
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }

    write_line(lines, f"[OK] Saved samples to: {paths['samples_file']}")
    write_line(lines, f"y min:  {summary['y_min']}")
    write_line(lines, f"y max:  {summary['y_max']}")
    write_line(lines, f"y mean: {summary['y_mean']}")
    write_line(lines, f"y std:  {summary['y_std']}")

    return summary


def save_real_generated_function(
    paths,
    function_tag,
    feature_targets,
    solution,
    code,
    class_name,
    llm_info,
    validation_result,
    sampling_summary,
    lines,
):
    write_line(lines, "\n" + "=" * 80)
    write_line(lines, "8. SAVE REAL GENERATED FUNCTION AND METADATA")
    write_line(lines, "=" * 80)

    header = f'''"""
Real generated function from New Function Task E.

Generated by:
scripts/new_function_task_E_real_automatic_landscape_generation.py

Run ID:
{RUN_ID}

Function tag:
{function_tag}

Feature targets:
{feature_targets}

Backend:
{llm_info}

Interface:
obj = {class_name}(dim=5)
y = obj.f(x)
"""
'''

    paths["function_file"].write_text(header + "\n\n" + code + "\n", encoding="utf-8")

    metadata = {
        "task": "new_function_task_E_real_automatic_landscape_generation",
        "run_id": RUN_ID,
        "function_tag": function_tag,
        "source": "real_llm_generation",
        "package_dir": str(EXTRACTED_DIR),
        "function_file": str(paths["function_file"]),
        "metadata_file": str(paths["metadata_file"]),
        "solution_jsonl_file": str(paths["jsonl_file"]),
        "samples_file": str(paths["samples_file"]),
        "llm_info": llm_info,
        "class_name": class_name,
        "description": getattr(solution, "description", ""),
        "fitness": make_json_serializable(getattr(solution, "fitness", None)),
        "feedback": getattr(solution, "feedback", ""),
        "error": getattr(solution, "error", ""),
        "features_targeted_by_ELAproblem": feature_targets,
        "dims": [DIM],
        "llamea_budget": LLAMEA_BUDGET,
        "n_parents": N_PARENTS,
        "n_offspring": N_OFFSPRING,
        "max_workers": MAX_WORKERS,
        "eval_timeout": EVAL_TIMEOUT,
        "fitness_sharing_enabled": ENABLE_FITNESS_SHARING,
        "niching": "sharing" if ENABLE_FITNESS_SHARING else None,
        "distance_metric": "ela_distance" if ENABLE_FITNESS_SHARING else None,
        "niche_radius": NICHE_RADIUS,
        "adaptive_niche_radius": ADAPTIVE_NICHE_RADIUS,
        "elitism": ELITISM,
        "validation_result": validation_result,
        "sampling_summary": sampling_summary,
        "next_step": (
            "Compute ELA features ic.eps.ratio and ela_meta.lin_simple.adj_r2 "
            "for this real generated function, then assign it to an existing BBOB group."
        ),
    }

    paths["metadata_file"].write_text(
        json.dumps(make_json_serializable(metadata), indent=2),
        encoding="utf-8",
    )

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

    with paths["jsonl_file"].open("w", encoding="utf-8") as f:
        f.write(json.dumps(solution_dict) + "\n")

    write_line(lines, f"[OK] Function saved to: {paths['function_file']}")
    write_line(lines, f"[OK] Metadata saved to: {paths['metadata_file']}")
    write_line(lines, f"[OK] Solution jsonl saved to: {paths['jsonl_file']}")
    write_line(lines, f"[OK] Samples saved to: {paths['samples_file']}")


def run_single_generation(function_tag, llamea_module, LLaMEA, ELAproblem, ela_distance, llm, llm_info):
    """
    Generate one real LLM function and save it independently.
    """

    lines = []
    paths = build_output_paths(function_tag)
    feature_targets = get_feature_targets_for_tag(function_tag)

    write_line(lines, "=" * 80)
    write_line(lines, f"NEW FUNCTION TASK E: REAL GENERATION {function_tag}")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root:     {PROJECT_ROOT}")
    write_line(lines, f"Extracted dir:    {EXTRACTED_DIR}")
    write_line(lines, f"Run ID:           {RUN_ID}")
    write_line(lines, f"Function tag:     {function_tag}")
    write_line(lines, f"Function file:    {paths['function_file']}")
    write_line(lines, f"Metadata file:    {paths['metadata_file']}")
    write_line(lines, f"Solution jsonl:   {paths['jsonl_file']}")
    write_line(lines, f"Samples file:     {paths['samples_file']}")
    write_line(lines, f"Summary file:     {paths['summary_file']}")
    write_line(lines, f"Target features:  {feature_targets}")
    write_line(lines, f"Budget:           {LLAMEA_BUDGET}")
    write_line(lines, f"Fitness sharing:  {ENABLE_FITNESS_SHARING}")
    write_line(lines, f"Niche radius:     {NICHE_RADIUS}")
    write_line(lines, f"Adaptive radius:  {ADAPTIVE_NICHE_RADIUS}")
    write_line(lines, f"Elitism:          {ELITISM}")

    try:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "4. RUN LLAMEA REAL GENERATION")
        write_line(lines, "=" * 80)

        problem = ELAproblem(
            logger=None,
            name=f"ELA_RealGeneration_{function_tag}",
            features=feature_targets,
            dims=[DIM],
            eval_timeout=EVAL_TIMEOUT,
        )

        task_prompt = problem.get_prompt()

        write_line(lines, "Created ELAproblem:")
        write_line(lines, f"  features = {problem.features}")
        write_line(lines, f"  dims     = {problem.dims}")
        write_line(lines, f"  prompt length = {len(task_prompt)}")

        optimizer = LLaMEA(
            f=problem.evaluate_function,
            llm=llm,
            n_parents=N_PARENTS,
            n_offspring=N_OFFSPRING,
            role_prompt="You are an expert in continuous black-box optimization benchmark generation.",
            task_prompt=task_prompt,
            experiment_name=f"new-function-task-E-real-generation-{RUN_ID}-{function_tag}-sharing",
            elitism=ELITISM,
            budget=LLAMEA_BUDGET,
            log=True,
            minimization=False,
            max_workers=MAX_WORKERS,
            niching="sharing" if ENABLE_FITNESS_SHARING else None,
            distance_metric=ela_distance if ENABLE_FITNESS_SHARING else None,
            niche_radius=NICHE_RADIUS,
            adaptive_niche_radius=ADAPTIVE_NICHE_RADIUS,
        )

        write_line(lines, "Initialized LLaMEA:")
        write_line(lines, f"  n_parents   = {N_PARENTS}")
        write_line(lines, f"  n_offspring = {N_OFFSPRING}")
        write_line(lines, f"  budget      = {LLAMEA_BUDGET}")
        write_line(lines, "  log         = True")
        write_line(lines, f"  max_workers = {MAX_WORKERS}")
        write_line(lines, f"  elitism     = {ELITISM}")
        write_line(lines, f"  niching     = {'sharing' if ENABLE_FITNESS_SHARING else None}")
        write_line(lines, f"  distance    = {'ela_distance' if ENABLE_FITNESS_SHARING else None}")
        write_line(lines, f"  radius      = {NICHE_RADIUS}")
        write_line(lines, f"  adaptive    = {ADAPTIVE_NICHE_RADIUS}")

        result = optimizer.run()

        if isinstance(result, tuple):
            best_solution = result[0]
        else:
            best_solution = result

        write_line(lines, "[OK] optimizer.run() completed.")

        code, parsed_class_name = validate_solution_object(best_solution, lines)

        class_name, instance, validation_result = validate_generated_function_code(
            code,
            parsed_class_name,
            lines,
        )

        sampling_summary = sample_function(instance, paths, lines)

        save_real_generated_function(
            paths=paths,
            function_tag=function_tag,
            feature_targets=feature_targets,
            solution=best_solution,
            code=code,
            class_name=class_name,
            llm_info=llm_info,
            validation_result=validation_result,
            sampling_summary=sampling_summary,
            lines=lines,
        )

        write_line(lines, "\n" + "=" * 80)
        write_line(lines, "TASK E SINGLE GENERATION CONCLUSION")
        write_line(lines, "=" * 80)
        write_line(lines, f"[SUCCESS] Real automatic landscape generation completed for {function_tag}.")
        write_line(lines, f"Feature target used: {feature_targets}")
        write_line(lines, "Next step: compute ic.eps.ratio and ela_meta.lin_simple.adj_r2 for this function.")

        return {
            "function_tag": function_tag,
            "feature_targets": feature_targets,
            "status": "success",
            "paths": {k: str(v) for k, v in paths.items()},
        }

    except Exception as e:
        write_line(lines, "\n" + "=" * 80)
        write_line(lines, f"TASK E FAILED FOR {function_tag}")
        write_line(lines, "=" * 80)
        write_line(lines, f"Feature target used: {feature_targets}")
        write_line(lines, str(e))
        write_line(lines, traceback.format_exc())

        return {
            "function_tag": function_tag,
            "feature_targets": feature_targets,
            "status": "failed",
            "error": str(e),
            "paths": {k: str(v) for k, v in paths.items()},
        }

    finally:
        paths["summary_file"].write_text("\n".join(lines), encoding="utf-8")
        print("\nSummary written to:")
        print(paths["summary_file"])


def main():
    master_lines = []

    master_summary_file = OUTPUT_DIR / f"task_E_real_generation_master_summary_{RUN_ID}.txt"

    write_line(master_lines, "=" * 80)
    write_line(master_lines, "NEW FUNCTION TASK E: MULTI REAL AUTOMATIC LANDSCAPE GENERATION")
    write_line(master_lines, "=" * 80)
    write_line(master_lines, f"Project root:     {PROJECT_ROOT}")
    write_line(master_lines, f"Extracted dir:    {EXTRACTED_DIR}")
    write_line(master_lines, f"Local env file:   {LOCAL_ENV_FILE}")
    write_line(master_lines, f"Run ID:           {RUN_ID}")
    write_line(master_lines, f"N_FUNCTIONS:      {N_FUNCTIONS}")
    write_line(master_lines, f"Manual targets:   {MANUAL_FEATURE_TARGETS}")
    write_line(master_lines, f"Default targets:  {DEFAULT_FEATURE_TARGETS_BY_TAG}")
    write_line(master_lines, f"Budget:           {LLAMEA_BUDGET}")
    write_line(master_lines, f"Master summary:   {master_summary_file}")

    try:
        load_local_env_file(master_lines)
        add_external_package_to_path(master_lines)
        llamea_module, LLaMEA, ELAproblem, ela_distance = import_llamea_components(master_lines)
        llm, llm_info = create_real_llm(llamea_module, master_lines)

        results = []

        for idx in range(1, N_FUNCTIONS + 1):
            function_tag = f"n{idx}"
            feature_targets = get_feature_targets_for_tag(function_tag)

            print("\n" + "=" * 80)
            print(f"STARTING GENERATION {function_tag}")
            print(f"FEATURE TARGETS: {feature_targets}")
            print("=" * 80)

            result = run_single_generation(
                function_tag=function_tag,
                llamea_module=llamea_module,
                LLaMEA=LLaMEA,
                ELAproblem=ELAproblem,
                ela_distance=ela_distance,
                llm=llm,
                llm_info=llm_info,
            )

            results.append(result)

            write_line(master_lines, "")
            write_line(master_lines, "-" * 80)
            write_line(master_lines, f"Result for {function_tag}: {result['status']}")
            write_line(master_lines, f"Feature targets: {result.get('feature_targets')}")

            if result["status"] == "success":
                write_line(master_lines, f"Function file: {result['paths']['function_file']}")
                write_line(master_lines, f"Metadata file: {result['paths']['metadata_file']}")
                write_line(master_lines, f"Samples file:  {result['paths']['samples_file']}")
            else:
                write_line(master_lines, f"Error: {result.get('error')}")

        write_line(master_lines, "\n" + "=" * 80)
        write_line(master_lines, "ALL GENERATIONS FINISHED")
        write_line(master_lines, "=" * 80)

        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        write_line(master_lines, f"Success: {success_count}")
        write_line(master_lines, f"Failed:  {failed_count}")

        write_line(master_lines, "\nGenerated outputs:")
        for r in results:
            write_line(master_lines, f"- {r['function_tag']}: {r['status']}")
            write_line(master_lines, f"  feature_targets: {r.get('feature_targets')}")
            write_line(master_lines, f"  function: {r['paths']['function_file']}")
            write_line(master_lines, f"  metadata: {r['paths']['metadata_file']}")
            write_line(master_lines, f"  samples:  {r['paths']['samples_file']}")

        master_json_file = OUTPUT_DIR / f"task_E_real_generation_master_summary_{RUN_ID}.json"
        master_json_file.write_text(
            json.dumps(make_json_serializable(results), indent=2),
            encoding="utf-8",
        )

        write_line(master_lines, f"\nMaster JSON saved to: {master_json_file}")

    except Exception as e:
        write_line(master_lines, "\n" + "=" * 80)
        write_line(master_lines, "TASK E MULTI GENERATION FAILED")
        write_line(master_lines, "=" * 80)
        write_line(master_lines, str(e))
        write_line(master_lines, traceback.format_exc())
        raise

    finally:
        master_summary_file.write_text("\n".join(master_lines), encoding="utf-8")
        print("\nMaster summary written to:")
        print(master_summary_file)


if __name__ == "__main__":
    main()