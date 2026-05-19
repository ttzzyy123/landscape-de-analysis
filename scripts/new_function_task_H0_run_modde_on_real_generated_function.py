# scripts/new_function_task_H0_run_modde_on_real_generated_function.py

from pathlib import Path
from multiprocessing import Pool, cpu_count
import importlib.util
import json
import os
import traceback

import ioh
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from modde import ModularDE
from tqdm import tqdm


# ============================================================
# New Function Task H0
# Batch run Niels-style modDE experiments on selected real generated functions
#
# Selected functions:
#   n3, n7, n5 from run 20260517_125137
#
# Default full setting:
#   1000 configs
#   3 repeats
#   10000 evaluations
#
# Output:
# intermediate/new_function_task/modde_real_function_results/
#   real_generated_function_20260517_125137_n3_modde_results_processed.pkl
#   real_generated_function_20260517_125137_n7_modde_results_processed.pkl
#   real_generated_function_20260517_125137_n5_modde_results_processed.pkl
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_RUN_ID = os.environ.get("REAL_FUNCTION_BASE_RUN_ID", "20260517_125137")

# Run these generated functions by default.
TARGET_FUNCTION_TAGS = os.environ.get("REAL_FUNCTION_TAGS", "n3,n7,n5").split(",")
TARGET_FUNCTION_TAGS = [x.strip() for x in TARGET_FUNCTION_TAGS if x.strip()]

DIM = int(os.environ.get("REAL_FUNCTION_DIM", "5"))

# Use separate fid values so later files can be distinguished safely.
FID_BASE = int(os.environ.get("REAL_FUNCTION_FID_BASE", "25"))
IID_NEW = int(os.environ.get("REAL_FUNCTION_IID", "1"))

SAMPLE_SIZE = int(os.environ.get("MODDE_SAMPLE_SIZE", "1000"))
REPS = int(os.environ.get("MODDE_REPS", "3"))
BUDGET = int(os.environ.get("MODDE_BUDGET", "10000"))
SEED = int(os.environ.get("MODDE_CONFIG_SEED", "42"))

PARALLEL = os.environ.get("MODDE_PARALLEL", "true").lower() in {"1", "true", "yes", "y"}
N_WORKERS = int(os.environ.get("MODDE_N_WORKERS", str(max(1, min(cpu_count(), 16)))))

START_INDEX = int(os.environ.get("MODDE_START_INDEX", "0"))
STOP_AFTER = os.environ.get("MODDE_STOP_AFTER", "")
STOP_AFTER = int(STOP_AFTER) if STOP_AFTER.strip() else None

REAL_FUNCTION_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "real_generated_functions"
)

OUT_DIR = (
    PROJECT_ROOT
    / "intermediate"
    / "new_function_task"
    / "modde_real_function_results"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Same modDE configuration space as Niels' config.py
# ============================================================


de_cs = ConfigurationSpace(
    {
        "F": (0.0, 1.0),
        "CR": (0.001, 1.0),
        "lambda_": (10, 50),
        "mutation_base": ["target", "best", "rand"],
        "mutation_reference": ["pbest", "rand", "nan", "best"],
        "mutation_n_comps": [1, 2],
        "use_archive": [False, True],
        "crossover": ["exp", "bin"],
        "adaptation_method": ["nan"],
        "lpsr": [False, True],
    }
)

DE_COLUMNS = [
    "F",
    "CR",
    "lambda_",
    "mutation_base",
    "mutation_reference",
    "mutation_n_comps",
    "use_archive",
    "crossover",
    "adaptation_method",
    "lpsr",
]


# ============================================================
# AOC logger copied/adapted from IOHxplainer utils.py
# ============================================================


class aoc_logger(ioh.logger.AbstractLogger):
    def __init__(
        self,
        budget,
        lower=1e-8,
        upper1=1e2,
        upper2=1e8,
        scale_log=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aoc1 = 0
        self.aoc2 = 0
        self.lower = lower
        self.upper1 = upper1
        self.upper2 = upper2
        self.budget = budget
        self.transform = lambda x: np.log10(x) if scale_log else x
        self.auc1_list = []
        self.auc2_list = []

    def __call__(self, log_info: ioh.LogInfo):
        if log_info.evaluations >= self.budget:
            return

        y_value1 = np.clip(log_info.raw_y_best, self.lower, self.upper1)
        fraction1 = (self.transform(y_value1) - self.transform(self.lower)) / (
            self.transform(self.upper1) - self.transform(self.lower)
        )
        self.aoc1 += fraction1

        y_value2 = np.clip(log_info.raw_y_best, self.lower, self.upper2)
        fraction2 = (self.transform(y_value2) - self.transform(self.lower)) / (
            self.transform(self.upper2) - self.transform(self.lower)
        )
        self.aoc2 += fraction2

        if log_info.evaluations % (self.budget / 10) == 0:
            corrected_aoc1 = (
                self.aoc1
                + np.clip(self.budget - log_info.evaluations, 0, self.budget) * fraction1
            ) / self.budget
            self.auc1_list.append(1 - corrected_aoc1)

            corrected_aoc2 = (
                self.aoc2
                + np.clip(self.budget - log_info.evaluations, 0, self.budget) * fraction2
            ) / self.budget
            self.auc2_list.append(1 - corrected_aoc2)

    def reset(self, func=None):
        super().reset()
        self.aoc1 = 0
        self.aoc2 = 0
        self.auc1_list = []
        self.auc2_list = []


def correct_aoc(ioh_function, logger, budget):
    y_best = ioh_function.state.current_best_internal.y

    fraction = (
        logger.transform(np.clip(y_best, logger.lower, logger.upper1))
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper1) - logger.transform(logger.lower))
    aoc1 = (
        logger.aoc1
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    fraction = (
        logger.transform(np.clip(y_best, logger.lower, logger.upper2))
        - logger.transform(logger.lower)
    ) / (logger.transform(logger.upper2) - logger.transform(logger.lower))
    aoc2 = (
        logger.aoc2
        + np.clip(budget - ioh_function.state.evaluations, 0, budget) * fraction
    ) / budget

    return 1 - aoc1, 1 - aoc2


# ============================================================
# Per-function context
# ============================================================


CURRENT_CONTEXT = {}


def set_context(context):
    global CURRENT_CONTEXT
    CURRENT_CONTEXT = context


def get_function_file(full_id):
    return REAL_FUNCTION_DIR / f"real_generated_function_{full_id}.py"


def get_output_paths(full_id):
    return {
        "raw_pkl": OUT_DIR / f"real_generated_function_{full_id}_modde_results_raw.pkl",
        "processed_pkl": OUT_DIR / f"real_generated_function_{full_id}_modde_results_processed.pkl",
        "checkpoint": OUT_DIR / f"real_generated_function_{full_id}_modde_intermediate.csv",
        "configs": OUT_DIR / f"real_generated_function_{full_id}_modde_sampled_configs.csv",
        "summary": OUT_DIR / f"real_generated_function_{full_id}_modde_run_summary.json",
    }


# ============================================================
# Generated function loading and IOH wrapping
# ============================================================


def load_generated_class(function_file):
    if not function_file.exists():
        raise FileNotFoundError(f"Real generated function file not found: {function_file}")

    spec = importlib.util.spec_from_file_location(function_file.stem, function_file)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {function_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and not name.startswith("_"):
            candidates.append((name, obj))

    if not candidates:
        raise RuntimeError("No generated class found in real generated function file.")

    # Prefer class named landscape if present.
    for name, cls in candidates:
        if name == "landscape":
            return name, cls

    return candidates[0]


def make_ioh_problem(dim, problem_name):
    function_file = CURRENT_CONTEXT["function_file"]

    class_name, cls = load_generated_class(function_file)
    landscape = cls(dim=dim)

    lower = float(getattr(landscape, "lower_bound", getattr(landscape, "bounds_lb", -5.0)))
    upper = float(getattr(landscape, "upper_bound", getattr(landscape, "bounds_ub", 5.0)))

    def objective(x):
        return float(landscape.f(np.asarray(x, dtype=float)))

    try:
        problem = ioh.wrap_problem(
            objective,
            name=problem_name,
            lb=lower,
            ub=upper,
        )
    except TypeError:
        problem = ioh.wrap_problem(
            objective,
            name=problem_name,
            optimization_type=ioh.OptimizationType.MIN,
            problem_type=ioh.ProblemType.REAL,
            lb=lower,
            ub=upper,
        )

    return problem


# ============================================================
# modDE runner copied/adapted from config.py
# ============================================================


def run_de(func, config, budget, dim, *args, **kwargs):
    lam = config.get("lambda_")
    if config.get("lambda_") == "nan":
        lam = None
    else:
        lam = int(config.get("lambda_")) * dim

    mut = config.get("mutation_reference")
    if config.get("mutation_reference") == "nan":
        mut = None

    archive = config.get("use_archive")
    if config.get("use_archive") == "False":
        archive = False
    elif config.get("use_archive") == "True":
        archive = True

    lpsr = config.get("lpsr")
    if config.get("lpsr") == "False":
        lpsr = False
    elif config.get("lpsr") == "True":
        lpsr = True

    cross = config.get("crossover")
    if config.get("crossover") == "nan":
        cross = None

    adaptation_method = config.get("adaptation_method")
    if config.get("adaptation_method") == "nan":
        adaptation_method = None

    item = {
        "F": np.array([float(config.get("F"))]),
        "CR": np.array([float(config.get("CR"))]),
        "lambda_": lam,
        "mutation_base": config.get("mutation_base"),
        "mutation_reference": mut,
        "mutation_n_comps": int(config.get("mutation_n_comps")),
        "use_archive": archive,
        "crossover": cross,
        "adaptation_method_F": adaptation_method,
        "adaptation_method_CR": adaptation_method,
        "lpsr": lpsr,
    }
    item["budget"] = int(budget)

    c = ModularDE(func, **item)

    try:
        c.run()
        return []
    except Exception:
        print("ModularDE run failed with config:")
        print(item)
        traceback.print_exc()
        return []


# ============================================================
# Experiment execution
# ============================================================


def config_to_dict(config):
    if hasattr(config, "get_dictionary"):
        d = config.get_dictionary()
    else:
        d = dict(config)

    clean = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        clean[k] = v

    return clean


def sample_configurations(configs_file):
    de_cs.seed(SEED)
    configs = de_cs.sample_configuration(SAMPLE_SIZE)

    if not isinstance(configs, list):
        configs = [configs]

    rows = []
    for i, conf in enumerate(configs):
        d = config_to_dict(conf)
        d["config_index"] = i
        rows.append(d)

    df_configs = pd.DataFrame(rows)
    df_configs.to_csv(configs_file, index=False)

    return rows


def run_one_config(task):
    config_index, config = task

    rows = []

    full_id = CURRENT_CONTEXT["full_id"]
    fid_new = CURRENT_CONTEXT["fid_new"]

    for seed in range(REPS):
        np.random.seed(seed)

        problem_name = f"real_generated_{full_id}_cfg{config_index}_seed{seed}"
        func = make_ioh_problem(DIM, problem_name)

        logger = aoc_logger(
            BUDGET,
            upper1=1e2,
            upper2=1e8,
            triggers=[ioh.logger.trigger.ALWAYS],
        )
        func.attach_logger(logger)

        try:
            run_de(func, config, budget=BUDGET, dim=DIM, seed=seed)
            auc1, auc2 = correct_aoc(func, logger, BUDGET)

            row = {
                "function_tag": CURRENT_CONTEXT["function_tag"],
                "full_id": full_id,
                "fid": fid_new,
                "iid": IID_NEW,
                "dim": DIM,
                "seed": seed,
                **config,
                "auc": auc1,
                "aucLarge": auc2,
                "auc_list": logger.auc1_list,
                "aucLarge_list": logger.auc2_list,
            }
            rows.append(row)

        except Exception as e:
            row = {
                "function_tag": CURRENT_CONTEXT["function_tag"],
                "full_id": full_id,
                "fid": fid_new,
                "iid": IID_NEW,
                "dim": DIM,
                "seed": seed,
                **config,
                "auc": np.nan,
                "aucLarge": np.nan,
                "auc_list": [],
                "aucLarge_list": [],
                "error": repr(e),
            }
            rows.append(row)
            traceback.print_exc()

        finally:
            try:
                func.reset()
                logger.reset(func)
            except Exception:
                pass

    return rows


def append_checkpoint(rows, checkpoint_file):
    df = pd.DataFrame(rows)
    file_exists = checkpoint_file.exists()
    df.to_csv(checkpoint_file, mode="a", header=not file_exists, index=False)


def process_results(df_raw):
    df = df_raw.copy()

    for col in ["mutation_reference", "adaptation_method", "lambda_"]:
        if col in df.columns:
            df[col] = df[col].replace(np.nan, "nan")

    for col in ["crossover", "mutation_base", "mutation_reference"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def run_for_one_function(function_tag, fid_new):
    full_id = f"{BASE_RUN_ID}_{function_tag}"
    function_file = get_function_file(full_id)
    paths = get_output_paths(full_id)

    set_context(
        {
            "function_tag": function_tag,
            "full_id": full_id,
            "function_file": function_file,
            "fid_new": fid_new,
        }
    )

    print("\n" + "=" * 80)
    print(f"RUNNING H0 FOR {function_tag} ({full_id})")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Real function file: {function_file}")
    print(f"Full ID: {full_id}")
    print(f"DIM: {DIM}")
    print(f"FID_NEW: {fid_new}")
    print(f"IID_NEW: {IID_NEW}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Repeats: {REPS}")
    print(f"Budget: {BUDGET}")
    print(f"Parallel: {PARALLEL}")
    print(f"Workers: {N_WORKERS}")
    print(f"Checkpoint: {paths['checkpoint']}")

    if not function_file.exists():
        raise FileNotFoundError(f"Function file not found: {function_file}")

    # Remove previous checkpoint for this exact function to avoid accidental append.
    if paths["checkpoint"].exists():
        print(f"[WARN] Removing existing checkpoint: {paths['checkpoint']}")
        paths["checkpoint"].unlink()

    print("\nSmoke test IOH wrapping...")
    test_func = make_ioh_problem(DIM, f"real_generated_{full_id}_smoke_test")
    y0 = test_func(np.zeros(DIM))
    print(f"f(0) via IOH wrapper = {y0}")
    test_func.reset()

    configs = sample_configurations(paths["configs"])
    print(f"Sampled configs saved to: {paths['configs']}")

    end_index = len(configs) if STOP_AFTER is None else min(len(configs), START_INDEX + STOP_AFTER)
    selected = configs[START_INDEX:end_index]
    tasks = [(START_INDEX + i, conf) for i, conf in enumerate(selected)]

    print(f"Running config indices [{START_INDEX}, {end_index})")

    all_rows = []

    if PARALLEL:
        with Pool(processes=min(N_WORKERS, len(tasks)), initializer=set_context, initargs=(CURRENT_CONTEXT,)) as pool:
            for rows in tqdm(pool.imap_unordered(run_one_config, tasks), total=len(tasks)):
                flat_rows = list(rows)
                append_checkpoint(flat_rows, paths["checkpoint"])
                all_rows.extend(flat_rows)
    else:
        for task in tqdm(tasks):
            rows = run_one_config(task)
            append_checkpoint(rows, paths["checkpoint"])
            all_rows.extend(rows)

    if paths["checkpoint"].exists():
        df_raw = pd.read_csv(paths["checkpoint"])
    else:
        df_raw = pd.DataFrame(all_rows)

    unnamed_cols = [c for c in df_raw.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df_raw = df_raw.drop(columns=unnamed_cols)

    df_raw.to_pickle(paths["raw_pkl"])

    df_processed = process_results(df_raw)
    df_processed.to_pickle(paths["processed_pkl"])

    summary = {
        "task": "new_function_task_H0_run_modde_on_real_generated_function_batch",
        "base_run_id": BASE_RUN_ID,
        "function_tag": function_tag,
        "full_id": full_id,
        "real_function_file": str(function_file),
        "raw_pkl_file": str(paths["raw_pkl"]),
        "processed_pkl_file": str(paths["processed_pkl"]),
        "checkpoint_file": str(paths["checkpoint"]),
        "configs_file": str(paths["configs"]),
        "dim": DIM,
        "fid_new": fid_new,
        "iid_new": IID_NEW,
        "sample_size": SAMPLE_SIZE,
        "reps": REPS,
        "budget": BUDGET,
        "n_rows_raw": int(len(df_raw)),
        "n_rows_processed": int(len(df_processed)),
        "n_auc_nan": int(df_processed["auc"].isna().sum()) if "auc" in df_processed.columns else None,
        "next_step": "Run Task H individual SHAP plotting script on the processed pkl.",
    }

    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved raw pkl:", paths["raw_pkl"])
    print("Saved processed pkl:", paths["processed_pkl"])
    print("Saved summary:", paths["summary"])
    print("Data shape:", df_processed.shape)
    print(f"Done for {function_tag}.")

    return summary


def main():
    print("=" * 80)
    print("NEW FUNCTION TASK H0: BATCH RUN modDE ON SELECTED REAL GENERATED FUNCTIONS")
    print("=" * 80)
    print(f"BASE_RUN_ID: {BASE_RUN_ID}")
    print(f"TARGET_FUNCTION_TAGS: {TARGET_FUNCTION_TAGS}")
    print(f"Full setting: sample_size={SAMPLE_SIZE}, reps={REPS}, budget={BUDGET}")
    print(f"Parallel: {PARALLEL}, workers={N_WORKERS}")

    summaries = []

    for idx, function_tag in enumerate(TARGET_FUNCTION_TAGS):
        fid_new = FID_BASE + idx
        summary = run_for_one_function(function_tag=function_tag, fid_new=fid_new)
        summaries.append(summary)

    batch_summary_file = OUT_DIR / f"real_generated_function_{BASE_RUN_ID}_{'_'.join(TARGET_FUNCTION_TAGS)}_modde_batch_summary.json"
    batch_summary_file.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BATCH H0 DONE")
    print("=" * 80)
    print(f"Batch summary saved to: {batch_summary_file}")

    for s in summaries:
        print(f"{s['function_tag']} -> {s['processed_pkl_file']}")


if __name__ == "__main__":
    main()