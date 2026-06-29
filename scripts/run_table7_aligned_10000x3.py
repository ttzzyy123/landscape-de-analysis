from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import importlib.util
import inspect
import json
import os
import traceback

import ioh
import numpy as np
import pandas as pd
from modde import ModularDE
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIM = 5
REPS = 3
BUDGET = 10000

DATA_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"
AFFINE_FUNCTION_FILE = PROJECT_ROOT / "data" / "affine_functions_tony.py"
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
    / "modde_table7_aligned_10000x3_results"
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

AFFINE_TARGETS = [
    {"function_type": "affine", "function_id": "f10_f6", "fid1": 10, "fid2": 6, "alpha": 0.9, "iid1": 1, "iid2": 1},
    {"function_type": "affine", "function_id": "f15_f8", "fid1": 15, "fid2": 8, "alpha": 0.9, "iid1": 1, "iid2": 1},
    {"function_type": "affine", "function_id": "f1_f23", "fid1": 1, "fid2": 23, "alpha": 0.9, "iid1": 1, "iid2": 1},
    {"function_type": "affine", "function_id": "f3_f6", "fid1": 3, "fid2": 6, "alpha": 0.9, "iid1": 1, "iid2": 1},
    {"function_type": "affine", "function_id": "f9_f12", "fid1": 9, "fid2": 12, "alpha": 0.9, "iid1": 1, "iid2": 1},
]

LLAMEA_TARGETS = [
    {"function_type": "llamea", "function_id": "20260516_115256"},
    {"function_type": "llamea", "function_id": "20260517_125137_n3"},
    {"function_type": "llamea", "function_id": "20260517_125137_n5"},
    {"function_type": "llamea", "function_id": "20260517_125137_n7"},
]

CURRENT_CONTEXT = {}


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


def set_context(context):
    global CURRENT_CONTEXT
    CURRENT_CONTEXT = context


def load_python_module(path):
    module_name = f"loaded_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_affine_callable(target):
    module = load_python_module(AFFINE_FUNCTION_FILE)
    if not hasattr(module, "get_bbob_function"):
        raise RuntimeError(f"{AFFINE_FUNCTION_FILE} does not define get_bbob_function().")
    return module.get_bbob_function(
        fid1=target["fid1"],
        iid1=target["iid1"],
        fid2=target["fid2"],
        iid2=target["iid2"],
        dim=DIM,
        alpha=target["alpha"],
    )


def instantiate_llamea_function(function_id):
    path = REAL_FUNCTION_DIR / f"real_generated_function_{function_id}.py"
    if not path.exists():
        raise FileNotFoundError(f"LLaMEA function file not found: {path}")

    module = load_python_module(path)
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and hasattr(obj, "f"):
            candidates.append(obj)

    if not candidates:
        raise RuntimeError(f"No generated class with an f(x) method found in {path}")

    cls = candidates[0]
    try:
        return cls(dim=DIM)
    except TypeError:
        try:
            return cls(DIM)
        except TypeError:
            return cls()


def make_ioh_problem(target, problem_name):
    if target["function_type"] == "affine":
        affine_func = load_affine_callable(target)

        def objective(x):
            return float(affine_func(np.asarray(x, dtype=float)))

    elif target["function_type"] == "llamea":
        generated = instantiate_llamea_function(target["function_id"])

        def objective(x):
            return float(generated.f(np.asarray(x, dtype=float)))

    else:
        raise ValueError(f"Unknown function_type: {target['function_type']}")

    try:
        return ioh.wrap_problem(
            objective,
            name=problem_name,
            lb=-5.0,
            ub=5.0,
        )
    except TypeError:
        return ioh.wrap_problem(
            objective,
            name=problem_name,
            optimization_type=ioh.OptimizationType.MIN,
            problem_type=ioh.ProblemType.REAL,
            lb=-5.0,
            ub=5.0,
        )


def normalize_config_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return "nan"
    return value


def load_bbob_configurations(config_file):
    if config_file.exists():
        df_configs = pd.read_csv(config_file)
    else:
        print(f"Loading BBOB experiment data from: {DATA_FILE}")
        df = pd.read_pickle(DATA_FILE)
        missing = [c for c in DE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing configuration columns in {DATA_FILE}: {missing}")

        df_configs = df[DE_COLUMNS].drop_duplicates().reset_index(drop=True)
        df_configs.insert(0, "config_index", np.arange(len(df_configs)))
        config_file.parent.mkdir(parents=True, exist_ok=True)
        df_configs.to_csv(config_file, index=False)

    if len(df_configs) != 10000:
        raise RuntimeError(f"Expected 10000 BBOB configurations, found {len(df_configs)}")

    rows = []
    for _, row in df_configs.iterrows():
        config = {
            col: normalize_config_value(row[col])
            for col in DE_COLUMNS
        }
        rows.append((int(row["config_index"]), config))

    return rows


def run_de(func, config, budget, dim):
    lam = config.get("lambda_")
    if str(lam) == "nan":
        lam = None
    else:
        lam = int(lam) * dim

    mut = config.get("mutation_reference")
    if str(mut) == "nan":
        mut = None

    archive = config.get("use_archive")
    if str(archive) == "False":
        archive = False
    elif str(archive) == "True":
        archive = True

    lpsr = config.get("lpsr")
    if str(lpsr) == "False":
        lpsr = False
    elif str(lpsr) == "True":
        lpsr = True

    cross = config.get("crossover")
    if str(cross) == "nan":
        cross = None

    adaptation_method = config.get("adaptation_method")
    if str(adaptation_method) == "nan":
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
        "budget": int(budget),
    }

    runner = ModularDE(func, **item)
    runner.run()


def run_one_config(task):
    config_index, config = task
    target = CURRENT_CONTEXT["target"]
    rows = []

    for seed in range(REPS):
        np.random.seed(seed)
        problem_name = f"table7_aligned_{target['function_id']}_cfg{config_index}_seed{seed}"
        func = make_ioh_problem(target, problem_name)
        logger = aoc_logger(
            BUDGET,
            upper1=1e2,
            upper2=1e8,
            triggers=[ioh.logger.trigger.ALWAYS],
        )
        func.attach_logger(logger)

        row = {
            "function_type": target["function_type"],
            "function_id": target["function_id"],
            "dim": DIM,
            "seed": seed,
            "config_index": config_index,
            **target,
            **config,
        }

        try:
            run_de(func, config, budget=BUDGET, dim=DIM)
            auc1, auc2 = correct_aoc(func, logger, BUDGET)
            row.update(
                {
                    "auc": auc1,
                    "aucLarge": auc2,
                    "auc_list": logger.auc1_list,
                    "aucLarge_list": logger.auc2_list,
                    "error": "",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "auc": np.nan,
                    "aucLarge": np.nan,
                    "auc_list": [],
                    "aucLarge_list": [],
                    "error": repr(exc),
                }
            )
            traceback.print_exc()
        finally:
            try:
                func.reset()
                logger.reset(func)
            except Exception:
                pass

        rows.append(row)

    return rows


def output_paths(function_id):
    safe_id = function_id.replace(".", "p")
    prefix = f"table7_aligned_{safe_id}_10000x3"
    return {
        "checkpoint": OUT_DIR / f"{prefix}_checkpoint.csv",
        "raw_pkl": OUT_DIR / f"{prefix}_raw.pkl",
        "processed_pkl": OUT_DIR / f"{prefix}_processed.pkl",
        "summary": OUT_DIR / f"{prefix}_summary.json",
    }


def completed_config_indices(checkpoint):
    if not checkpoint.exists():
        return set()
    df = pd.read_csv(checkpoint, usecols=["config_index", "seed"])
    counts = df.drop_duplicates(["config_index", "seed"]).groupby("config_index")["seed"].nunique()
    return set(counts[counts >= REPS].index.astype(int).tolist())


def append_checkpoint(rows, checkpoint):
    df = pd.DataFrame(rows)
    file_exists = checkpoint.exists()
    df.to_csv(checkpoint, mode="a", header=not file_exists, index=False)


def process_checkpoint(checkpoint):
    df = pd.read_csv(checkpoint)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df = df.drop_duplicates(["function_id", "config_index", "seed"], keep="last")
    return df


def run_target(target, configs, workers, force=False, stop_after=None):
    paths = output_paths(target["function_id"])
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    if force:
        for path in paths.values():
            if path.exists():
                path.unlink()

    done = completed_config_indices(paths["checkpoint"])
    tasks = [(idx, conf) for idx, conf in configs if idx not in done]
    if stop_after is not None:
        tasks = tasks[:stop_after]

    context = {"target": target}
    set_context(context)

    print("\n" + "=" * 80)
    print(f"Running {target['function_type']} {target['function_id']}")
    print(f"Already complete configs: {len(done)}")
    print(f"Remaining configs this run: {len(tasks)}")
    print(f"Checkpoint: {paths['checkpoint']}")
    print("=" * 80)

    if tasks:
        if workers > 1:
            with Pool(processes=workers, initializer=set_context, initargs=(context,)) as pool:
                for rows in tqdm(pool.imap_unordered(run_one_config, tasks), total=len(tasks)):
                    append_checkpoint(rows, paths["checkpoint"])
        else:
            for task in tqdm(tasks):
                rows = run_one_config(task)
                append_checkpoint(rows, paths["checkpoint"])

    df = process_checkpoint(paths["checkpoint"])
    df.to_pickle(paths["raw_pkl"])
    df.to_pickle(paths["processed_pkl"])

    summary = {
        "task": "table7_aligned_generated_function_modde_evaluation",
        "function_type": target["function_type"],
        "function_id": target["function_id"],
        "dim": DIM,
        "budget": BUDGET,
        "reps": REPS,
        "configuration_source": str(DATA_FILE),
        "configuration_count": int(df["config_index"].nunique()),
        "expected_configuration_count": 10000,
        "n_rows": int(len(df)),
        "expected_n_rows": 10000 * REPS,
        "n_auc_nan": int(df["auc"].isna().sum()) if "auc" in df.columns else None,
        "checkpoint": str(paths["checkpoint"]),
        "raw_pkl": str(paths["raw_pkl"]),
        "processed_pkl": str(paths["processed_pkl"]),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved processed pkl: {paths['processed_pkl']}")
    print(f"Rows: {summary['n_rows']}, configs: {summary['configuration_count']}, NaN AUC: {summary['n_auc_nan']}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Table 7 affine and LLaMEA functions using the exact 10000 BBOB modDE configurations."
    )
    parser.add_argument("--only", nargs="*", default=None, help="Optional function ids to run, e.g. f15_f8 20260517_125137_n3")
    parser.add_argument("--workers", type=int, default=max(1, min(cpu_count(), 16)))
    parser.add_argument("--force", action="store_true", help="Delete existing aligned outputs before running.")
    parser.add_argument("--stop-after", type=int, default=None, help="Debug option: run only this many remaining configs per function.")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_file = OUT_DIR / "bbob_de_final_5_unique_10000_configs.csv"
    configs = load_bbob_configurations(config_file)
    print(f"Loaded {len(configs)} aligned BBOB configurations from {config_file}")

    targets = AFFINE_TARGETS + LLAMEA_TARGETS
    if args.only:
        wanted = set(args.only)
        targets = [target for target in targets if target["function_id"] in wanted]
        missing = wanted - {target["function_id"] for target in targets}
        if missing:
            raise ValueError(f"Unknown --only function ids: {sorted(missing)}")

    summaries = []
    for target in targets:
        summaries.append(
            run_target(
                target=target,
                configs=configs,
                workers=args.workers,
                force=args.force,
                stop_after=args.stop_after,
            )
        )

    batch_summary = OUT_DIR / "table7_aligned_10000x3_batch_summary.json"
    batch_summary.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nBatch summary saved to: {batch_summary}")


if __name__ == "__main__":
    main()
