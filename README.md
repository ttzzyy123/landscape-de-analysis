# Landscape-Aware Explainability for Modular Differential Evolution

This repository contains the experimental code, processed outputs, and table-generation scripts used for a bachelor thesis on landscape-aware explainable benchmarking for Modular Differential Evolution (modDE).

The project investigates whether Exploratory Landscape Analysis (ELA) features can be used to group continuous black-box optimisation problems and explain modDE hyper-parameter importance beyond fixed benchmark-function identities.

## Project Overview

The main workflow is:

1. Extract or load ELA features for BBOB function instances.
2. Construct landscape groups from selected ELA features.
3. Train CatBoost surrogate models on modDE performance data.
4. Compute SHAP explanations for hyper-parameter importance.
5. Compare group-level, function-level, and instance-level explanation patterns.
6. Evaluate partial generalisation on generated unseen landscapes, including affine BBOB combinations and LLaMEA-generated functions.

The thesis focuses on two ELA features:

- epsilon ratio
- adjusted R2 of a linear model

These features are used to build the main six-group landscape grouping scheme and the finer instance-level grouping used in later analysis.

## Repository Structure

```text
catboost_info/       CatBoost training metadata and logs
data/                Processed input data used by the analysis scripts
external/            External packages and LLaMEA-related materials
intermediate/        Intermediate experiment outputs and generated-function results
output/              Final plots, tables, SHAP summaries, and comparison outputs
scripts/             Main analysis, plotting, SHAP, and table-generation scripts
```

Important generated outputs are stored under:

```text
output/0307_main_comparison/
output/step7_instance_level_group_and_partial_function_plots/
output/new_function_task/
```

## Key Scripts

The most relevant scripts are:

```text
scripts/step1_extract_features.py
scripts/step2_define_bins.py
scripts/step3_prepare_group_data_tiebreak.py
scripts/step4_shap_analysis_manual_bins_niles.py
scripts/step6_shap_individual_manual_bins_niles.py
scripts/step8b_export_missing_importance_and_similarity.py
scripts/step9_hall_of_fame_0307.py
scripts/step10d_generate_final_supervisor_tables_0307.py
scripts/run_table7_aligned_10000x3.py
```

`run_table7_aligned_10000x3.py` evaluates the generated functions using the same 10,000 modDE configurations and three stochastic repetitions as the BBOB experiments. This aligned setting is used for the final generated-function Hall-of-Fame and SHAP comparison tables.

## Generated Functions and API Keys

The scripts with the prefix `new_function_task_` are used for the generated-function part of the thesis. The LLaMEA-related generation scripts may require access to an external LLM API when new functions are created, especially:

```text
scripts/new_function_task_C_dryrun_generate_one_function.py
scripts/new_function_task_E_real_automatic_landscape_generation.py
```

API keys should be supplied through local environment variables or an ignored `.env` file. Do not commit API keys, tokens, or local credential files to Git. The repository `.gitignore` already excludes `*.env` and `secrets/`.

Example on Linux/macOS:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Example on Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

After generated functions have been created locally, the later analysis steps, including ELA feature computation, group assignment, modDE evaluation, SHAP analysis, and final table generation, can be rerun from the saved generated-function files without making new API calls.

## Reproducing Final Tables

After the required data files are available, the final supervisor-requested tables can be regenerated with:

```bash
python scripts/step10d_generate_final_supervisor_tables_0307.py
```

To regenerate SHAP outputs for generated functions from the aligned 10,000 x 3 setting:

```bash
STEP8B_USE_ALIGNED_GENERATED_MODDE=1 \
STEP8B_FORCE_REEXPORT_RAW=1 \
STEP8B_SAMPLE_SIZE=10000 \
python scripts/step8b_export_missing_importance_and_similarity.py
```

On Windows PowerShell, the same environment variables can be set with:

```powershell
$env:STEP8B_USE_ALIGNED_GENERATED_MODDE="1"
$env:STEP8B_FORCE_REEXPORT_RAW="1"
$env:STEP8B_SAMPLE_SIZE="10000"
python scripts\step8b_export_missing_importance_and_similarity.py
```

## Environment

The main analysis uses Python with packages including:

- numpy
- pandas
- scipy
- matplotlib
- seaborn
- catboost
- shap
- ioh
- iohxplainer
- ConfigSpace

Detailed local setup notes are available in:

```text
README_LOCAL_ENV_SETUP.md
```

## Data Bundle

Large raw and processed experiment files are intentionally not committed to Git, especially `.pkl` and `.csv` files generated during repeated experiments. These files are provided separately as a GitHub Release asset:

- Release page: [Thesis data bundle](https://github.com/ttzzyy123/landscape-de-analysis/releases/tag/data-v1.0)
- Main data archive: [csv_pkl_data_bundle_20260629.zip](https://github.com/ttzzyy123/landscape-de-analysis/releases/download/data-v1.0/csv_pkl_data_bundle_20260629.zip)
- Archive manifest: [csv_pkl_data_bundle_20260629_manifest.csv](https://github.com/ttzzyy123/landscape-de-analysis/releases/download/data-v1.0/csv_pkl_data_bundle_20260629_manifest.csv)
- Skipped-file log: [csv_pkl_data_bundle_20260629_skipped.csv](https://github.com/ttzzyy123/landscape-de-analysis/releases/download/data-v1.0/csv_pkl_data_bundle_20260629_skipped.csv)

The archive contains the local `.csv` and `.pkl` files used for the thesis experiments, including processed BBOB/modDE data, generated-function aligned `10000 x 3` results, SHAP intermediate outputs, and final CSV/PKL experiment outputs. The manifest lists the relative paths and file sizes of the archived files.

To restore the data locally, download and extract the archive into the project parent or the same directory structure used during the experiments. The relative paths inside the archive preserve the original experiment-folder layout.

## Thesis Context

This code supports a thesis on:

> Landscape-aware explainability for Modular Differential Evolution on continuous black-box optimisation problems.

The main claim tested in the experiments is that ELA-based landscape grouping can provide transferable behavioural references for explaining modDE hyper-parameter importance, while still requiring function-specific analysis for exact configuration selection and performance prediction.
