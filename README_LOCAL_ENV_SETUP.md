# Local Environment Setup Guide

This folder contains the local backup of the LLaMEA paper experiments and related landscape-analysis files. The original server setup used two layers of environment activation:

```bash
source /software/anaconda3/etc/profile.d/conda.sh
conda activate llamea_311
source external/LLaMEA-paper-ela/XAI-liacs-LLaMEA-6d8b3c1/.venv/bin/activate
```

On Windows, the equivalent setup is described below.

## 1. Main Landscape Analysis Environment: `ioh_new`

Use this environment for the main `landscape-de-analysis-main` scripts, including SHAP analysis, Step 8B, Step 10D, and table generation.

Create the environment:

```bat
E:\anacoda\condabin\conda.bat create -n ioh_new python=3.10 -y
```

Install dependencies:

```bat
E:\anacoda\envs\ioh_new\python.exe -m pip install --upgrade pip setuptools wheel
E:\anacoda\envs\ioh_new\python.exe -m pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm joblib shap catboost xgboost lightgbm statsmodels ioh iohxplainer ConfigSpace jupyter ipykernel tabulate openpyxl pyarrow
```

Register the Jupyter kernel:

```bat
E:\anacoda\envs\ioh_new\python.exe -m ipykernel install --sys-prefix --name ioh_new --display-name "Python (ioh_new)"
```

Validate:

```bat
E:\anacoda\envs\ioh_new\python.exe -c "import numpy, pandas, scipy, sklearn, shap, catboost, xgboost, lightgbm, statsmodels, ioh, iohxplainer, ConfigSpace; print('ioh_new imports ok')"
```

Use it from Anaconda Prompt:

```bat
conda activate ioh_new
cd /d "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main"
```

Or use it directly from PowerShell/cmd:

```bat
E:\anacoda\envs\ioh_new\python.exe scripts\step10d_generate_final_supervisor_tables_0307.py
```

## 2. LLaMEA Environment: `llamea_311` plus Project `.venv`

The LLaMEA repository requires Python 3.11 and uses `uv` according to its README.

Repository path:

```text
E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main\external\LLaMEA-paper-ela\XAI-liacs-LLaMEA-6d8b3c1
```

Create the Conda base environment:

```bat
E:\anacoda\condabin\conda.bat create -n llamea_311 python=3.11 -y
E:\anacoda\envs\llamea_311\python.exe -m pip install --upgrade pip setuptools wheel uv
```

Install the LLaMEA project dependencies from the repository root:

```bat
cd /d "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main\external\LLaMEA-paper-ela\XAI-liacs-LLaMEA-6d8b3c1"
E:\anacoda\envs\llamea_311\python.exe -m uv sync --dev --group examples
```

This creates the project virtual environment here:

```text
XAI-liacs-LLaMEA-6d8b3c1\.venv
```

## 3. Important Windows Path-Length Fix

The full local path is very deep. Some Python imports fail on Windows because of the traditional path-length limit. To avoid this, map the LLaMEA repository to a short temporary drive letter before running it:

```bat
subst L: "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main\external\LLaMEA-paper-ela\XAI-liacs-LLaMEA-6d8b3c1"
```

Then use the short path:

```bat
L:\.venv\Scripts\activate
```

Or run Python directly:

```bat
L:\.venv\Scripts\python.exe L:\ELA.py
```

Validate the LLaMEA environment:

```bat
L:\.venv\Scripts\python.exe -c "import llamea, numpy, pandas, openai, google.generativeai, ConfigSpace, ioh, sklearn, xgboost, pflacco; print('LLaMEA imports ok')"
```

If you want to remove the temporary drive mapping later:

```bat
subst L: /D
```

## 4. Basins Attribution Dependencies

The folder `basinsattribution-main` has its own `requirements.txt`:

```text
matplotlib==3.9.2
numpy==1.26.4
pybind11==2.13.6
torch==2.8.0
jax==0.7.2
```

However, this file has a dependency conflict:

- LLaMEA requires `numpy < 2`.
- `jax==0.7.2` requires `numpy >= 2.0`.

The working local solution is to keep `numpy==1.26.4` and install a compatible JAX version:

```bat
subst L: "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main\external\LLaMEA-paper-ela\XAI-liacs-LLaMEA-6d8b3c1"
E:\anacoda\envs\llamea_311\python.exe -m uv pip install --python L:\.venv\Scripts\python.exe matplotlib==3.9.2 numpy==1.26.4 pybind11==2.13.6 torch==2.8.0 jax==0.4.35
```

Validate basins attribution imports:

```bat
cd /d L:\basinsattribution-main
L:\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); import utils; import process_basins; import analyze_basins; import torch, jax; print('basins imports ok')"
```

Expected versions from the tested local setup:

```text
numpy 1.26.4
torch 2.8.0+cpu
jax 0.4.35
```

## 5. Quick Usage Summary

For main thesis SHAP/table scripts:

```bat
conda activate ioh_new
cd /d "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main"
python scripts\step10d_generate_final_supervisor_tables_0307.py
```

For LLaMEA / ELA experiments:

```bat
subst L: "E:\DS AI\6\Thesis\server_full_backup_20260624\landscape-de-analysis-main\landscape-de-analysis-main\external\LLaMEA-paper-ela\XAI-liacs-LLaMEA-6d8b3c1"
L:\.venv\Scripts\activate
python L:\ELA.py
```

## 6. Notes

- The `ioh_new` environment is for the main landscape-analysis scripts.
- The `llamea_311` Conda environment is mainly used to provide Python 3.11 and `uv`.
- The actual LLaMEA packages live in the project `.venv` created by `uv sync`.
- Use the `L:` short-path mapping on Windows to avoid import errors caused by very long paths.
- API-based LLaMEA runs require environment variables such as `OPENAI_API_KEY` or Google/Gemini credentials, depending on which LLM backend is used.
