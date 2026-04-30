from pathlib import Path

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "de_final_5_processed.pkl"
GROUP_DATA_DIR = PROJECT_ROOT / "intermediate" / "group_data_eps2bins_3_r025065"

OUTPUT_DIR = PROJECT_ROOT / "output" / "manual_bins_shap_beeswarm_encoding"
FUNCTION_OUT_DIR = OUTPUT_DIR / "functions"
GROUP_OUT_DIR = OUTPUT_DIR / "groups"

FUNCTION_OUT_DIR.mkdir(parents=True, exist_ok=True)
GROUP_OUT_DIR.mkdir(parents=True, exist_ok=True)

FUNCTION_SAMPLE_SIZE = 10000
GROUP_SAMPLE_SIZE = 20000


def prepare_features(df: pd.DataFrame):
    """
    Encoding version:
    - 数值列直接保留
    - 类别列使用 categorical encoding，不使用 one-hot
    - 加入 iid 和 seed，命名为 Instance variance / Stochastic variance
    - 删除常量列
    """
    df = df.copy()

    numeric_features = [
        "CR",
        "F",
        "lambda_",
        "lpsr",
        "mutation_n_comps",
        "use_archive",
    ]

    categorical_features = [
        "mutation_base",
        "mutation_reference",
        "crossover",
        "adaptation_method",
    ]

    variance_features = [
        "iid",
        "seed",
    ]

    target_col = "auc"

    required_cols = (
        numeric_features
        + categorical_features
        + variance_features
        + [target_col]
    )

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[
        numeric_features
        + categorical_features
        + variance_features
    ].copy()

    X = X.rename(
        columns={
            "iid": "Instance variance",
            "seed": "Stochastic variance",
        }
    )

    for col in categorical_features:
        X[col] = pd.Categorical(X[col]).codes

    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]

    if constant_cols:
        print(f"Dropping constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    y = df[target_col].copy()

    return X, y


def train_and_plot_beeswarm(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    out_file: Path,
    sample_size: int,
):
    if len(df) > sample_size:
        print(f"Sampling from {len(df)} -> {sample_size}")
        df = df.sample(sample_size, random_state=42)

    X, y = prepare_features(df)

    print(f"Training model: {title}")
    print("X shape:", X.shape)
    print("Features:", list(X.columns))

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X)

    order = list(range(len(X.columns)))

    shap.plots.beeswarm(
        shap_explanation,
        show=False,
        order=order,
        max_display=20,
        color=plt.get_cmap("viridis"),
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()

    print(f"Saved: {out_file}")


def plot_function_level():
    print("\n==============================")
    print("Function-level SHAP beeswarm")
    print("==============================")

    df = pd.read_pickle(DATA_FILE)

    if "dim" in df.columns:
        df = df[df["dim"] == 5].copy()

    functions = sorted(df["fid"].dropna().unique().tolist())

    for fid in functions:
        print(f"\nProcessing function f{fid}")

        sub = df[df["fid"] == fid].copy()

        out_file = FUNCTION_OUT_DIR / f"summary_f{fid}_d5.png"

        train_and_plot_beeswarm(
            df=sub,
            title=f"$f_{{{fid}}}$ in $d=5$",
            xlabel=f"Hyper-parameter contributions on $f_{{{fid}}}$ in $d=5$",
            out_file=out_file,
            sample_size=FUNCTION_SAMPLE_SIZE,
        )


def discover_group_files(group_dir: Path):
    files = sorted(group_dir.glob("de_final_5_group_*.pkl"))
    pairs = []

    for f in files:
        try:
            group_id = int(f.stem.split("_")[-1])
            pairs.append((group_id, f))
        except ValueError:
            continue

    return pairs


def plot_group_level():
    print("\n==========================")
    print("Group-level SHAP beeswarm")
    print("==========================")

    group_files = discover_group_files(GROUP_DATA_DIR)

    if not group_files:
        raise FileNotFoundError(f"No group files found in {GROUP_DATA_DIR}")

    for group_id, file in group_files:
        print(f"\nProcessing Group {group_id}")
        print("File:", file)

        df = pd.read_pickle(file)

        group_label = None
        if "group_label" in df.columns and df["group_label"].notna().any():
            group_label = df["group_label"].dropna().iloc[0]

        if group_label is None:
            title = f"Group {group_id} in $d=5$"
        else:
            title = f"Group {group_id}: {group_label}"

        out_file = GROUP_OUT_DIR / f"summary_group_{group_id}_d5.png"

        train_and_plot_beeswarm(
            df=df,
            title=title,
            xlabel=f"Hyper-parameter contributions on Group {group_id} in $d=5$",
            out_file=out_file,
            sample_size=GROUP_SAMPLE_SIZE,
        )


def main():
    plot_function_level()
    plot_group_level()

    print("\nDone.")
    print("Function plots:", FUNCTION_OUT_DIR)
    print("Group plots:", GROUP_OUT_DIR)


if __name__ == "__main__":
    main()