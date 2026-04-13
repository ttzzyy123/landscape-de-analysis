import pandas as pd
import os

INPUT_FILE = "data/de_final_5_processed.pkl"
OUTPUT_DIR = "intermediate/modde_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading data...")
    df = pd.read_pickle(INPUT_FILE)

    print("Total shape:", df.shape)
    print("\nCrossover value counts:")
    print(df["crossover"].value_counts())

    # ===== 分离 =====
    df_bin = df[df["crossover"] == "bin"].copy()
    df_exp = df[df["crossover"] == "exp"].copy()

    print("\nBin shape:", df_bin.shape)
    print("Exp shape:", df_exp.shape)

    # ===== 保存 =====
    bin_path = os.path.join(OUTPUT_DIR, "de_final_5_bin.pkl")
    exp_path = os.path.join(OUTPUT_DIR, "de_final_5_exp.pkl")

    df_bin.to_pickle(bin_path)
    df_exp.to_pickle(exp_path)

    print("\nSaved:")
    print(bin_path)
    print(exp_path)


if __name__ == "__main__":
    main()