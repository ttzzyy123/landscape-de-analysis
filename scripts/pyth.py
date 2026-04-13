import pandas as pd

pkl_path = "/data/s3795888/ioh_project/my_landscape_experiments/data/de_final_5_processed.pkl"

df = pd.read_pickle(pkl_path)

print("TYPE:")
print(type(df))

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nSHAPE:")
print(df.shape)

print("\nHEAD:")
print(df.head())