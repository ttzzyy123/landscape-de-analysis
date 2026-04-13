import pandas as pd
import matplotlib.pyplot as plt

base = "output"

for g in [0, 1, 2]:
    df = pd.read_csv(f"{base}/shap_group_{g}.csv")

    plt.figure()
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"SHAP Importance - Group {g}")
    plt.xlabel("Importance")

    plt.savefig(f"{base}/plot_group_{g}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    
    
    import pandas as pd
import matplotlib.pyplot as plt

base = "output"

for g in [0, 1, 2]:
    df_bin = pd.read_csv(f"{base}/shap_bin_group_{g}.csv")
    df_exp = pd.read_csv(f"{base}/shap_exp_group_{g}.csv")

    df = df_bin.merge(df_exp, on="feature", suffixes=("_bin", "_exp"))

    plt.figure()
    x = range(len(df))

    plt.barh(x, df["importance_bin"], alpha=0.7, label="bin")
    plt.barh(x, df["importance_exp"], alpha=0.7, label="exp")

    plt.yticks(x, df["feature"])
    plt.gca().invert_yaxis()
    plt.legend()

    plt.title(f"Bin vs Exp - Group {g}")
    plt.xlabel("Importance")

    plt.savefig(f"{base}/plot_bin_vs_exp_group_{g}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("output/shap_individual/shap_all_functions.csv")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="feature", y="importance")

plt.xticks(rotation=45)
plt.title("Individual Function SHAP Distribution")

plt.savefig("output/plot_individual_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()    




import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/grouping_sensitivity/function_level_features.csv")
groups = pd.read_csv("output/grouping_sensitivity/assignments_kmeans_k3.csv")

df = df.merge(groups[["Function", "group"]], on="Function")

plt.figure()

for g in [0, 1, 2]:
    sub = df[df["group"] == g]
    plt.scatter(sub["ic.eps.ratio"], sub["ela_meta.lin_simple.adj_r2"], label=f"Group {g}")

plt.xlabel("ic.eps.ratio")
plt.ylabel("adj_r2")
plt.legend()

plt.title("Landscape-based Grouping")

plt.savefig("output/plot_grouping_scatter.png", dpi=300, bbox_inches="tight")
plt.close()