import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------
# CONFIG
# -------------------------------------------
synthetic_df = pd.read_csv("../data/synthetic_dataset.csv")
FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]
iso1 = FEATURES[0]
iso2 = FEATURES[1]

# -------------------------------------------
# Scatter Plot
# -------------------------------------------
# plt.figure()
# plt.scatter(synthetic_df[iso1], synthetic_df[iso2])
# plt.xlabel(iso1)
# plt.ylabel(iso2)
# plt.title("Synthetic Data: First Two Isotopes")
# plt.savefig("../output/synthetic_plot.png")

plt.figure()

colors = {
    "herbivorous": "green",
    "piscivorous": "blue",
    "ibi": "orange"
}

for cluster, color in colors.items():
    subset = synthetic_df[synthetic_df["cluster"] == cluster]
    plt.scatter(subset[iso1], subset[iso2], label=cluster, color=color, alpha=0.85, s=20)

plt.xlabel(iso1)
plt.ylabel(iso2)
plt.title("Synthetic Data: First Two Isotopes")
plt.legend()

plt.savefig("../output/synthetic_plot.png")

# -------------------------------------------
# Statistical Analysis
# -------------------------------------------
# analysis_rows = []
# for feature in FEATURES:
#     data = synthetic_df[feature].values
#     skew = stats.skew(data)
#     kurt = stats.kurtosis(data)  # Fisher definition (0 = normal)

#     analysis_rows.append({
#         "feature": feature,
#         "mean": np.mean(data),
#         "std": np.std(data),
#         "skewness": skew,
#         "kurtosis": kurt
#     })

#     # QQ Plot
#     plt.figure()
#     stats.probplot(data, dist="norm", plot=plt)
#     plt.title(f"QQ-Plot: {feature}")
#     plt.savefig(f"../output/qqplot_{feature}.png", dpi=200, bbox_inches="tight")
#     plt.close()