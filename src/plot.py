import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
iso1 = FEATURES[2]
iso2 = FEATURES[3]

plt.figure()

colors = {
    "herbivorous": "green",
    "piscivorous": "blue",
    "ibi": "orange"
}

lower_limit_color = "black"

# -------------------------------------------
# Scatter plot
# -------------------------------------------
for cluster, color in colors.items():
    subset = synthetic_df[synthetic_df["cluster"] == cluster]
    plt.scatter(subset[iso1], subset[iso2], label=cluster, color=color, alpha=0.85, s=20)

# -------------------------------------------
# Lower Marine Limit
# -------------------------------------------
pisc = synthetic_df[synthetic_df["cluster"] == "piscivorous"]

lower_limit_iso1 = pisc[iso1].min()
lower_limit_iso2 = pisc[iso2].min()

# vertical axis (iso1)
plt.axvline(
    x=lower_limit_iso1,
    color=lower_limit_color,
    linestyle="--",
    linewidth=2,
    label="Lower marine limit"
)

# horizontal axis (iso2)
plt.axhline(
    y=lower_limit_iso2,
    color=lower_limit_color,
    linestyle="--",
    linewidth=2
)

# -------------------------------------------
# Labels
# -------------------------------------------
plt.xlabel(iso1)
plt.ylabel(iso2)
plt.title("Synthetic Data: First Two Isotopes with Lower Marine Limit")
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