"""
Synthetic dataset generation

- Define reference clusters
- For each cluster and each feature:
    mu  = mean(real_data)
    sigma ≈ (max - min) / 4   (range/4 heuristic)
- Sample synthetic points from independent Gaussian distributions:
    x ~ N(mu, sigma^2) per feature
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_XLSX = "../data/terrherbmarin.xlsx"
SHEET_NAME = 0
NROWS = 88                 # 88, since the last rows are only notes.
SEED = 0
FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]
SIZE = 200

# -------------------------------------------------
# Manual adjustment of mean and std per cluster & feature
# Factors multiply existing sigma, shifts are added to meanA
# -------------------------------------------------
MANUAL_ADJUSTMENTS = {
    "herbivorous": {
        "δ13C coll": {"mean_shift": 0.0, "sigma_factor": 0.8},
        "δ15N coll": {"mean_shift": -1.0, "sigma_factor": 0.8},
        "δ13C carb": {"mean_shift": -3.0, "sigma_factor": 0.6},
        "δ18O carb": {"mean_shift": -4.0, "sigma_factor": 0.6},
    },
    "piscivorous": {
        "δ13C coll": {"mean_shift": 0.0, "sigma_factor": 0.8},
        "δ15N coll": {"mean_shift": 1.0, "sigma_factor": 0.8},
        "δ13C carb": {"mean_shift": 3.0, "sigma_factor": 0.6},
        "δ18O carb": {"mean_shift": 4.0, "sigma_factor": 0.6},
    }
}

# IBI parameters
IBI_SIZE = 20
IBI_STD_FACTOR = 1.0

# -------------------------------------------------
# 1) Load data
# -------------------------------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl", sheet_name=SHEET_NAME, nrows=NROWS)

# Normalize strings
if "diet" in df.columns:
    df["diet"] = df["diet"].astype(str).str.lower().str.strip()

# -------------------------------------------------
# 2) Preprocessing
# -------------------------------------------------
df["diet"] = df["diet"].astype(str).str.lower().str.strip()
df["species"] = df["species"].astype(str).str.lower().str.strip()

# Keep only herbivorous and piscivorous
# Remove "common seal" from piscivorous, since it would shift the
# piscivorous mean and variance into the herbivorous direction.
df = df[
    (df["diet"] == "herbivorous") |
    ((df["diet"] == "piscivorous") & (df["species"] != "common seal"))
].copy()

# Ensure isotope features exist and are not NaN
df = df.dropna(subset=FEATURES).copy()

df["cluster"] = df["diet"]
df = df.dropna(subset=["cluster"]).copy()

# -------------------------------------------------
# 3) Compute params: mu and sigma≈range/4 per cluster & feature
# -------------------------------------------------
params_rows = []
params = {}  # cluster -> dict(mu, sigma, n)

for cluster_name, sub in df.groupby("cluster"):
    mus = []
    sigmas = []
    n = len(sub)

    for f in FEATURES:
        mu = float(sub[f].mean())
        r = float(sub[f].max() - sub[f].min())
        sigma = r / 4.0

        if sigma == 0.0:
            sigma = 1e-6

        mus.append(mu)
        sigmas.append(sigma)

        params_rows.append(
            {
                "cluster": cluster_name,
                "feature": f,
                "mu": mu,
                "range": r,
                "sigma_range_over_4": sigma,
                "n_samples_in_cluster": n,
            }
        )

    params[cluster_name] = {"mu": np.array(mus), "sigma": np.array(sigmas), "n": n}

params_table = pd.DataFrame(params_rows)

# -------------------------------------------------
# Save original parameters for later comparison
# -------------------------------------------------
original_params = {}

for cluster_name, p in params.items():
    original_params[cluster_name] = {
        "mu": p["mu"].copy(),
        "sigma": p["sigma"].copy()
    }

# -------------------------------------------------
# 4) Manual adjustment of means and sigmas
# -------------------------------------------------
cluster_names = list(params.keys())
print(cluster_names)

for cluster_name in cluster_names:
    if cluster_name not in MANUAL_ADJUSTMENTS:
        continue

    for i, feature in enumerate(FEATURES):
        adj = MANUAL_ADJUSTMENTS[cluster_name].get(
            feature,
            {"mean_shift": 0.0, "sigma_factor": 1.0}
        )

        mean_shift = float(adj.get("mean_shift", 0.0))
        sigma_factor = float(adj.get("sigma_factor", 1.0))

        params[cluster_name]["mu"][i] += mean_shift
        params[cluster_name]["sigma"][i] *= sigma_factor

        if params[cluster_name]["sigma"][i] <= 0.0:
            params[cluster_name]["sigma"][i] = 1e-6

# -------------------------------------------------
# Print comparison table: original vs changed
# -------------------------------------------------
comparison_rows = []

for cluster_name in params.keys():
    for i, f in enumerate(FEATURES):
        comparison_rows.append({
            "cluster": cluster_name,
            "feature": f,
            "mean": original_params[cluster_name]["mu"][i],
            "std": original_params[cluster_name]["sigma"][i],
            "mean_changed": params[cluster_name]["mu"][i],
            "std_changed": params[cluster_name]["sigma"][i],
        })

comparison_table = pd.DataFrame(comparison_rows)

print("\nMean/Std comparison per cluster and feature:\n")
print(
    comparison_table
    .pivot(index="feature", columns="cluster", values=["mean", "std", "mean_changed", "std_changed"])
    .round(3)
)

# -------------------------------------------------
# 5) Generate synthetic samples (Gaussian per feature)
# -------------------------------------------------
rng = np.random.default_rng(SEED)

synthetic_parts = []
for cluster_name, p in params.items():
    X = rng.normal(loc=p["mu"], scale=p["sigma"], size=(SIZE, len(FEATURES)))
    syn = pd.DataFrame(X, columns=FEATURES)
    syn["cluster"] = cluster_name
    synthetic_parts.append(syn)

c1, c2 = cluster_names

ibi_center = 0.5 * (params[c1]["mu"] + params[c2]["mu"])
ibi_sigma = 0.5 * (params[c1]["sigma"] + params[c2]["sigma"]) * IBI_STD_FACTOR

# Since we want the IBI's d13C and d15N values to be equivalent to the herbivorous ones,
# we just set overwrite the named 2 dimensions. 
ibi_center[FEATURES.index("δ13C coll")] = params["herbivorous"]["mu"][FEATURES.index("δ13C coll")]
ibi_center[FEATURES.index("δ15N coll")] = params["herbivorous"]["mu"][FEATURES.index("δ15N coll")]

ibi_sigma[FEATURES.index("δ13C coll")] = params["herbivorous"]["sigma"][FEATURES.index("δ13C coll")]
ibi_sigma[FEATURES.index("δ15N coll")] = params["herbivorous"]["sigma"][FEATURES.index("δ15N coll")]

ibi_X = rng.normal(
    loc=ibi_center,
    scale=ibi_sigma,
    size=(IBI_SIZE, len(FEATURES))
)

ibi_df = pd.DataFrame(ibi_X, columns=FEATURES)
ibi_df["cluster"] = "ibi"
synthetic_parts.append(ibi_df)

synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
synthetic_df.to_csv("../data/synthetic_dataset.csv", index=False)

print("Done.")