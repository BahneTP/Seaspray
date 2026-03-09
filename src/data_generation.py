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

# Separation parameters
SHRINK_SIGMA = 1#0.6
SHIFT_MEAN = 0#2.0

# IBI parameters
IBI_SIZE = 20
IBI_STD_FACTOR = 1.25

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
# 4) Separation adjustment (mean shift + sigma shrink)
# -------------------------------------------------
cluster_names = list(params.keys())
print(cluster_names)

c1, c2 = cluster_names

for i in range(len(FEATURES)):
    mu1 = params[c1]["mu"][i]
    mu2 = params[c2]["mu"][i]
    s1 = params[c1]["sigma"][i]
    s2 = params[c2]["sigma"][i]

    s1_new = s1 * SHRINK_SIGMA
    s2_new = s2 * SHRINK_SIGMA

    mu1_new = mu1 - SHIFT_MEAN * s1_new
    mu2_new = mu2 + SHIFT_MEAN * s2_new

    params[c1]["mu"][i] = mu1_new
    params[c2]["mu"][i] = mu2_new
    params[c1]["sigma"][i] = s1_new
    params[c2]["sigma"][i] = s2_new


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
ibi_std = float(np.mean(0.5 * (params[c1]["sigma"] + params[c2]["sigma"])) * IBI_STD_FACTOR)

ibi_X, _ = make_blobs(
    n_samples=IBI_SIZE,
    centers=[ibi_center],
    cluster_std=ibi_std,
    n_features=len(FEATURES),
    random_state=SEED
)

ibi_df = pd.DataFrame(ibi_X, columns=FEATURES)
ibi_df["cluster"] = "ibi"
synthetic_parts.append(ibi_df)

synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
synthetic_df.to_csv("../data/synthetic_dataset.csv", index=False)

print("Done.")
print(synthetic_df["cluster"].value_counts())
print(synthetic_df.head())