"""
Synthetic dataset generation (Thesis-style)

Implements the synthetic-data setup described in the thesis:
- Define reference clusters
- For each cluster and each feature:
    mu  = mean(real_data)
    sigma ≈ (max - min) / 4   (range/4 heuristic)
- Sample synthetic points from independent Gaussian distributions:
    x ~ N(mu, sigma^2) per feature
- Reproducible via fixed random seed
"""

import numpy as np
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_XLSX = "../data/terrherbmarin.xlsx"
SHEET_NAME = 0               # or e.g. "Sheet1"
NROWS = 88                 # e.g. 88, or None for all
SEED = 0
FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]
SIZE = 300                 

# Output files
OUT_CSV = "synthetic_dataset.csv"
OUT_XLSX = "synthetic_dataset.xlsx"


# -------------------------------------------------
# 1) Load data
# -------------------------------------------------
df = pd.read_excel(INPUT_XLSX, engine="openpyxl", sheet_name=SHEET_NAME, nrows=NROWS)

# Normalize strings (safe even if NaN)
if "diet" in df.columns:
    df["diet"] = df["diet"].astype(str).str.lower().str.strip()


# -------------------------------------------------
# 2) ONE-HOT Encoding
# -------------------------------------------------
# (kept as-is; now uses diet categories inferred from FEATURES' names is not intended,
# so we skip one-hot creation if FEATURES are isotopes)
# If you still want one-hot diet columns in addition to isotopes, define DIET_FEATURES separately.
pass


# -------------------------------------------------
# 3) Define reference clusters
# -------------------------------------------------
# Use diet as clusters (same as your original logic), but do NOT filter by FEATURES anymore
# because FEATURES are isotope column names now.
df["cluster"] = df["diet"]
df = df.dropna(subset=["cluster"]).copy()

# Ensure isotope features exist and are not NaN
df = df.dropna(subset=FEATURES).copy()


# -------------------------------------------------
# 4) Compute params: mu and sigma≈range/4 per cluster & feature
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

        # Avoid sigma=0 (would make feature constant in synthetic data)
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
# 5) Generate synthetic samples (Gaussian per feature)
# -------------------------------------------------
rng = np.random.default_rng(SEED)

synthetic_parts = []
for cluster_name, p in params.items():
    X = rng.normal(loc=p["mu"], scale=p["sigma"], size=(SIZE, len(FEATURES)))
    syn = pd.DataFrame(X, columns=FEATURES)
    syn["cluster"] = cluster_name
    synthetic_parts.append(syn)

synthetic_df = pd.concat(synthetic_parts, ignore_index=True)


# -------------------------------------------------
# 6) Save outputs
# -------------------------------------------------
synthetic_df.to_csv(OUT_CSV, index=False)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    synthetic_df.to_excel(writer, index=False, sheet_name="synthetic")
    params_table.to_excel(writer, index=False, sheet_name="params_mu_sigma")
    # quick real summary
    real_summary = df.groupby("cluster")[FEATURES].agg(["mean", "min", "max", "std", "count"])
    real_summary.to_excel(writer, sheet_name="real_summary")

print("Done.")
print("Clusters:", df["cluster"].value_counts().to_dict())
print("Saved:", OUT_CSV, "and", OUT_XLSX)
print(synthetic_df.head())