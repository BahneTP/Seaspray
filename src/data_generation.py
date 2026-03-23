import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd

def run(config):
    c = config.Config
    ma = config.Manual_Adjustments

    # -------------------------------------------------
    # 1) Load data
    # -------------------------------------------------
    df = pd.read_excel(c.INPUT_XLSX, engine="openpyxl", sheet_name=c.SHEET_NAME, nrows=c.NROWS)

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
    df = df.dropna(subset=c.FEATURES).copy()

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

        for f in c.FEATURES:
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
        if cluster_name not in ma:
            continue

        for i, feature in enumerate(c.FEATURES):
            adj = ma[cluster_name].get(
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
    # 5) Generate synthetic samples (Gaussian per feature)
    # -------------------------------------------------
    rng = np.random.default_rng(c.SEED)

    synthetic_parts = []
    for cluster_name, p in params.items():
        X = rng.normal(loc=p["mu"], scale=p["sigma"], size=(c.SIZE, len(c.FEATURES)))
        syn = pd.DataFrame(X, columns=c.FEATURES)
        syn["cluster"] = cluster_name
        synthetic_parts.append(syn)

    c1, c2 = cluster_names

    ibi_center = 0.5 * (params[c1]["mu"] + params[c2]["mu"])
    ibi_sigma = 0.5 * (params[c1]["sigma"] + params[c2]["sigma"]) * c.IBI_STD_FACTOR

    # Following "for" and "before-values" only for the table.
    ibi_center_before = ibi_center.copy()
    ibi_sigma_before = ibi_sigma.copy()

    for feature in ["δ13C coll", "δ15N coll"]:
        i = c.FEATURES.index(feature)
        ibi_center_before[i] = params["herbivorous"]["mu"][i]
        ibi_sigma_before[i] = params["herbivorous"]["sigma"][i]

    # coll's as herbivores.
    for feature in ["δ13C coll", "δ15N coll"]:
        i = c.FEATURES.index(feature)
        adj = ma["ibi"][feature]

        ibi_center[i] = params["herbivorous"]["mu"][i] + float(adj["mean_shift"])
        ibi_sigma[i] = params["herbivorous"]["sigma"][i] * float(adj["sigma_factor"])

    # Carbs in-between of piscivorous and herbivorous, but shifted and factored.
    for feature in ["δ13C carb", "δ18O carb"]:
        i = c.FEATURES.index(feature)
        adj = ma["ibi"][feature]

        ibi_center[i] = ibi_center[i] + float(adj["mean_shift"])
        ibi_sigma[i] = ibi_sigma[i] * float(adj["sigma_factor"])

    ibi_X = rng.normal(
        loc=ibi_center,
        scale=ibi_sigma,
        size=(c.IBI_SIZE, len(c.FEATURES))
    )

    ibi_df = pd.DataFrame(ibi_X, columns=c.FEATURES)
    ibi_df["cluster"] = "ibi"
    synthetic_parts.append(ibi_df)

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
    synthetic_df.to_csv(f"../output/synthetic_datasets/{c.NAME}.csv", index=False)


    # -------------------------------------------------
    # Print comparison table at the end: before/after
    # -------------------------------------------------
    comparison_rows = []

    # herbivorous + piscivorous
    for cluster_name in ["herbivorous", "piscivorous"]:
        for i, f in enumerate(c.FEATURES):
            comparison_rows.append({
                "cluster": cluster_name,
                "feature": f,
                "mean": f'{original_params[cluster_name]["mu"][i]:.3f}/{params[cluster_name]["mu"][i]:.3f}',
                "std": f'{original_params[cluster_name]["sigma"][i]:.3f}/{params[cluster_name]["sigma"][i]:.3f}',
            })

    # ibi
    for i, f in enumerate(c.FEATURES):
        comparison_rows.append({
            "cluster": "ibi",
            "feature": f,
            "mean": f"{ibi_center_before[i]:.3f}/{ibi_center[i]:.3f}",
            "std": f"{ibi_sigma_before[i]:.3f}/{ibi_sigma[i]:.3f}",
        })

    comparison_table = pd.DataFrame(comparison_rows)

    print("\nMean/Std comparison per cluster and feature (before/after):\n")
    print(
        comparison_table
        .pivot(index="feature", columns="cluster", values=["mean", "std"])
    )