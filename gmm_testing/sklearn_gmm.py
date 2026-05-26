import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


FEATURES = ["x1", "x2"]


def scale_like_r_scale(x, center=None, scale=None):
    if center is None:
        center = x.mean(axis=0)
    if scale is None:
        scale = x.std(axis=0, ddof=1)
    return (x - center) / scale, center, scale


def component_for_cluster_a(gmm, x_train_scaled, labels):
    posterior = gmm.predict_proba(x_train_scaled)
    labels = np.asarray(labels)
    a_means = posterior[labels == "A"].mean(axis=0)
    return int(np.argmax(a_means))


def run(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    train_df = df[df["cluster"].isin(["A", "B", "middle"])].copy() #["A", "B"]

    x_train = train_df[FEATURES].to_numpy(dtype=float)
    x_all = df[FEATURES].to_numpy(dtype=float)

    x_train_scaled, center, scale = scale_like_r_scale(x_train)
    x_all_scaled, _, _ = scale_like_r_scale(x_all, center=center, scale=scale)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=42,
        n_init=20,
        reg_covar=1e-6,
    )
    gmm.fit(x_train_scaled)

    cluster_a_component = component_for_cluster_a(
        gmm,
        x_train_scaled,
        train_df["cluster"].to_numpy(),
    )
    posterior = gmm.predict_proba(x_all_scaled)

    result = pd.DataFrame(
        {
            "row_index": np.arange(len(df)),
            "sklearn_prob_a": posterior[:, cluster_a_component],
            "sklearn_model": "full",
        }
    )
    result.to_csv(output_path, index=False)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python3 sklearn_gmm.py <dataset.csv> <output.csv>")
    run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
