import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


DATA_FILE = "synthetic_gmm_data.csv"
TEST_FILE = "test_points.csv"
FEATURES = ["x1", "x2"]


def scale_like_r_scale(x, center=None, scale=None):
    if center is None:
        center = x.mean(axis=0)
    if scale is None:
        scale = x.std(axis=0, ddof=1)
    return (x - center) / scale, center, scale


def main():
    df = pd.read_csv(DATA_FILE)
    test_points = pd.read_csv(TEST_FILE)

    x = df[FEATURES].to_numpy(dtype=float)
    x_test = test_points[FEATURES].to_numpy(dtype=float)

    x_scaled, center, scale = scale_like_r_scale(x)
    x_test_scaled, _, _ = scale_like_r_scale(x_test, center=center, scale=scale)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=42,
        n_init=20,
        reg_covar=1e-6,
    )
    gmm.fit(x_scaled)

    posterior = gmm.predict_proba(x_test_scaled)
    prediction = gmm.predict(x_test_scaled) + 1
    uncertainty = 1.0 - posterior.max(axis=1)

    result = test_points.copy()
    for component in range(posterior.shape[1]):
        result[f"component_{component + 1}"] = posterior[:, component]
    result["prediction"] = prediction
    result["uncertainty"] = uncertainty

    print("=== Python / sklearn ===")
    print(f"Converged: {gmm.converged_}")
    print(f"Iterations: {gmm.n_iter_}")
    print()
    print("Posterior probabilities for test points:")
    print(result.to_string(index=False))
    print()
    print("Weights:")
    print(gmm.weights_)
    print()
    print("Means:")
    print(gmm.means_)
    print()
    print("Covariances:")
    print(gmm.covariances_)


if __name__ == "__main__":
    main()
