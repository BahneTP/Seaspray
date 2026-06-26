import numpy as np

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


def _fit_herbivorous_distribution(
    df_train,
    feature_cols,
    herbivorous_label="herbivorous",
    regularization=1e-6,
):
    herbivorous = df_train[df_train["cluster"] == herbivorous_label]

    if herbivorous.empty:
        raise ValueError(f"No rows found for herbivorous label: {herbivorous_label}")

    values = herbivorous[feature_cols].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])

    covariance = covariance + np.eye(len(feature_cols)) * regularization
    inverse_covariance = np.linalg.pinv(covariance)

    return mean, inverse_covariance


def add_herbivorous_mahalanobis(
    df_train,
    df_eval,
    feature_cols,
    herbivorous_label="herbivorous",
):
    mean, inverse_covariance = _fit_herbivorous_distribution(
        df_train,
        feature_cols,
        herbivorous_label=herbivorous_label,
    )
    df_results = df_eval.copy()
    diff = df_results[feature_cols].to_numpy(dtype=float) - mean
    squared_distance = np.einsum("ij,jk,ik->i", diff, inverse_covariance, diff)
    distance = np.sqrt(np.maximum(squared_distance, 0))

    df_results["herbivorous_mahalanobis_distance"] = distance
    df_results["herbivorous_mahalanobis"] = 1 - np.exp(-0.5 * squared_distance)

    return df_results


def detect(
    df_train,
    df_eval,
    feature_cols,
    threshold=0.2,
    df_model=None,
    printing=True,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
):
    if df_model is not None and "herbivorous_mahalanobis" in df_model.columns:
        df_results = df_model.copy()
    else:
        df_results = add_herbivorous_mahalanobis(
            df_train,
            df_eval,
            feature_cols,
        )

    df_results = add_ibi_prediction(
        df_results,
        df_results["herbivorous_mahalanobis"] >= threshold,
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(
            metrics,
            mode="herbivorous_mahalanobis",
            gmm_threshold=threshold,
        )

    return df_results
