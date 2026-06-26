from .combined import detect_combined
from .gmm import detect_gmm
from .plotting import plot_ibi_threshold_metrics
from .svm import detect_svm


def detect_ibi(
    df,
    mode="svm",
    svm_threshold=1.0,
    gmm_threshold=0.2,
    combined_operator="or",
    printing=True,
    potential_ibi_cluster=None,
    upper_marine_limit_detection=False,
    feature_cols=None,
    upper_marine_limits=None,
):
    mode = mode.lower()

    if mode == "svm":
        return detect_svm(
            df,
            threshold=svm_threshold,
            printing=printing,
            upper_marine_limit_detection=upper_marine_limit_detection,
            feature_cols=feature_cols,
            upper_marine_limits=upper_marine_limits,
        )

    if mode == "gmm":
        return detect_gmm(
            df,
            threshold=gmm_threshold,
            printing=printing,
            potential_ibi_cluster=potential_ibi_cluster,
            upper_marine_limit_detection=upper_marine_limit_detection,
            feature_cols=feature_cols,
            upper_marine_limits=upper_marine_limits,
        )

    if mode == "combined":
        return detect_combined(
            df,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            operator=combined_operator,
            printing=printing,
            potential_ibi_cluster=potential_ibi_cluster,
            upper_marine_limit_detection=upper_marine_limit_detection,
            feature_cols=feature_cols,
            upper_marine_limits=upper_marine_limits,
        )

    raise ValueError("mode must be one of: 'svm', 'gmm', 'combined'")


__all__ = [
    "detect_ibi",
    "plot_ibi_threshold_metrics",
    "detect_svm",
    "detect_gmm",
    "detect_combined",
]
