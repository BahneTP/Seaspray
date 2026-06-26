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
):
    mode = mode.lower()

    if mode == "svm":
        return detect_svm(
            df,
            threshold=svm_threshold,
            printing=printing,
        )

    if mode == "gmm":
        return detect_gmm(
            df,
            threshold=gmm_threshold,
            printing=printing,
        )

    if mode == "combined":
        return detect_combined(
            df,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            operator=combined_operator,
            printing=printing,
        )

    raise ValueError("mode must be one of: 'svm', 'gmm', 'combined'")


__all__ = [
    "detect_ibi",
    "plot_ibi_threshold_metrics",
    "detect_svm",
    "detect_gmm",
    "detect_combined",
]
