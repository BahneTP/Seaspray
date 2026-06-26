from . import (
    combined,
    gmm_ibi_cluster,
    gmm_inverse_prob,
    herbivorous_mahalanobis,
    svm_margin,
)


GMM_MODES = {"gmm", "gmm_ibi_cluster"}
SVM_MODES = {"svm", "svm_margin"}


def detect_ibi(
    df_train,
    df_eval,
    feature_cols,
    mode="gmm_ibi_cluster",
    svm_threshold=1.0,
    gmm_threshold=0.2,
    printing=True,
    df_model=None,
    potential_ibi_cluster=None,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
    **_,
):
    mode = mode.lower()

    if mode in GMM_MODES:
        return gmm_ibi_cluster.detect(
            df_train=df_train,
            df_eval=df_eval,
            feature_cols=feature_cols,
            threshold=gmm_threshold,
            df_model=df_model,
            potential_ibi_cluster=potential_ibi_cluster,
            printing=printing,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )

    if mode == "gmm_inverse_prob":
        return gmm_inverse_prob.detect(
            df_train=df_train,
            df_eval=df_eval,
            feature_cols=feature_cols,
            threshold=gmm_threshold,
            df_model=df_model,
            printing=printing,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )

    if mode == "herbivorous_mahalanobis":
        return herbivorous_mahalanobis.detect(
            df_train=df_train,
            df_eval=df_eval,
            feature_cols=feature_cols,
            threshold=gmm_threshold,
            df_model=df_model,
            printing=printing,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )

    if mode in SVM_MODES:
        return svm_margin.detect(
            df_train=df_train,
            df_eval=df_eval,
            feature_cols=feature_cols,
            threshold=svm_threshold,
            df_model=df_model,
            printing=printing,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )

    if mode == "combined":
        return combined.detect(
            df_train=df_train,
            df_eval=df_eval,
            feature_cols=feature_cols,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            operator=_.get("combined_operator", "or"),
            df_model=df_model,
            potential_ibi_cluster=potential_ibi_cluster,
            printing=printing,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )

    raise ValueError(
        "mode must be one of: 'gmm', 'gmm_ibi_cluster', 'gmm_inverse_prob', "
        "'herbivorous_mahalanobis', 'svm', 'svm_margin', 'combined'"
    )
