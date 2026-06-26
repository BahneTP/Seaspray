from models.gmm import run_gmm

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report
from .gmm_ibi_cluster import _get_gmm_prob_cols


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
    df_results = df_model.copy() if df_model is not None else run_gmm(
        df_train,
        df_eval,
        feature_cols,
    )
    prob_cols = _get_gmm_prob_cols(df_results)
    df_results["gmm_inverse_prob"] = 1 - df_results[prob_cols].max(axis=1)
    df_results = add_ibi_prediction(
        df_results,
        df_results["gmm_inverse_prob"] >= threshold,
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(metrics, mode="gmm_inverse_prob", gmm_threshold=threshold)

    return df_results
