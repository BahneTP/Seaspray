from models.svm import run_svm

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


def get_svm_margin_mask(df, threshold):
    return df["svm_margin_distance"] <= threshold


def detect(
    df_train,
    df_eval,
    feature_cols,
    threshold=1.0,
    df_model=None,
    printing=True,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
):
    df_results = df_model.copy() if df_model is not None else run_svm(
        df_train,
        df_eval,
        feature_cols,
    )
    df_results = add_ibi_prediction(
        df_results,
        get_svm_margin_mask(df_results, threshold),
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(metrics, mode="svm_margin", svm_threshold=threshold)

    return df_results
