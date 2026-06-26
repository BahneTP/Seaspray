from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


def get_svm_mask(df, threshold):
    return df["svm_margin_distance"] <= threshold


def detect_svm(
    df,
    threshold=1.0,
    printing=True,
    upper_marine_limit_detection=False,
    feature_cols=None,
    upper_marine_limits=None,
):
    df_results = add_ibi_prediction(
        df,
        get_svm_mask(df, threshold),
        upper_marine_limit_detection=upper_marine_limit_detection,
        feature_cols=feature_cols,
        upper_marine_limits=upper_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(
            metrics,
            mode="svm",
            svm_threshold=threshold,
        )

    return df_results
