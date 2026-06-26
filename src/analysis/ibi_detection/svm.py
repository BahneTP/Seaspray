from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


def get_svm_mask(df, threshold):
    return df["svm_margin_distance"] <= threshold


def detect_svm(df, threshold=1.0, printing=True):
    df_results = add_ibi_prediction(df, get_svm_mask(df, threshold))
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(
            metrics,
            mode="svm",
            svm_threshold=threshold,
        )

    return df_results
