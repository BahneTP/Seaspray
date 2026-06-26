from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report
from .gmm import add_gmm_ibi_prob
from .svm import get_svm_mask


def get_combined_mask(df, svm_threshold, gmm_threshold, operator="or"):
    operator = operator.lower()

    if operator not in ["or", "and"]:
        raise ValueError("combined_operator must be either 'or' or 'and'")

    df_results = add_gmm_ibi_prob(df)
    svm_detected = get_svm_mask(df_results, svm_threshold)
    gmm_detected = df_results["gmm_ibi_prob"] >= gmm_threshold

    if operator == "and":
        return svm_detected & gmm_detected

    return svm_detected | gmm_detected


def detect_combined(
    df,
    svm_threshold=1.0,
    gmm_threshold=0.2,
    operator="or",
    printing=True,
):
    df_results = add_gmm_ibi_prob(df)
    df_results = add_ibi_prediction(
        df_results,
        get_combined_mask(
            df_results,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            operator=operator,
        ),
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(
            metrics,
            mode="combined",
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=operator,
        )

    return df_results
