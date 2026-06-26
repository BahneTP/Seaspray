from models.svm import run_svm

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report
from .gmm_ibi_cluster import prepare_gmm_ibi_cluster
from .svm_margin import get_svm_margin_mask


def prepare_combined(
    df_train,
    df_eval,
    feature_cols,
    df_model=None,
    potential_ibi_cluster=None,
    printing=False,
):
    df_results, potential_ibi_cluster = prepare_gmm_ibi_cluster(
        df_train=df_train,
        df_eval=df_eval,
        feature_cols=feature_cols,
        df_model=df_model,
        potential_ibi_cluster=potential_ibi_cluster,
        printing=printing,
    )

    if "svm_margin_distance" not in df_results.columns:
        df_results = run_svm(df_train, df_results, feature_cols)

    return df_results, potential_ibi_cluster


def get_combined_mask(
    df_model,
    svm_threshold,
    gmm_threshold,
    operator="or",
):
    operator = operator.lower()

    if operator not in ["or", "and"]:
        raise ValueError("combined_operator must be either 'or' or 'and'")

    svm_detected = get_svm_margin_mask(df_model, svm_threshold)
    gmm_detected = df_model["gmm_ibi_prob"] >= gmm_threshold

    if operator == "and":
        return svm_detected & gmm_detected

    return svm_detected | gmm_detected


def detect(
    df_train,
    df_eval,
    feature_cols,
    svm_threshold=1.0,
    gmm_threshold=0.2,
    operator="or",
    df_model=None,
    potential_ibi_cluster=None,
    printing=True,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
):
    df_results, potential_ibi_cluster = prepare_combined(
        df_train=df_train,
        df_eval=df_eval,
        feature_cols=feature_cols,
        df_model=df_model,
        potential_ibi_cluster=potential_ibi_cluster,
        printing=printing,
    )
    df_results = add_ibi_prediction(
        df_results,
        get_combined_mask(
            df_results,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            operator=operator,
        ),
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
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
