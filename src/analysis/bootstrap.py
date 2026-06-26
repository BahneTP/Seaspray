import numpy as np
import pandas as pd

from gmm_r import GMM_R
from ibi_detection import detect_ibi
from ibi_detection.common import calculate_ibi_metrics, calculate_upper_marine_limits
from ibi_detection.gmm import add_gmm_cluster, find_potential_ibi_cluster
from svm import SVM


def _models_for_mode(mode):
    mode = mode.lower()

    if mode == "gmm":
        return True, False

    if mode == "svm":
        return False, True

    if mode == "combined":
        return True, True

    raise ValueError("mode must be one of: 'svm', 'gmm', 'combined'")


def make_stratified_bootstrap_split(
    df,
    stratify_col="cluster",
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    train_indices = []
    oob_indices = []

    for _, group in df.groupby(stratify_col):
        indices = group.index.to_numpy()
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sampled_unique = np.unique(sampled)
        oob = np.setdiff1d(indices, sampled_unique)

        train_indices.extend(sampled.tolist())
        oob_indices.extend(oob.tolist())

    df_train = df.loc[train_indices].reset_index(drop=True)
    df_test = df.loc[oob_indices].reset_index(drop=True)

    return df_train, df_test


def _add_model_outputs(
    df_train,
    df_eval,
    feature_cols,
    mode,
):
    use_gmm, use_svm = _models_for_mode(mode)
    df_results = df_eval.copy()

    if use_gmm:
        gmm = GMM_R(df_train, feature_cols)
        df_results = gmm.save(df_results)

    if use_svm:
        svm = SVM(df_train, df_eval, feature_cols)
        df_results = svm.save(df_results)

    return df_results


def _get_potential_ibi_cluster(df_train_model):
    df_clustered = add_gmm_cluster(df_train_model)
    potential_ibi_cluster, _, _ = find_potential_ibi_cluster(df_clustered)
    return potential_ibi_cluster


def _calculate_split_metrics(
    df_model,
    mode,
    split_name,
    iteration,
    svm_threshold,
    gmm_threshold,
    combined_operator,
    potential_ibi_cluster,
    upper_marine_limit_detection,
    feature_cols,
    upper_marine_limits,
):
    df_detected = detect_ibi(
        df_model,
        mode=mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        printing=False,
        potential_ibi_cluster=potential_ibi_cluster,
        upper_marine_limit_detection=upper_marine_limit_detection,
        feature_cols=feature_cols,
        upper_marine_limits=upper_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_detected)
    metrics["iteration"] = iteration
    metrics["split"] = split_name
    metrics["n_rows"] = len(df_detected)

    return metrics


def _evaluate_iteration(
    df_train,
    df_test,
    feature_cols,
    mode,
    iteration,
    svm_threshold,
    gmm_threshold,
    combined_operator,
    upper_marine_limit_detection,
):
    use_gmm, _ = _models_for_mode(mode)
    upper_marine_limits = None

    if upper_marine_limit_detection:
        upper_marine_limits = calculate_upper_marine_limits(df_train, feature_cols)

    df_train_model = _add_model_outputs(
        df_train=df_train,
        df_eval=df_train,
        feature_cols=feature_cols,
        mode=mode,
    )
    potential_ibi_cluster = None

    if use_gmm:
        potential_ibi_cluster = _get_potential_ibi_cluster(df_train_model)

    df_test_model = _add_model_outputs(
        df_train=df_train,
        df_eval=df_test,
        feature_cols=feature_cols,
        mode=mode,
    )

    return [
        _calculate_split_metrics(
            df_model=df_train_model,
            mode=mode,
            split_name="train",
            iteration=iteration,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            upper_marine_limit_detection=upper_marine_limit_detection,
            feature_cols=feature_cols,
            upper_marine_limits=upper_marine_limits,
        ),
        _calculate_split_metrics(
            df_model=df_test_model,
            mode=mode,
            split_name="test",
            iteration=iteration,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            upper_marine_limit_detection=upper_marine_limit_detection,
            feature_cols=feature_cols,
            upper_marine_limits=upper_marine_limits,
        ),
    ]


def run_bootstrap_evaluation(
    df,
    feature_cols,
    mode="gmm",
    n_iterations=200,
    svm_threshold=1.0,
    gmm_threshold=0.18,
    combined_operator="or",
    upper_marine_limit_detection=True,
    random_state=42,
):
    rows = []
    rng = np.random.default_rng(random_state)

    for iteration in range(n_iterations):
        split_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        df_train, df_test = make_stratified_bootstrap_split(
            df,
            stratify_col="cluster",
            random_state=split_seed,
        )

        if df_test.empty:
            continue

        rows.extend(
            _evaluate_iteration(
                df_train=df_train,
                df_test=df_test,
                feature_cols=feature_cols,
                mode=mode,
                iteration=iteration,
                svm_threshold=svm_threshold,
                gmm_threshold=gmm_threshold,
                combined_operator=combined_operator,
                upper_marine_limit_detection=upper_marine_limit_detection,
            )
        )

    return pd.DataFrame(rows)


def summarize_bootstrap_metrics(metrics_df):
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "detected_ibi",
        "true_ibi",
        "correct_ibi",
        "false_positive_ibi",
    ]
    rows = []

    for split_name, split_df in metrics_df.groupby("split"):
        for metric in metric_cols:
            values = split_df[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    "min": np.min(values),
                    "max": np.max(values),
                    "q025": np.quantile(values, 0.025),
                    "q975": np.quantile(values, 0.975),
                }
            )

    return pd.DataFrame(rows)
