import numpy as np
import pandas as pd

from ibi_detection import detect_ibi
from ibi_detection.common import calculate_ibi_metrics, calculate_lower_marine_limits
from ibi_detection.gmm_ibi_cluster import add_gmm_cluster, find_potential_ibi_cluster
from ibi_detection.herbivorous_mahalanobis import add_herbivorous_mahalanobis
from models.gmm import run_gmm
from models.svm import run_svm


def _models_for_mode(mode):
    mode = mode.lower()

    if mode in ["gmm", "gmm_ibi_cluster", "gmm_inverse_prob"]:
        return True, False

    if mode in ["svm", "svm_margin"]:
        return False, True

    if mode == "combined":
        return True, True

    if mode == "herbivorous_mahalanobis":
        return False, False

    raise ValueError(
        "mode must be one of: 'svm', 'gmm', 'gmm_inverse_prob', "
        "'herbivorous_mahalanobis', 'combined'"
    )


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
        df_results = run_gmm(df_train, df_results, feature_cols)

    if use_svm:
        df_results = run_svm(df_train, df_results, feature_cols)

    if mode.lower() == "herbivorous_mahalanobis":
        df_results = add_herbivorous_mahalanobis(
            df_train,
            df_results,
            feature_cols,
        )

    return df_results


def _get_potential_ibi_cluster(df_train_model, feature_cols):
    df_clustered = add_gmm_cluster(df_train_model)
    potential_ibi_cluster, _, _ = find_potential_ibi_cluster(
        df_clustered,
        feature_cols,
    )
    return potential_ibi_cluster


def _calculate_split_metrics(
    df_model,
    mode,
    split_name,
    iteration,
    threshold,
    svm_threshold,
    gmm_threshold,
    combined_operator,
    potential_ibi_cluster,
    lower_marine_limit_detection,
    feature_cols,
    lower_marine_limits,
    best_f1_gmm_threshold=None,
):
    df_detected = detect_ibi(
        df_train=df_model,
        df_eval=df_model,
        feature_cols=feature_cols,
        mode=mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        printing=False,
        df_model=df_model,
        potential_ibi_cluster=potential_ibi_cluster,
        lower_marine_limit_detection=lower_marine_limit_detection,
        lower_marine_limits=lower_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_detected)
    metrics["iteration"] = iteration
    metrics["split"] = split_name
    metrics["threshold"] = threshold
    metrics["svm_threshold"] = svm_threshold
    metrics["gmm_threshold"] = gmm_threshold
    metrics["n_rows"] = len(df_detected)

    if best_f1_gmm_threshold is not None:
        metrics["best_f1_gmm_threshold"] = best_f1_gmm_threshold

    return metrics


def _threshold_values(
    thresholds,
    start_threshold,
    end_threshold,
    step_size,
):
    if thresholds is not None:
        return np.array(thresholds, dtype=float)

    return np.arange(start_threshold, end_threshold + step_size / 2, step_size)


def _thresholds_for_mode(
    mode,
    threshold,
    sweep_threshold,
    fixed_svm_threshold,
    fixed_gmm_threshold,
):
    mode = mode.lower()

    if sweep_threshold == "auto":
        sweep_threshold = "svm" if mode in ["svm", "svm_margin", "combined"] else "gmm"
    else:
        sweep_threshold = sweep_threshold.lower()

    if sweep_threshold not in ["svm", "gmm", "both"]:
        raise ValueError("sweep_threshold must be one of: 'auto', 'svm', 'gmm', 'both'")

    svm_threshold = fixed_svm_threshold
    gmm_threshold = fixed_gmm_threshold

    if sweep_threshold in ["svm", "both"]:
        svm_threshold = threshold

    if sweep_threshold in ["gmm", "both"]:
        gmm_threshold = threshold

    return svm_threshold, gmm_threshold


def _fixed_threshold_for_mode(mode, svm_threshold, gmm_threshold):
    return svm_threshold if mode.lower() in ["svm", "svm_margin"] else gmm_threshold


def _find_best_f1_threshold(
    df_model,
    mode,
    threshold_values,
    sweep_threshold,
    fixed_svm_threshold,
    fixed_gmm_threshold,
    combined_operator,
    potential_ibi_cluster,
    lower_marine_limit_detection,
    feature_cols,
    lower_marine_limits,
):
    rows = []

    for threshold in threshold_values:
        svm_threshold, gmm_threshold = _thresholds_for_mode(
            mode=mode,
            threshold=threshold,
            sweep_threshold=sweep_threshold,
            fixed_svm_threshold=fixed_svm_threshold,
            fixed_gmm_threshold=fixed_gmm_threshold,
        )
        df_detected = detect_ibi(
            df_train=df_model,
            df_eval=df_model,
            feature_cols=feature_cols,
            mode=mode,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            printing=False,
            df_model=df_model,
            potential_ibi_cluster=potential_ibi_cluster,
            lower_marine_limit_detection=lower_marine_limit_detection,
            lower_marine_limits=lower_marine_limits,
        )
        metrics = calculate_ibi_metrics(df_detected)
        rows.append(
            {
                "threshold": threshold,
                "f1_score": metrics["f1_score"],
            }
        )

    threshold_metrics = pd.DataFrame(rows)
    best_row = threshold_metrics.loc[threshold_metrics["f1_score"].idxmax()]

    return best_row["threshold"]


def _calculate_threshold_metrics(
    df_model,
    mode,
    split_name,
    iteration,
    threshold_values,
    sweep_threshold,
    fixed_svm_threshold,
    fixed_gmm_threshold,
    combined_operator,
    potential_ibi_cluster,
    lower_marine_limit_detection,
    feature_cols,
    lower_marine_limits,
):
    rows = []

    for threshold in threshold_values:
        svm_threshold, gmm_threshold = _thresholds_for_mode(
            mode=mode,
            threshold=threshold,
            sweep_threshold=sweep_threshold,
            fixed_svm_threshold=fixed_svm_threshold,
            fixed_gmm_threshold=fixed_gmm_threshold,
        )
        rows.append(
            _calculate_split_metrics(
                df_model=df_model,
                mode=mode,
                split_name=split_name,
                iteration=iteration,
                threshold=threshold,
                svm_threshold=svm_threshold,
                gmm_threshold=gmm_threshold,
                combined_operator=combined_operator,
                potential_ibi_cluster=potential_ibi_cluster,
                lower_marine_limit_detection=lower_marine_limit_detection,
                feature_cols=feature_cols,
                lower_marine_limits=lower_marine_limits,
            )
        )

    return rows


def _evaluate_iteration(
    df_train,
    df_test,
    feature_cols,
    mode,
    iteration,
    threshold_values,
    sweep_threshold,
    fixed_svm_threshold,
    fixed_gmm_threshold,
    combined_operator,
    lower_marine_limit_detection,
    include_threshold_metrics=False,
):
    use_gmm, _ = _models_for_mode(mode)
    lower_marine_limits = None

    if lower_marine_limit_detection:
        lower_marine_limits = calculate_lower_marine_limits(df_train, feature_cols)

    df_train_model = _add_model_outputs(
        df_train=df_train,
        df_eval=df_train,
        feature_cols=feature_cols,
        mode=mode,
    )
    potential_ibi_cluster = None

    if use_gmm:
        potential_ibi_cluster = _get_potential_ibi_cluster(
            df_train_model,
            feature_cols,
        )

    df_test_model = _add_model_outputs(
        df_train=df_train,
        df_eval=df_test,
        feature_cols=feature_cols,
        mode=mode,
    )

    fixed_threshold = _fixed_threshold_for_mode(
        mode=mode,
        svm_threshold=fixed_svm_threshold,
        gmm_threshold=fixed_gmm_threshold,
    )
    best_train_threshold = _find_best_f1_threshold(
        df_model=df_train_model,
        mode=mode,
        threshold_values=threshold_values,
        sweep_threshold=sweep_threshold,
        fixed_svm_threshold=fixed_svm_threshold,
        fixed_gmm_threshold=fixed_gmm_threshold,
        combined_operator=combined_operator,
        potential_ibi_cluster=potential_ibi_cluster,
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )
    best_test_threshold = _find_best_f1_threshold(
        df_model=df_test_model,
        mode=mode,
        threshold_values=threshold_values,
        sweep_threshold=sweep_threshold,
        fixed_svm_threshold=fixed_svm_threshold,
        fixed_gmm_threshold=fixed_gmm_threshold,
        combined_operator=combined_operator,
        potential_ibi_cluster=potential_ibi_cluster,
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )

    fixed_rows = [
        _calculate_split_metrics(
            df_model=df_train_model,
            mode=mode,
            split_name="train",
            iteration=iteration,
            threshold=fixed_threshold,
            svm_threshold=fixed_svm_threshold,
            gmm_threshold=fixed_gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
            best_f1_gmm_threshold=best_train_threshold,
        ),
        _calculate_split_metrics(
            df_model=df_test_model,
            mode=mode,
            split_name="test",
            iteration=iteration,
            threshold=fixed_threshold,
            svm_threshold=fixed_svm_threshold,
            gmm_threshold=fixed_gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
            best_f1_gmm_threshold=best_test_threshold,
        ),
    ]

    if not include_threshold_metrics:
        return fixed_rows

    threshold_rows = []
    threshold_rows.extend(
        _calculate_threshold_metrics(
            df_model=df_train_model,
            mode=mode,
            split_name="train",
            iteration=iteration,
            threshold_values=threshold_values,
            sweep_threshold=sweep_threshold,
            fixed_svm_threshold=fixed_svm_threshold,
            fixed_gmm_threshold=fixed_gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
        )
    )
    threshold_rows.extend(
        _calculate_threshold_metrics(
            df_model=df_test_model,
            mode=mode,
            split_name="test",
            iteration=iteration,
            threshold_values=threshold_values,
            sweep_threshold=sweep_threshold,
            fixed_svm_threshold=fixed_svm_threshold,
            fixed_gmm_threshold=fixed_gmm_threshold,
            combined_operator=combined_operator,
            potential_ibi_cluster=potential_ibi_cluster,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
        )
    )

    return fixed_rows, threshold_rows


def run_bootstrap_evaluation(
    df,
    feature_cols,
    mode="gmm",
    n_iterations=200,
    svm_threshold=1.0,
    gmm_threshold=0.18,
    combined_operator="or",
    lower_marine_limit_detection=True,
    thresholds=None,
    start_threshold=0.0,
    end_threshold=1.0,
    step_size=0.01,
    sweep_threshold="auto",
    random_state=42,
    return_threshold_metrics=False,
):
    rows = []
    threshold_rows = []
    rng = np.random.default_rng(random_state)
    threshold_values = _threshold_values(
        thresholds=thresholds,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        step_size=step_size,
    )

    for iteration in range(n_iterations):
        split_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        df_train, df_test = make_stratified_bootstrap_split(
            df,
            stratify_col="cluster",
            random_state=split_seed,
        )

        if df_test.empty:
            continue

        iteration_results = _evaluate_iteration(
            df_train=df_train,
            df_test=df_test,
            feature_cols=feature_cols,
            mode=mode,
            iteration=iteration,
            threshold_values=threshold_values,
            sweep_threshold=sweep_threshold,
            fixed_svm_threshold=svm_threshold,
            fixed_gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            lower_marine_limit_detection=lower_marine_limit_detection,
            include_threshold_metrics=return_threshold_metrics,
        )

        if return_threshold_metrics:
            fixed_rows, iteration_threshold_rows = iteration_results
            rows.extend(fixed_rows)
            threshold_rows.extend(iteration_threshold_rows)
        else:
            rows.extend(iteration_results)

    metrics_df = pd.DataFrame(rows)

    if return_threshold_metrics:
        return metrics_df, pd.DataFrame(threshold_rows)

    return metrics_df


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
        "best_f1_gmm_threshold",
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


def summarize_bootstrap_threshold_metrics(threshold_metrics_df):
    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    rows = []

    for (split_name, threshold), split_df in threshold_metrics_df.groupby(["split", "threshold"]):
        for metric in metric_cols:
            values = split_df[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "split": split_name,
                    "threshold": threshold,
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
