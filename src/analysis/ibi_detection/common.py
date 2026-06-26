import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_ibi_metrics(df):
    true_ibi = df["cluster"] == "ibi"
    detected_ibi = df["ibi_detected"]

    return {
        "accuracy": accuracy_score(true_ibi, detected_ibi),
        "precision": precision_score(true_ibi, detected_ibi, zero_division=0),
        "recall": recall_score(true_ibi, detected_ibi, zero_division=0),
        "f1_score": f1_score(true_ibi, detected_ibi, zero_division=0),
        "detected_ibi": int(detected_ibi.sum()),
        "true_ibi": int(true_ibi.sum()),
        "correct_ibi": int((true_ibi & detected_ibi).sum()),
        "false_positive_ibi": int((~true_ibi & detected_ibi).sum()),
    }


def calculate_lower_marine_limits(
    df,
    feature_cols,
    marine_label="piscivorous",
):
    marine = df[df["cluster"] == marine_label]

    if marine.empty:
        raise ValueError(f"No rows found for marine label: {marine_label}")

    return marine[feature_cols].min()


def calculate_upper_marine_limits(
    df,
    feature_cols,
    marine_label="piscivorous",
):
    return calculate_lower_marine_limits(
        df,
        feature_cols,
        marine_label=marine_label,
    )


def get_lower_marine_limit_mask(
    df,
    feature_cols,
    lower_marine_limits,
    marine_label="piscivorous",
):
    if lower_marine_limits is None:
        lower_marine_limits = calculate_lower_marine_limits(
            df,
            feature_cols,
            marine_label=marine_label,
        )

    non_marine = df["cluster"] != marine_label
    above_all_limits = (df[feature_cols] >= lower_marine_limits).all(axis=1)

    return non_marine & above_all_limits


def get_upper_marine_limit_mask(
    df,
    feature_cols,
    upper_marine_limits,
    marine_label="piscivorous",
):
    return get_lower_marine_limit_mask(
        df,
        feature_cols=feature_cols,
        lower_marine_limits=upper_marine_limits,
        marine_label=marine_label,
    )


def add_ibi_prediction(
    df,
    mask,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
    upper_marine_limit_detection=False,
    feature_cols=None,
    upper_marine_limits=None,
    marine_label="piscivorous",
):
    df_results = df.copy()
    use_marine_limit_detection = (
        lower_marine_limit_detection or upper_marine_limit_detection
    )

    if lower_marine_limits is None:
        lower_marine_limits = upper_marine_limits

    if use_marine_limit_detection:
        if feature_cols is None:
            raise ValueError(
                "feature_cols must be provided when marine limit detection is enabled."
            )

        mask = mask | get_lower_marine_limit_mask(
            df_results,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
            marine_label=marine_label,
        )

    df_results["ibi_detected"] = mask
    df_results["ibi_pred"] = np.where(
        df_results["ibi_detected"],
        "ibi",
        df_results.get("svm_pred", "not_ibi"),
    )
    return df_results


def print_detection_report(
    metrics,
    mode,
    svm_threshold=None,
    gmm_threshold=None,
    combined_operator=None,
):
    print("\n==================== IBI DETECTION: ====================")
    print("Mode:", mode)

    if svm_threshold is not None:
        print("SVM threshold:", svm_threshold)

    if gmm_threshold is not None:
        print("GMM threshold:", gmm_threshold)

    if combined_operator is not None:
        print("Combined operator:", combined_operator)

    print("Detected IBI entries:", metrics["detected_ibi"])
    print("True IBI entries:", metrics["true_ibi"])
    print("Correctly detected IBI:", metrics["correct_ibi"])
    print("False positive IBI:", metrics["false_positive_ibi"])
    print("\nIBI Metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-score:", metrics["f1_score"])
    print("========================================================\n")


def bootstrap_metrics(df, metric_func, n_bootstraps=1000, ci=95, random_state=None):
    """
    Bootstrap metric estimation with confidence intervals.
    
    Args:
        df: DataFrame with data
        metric_func: Function that takes df and returns a dict of metrics
        n_bootstraps: Number of bootstrap samples
        ci: Confidence interval percentage (e.g., 95)
        random_state: Random seed for reproducibility
    
    Returns:
        dict with metric names as keys and dicts containing 'mean' and 'ci' as values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(df)
    bootstrap_results = {}
    
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[indices]
        
        # Calculate metrics for this bootstrap sample
        metrics = metric_func(df_boot)
        
        # Store results
        for key, value in metrics.items():
            if key not in bootstrap_results:
                bootstrap_results[key] = []
            bootstrap_results[key].append(value)
    
    # Calculate mean and confidence intervals
    alpha = (100 - ci) / 2
    result = {}
    
    for key, values in bootstrap_results.items():
        values_array = np.array(values)
        result[key] = {
            "mean": np.mean(values_array),
            "std": np.std(values_array),
            "ci_lower": np.percentile(values_array, alpha),
            "ci_upper": np.percentile(values_array, 100 - alpha),
            "values": values_array,  # Store all bootstrap values for later use
        }
    
    return result
