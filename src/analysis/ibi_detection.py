import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
        "false_positive_ibi": int((~true_ibi & detected_ibi).sum())
    }


def _get_ibi_mask(df, mode, svm_threshold, gmm_threshold, combined_operator):
    mode = mode.lower()
    combined_operator = combined_operator.lower()

    if mode not in ["svm", "gmm", "combined"]:
        raise ValueError("mode must be one of: 'svm', 'gmm', 'combined'")

    if combined_operator not in ["or", "and"]:
        raise ValueError("combined_operator must be either 'or' or 'and'")

    if mode == "svm":
        return df["svm_margin_distance"] <= svm_threshold

    if mode == "gmm":
        return df["gmm_prob_distance"] <= gmm_threshold

    svm_detected = df["svm_margin_distance"] <= svm_threshold
    gmm_detected = df["gmm_prob_distance"] <= gmm_threshold

    if combined_operator == "and":
        return svm_detected & gmm_detected

    return svm_detected | gmm_detected


def detect_ibi(
    df,
    mode="svm",
    svm_threshold=1.0,
    gmm_threshold=0.2,
    combined_operator="or",
    printing=True
):
    df_results = df.copy()

    df_results["ibi_detected"] = _get_ibi_mask(
        df_results,
        mode=mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator
    )
    df_results["ibi_pred"] = np.where(
        df_results["ibi_detected"],
        "ibi",
        df_results.get("svm_pred", "not_ibi")
    )

    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print("\n==================== IBI DETECTION: ====================")
        print("Mode:", mode)
        print("SVM threshold:", svm_threshold)
        print("GMM threshold:", gmm_threshold)
        if mode == "combined":
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

    return df_results


def plot_ibi_threshold_metrics(
    df,
    mode="svm",
    start_threshold=0.1,
    end_threshold=1.5,
    step_size=0.1,
    output_path="ibi_threshold_metrics.png",
    fixed_svm_threshold=1.0,
    fixed_gmm_threshold=0.2,
    combined_operator="or",
    sweep_threshold="auto"
):
    rows = []
    thresholds = np.arange(start_threshold, end_threshold + step_size / 2, step_size)
    mode = mode.lower()

    if sweep_threshold == "auto":
        sweep_threshold = "svm" if mode == "combined" else mode
    else:
        sweep_threshold = sweep_threshold.lower()

    if sweep_threshold not in ["svm", "gmm", "both"]:
        raise ValueError("sweep_threshold must be one of: 'auto', 'svm', 'gmm', 'both'")

    for threshold in thresholds:
        svm_threshold = fixed_svm_threshold
        gmm_threshold = fixed_gmm_threshold

        if sweep_threshold in ["svm", "both"]:
            svm_threshold = threshold

        if sweep_threshold in ["gmm", "both"]:
            gmm_threshold = threshold

        df_threshold = detect_ibi(
            df,
            mode=mode,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            printing=False
        )
        metrics = calculate_ibi_metrics(df_threshold)
        metrics["threshold"] = threshold
        metrics["svm_threshold"] = svm_threshold
        metrics["gmm_threshold"] = gmm_threshold
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))
    plt.plot(metrics_df["threshold"], metrics_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(metrics_df["threshold"], metrics_df["precision"], marker="o", label="Precision")
    plt.plot(metrics_df["threshold"], metrics_df["recall"], marker="o", label="Recall")
    plt.plot(metrics_df["threshold"], metrics_df["f1_score"], marker="o", label="F1-score")
    plt.xlabel(f"{sweep_threshold.upper()} threshold")
    plt.ylabel("Metric value")
    plt.title(f"IBI detection metrics by threshold ({mode})")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    best_row = metrics_df.loc[metrics_df["f1_score"].idxmax()]
    print("\n==================== IBI THRESHOLD SEARCH: ====================")
    print("Mode:", mode)
    print("Swept threshold:", sweep_threshold)
    print("Start threshold:", start_threshold)
    print("End threshold:", end_threshold)
    print("Step size:", step_size)
    print("Best F1 threshold:", best_row["threshold"])
    print("Best F1-score:", best_row["f1_score"])
    print("Diagram saved to:", output_path)
    print("===============================================================\n")

    return metrics_df
