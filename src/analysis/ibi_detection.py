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


def detect_ibi(df, margin_threshold=1.0, printing=True):
    df_results = df.copy()

    df_results["svm_margin_distance"] = df_results["svm_margin"].abs()
    df_results["ibi_detected"] = df_results["svm_margin_distance"] <= margin_threshold
    df_results["ibi_pred"] = np.where(
        df_results["ibi_detected"],
        "ibi",
        df_results["svm_pred"]
    )

    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print("\n==================== IBI DETECTION: ====================")
        print("Margin threshold:", margin_threshold)
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
    start_threshold=0.1,
    end_threshold=1.5,
    step_size=0.1,
    output_path="ibi_threshold_metrics.png"
):
    rows = []
    thresholds = np.arange(start_threshold, end_threshold + step_size / 2, step_size)

    for threshold in thresholds:
        df_threshold = detect_ibi(df, margin_threshold=threshold, printing=False)
        metrics = calculate_ibi_metrics(df_threshold)
        metrics["threshold"] = threshold
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))
    plt.plot(metrics_df["threshold"], metrics_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(metrics_df["threshold"], metrics_df["precision"], marker="o", label="Precision")
    plt.plot(metrics_df["threshold"], metrics_df["recall"], marker="o", label="Recall")
    plt.plot(metrics_df["threshold"], metrics_df["f1_score"], marker="o", label="F1-score")
    plt.xlabel("SVM margin threshold")
    plt.ylabel("Metric value")
    plt.title("IBI detection metrics by threshold")
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
    print("Start threshold:", start_threshold)
    print("End threshold:", end_threshold)
    print("Step size:", step_size)
    print("Best F1 threshold:", best_row["threshold"])
    print("Best F1-score:", best_row["f1_score"])
    print("Diagram saved to:", output_path)
    print("===============================================================\n")

    return metrics_df
