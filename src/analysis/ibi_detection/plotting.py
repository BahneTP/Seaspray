from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import calculate_ibi_metrics
from .ibi_detection import detect_ibi


def _detect_ibi_for_plot(
    df,
    mode,
    svm_threshold,
    gmm_threshold,
    combined_operator,
    lower_marine_limit_detection=False,
    feature_cols=None,
    lower_marine_limits=None,
    df_train=None,
):
    return detect_ibi(
        df_train=df_train if df_train is not None else df,
        df_eval=df,
        feature_cols=feature_cols,
        mode=mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        printing=False,
        lower_marine_limit_detection=lower_marine_limit_detection,
        lower_marine_limits=lower_marine_limits,
    )


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
    sweep_threshold="auto",
    lower_marine_limit_detection=False,
    feature_cols=None,
    lower_marine_limits=None,
    df_train=None,
):
    rows = []
    thresholds = np.arange(start_threshold, end_threshold + step_size / 2, step_size)
    mode = mode.lower()

    if sweep_threshold == "auto":
        sweep_threshold = "svm" if mode in ["svm", "svm_margin", "combined"] else "gmm"
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

        df_threshold = _detect_ibi_for_plot(
            df,
            mode=mode,
            svm_threshold=svm_threshold,
            gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
            df_train=df_train,
        )
        metrics = calculate_ibi_metrics(df_threshold)
        metrics["threshold"] = threshold
        metrics["svm_threshold"] = svm_threshold
        metrics["gmm_threshold"] = gmm_threshold
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))
    # Colorblind-friendly palette and thicker lines (no markers)
    plt.plot(
        metrics_df["threshold"],
        metrics_df["accuracy"],
        label="Accuracy",
        color="#0072B2",
        linewidth=2.5,
    )
    plt.plot(
        metrics_df["threshold"],
        metrics_df["precision"],
        label="Precision",
        color="#E69F00",
        linewidth=2.5,
    )
    plt.plot(
        metrics_df["threshold"],
        metrics_df["recall"],
        label="Recall",
        color="#009E73",
        linewidth=2.5,
    )
    plt.plot(
        metrics_df["threshold"],
        metrics_df["f1_score"],
        label="F1-score",
        color="#CC79A7",
        linewidth=2.5,
    )

    # Slightly larger axis labels, no title for cleaner export
    plt.xlabel(f"{sweep_threshold.upper()} threshold", fontsize=12)
    plt.ylabel("Metric value", fontsize=12)
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


def plot_bootstrap_metrics(
    bootstrap_results_df,
    output_path="bootstrap_ibi_metrics.png",
):
    """
    Plot bootstrap metrics with confidence intervals as bands around lines.
    
    Args:
        bootstrap_results_df: DataFrame with columns:
            threshold, metric, mean, q025, q975
        output_path: Path to save the figure
    """
    plt.figure(figsize=(9, 5))

    # Colorblind-friendly palette
    colors = {
        "accuracy": "#0072B2",
        "precision": "#E69F00",
        "recall": "#009E73",
        "f1_score": "#CC79A7",
    }

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    for metric in metrics:
        metric_data = bootstrap_results_df[bootstrap_results_df["metric"] == metric]
        metric_data = metric_data.sort_values("threshold")

        # Plot mean line
        plt.plot(
            metric_data["threshold"],
            metric_data["mean"],
            label=metric.replace("_", "-").title(),
            color=colors[metric],
            linewidth=2.5,
        )

        # Plot confidence interval as band
        plt.fill_between(
            metric_data["threshold"],
            metric_data["q025"],
            metric_data["q975"],
            color=colors[metric],
            alpha=0.15,  # Semi-transparent
        )

    plt.xlabel("GMM threshold", fontsize=12)
    plt.ylabel("Metric value", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nBootstrap plot saved to: {output_path}")
