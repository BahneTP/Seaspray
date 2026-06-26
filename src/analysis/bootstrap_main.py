import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

from bootstrap import (
    run_bootstrap_evaluation,
    summarize_bootstrap_metrics,
    summarize_bootstrap_threshold_metrics,
)
from ibi_detection.plotting import plot_bootstrap_metrics


class Tee():
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CONFIG_NR = 5


def run_bootstrap_analysis(config_nr=CONFIG_NR):
    output_dir = PROJECT_ROOT / "output" / str(config_nr)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{config_nr}.csv"

    df = pd.read_csv(dataset_path)
    feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

    ibi_detection_mode = "gmm"
    svm_threshold = 1.0
    gmm_threshold = 0.18
    combined_operator = "or"
    upper_marine_limit_detection = True
    n_iterations = 10
    start_threshold = 0
    end_threshold = 1
    step_size = 0.01
    sweep_threshold = "auto"
    random_state = 42

    metrics_df, threshold_metrics_df = run_bootstrap_evaluation(
        df=df,
        feature_cols=feature_cols,
        mode=ibi_detection_mode,
        n_iterations=n_iterations,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        upper_marine_limit_detection=upper_marine_limit_detection,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        step_size=step_size,
        sweep_threshold=sweep_threshold,
        random_state=random_state,
        return_threshold_metrics=True,
    )
    summary_df = summarize_bootstrap_metrics(metrics_df)
    threshold_summary_df = summarize_bootstrap_threshold_metrics(threshold_metrics_df)

    metrics_path = output_dir / f"{config_nr}_{ibi_detection_mode}_bootstrap_metrics.csv"
    summary_path = output_dir / f"{config_nr}_{ibi_detection_mode}_bootstrap_summary.csv"
    threshold_summary_path = output_dir / f"{config_nr}_{ibi_detection_mode}_bootstrap_threshold_summary.csv"
    threshold_plot_path = output_dir / f"{config_nr}_{ibi_detection_mode}_bootstrap_threshold_metrics.png"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    threshold_summary_df.to_csv(threshold_summary_path, index=False)
    plot_bootstrap_metrics(
        threshold_summary_df[threshold_summary_df["split"] == "test"],
        output_path=threshold_plot_path,
    )

    print("Bootstrap iterations:", n_iterations)
    print("Threshold range:", start_threshold, "to", end_threshold)
    print("Threshold step size:", step_size)
    print("Metrics saved to:", metrics_path)
    print("Summary saved to:", summary_path)
    print("Threshold summary saved to:", threshold_summary_path)
    print("Threshold plot saved to:", threshold_plot_path)
    print("\nBootstrap summary:")
    print(summary_df)


def main():
    config_nr = int(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_NR
    log_path = PROJECT_ROOT / "output" / str(config_nr) / f"bootstrap_{config_nr}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print("Log file:", log_path)
            run_bootstrap_analysis(config_nr=config_nr)


if __name__ == "__main__":
    main()
