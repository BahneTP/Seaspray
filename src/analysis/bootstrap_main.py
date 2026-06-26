import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd

from bootstrap import run_bootstrap_evaluation, summarize_bootstrap_metrics


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
CONFIG_NR = 6


def run_bootstrap_analysis():
    output_dir = PROJECT_ROOT / "output" / str(CONFIG_NR)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{CONFIG_NR}.csv"

    df = pd.read_csv(dataset_path)
    feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

    ibi_detection_mode = "gmm"
    svm_threshold = 1.0
    gmm_threshold = 0.18
    combined_operator = "or"
    upper_marine_limit_detection = True
    n_iterations = 10
    random_state = 42

    metrics_df = run_bootstrap_evaluation(
        df=df,
        feature_cols=feature_cols,
        mode=ibi_detection_mode,
        n_iterations=n_iterations,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        upper_marine_limit_detection=upper_marine_limit_detection,
        random_state=random_state,
    )
    summary_df = summarize_bootstrap_metrics(metrics_df)

    metrics_path = output_dir / f"{CONFIG_NR}_{ibi_detection_mode}_bootstrap_metrics.csv"
    summary_path = output_dir / f"{CONFIG_NR}_{ibi_detection_mode}_bootstrap_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Bootstrap iterations:", n_iterations)
    print("Metrics saved to:", metrics_path)
    print("Summary saved to:", summary_path)
    print("\nBootstrap summary:")
    print(summary_df)


def main():
    log_path = PROJECT_ROOT / "output" / str(CONFIG_NR) / f"bootstrap_{CONFIG_NR}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print("Log file:", log_path)
            run_bootstrap_analysis()


if __name__ == "__main__":
    main()
