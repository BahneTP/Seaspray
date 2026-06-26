import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gmm_r import GMM_R
from ibi_detection import detect_ibi, plot_ibi_threshold_metrics
from svm import SVM


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
CONFIG_NR = 4


def run_analysis():
    # =========================
    # 1. Load Data.
    # =========================
    output_dir = PROJECT_ROOT / "output" / str(CONFIG_NR)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{CONFIG_NR}.csv"

    df = pd.read_csv(dataset_path)
    feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

    # =========================
    # 2. Split Data.
    # =========================
    df_labeled = df[df["cluster"].isin(["herbivorous", "piscivorous", "ibi"])].copy()
    df_train, df_test = train_test_split(
        df_labeled,
        test_size=0.2,
        random_state=42,
        stratify=df_labeled["cluster"]
    )

    # =========================
    # 3. Run GMM & SVM
    # =========================
    gmm = GMM_R(df_train, feature_cols)
    df = gmm.save(df)

    # svm = SVM(df_train, df_test, feature_cols)
    # svm.report()
    # df = svm.save(df)

    # =========================
    # 4. Ibi Detection
    # =========================
    ibi_detection_mode = "gmm"
    svm_threshold = 1.0
    gmm_threshold = 0.18    
    combined_operator = "or"

    df = detect_ibi(
        df,
        mode=ibi_detection_mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator
    )
    ibi_metrics = plot_ibi_threshold_metrics(
        df,
        mode=ibi_detection_mode,
        start_threshold=0,
        end_threshold=1,
        step_size=0.01,
        fixed_svm_threshold=svm_threshold,
        fixed_gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        output_path=output_dir / f"{CONFIG_NR}_{ibi_detection_mode}_ibi_threshold_metrics.png"
    )

    # =========================
    # 6. Grid-plot
    # ### WORK IN PROGRESS ###
    # =========================
    # grid_plot.run(df, output_path=output_dir / "gridplot.png")

    # =========================
    # 7. Data Saving
    # =========================
    # print(df[df["cluster"] == "ibi"])
    print(df)
    # df.to_csv(output_dir / "analysis.csv", index=False)
    # ibi_metrics.to_csv(output_dir / "ibi_threshold_metrics.csv", index=False)


def main():
    log_path = PROJECT_ROOT / "output" / str(CONFIG_NR) / f"analysis_{CONFIG_NR}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print("Log file:", log_path)
            run_analysis()


if __name__ == "__main__":
    main()
