from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ibi_detection import detect_ibi, plot_ibi_threshold_metrics
from ibi_detection.common import calculate_lower_marine_limits


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CONFIG_NR = 4
SAVE_FILES = False


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
    # 3. Ibi Detection
    # =========================
    ibi_detection_mode = 'gmm_inverse_prob' #"gmm_ibi_cluster"
    svm_threshold = 1.0
    gmm_threshold = 0.02    
    combined_operator = "or"
    lower_marine_limit_detection = True
    lower_marine_limits = calculate_lower_marine_limits(df_train, feature_cols)

    df = detect_ibi(
        df_train=df_train,
        df_eval=df,
        feature_cols=feature_cols,
        mode=ibi_detection_mode,
        svm_threshold=svm_threshold,
        gmm_threshold=gmm_threshold,
        combined_operator=combined_operator,
        lower_marine_limit_detection=lower_marine_limit_detection,
        lower_marine_limits=lower_marine_limits,
    )
    if SAVE_FILES:
        ibi_metrics = plot_ibi_threshold_metrics(
            df,
            mode=ibi_detection_mode,
            start_threshold=0,
            end_threshold=1,
            step_size=0.01,
            fixed_svm_threshold=svm_threshold,
            fixed_gmm_threshold=gmm_threshold,
            combined_operator=combined_operator,
            lower_marine_limit_detection=lower_marine_limit_detection,
            feature_cols=feature_cols,
            lower_marine_limits=lower_marine_limits,
            df_train=df_train,
            output_path=output_dir / f"{CONFIG_NR}_{ibi_detection_mode}_ibi_threshold_metrics.png"
        )
        ibi_metrics.to_csv(output_dir / "ibi_threshold_metrics.csv", index=False)

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

    if SAVE_FILES:
        df.to_csv(output_dir / "analysis.csv", index=False)


def main():
    run_analysis()


if __name__ == "__main__":
    main()
