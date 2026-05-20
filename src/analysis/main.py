import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from svm import *
from gmm import *
from ibi_detection import *
import grid_plot

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# =========================
# 1. Load Data.
# =========================
config_nr = 2
DATASET_PATH = PROJECT_ROOT / "output" / "synthetic_datasets" / f"{config_nr}.csv"
PLOT_DIR = PROJECT_ROOT / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

# =========================
# 2. Split Data.
# =========================
df_labeled = df[df["cluster"].isin(["herbivorous", "piscivorous"])].copy()
df_train, df_test = train_test_split(
    df_labeled,
    test_size=0.2,
    random_state=42,
    stratify=df_labeled["cluster"]
)

# =========================
# 3. Run SVM & GMM
# =========================
svm = SVM(df_train, df_test, feature_cols)
svm.report()
df = svm.save(df)

gmm = GMM(df_train, feature_cols)
df = gmm.save(df)

# =========================
# 4. Ibi Detection
# =========================
df = detect_ibi(df, margin_threshold=1.0)
ibi_metrics = plot_ibi_threshold_metrics(
    df,
    start_threshold=0.1,
    end_threshold=1.5,
    step_size=0.1,
    output_path=PLOT_DIR / f"{config_nr}_ibi_threshold_metrics.png"
)

# =========================
# 5. Grid-plot
# ### WORK IN PROGRESS ###
# =========================
# grid_plot.run(df, output_path="gridplot.png")

# =========================
# 6. Data Saving
# =========================
print(df)
# df.to_csv(SCRIPT_DIR / "analysis.csv", index=False)
# ibi_metrics.to_csv(SCRIPT_DIR / "ibi_threshold_metrics.csv", index=False)
