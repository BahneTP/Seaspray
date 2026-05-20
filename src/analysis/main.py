import pandas as pd
import numpy as np

from svm import *
from gmm import *
import grid_plot

# =========================
# 1. Load Data.
# =========================
# df = pd.read_csv("../../output/synthetic_datasets/0.csv")
# feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

# # =========================
# # 2. Run SVM & GMM
# # =========================
# svm = SVM(df, feature_cols)
# svm.report()ss
# df = svm.save(df)

# gmm = GMM(df, feature_cols)
# df = gmm.save(df)
# print(df)
# df.to_csv("analysis.csv", index=False)

df = pd.read_csv("analysis.csv")

# for i in range(len(df["gmm_prob_0"])):
#     if df["cluster"][i] == "herbivorous":
#         print(df["gmm_prob_0"][i])

# =========================
# 3. Grid-plot
# =========================
grid_plot.run(df, output_path="gridplot.png")