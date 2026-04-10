import pandas as pd
import numpy as np

import svm

# =========================
# 1. Load Data.
# =========================
df = pd.read_csv("../../output/synthetic_datasets/2.csv")
svm.run(df)

