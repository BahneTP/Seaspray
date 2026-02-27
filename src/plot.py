import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

synthetic_df = pd.read_csv("../data/synthetic_dataset.csv")

iso1 = FEATURES[0]
iso2 = FEATURES[1]

plt.figure()
plt.scatter(synthetic_df[iso1], synthetic_df[iso2])
plt.xlabel(iso1)
plt.ylabel(iso2)
plt.title("Synthetic Data: First Two Isotopes")

plt.savefig("../output/synthetic_plot.png")