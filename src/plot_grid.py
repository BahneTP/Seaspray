import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------
# CONFIG
# -------------------------------------------
synthetic_df = pd.read_csv("../data/synthetic_dataset.csv")

FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

FEATURE_LABELS = {
    "δ13C coll": r"$\delta^{13}C_{\mathrm{collagen}}$" + "\n[‰ VPDB]",
    "δ15N coll": r"$\delta^{15}N_{\mathrm{collagen}}$" + "\n[‰ AIR]",
    "δ13C carb": r"$\delta^{13}C_{\mathrm{carbonate}}$" + "\n[‰ VPDB]",
    "δ18O carb": r"$\delta^{18}O_{\mathrm{carbonate}}$" + "\n[‰ VPDB]",
}

colors = {
    "herbivorous": "green",
    "piscivorous": "blue",
    "ibi": "orange"
}

lower_limit_color = "black"

# -------------------------------------------
# Data
# -------------------------------------------
pisc = synthetic_df[synthetic_df["cluster"] == "piscivorous"]

n_features = len(FEATURES)

fig, axes = plt.subplots(
    n_features,
    n_features,
    figsize=(12, 12),
    squeeze=False
)

# -------------------------------------------
# consistent axis limits
# -------------------------------------------
feature_limits = {}

for feature in FEATURES:

    vmin = synthetic_df[feature].min()
    vmax = synthetic_df[feature].max()

    padding = (vmax - vmin) * 0.05

    feature_limits[feature] = (
        vmin - padding,
        vmax + padding
    )

# -------------------------------------------
# Plot matrix
# -------------------------------------------
for row, y_feature in enumerate(FEATURES):

    for col, x_feature in enumerate(FEATURES):

        ax = axes[row, col]

        ax.set_xlim(feature_limits[x_feature])
        ax.set_ylim(feature_limits[y_feature])

        # -------------------------------------------
        # Diagonal
        # -------------------------------------------
        if row == col:

            ax.text(
                0.5,
                0.5,
                FEATURE_LABELS[x_feature],
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes
            )

            ax.set_xticks([])
            ax.set_yticks([])

            continue

        # -------------------------------------------
        # Scatter points
        # -------------------------------------------
        for cluster, color in colors.items():

            subset = synthetic_df[synthetic_df["cluster"] == cluster]

            ax.scatter(
                subset[x_feature],
                subset[y_feature],
                color=color,
                s=20,
                alpha=0.85
            )

        # -------------------------------------------
        # Lower Marine Limit
        # -------------------------------------------
        lower_limit_x = pisc[x_feature].min()
        lower_limit_y = pisc[y_feature].min()

        ax.axvline(
            x=lower_limit_x,
            color=lower_limit_color,
            linestyle="--",
            linewidth=1.5
        )

        ax.axhline(
            y=lower_limit_y,
            color=lower_limit_color,
            linestyle="--",
            linewidth=1.5
        )

        # -------------------------------------------
        # TICKS ONLY OUTSIDE
        # -------------------------------------------
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False
        )

        # bottom ticks
        if row == n_features - 1:

            ax.tick_params(
                bottom=True,
                labelbottom=True
            )

        # top ticks
        if row == 0:

            ax.tick_params(
                top=True,
                labeltop=True
            )

        # left ticks
        if col == 0:

            ax.tick_params(
                left=True,
                labelleft=True
            )

        # right ticks
        if col == n_features - 1:

            ax.tick_params(
                right=True,
                labelright=True
            )

        # -------------------------------------------
        # Labels
        # -------------------------------------------
        if row == n_features - 1:

            ax.set_xlabel(FEATURE_LABELS[x_feature], fontsize=10)

        if row == 0:

            ax.xaxis.set_label_position("top")
            ax.set_xlabel(FEATURE_LABELS[x_feature], fontsize=10)

        if col == 0:

            ax.set_ylabel(FEATURE_LABELS[y_feature], fontsize=10)

        if col == n_features - 1:

            ax.yaxis.set_label_position("right")
            ax.set_ylabel(FEATURE_LABELS[y_feature], fontsize=10)

# -------------------------------------------
# Layout
# -------------------------------------------
plt.subplots_adjust(
    wspace=0.02,
    hspace=0.02
)

plt.savefig(
    "../output/synthetic_plot_matrix.png",
    dpi=300
)

print("Plot saved.")