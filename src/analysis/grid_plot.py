import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(df, output_path="../output/plots/svm_gmm_plot_matrix.png"):
    FEATURES = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

    FEATURE_LABELS = {
        "δ13C coll": r"$\delta^{13}C_{\mathrm{collagen}}$" + "\n[‰ VPDB]",
        "δ15N coll": r"$\delta^{15}N_{\mathrm{collagen}}$" + "\n[‰ AIR]",
        "δ13C carb": r"$\delta^{13}C_{\mathrm{carbonate}}$" + "\n[‰ VPDB]",
        "δ18O carb": r"$\delta^{18}O_{\mathrm{carbonate}}$" + "\n[‰ VPDB]",
    }

    svm_colors = {
        "herbivorous": "green",
        "piscivorous": "blue",
        "ibi": "orange",
    }

    # WICHTIG:
    # Hier Zuordnung der GMM-Komponenten zu den Klassen festlegen.
    # Falls dein GMM anders herum codiert ist, hier tauschen.
    COMPONENT_PROB_BY_CLUSTER = {
        "herbivorous": "gmm_prob_0",
        "piscivorous": "gmm_prob_1",
    }

    lower_limit_color = "black"

    df = df.copy()

    # -------------------------------------------
    # Hilfsfunktionen
    # -------------------------------------------
    def get_assigned_uncertainty(row):
        pred = row["svm_pred"]

        # Falls eine Klasse vorkommt, die keine GMM-Komponente zugeordnet hat
        if pred not in COMPONENT_PROB_BY_CLUSTER:
            return np.nan

        prob_col = COMPONENT_PROB_BY_CLUSTER[pred]
        assigned_prob = row[prob_col]

        return assigned_prob

    # -------------------------------------------
    # Data preparation
    # -------------------------------------------
    pisc = df[df["cluster"] == "piscivorous"].copy()

    df["gmm_uncertainty_assigned"] = df.apply(get_assigned_uncertainty, axis=1)

    valid_unc = df["gmm_uncertainty_assigned"].dropna()

    if len(valid_unc) == 0:
        raise ValueError("No valid assigned GMM uncertainty values could be computed.")

    unc_min = valid_unc.min()
    unc_max = valid_unc.max()

    if np.isclose(unc_max, unc_min):
        unc_max = unc_min + 1e-12

    df["unc_norm"] = (df["gmm_uncertainty_assigned"])# - unc_min) / (unc_max - unc_min)
    # print(df["unc_norm"])
    # Punktgröße und Alpha nur aus uncertainty
    df["plot_size"] = 15 + 85 * df["unc_norm"]
    df["plot_alpha"] = 0.20 + 0.80 * df["unc_norm"]

    n_features = len(FEATURES)

    fig, axes = plt.subplots(
        n_features,
        n_features,
        figsize=(12, 12),
        squeeze=False
    )

    # -------------------------------------------
    # Consistent axis limits
    # -------------------------------------------
    feature_limits = {}
    for feature in FEATURES:
        vmin = df[feature].min()
        vmax = df[feature].max()
        padding = (vmax - vmin) * 0.05
        feature_limits[feature] = (vmin - padding, vmax + padding)

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
            # Upper triangle:
            # SVM cluster assignment
            # -------------------------------------------
            if row < col:
                for pred_class, color in svm_colors.items():
                    subset = df[df["svm_pred"] == pred_class]

                    if len(subset) == 0:
                        continue

                    ax.scatter(
                        subset[x_feature],
                        subset[y_feature],
                        color=color,
                        s=20,
                        alpha=0.85,
                        edgecolors="none",
                        linewidths=0.2
                    )

            # -------------------------------------------
            # Lower triangle:
            # color = assigned class
            # size + alpha = uncertainty of assigned class
            # -------------------------------------------
            else:
                subset_valid = df.dropna(subset=["gmm_uncertainty_assigned", "unc_norm"])

                for _, row_data in subset_valid.iterrows():
                    pred_class = row_data["svm_pred"]
                    color = svm_colors.get(pred_class, "gray")

                    ax.scatter(
                        row_data[x_feature],
                        row_data[y_feature],
                        color=color,
                        s=row_data["plot_size"],
                        alpha=row_data["plot_alpha"],
                        edgecolors="none",
                        linewidths=0.25
                    )

            # -------------------------------------------
            # Lower marine limit
            # -------------------------------------------
            if len(pisc) > 0:
                lower_limit_x = pisc[x_feature].min()
                lower_limit_y = pisc[y_feature].min()

                ax.axvline(
                    x=lower_limit_x,
                    color=lower_limit_color,
                    linestyle="--",
                    linewidth=1.2
                )

                ax.axhline(
                    y=lower_limit_y,
                    color=lower_limit_color,
                    linestyle="--",
                    linewidth=1.2
                )

            # -------------------------------------------
            # Ticks only outside
            # -------------------------------------------
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False
            )

            if row == n_features - 1:
                ax.tick_params(bottom=True, labelbottom=True)

            if row == 0:
                ax.tick_params(top=True, labeltop=True)

            if col == 0:
                ax.tick_params(left=True, labelleft=True)

            if col == n_features - 1:
                ax.tick_params(right=True, labelright=True)

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
    # Layout and save
    # -------------------------------------------
    plt.subplots_adjust(
        wspace=0.02,
        hspace=0.02
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Plot saved to: {output_path}")