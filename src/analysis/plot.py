import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# 7. PCA to 2D for a single plot with ALL points
# =========================
# Use the same scaling logic as the SVM
scaler = svm_model.named_steps["scaler"]
X_all_scaled = scaler.transform(X_all)

pca = PCA(n_components=2, random_state=42)
X_all_pca = pca.fit_transform(X_all_scaled)

df_pca = df_all_results.copy()
df_pca["PC1"] = X_all_pca[:, 0]
df_pca["PC2"] = X_all_pca[:, 1]

print("\nExplained variance ratio:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")

# =========================
# 8. Single PCA plot with all labels
# =========================
colors = {
    "herbivorous": "green",
    "piscivorous": "blue",
    "ibi": "orange"
}

plt.figure(figsize=(10, 8))

for cluster, color in colors.items():
    subset = df_pca[df_pca["cluster"] == cluster]

    marker = "o"
    size = 35
    linewidth = 0.0

    if cluster == "ibi":
        marker = "x"
        size = 70
        linewidth = 1.5

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        color=color,
        s=size,
        alpha=0.85,
        marker=marker,
        linewidths=linewidth,
        label=cluster
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA projection of all data (including IBI)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../output/pca_all_clusters.png", dpi=300)
plt.close()

print("PCA plot saved: ../output/pca_all_clusters.png")