import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)

# =========================
# 1. Synthetic CLEAR dataset
# =========================
np.random.seed(42)

# Cluster A (links)
A = np.random.normal(loc=[-3, 0], scale=0.3, size=(50, 2))

# Cluster B (rechts)
B = np.random.normal(loc=[3, 0], scale=0.3, size=(50, 2))

data = np.vstack([A, B])
labels = np.array(["A"] * len(A) + ["B"] * len(B))

df = pd.DataFrame(data, columns=["x1", "x2"])
df["cluster"] = labels

# =========================
# 2. Train GMM
# =========================
X = df[["x1", "x2"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",#"spherical",
    random_state=42
)
gmm.fit(X_scaled)

# =========================
# 3. Test point
# =========================
test_points = np.array([
    [-3, 0],   # klar A
    [0, 0],    # Mitte
    [3, 0]     # klar B
])
test_point_scaled = scaler.transform(test_points)

probs = gmm.predict_proba(test_point_scaled)
print("Posterior probabilities:", probs)

# =========================
# 4. Simple Plot
# =========================
plt.figure()

# Punkte nach Klasse
plt.scatter(
    X_scaled[labels == "A", 0],
    X_scaled[labels == "A", 1],
    label="Class A"
)

plt.scatter(
    X_scaled[labels == "B", 0],
    X_scaled[labels == "B", 1],
    label="Class B"
)

# Testpunkt hervorheben
plt.scatter(
    test_point_scaled[:, 0],
    test_point_scaled[:, 1],
    color="red",
    s=100,
    label="Test Points"
)
plt.legend()

# speichern
plt.savefig("simple_points_plot.png", dpi=300)
plt.close()