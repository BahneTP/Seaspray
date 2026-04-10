import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Synthetic symmetric dataset
# =========================
data = np.array([
    [-1,  1],   # class A
    [-1, -1],   # class A
    [ 1,  1],   # class B
    [ 1, -1]    # class B
])

labels = np.array(["A", "A", "B", "B"])

df = pd.DataFrame(data, columns=["x1", "x2"])
df["cluster"] = labels

# =========================
# 2. Train only on A + B
# =========================
X = df[["x1", "x2"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=42
)
gmm.fit(X_scaled)

# =========================
# 3. Test point in the middle
# =========================
test_point = np.array([[0, 0]])
test_point_scaled = scaler.transform(test_point)

probs = gmm.predict_proba(test_point_scaled)

print("Posterior probabilities:", probs)