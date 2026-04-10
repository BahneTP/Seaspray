import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load Data.
# =========================
df = pd.read_csv("../../output/synthetic_datasets/2.csv")

feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

# =========================
# 2. Preprocessing.
# =========================
train_df = df[df["cluster"].isin(["herbivorous", "piscivorous"])].copy()
X_train = train_df[feature_cols].values

# Standardizing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =========================
# 4. Train GMM.
# =========================
# 2 components, herbivorous and piscivorous.
gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=42
)
gmm.fit(X_train_scaled)

# =========================
# 5. Transform all points.
# =========================
X_all = df[feature_cols].values
X_all_scaled = scaler.transform(X_all)

# =========================
# 6. Posterior probabilities
# =========================
probs = gmm.predict_proba(X_all_scaled)   # shape: (n_samples, 2)

# Höchste Cluster-Wahrscheinlichkeit
max_prob = probs.max(axis=1)

# Uncertainty Score:
# hoch = unsicher, niedrig = sicher
uncertainty = 1 - max_prob

df["gmm_prob_cluster_0"] = probs[:, 0]
df["gmm_prob_cluster_1"] = probs[:, 1]
df["gmm_max_prob"] = max_prob
df["gmm_uncertainty"] = uncertainty

# =========================
# 7. Optional: Log-Likelihood als atypischkeitsmaß
# =========================
# niedrig = Punkt passt schlecht ins Modell
log_likelihood = gmm.score_samples(X_all_scaled)
df["gmm_log_likelihood"] = log_likelihood

# =========================
# 8. Threshold setzen
# =========================
# Variante A: Threshold aus terrestrischen Trainingsdaten
train_probs = gmm.predict_proba(X_train_scaled)
train_max_prob = train_probs.max(axis=1)
train_uncertainty = 1 - train_max_prob

# z.B. oberste 10% Unsicherheit der bekannten terrestrischen Punkte
uncertainty_threshold = np.quantile(train_uncertainty, 0.9)

print(f"GMM uncertainty threshold: {uncertainty_threshold:.4f}")

# =========================
# 9. Unsichere Punkte markieren
# =========================
df["gmm_uncertain"] = df["gmm_uncertainty"] > uncertainty_threshold

# =========================
# 10. Speziell IBI-Kandidaten markieren
# =========================
df["ibi_candidate_by_gmm"] = df["gmm_uncertain"]

# Wenn du nur echte ibi-Zeilen anschauen willst:
df["true_ibi_detected_by_gmm"] = (df["cluster"] == "ibi") & (df["gmm_uncertain"])

# =========================
# 11. Überblick
# =========================
print("\nUnsichere Punkte insgesamt:")
print(df["gmm_uncertain"].value_counts())

print("\nWie viele echte IBI-Punkte werden als unsicher erkannt?")
ibi_df = df[df["cluster"] == "ibi"].copy()
if len(ibi_df) > 0:
    print(ibi_df["gmm_uncertain"].value_counts())

print("\nBeispielausgabe:")
print(
    df[["cluster", "gmm_max_prob", "gmm_uncertainty", "gmm_log_likelihood", "gmm_uncertain"]]
    .head(15)
)

# =========================
# 12. Speichern
# =========================
df.to_csv("gmm_results.csv", index=False)