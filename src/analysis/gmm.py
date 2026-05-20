import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# np.set_printoptions(suppress=True)

class GMM():
    def __init__(self, df_train, feature_cols):

        self.feature_cols = feature_cols

        # ===== Preprocessing. =====
        X_train = df_train[feature_cols].values

        # Standardizing.
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # ===== Train GMM. =====
        # 2 components, herbivorous and piscivorous.
        self.gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=42
        )
        self.gmm.fit(X_train_scaled)

    def save(self, df):
        # ===== Transform all points. =====
        X_all = df[self.feature_cols].values
        X_all_scaled = self.scaler.transform(X_all)

        # ===== Posterior probabilities =====
        probs = self.gmm.predict_proba(X_all_scaled)  # shape: (n_samples, 2)
        df["gmm_prob_0"] = probs[:, 0]
        df["gmm_prob_1"] = probs[:, 1]
        df["gmm_prob_distance"] = np.abs(probs[:, 0] - probs[:, 1])
        return df
