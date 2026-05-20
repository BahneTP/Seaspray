import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class SVM():
    def __init__(self, df, feature_cols):
        # ===== Splitting the data. =====
        df_train = df[df["cluster"].isin(["herbivorous", "piscivorous"])].copy()
        self.feature_cols = feature_cols

        X = df_train[feature_cols]
        y = df_train["cluster"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ===== SVM Pipeline =====
        # Standardizing.
        self.svm_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=1.0))
        ])

        # Train.
        self.svm_model.fit(self.X_train, self.y_train)

    def report(self):
    # Predictions on Test-data.

        y_pred = self.svm_model.predict(self.X_test)
        print("\n==================== SVM REPORT: ====================")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("=====================================================\n")

    def save(self, df):
    # Apply trained SVM to ALL data (including IBI)

        X_all = df[self.feature_cols]

        df_all_results = df.copy()
        df_all_results["svm_pred"] = self.svm_model.predict(X_all)
        df_all_results["svm_margin"] = self.svm_model.decision_function(X_all)

        return df_all_results