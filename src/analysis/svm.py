import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class SVM():
    def __init__(self, df_train, df_test, feature_cols):
        self.feature_cols = feature_cols

        self.X_train = df_train[feature_cols]
        self.y_train = df_train["cluster"]
        self.X_test = df_test[feature_cols]
        self.y_test = df_test["cluster"]

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
        df_all_results["svm_margin_distance"] = np.abs(self.svm_model.decision_function(X_all))

        return df_all_results
