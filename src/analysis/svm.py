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


def run(df):

    # =========================
    # 2. Splitting the data.
    # =========================
    df_svm = df[df["cluster"].isin(["herbivorous", "piscivorous"])].copy()

    X = df_svm[["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]]
    y = df_svm["cluster"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # 3. SVM Pipeline
    # =========================
    # Standardizing.
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0))
    ])

    # Train.
    svm_model.fit(X_train, y_train)

    # =========================
    # 4. Predictions on Test-data.
    # =========================
    y_pred = svm_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # =========================
    # 5. Decision function / Margin Scores
    # =========================
    # Distance to Margin.
    margin_scores = svm_model.decision_function(X)

    results = df_svm.copy()
    results["svm_margin"] = margin_scores
    results["svm_pred"] = svm_model.predict(X)
    print(results.head())

    # =========================
    # 6. Apply trained SVM to ALL data (including IBI)
    # =========================
    feature_cols = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]

    X_all = df[feature_cols]

    df_all_results = df.copy()
    df_all_results["svm_pred"] = svm_model.predict(X_all)
    df_all_results["svm_margin"] = svm_model.decision_function(X_all)

    # Optional speichern
    df_all_results.to_csv("../../output/synthetic_datasets/svm_results.csv", index=False)

    print("\nIBI points with SVM prediction and margin:")
    print(
        df_all_results.head()
    )