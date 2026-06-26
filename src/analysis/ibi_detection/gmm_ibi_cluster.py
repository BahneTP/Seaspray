import re

import numpy as np

from models.gmm import run_gmm

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


def _get_gmm_prob_cols(df):
    prob_cols = [
        col for col in df.columns if re.fullmatch(r"gmm_prob_\d+", col)
    ]
    return sorted(prob_cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def add_gmm_cluster(df):
    df_results = df.copy()
    prob_cols = _get_gmm_prob_cols(df_results)
    df_results["gmm_cluster"] = np.argmax(df_results[prob_cols].to_numpy(), axis=1)
    return df_results


def summarize_gmm_clusters(df, feature_cols):
    return (
        df.groupby("gmm_cluster")[feature_cols]
        .agg(["mean", "std", "count"])
        .sort_index()
    )


def find_potential_ibi_cluster(df, feature_cols):
    centers = df.groupby("gmm_cluster")[feature_cols].mean().sort_index()

    if len(centers) != 3:
        raise ValueError("Expected exactly 3 GMM clusters to identify a middle cluster.")

    feature_mean = df[feature_cols].mean()
    feature_std = df[feature_cols].std(ddof=0).replace(0, 1)
    scaled_centers = (centers - feature_mean) / feature_std
    clusters = list(scaled_centers.index)
    distances = {}

    for i, cluster_a in enumerate(clusters):
        for cluster_b in clusters[i + 1:]:
            distance = np.linalg.norm(
                scaled_centers.loc[cluster_a] - scaled_centers.loc[cluster_b]
            )
            distances[(cluster_a, cluster_b)] = distance

    outer_clusters = max(distances, key=distances.get)
    potential_ibi_cluster = next(
        cluster for cluster in clusters if cluster not in outer_clusters
    )

    return potential_ibi_cluster, summarize_gmm_clusters(df, feature_cols), distances


def print_gmm_cluster_report(potential_ibi_cluster, cluster_summary, distances):
    print("\n==================== GMM CLUSTERS: ====================")
    print("Cluster summary:")
    print(cluster_summary)
    print("\nDistances between scaled cluster centers:")

    for clusters, distance in distances.items():
        print(f"{clusters[0]} - {clusters[1]}: {distance}")

    print("Potential IBI cluster:", potential_ibi_cluster)
    print("========================================================\n")


def add_gmm_ibi_prob(df, potential_ibi_cluster):
    df_results = df.copy()
    df_results["gmm_ibi_prob"] = df_results[f"gmm_prob_{potential_ibi_cluster}"]
    return df_results


def prepare_gmm_ibi_cluster(
    df_train,
    df_eval,
    feature_cols,
    df_model=None,
    potential_ibi_cluster=None,
    printing=False,
):
    df_results = df_model.copy() if df_model is not None else run_gmm(
        df_train,
        df_eval,
        feature_cols,
    )
    cluster_summary = None
    distances = None

    if potential_ibi_cluster is None:
        df_train_model = run_gmm(df_train, df_train, feature_cols)
        df_clustered = add_gmm_cluster(df_train_model)
        potential_ibi_cluster, cluster_summary, distances = find_potential_ibi_cluster(
            df_clustered,
            feature_cols,
        )

    if printing and cluster_summary is not None:
        print_gmm_cluster_report(potential_ibi_cluster, cluster_summary, distances)

    return add_gmm_ibi_prob(df_results, potential_ibi_cluster), potential_ibi_cluster


def detect(
    df_train,
    df_eval,
    feature_cols,
    threshold=0.2,
    df_model=None,
    potential_ibi_cluster=None,
    printing=True,
    lower_marine_limit_detection=False,
    lower_marine_limits=None,
):
    df_results, potential_ibi_cluster = prepare_gmm_ibi_cluster(
        df_train=df_train,
        df_eval=df_eval,
        feature_cols=feature_cols,
        df_model=df_model,
        potential_ibi_cluster=potential_ibi_cluster,
        printing=printing,
    )
    df_results = add_ibi_prediction(
        df_results,
        df_results["gmm_ibi_prob"] >= threshold,
        lower_marine_limit_detection=lower_marine_limit_detection,
        feature_cols=feature_cols,
        lower_marine_limits=lower_marine_limits,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(metrics, mode="gmm_ibi_cluster", gmm_threshold=threshold)

    return df_results
