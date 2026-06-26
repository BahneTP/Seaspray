import numpy as np
import re

from .common import add_ibi_prediction, calculate_ibi_metrics, print_detection_report


FEATURE_COLS = ["δ13C coll", "δ15N coll", "δ13C carb", "δ18O carb"]


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


def summarize_gmm_clusters(df, feature_cols=FEATURE_COLS):
    return (
        df.groupby("gmm_cluster")[feature_cols]
        .agg(["mean", "std", "count"])
        .sort_index()
    )


def find_potential_ibi_cluster(df, feature_cols=FEATURE_COLS):
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

    return (
        potential_ibi_cluster,
        summarize_gmm_clusters(df, feature_cols),
        distances,
    )


def print_gmm_cluster_report(potential_ibi_cluster, cluster_summary, distances):
    print("\n==================== GMM CLUSTERS: ====================")
    print("Cluster summary:")
    print(cluster_summary)
    print("\nDistances between scaled cluster centers:")

    for clusters, distance in distances.items():
        print(f"{clusters[0]} - {clusters[1]}: {distance}")

    print("Potential IBI cluster:", potential_ibi_cluster)
    print("========================================================\n")


def _get_ibi_prob(df, potential_ibi_cluster):
    ibi_prob_col = f"gmm_prob_{potential_ibi_cluster}"
    return df[ibi_prob_col]


def _prepare_gmm_metric(df):
    df_clustered = add_gmm_cluster(df)
    (
        potential_ibi_cluster,
        cluster_summary,
        distances,
    ) = find_potential_ibi_cluster(df_clustered)

    df_results = df.copy()
    df_results["gmm_ibi_prob"] = _get_ibi_prob(
        df_results,
        potential_ibi_cluster,
    )

    return df_results, potential_ibi_cluster, cluster_summary, distances


def add_gmm_ibi_prob(df):
    df_results, _, _, _ = _prepare_gmm_metric(df)
    return df_results


def get_gmm_mask(df, threshold):
    df_results = add_gmm_ibi_prob(df)
    return df_results["gmm_ibi_prob"] >= threshold


def detect_gmm(df, threshold=0.2, printing=True):
    df_results, potential_ibi_cluster, cluster_summary, distances = _prepare_gmm_metric(
        df
    )

    if printing:
        print_gmm_cluster_report(potential_ibi_cluster, cluster_summary, distances)

    df_results = add_ibi_prediction(
        df_results,
        df_results["gmm_ibi_prob"] >= threshold,
    )
    metrics = calculate_ibi_metrics(df_results)

    if printing:
        print_detection_report(
            metrics,
            mode="gmm",
            gmm_threshold=threshold,
        )

    return df_results


def apply_gmm_threshold(df, threshold=0.2):
    """
    Apply GMM threshold to data that already has gmm_ibi_prob column.
    Used for bootstrapping to avoid re-clustering.
    
    Args:
        df: DataFrame with 'cluster' and 'gmm_ibi_prob' columns
        threshold: Detection threshold
    
    Returns:
        DataFrame with 'ibi_detected' and metrics
    """
    df_results = df.copy()
    
    if 'gmm_ibi_prob' not in df_results.columns:
        raise ValueError("DataFrame must have 'gmm_ibi_prob' column")
    
    df_results = add_ibi_prediction(
        df_results,
        df_results["gmm_ibi_prob"] >= threshold,
    )
    
    return df_results
