import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent


def gaussian_points(rng, n, center, sd, cluster, point_type="cluster"):
    rows = []
    for _ in range(n):
        rows.append(
            {
                "x1": rng.gauss(center[0], sd),
                "x2": rng.gauss(center[1], sd),
                "cluster": cluster,
                "point_type": point_type,
            }
        )
    return rows


def middle_points(rng, n, center, sd):
    rows = []
    for _ in range(n):
        rows.append(
            {
                "x1": rng.gauss(center[0], sd),
                "x2": rng.gauss(center[1], sd),
                "cluster": "middle",
                "point_type": "middle",
            }
        )
    return rows


def point(x1, x2, cluster, point_type="cluster"):
    return {
        "x1": x1,
        "x2": x2,
        "cluster": cluster,
        "point_type": point_type,
    }


def write_dataset(filename, rows):
    path = OUT_DIR / filename
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x1", "x2", "cluster", "point_type"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path.name}")


def plot_dataset(rows, filename):
    colors = {
        "A": "tab:blue",
        "B": "tab:orange",
        "middle": "tab:red",
    }
    labels = {
        "A": "Cluster A",
        "B": "Cluster B",
        "middle": "Middle points",
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for cluster in ["A", "B", "middle"]:
        cluster_rows = [row for row in rows if row["cluster"] == cluster]
        ax.scatter(
            [row["x1"] for row in cluster_rows],
            [row["x2"] for row in cluster_rows],
            label=labels[cluster],
            color=colors[cluster],
            s=70 if cluster == "middle" else 35,
            marker="x" if cluster == "middle" else "o",
            alpha=0.95 if cluster == "middle" else 0.75,
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=8,
        framealpha=0.9,
        borderaxespad=0,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0.6)

    path = OUT_DIR / filename
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote plot to {path.name}")


def make_dataset_0():
    rng = random.Random(42)
    rows = []
    rows += gaussian_points(rng, 60, center=(-3.0, 0.0), sd=0.35, cluster="A")
    rows += gaussian_points(rng, 60, center=(3.0, 0.0), sd=0.35, cluster="B")
    rows += middle_points(rng, 10, center=(0.0, 0.0), sd=0.12)
    return rows


def make_dataset_1():
    rng = random.Random(43)
    rows = []
    rows += gaussian_points(rng, 60, center=(-2.4, -0.3), sd=0.45, cluster="A")
    rows += gaussian_points(rng, 60, center=(2.4, 0.3), sd=0.45, cluster="B")
    rows += middle_points(rng, 10, center=(0.0, 0.0), sd=0.16)
    return rows


def make_dataset_2():
    rng = random.Random(44)
    rows = []
    rows += gaussian_points(rng, 60, center=(-2.8, 1.2), sd=0.4, cluster="A")
    rows += gaussian_points(rng, 60, center=(2.8, -1.2), sd=0.4, cluster="B")
    rows += middle_points(rng, 10, center=(0.0, 0.0), sd=0.14)
    return rows


def make_dataset_3():
    return [
        point(-2.0, -0.35, "A"),
        point(-2.0, 0.35, "A"),
        point(2.0, -0.35, "B"),
        point(2.0, 0.35, "B"),
        point(0.0, 0.0, "middle", "middle"),
        point(0.0, -0.2, "middle", "middle"),
        point(0.0, 0.2, "middle", "middle"),
        point(-0.1, 0.0, "middle", "middle"),
        point(0.1, 0.0, "middle", "middle"),
        point(-1, 0.0, "middle", "middle"),
        point(1, 0.0, "middle", "middle"),
    ]


def make_dataset_4():
    rng = random.Random(45)
    rows = []
    rows += gaussian_points(rng, 12, center=(-3.0, 0.0), sd=0.35, cluster="A")
    rows += gaussian_points(rng, 12, center=(3.0, 0.0), sd=0.35, cluster="B")
    rows += [
        point(0.0, -0.3, "middle", "middle"),
        point(0.0, -0.1, "middle", "middle"),
        point(0.0, 0.1, "middle", "middle"),
        point(0.0, 0.3, "middle", "middle"),
    ]
    return rows


def main():
    datasets = [
        ("0", make_dataset_0()),
        ("1", make_dataset_1()),
        ("2", make_dataset_2()),
        ("3", make_dataset_3()),
        ("4", make_dataset_4()),
    ]

    for name, rows in datasets:
        write_dataset(f"{name}.csv", rows)
        plot_dataset(rows, f"{name}.png")


if __name__ == "__main__":
    main()
