import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_DIR = BASE_DIR / "datasets"
PLOT_DIR = SCRIPT_DIR


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compact_probability(value):
    text = "{:.2f}".format(float(value))
    if text.startswith("0"):
        return text[1:]
    return text


def plot_dataset(dataset, dataset_rows, result_rows):
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
    results_by_row = {row["row_index"]: row for row in result_rows}

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    middle_rows = [row for row in dataset_rows if row["cluster"] == "middle"]
    ax.scatter(
        [float(row["x1"]) for row in middle_rows],
        [float(row["x2"]) for row in middle_rows],
        label=labels["middle"],
        color=colors["middle"],
        s=45,
        marker="x",
        alpha=0.95,
    )

    y_values = [float(row["x2"]) for row in middle_rows]
    y_offset = (max(y_values) - min(y_values)) * 0.045 if len(y_values) > 1 else 0.1

    for row_index, row in enumerate(dataset_rows):
        result = results_by_row.get(str(row_index))
        if result is None:
            continue

        x = float(row["x1"])
        y = float(row["x2"])
        label = "{}/{}".format(
            compact_probability(result["sklearn_prob_a"]),
            compact_probability(result["mclust_prob_a"]),
        )
        ax.text(x, y + y_offset, label, ha="center", va="bottom", fontsize=8)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.margins(x=0.12, y=0.24)
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

    path = PLOT_DIR / "{}.png".format(dataset)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("Wrote {}".format(path))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 plot.py <gmm_results.csv>")

    grouped_results = defaultdict(list)
    for row in read_csv(Path(sys.argv[1])):
        grouped_results[row["dataset"]].append(row)

    for dataset in sorted(grouped_results, key=int):
        dataset_rows = read_csv(DATASET_DIR / "{}.csv".format(dataset))
        plot_dataset(dataset, dataset_rows, grouped_results[dataset])


if __name__ == "__main__":
    main()
