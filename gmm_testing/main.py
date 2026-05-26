import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
SKLEARN_SCRIPT = BASE_DIR / "sklearn_gmm.py"
MCLUST_SCRIPT = BASE_DIR / "mclust_gmm.R"
PLOT_SCRIPT = BASE_DIR / "gmm_plots" / "plot.py"
COMBINE_SCRIPT = BASE_DIR / "combined_plots" / "combine_plots.py"
RSCRIPT = Path("/zfshome/sunle073/miniconda3/envs/r-env/bin/Rscript")
PYTHON = Path("/usr/bin/python3")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command, env=None):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n{}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}".format(
                " ".join(str(part) for part in command),
                result.stdout,
                result.stderr,
            )
        )
    return result


def run_sklearn(dataset_path, output_path):
    run_command([
        str(PYTHON),
        str(SKLEARN_SCRIPT),
        str(dataset_path),
        str(output_path),
    ])


def run_mclust(dataset_path, output_path):
    rscript = str(RSCRIPT) if RSCRIPT.exists() else "Rscript"
    run_command([rscript, str(MCLUST_SCRIPT), str(dataset_path), str(output_path)])


def run_plot(results_path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    run_command([str(PYTHON), str(PLOT_SCRIPT), str(results_path)], env=env)


def run_combine_plots():
    run_command([str(PYTHON), str(COMBINE_SCRIPT)])


def merge_dataset_results(dataset_path, sklearn_rows, mclust_rows):
    dataset_rows = read_csv(dataset_path)
    sklearn_by_row = {row["row_index"]: row for row in sklearn_rows}
    mclust_by_row = {row["row_index"]: row for row in mclust_rows}

    merged = []
    for row_index, row in enumerate(dataset_rows):
        if row["point_type"] != "middle":
            continue
        key = str(row_index)
        merged.append(
            {
                "dataset": dataset_path.stem,
                "row_index": key,
                "x1": row["x1"],
                "x2": row["x2"],
                "cluster": row["cluster"],
                "point_type": row["point_type"],
                "sklearn_prob_a": sklearn_by_row[key]["sklearn_prob_a"],
                "mclust_prob_a": mclust_by_row[key]["mclust_prob_a"],
                "sklearn_model": sklearn_by_row[key]["sklearn_model"],
                "mclust_model": mclust_by_row[key]["mclust_model"],
            }
        )
    return merged


def main():
    dataset_paths = sorted(DATASET_DIR.glob("*.csv"), key=lambda path: int(path.stem))
    all_rows = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for dataset_path in dataset_paths:
            sklearn_output = tmp_dir / "{}_sklearn.csv".format(dataset_path.stem)
            mclust_output = tmp_dir / "{}_mclust.csv".format(dataset_path.stem)

            print("Running dataset {}".format(dataset_path.name))
            run_sklearn(dataset_path, sklearn_output)
            run_mclust(dataset_path, mclust_output)

            all_rows.extend(
                merge_dataset_results(
                    dataset_path,
                    read_csv(sklearn_output),
                    read_csv(mclust_output),
                )
            )

        results_file = tmp_dir / "gmm_results.csv"
        write_csv(
            results_file,
            all_rows,
            [
                "dataset",
                "row_index",
                "x1",
                "x2",
                "cluster",
                "point_type",
                "sklearn_prob_a",
                "mclust_prob_a",
                "sklearn_model",
                "mclust_model",
            ],
        )
        run_plot(results_file)
        run_combine_plots()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
