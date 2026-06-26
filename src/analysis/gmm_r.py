import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


class GMM_R():
    def __init__(self, df_train, feature_cols, rscript_path=None):
        self.df_train = df_train.copy()
        self.feature_cols = feature_cols
        self.r_script = Path(__file__).resolve().with_name("gmm_r.R")
        self.rscript_path = rscript_path or self._find_rscript()

    def _find_rscript(self):
        local_rscript = Path("/zfshome/sunle073/miniconda3/envs/r-env/bin/Rscript")
        if local_rscript.exists():
            return str(local_rscript)

        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError("Rscript was not found. Install R or pass rscript_path.")

        return rscript

    def save(self, df):
        df_results = df.copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            train_path = tmp_dir / "train.csv"
            all_path = tmp_dir / "all.csv"
            output_path = tmp_dir / "gmm_r_output.csv"

            self.df_train.to_csv(train_path, index=False)
            df_results.to_csv(all_path, index=False)

            command = [
                self.rscript_path,
                str(self.r_script),
                str(train_path),
                str(all_path),
                str(output_path),
                *self.feature_cols,
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "R GMM failed:\n{}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}".format(
                        " ".join(command),
                        result.stdout,
                        result.stderr,
                    )
                )

            if result.stdout.strip():
                print(result.stdout.strip())

            gmm_results = pd.read_csv(output_path)

        gmm_results = gmm_results.sort_values("row_index").reset_index(drop=True)
        df_results["gmm_prob_0"] = gmm_results["gmm_prob_0"].to_numpy()
        df_results["gmm_prob_1"] = gmm_results["gmm_prob_1"].to_numpy()
        df_results["gmm_prob_2"] = gmm_results["gmm_prob_2"].to_numpy()

        return df_results
