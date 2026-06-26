import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    for config_nr in range(7):
        print(f"\nRunning bootstrap config {config_nr}")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "bootstrap_main.py"), str(config_nr)],
            check=True,
        )


if __name__ == "__main__":
    main()
