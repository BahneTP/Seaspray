import argparse
import importlib
import data_generation
import plot_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name der Config, z. B. config_0"
    )
    args = parser.parse_args()

    config_module = importlib.import_module(f"configs.{args.config}")

    data_generation.run(config_module)
    plot_grid.run(config_module)


if __name__ == "__main__":
    main()