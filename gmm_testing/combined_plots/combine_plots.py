from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_PLOT_DIR = BASE_DIR / "datasets"
GMM_PLOT_DIR = BASE_DIR / "gmm_plots"
OUT_DIR = SCRIPT_DIR


def load_image(path):
    return Image.open(path).convert("RGBA")


def stack_vertical(top, bottom):
    width = max(top.width, bottom.width)
    height = top.height + bottom.height
    combined = Image.new("RGBA", (width, height), "WHITE")

    top_x = (width - top.width) // 2
    bottom_x = (width - bottom.width) // 2
    combined.paste(top, (top_x, 0), top)
    combined.paste(bottom, (bottom_x, top.height), bottom)

    return combined.convert("RGB")


def main():
    dataset_plots = sorted(DATASET_PLOT_DIR.glob("*.png"), key=lambda path: int(path.stem))
    for dataset_plot in dataset_plots:
        gmm_plot = GMM_PLOT_DIR / dataset_plot.name
        if not gmm_plot.exists():
            print("Skipping {}: missing {}".format(dataset_plot.name, gmm_plot))
            continue

        combined = stack_vertical(
            load_image(dataset_plot),
            load_image(gmm_plot),
        )
        output_path = OUT_DIR / dataset_plot.name
        combined.save(output_path)
        print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
