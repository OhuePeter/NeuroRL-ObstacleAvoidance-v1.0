import os
import numpy as np

from src.visualization.plotting import create_figure, save_figure
from src.visualization.colors import COLORS


def test_create_figure_returns_figure_and_axes():
    fig, ax = create_figure()

    assert fig is not None
    assert ax is not None


def test_save_figure_writes_png_pdf_svg(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        fig, ax = create_figure()

        x = np.linspace(0, 10, 200)
        y = np.sin(x)

        ax.plot(x, y, color=COLORS["P0"], label="Example")

        save_figure(fig, "Fig00_publication_test")

        out_root = tmp_path / "figures" / "manuscript"
        assert (out_root / "Fig00_publication_test.png").exists()
        assert (out_root / "Fig00_publication_test.pdf").exists()
        assert (out_root / "Fig00_publication_test.svg").exists()
    finally:
        os.chdir(original_cwd)