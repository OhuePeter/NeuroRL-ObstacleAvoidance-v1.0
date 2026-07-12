import numpy as np

from src.visualization.plotting import (
    create_figure,
    save_figure
)

from src.visualization.colors import COLORS


def main():

    fig, ax = create_figure()

    x = np.linspace(0, 10, 200)

    y = np.sin(x)

    ax.plot(

        x,

        y,

        color=COLORS["P0"],

        label="Example"

    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Signal")

    ax.legend()

    save_figure(

        fig,

        "Fig00_publication_test"

    )

    print("Publication figure saved successfully.")


if __name__ == "__main__":
    main()